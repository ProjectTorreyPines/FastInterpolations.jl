# ========================================
# HeteroInterpolantND — Per-Axis Window Traits
# ========================================
# Cell-local stencil windows for the unified mini-collapse OnTheFly path.
#
# When the OnTheFly path encounters local methods (PCHIP/Cardinal/Akima),
# `_eval_hetero_nd` (in hetero_eval.jl) computes a per-axis window around
# the query cell — only those grid points are read by the kernel. This
# turns the dim-1 collapse loop from O(n^(N-1)) into O(stencil^(N-1)).
#
# Design invariants enforced here:
# 1. Each method's window length is constant per `(method type, axis length)`
#    pair: for grids large enough to satisfy the stencil, the length is
#    `_fixed_window_size(m)` (a method-only constant); for tiny grids it
#    falls back to `1:n` (constant in n). Within a single interpolant n is
#    fixed, so the pool buffer slot size stays identical across all queries
#    of that interpolant — no resize churn, zero alloc after warmup.
# 2. The window contains enough grid points such that the inner 1D oneshot
#    call on the windowed view is *numerically identical* to the same call
#    on the full grid. For PCHIP/Cardinal this means n_window ≥ 3; for
#    Akima ≥ 4 (Akima 1D has a special n==3 branch that diverges from the
#    general n≥4 formula — see hermite_local_slopes.jl:99-107). On tiny
#    grids the windowed call and the full-grid call are literally the same
#    call (both use `1:n`), so equivalence holds trivially.
# 3. Global-solve methods (Cubic, Quadratic) always return the full axis
#    — they cannot be windowed because the spline solve is non-local.

# ── Fixed window size per method type ──
#
# Format: 2*radius + 2 grid points centered on the cell endpoints (ix, ix+1).
# - PCHIP/Cardinal need ±1 neighbors for slopes at the cell endpoints → 4 points
# - Akima needs ±2 neighbors for the 4-secant formula → 6 points
# - Linear/Constant need only the 2 cell endpoints → 2 points (kernel ignores
#   the extras anyway, but we keep the size minimal to maximize speedup)
@inline _fixed_window_size(::PchipInterp) = 4
@inline _fixed_window_size(::CardinalInterp) = 4
@inline _fixed_window_size(::AkimaInterp) = 6
@inline _fixed_window_size(::LinearInterp) = 2
@inline _fixed_window_size(::ConstantInterp) = 2

# ── Window builder for "windowable" methods ──
#
# Returns a UnitRange{Int} of length `_fixed_window_size(m)`, centered on the
# cell endpoints (ix, ix+1) and asymmetrically extended to stay within [1, n].
# When the grid is too small (n < fixed_window), returns the full axis 1:n.
@inline function _axis_window(
        m::Union{PchipInterp, CardinalInterp, AkimaInterp, LinearInterp, ConstantInterp},
        ix::Int, n::Int
    )
    fw = _fixed_window_size(m)
    n < fw && return 1:n   # tiny-grid fallback (full axis)
    # Symmetric stencil around (ix, ix+1): ix - r .. ix + 1 + r, where r = (fw - 2) ÷ 2
    r = (fw - 2) >> 1
    lo = ix - r
    hi = ix + 1 + r
    # Clamp to [1, n] with asymmetric extension to keep length == fw
    if lo < 1
        hi += 1 - lo
        lo = 1
    elseif hi > n
        lo -= hi - n
        hi = n
    end
    return lo:hi
end

# ── Global-solve methods: full axis, no windowing ──
@inline _axis_window(::CubicInterp, ix::Int, n::Int) = 1:n
@inline _axis_window(::QuadraticInterp, ix::Int, n::Int) = 1:n

# ── PeriodicBC variants: full axis, no windowing ──
# A cell-local window for PeriodicBC would split the seam cell across the grid
# boundary, breaking the 1D entry's `bc.period - (x[end] - x[1])` seam-cell
# computation (the windowed `x` view is a sub-range, not the full periodic
# grid). Forwarding the FULL grid lets the 1D entry's zero-copy `_periodic_secant`
# path operate correctly. Trade-off: full-axis fiber scan instead of
# cell-local stencil for PeriodicBC axes — perf reverts to global-solve-like
# pattern, which is acceptable for the (less-common) periodic case.
@inline _axis_window(::PchipInterp{<:PeriodicBC}, ix::Int, n::Int) = 1:n
@inline _axis_window(::CardinalInterp{T, <:PeriodicBC}, ix::Int, n::Int) where {T} = 1:n
@inline _axis_window(::AkimaInterp{<:PeriodicBC}, ix::Int, n::Int) = 1:n

# ── Windowable-method trait (persistent-path gate) ──
#
# A method is "windowable" iff it evaluates from a fixed-size cell-local stencil
# (i.e. `_axis_window(m, ix, n)` returns a sub-range narrower than `1:n` for
# large grids). This is a STRICT SUPERSET of `_is_local_method` (which is about
# "computes local slopes" and is used by the OnTheFly-vs-PreCompute resolver).
#
# Linear/Constant are windowable (2-point stencil) even though they're not
# "local-slope" methods. We use a separate trait for them because extending
# `_is_local_method` would change the scalar/batch `AutoCoeffs` resolver's
# behavior, which we want to keep frozen.
#
# ⚠️  ASYMMETRY WARNING — read before changing gate sites:
#
# This trait is deliberately used ONLY at the persistent-interpolant call
# sites (`_eval_hetero_nd(<:Array)`, `_locate_cell(<:Array)`, the `@generated
# _eval_nointerp` path). The scalar-oneshot path (`_interp_nd_oneshot_onthefly`)
# and the batch dispatcher's spacings pre-compute still use `_has_any_local_method`.
#
# Rationale: the windowed path runs `_search_all_intervals(q_eval, grids,
# spacings, ...)` and needs per-axis spacings to binary-search for the cell
# index. The persistent path has `itp.spacings` PRE-COMPUTED at construction
# and stored in the struct — zero per-call allocation. The scalar one-shot
# path has to build spacings inline via `map(_create_spacing, grids)`, and for
# Vector grids `_create_spacing` allocates `h` and `inv_h` buffers (~1024 B
# for a 30×25 grid). For Hermite methods the stencil win (~5-20× for local-
# slope kernels) justifies that allocation; for pure Linear/Constant with
# Vector grids it would be a net loss for scalar one-shot, so we leave the
# one-shot gate narrower.
#
# Persistent interpolant + pure Linear/Constant: 667 ns → ~50 ns (13×) for
# 100×100, no per-call allocation (spacings are cached in itp.spacings).
@inline _is_windowable_method(::PchipInterp) = true
@inline _is_windowable_method(::CardinalInterp) = true
@inline _is_windowable_method(::AkimaInterp) = true
@inline _is_windowable_method(::LinearInterp) = true
@inline _is_windowable_method(::ConstantInterp) = true
@inline _is_windowable_method(::AbstractInterpMethod) = false
@inline _has_any_windowable_method(methods::Tuple) = any(_is_windowable_method, methods)
