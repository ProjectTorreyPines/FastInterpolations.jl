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

# ── PeriodicBC variants on the persistent path: full axis ──
#
# Persistent interpolants reach `_axis_window` via `_eval_hetero_nd` /
# `_locate_cell` with `itp.methods` still carrying `PeriodicBC` and the
# extended grid. A windowed view there does not satisfy the
# `y[1] ≈ y[end]` BC check inside the inner 1D oneshot, so the persistent
# path keeps full-axis fallback. The OnTheFly oneshot path uses the
# wrap-aware helpers below (`_axis_window_pooled` + `_axis_grid_pooled`)
# instead, which bake the wrap into pool-allocated index/grid buffers and
# strip BC for the inner call.
@inline _axis_window(::PchipInterp{<:PeriodicBC}, ix::Int, n::Int) = 1:n
@inline _axis_window(::CardinalInterp{T, <:PeriodicBC}, ix::Int, n::Int) where {T} = 1:n
@inline _axis_window(::AkimaInterp{<:PeriodicBC}, ix::Int, n::Int) = 1:n

# ── OnTheFly oneshot wrap-aware window + grid helpers ──
#
# Per-axis dispatch entries (`_axis_window_pooled`, `_axis_grid_pooled`):
# only the dispatch entry takes `pool` — for periodic methods it acquires a
# small (4-6 element) buffer and hands it to the pure in-place fill helpers
# (`_fill_periodic_window!`, `_fill_periodic_grid!`). The fill helpers are
# pool-agnostic (mirrors codebase convention: `_compute_*!`, `_slope_1d!`).
# Non-periodic methods go through the default method (UnitRange / grid view,
# no pool acquire).

@inline _axis_window_pooled(pool, m::AbstractInterpMethod, x::AbstractVector, ix::Int) =
    _axis_window(m, ix, length(x))
@inline _axis_window_pooled(pool, m::PchipInterp{<:PeriodicBC}, x::AbstractVector, ix::Int) =
    _fill_periodic_window!(acquire!(pool, Int, _fixed_window_size(m)), m, ix, length(x))
@inline _axis_window_pooled(pool, m::CardinalInterp{T, <:PeriodicBC}, x::AbstractVector, ix::Int) where {T} =
    _fill_periodic_window!(acquire!(pool, Int, _fixed_window_size(m)), m, ix, length(x))
@inline _axis_window_pooled(pool, m::AkimaInterp{<:PeriodicBC}, x::AbstractVector, ix::Int) =
    _fill_periodic_window!(acquire!(pool, Int, _fixed_window_size(m)), m, ix, length(x))

@inline _axis_grid_pooled(pool, ::AbstractInterpMethod, x::AbstractVector, w::AbstractVector{Int}, ::Int) =
    view(x, w)
@inline _axis_grid_pooled(pool, m::PchipInterp{<:PeriodicBC}, x::AbstractVector{Tg}, ::AbstractVector{Int}, ix::Int) where {Tg} =
    _fill_periodic_grid!(acquire!(pool, Tg, _fixed_window_size(m)), m, x, ix)
@inline _axis_grid_pooled(pool, m::CardinalInterp{T, <:PeriodicBC}, x::AbstractVector{Tg}, ::AbstractVector{Int}, ix::Int) where {T, Tg} =
    _fill_periodic_grid!(acquire!(pool, Tg, _fixed_window_size(m)), m, x, ix)
@inline _axis_grid_pooled(pool, m::AkimaInterp{<:PeriodicBC}, x::AbstractVector{Tg}, ::AbstractVector{Int}, ix::Int) where {Tg} =
    _fill_periodic_grid!(acquire!(pool, Tg, _fixed_window_size(m)), m, x, ix)

# Pure in-place fill helpers (no pool knowledge).
@inline function _fill_periodic_window!(buf::AbstractVector{Int}, m, ix::Int, n::Int)
    fw = length(buf)
    cycle = _wrap_cycle(m.bc, n)
    r = (fw - 2) >> 1
    @inbounds for k in 1:fw
        buf[k] = mod1(ix - r + (k - 1), cycle)
    end
    return buf
end

@inline function _fill_periodic_grid!(buf::AbstractVector{Tg}, m, x::AbstractVector{Tg}, ix::Int) where {Tg}
    fw = length(buf)
    n = length(x)
    bc = m.bc
    cycle = _wrap_cycle(bc, n)
    period = _wrap_period(x, bc)
    r = (fw - 2) >> 1
    @inbounds for k in 1:fw
        raw = ix - r + (k - 1)
        wrap = mod1(raw, cycle)
        offset = (raw - wrap) ÷ cycle
        buf[k] = x[wrap] + offset * period
    end
    return buf
end

# Cycle period for `mod1(k, cycle)` index wrap.
# - Inclusive (y[1]==y[n]): cycle = n-1, so x[n] ≡ x[1] + period.
# - Exclusive (n distinct samples on [x[1], x[1]+period)): cycle = n.
@inline _wrap_cycle(::PeriodicBC{:inclusive}, n::Int) = n - 1
@inline _wrap_cycle(::PeriodicBC{:exclusive}, n::Int) = n

@inline _wrap_period(x::AbstractVector, ::PeriodicBC{:inclusive}) = last(x) - first(x)
@inline _wrap_period(x::AbstractVector, bc::PeriodicBC{:exclusive}) = _resolve_exclusive_period(x, bc)

# ── BC strip helpers: PeriodicBC → NoBC, WrapExtrap → NoExtrap ──
#
# When the wrap-aware helpers above bake the periodic seam into pool-allocated
# index + monotonic shifted x buffers, the inner 1D oneshot must NOT re-apply
# periodic slope wrap. We strip per-axis: identity for non-periodic methods.
@inline _strip_periodic_bc(::PchipInterp{<:PeriodicBC}) = PchipInterp(NoBC())
@inline _strip_periodic_bc(m::CardinalInterp{T, <:PeriodicBC}) where {T} =
    CardinalInterp(m.tension, NoBC())
@inline _strip_periodic_bc(::AkimaInterp{<:PeriodicBC}) = AkimaInterp(NoBC())
@inline _strip_periodic_bc(m::AbstractInterpMethod) = m

@inline _is_periodic_method(::PchipInterp{<:PeriodicBC}) = true
@inline _is_periodic_method(::CardinalInterp{T, <:PeriodicBC}) where {T} = true
@inline _is_periodic_method(::AkimaInterp{<:PeriodicBC}) = true
@inline _is_periodic_method(::AbstractInterpMethod) = false

@inline _has_any_periodic_method(methods::Tuple) = any(_is_periodic_method, methods)

@inline _strip_wrap_extrap(e, m) = _is_periodic_method(m) ? NoExtrap() : e

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
