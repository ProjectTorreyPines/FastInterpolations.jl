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
# 1. Each method's window length is FIXED per method type (compile-time
#    constant), so the pool buffer slot size is identical across queries
#    of the same interpolant — no resize churn, zero alloc after warmup.
# 2. The window contains enough grid points such that the inner 1D oneshot
#    call on the windowed view is *numerically identical* to the same call
#    on the full grid. For PCHIP/Cardinal this means n_window ≥ 3; for
#    Akima ≥ 4 (Akima 1D has a special n==3 branch that diverges from the
#    general n≥4 formula — see hermite_local_slopes.jl:99-107).
# 3. Tiny grids (n < fixed_window) fall back to a full-axis range. This
#    is rare and the size is still compile-time-constant per (method, n_axis).
# 4. Global-solve methods (Cubic, Quadratic) always return the full axis
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
