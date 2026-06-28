# ========================================
# Periodic Slope Infrastructure for Hermite Methods
# ========================================
#
# Two endpoint variants of `PeriodicBC` flow through the slope code uniformly
# via the wrap-aware secant access primitives below.
#
# Used by two different paths with different copy strategies:
#
#   1. Oneshot (OnTheFly + PreCompute scalar/vector entry points):
#      zero-copy — the user's n-length grid is consumed as-is at slope time;
#      `_resolve_search`'s seam dispatch handles eval-time wrap.
#
#   2. Persistent (`pchip_interp(x, y; bc=...)` etc.): mirrors the
#      Linear/Constant convention via `_periodic_extend_1d` for both
#      endpoints, then renormalizes `bc` to `:inclusive` via
#      `_bc_after_extend`. The (n+1) extension is a one-time construction
#      cost; per-query eval has zero overhead. The wrap-aware primitives
#      below still drive the boundary slopes on the extended grid.
#
# Closed-cycle index conventions:
#
#   :inclusive — n grid points, n-1 cells; user provides matched endpoints
#                (y[n] ≈ y[1], x[n] = x[1] + L). Real secants are m_1..m_{n-1}.
#                The last secant m_{n-1} is itself the seam-cell secant. The
#                cycle of secant indices has period n-1, so wrapping uses
#                `mod1(j, n-1)`.
#
#   :exclusive — n grid points, n cells (cell n is the seam cell from x[n] to
#                x[1] + period). Real secants exist for cells 1..n-1; the
#                seam-cell secant m_n is virtual: `(y[1] - y[n]) / seam_h`
#                where `seam_h = period - (x[n] - x[1])`. The cycle of secant
#                indices has period n, so wrapping uses `mod1(j, n)` and the
#                seam case is the unique cell n.
#
# `_periodic_secant(x, y, j, n, bc)` and `_periodic_cell_width(x, j, n, bc)`
# absorb both differences. All boundary slope helpers consume them and stay
# endpoint-agnostic.

# ────────────────────────────────────────────
# Wrap-aware secant access
# ────────────────────────────────────────────

# Cycle period for the secant index sequence (number of cells in closed cycle).
# `:inclusive` and `:extended` share the same length-(n+1) closed-cycle data
# layout — only the BC symbol differs (user-supplied vs library-promoted).
@inline _secant_cycle_length(n::Int, ::PeriodicBC{:inclusive}) = n - 1
@inline _secant_cycle_length(n::Int, ::PeriodicBC{:extended}) = n - 1
@inline _secant_cycle_length(n::Int, ::PeriodicBC{:exclusive}) = n

"""
    _periodic_secant(x, y, j, n, bc) -> Tv

Closed-cycle secant `m_j = (y[j+1] - y[j]) / (x[j+1] - x[j])` with wrap-aware
index resolution. `j` may be any integer; it is wrapped into the cycle
`[1, _secant_cycle_length(n, bc)]` via `mod1`.

For `:exclusive`, the wrapped index `n` denotes the *virtual seam cell* and
returns `(y[1] - y[n]) / seam_h` instead of an out-of-bounds grid lookup.

For `:inclusive`, all wrapped indices fall in `[1, n-1]` and return real
grid secants directly.
"""
@inline function _periodic_secant(x::AbstractVector, y::AbstractVector, j::Int, n::Int, ::PeriodicBC{:inclusive})
    nm1 = n - 1
    jw = mod1(j, nm1)
    Tc = _promote_eltype(_coeff_op, eltype(x), eltype(y))
    @inbounds return _fielddiff(Tc, y[jw + 1], y[jw]) / (x[jw + 1] - x[jw])
end

# `:extended` shares the `:inclusive` data layout (length n+1 closed-cycle);
# secant cycle is identical to `:inclusive`.
@inline function _periodic_secant(x::AbstractVector, y::AbstractVector, j::Int, n::Int, ::PeriodicBC{:extended})
    nm1 = n - 1
    jw = mod1(j, nm1)
    Tc = _promote_eltype(_coeff_op, eltype(x), eltype(y))
    @inbounds return _fielddiff(Tc, y[jw + 1], y[jw]) / (x[jw + 1] - x[jw])
end

# Cast the resolved exclusive period to the grid's promoted-float type so
# seam-cell arithmetic (`seam_h`, seam secant) stays in the grid eltype.
# Without this cast, a `Float64` user-supplied period combined with a
# `Float32` grid would silently widen Float32 → Float64 (PCHIP/Cardinal) or
# trigger a `MethodError` in `_akima_weighted_slope` (which requires all 4
# secants to share `Tv`). Mirrors the duck-safe `_PromotableValue` lift
# used by `_extend_exclusive` in `core/periodic.jl`.
@inline function _resolve_seam_period(x::AbstractVector, bc::PeriodicBC{:exclusive})
    period_raw = _resolve_exclusive_period(x, bc)
    Tg_raw = eltype(x)
    Tg = Tg_raw <: _PromotableValue ? float(Tg_raw) : Tg_raw
    return Tg(period_raw)
end

@inline function _periodic_secant(x::AbstractVector, y::AbstractVector, j::Int, n::Int, bc::PeriodicBC{:exclusive})
    jw = mod1(j, n)
    if jw == n
        period = _resolve_seam_period(x, bc)
        seam_h = period - (@inbounds x[n] - x[1])
        Tc = _promote_eltype(_coeff_op, eltype(x), eltype(y))
        @inbounds return _fielddiff(Tc, y[1], y[n]) / seam_h
    end
    Tc = _promote_eltype(_coeff_op, eltype(x), eltype(y))
    @inbounds return _fielddiff(Tc, y[jw + 1], y[jw]) / (x[jw + 1] - x[jw])
end

"""
    _periodic_cell_width(x, j, n, bc) -> Tg

Width of the cell whose secant is `m_j` in closed-cycle representation.
Mirrors `_periodic_secant`'s wrapping. Used by methods that need the cell
width (PCHIP harmonic mean) in addition to the secant value.
"""
@inline function _periodic_cell_width(x::AbstractVector, j::Int, n::Int, ::PeriodicBC{:inclusive})
    nm1 = n - 1
    jw = mod1(j, nm1)
    @inbounds return x[jw + 1] - x[jw]
end

# `:extended` shares the `:inclusive` data layout — cell-width logic is identical.
@inline function _periodic_cell_width(x::AbstractVector, j::Int, n::Int, ::PeriodicBC{:extended})
    nm1 = n - 1
    jw = mod1(j, nm1)
    @inbounds return x[jw + 1] - x[jw]
end

@inline function _periodic_cell_width(x::AbstractVector, j::Int, n::Int, bc::PeriodicBC{:exclusive})
    jw = mod1(j, n)
    if jw == n
        period = _resolve_seam_period(x, bc)
        return period - (@inbounds x[n] - x[1])
    end
    @inbounds return x[jw + 1] - x[jw]
end

# `_bc_after_extend` lives in `src/core/periodic.jl` next to `_periodic_extend_1d`
# — it is generic post-extension BC normalization, not Hermite-specific.
