# ========================================
# Periodic Slope Infrastructure for Hermite Methods
# ========================================
#
# Two endpoint variants of `PeriodicBC` flow through the slope code uniformly
# via the wrap-aware secant access primitives below. No grid extension is
# required; the user's n-length grid is consumed as-is at slope time, while
# `_resolve_search`'s seam dispatch (master infrastructure) handles eval-time
# wrap. This is the genuine "zero-copy exclusive" path.
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
@inline _secant_cycle_length(n::Int, ::PeriodicBC{:inclusive}) = n - 1
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
    @inbounds return (y[jw + 1] - y[jw]) / (x[jw + 1] - x[jw])
end

@inline function _periodic_secant(x::AbstractVector, y::AbstractVector, j::Int, n::Int, bc::PeriodicBC{:exclusive})
    jw = mod1(j, n)
    if jw == n
        # `bc.period` may be `nothing` (Range-inferred) — `_resolve_exclusive_period`
        # returns either the user-supplied period or the Range-inferred value.
        period = _resolve_exclusive_period(x, bc)
        seam_h = period - (@inbounds x[n] - x[1])
        @inbounds return (y[1] - y[n]) / seam_h
    end
    @inbounds return (y[jw + 1] - y[jw]) / (x[jw + 1] - x[jw])
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

@inline function _periodic_cell_width(x::AbstractVector, j::Int, n::Int, bc::PeriodicBC{:exclusive})
    jw = mod1(j, n)
    if jw == n
        # `bc.period` may be `nothing` (Range-inferred). Resolve via the same
        # helper used by `_resolve_extrap` / `_resolve_search`.
        period = _resolve_exclusive_period(x, bc)
        return period - (@inbounds x[n] - x[1])
    end
    @inbounds return x[jw + 1] - x[jw]
end

# ────────────────────────────────────────────
# Slope computation dispatch (method-agnostic, passes bc)
# ────────────────────────────────────────────
_compute_slopes!(dy, x, y, sm::PchipSlopes) = _pchip_slopes!(dy, x, y; bc = sm.bc)
_compute_slopes!(dy, x, y, sm::CardinalSlopes) = _cardinal_slopes!(dy, x, y, sm.tension; bc = sm.bc)
_compute_slopes!(dy, x, y, sm::AkimaSlopes) = _akima_slopes!(dy, x, y; bc = sm.bc)

# ────────────────────────────────────────────
# BC normalization after grid extension
# ────────────────────────────────────────────
# `_periodic_extend_1d` produces a closed-cycle (n+1) grid for `:exclusive`
# input and a passthrough n-grid for `:inclusive` (already closed-cycle).
# After this normalization, the slope side should treat the grid as
# `:inclusive` regardless of the user's original `bc.endpoint` — the seam
# cell is now the last cell of the extended grid (or already in place for
# `:inclusive`). `check=false` skips redundant endpoint validation since
# the extension constructs `y_eff[end] = y_eff[1]` by definition.
@inline _bc_after_extend(bc::AbstractBC) = bc
@inline _bc_after_extend(::PeriodicBC) = PeriodicBC(endpoint = :inclusive, check = false)
