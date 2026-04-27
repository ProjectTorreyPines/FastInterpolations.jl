# ========================================
# Local (Per-Index) Slope Computation
# ========================================
# O(1) slope computation at a single grid index using a small stencil.
# Used by the OnTheFly coefficient strategy to avoid precomputing all slopes.
#
# Each _local_slope(method, x, y, i, n) returns the slope dy[i] that would
# be produced by the corresponding bulk _*_slopes! function.
#
# Boundary indices (i ∈ {1, n}) dispatch on `sm.bc` via per-method
# `_*_boundary_slope(x, y, i, n, …, bc)` helpers. Akima additionally needs
# wrap-aware paths at i ∈ {2, n-1} because its K=5 stencil crosses the join
# from those indices too.
#
# PeriodicBC `inclusive` vs `exclusive` differ in:
#   - secant cycle length (n-1 vs n)
#   - existence of a "virtual" seam-cell secant (only `:exclusive`)
# Both are absorbed by `_periodic_secant` / `_periodic_cell_width` (in
# hermite_periodic_slopes.jl) — the boundary helpers below stay
# endpoint-agnostic and call those primitives.
#
# Reuses existing helpers:
#   _pchip_endpoint_slope (pchip_slopes.jl)
#   _akima_weighted_slope (akima_slopes.jl)

# ========================================
# PCHIP Local Slope (Fritsch-Carlson)
# ========================================
# Stencil: 3 points — y[i-1], y[i], y[i+1]
# Boundary (NoBC):       _pchip_endpoint_slope (3-point one-sided FD + monotonicity clamping)
# Boundary (PeriodicBC): closed-cycle interior formula via abstraction

@inline function _local_slope(sm::PchipSlopes, x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    # Special case: 2 points → linear
    if n == 2
        @inbounds return (y[2] - y[1]) / (x[2] - x[1])
    end

    if i == 1 || i == n
        return _pchip_boundary_slope(x, y, i, n, sm.bc)
    end

    # Interior: weighted harmonic mean (Fritsch-Carlson)
    @inbounds begin
        h_prev = x[i] - x[i - 1]
        h_curr = x[i + 1] - x[i]
        δ_prev = (y[i] - y[i - 1]) / h_prev
        δ_curr = (y[i + 1] - y[i]) / h_curr
    end
    if sign(δ_prev) != sign(δ_curr)
        return zero(Tv)
    else
        w1 = 2 * h_curr + h_prev
        w2 = h_curr + 2 * h_prev
        return (w1 + w2) / (w1 / δ_prev + w2 / δ_curr)
    end
end

# ── PCHIP boundary slope dispatch ──
# NoBC: original 3-point one-sided FD with monotonicity clamping
@inline function _pchip_boundary_slope(x, y, i, n, ::NoBC)
    if i == 1
        @inbounds begin
            h1 = x[2] - x[1]
            h2 = x[3] - x[2]
            δ1 = (y[2] - y[1]) / h1
            δ2 = (y[3] - y[2]) / h2
        end
        return _pchip_endpoint_slope(h1, h2, δ1, δ2)
    else  # i == n
        @inbounds begin
            h_last = x[n] - x[n - 1]
            h_prev = x[n - 1] - x[n - 2]
            δ_last = (y[n] - y[n - 1]) / h_last
            δ_prev = (y[n - 1] - y[n - 2]) / h_prev
        end
        return _pchip_endpoint_slope(h_last, h_prev, δ_last, δ_prev)
    end
end

# PeriodicBC (both endpoints): closed-cycle interior formula. The wrap-aware
# secant/cell-width primitives absorb the inclusive vs exclusive index shift,
# so this single body handles both endpoints identically.
#
# At inclusive i ∈ {1, n}: secant pair (m_{i-1}, m_i) in closed cycle yields
#   the same value at both ends (dy[1] == dy[n]) — C¹ at join automatic.
# At exclusive i = 1:  m_0 = seam, m_1 = cell[1,2]. dy[1] uses seam.
# At exclusive i = n:  m_{n-1} = cell[n-1,n], m_n = seam. dy[n] uses seam too.
#   dy[1] ≠ dy[n] in general for exclusive; both endpoints are correctly
#   wrap-aware so eval-time C¹ at the seam holds via search's `idx_R = 1`.
@inline function _pchip_boundary_slope(x::AbstractVector{Tg}, y::AbstractVector{Tv}, i, n, bc::PeriodicBC) where {Tg, Tv}
    δ_prev = _periodic_secant(x, y, i - 1, n, bc)
    δ_curr = _periodic_secant(x, y, i, n, bc)
    h_prev = _periodic_cell_width(x, i - 1, n, bc)
    h_curr = _periodic_cell_width(x, i, n, bc)
    if sign(δ_prev) != sign(δ_curr)
        return zero(promote_type(Tv, Tg))
    else
        w1 = 2 * h_curr + h_prev
        w2 = h_curr + 2 * h_prev
        return (w1 + w2) / (w1 / δ_prev + w2 / δ_curr)
    end
end

# ========================================
# Cardinal Local Slope
# ========================================
# Stencil: 3 points — y[i-1], y[i], y[i+1]
# Boundary (NoBC):       one-sided 2-point FD × scale
# Boundary (PeriodicBC): closed-cycle central FD × scale via abstraction

@inline function _local_slope(sm::CardinalSlopes, x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    scale = one(Tg) - sm.tension

    # Special case: 2 points → linear
    if n == 2
        @inbounds return scale * (y[2] - y[1]) / (x[2] - x[1])
    end

    if i == 1 || i == n
        return _cardinal_boundary_slope(x, y, i, n, scale, sm.bc)
    end
    @inbounds return scale * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
end

@inline function _cardinal_boundary_slope(x, y, i, n, scale, ::NoBC)
    return if i == 1
        @inbounds scale * (y[2] - y[1]) / (x[2] - x[1])
    else
        @inbounds scale * (y[n] - y[n - 1]) / (x[n] - x[n - 1])
    end
end

# PeriodicBC closed-cycle central FD over the join. Numerator and denominator
# expressed in terms of cell secants and widths so the abstraction handles
# inclusive (m_n = m_1 wrap) vs exclusive (m_n = seam virtual) uniformly.
# Equivalent to scale * (y[i+1] - y_wrapped[i-1]) / (x[i+1] - x_wrapped[i-1]).
@inline function _cardinal_boundary_slope(x, y, i, n, scale, bc::PeriodicBC)
    h_prev = _periodic_cell_width(x, i - 1, n, bc)
    h_curr = _periodic_cell_width(x, i, n, bc)
    m_prev = _periodic_secant(x, y, i - 1, n, bc)
    m_curr = _periodic_secant(x, y, i, n, bc)
    return scale * (m_prev * h_prev + m_curr * h_curr) / (h_prev + h_curr)
end

# ========================================
# Akima Local Slope
# ========================================
# Stencil: up to 5 points — y[i-2..i+2] (4 secants m[i-2..i+1])
# Boundary (NoBC):       virtual secants via linear extrapolation of secant sequence
# Boundary (PeriodicBC): real wrapped secants — full closed-cycle support.
#
# Akima's K=5 stencil crosses the join at FOUR indices: i ∈ {1, 2, n-1, n}.
# All four take the wrap-aware path. Indices 3..n-2 stay on the real-secants
# fast path (no wrap overhead).

@inline function _local_slope(sm::AkimaSlopes, x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    # Special case: 2 points → linear
    if n == 2
        @inbounds return (y[2] - y[1]) / (x[2] - x[1])
    end

    # Special case: 3 points
    if n == 3
        # PeriodicBC: every index lies within K=5 stencil reach of the join,
        # so use the wrap-aware path (cycle of 2 secants for inclusive, 3 for
        # exclusive — `_periodic_secant` handles both via mod1).
        if sm.bc isa PeriodicBC
            return _akima_local_4secant_periodic(x, y, i, n, sm.bc)
        end
        @inbounds begin
            m1 = (y[2] - y[1]) / (x[2] - x[1])
            m2 = (y[3] - y[2]) / (x[3] - x[2])
        end
        i == 1 && return m1
        i == 3 && return m2
        return (m1 + m2) / 2  # i == 2
    end

    # General case: n ≥ 4
    # PeriodicBC: wrap-aware path for i ∈ {1, 2, n-1, n}.
    if sm.bc isa PeriodicBC && (i <= 2 || i >= n - 1)
        return _akima_local_4secant_periodic(x, y, i, n, sm.bc)
    end
    return _akima_local_4secant(x, y, i, n)
end

# Akima 4-secant computation with wrap-aware secant access. Dispatches via
# `_periodic_secant` so inclusive (m_n = m_1 wrap, mod1 j n-1) and exclusive
# (m_n = seam virtual, mod1 j n) both work with one body.
@inline function _akima_local_4secant_periodic(x, y, i::Int, n::Int, bc::PeriodicBC)
    @inline _ws(j) = _periodic_secant(x, y, j, n, bc)
    return _akima_weighted_slope(_ws(i - 2), _ws(i - 1), _ws(i), _ws(i + 1))
end

# Helper: compute the 4 secants needed for Akima slope at index i (NoBC).
# Secant at interval j → j+1 is m_j = (y[j+1] - y[j]) / (x[j+1] - x[j])
# Real secants exist for j = 1..n-1. Virtual secants extend the sequence linearly.
@inline function _akima_local_4secant(x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    # Akima slope at grid index i needs 4 secants: m[i-2], m[i-1], m[i], m[i+1]
    # where m[k] = secant at interval [k, k+1].
    # Real secant range: 1 to n-1.
    # Out-of-range secants are virtual (linearly extrapolated from the secant sequence).

    @inline _secant(j) = @inbounds (y[j + 1] - y[j]) / (x[j + 1] - x[j])

    # Virtual secants extend the real sequence linearly:
    #   m[0]   = 2*m[1] - m[2]
    #   m[-1]  = 2*m[0] - m[1]  = 3*m[1] - 2*m[2]
    #   m[n]   = 2*m[n-1] - m[n-2]
    #   m[n+1] = 2*m[n] - m[n-1] = 3*m[n-1] - 2*m[n-2]

    nm1 = n - 1  # last valid secant index

    @inline function _safe_secant(j)
        if 1 <= j <= nm1
            return _secant(j)
        elseif j == 0
            return 2 * _secant(1) - _secant(2)
        elseif j == -1
            m1 = _secant(1)
            m2 = _secant(2)
            return 3 * m1 - 2 * m2
        elseif j == nm1 + 1  # n
            return 2 * _secant(nm1) - _secant(nm1 - 1)
        elseif j == nm1 + 2  # n+1
            m_last = _secant(nm1)
            m_prev = _secant(nm1 - 1)
            return 3 * m_last - 2 * m_prev
        else
            error("_safe_secant: j=$j out of expected range [-1, $(nm1 + 2)] for n=$(nm1 + 1)")
        end
    end

    return _akima_weighted_slope(
        _safe_secant(i - 2), _safe_secant(i - 1), _safe_secant(i), _safe_secant(i + 1)
    )
end
