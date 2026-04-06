# ========================================
# Local (Per-Index) Slope Computation
# ========================================
# O(1) slope computation at a single grid index using a small stencil.
# Used by the OnTheFly coefficient strategy to avoid precomputing all slopes.
#
# Each _local_slope(method, x, y, i, n) returns the slope dy[i] that would
# be produced by the corresponding bulk _*_slopes! function.
#
# Reuses existing helpers:
#   _pchip_endpoint_slope (pchip_slopes.jl)
#   _akima_weighted_slope (akima_slopes.jl)

# ========================================
# PCHIP Local Slope (Fritsch-Carlson)
# ========================================
# Stencil: 3 points — y[i-1], y[i], y[i+1]
# Boundary: _pchip_endpoint_slope (3-point one-sided FD + monotonicity clamping)

@inline function _local_slope(::PchipSlopes, x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    # Special case: 2 points → linear
    if n == 2
        @inbounds return (y[2] - y[1]) / (x[2] - x[1])
    end

    if i == 1
        # Left endpoint: 3-point one-sided FD with monotonicity clamping
        @inbounds begin
            h1 = x[2] - x[1]
            h2 = x[3] - x[2]
            δ1 = (y[2] - y[1]) / h1
            δ2 = (y[3] - y[2]) / h2
        end
        return _pchip_endpoint_slope(h1, h2, δ1, δ2)
    elseif i == n
        # Right endpoint: reversed args (boundary interval last)
        @inbounds begin
            h_last = x[n] - x[n - 1]
            h_prev = x[n - 1] - x[n - 2]
            δ_last = (y[n] - y[n - 1]) / h_last
            δ_prev = (y[n - 1] - y[n - 2]) / h_prev
        end
        return _pchip_endpoint_slope(h_last, h_prev, δ_last, δ_prev)
    else
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
end

# ========================================
# Cardinal Local Slope
# ========================================
# Stencil: 3 points — y[i-1], y[i], y[i+1]
# Boundary: one-sided 2-point FD

@inline function _local_slope(sm::CardinalSlopes, x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    scale = one(Tg) - sm.tension

    # Special case: 2 points → linear
    if n == 2
        @inbounds return scale * (y[2] - y[1]) / (x[2] - x[1])
    end

    return if i == 1
        @inbounds scale * (y[2] - y[1]) / (x[2] - x[1])
    elseif i == n
        @inbounds scale * (y[n] - y[n - 1]) / (x[n] - x[n - 1])
    else
        @inbounds scale * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    end
end

# ========================================
# Akima Local Slope
# ========================================
# Stencil: up to 5 points — y[i-2..i+2] (4 secants)
# Boundary: virtual secants via linear extrapolation of secant sequence
# Reuses _akima_weighted_slope from akima_slopes.jl

@inline function _local_slope(::AkimaSlopes, x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    # Special case: 2 points → linear
    if n == 2
        @inbounds return (y[2] - y[1]) / (x[2] - x[1])
    end

    # Special case: 3 points
    if n == 3
        @inbounds begin
            m1 = (y[2] - y[1]) / (x[2] - x[1])
            m2 = (y[3] - y[2]) / (x[3] - x[2])
        end
        i == 1 && return m1
        i == 3 && return m2
        return (m1 + m2) / 2  # i == 2
    end

    # General case: n ≥ 4
    # Need 4 secants for _akima_weighted_slope: m[i-2], m[i-1], m[i], m[i+1]
    # where m[k] = secant at interval [k, k+1].
    # For slope at grid index i, the 4 secants are at intervals:
    #   [i-2,i-1], [i-1,i], [i,i+1], [i+1,i+2]
    # At boundaries, fill missing secants with virtual (linearly extrapolated) secants.
    return _akima_local_4secant(x, y, i, n)
end

# Helper: compute the 4 secants needed for Akima slope at index i.
# Secant at interval j → j+1 is m_j = (y[j+1] - y[j]) / (x[j+1] - x[j])
# Real secants exist for j = 1..n-1. Virtual secants extend the sequence linearly.
@inline function _akima_local_4secant(x::AbstractVector{Tg}, y::AbstractVector{Tv}, i::Int, n::Int) where {Tg, Tv}
    # Akima slope at grid index i needs 4 secants: m[i-2], m[i-1], m[i], m[i+1]
    # where m[k] = secant at interval [k, k+1].
    # Real secant range: 1 to n-1.
    # Out-of-range secants are virtual (linearly extrapolated from the secant sequence).

    @inline _secant(j) = @inbounds (y[j + 1] - y[j]) / (x[j + 1] - x[j])

    # Compute each of the 4 needed secants — real if in [1, n-1], else virtual.
    # Needed intervals: i-2, i-1, i, i+1
    # Use a general approach that works for all boundary cases.

    # Helper: get secant at interval j, real or virtual
    # Virtual secants extend the real sequence linearly:
    #   m[0]   = 2*m[1] - m[2]
    #   m[-1]  = 2*m[0] - m[1]  = 3*m[1] - 2*m[2]
    #   m[n]   = 2*m[n-1] - m[n-2]
    #   m[n+1] = 2*m[n] - m[n-1] = 3*m[n-1] - 2*m[n-2]

    # Compute all 4 secants in order
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
