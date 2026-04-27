# ========================================
# PCHIP Slope Computation (Fritsch-Carlson 1980)
# ========================================
# Monotone-preserving local slope estimation.
# O(n), single pass, in-place into pre-allocated dy buffer.
#
# Reference: Fritsch & Carlson, "Monotone Piecewise Cubic Interpolation",
# SIAM J. Numer. Anal. 17(2), 1980, pp. 238-246.
#
# Note: Requires Tv to support sign(), abs(), zero(), and division.
# This restricts Tv to Real subtypes (Int, Float32, Float64, etc.).
# For non-real value types (SVector, Complex, duck types), use
# hermite_interp(x, y, dy) with user-computed slopes instead.

"""
    _pchip_endpoint_slope(h1, h2, δ1, δ2)

Compute one-sided 3-point finite difference with monotonicity clamping
for PCHIP endpoints.

# Arguments
- `h1`: width of the boundary interval
- `h2`: width of the next interval
- `δ1`: secant slope of the boundary interval
- `δ2`: secant slope of the next interval

# Returns
Clamped endpoint slope that preserves monotonicity.
"""
@inline function _pchip_endpoint_slope(h1::Tg, h2::Tg, δ1::Tv, δ2::Tv) where {Tg, Tv}
    # 3-point one-sided finite difference
    d = ((2 * h1 + h2) * δ1 - h1 * δ2) / (h1 + h2)

    # Monotonicity clamping
    if sign(d) != sign(δ1)
        d = zero(d)
    elseif sign(δ1) != sign(δ2) && abs(d) > abs(3 * δ1)
        d = 3 * δ1
    end
    return d
end

"""
    _pchip_slopes!(dy, x, y)

Compute PCHIP (Fritsch-Carlson) monotone-preserving slopes in-place.

# Arguments
- `dy::AbstractVector{Tv}`: output slopes (pre-allocated, length n)
- `x::AbstractVector{Tg}`: grid points (sorted, length n)
- `y::AbstractVector{Tv}`: function values (length n)

# Algorithm
- Interior: weighted harmonic mean with zero-clamping at local extrema
- Endpoints: 3-point one-sided finite difference with monotonicity clamping
- Special case: n=2 → linear slopes

# Complexity
O(n), single pass, zero allocation (writes into `dy`).
"""
function _pchip_slopes!(
        dy::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector;
        bc::AbstractBC = NoBC()
    ) where {Tg}
    n = length(x)
    @assert n >= 2 "PCHIP requires at least 2 points"
    @assert length(y) == n "y length must match x"
    @assert length(dy) == n "dy length must match x"

    # Special case: 2 points → linear
    if n == 2
        @inbounds begin
            δ = (y[2] - y[1]) / (x[2] - x[1])
            dy[1] = δ
            dy[2] = δ
        end
        return dy
    end

    # Compute secant slopes for first two intervals (needed for first interior)
    @inbounds h_prev = x[2] - x[1]
    @inbounds δ_prev = (y[2] - y[1]) / h_prev

    @inbounds h_curr = x[3] - x[2]
    @inbounds δ_curr = (y[3] - y[2]) / h_curr

    # Left endpoint: bc-dispatched helper.
    # NoBC: one-sided 3-point FD with monotonicity clamping.
    # PeriodicBC: closed-cycle interior formula via wrap-aware abstraction.
    @inbounds dy[1] = _pchip_boundary_slope(x, y, 1, n, bc)

    # Interior slopes (k = 2:n-1) — unchanged. K=3 stencil never crosses join.
    @inbounds for k in 2:(n - 1)
        if sign(δ_prev) != sign(δ_curr)
            # Local extremum: zero slope preserves monotonicity
            dy[k] = zero(eltype(dy))
        else
            # Weighted harmonic mean (Fritsch-Carlson formula)
            w1 = 2 * h_curr + h_prev
            w2 = h_curr + 2 * h_prev
            dy[k] = (w1 + w2) / (w1 / δ_prev + w2 / δ_curr)
        end

        # Advance to next interval (k < n-1 means there's a next interval)
        if k < n - 1
            h_prev = h_curr
            δ_prev = δ_curr
            h_curr = x[k + 2] - x[k + 1]
            δ_curr = (y[k + 2] - y[k + 1]) / h_curr
        end
    end

    # Right endpoint: same bc-dispatched helper. The wrap-aware abstraction
    # makes this self-consistent across endpoints:
    # - PeriodicBC{:inclusive}: helper at i=n yields the same value as at i=1
    #   (closed cycle on n-1 cells → m_{n-1}, m_n=m_1 produces same pair) →
    #   dy[n] == dy[1] automatically.
    # - PeriodicBC{:exclusive}: helper at i=n uses (m_{n-1}, m_n=seam) which
    #   differs from i=1's (m_0=seam, m_1) — dy[1] ≠ dy[n] in general, both
    #   correctly wrap-aware via the seam secant.
    # - NoBC: helper falls back to the original one-sided FD.
    @inbounds dy[n] = _pchip_boundary_slope(x, y, n, n, bc)

    return dy
end
