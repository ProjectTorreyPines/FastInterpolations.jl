# ========================================
# Cardinal Spline Slope Computation
# ========================================
# Cardinal slopes: d_k = (1 - tension) * (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])
# When tension=0, this is Catmull-Rom (simple central finite difference).
# O(n), single pass, in-place.

"""
    _cardinal_slopes!(dy, x, y, tension)

Compute cardinal spline slopes in-place.

# Arguments
- `dy::AbstractVector{Tv}`: output slopes (pre-allocated, length n)
- `x::AbstractVector{Tg}`: grid points (sorted, length n, ≥ 2)
- `y::AbstractVector{Tv}`: function values (length n)
- `tension::Tg`: tension parameter (0 = CatmullRom, 1 = zero slopes)

# Algorithm
- Interior (k = 2:n-1): `d_k = (1 - tension) * (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])`
- Endpoints: one-sided 2-point finite difference scaled by `(1 - tension)`
- Special case: n=2 → linear slopes scaled by `(1 - tension)`

# Complexity
O(n), single pass, zero allocation (writes into `dy`).
"""
function _cardinal_slopes!(
        dy::AbstractVector{Tv},
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        tension::Tg
    ) where {Tg <: AbstractFloat, Tv}
    n = length(x)
    @assert n >= 2 "Cardinal spline requires at least 2 points"
    @assert length(y) == n "y length must match x"
    @assert length(dy) == n "dy length must match x"

    scale = one(Tg) - tension

    # Special case: 2 points → linear
    if n == 2
        @inbounds begin
            δ = (y[2] - y[1]) / (x[2] - x[1])
            dy[1] = scale * δ
            dy[2] = scale * δ
        end
        return dy
    end

    # Left endpoint: one-sided 2-point FD
    @inbounds dy[1] = scale * (y[2] - y[1]) / (x[2] - x[1])

    # Interior: central finite difference
    @inbounds for k in 2:(n - 1)
        dy[k] = scale * (y[k + 1] - y[k - 1]) / (x[k + 1] - x[k - 1])
    end

    # Right endpoint: one-sided 2-point FD
    @inbounds dy[n] = scale * (y[n] - y[n - 1]) / (x[n] - x[n - 1])

    return dy
end
