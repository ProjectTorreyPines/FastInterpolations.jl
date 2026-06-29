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
        dy::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        tension;
        bc::AbstractBC = NoBC()
    ) where {Tg}
    n = length(x)
    @assert n >= 2 "Cardinal spline requires at least 2 points"
    @assert length(y) == n "y length must match x"
    @assert length(dy) == n "dy length must match x"

    scale = one(Tg) - tension

    # Special case: 2 points. PeriodicBC routes through the wrap-aware central
    # FD helper (see PCHIP n=2 note for rationale).
    if n == 2
        if bc isa PeriodicBC
            @inbounds dy[1] = _cardinal_boundary_slope(x, y, 1, n, scale, bc)
            @inbounds dy[2] = _cardinal_boundary_slope(x, y, 2, n, scale, bc)
            return dy
        end
        @inbounds begin
            δ = _forward_secant(x, y, 1)
            dy[1] = scale * δ
            dy[2] = scale * δ
        end
        return dy
    end

    # Left endpoint: bc-dispatched helper.
    @inbounds dy[1] = _cardinal_boundary_slope(x, y, 1, n, scale, bc)

    # Interior: central finite difference (K=3, no wrap needed).
    @inbounds for k in 2:(n - 1)
        dy[k] = scale * _centered_secant(x, y, k)
    end

    # Right endpoint: same bc-dispatched helper. PeriodicBC{:inclusive} yields
    # dy[n] == dy[1] automatically (closed-cycle symmetry); :exclusive yields
    # a different value using the seam secant — the helper handles both via
    # `_periodic_secant`/`_periodic_cell_width`.
    @inbounds dy[n] = _cardinal_boundary_slope(x, y, n, n, scale, bc)

    return dy
end
