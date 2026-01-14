# ========================================
# CubicSplineCache Constructor
# ========================================
# Separated from cubic_interp.jl for clarity.
# Creates cache with pre-computed LU factorization.

"""
    CubicSplineCache(x::AbstractVector{T}; bc=NaturalBC()) where {T<:AbstractFloat}

Construct a cubic spline cache for grid points `x`.

Pre-computes and factorizes the tridiagonal matrix that depends only on x geometry.
This factorization can be reused for interpolating different y vectors.

# Arguments
- `x::AbstractVector{T}`: Grid points (must be sorted, length >= 3)
- `bc`: Boundary condition specification:
  - `NaturalBC()`: Zero curvature at both ends (default)
  - `ClampedBC()`: Zero slope at both ends
  - `PeriodicBC()`: Periodic boundary condition
  - `Deriv1(val)` or `Deriv2(val)`: Symmetric BC (same at both ends)
  - `BCPair(Deriv1(v1), Deriv2(v2))`: Asymmetric BC pair

# Example
```julia
x = range(0.0, 1.0, 51)
cache = CubicSplineCache(x)                              # Natural BC (default)
cache = CubicSplineCache(x; bc=ClampedBC())              # Zero slope at both ends
cache = CubicSplineCache(x; bc=Deriv1(0.5))              # Slope=0.5 at both ends
cache = CubicSplineCache(x; bc=BCPair(Deriv1(0.5), Deriv2(0)))   # Mixed: slope left, natural right
cache_periodic = CubicSplineCache(x; bc=PeriodicBC())   # Periodic BC

# Reuse for multiple y vectors
y1 = sin.(x)
y2 = cos.(x)
result1 = cubic_interp(cache, y1, [0.25, 0.75])
result2 = cubic_interp(cache, y2, [0.25, 0.75])
```
"""
function CubicSplineCache(x::AbstractVector{T}; bc::AbstractBC=NaturalBC()) where {T<:AbstractFloat}
    # Periodic BC
    if _is_periodic_bc(bc)
        return _build_periodic_cache(x)
    end

    # Normalize BC to BCPair (NaturalBC/ClampedBC → BCPair, PointBC → symmetric BCPair)
    bc_normalized = _normalize_bc(bc, T)

    # All non-periodic BC use unified BCPair path
    return _build_derivative_bc_cache(x, bc_normalized.left, bc_normalized.right)
end
