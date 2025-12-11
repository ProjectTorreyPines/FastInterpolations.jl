# Internal utility functions for FastInterpolations.jl

"""
    _find_interval_with_bounds(x::AbstractRange{FT}, xi::FT) where {FT<:AbstractFloat}

Find interpolation interval using O(1) direct calculation for uniform grids.

Returns `(idx, x0, x1)` where:
- `idx`: interval index in [1, length(x)-1]
- `x0`: left boundary value x[idx]
- `x1`: right boundary value x[idx+1]
"""
@inline function _find_interval_with_bounds(
    x::AbstractRange{FT},
    xi::FT
) where {FT<:AbstractFloat}
    n = length(x)
    x_min = first(x)
    dx = Base.step(x)

    # epsilon handles floating point errors (e.g., 1.999999 should map to index 2, not 1)
    idx = clamp(floor(Int, (xi - x_min) / dx + 1 + 10*eps(FT)), 1, n - 1)

    # Direct calculation to avoid expensive TwicePrecision indexing
    x0 = x_min + (idx - 1) * dx
    x1 = x0 + dx
    return idx, x0, x1
end

"""
    _find_interval_with_bounds(x::AbstractVector{FT}, xi::FT) where {FT<:AbstractFloat}

Find interpolation interval using O(log n) binary search for non-uniform grids.

Returns `(idx, x0, x1)` where:
- `idx`: interval index in [1, length(x)-1]
- `x0`: left boundary value x[idx]
- `x1`: right boundary value x[idx+1]
"""
@inline function _find_interval_with_bounds(
    x::AbstractVector{FT},
    xi::FT
) where {FT<:AbstractFloat}
    n = length(x)

    # Find interval using binary search
    @inbounds begin
        if xi <= x[1]
            idx = 1
        elseif xi >= x[end]
            idx = n - 1
        else
            lo, hi = 1, n
            while hi - lo > 1
                mid = (lo + hi) >> 1 # Fast division by 2
                if x[mid] <= xi
                    lo = mid
                else
                    hi = mid
                end
            end
            idx = lo
        end
    end

    # Return idx and boundary values for alpha calculation
    @inbounds x0, x1 = x[idx], x[idx + 1]
    return idx, x0, x1
end
