# ========================================
# Linear Interpolant Types
# ========================================
# Type definition for LinearInterpolant.
# Constructor and callable methods are in linear_interpolant.jl.

"""
    LinearInterpolant{T,X,Y,P}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `extrap::Val`: Extrapolation mode (Val(:none), Val(:extension), Val(:constant), or Val(:wrap))
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
# Create interpolator (minimal allocation)
itp = linear_interp(x, y)  # default extrap=:none, search=Binary()

# Create with custom search policy (baked-in default)
itp = linear_interp(x, y; search=LinearBinary())

# Use in broadcast (fused, no intermediate arrays)
result = @. coef * itp(rho) * other_terms

# Reuse interpolator multiple times
vals1 = itp.(query_points1)
vals2 = @. compute(itp(query_points2))

# Extrapolation options
itp_ext = linear_interp(x, y; extrap=:extension)  # linear extrap
itp_const = linear_interp(x, y; extrap=:constant)  # clamp to boundary values
itp_wrap = linear_interp(x, y; extrap=:wrap)  # wrap to domain
val = itp_wrap(2.5)  # wraps to domain

# Override search policy at call time
itp(0.5; search=Binary())  # override stored policy
```
"""
struct LinearInterpolant{T<:AbstractFloat,X<:AbstractVector{T},Y<:AbstractVector{T},P<:AbstractSearchPolicy} <: AbstractInterpolant{T}
    x::X
    y::Y
    extrap::ExtrapVal  # Extrapolation mode (concrete union for union-splitting)
    search_policy::P  # Default search policy (immutable, thread-safe)

    function LinearInterpolant(
        x::X,
        y::Y;
        extrap::Symbol=:none,
        search::P=Binary()
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"

        # Manual dispatch to avoid union-splitting with 4 Val types
        @_dispatch_extrap extrap => ev begin
            return new{T,X,Y,P}(x, y, ev, search)
        end
    end
end
