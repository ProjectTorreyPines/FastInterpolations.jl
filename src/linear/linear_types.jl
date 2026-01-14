# ========================================
# Linear Interpolant Types
# ========================================
# Type definition for LinearInterpolant.
# Constructor and callable methods are in linear_interpolant.jl.

"""
    LinearInterpolant{T,X,Y}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `mode::Val`: Evaluation mode (Val(:none), Val(:extension), Val(:constant), or Val(:wrap))

# Usage
```julia
# Create interpolator (minimal allocation)
itp = linear_interp(x, y)  # default extrap=:none (throws error if outside domain)

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
```
"""
struct LinearInterpolant{T<:AbstractFloat,X<:AbstractVector{T},Y<:AbstractVector{T}} <: AbstractInterpolant{T}
    x::X
    y::Y
    mode::ExtrapVal  # Evaluation mode (concrete union for union-splitting)

    function LinearInterpolant(
        x::X,
        y::Y;
        extrap::Symbol=:none
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        @assert length(x) == length(y) "x and y must have same length"

        # Manual dispatch to avoid union-splitting with 4 Val types
        @_dispatch_extrap extrap => ev begin
            return new{T,X,Y}(x, y, ev)
        end
    end
end
