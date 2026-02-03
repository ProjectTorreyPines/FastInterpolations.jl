# ========================================
# Linear Interpolant Types
# ========================================
# Type definition for LinearInterpolant.
# Constructor and callable methods are in linear_interpolant.jl.

"""
    LinearInterpolant{Tg,Tv,X,Y,P}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32/Float64) - coordinates and geometry
- `Tv`: Value type - element type of y (can be Tg, Complex{Tg}, or other Number)
- `X<:AbstractVector{Tg}`: Grid vector type (preserves Range for O(1) lookup)
- `Y<:AbstractVector{Tv}`: Values vector type
- `P<:AbstractSearchPolicy`: Search policy type

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

# Complex-valued interpolation
y_complex = exp.(2im .* x)
itp_c = linear_interp(x, y_complex)  # Works natively with Complex

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
struct LinearInterpolant{
    Tg<:AbstractFloat,
    Tv,
    X<:AbstractVector{Tg},
    Y<:AbstractVector{Tv},
    P<:AbstractSearchPolicy
} <: AbstractInterpolant{Tg, Tv}
    x::X
    y::Y
    extrap::ExtrapVal  # Extrapolation mode (concrete union for union-splitting)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function LinearInterpolant{Tg,Tv,X,Y,P}(
        x::X, y::Y, ev::ExtrapVal, search::P
    ) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        new{Tg,Tv,X,Y,P}(x, y, ev, search)
    end
end

# ========================================
# Outer Constructor: typed inputs only
# ========================================
# - Symbol → Val dispatch
# - Call inner constructor
#
# PERFORMANCE: Typed signature + @inline enables compile-time specialization.
# Use linear_interp() for automatic type promotion from Real inputs.
@inline function LinearInterpolant(
    x::X,
    y::Y;
    extrap::Symbol=:none,
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy}
    @_dispatch_extrap extrap => ev begin
        return LinearInterpolant{Tg,Tv,X,Y,P}(x, y, ev, search)
    end
end

# ========================================
# Type Aliases for Common Cases
# ========================================

"""Real-valued linear interpolant (matches original behavior)."""
const RealLinearInterpolant{T} = LinearInterpolant{T, T} where {T<:AbstractFloat}

"""Complex-valued linear interpolant."""
const ComplexLinearInterpolant{T} = LinearInterpolant{T, Complex{T}} where {T<:AbstractFloat}
