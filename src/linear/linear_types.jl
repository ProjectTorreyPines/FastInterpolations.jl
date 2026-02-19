# ========================================
# Linear Interpolant Types
# ========================================
# Type definition for LinearInterpolant.
# Constructor and callable methods are in linear_interpolant.jl.

"""
    LinearInterpolant{Tg,Tv,X,Y,E,P}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32/Float64) - coordinates and geometry
- `Tv`: Value type - element type of y (can be Tg, Complex{Tg}, or other Number)
- `X<:AbstractVector{Tg}`: Grid vector type (preserves Range for O(1) lookup)
- `Y<:AbstractVector{Tv}`: Values vector type
- `E<:AbstractExtrapMode`: Extrapolation mode type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `extrap::E`: Extrapolation mode (NoExtrap(), ExtendExtrap(), ConstExtrap(), or WrapExtrap())
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
    E<:AbstractExtrapMode,
    P<:AbstractSearchPolicy
} <: AbstractInterpolant{Tg, Tv}
    x::X
    y::Y
    extrap::E  # Extrapolation mode (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function LinearInterpolant{Tg,Tv,X,Y,E,P}(
        x::X, y::Y, ev::E, search::P
    ) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, E<:AbstractExtrapMode, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        new{Tg,Tv,X,Y,E,P}(x, y, ev, search)
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
    extrap::Union{Symbol,AbstractExtrapMode}=NoExtrap(),
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy}
    extrap isa Symbol && Base.depwarn(_EXTRAP_SYMBOL_DEPWARN, :linear_interp)
    mode = extrap isa Symbol ? _symbol_to_extrap_mode(extrap) : extrap
    E = typeof(mode)
    return LinearInterpolant{Tg,Tv,X,Y,E,P}(x, y, mode, search)
end

# ========================================
# Type Aliases for Common Cases
# ========================================

"""Real-valued linear interpolant (matches original behavior)."""
const RealLinearInterpolant{T} = LinearInterpolant{T, T} where {T<:AbstractFloat}

"""Complex-valued linear interpolant."""
const ComplexLinearInterpolant{T} = LinearInterpolant{T, Complex{T}} where {T<:AbstractFloat}
