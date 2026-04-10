# ========================================
# Linear Interpolant Types
# ========================================
# Type definition for LinearInterpolant.
# Constructor and callable methods are in linear_interpolant.jl.

"""
    LinearInterpolant{Tg,Tv,X,Y,S,E,P}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg`: Grid type — normally `Float32`/`Float64`, but **unconstrained** to admit
        duck-typed grid scalars (e.g. `ForwardDiff.Dual`) that satisfy the grid
        arithmetic/ordering protocol (`-`, `inv`, `*`, `<`). Duck grids take a
        pass-through normalization path and build a differentiable `VectorSpacing{Tg}`.
- `Tv`: Value type (unconstrained)
- `X<:AbstractVector{Tg}`: Grid vector type (preserves Range for O(1) lookup)
- `Y<:AbstractVector{Tv}`: Values vector type
- `S<:AbstractGridSpacing{Tg}`: Grid spacing type (ScalarSpacing or VectorSpacing)
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `spacing::S`: Precomputed grid spacing (avoids TwicePrecision overhead on Range grids)
- `extrap::E`: Extrapolation mode (NoExtrap(), ExtendExtrap(), ClampExtrap(), or WrapExtrap())
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
# Create interpolator (minimal allocation)
itp = linear_interp(x, y)  # default extrap=NoExtrap(), search=AutoSearch()

# Create with custom search policy (overrides default AutoSearch)
itp = linear_interp(x, y; search=BinarySearch())

# Complex-valued interpolation
y_complex = exp.(2im .* x)
itp_c = linear_interp(x, y_complex)  # Works natively with Complex

# Use in broadcast (fused, no intermediate arrays)
result = @. coef * itp(rho) * other_terms

# Reuse interpolator multiple times
vals1 = itp.(query_points1)
vals2 = @. compute(itp(query_points2))

# Extrapolation options
itp_ext = linear_interp(x, y; extrap=ExtendExtrap())  # linear extrap
itp_const = linear_interp(x, y; extrap=ClampExtrap())  # clamp to boundary values
itp_wrap = linear_interp(x, y; extrap=WrapExtrap())  # wrap to domain
val = itp_wrap(2.5)  # wraps to domain

# Override search policy at call time
itp(0.5; search=BinarySearch())  # override stored policy
```
"""
struct LinearInterpolant{
        Tg,
        Tv,
        X <: AbstractVector{Tg},
        Y <: AbstractVector{Tv},
        S <: AbstractGridSpacing{Tg},
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
    } <: AbstractInterpolant1D{Tg, Tv}
    x::X
    y::Y
    spacing::S  # Precomputed grid spacing (ScalarSpacing for Range, VectorSpacing for Vector)
    extrap::E  # Extrapolation mode (compile-time specialized)
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function LinearInterpolant{Tg, Tv, X, Y, S, E, P}(
            x::AbstractVector{Tg}, y::AbstractVector{Tv}, spacing::S, ev::E, search::P
        ) where {Tg, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        # Copy to ensure immutability: once constructed, the interpolant owns
        # its data and returns identical results regardless of external mutation.
        # copy() on immutable Range types is a no-op (zero allocation).
        # typeof() rebinds X/Y to the post-copy concrete type (e.g. SubArray → Vector).
        xc, yc = copy(x), copy(y)
        return new{Tg, Tv, typeof(xc), typeof(yc), S, E, P}(xc, yc, spacing, ev, search)
    end
end

# ========================================
# Outer Constructor: typed inputs only
# ========================================
#
# PERFORMANCE: Typed signature + @inline enables compile-time specialization.
# Use linear_interp() for automatic type promotion from Real inputs.
@inline function LinearInterpolant(
        x::X,
        y::Y;
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch()
    ) where {Tg, Tv, X <: AbstractVector{Tg}, Y <: AbstractVector{Tv}, P <: AbstractSearchPolicy}
    E = typeof(extrap)
    spacing = _create_spacing(x)
    S = typeof(spacing)
    return LinearInterpolant{Tg, Tv, X, Y, S, E, P}(x, y, spacing, extrap, search)
end

# ========================================
# Type Aliases for Common Cases
# ========================================

"""Real-valued linear interpolant (matches original behavior)."""
const RealLinearInterpolant{T} = LinearInterpolant{T, T} where {T <: AbstractFloat}

"""Complex-valued linear interpolant."""
const ComplexLinearInterpolant{T} = LinearInterpolant{T, Complex{T}} where {T <: AbstractFloat}
