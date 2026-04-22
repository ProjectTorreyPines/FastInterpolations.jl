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

    # Inner constructor: promotes x/y, creates spacing, stores everything.
    # _store_grid: Vector → _convert_copy (single alloc), Range → _CachedRange (stack).
    # _convert_copy: same-type → copy(), different-type → Vector{T}(v).
    function LinearInterpolant(
            x::AbstractVector, y::AbstractVector, ev::E, search::P
        ) where {E <: AbstractExtrap, P <: AbstractSearchPolicy}
        length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
        Tg = _promote_grid_float(eltype(x), eltype(y))
        Tv = _value_type(eltype(y), Tg)
        xc = _store_grid(x, Tg)
        yc = _convert_copy(y, Tv)
        spacing = _create_spacing(xc)
        return new{Tg, Tv, typeof(xc), typeof(yc), typeof(spacing), E, P}(xc, yc, spacing, ev, search)
    end
end

# ========================================
# Outer Constructor: convenience kwarg wrapper
# ========================================
@inline function LinearInterpolant(
        x::AbstractVector,
        y::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    )
    # Materialize WrapExtrap{Nothing} against grid so kernels never see the
    # unmaterialized singleton when users construct the struct directly.
    return LinearInterpolant(x, y, _materialize_extrap(x, extrap), search)
end
