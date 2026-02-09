# ========================================
# LinearInterpolantND API
# ========================================
#
# Public API for N-dimensional linear (multilinear) interpolation.
# Extends linear_interp() to accept tuple-grid input.

# ========================================
# Grid Conversion Helpers
# ========================================

"""
    _convert_grid_linear(x, Tg) -> AbstractVector{Tg}

Convert grid to target element type, preserving Range structure.
"""
function _convert_grid_linear(x::AbstractRange, ::Type{Tg}) where {Tg}
    eltype(x) === Tg && return x
    return range(Tg(first(x)), Tg(last(x)), length(x))
end

function _convert_grid_linear(x::AbstractVector, ::Type{Tg}) where {Tg}
    eltype(x) === Tg && return x
    return Tg.(x)
end

# ========================================
# Constructor API
# ========================================

"""
    linear_interp(grids::NTuple{N,AbstractVector}, data::AbstractArray{<:Any,N}; kwargs...)

Create an N-dimensional multilinear interpolant with tuple-grid API.

Performs bilinear interpolation in 2D, trilinear in 3D, and n-linear in N dimensions.
The interpolation is exact at grid points and linearly blended between them.

# Arguments
- `grids`: Tuple of grid vectors, one per dimension (e.g., `(x, y)` or `(x, y, z)`)
- `data`: N-dimensional data array where `size(data, d) == length(grids[d])`

# Keyword Arguments
- `extrap=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`, `:wrap`) or per-axis tuple
- `search=Binary()`: Search policy or per-axis tuple

# Returns
- `LinearInterpolantND{Tg,Tv,N}`: Callable interpolant object

# Examples
```julia
# 2D bilinear interpolation
x = range(0.0, 1.0, 11)
y = range(0.0, 2.0, 21)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

itp = linear_interp((x, y), data)
val = itp((0.5, 1.0))              # Scalar query
vals = itp((xs, ys))               # Batch SoA
vals = itp(points)                 # Batch AoS

# Derivatives
∂f_∂x = itp((0.5, 1.0); deriv=Val((1,0)))  # ∂f/∂x
∂f_∂y = itp((0.5, 1.0); deriv=Val((0,1)))  # ∂f/∂y
grad = (∂f_∂x, ∂f_∂y)                       # Gradient

# 3D trilinear interpolation
z = range(0.0, 1.0, 11)
data3d = [xi + yj + zk for xi in x, yj in y, zk in z]
itp3d = linear_interp((x, y, z), data3d)
val3d = itp3d((0.5, 1.0, 0.3))

# Per-axis configuration
itp = linear_interp((x, y), data;
    extrap=(:none, :wrap),
    search=(Binary(), LinearBinary())
)
```
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary()
) where {N, Tv_raw}
    # Validate grid dimensions
    _validate_nd_grids(grids, data)

    # Determine grid type (promote Int → Float64 for consistency with 1D API)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64

    # Convert grids to target type (preserving Range structure)
    grids_typed = _convert_grids_typed_linear(grids, Tg)

    # Create spacings
    spacings = _create_spacings_typed(grids_typed)

    # Promote data type
    Tv = _value_type(Tv_raw, Tg)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    # Resolve per-axis configuration
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Convert symbols to Val types
    extrap_vals = _to_extrap_vals_linear(extraps)

    return LinearInterpolantND{Tg, Tv, N,
        typeof(grids_typed), typeof(spacings), typeof(extrap_vals), typeof(searches)}(
        grids_typed, spacings, Array(data_typed), extrap_vals, searches
    )
end

# ========================================
# Grid Conversion (Generated for Zero-Alloc)
# ========================================

@generated function _convert_grids_typed_linear(grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    exprs = [:(FastInterpolations._convert_grid_linear(grids[$i], Tg)) for i in 1:N]
    :(($(exprs...),))
end

# ========================================
# Val Type Conversion
# ========================================

@inline function _to_extrap_vals_linear(extraps::NTuple{N, Symbol}) where {N}
    return ntuple(i -> Val(extraps[i]), Val(N))
end

# ========================================
# One-Shot API
# ========================================

"""
    linear_interp(grids, data, query; kwargs...)

One-shot N-dimensional linear interpolation (scalar query).
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    query::Tuple{Vararg{Real, N}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {N}
    itp = linear_interp(grids, data; extrap=extrap, search=search)
    return itp(query; deriv=deriv)
end

"""
    linear_interp(grids, data, queries::NTuple{N,AbstractVector}; kwargs...)

One-shot N-dimensional linear interpolation (batch SoA query).
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    queries::NTuple{N, AbstractVector{<:Real}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {N}
    itp = linear_interp(grids, data; extrap=extrap, search=search)
    return itp(queries; deriv=deriv)
end

"""
    linear_interp(grids, data, queries::AbstractVector{<:NTuple}; kwargs...)

One-shot N-dimensional linear interpolation (batch AoS query).
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {N}
    itp = linear_interp(grids, data; extrap=extrap, search=search)
    return itp(queries; deriv=deriv)
end
