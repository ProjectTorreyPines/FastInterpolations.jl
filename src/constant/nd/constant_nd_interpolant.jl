# ========================================
# ConstantInterpolantND — Interpolant Construction
# ========================================
#
# Constructor API and helpers for ConstantInterpolantND.
# One-shot evaluation is in constant_nd_oneshot.jl.

# ========================================
# Grid Conversion Helpers
# ========================================

"""
    _convert_grid_constant(x, Tg) -> AbstractVector{Tg}

Convert grid to target element type, preserving Range structure.
"""
function _convert_grid_constant(x::AbstractRange, ::Type{Tg}) where {Tg}
    eltype(x) === Tg && return x
    return range(Tg(first(x)), Tg(last(x)), length(x))
end

function _convert_grid_constant(x::AbstractVector, ::Type{Tg}) where {Tg}
    eltype(x) === Tg && return x
    return Tg.(x)
end

# ========================================
# Constructor API
# ========================================

"""
    constant_interp(grids::NTuple{N,AbstractVector}, data::AbstractArray{<:Any,N}; kwargs...)

Create an N-dimensional constant interpolant with tuple-grid API.

# Arguments
- `grids`: Tuple of grid vectors, one per dimension (e.g., `(x, y)` or `(x, y, z)`)
- `data`: N-dimensional data array where `size(data, d) == length(grids[d])`

# Keyword Arguments
- `side=:nearest`: Side selection mode (`:nearest`, `:left`, `:right`) or per-axis tuple
- `extrap=:none`: Extrapolation mode (`:none`, `:constant`, `:extension`, `:wrap`) or per-axis tuple
- `search=Binary()`: Search policy or per-axis tuple

# Returns
- `ConstantInterpolantND{Tg,Tv,N}`: Callable interpolant object

# Examples
```julia
# 2D constant interpolation
x = [0.0, 1.0, 2.0]
y = [0.0, 1.0, 2.0, 3.0]
data = rand(3, 4)

itp = constant_interp((x, y), data)
val = itp((0.5, 1.5))           # Scalar query
vals = itp((xs, ys))            # Batch SoA
vals = itp(points)              # Batch AoS

# Per-axis configuration
itp = constant_interp((x, y), data;
    side=(:left, :right),
    extrap=(:none, :wrap),
    search=(Binary(), LinearBinary())
)
```
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N};
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary()
) where {N, Tv_raw}
    # Validate grid dimensions
    _validate_nd_grids(grids, data)

    # Determine grid type (promote Int → Float64 for consistency with 1D API)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64

    # Convert grids to target type (preserving Range structure)
    grids_typed = _convert_grids_typed_constant(grids, Tg)

    # Create spacings
    spacings = _create_spacings_typed(grids_typed)

    # Promote data type
    Tv = _value_type(Tv_raw, Tg)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    # Resolve per-axis configuration
    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Convert symbols to Val types
    extrap_vals = _to_extrap_vals(extraps)
    side_vals = _to_side_vals(sides)

    return ConstantInterpolantND{Tg, Tv, N,
        typeof(grids_typed), typeof(spacings), typeof(extrap_vals), typeof(side_vals), typeof(searches)}(
        grids_typed, spacings, Array(data_typed), extrap_vals, side_vals, searches
    )
end

# ========================================
# Grid Conversion (Generated for Zero-Alloc)
# ========================================

@generated function _convert_grids_typed_constant(grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    exprs = [:(FastInterpolations._convert_grid_constant(grids[$i], Tg)) for i in 1:N]
    :(($(exprs...),))
end

# ========================================
# Val Type Conversion
# ========================================

@inline function _to_extrap_vals(extraps::NTuple{N, Symbol}) where {N}
    return ntuple(i -> _to_extrap_val(extraps[i]), Val(N))
end

@inline _to_extrap_val(s::Symbol) = Val(s)

@inline function _to_side_vals(sides::NTuple{N, Symbol}) where {N}
    return ntuple(i -> _to_side_val(sides[i]), Val(N))
end

@inline _to_side_val(s::Symbol) = Val(s)
