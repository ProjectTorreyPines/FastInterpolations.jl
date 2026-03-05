# ========================================
# ConstantInterpolantND — Interpolant Construction
# ========================================
#
# Constructor API and helpers for ConstantInterpolantND.
# One-shot evaluation is in constant_nd_oneshot.jl.

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
- `side=NearestSide()`: Side selection mode (`NearestSide()`, `LeftSide()`, `RightSide()`) or per-axis tuple
- `extrap=NoExtrap()`: Extrapolation mode (`NoExtrap()`, `ClampedExtrap()`, `ExtendExtrap()`, `WrapExtrap()`) or per-axis tuple
- `search=AutoSearch()`: Search policy or per-axis tuple

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
    side=(LeftSide(), RightSide()),
    extrap=(NoExtrap(), WrapExtrap()),
    search=(BinarySearch(), LinearBinarySearch())
)
```
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N};
    side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
    extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch()
) where {N, Tv_raw}
    # Validate grid dimensions
    _validate_nd_grids(grids, data)

    # Determine grid type (promote Int → Float64 for consistency with 1D API)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64

    # Convert grids to target type (preserving Range structure)
    grids_typed = _convert_grids_typed(grids, Tg)

    # Create spacings
    spacings = _create_spacings_typed(grids_typed)

    # Promote data type
    Tv = _value_type(Tv_raw, Tg)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    # Resolve per-axis configuration
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    extrap_vals = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    return ConstantInterpolantND{Tg, Tv, N,
        typeof(grids_typed), typeof(spacings), typeof(extrap_vals), typeof(sides), typeof(searches)}(
        grids_typed, spacings, Array(data_typed), extrap_vals, sides, searches
    )
end
