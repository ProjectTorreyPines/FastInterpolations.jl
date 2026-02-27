# ========================================
# LinearInterpolantND — Interpolant Construction
# ========================================
#
# Constructor API and helpers for LinearInterpolantND.
# One-shot evaluation is in linear_nd_oneshot.jl.

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
- `extrap=NoExtrap()`: Extrapolation mode (`NoExtrap()`, `ConstExtrap()`, `ExtendExtrap()`, `WrapExtrap()`) or per-axis tuple
- `search=AutoSearch()`: Search policy or per-axis tuple

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
    extrap=(NoExtrap(), WrapExtrap()),
    search=(BinarySearch(), LinearBinarySearch())
)
```
"""
function linear_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N};
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
    searches = _resolve_search_nd(search, Val(N))

    extrap_vals = _resolve_extrap_nd(extrap, nothing, Val(N))
    return LinearInterpolantND{Tg, Tv, N,
        typeof(grids_typed), typeof(spacings), typeof(extrap_vals), typeof(searches)}(
        grids_typed, spacings, Array(data_typed), extrap_vals, searches
    )
end
