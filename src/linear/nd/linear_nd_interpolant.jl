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
- `extrap=NoExtrap()`: Extrapolation mode (`NoExtrap()`, `ClampExtrap()`, `ExtendExtrap()`, `WrapExtrap()`) or per-axis tuple
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
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        store::StorePolicy = StorePolicy()
    ) where {N, Tv_raw}
    # Validate grid dimensions
    _validate_nd_grids(grids, data)

    # Promote grid/data types
    grids_typed, _, Tv, _ = _nd_promote_grids(grids, data)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    # Resolve per-axis configuration
    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Extend `:exclusive` axes/data to closed-cycle (n+1) layout; periodic
    # bcs are promoted to `:extended` by `_prepare_periodic_nd`, then per-axis
    # `_cache_axis` wraps (raw → wrapped, pre-wrapped → passthrough).
    grids_typed, data_typed, bcs_post = _prepare_periodic_nd(grids_typed, data_typed, bcs)
    grids_typed = map(_cache_axis, grids_typed, bcs_post)

    # Per-axis extrap: validate + auto-promote `WrapExtrap` on periodic axes.
    extrap_vals = _resolve_extrap(extrap, bcs, Val(N), Tv)
    extrap_vals = map(_resolve_extrap, extrap_vals, grids_typed)
    return LinearInterpolantND(grids_typed, data_typed, extrap_vals, searches; bcs = bcs_post, store = store)
end
