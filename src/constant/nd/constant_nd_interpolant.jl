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
- `extrap=NoExtrap()`: Extrapolation mode (`NoExtrap()`, `ClampExtrap()`, `ExtendExtrap()`, `WrapExtrap()`) or per-axis tuple
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
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch()
    ) where {N, Tv_raw}
    # Validate grid dimensions
    _validate_nd_grids(grids, data)

    # Selection kernel → raw eltype contract (Int in → Int out), N-axis
    # generalization of the 1D policy: `Tg = promote_type(eltype.(grids)...)`
    # without Float widening, `Tv = eltype(data)`.
    grids_typed, _, Tv = _nd_promote_grids_raw(grids, data)
    data_typed = data

    # Resolve per-axis configuration
    bcs = _resolve_bcs_nd(bc, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Extend `:exclusive` axes/data to `:inclusive` form, then per-axis
    # `_cache_axis` (raw → wrapped, pre-wrapped → passthrough).
    grids_typed, data_typed, bcs_post = _prepare_periodic_nd(grids_typed, data_typed, bcs)
    grids_typed = map(_cache_axis, grids_typed, bcs_post)

    # Per-axis extrap: validate + auto-promote `WrapExtrap` on periodic axes.
    extrap_vals = _resolve_extrap(extrap, bcs, Val(N), Tv)
    extrap_vals = map(_resolve_extrap, extrap_vals, grids_typed)
    return ConstantInterpolantND(grids_typed, data_typed, extrap_vals, sides, searches; bcs = bcs_post)
end
