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
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        store::StorePolicy = StorePolicy()
    ) where {N, Tv_raw}
    # Validate grid dimensions
    _validate_nd_grids(grids, data)

    # Raw storage (no Float widening): `Tg = promote_type(eltype.(grids)...)`,
    # `Tv = eltype(data)`. Kernel handles return-type widening via per-axis
    # `* one(dL_d)`.
    grids_typed, _, Tv = _nd_promote_grids_raw(grids, data)
    data_typed = data

    # Resolve per-axis configuration
    bcs = _resolve_bcs_nd(bc, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Extend `:exclusive` axes/data to closed-cycle (n+1) layout; periodic
    # bcs are promoted to `:extended` by `_prepare_periodic_nd`, then per-axis
    # `_policy_axes` wraps store-aware (raw → wrapped, pre-wrapped → passthrough).
    grids_typed, data_typed, bcs_post = _prepare_periodic_nd(grids_typed, data_typed, bcs)
    grids_typed = _policy_axes(grids_typed, bcs_post, store)

    # Per-axis extrap: validate + auto-promote `WrapExtrap` on periodic axes.
    extrap_vals = _resolve_extrap(extrap, bcs, Val(N), Tv)
    extrap_vals = map(_resolve_extrap, extrap_vals, grids_typed)
    return ConstantInterpolantND(grids_typed, data_typed, extrap_vals, sides, searches; bcs = bcs_post, store = store)
end

# N=1 collapse: a 1-axis grid tuple forwards to the genuine 1D constant path (lean
# 1D batch loop; per-axis 1-tuple kwargs unwrap to scalar). More specific than the
# `NTuple{N}` method above, so it only claims N=1. See linear_nd_interpolant.jl.
@inline constant_interp(grids::Tuple{AbstractVector}, data::AbstractVector; kwargs...) =
    constant_interp(only(grids), data; _unwrap_nd_kwargs(values(kwargs))...)

# N=1 scalar one-shot: bare scalar → scalar query `(q,)` → ND scalar one-shot
# (scalar output, not `[val]`). See linear_nd_interpolant.jl.
@inline constant_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Number; kwargs...) =
    constant_interp(grids, data, (q,); kwargs...)

# N=1 batch one-shot → lean 1D batch one-shot (bit-identical). See linear_nd_interpolant.jl.
@inline constant_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractArray; kwargs...) =
    constant_interp(only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
@inline constant_interp!(output::AbstractArray, grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractArray; kwargs...) =
    constant_interp!(output, only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
# Single-axis SoA `(xv,)` → 1D batch. See linear_nd_interpolant.jl.
@inline constant_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractArray}; kwargs...) =
    constant_interp(only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)
@inline constant_interp!(output::AbstractArray, grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractArray}; kwargs...) =
    constant_interp!(output, only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)
