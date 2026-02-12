# ========================================
# ConstantInterpolantND API
# ========================================
#
# Public API for N-dimensional constant interpolation.
# Extends constant_interp() to accept tuple-grid input.

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

# ========================================
# ZERO-ALLOC ONE-SHOT IMPLEMENTATION
# ========================================
#
# Standalone evaluation functions that bypass Interpolant construction.
# Zero heap allocation for scalar queries after warmup.

"""
    _constant_interp_nd_oneshot(grids, data, query, extraps_val, side_vals, searches)

Zero-allocation scalar one-shot ND constant evaluation.
Evaluates directly from grids + data without constructing a ConstantInterpolantND.
"""
@inline function _constant_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    extraps_val::NTuple{N, Val},
    side_vals::NTuple{N, SideVal},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    spacings = _create_spacings_typed(grids)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
    return _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
end

"""
    _constant_interp_nd_oneshot_soa(grids, data, queries, extraps_val, side_vals, searches)

SoA batch one-shot ND constant evaluation.
Only allocates the output vector (return value).
"""
function _constant_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    extraps_val::NTuple{N, Val},
    side_vals::NTuple{N, SideVal},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end
    spacings = _create_spacings_typed(grids)
    output = Vector{Tv}(undef, n_queries)
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

"""
    _constant_interp_nd_oneshot_aos(grids, data, queries, extraps_val, side_vals, searches)

AoS batch one-shot ND constant evaluation.
Only allocates the output vector (return value).
"""
function _constant_interp_nd_oneshot_aos(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    extraps_val::NTuple{N, Val},
    side_vals::NTuple{N, SideVal},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries)
    spacings = _create_spacings_typed(grids)
    output = Vector{Tv}(undef, n_queries)
    @inbounds for k in 1:n_queries
        query_k = queries[k]
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        output[k] = _constant_nd_kernel(data, spacings, side_vals, indices, q_eval, Ls)
    end
    return output
end

# ========================================
# Derivative Check Helper
# ========================================

@inline _is_any_deriv(d::Int) = d != 0
@inline _is_any_deriv(::Val{T}) where {T} = any(!=(0), T)
@inline _is_any_deriv(d::NTuple{N, Int}) where {N} = any(!=(0), d)

# ========================================
# One-Shot API
# ========================================

"""
    constant_interp(grids, data, query; kwargs...)

One-shot N-dimensional constant interpolation (scalar query).
Zero-allocation after warmup.
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}};
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    # Any derivative of constant interpolation is zero
    if _is_any_deriv(deriv)
        return zero(Tv)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed_constant(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))
    side_vals = _to_side_vals(sides)

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        return _constant_interp_nd_oneshot(
            grids_typed, data, query, extraps_val, side_vals, searches)::Tv
    end
end

"""
    constant_interp(grids, data, queries::NTuple{N,AbstractVector}; kwargs...)

One-shot N-dimensional constant interpolation (batch SoA query).
Only allocates the output vector.
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::NTuple{N, AbstractVector{<:Real}};
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    if _is_any_deriv(deriv)
        n_queries = length(queries[1])
        return zeros(Tv, n_queries)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed_constant(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))
    side_vals = _to_side_vals(sides)

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        return _constant_interp_nd_oneshot_soa(
            grids_typed, data, queries, extraps_val, side_vals, searches)::Vector{Tv}
    end
end

"""
    constant_interp(grids, data, queries::AbstractVector{<:NTuple}; kwargs...)

One-shot N-dimensional constant interpolation (batch AoS query).
Only allocates the output vector.
"""
function constant_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    side::Union{Symbol, NTuple{N, Symbol}} = :nearest,
    extrap::Union{Symbol, NTuple{N, Symbol}} = :none,
    search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = Binary(),
    deriv::Union{Int, Val, NTuple{N,Int}} = 0
) where {Tv, N}
    if _is_any_deriv(deriv)
        n_queries = length(queries)
        return zeros(Tv, n_queries)
    end

    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed_constant(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    extraps = _resolve_extrap_nd(extrap, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    searches = _resolve_search_nd(search, Val(N))
    side_vals = _to_side_vals(sides)

    @_dispatch_extrap_nd extraps nothing => extraps_val begin
        return _constant_interp_nd_oneshot_aos(
            grids_typed, data, queries, extraps_val, side_vals, searches)::Vector{Tv}
    end
end
