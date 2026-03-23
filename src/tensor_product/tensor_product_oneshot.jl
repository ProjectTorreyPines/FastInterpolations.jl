# ========================================
# TensorProductInterpolantND — One-Shot API
# ========================================
# Zero-allocation one-shot evaluation: interp(grids, data, query; method=...)
# Bypasses interpolant construction — builds partials in pool, evaluates, releases.
#
# Homogeneous methods → dispatch to existing one-shot APIs (cubic_interp, etc.)
# Heterogeneous methods → pool-based compact partials + hetero kernel
#
# Pattern follows cubic_nd_oneshot.jl: @with_pool + acquire! for zero-alloc.

# ========================================
# Pool-Based Heterogeneous Core (Scalar)
# ========================================

@with_pool pool function _interp_nd_hetero_oneshot(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing,
    ) where {Tg <: AbstractFloat, N}
    # 0. Domain check + FillExtrap short-circuit
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Extend exclusive periodic axes (pool-based)
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_p, data_p, _ = _prepare_periodic_nd_pooled(pool, grids, data, bcs_periodic)

    # 2. Pool-allocate compact partials (promoted Tv, matching constructor path)
    Tv = _value_type(eltype(data), Tg)
    sizes = map(_deriv_size, methods)
    n_partials = prod(sizes)
    partials = acquire!(pool, Tv, (n_partials, size(data_p)...))

    # 3. Compute heterogeneous partials in-place (reuses build.jl core)
    _compute_nd_partials_hetero!(partials, grids_p, data_p, methods, sizes)

    # 4. Pool-based spacings
    spacings = _create_spacings_pooled(pool, grids_p)

    # 5. Eval pipeline (all standalone functions from nd_utils.jl)
    q_eval = _handle_all_extraps(query, grids_p, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids_p, spacings, searches, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)

    # 6. Heterogeneous tensor-product kernel
    return _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, methods)
end

# ========================================
# Pool-Based Heterogeneous Core (Batch In-Place)
# ========================================

@with_pool pool function _interp_nd_hetero_oneshot_batch!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{<:Any, N},
        queries,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing,
    ) where {Tg <: AbstractFloat, N}
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps_val)

    # Build phase (ONE-TIME)
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_p, data_p, _ = _prepare_periodic_nd_pooled(pool, grids, data, bcs_periodic)

    Tv = _value_type(eltype(data), Tg)
    sizes = map(_deriv_size, methods)
    n_partials = prod(sizes)
    partials = acquire!(pool, Tv, (n_partials, size(data_p)...))
    _compute_nd_partials_hetero!(partials, grids_p, data_p, methods, sizes)
    spacings = _create_spacings_pooled(pool, grids_p)

    # Eval loop (per query)
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_p, extraps_val, ops, first(data_p))
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        q_eval = _handle_all_extraps(query_k, grids_p, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids_p, spacings, searches, hints)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, methods)
    end
    return output
end

# ========================================
# Function Barrier
# ========================================
# Forces search type specialization before entering @with_pool boundary.
# NOT @inline — specialization requires real call.

function _interp_nd_hetero_batch_dispatch!(output, grids, data, queries, methods, extraps, searches, ops, hints)
    return _interp_nd_hetero_oneshot_batch!(output, grids, data, queries, methods, extraps, searches, ops, hints)
end

# ========================================
# Homogeneous One-Shot Dispatch (Scalar)
# ========================================
# Pass user's original kwargs — existing one-shot APIs handle resolution internally.

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{<:CubicInterp, Vararg{<:CubicInterp}},
        deriv, extrap, search, hints,
    )
    bcs = map(m -> m.bc, methods)
    return cubic_interp(grids, data, query; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        ::Tuple{LinearInterp, Vararg{LinearInterp}},
        deriv, extrap, search, hints,
    )
    return linear_interp(grids, data, query; extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{<:QuadraticInterp, Vararg{<:QuadraticInterp}},
        deriv, extrap, search, hints,
    )
    bcs = map(m -> m.bc, methods)
    return quadratic_interp(grids, data, query; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{<:ConstantInterp, Vararg{<:ConstantInterp}},
        deriv, extrap, search, hints,
    )
    sides = map(m -> m.side, methods)
    return constant_interp(grids, data, query; side = sides, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

# Heterogeneous fallback → resolve kwargs → pool core
function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        deriv, extrap, search, hints,
    ) where {N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)
    Tv = _value_type(eltype(data), Tg)
    Tr = _output_eltype(eltype(data), Tg, typeof.(query)...)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    searches = _resolve_search_nd(search, Val(N), query)
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_axis_methods(grids_typed, methods, extraps_val)

    return _interp_nd_hetero_oneshot(grids_typed, data, query, methods, extraps_val, searches, ops, hints)::Tr
end

# ========================================
# Homogeneous Batch Dispatch (In-Place)
# ========================================

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{<:CubicInterp, Vararg{<:CubicInterp}},
        deriv, extrap, search, hints,
    )
    bcs = map(m -> m.bc, methods)
    return cubic_interp!(output, grids, data, queries; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        ::Tuple{LinearInterp, Vararg{LinearInterp}},
        deriv, extrap, search, hints,
    )
    return linear_interp!(output, grids, data, queries; extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{<:QuadraticInterp, Vararg{<:QuadraticInterp}},
        deriv, extrap, search, hints,
    )
    bcs = map(m -> m.bc, methods)
    return quadratic_interp!(output, grids, data, queries; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{<:ConstantInterp, Vararg{<:ConstantInterp}},
        deriv, extrap, search, hints,
    )
    sides = map(m -> m.side, methods)
    return constant_interp!(output, grids, data, queries; side = sides, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

# Heterogeneous fallback → resolve kwargs → function barrier → pool batch
function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        deriv, extrap, search, hints,
    ) where {N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)
    _query_check_ndims(queries, Val(N))
    Tv = _value_type(eltype(data), Tg)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    searches = _resolve_search_nd_uniform(search, Val(N), queries, hints)
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_axis_methods(grids_typed, methods, extraps_val)

    return _interp_nd_hetero_batch_dispatch!(output, grids_typed, data, queries, methods, extraps_val, searches, ops, hints)
end

# ========================================
# Public API — Scalar One-Shot
# ========================================

"""
    interp(grids, data, query; method, deriv=EvalValue(), extrap=NoExtrap(), search=AutoSearch(), hint=nothing)

One-shot N-dimensional interpolation at a single point.
Zero-allocation after warmup. No interpolant object is created.

Homogeneous methods auto-dispatch to optimized one-shot APIs (`cubic_interp`, etc.).
Heterogeneous methods use pool-based compact partial derivative computation.

# Examples
```julia
x, y = range(0, 1, 50), range(0, 1, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# One-shot (no interpolant created)
val = interp((x, y), data, (0.5, 0.3); method=(CubicInterp(), LinearInterp()))

# With derivative
dfdx = interp((x, y), data, (0.5, 0.3);
    method=(CubicInterp(), LinearInterp()), deriv=(DerivOp(1), DerivOp(0)))
```
"""
function interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method
    return _interp_nd_oneshot_dispatch(grids, data, query, method_tuple, deriv, extrap, search, hint)
end

# ========================================
# Public API — Batch In-Place
# ========================================

"""
    interp!(output, grids, data, queries; method, kwargs...)

In-place one-shot N-dimensional interpolation at multiple points.
Builds partials once, evaluates at all query points.
"""
function interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method
    return _interp_nd_oneshot_batch_dispatch!(output, grids, data, queries, method_tuple, deriv, extrap, search, hint)
end

# ========================================
# Public API — Batch Allocating
# ========================================

"""
    interp(grids, data, queries; method, kwargs...)

Allocating one-shot N-dimensional interpolation at multiple points.
Returns a `Vector` of interpolated values.
"""
function interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    Tq = _query_eltype(queries)
    Tr = _output_eltype(Tv, Tg, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    interp!(output, grids, data, queries; method = method, deriv = deriv, extrap = extrap, search = search, hint = hint)
    return output
end
