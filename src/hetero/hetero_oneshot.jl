# ========================================
# HeteroInterpolantND — One-Shot API
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
    ) where {Tg, N}
    # 0. Domain check + FillExtrap short-circuit
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Extend exclusive periodic axes (pool-based)
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs_periodic)

    # 1a. Per-axis materialization: upgrade WrapExtrap{Nothing} → WrapExtrap{T} against
    # the post-extension grid so the downstream eval pipeline never sees the singleton.
    extraps_eff = map(_materialize_extrap, grids_p, bcs_p, extraps_val)

    # 2. Pool-allocate compact partials (widened with Tg for Dual grid support)
    Tv = _value_type(eltype(data), Tg)
    Tz = _output_eltype(Tv, Tg)
    sizes = map(_deriv_size, methods)
    n_partials = prod(sizes)
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))

    # 3. Compute heterogeneous partials in-place (reuses build.jl core)
    _compute_nd_partials_hetero!(partials, grids_p, data_p, methods, sizes)

    # 4. Pool-based spacings
    spacings = _create_spacings_pooled(pool, grids_p)

    # 5. Eval pipeline (all standalone functions from nd_utils.jl)
    q_eval = _handle_all_extraps(query, grids_p, extraps_eff)
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
        policies::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints,  # Nothing or NTuple{N, Ref{Int}}
        mono::NTuple{N, Bool},
    ) where {Tg, N}
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps_val)

    # Build phase (ONE-TIME)
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs_periodic)

    # Per-axis materialization against the (possibly extended) grid.
    extraps_eff = map(_materialize_extrap, grids_p, bcs_p, extraps_val)

    Tv = _value_type(eltype(data), Tg)
    Tz = _output_eltype(Tv, Tg)
    sizes = map(_deriv_size, methods)
    n_partials = prod(sizes)
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))
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
        q_eval = _handle_all_extraps(query_k, grids_p, extraps_eff)
        indices, Ls, _ = _search_all_intervals(q_eval, grids_p, spacings, policies, hints, mono)
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

function _interp_nd_hetero_batch_dispatch!(output, grids, data, queries, methods, extraps, policies, ops, hints, mono)
    return _interp_nd_hetero_oneshot_batch!(output, grids, data, queries, methods, extraps, policies, ops, hints, mono)
end

# ========================================
# OnTheFly One-Shot (Scalar)
# ========================================
# Scalar evaluation via _collapse_dims (sequential 1D one-shot per fiber).
# No partials computation — 2^N× less work than PreCompute for a single query.
# Periodic BC is handled internally by each _oneshot_eval_1d call.

@inline @with_pool pool function _interp_nd_oneshot_onthefly(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing,
        spacings::Union{Nothing, Tuple{Vararg{AbstractGridSpacing, N}}} = nothing,
    ) where {Tg, Tv, N}
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result
    # OnTheFly does not extend data — materialize extraps against the original grid
    # with per-axis bcs derived from methods. Periodic axes become WrapExtrap with
    # the correct `[x_min, x_min+period)` domain via the bc-aware constructor.
    bcs = map(_bc_for_periodic_check, methods)
    extraps_eff = map(_materialize_extrap, grids, bcs, extraps_val)
    q_eval = _handle_all_extraps(query, grids, extraps_eff)
    # Tr promotes data eltype with grid + query eltypes → Dual-safe pool buffers for AD.
    # Grid eltype included: when grid is Dual, 1D oneshot returns Dual-typed results
    # that must fit into _collapse_dims intermediate buffers.
    Tr = _output_eltype(Tv, Tg, typeof.(q_eval)...)

    # GridIdx safety gate: same reason as the persistent path — a GridIdx on a
    # windowable axis would be aliased to the wrong grid entry once the data
    # view is sliced to a cell-local stencil. Fall through to the full-fiber
    # path instead (correct behavior, tiny perf cost on the rare mixed query).
    #
    # NOTE: this gate is defensive — the public scalar `interp(...)` API and
    # the public batch `interp!` both promote GridIdx → NoInterp before any
    # call reaches this function (see hetero_oneshot.jl:366-377 and
    # hetero_nointerp.jl:_interp_batch_with_grididx!), so under normal use the
    # `_has_grididx` branch is unreachable. The gate exists to prevent silent
    # corruption if a future internal caller forgets to strip GridIdx first.
    if _has_any_local_method(methods) && !_has_grididx(typeof(query))
        # Resolve spacings:
        # - Caller-provided (batch dispatcher precomputes once outside its loop) → reuse.
        # - nothing → this is scalar oneshot. Acquire h/inv_h from the pool so Vector
        #   grids don't heap-allocate per call. Range grids return `ScalarSpacing`
        #   (struct, no pool touch). Pool scope is this function's `@with_pool`, so
        #   the acquired buffers are reclaimed when we return.
        sp = spacings === nothing ? _create_spacings_pooled(pool, grids) : spacings
        indices, _, _ = _search_all_intervals(q_eval, grids, sp, searches, hints)
        # `map` over heterogeneous tuples for closure-free unrolled per-axis dispatch.
        windows = map(_axis_window, methods, indices, map(length, grids))
        data_local = view(data, windows...)
        grids_local = map(view, grids, windows)
        rel_windows = map(Base.OneTo ∘ length, windows)
        return _collapse_dims(
            Tr, data_local, grids_local, methods, extraps_eff,
            q_eval, ops, searches, nothing, rel_windows,
        )
    end

    # Pure global-solve path: no pre-search, full windows, bit-for-bit pre-Phase-3 behavior.
    full_windows = map(Base.OneTo, size(data))
    return _collapse_dims(Tr, data, grids, methods, extraps_eff, q_eval, ops, searches, hints, full_windows)
end

# ========================================
# Homogeneous One-Shot Dispatch (Scalar)
# ========================================
# Pass user's original kwargs — existing one-shot APIs handle resolution internally.
# `coeffs` is forwarded to APIs that support it (Cubic, Quadratic).
# Linear/Constant ignore it (no global solve).

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{CubicInterp, Vararg{CubicInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    bcs = map(m -> m.bc, methods)
    return cubic_interp(grids, data, query; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints, coeffs = coeffs)
end

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        ::Tuple{LinearInterp, Vararg{LinearInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    return linear_interp(grids, data, query; extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{QuadraticInterp, Vararg{QuadraticInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    bcs = map(m -> m.bc, methods)
    return quadratic_interp(grids, data, query; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints, coeffs = coeffs)
end

function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{ConstantInterp, Vararg{ConstantInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    sides = map(m -> m.side, methods)
    return constant_interp(grids, data, query; side = sides, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

# Heterogeneous fallback → resolve kwargs → OnTheFly or PreCompute pool core
function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        deriv, extrap, search, hints, coeffs,
    ) where {N}
    grids_typed, Tg, Tv, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)
    Tr = _output_eltype(eltype(data), Tg, typeof.(query)...)

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    searches = _resolve_search_nd(search, Val(N), query)
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_axis_methods(grids_typed, methods, extraps_val)

    if coeffs isa OnTheFly
        return _interp_nd_oneshot_onthefly(grids_typed, data, query, methods, extraps_val, searches, ops, hints)::Tr
    end
    return _interp_nd_hetero_oneshot(grids_typed, data, query, methods, extraps_val, searches, ops, hints)::Tr
end

# ========================================
# Homogeneous Batch Dispatch (In-Place)
# ========================================
# Batch always uses PreCompute (amortize build over many queries).
# `coeffs` forwarded to APIs that support it; Linear/Constant ignore.

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{CubicInterp, Vararg{CubicInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    bcs = map(m -> m.bc, methods)
    return cubic_interp!(output, grids, data, queries; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints, coeffs = coeffs)
end

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        ::Tuple{LinearInterp, Vararg{LinearInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    return linear_interp!(output, grids, data, queries; extrap = extrap, search = search, deriv = deriv, hint = hints)
end

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{QuadraticInterp, Vararg{QuadraticInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    bcs = map(m -> m.bc, methods)
    return quadratic_interp!(output, grids, data, queries; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints, coeffs = coeffs)
end

function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{ConstantInterp, Vararg{ConstantInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    sides = map(m -> m.side, methods)
    return constant_interp!(output, grids, data, queries; side = sides, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

# Heterogeneous fallback → resolve kwargs → function barrier → pool batch.
# Branches on the resolved `coeffs`: PreCompute uses the amortized partials build,
# OnTheFly loops the existing scalar one-shot per query (used when the method tuple
# contains a local Hermite method with no PreCompute backend).
@with_pool pool function _interp_nd_oneshot_batch_dispatch!(
        output, grids, data, queries,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        deriv, extrap, search, hints, coeffs,
    ) where {N}
    grids_typed, _, Tv, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)
    _query_check_ndims(queries, Val(N))

    extraps_val = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    policies = _resolve_search_nd(search, Val(N))
    mono = _check_mono_nd(policies, queries)
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_axis_methods(grids_typed, methods, extraps_val)

    if coeffs isa OnTheFly
        nq = _query_length(queries)
        length(output) == nq || _throw_query_output_mismatch(nq, length(output))
        _query_validate(queries)
        # Phase 5a: precompute per-axis spacings ONCE if any local-Hermite axis is present.
        # The windowed path inside `_interp_nd_oneshot_onthefly` reuses these across all
        # `nq` queries, so the cell-local stencil benefit is realized in the batch loop.
        # For pure global-solve method tuples, `spacings` is left as `nothing` (unused).
        # Use `_create_spacings_pooled` so Vector grids acquire h/inv_h from THIS
        # function's `@with_pool` scope — the buffers outlive the per-query
        # `_interp_nd_oneshot_onthefly` calls (which have their own inner pool
        # scope) because nested `@with_pool` scopes don't reclaim the outer
        # scope's arrays. Range grids return `ScalarSpacing` with zero pool touch.
        spacings = _has_any_local_method(methods) ? _create_spacings_pooled(pool, grids_typed) : nothing
        @inbounds for k in 1:nq
            query_k = _extract_query_point(queries, k, Val(N))
            output[k] = _interp_nd_oneshot_onthefly(grids_typed, data, query_k, methods, extraps_val, policies, ops, hints, spacings)
        end
        return output
    end

    return _interp_nd_hetero_batch_dispatch!(output, grids_typed, data, queries, methods, extraps_val, policies, ops, hints, mono)
end

# ========================================
# Public API — Scalar One-Shot
# ========================================

"""
    interp(grids, data, query; method, coeffs=AutoCoeffs(), deriv=EvalValue(), extrap=NoExtrap(), search=AutoSearch(), hint=nothing)

One-shot N-dimensional interpolation at a single point.
Zero-allocation after warmup. No interpolant object is created.

With `coeffs=AutoCoeffs()` (default), scalar queries use `OnTheFly()` strategy
(2^N× less work than `PreCompute()` for a single point).

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
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method
    resolved_query = map(_resolve_grididx, query, grids)
    # GridIdx auto-promotion: when all derivs are EvalValue (scalar or tuple),
    # GridIdx axes need no interpolation — replace their method with NoInterp()
    # for pre-slice dimension reduction (e.g., 3D cubic build → 1D: ~5000x speedup).
    # MUST run before `_validate_nd_coeffs` so that `coeffs=PreCompute()` is
    # validated against the post-promotion methods: a Pchip axis that is
    # GridIdx-sliced becomes NoInterp and no longer trips the local-Hermite check.
    if _all_eval_value(deriv)
        method_tuple = _promote_grididx_to_nointerp(method_tuple, resolved_query)
    end
    coeffs_resolved = _resolve_coeffs_nd_oneshot(coeffs, resolved_query, method_tuple)
    # Reject unsupported strategy/method combinations (e.g., PreCompute + local Hermite ND)
    # Mirrors the interpolant-construction validation in hetero_interpolant.jl.
    _validate_nd_coeffs(coeffs_resolved, method_tuple)
    # NoInterp routing: method-based (not query-type-based)
    if _has_nointerp_method(typeof(method_tuple))
        _validate_nointerp_grididx(method_tuple, resolved_query)
        return _interp_nointerp_oneshot(grids, data, resolved_query, method_tuple, deriv, extrap, search, hint)
    end
    return _interp_nd_oneshot_dispatch(grids, data, resolved_query, method_tuple, deriv, extrap, search, hint, coeffs_resolved)
end

# ========================================
# Public API — Batch In-Place
# ========================================

"""
    interp!(output, grids, data, queries; method, coeffs=AutoCoeffs(), kwargs...)

In-place one-shot N-dimensional interpolation at multiple points.
Builds partials once, evaluates at all query points.
"""
function interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint = nothing,
    ) where {N}
    # Mixed queries with GridIdx → delegate to GridIdx batch path. Forward
    # `coeffs` so the reduced (post-slice) sub-problem honors and validates the
    # caller's strategy choice rather than silently falling back to AutoCoeffs.
    if queries isa Tuple && _has_grididx(typeof(queries))
        return _interp_batch_with_grididx!(
            output, grids, data, queries;
            method = method, deriv = deriv, extrap = extrap,
            search = search, hint = hint, coeffs = coeffs,
        )
    end
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method
    coeffs_resolved = _resolve_coeffs_nd_oneshot(coeffs, queries, method_tuple)
    # Reject explicit unsupported combinations (PreCompute + local Hermite); the
    # AutoCoeffs path never trips this because resolution returns OnTheFly for
    # local methods. Mirrors the scalar `interp` validation at line 309.
    _validate_nd_coeffs(coeffs_resolved, method_tuple)
    return _interp_nd_oneshot_batch_dispatch!(output, grids, data, queries, method_tuple, deriv, extrap, search, hint, coeffs_resolved)
end

# ========================================
# Public API — Batch Allocating
# ========================================

"""
    interp(grids, data, queries; method, coeffs=AutoCoeffs(), kwargs...)

Allocating one-shot N-dimensional interpolation at multiple points.
Returns a `Vector` of interpolated values.
"""
function interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {Tv, N}
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _output_eltype(Tv, Tg, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    interp!(output, grids, data, queries; method = method, coeffs = coeffs, deriv = deriv, extrap = extrap, search = search, hint = hint)
    return output
end
