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
    # 0. Validate + promote (in-domain NoExtrap axis → InBounds for the lean search) + FillExtrap
    #    short-circuit. InBounds no-ops through `_try_fill_oob` / periodic extension /
    #    `_resolve_extrap` / `_handle_all_extraps` and reaches the extrap-aware search below.
    extraps_val = _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Extend exclusive periodic axes (pool-based). `bcs_p` is the
    # post-extension bc tuple, threaded into the build below.
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs_periodic)

    # 1a. Per-axis extrap passthrough against the post-extension grid.
    # Post-extension: each axis's `(first, last)` IS the wrap domain — the
    # 2-arg primitive is identity for tag-struct extraps.
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)

    # 2. Pool-allocate compact partials (widened with Tg for Dual grid support)
    Tv = _value_type(eltype(data), Tg)
    Tz = _promote_eltype(_coeff_op, Tg, Tv)
    sizes = map(_deriv_size, methods)
    n_partials = prod(sizes)
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))

    # 3. Compute heterogeneous partials in-place (reuses build.jl core)
    _compute_nd_partials_hetero!(partials, grids_p, data_p, methods, bcs_p, sizes)

    # 4. Eval pipeline (axis-only — `grids_p` carries `h`/`inv_h` directly)
    q_eval = _handle_all_extraps(query, grids_p, extraps_eff)
    # 5-arg search: per-axis `extraps_eff` → InBounds range axes take the lean direct search.
    indices, Ls, _ = _search_all_intervals(q_eval, grids_p, searches, hints, extraps_eff)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, grids_p, indices, Ls)

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
        ops::NTuple{N, AbstractEvalOp},
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tg, N}
    # Resolve here so the fresh Ref tuple stays local to this frame (stack-elidable).
    policies, hints = _resolve_oneshot_search_nd(search, queries, hint, Val(N))
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)

    # Build phase (ONE-TIME)
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs_periodic)

    # Per-axis materialization against the (possibly extended) grid.
    # Post-extension: grid-span IS the wrap domain → 2-arg primitive per-axis.
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)
    # Batch-level InBounds promotion: see cubic_nd_oneshot.jl for pattern.
    extraps_eff = _validate_nd_domain(grids_p, queries, extraps_eff)

    Tv = _value_type(eltype(data), Tg)
    Tz = _promote_eltype(_coeff_op, Tg, Tv)
    sizes = map(_deriv_size, methods)
    n_partials = prod(sizes)
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))
    _compute_nd_partials_hetero!(partials, grids_p, data_p, methods, bcs_p, sizes)

    # Eval loop (per query) — axis-only helpers read `h`/`inv_h` from `grids_p`
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_p, extraps_eff, ops, first(data_p))
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        q_eval = _handle_all_extraps(query_k, grids_p, extraps_eff)
        # 5-arg search: `extraps_eff` is InBounds-promoted (via `_validate_nd_domain` above) for
        # an in-domain batch → InBounds range axes take the lean direct search.
        indices, Ls, _ = _search_all_intervals(q_eval, grids_p, policies, hints, extraps_eff)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, grids_p, indices, Ls)
        output[k] = _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, methods)
    end
    return output
end

# Function barrier — specializes on concrete `search` type.
function _interp_nd_hetero_batch_dispatch!(output, grids, data, queries, methods, extraps, ops, search, hint)
    return _interp_nd_hetero_oneshot_batch!(output, grids, data, queries, methods, extraps, ops, search, hint)
end

# ========================================
# OnTheFly One-Shot (Scalar)
# ========================================
# Scalar evaluation via _collapse_dims (sequential 1D one-shot per fiber).
# No partials computation — 2^N× less work than PreCompute for a single query.
# Periodic BC is handled internally by each _oneshot_eval_1d call.

@inline @with_pool pool function _interp_nd_oneshot_onthefly(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing,
    ) where {Tv, N}
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result
    # OnTheFly does not extend data — per-axis surface-level axis-as-truth
    # wrap so `:exclusive` axes carry the validated period through `last(g)` /
    # `_wrap_to_domain` / `search_interval` without the raw n-length Vector's
    # `last - first ≠ period` mismatch (Linear/Constant ND mirror this pattern).
    bcs = map(_bc_for_periodic_check, methods)
    # Value-matched grid float (Int grid + Float32 data → Float32) — output matches the caller's witness.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    # @generated static-Tg unroll: a Type captured in a closure (or an
    # `ntuple(_ -> Tg, …)` element) de-optimizes under weak const-prop —
    # LTS per-fiber heap, and per-axis dynamic dispatch on 1.12 CI workers.
    grids_eff = _resolve_axes(grids, bcs, Tg)
    # NOTE: inclusive PeriodicBC slice validation is NOT performed here — it is
    # hoisted to the callers (`_interp_nd_oneshot_dispatch` and the OnTheFly
    # branch of `_interp_nd_oneshot_batch_dispatch!`) so the batch path pays the
    # O(boundary-size) check once per batch instead of once per query.
    extraps_eff = map(_resolve_extrap, extraps_val, bcs, grids_eff)
    q_eval = _handle_all_extraps(query, grids_eff, extraps_eff)
    # Tr promotes data with grid + query eltypes → Dual-safe pool buffers for AD.
    # `Tg` (value-matched above) already floats Int; reuse it here.
    Tr = _promote_eltype(Tv, Tg, typeof.(q_eval)...)

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
        # BC-aware per-axis search; on `PeriodicBC{:exclusive}` axes the seam
        # cell returns `idx_R=1` so the windowing below picks the right cell.
        stencils, _, _ = _search_all_intervals_stencil(q_eval, grids_eff, searches, hints)
        indices = map(first, stencils)
        # Per-axis windows — generic `AbstractVector{Int}`:
        #   - non-periodic windowable: `UnitRange{Int}` (cell-local, asymmetric clamp)
        #   - periodic windowable:     `Vector{Int}` from pool (wrap-aware indices)
        # Per-axis grid local — `AbstractVector{Tg}`:
        #   - non-periodic: `view(grid, window)`
        #   - periodic:     `Vector{Tg}` from pool (monotonic shifted x)
        # Each axis's return type is determined at compile time by the method
        # type → tuple is concrete, no Union boxing.
        windows = map((m, x, ix) -> _axis_window_pooled(pool, m, x, ix), methods, grids_eff, indices)
        grids_local = map((m, x, w, ix) -> _axis_grid_pooled(pool, m, x, w, ix), methods, grids_eff, windows, indices)
        # Wrap is baked into the windowed grid → strip BC and any WrapExtrap so
        # the inner 1D oneshot evaluates the local mini-grid as non-periodic.
        methods_inner = map(_strip_periodic_bc, methods)
        extraps_inner = map(_strip_wrap_extrap, extraps_eff, methods)
        # Pass full `data` + `windows` (not pre-sliced) to `_collapse_dims`.
        # Inside, fibers are built via single-level `view(data, windows[1], scalars...)`
        # — no nested-SubArray alloc even when `windows[1]` is `Vector{Int}`.
        return _collapse_dims(
            Tr, data, grids_local, methods_inner, extraps_inner,
            q_eval, ops, searches, nothing, windows,
        )
    end

    # Global-solve path: grids straight through. The caller
    # (`_interp_nd_oneshot_dispatch`) promotes types only — the axes arrive raw,
    # and each inner 1D one-shot value-matches its own axis via the data-aware
    # cache / width-first geometry (and wraps `:exclusive` axes itself — user
    # length n, not the wrapped virtual n+1). `_collapse_dims` emits `Tr` directly.
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
        methods::Tuple{LinearInterp, Vararg{LinearInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    bcs = map(m -> m.bc, methods)
    return linear_interp(grids, data, query; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
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
    bcs = map(m -> m.bc, methods)
    return constant_interp(grids, data, query; side = sides, bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
end

# Heterogeneous fallback → resolve kwargs → OnTheFly or PreCompute pool core
function _interp_nd_oneshot_dispatch(
        grids, data, query,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        deriv, extrap, search, hints, coeffs,
    ) where {N}
    # Type-only promotion: no eager grid convert. The OnTheFly path value-matches
    # each raw axis at its own inner 1D surface (data-aware caches / width-first
    # geometry), so it takes the raw `grids` straight through. The PreCompute path
    # (`_interp_nd_hetero_oneshot`) requires homogeneous-eltype `Tg` grids and uses
    # them directly, so it still gets the converted grids.
    Tg, Tv, _ = _nd_promote_types(grids, data)
    _validate_nd_grids(grids, data)
    Tr = _promote_eltype(eltype(data), Tg, typeof.(query)...)

    # bc-aware extrap: NoExtrap → WrapExtrap on PeriodicBC axes.
    bcs = map(_bc_for_periodic_check, methods)
    # Inclusive PeriodicBC requires `data[1, ...] ≈ data[end, ...]` per axis.
    # Validated once here (not inside `_interp_nd_oneshot_onthefly`) so that
    # the equivalent batch dispatch can hoist the same check above its loop —
    # otherwise the boundary-slice scan runs once per query.
    _validate_periodic_slices_nd(data, bcs, Val(N))
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    searches = _resolve_search_nd(search, Val(N), query)
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_axis_methods(grids, methods, extraps_val)

    if coeffs isa OnTheFly
        return _interp_nd_oneshot_onthefly(grids, data, query, methods, extraps_val, searches, ops, hints)::Tr
    end
    return _interp_nd_hetero_oneshot(_convert_grids_typed(grids, Tg), data, query, methods, extraps_val, searches, ops, hints)::Tr
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
        methods::Tuple{LinearInterp, Vararg{LinearInterp}},
        deriv, extrap, search, hints, coeffs,
    )
    bcs = map(m -> m.bc, methods)
    return linear_interp!(output, grids, data, queries; bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
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
    bcs = map(m -> m.bc, methods)
    return constant_interp!(output, grids, data, queries; side = sides, bc = bcs, extrap = extrap, search = search, deriv = deriv, hint = hints)
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
    # Type-only promotion (see the scalar dispatch): OnTheFly takes raw `grids`
    # and value-matches per axis; PreCompute gets the converted grids.
    Tg, Tv, _ = _nd_promote_types(grids, data)
    _validate_nd_grids(grids, data)
    _query_check_ndims(queries, Val(N))

    # bc-aware extrap (matches scalar dispatch).
    bcs = map(_bc_for_periodic_check, methods)
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    _validate_axis_methods(grids, methods, extraps_val)

    if coeffs isa OnTheFly
        nq = _query_length(queries)
        length(output) == nq || _throw_query_output_mismatch(nq, length(output))
        _query_validate(queries)
        # Validate inclusive periodic boundary slices ONCE per batch.
        _validate_periodic_slices_nd(data, bcs, Val(N))
        # OTF batch uses the master shape: pass `search` (unresolved AutoSearch
        # tuple) + `hints` (as-is) per query. Eager 4-arg resolution + pre-built
        # hints would propagate Union policies across `_interp_nd_oneshot_onthefly`'s
        # function boundary — it's too large to inline, so the Union element
        # boxes into 4 allocs/query (≈ 100 B/query). The architectural fix is a
        # dedicated OTF batch function that hosts the per-query work locally
        # (follow-up).
        searches = _resolve_search_nd(search, Val(N))
        @inbounds for k in 1:nq
            query_k = _extract_query_point(queries, k, Val(N))
            output[k] = _interp_nd_oneshot_onthefly(grids, data, query_k, methods, extraps_val, searches, ops, hints)
        end
        return output
    end

    return _interp_nd_hetero_batch_dispatch!(output, _convert_grids_typed(grids, Tg), data, queries, methods, extraps_val, ops, search, hints)
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
        output::AbstractArray,
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
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method
    # Separable fast path (before any flattening, so the N-D output reaches the
    # gridded kernel without a reshape): true iff a gridded evaluator exists for
    # this (query, method) — a GriddedQuery on an all-linear method today.
    _try_gridded_separable!(output, grids, data, queries, method_tuple, extrap, deriv) && return output
    # The batch cores fill by LINEAR index, so a shaped output (e.g. an N-D array
    # for a GriddedQuery) is written through a flat 1-D view. `vec`/`reshape`
    # aliases (never copies), so the caller's array is filled in place; we hand
    # back the caller's original `output`, not the flat view.
    flat = output isa AbstractVector ? output : vec(output)
    # Mixed queries with GridIdx → delegate to GridIdx batch path. Forward
    # `coeffs` so the reduced (post-slice) sub-problem honors and validates the
    # caller's strategy choice rather than silently falling back to AutoCoeffs.
    if queries isa Tuple && _has_grididx(typeof(queries))
        _interp_batch_with_grididx!(
            flat, grids, data, queries;
            method = method, deriv = deriv, extrap = extrap,
            search = search, hint = hint, coeffs = coeffs,
        )
        return output
    end
    coeffs_resolved = _resolve_coeffs_nd_oneshot(coeffs, queries, method_tuple)
    # Reject explicit unsupported combinations (PreCompute + local Hermite); the
    # AutoCoeffs path never trips this because resolution returns OnTheFly for
    # local methods. Mirrors the scalar `interp` validation at line 309.
    _validate_nd_coeffs(coeffs_resolved, method_tuple)
    _interp_nd_oneshot_batch_dispatch!(flat, grids, data, queries, method_tuple, deriv, extrap, search, hint, coeffs_resolved)
    return output
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
    Tr = _promote_eltype(Tv, Tg, Tq)
    # Output takes the query's shape: a flat vector for ordinary batches, the
    # N-D `size(gq)` array for a shaped container like GriddedQuery.
    output = Array{Tr}(undef, _query_size(queries))
    interp!(output, grids, data, queries; method = method, coeffs = coeffs, deriv = deriv, extrap = extrap, search = search, hint = hint)
    return output
end
