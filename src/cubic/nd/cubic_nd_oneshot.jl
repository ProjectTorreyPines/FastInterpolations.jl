# ========================================
# ND Cubic Interpolation — One-Shot Evaluation
# ========================================
#
# Zero-allocation one-shot API and pool-based backends for ND cubic interpolation.
# Interpolant construction is in cubic_nd_interpolant.jl.

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

# One-shot fill/extrap payload space: data is not eagerly converted here, so Real
# axes widen by the coeff witness; a non-Real tag cannot run the witness — the
# value space itself serves (the persistent entry's convention).
@inline _oneshot_fill_eltype(::Type{Tg}, ::Type{Tv}) where {Tg <: Real, Tv} =
    _promote_eltype(_coeff_op2, Tg, Tv)
@inline _oneshot_fill_eltype(::Type{Tg}, ::Type{Tv}) where {Tg, Tv} = _value_type(Tv, Tg)

"""
    cubic_interp(grids, data, query; deriv=EvalValue(), kwargs...)

One-shot ND cubic interpolation at a single point.
Zero-allocation after warmup: uses pool-based partials instead of constructing an Interpolant.

Non-Real (unit-carrying) axes mirror the persistent scaled-store build: the solve runs
on dimensionless axis twins and the result is restored to grid⁻ᵏ units. The zero-alloc
contract is the Real-axis path's; the unit arm allocates its per-call twins.

# Keywords
- `deriv`: `DerivOp` or `NTuple{N,DerivOp}` for mixed partials
- `bc`, `extrap`, `search`, `coeffs`: Same as the Interpolant constructor form

!!! note "Periodic BC validation"
    Periodic data integrity (`data[..., 1, ...] ≈ data[..., end, ...]`) **is** validated
    for `PeriodicBC` dimensions, just as in the `CubicInterpolant` constructor.
    The check is zero-allocation: it uses a `@generated` nested loop with direct indexing
    instead of `selectdim` (which would heap-allocate a `SubArray`).
"""
function cubic_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Number, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    # Scalar one-shot: raw grids — a stable grid id lets `_get_cubic_cache` memoise
    # (a per-call copy would miss every time + alloc). `Tg` is value-matched (Int/OneTo grid +
    # Float32 data → Float32), so the OnTheFly eval + witness `Tr` agree. Batch keeps eager-convert.
    Tg_raw = _promote_grid_eltype(grids)
    Tg = _promote_grid_float(Tg_raw, Tv)
    Tv_p = _oneshot_fill_eltype(Tg, Tv)
    _validate_nd_grids(grids, data)
    # Reparameterizable axes only (Real or unit-carrying) — the persistent builder's gate.
    _check_nd_reparam_grid(grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    # `ops` folds into the witness BEFORE the assertion (linear canonical): a derivative
    # query lands in value/coordᴺ — identity on Real grids and on `DerivOp{0}`.
    Tr = _deriv_eltype_nd(
        _nd_value_eltype(_interp_op, Tv, grids, promote_type(typeof.(query)...)), grids, ops
    )

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    # Validate BC requirements (once, before dispatch).
    _validate_nd_bcs!(grids, bcs, data, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv_p)

    # OnTheFly: skip full partials build — sequential 1D collapse (2^N× less work).
    # Non-Real queries resolve PreCompute by dispatch (Real-tuple arm); an explicit
    # OnTheFly rides the hetero collapse, whose gate keeps the refusal friendly.
    coeffs_resolved = _resolve_coeffs_nd_oneshot(coeffs, query, ntuple(_ -> CubicInterp(), Val(N)))
    if coeffs_resolved isa OnTheFly
        _check_nd_hetero_grid(Tg_raw)
        methods = map(CubicInterp, bcs)
        return _interp_nd_oneshot_onthefly(grids, data, query, methods, extraps_val, searches, ops, hint)::Tr
    end
    return _cubic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops, hint)::Tr
end

"""
    cubic_interp(grids, data, queries; deriv=EvalValue(), kwargs...)

One-shot ND cubic interpolation at multiple points (batch).
Accepts any query format implementing the query protocol
(`_query_length`, `_query_extract`, `_query_eltype`).
Zero-allocation for workspace after warmup; output vector is heap-allocated.
"""
# Public ND allocating batch one-shot (N≥2; N=1 intercepted by the collapse method,
# reaching here only for the no-1D-equivalent OnTheFly branch).
@inline cubic_interp(grids::NTuple{N, AbstractVector}, data::AbstractArray{<:Any, N}, queries; kwargs...) where {N} =
    _cubic_interp_nd_oneshot_alloc(grids, data, queries; kwargs...)

function _cubic_interp_nd_oneshot_alloc(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _check_nd_reparam_grid(grids)
    Tq = _query_eltype(queries)
    # Same fold as the scalar entry (linear canonical): the buffer must be sized in
    # ∂-units, else a unit-grid derivative batch throws on the first store.
    ops = _resolve_deriv_nd(deriv, Val(N))
    Tr = _deriv_eltype_nd(_nd_value_eltype(_interp_op, Tv, grids, Tq), grids, ops)
    output = _alloc_query_output(Tr, queries)
    _cubic_interp_nd_oneshot_batch!(output, grids, data, queries; deriv, bc, extrap, search, coeffs, hint)
    return output
end

# ========================================
# POOL-BASED ND ONE-SHOT IMPLEMENTATION
# ========================================
#
# Zero-allocation ND one-shot evaluation using pool-based partials.
# Bypasses Interpolant construction entirely — computes partials in a pool buffer,
# evaluates at the query point(s), then releases all buffers on scope exit.
#
# All eval pipeline functions are standalone:
#   _handle_all_extraps, _search_all_intervals, _compute_all_local_params, _eval_nd_cell

"""
    _cubic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops)

Pool-based scalar one-shot ND cubic Hermite evaluation.
Computes 2^N partial derivatives in a pool buffer and evaluates at a single point.
Zero-allocation after warmup (pool reuse).

`extraps_val` must be a pre-resolved tuple of concrete `AbstractExtrap` instances
(e.g., `(NoExtrap(), ClampExtrap())`), computed via `_resolve_extrap_nd` in the API layer.
"""
@with_pool pool function _cubic_interp_nd_oneshot(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Number, N}},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tv, N}
    # Value-matched pooled wrap, symmetric with the quadratic scalar backend: Ranges
    # → isbits `_CachedRange{Tg}` (the per-axis spline caches still memoise via
    # value-deterministic objectid); a mismatched Vector converts into a POOL buffer
    # (warm one-shots stay zero-alloc), so the whole solve pipeline runs at `Tg`.
    # Non-Real axes wrap at their OWN eltype (abstract-tag arm of the @generated map).
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    grids = _cache_axes_pooled(pool, grids, Tg)  # @generated static-Tg unroll (no Type-captured closure)
    # Bare GridIdx(k).val is NaN → resolve to the grid coordinate for the value kernel (search still uses .idx).
    query = map(_resolve_grididx, query, grids)
    # 0. Validate (NoExtrap throw must precede FillExtrap short-circuit) AND promote per axis:
    #    an in-domain NoExtrap axis becomes InBounds for the search (lean); InBounds is a no-op
    #    for `_try_fill_oob` / periodic extension / `_handle_all_extraps` and reaches the
    #    extrap-aware `_search_all_intervals` below.
    extraps_val = _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Extend exclusive periodic axes (pool-based, zero heap alloc)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)

    # 1a. Per-axis extrap passthrough against the extended grid.
    # Post-extension: each axis's `(first, last)` IS the wrap domain — the
    # 2-arg primitive is identity for tag-struct extraps (Wrap, Clamp, ...).
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)

    # 2. Pool-allocate partials array (THE KEY: pool instead of heap). Tz widens Tv
    # with the solve-grid eltype (Dual grids → Dual derivs). Non-Real axes solve on
    # the dimensionless twins; search/local params below keep reading `grids_p`.
    grids_solve, bcs_solve = _reparam_solve_frame(grids_p, bcs_p, data_p)
    Tz = _promote_eltype(_coeff_op2, _promote_grid_eltype(grids_solve), Tv)
    n_partials = 1 << N
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))

    # 3. Compute all partial derivatives in-place
    #    (internally uses autocached 1D caches + nested @with_pool for temp buffers)
    _compute_nd_partials!(partials, grids_solve, data_p, bcs_solve)

    # 4. Eval pipeline (all standalone functions, no Interpolant needed).
    # Axis-only forms — `grids_p` axes carry `h`/`inv_h` directly via `_get_h`/
    # `_get_inv_h` (cached lookup for wrapped axes, on-the-fly diff for raw Vector);
    # the data-aware form width-types hs/inv_hs at `Tg` (raw Int-Vector axes included).
    q_evals = _handle_all_extraps(query, grids_p, extraps_eff)
    indices, Ls, _ = _search_all_intervals(q_evals, grids_p, searches, hints, extraps_eff)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls, Tg)

    # 6. Tensor-product kernel evaluation ([Y]-scaled partials → grid⁻ᵏ restore at the seam)
    r = _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
    return _restore_nd_deriv_scale(r, grids_p, ops)
end

"""
    _cubic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, searches, ops)

Pool-based in-place batch one-shot ND cubic Hermite evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
Uses query protocol (`_query_length`, `_query_extract`) — works with any query format.

`extraps_val` must be a pre-resolved tuple of concrete `AbstractExtrap` instances.
"""
# `grids` is NOT pinned to one shared axis eltype: mixed-unit grids (`s` × `m`)
# reach this backend as a heterogeneous per-axis-float tuple — requiring a common
# `Tg` excluded them from the batch path entirely (the linear sibling relaxed first).
@with_pool pool function _cubic_interp_nd_oneshot_batch!(
        output::AbstractArray,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}},
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}},
    ) where {Tv, N}
    # Resolve here so the fresh Ref tuple stays local to this frame (stack-elidable).
    policies, hints = _resolve_oneshot_search_nd(search, queries, hint, Val(N))
    nq = _query_length(queries)
    _check_query_output_size(output, queries)
    _query_validate(queries)
    Tg_raw = _promote_grid_eltype(grids)

    # Build phase (same as scalar, done once)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)
    # Per-axis materialization of extraps against the (possibly extended) grid.
    # Post-extension: grid-span IS the wrap domain → 2-arg primitive per-axis.
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)
    # Validate + batch-level InBounds promotion: throws on OOB NoExtrap and returns `InBounds()`
    # per axis when all its queries are in-bounds, so the per-query `_try_fill_oob` /
    # `_handle_all_extraps` branches compile away.
    extraps_eff = _validate_nd_domain(grids_p, queries, extraps_eff)
    # Non-Real axes solve on the dimensionless twins (persistent scaled-store mirror);
    # search + local params below keep reading the unit axes `grids_p`.
    grids_solve, bcs_solve = _reparam_solve_frame(grids_p, bcs_p, data_p)
    Tz = _promote_eltype(_coeff_op2, _promote_grid_eltype(grids_solve), Tv)
    n_partials = 1 << N
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))
    _compute_nd_partials!(partials, grids_solve, data_p, bcs_solve)

    # Eval loop: search + kernel per query point. Axis-only helpers read
    # `h`/`inv_h` directly from `grids_p` (no transient pool spacings).
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_p, extraps_eff, ops, first(data_p))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_evals = _handle_all_extraps(query_k, grids_p, extraps_eff)
        indices, Ls, _ = _search_all_intervals(q_evals, grids_p, policies, hints, extraps_eff)
        hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls, Tg_raw)
        r = _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
        output[k] = _restore_nd_deriv_scale(r, grids_p, ops)
    end
    return output
end

# Function barrier — specializes on concrete `search` type.
function _cubic_nd_batch_dispatch!(output, grids, data, queries, bcs, extraps, ops, search, hint)
    return _cubic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps, ops, search, hint)
end

# ========================================
# IN-PLACE PUBLIC API (ND batch)
# ========================================

"""
    cubic_interp!(output, grids, data, queries; deriv=EvalValue(), kwargs...)

In-place one-shot ND cubic interpolation at multiple points (batch).
Accepts any query format implementing the query protocol.
Writes results into pre-allocated `output` vector.
"""
# Public ND in-place batch one-shot (N≥2; N=1 is intercepted by the collapse method
# below and only reaches here via the OnTheFly branch, which has no 1D equivalent).
@inline function cubic_interp!(output::AbstractArray, grids::NTuple{N, AbstractVector}, data::AbstractArray{<:Any, N}, queries; kwargs...) where {N}
    _check_nd_reparam_grid(grids)
    return _cubic_interp_nd_oneshot_batch!(output, grids, data, queries; kwargs...)
end

function _cubic_interp_nd_oneshot_batch!(
        output::AbstractArray,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    _query_check_ndims(queries, Val(N))
    grids_typed, _, Tv_p, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    _validate_nd_bcs!(grids_typed, bcs, data, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv_p)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _cubic_nd_batch_dispatch!(output, grids_typed, data, queries, bcs, extraps_val, ops, search, hint)
end
