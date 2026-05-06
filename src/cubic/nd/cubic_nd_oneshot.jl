# ========================================
# ND Cubic Interpolation — One-Shot Evaluation
# ========================================
#
# Zero-allocation one-shot API and pool-based backends for ND cubic interpolation.
# Interpolant construction is in cubic_nd_interpolant.jl.

# ========================================
# ONE-SHOT PUBLIC API
# ========================================

"""
    cubic_interp(grids, data, query; deriv=EvalValue(), kwargs...)

One-shot ND cubic interpolation at a single point.
Zero-allocation after warmup: uses pool-based partials instead of constructing an Interpolant.

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
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tv, N}
    # Type promotion + validation (same as constructor path)
    grids_typed, Tg, Tv_p, Tz = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)
    Tr = _output_eltype(Tv_p, Tg, typeof.(query)...)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N), query)  # NTuple{N,Real} <: Tuple → BinarySearch/axis

    # Validate BC requirements (once, before dispatch).
    _validate_nd_bcs!(grids_typed, bcs, data, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv_p)
    ops = _resolve_deriv_nd(deriv, Val(N))

    # OnTheFly: skip full partials build — use sequential 1D collapse (2^N× less work)
    coeffs_resolved = _resolve_coeffs_nd_oneshot(coeffs, query, ntuple(_ -> CubicInterp(), Val(N)))
    if coeffs_resolved isa OnTheFly
        methods = map(CubicInterp, bcs)
        return _interp_nd_oneshot_onthefly(grids_typed, data, query, methods, extraps_val, searches, ops, hint)::Tr
    end
    return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops, hint)::Tr
end

"""
    cubic_interp(grids, data, queries; deriv=EvalValue(), kwargs...)

One-shot ND cubic interpolation at multiple points (batch).
Accepts any query format implementing the query protocol
(`_query_length`, `_query_extract`, `_query_eltype`).
Zero-allocation for workspace after warmup; output vector is heap-allocated.
"""
function cubic_interp(
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
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _output_eltype(Tv, Tg, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    cubic_interp!(output, grids, data, queries; deriv, bc, extrap, search, coeffs, hint)
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
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints = nothing
    ) where {Tg, Tv, N}
    # 0. NoExtrap domain check must precede FillExtrap short-circuit
    _validate_nd_domain(grids, query, extraps_val)
    oob_result = _try_fill_oob(query, grids, extraps_val, ops, @inbounds first(data))
    oob_result !== nothing && return oob_result

    # 1. Extend exclusive periodic axes (pool-based, zero heap alloc)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)

    # 1a. Per-axis materialization: upgrade WrapExtrap{Nothing} → WrapExtrap{T} against
    # the extended grid and force WrapExtrap on periodic axes so `_handle_all_extraps`
    # hits the typed form (kernels never see WrapExtrap{Nothing}).
    # Post-extension: grid-span IS the wrap domain → 2-arg primitive per-axis.
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)

    # 2. Pool-allocate partials array (THE KEY: pool instead of heap)
    # Tz widens Tv with Tg: when grid is Dual, derivatives = data × inv_h → Dual-typed.
    Tz = _output_eltype(Tv, Tg)
    n_partials = 1 << N
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))

    # 3. Compute all partial derivatives in-place
    #    (internally uses autocached 1D caches + nested @with_pool for temp buffers)
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)

    # 4. Eval pipeline (all standalone functions, no Interpolant needed).
    # Axis-only forms — `grids_p` axes carry `h`/`inv_h` directly via `_get_h`/
    # `_get_inv_h` (cached lookup for wrapped axes, on-the-fly diff for raw Vector).
    q_evals = _handle_all_extraps(query, grids_p, extraps_eff)
    indices, Ls, _ = _search_all_intervals(q_evals, grids_p, searches, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls)

    # 6. Tensor-product kernel evaluation
    return _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _cubic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps_val, searches, ops)

Pool-based in-place batch one-shot ND cubic Hermite evaluation.
Computes partials ONCE, then evaluates at all query points into `output`.
Uses query protocol (`_query_length`, `_query_extract`) — works with any query format.

`extraps_val` must be a pre-resolved tuple of concrete `AbstractExtrap` instances.
"""
@with_pool pool function _cubic_interp_nd_oneshot_batch!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        ops::NTuple{N, AbstractEvalOp},
        hints,  # Nothing or NTuple{N, Ref{Int}}
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps_val)

    # Build phase (same as scalar, done once)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)
    # Per-axis materialization of extraps against the (possibly extended) grid.
    # Post-extension: grid-span IS the wrap domain → 2-arg primitive per-axis.
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)
    Tz = _output_eltype(Tv, Tg)
    n_partials = 1 << N
    partials = acquire!(pool, Tz, (n_partials, size(data_p)...))
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)

    # Eval loop: search + kernel per query point. Axis-only helpers read
    # `h`/`inv_h` directly from `grids_p` (no transient pool spacings).
    @inbounds for k in 1:nq
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, grids_p, extraps_val, ops, first(data_p))
        if oob_val !== nothing
            output[k] = oob_val; continue
        end
        q_evals = _handle_all_extraps(query_k, grids_p, extraps_eff)
        indices, Ls, _ = _search_all_intervals(q_evals, grids_p, policies, hints, mono)
        hs, inv_hs, dLs = _compute_all_local_params(q_evals, grids_p, indices, Ls)
        output[k] = _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

# Function barrier: forces Julia to runtime-dispatch on the concrete
# searches tuple type, resolving per-element Union{BinarySearch,LinearBinarySearch} before
# entering the @with_pool boundary. NOT @inline — specialization requires real call.
function _cubic_nd_batch_dispatch!(output, grids, data, queries, bcs, extraps, policies, ops, hints, mono)
    return _cubic_interp_nd_oneshot_batch!(output, grids, data, queries, bcs, extraps, policies, ops, hints, mono)
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
function cubic_interp!(
        output::AbstractVector,
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
    policies = _resolve_search_nd(search, Val(N))
    hints_nd = hint  # pass user hints as-is (nothing or NTuple{N,Ref{Int}})
    mono = _check_mono_nd(policies, queries)

    _validate_nd_bcs!(grids_typed, bcs, data, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv_p)
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _cubic_nd_batch_dispatch!(output, grids_typed, data, queries, bcs, extraps_val, policies, ops, hints_nd, mono)
end
