# ========================================
# Shared Interpolant Callable Interface (1D + ND)
# ========================================
#
# All interpolant callables defined once on AbstractInterpolant1D / AbstractInterpolantND.
# Per-type files only need trait implementations.
#
# ── 1D Subtypes must implement:
#   _itp_eval_scalar(itp, xq, extrap, op, searcher)   — core scalar evaluation
#   _itp_vector_loop!(out, itp, xq, extrap, op, searcher) — core vector loop
#
# ── 1D Subtypes may override (defaults work for Linear/Quadratic/Constant):
#   _itp_grid(itp)    — grid vector (default: itp.x, Cubic overrides to itp.cache.x)
#   _itp_extrap(itp)  — extrap mode (default: itp.extrap)
#   _itp_search(itp)  — search policy (default: itp.search_policy)

# ╔══════════════════════════════════════╗
# ║       1D Interpolant Protocol        ║
# ╚══════════════════════════════════════╝

# ── Trait Defaults ──

@inline _itp_grid(itp::AbstractInterpolant1D) = itp.x
@inline _itp_extrap(itp::AbstractInterpolant1D) = itp.extrap
@inline _itp_search(itp::AbstractInterpolant1D) = itp.search_policy

# ── Output eltype trait ──
# Default: arithmetic kernels (Linear/Cubic/Quadratic) divide by `h`, so route
# through the shared `_arithmetic_kernel_shape` for Julia inference. Selection
# kernels (Constant) and Hermite-family (with `dy`) override this default.
@inline _output_eltype(::AbstractInterpolant1D{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq} =
    _output_eltype(_arithmetic_kernel_shape, Tg, Tv, Tq)

# ========================================
# 1D Scalar Call — Hot Path
# ========================================
# @inline for broadcast fusion. Accepts any xq (Real, Dual for AD).
# @boundscheck normalized: always checked here, not in per-type eval.

@inline function (itp::AbstractInterpolant1D{Tg, Tv})(
        xq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv}
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    # Domain check is delegated to the per-method `_eval_*_at_point` callable
    # below — NoExtrap throws via that path's `@boundscheck _check_domain`,
    # while Clamp/Fill/Wrap handle OOB inside their specialized eval methods
    # without ever calling `_check_domain`. Outer redundancy removed.
    searcher = _resolve_search(grid, xq, search, hint)
    return _itp_eval_scalar(itp, xq, extrap, deriv, searcher)
end

# ========================================
# 1D Vector Call — Allocating
# ========================================
# Buffer eltype comes from the `_output_eltype(itp, Tq)` trait — generic for
# arithmetic kernels (Int→Float upgrade) and `_select_output_eltype` for
# selection kernels (Constant). Duck `SVector × Dual` resolves via the
# trait's `Base.promote_op` fallback.

function (itp::AbstractInterpolant1D{Tg, Tv})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    output = Vector{_output_eltype(itp, Tq)}(undef, length(xq))
    searcher = _resolve_search(grid, xq, search, hint)
    _itp_vector_loop!(output, itp, xq, extrap, deriv, searcher)
    return output
end

# ========================================
# 1D Vector Call — In-Place
# ========================================

function (itp::AbstractInterpolant1D{Tg, Tv})(
        output::AbstractVector,
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    @assert length(output) == length(xq) "output length must match xq length"
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    searcher = _resolve_search(grid, xq, search, hint)
    _itp_vector_loop!(output, itp, xq, extrap, deriv, searcher)
    return output
end

# ╔══════════════════════════════════════╗
# ║       ND Interpolant Protocol        ║
# ╚══════════════════════════════════════╝

# ── Derivative zero-fill trait ──
# Linear: 2nd+ derivative → all zeros. Constant: any derivative → all zeros.
# Default: no zero-fill (Cubic, Quadratic evaluate all derivative orders).

@inline _deriv_zero_fill(::AbstractInterpolantND, ::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} = false

# ── Output eltype trait (ND) — mirrors the 1D version. ──
@inline _output_eltype(::AbstractInterpolantND{Tg, Tv, N}, ::Type{Tq}) where {Tg, Tv, N, Tq} =
    _output_eltype(_arithmetic_kernel_shape, Tg, Tv, Tq)

# ========================================
# Unified Batch Interpolant Evaluation (Generic ND)
# ========================================
#
# Single batch loop for all AbstractInterpolantND subtypes.
# Query extraction dispatches via _query_extract on query container type.

@inline function _interp_nd_batch!(
        output::AbstractVector,
        itp::AbstractInterpolantND{Tg, Tv, N},
        queries,
        extraps_eff::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    zref = _value_sample(itp)
    @inbounds for k in 1:_query_length(queries)
        query_k = _extract_query_point(queries, k, Val(N))
        # `extraps_eff` carries per-axis `InBounds()` from `_check_domain_nd`
        # when all batch queries on that axis are in-bounds (1D `_check_domain`
        # union-split per axis via the heterogeneous `map`). `_try_fill_oob` and
        # `_locate_cell` compile away the wrap/clamp/fill per-query branches on
        # the InBounds axes — see `_handle_axis_extrap(::InBounds) = q` and
        # `_is_fill_oob`'s @generated `fill_dims` filtering.
        oob_val = _try_fill_oob(query_k, itp.grids, extraps_eff, ops, zref)
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        cell = _locate_cell(itp, query_k, extraps_eff, policies, hints, mono)
        output[k] = _eval_at_cell(itp, cell, ops)
    end
    return output
end

# Scalar-path forwarder: vector_calculus and per-method scalar paths call
# the 5-arg form; we inject `itp.extraps` so the canonical 6-arg
# `_locate_cell` works uniformly. The 6-arg form is the source of truth —
# batch callers (`_interp_nd_batch!`) and the windowed/hetero paths can pass
# a different `extraps_eff` (e.g., InBounds-promoted) without affecting
# scalar callers. Pure `getfield` body — Julia elides this trivially.
@inline _locate_cell(itp::AbstractInterpolantND, q, policies, hints, mono) =
    _locate_cell(itp, q, itp.extraps, policies, hints, mono)

# ========================================
# Unified Scalar Interpolant Evaluation (Generic ND)
# ========================================
#
# Single scalar entry point for AbstractInterpolantND subtypes whose eval
# structure matches `validate → try_fill_oob → deriv_zero_fill → locate →
# eval` (Cubic / Linear / Constant / Quadratic). Each method's callable
# resolves search/hints/ops then delegates here.
#
# Trait dispatch: `_deriv_zero_fill(itp, ops, Val(N))` (Linear: 2nd+ deriv,
# Constant: any deriv, Cubic/Quadratic default false). `_value_sample(itp)`
# supplies the per-method zero element (data first / partials first).
#
# Hetero is *not* routed through here — its callable has GridIdx/NoInterp
# branches and `_eval_hetero_nd` uses a non-`_locate_cell` path (recursive
# `_collapse_dims` / `_eval_hetero_precomputed`).
@inline function _eval_nd_at_point(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    # NoExtrap throw must precede FillExtrap short-circuit (mixed-extrap configs).
    _validate_nd_domain(itp.grids, query, itp.extraps)
    oob_result = _try_fill_oob(query, itp.grids, itp.extraps, ops, _value_sample(itp))
    oob_result !== nothing && return oob_result
    if _deriv_zero_fill(itp, ops, Val(N))
        Tq = promote_type(map(typeof, query)...)
        T_out = _output_eltype(itp, Tq)
        return _deriv_zero_value(T_out, _value_sample(itp), query)
    end
    cell = _locate_cell(itp, query, policies, hints, mono)
    return _eval_at_cell(itp, cell, ops)
end

# ========================================
# ND Scalar: Vector query → tuple conversion
# ========================================
# ForwardDiff/Optim compatibility — AbstractVector{<:Real} queries.

@inline function (itp::AbstractInterpolantND{Tg, Tv, N})(
        query::AbstractVector{<:Real};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return itp(query_tuple; deriv = deriv, search = search, hint = hint)
end

# ========================================
# ND Scalar: Vararg convenience
# ========================================
# Converts vararg calls to tuple form: itp(0.5, GridIdx(3)) → itp((0.5, GridIdx(3)))
# Per-type callables (Cubic, Linear, etc.) accept Tuple{Vararg{Real, N}} directly,
# resolving GridIdx via _resolve_grididx internally. GridIdx <: Real, so Vararg{Real,N} matches.
@inline function (itp::AbstractInterpolantND{Tg, Tv, N})(
        q::Vararg{Real, N};
        kw...,
    ) where {Tg, Tv, N}
    return itp(q; kw...)
end

# ========================================
# ND In-Place Batch — Unified
# ========================================
#
# Single entry point for all batch in-place evaluation.
# Protocol functions (_query_length, _query_extract, _query_eltype) dispatch
# directly on query container type — no normalization needed.

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        output::AbstractVector,
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    nq = _query_length(queries)
    length(output) == nq || _throw_query_output_mismatch(nq, length(output))
    _query_validate(queries)
    # Batch-level InBounds promotion: SoA queries get per-axis InBounds when
    # every query on that axis is in-bounds; AoS/generic stays original.
    # Subsumes the NoExtrap throw via 1D `_check_domain`'s `@boundscheck
    # _is_all_inbounds || _throw_batch_oob`, so the separate `_validate_nd_domain`
    # call is no longer needed.
    extraps_eff = _check_domain_nd(itp.grids, queries, itp.extraps)
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _check_mono_nd(policies, queries)
    if _deriv_zero_fill(itp, ops, Val(N))
        nq = _query_length(queries)
        if nq > 0
            Tq = _query_eltype(queries)
            T_out = _output_eltype(itp, Tq)
            first_q = _extract_query_point(queries, 1, Val(N))
            fill!(output, _deriv_zero_value(T_out, _value_sample(itp), first_q))
        end
        return output
    end
    _interp_nd_batch!(output, itp, queries, extraps_eff, ops, policies, hints, mono)
    return output
end

# ========================================
# ND Allocating Batch — Unified
# ========================================
# Buffer eltype from `_output_eltype(itp, Tq)` (see 1D variant for kernel-
# type rationale); delegates to in-place — no sample-first eval.

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    Tq = _query_eltype(queries)
    output = Vector{_output_eltype(itp, Tq)}(undef, _query_length(queries))
    return itp(output, queries; deriv = deriv, search = search, hint = hint)
end
