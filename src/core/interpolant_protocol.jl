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
    ) where {Tg <: AbstractFloat, Tv}
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    @boundscheck _check_domain(grid, xq, extrap)
    searcher = _resolve_search(grid, xq, search, hint)
    return _itp_eval_scalar(itp, xq, extrap, deriv, searcher)
end

# ========================================
# 1D Vector Call — Allocating
# ========================================
# Output type promoted to wider type for precision preservation.

function (itp::AbstractInterpolant1D{Tg, Tv})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    T_out = promote_type(Tv, Tq)
    output = Vector{T_out}(undef, length(xq))
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
    ) where {Tg <: AbstractFloat, Tv, Tq <: Real}
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
        ops::NTuple{N, AbstractEvalOp},
        search::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints = nothing
    ) where {Tg, Tv, N}
    zref = _zero_ref(itp)
    @inbounds for k in 1:_query_length(queries)
        query_k = _extract_query_point(queries, k, Val(N))
        oob_val = _try_fill_oob(query_k, itp.grids, itp.extraps, ops, zref)
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        cell = _locate_cell(itp, query_k, search, hints)
        output[k] = _eval_at_cell(itp, cell, ops)
    end
    return output
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
# Per-type callables (Cubic, Linear, etc.) accept Tuple{Vararg{ScalarCoord, N}} directly,
# resolving GridIdx via _resolve_grididx internally.
@inline function (itp::AbstractInterpolantND{Tg, Tv, N})(
        q::Vararg{Union{Real, GridIdx}, N};
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
    _validate_nd_domain(itp.grids, queries, itp.extraps)
    search_tuple = _resolve_search_nd(search, Val(N), queries, hint)
    if _deriv_zero_fill(itp, ops, Val(N))
        fill!(output, zero(eltype(output)))
        return output
    end
    _interp_nd_batch!(output, itp, queries, ops, search_tuple, hint)
    return output
end

# ========================================
# ND Allocating Batch — Unified
# ========================================
# Allocates output via protocol, delegates to in-place above.

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    Tq = _query_eltype(queries)
    output = Vector{promote_type(Tv, Tg, Tq)}(undef, _query_length(queries))
    return itp(output, queries; deriv = deriv, search = search, hint = hint)
end
