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
# Shared ND Callable Interface
# ========================================
#
# All ND callables defined once on AbstractInterpolantND.
# Scalar (tuple/vector), batch (SoA/AoS), and generic fallback paths.
# Per-type eval files only need: scalar tuple callable.

# ── Scalar: Vector query → tuple conversion (ForwardDiff/Optim compat) ──

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

# ── In-place batch: unified ──
#
# Single entry point for all batch in-place evaluation.
# Protocol functions (_query_length, _query_extract, _query_eltype) dispatch
# directly on query container type — no normalization needed.
# Scalar queries (Tuple{Vararg{Real,N}}, AbstractVector{<:Real}) dispatch to
# more specific methods above, so this never intercepts scalar calls.

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

# ── Allocating batch: unified ──
#
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
