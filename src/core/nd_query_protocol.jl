# ========================================
# ND Query Protocol
# ========================================
#
# Extensible 3-function protocol for ND query containers.
#
# Generic defaults delegate to Base (length, getindex, eltype), so any type
# with correct Base semantics works out of the box — no protocol to implement.
# SoA (Tuple of Vectors) is the only built-in override (Base semantics differ).
#
# Override _query_* only when Base interfaces have wrong semantics for your type:
#   import FastInterpolations: _query_length, _query_extract, _query_eltype
#   _query_length(q::MyQueries) = ...
#   _query_extract(q::MyQueries, k) = ...       # return k-th point (any indexable)
#   _query_eltype(q::MyQueries) = ...

# ── Protocol function 1: query count ──

@inline _query_length(q::Tuple{Vararg{AbstractVector}}) = length(q[1])
@inline _query_length(q) = length(q)

# ── Protocol function 2: k-th query point extraction ──
# User-facing: simple getindex-like interface. No Val{N} needed.
# Override this for custom query types — just return the k-th point (any indexable).
# SoA returns NTuple directly (N known from tuple type parameter).

@inline _query_extract(q::Tuple{Vararg{AbstractVector, N}}, k) where {N} =
    ntuple(d -> @inbounds(q[d][k]), Val(N))
@inline _query_extract(q, k) = @inbounds q[k]

# ── Internal: NTuple-guaranteed extraction for kernel consumption ──
# Wraps _query_extract with _as_ntuple — ensures NTuple{N, <:Real} for @generated kernels.
# NOT part of the user protocol — users never override this.

@inline _extract_query_point(q, k, ::Val{N}) where {N} =
    _as_ntuple(_query_extract(q, k), Val(N))

@inline _as_ntuple(x::NTuple{N, <:Real}, ::Val{N}) where {N} = x
@inline _as_ntuple(x, ::Val{N}) where {N} = ntuple(d -> @inbounds(x[d]), Val(N))

# ── Protocol function 3: scalar element type for output allocation ──
# Extracts the scalar floating type from query elements (not the element type itself).
# e.g. Vector{NTuple{2,Float64}} → Float64, not NTuple{2,Float64}.

@inline _query_eltype(q::Tuple{Vararg{AbstractVector}}) = promote_type(map(eltype, q)...)
@inline _query_eltype(q) = _scalar_eltype(eltype(q))

# Scalar type extraction helpers — dispatch on element type
@inline _scalar_eltype(::Type{T}) where {T <: Real} = T
@inline _scalar_eltype(::Type{T}) where {T <: Tuple} = promote_type(fieldtypes(T)...)
@inline _scalar_eltype(::Type{T}) where {T} = eltype(T)

# ── Protocol function 4: query consistency validation ──
# Default: no-op (AoS, custom types — single container, no mismatch possible).
# SoA: per-axis length consistency check (axes are independent containers).
# Override for custom multi-container query types that need validation.

@inline _query_validate(q) = nothing

@inline function _query_validate(q::Tuple{Vararg{AbstractVector}})
    n = length(q[1])
    for d in 2:length(q)
        length(q[d]) == n || _throw_query_axis_mismatch(n, d, length(q[d]))
    end
    return nothing
end

# ── Error helpers (@noinline cold path) ──

@noinline _throw_query_output_mismatch(nq, no) =
    throw(DimensionMismatch("output length $no != query length $nq"))
@noinline _throw_query_axis_mismatch(n1, d, nd) =
    throw(DimensionMismatch("query axis lengths differ: dim 1 has $n1, dim $d has $nd"))

# ── Derivative zero-fill trait ──
# Linear: 2nd+ derivative → all zeros. Constant: any derivative → all zeros.
# Default: no zero-fill (Cubic, Quadratic evaluate all derivative orders).

@inline _deriv_zero_fill(::AbstractInterpolantND, ::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} = false


# ========================================
# Unified Batch Evaluation (Generic ND)
# ========================================
#
# Single batch loop for all AbstractInterpolantND subtypes.
# Query extraction dispatches via _query_extract on query container type.

@inline function _batch_nd_unified!(
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
    length(query) == N || throw(
        DimensionMismatch(
            "expected $N-element vector, got $(length(query))-element vector"
        )
    )
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
    search_tuple = _resolve_search_nd(search, Val(N), queries, hint)
    if _deriv_zero_fill(itp, ops, Val(N))
        fill!(output, zero(eltype(output)))
        return output
    end
    _batch_nd_unified!(output, itp, queries, ops, search_tuple, hint)
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
