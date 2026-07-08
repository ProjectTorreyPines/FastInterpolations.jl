# ============================================================================
# GriddedQuery — unified persistent callable for every ND interpolant
# ============================================================================
#
# One `itp(gq)` / `itp(out, gq)` pair on the abstract type, so ALL ND
# interpolants evaluate a `GriddedQuery` through a single entry and return the
# N-D `size(gq)` array. Persistent interpolants adapt themselves to the same
# prepared method-tuple dispatcher used by one-shot `interp!`:
#   * separable-capable method families add one `_gridded_eval_methods!` arm
#     beside their kernels (fast anchor-reuse path);
#   * unsupported tuples fall through to the shared point-wise batch core into a
#     flat view of `out`.
#
# Hoisting the pair to `AbstractInterpolantND` also removes the N = 1 method
# ambiguity the per-type functors had: those dispatched on the CONCRETE type
# (winning arg 1) while the batch callable `(itp)(::AbstractVector, queries)`
# won arg 2 — a split. Sharing the abstract arg 1 lets `GriddedQuery ⊂ Any`
# win arg 3 outright, so no per-N disambiguation method is needed.

"""
    (itp::AbstractInterpolantND)(gq::GriddedQuery; deriv = EvalValue(), extrap = nothing, search, hint)
    (itp::AbstractInterpolantND)(out::AbstractArray, gq::GriddedQuery; deriv = EvalValue(), extrap = nothing, search, hint)

Evaluate an N-D interpolant at every combination of `gq.axes` coordinates
(rectilinear / tensor-product), returning an N-D array of `size(gq)`.
Supported method families take the gridded fast path by resolving per-axis
anchors once and reusing them across the grid; unsupported methods evaluate
point-wise. `deriv`/`extrap` mirror point-wise ND eval; the 2-arg form writes
into `out`.
"""
function (itp::AbstractInterpolantND{Tg, Tv, N})(
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    extraps = _resolve_extrap_override_nd(itp, extrap)
    return _gridded_alloc_itp(itp, gq, ops, extraps, search, hint)
end

# Allocating arm. The default allocates the N-D output, then fills in place.
# Separable types that validate the domain during anchor build (linear) override
# this to build anchors — hence fire NoExtrap's throw — BEFORE the O(∏M_d)
# output allocation (pinned: a throwing query must not leave the full matrix
# behind). Point-wise methods have no such early exit and use the default.
@inline function _gridded_alloc_itp(
        itp::AbstractInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        ops,
        extraps,
        search,
        hint
    ) where {Tg, Tv, N}
    out = Array{_promote_eltype(itp, _query_eltype(gq)), N}(undef, size(gq))
    _gridded_eval_itp!(out, itp, gq, ops, extraps, search, hint)
    return out
end

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        out::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _gridded_functor_call!(out, itp, gq, deriv, extrap, search, hint)
end

# N = 1 disambiguation: a 1-axis GriddedQuery's output IS an `AbstractVector`
# (`AbstractArray{<:Any,1}`), so at N = 1 the in-place functor above overlaps the
# generic batch callable `(itp)(::AbstractVector, queries)` — arg 2 equal, arg 3
# `GriddedQuery ⊂ queries`, a split. One abstract-level method (more specific than
# both) closes it — vs the three CONCRETE per-type functors this replaced.
function (itp::AbstractInterpolantND{Tg, Tv, 1})(
        out::AbstractVector,
        gq::GriddedQuery{<:Tuple{Any}};
        deriv::Union{DerivOp, Tuple{DerivOp}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{AbstractSearchPolicy}} = itp.searches,
        hint::Union{Nothing, Tuple{Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv}
    return _gridded_functor_call!(out, itp, gq, deriv, extrap, search, hint)
end

@inline function _gridded_functor_call!(
        out::AbstractArray{<:Any, N},
        itp::AbstractInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        deriv,
        extrap,
        search,
        hint
    ) where {Tg, Tv, N}
    size(out) == size(gq) || throw(
        DimensionMismatch("output size $(size(out)) != query size $(size(gq))")
    )
    ops = _resolve_deriv_nd(deriv, Val(N))
    extraps = _resolve_extrap_override_nd(itp, extrap)
    _gridded_eval_itp!(out, itp, gq, ops, extraps, search, hint)
    return out
end

# ---- common separable method dispatch ----------------------------------------
# Both public surfaces feed this same prepared dispatcher:
#
#   * one-shot `interp!(out, grids, data, GriddedQuery(...); method = methods)`
#     resolves/caches raw inputs, then calls `_gridded_eval_methods!`;
#   * persistent `itp(out, gq)` extracts the interpolant's method tuple and calls
#     the same function with the already stored grids/data/extraps.
#
# Method families that have a separable tensor-product kernel add one
# `_gridded_eval_methods!` method beside that kernel. Unsupported method tuples
# return `false` and fall back to the point-wise batch core.
@inline _gridded_eval_methods!(
    out, grids, data, targets, methods, ops, extraps
) = false

struct _NoGriddedMethods end

@inline _gridded_methods(::AbstractInterpolantND) = _NoGriddedMethods()
@inline _gridded_methods(::LinearInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    ntuple(_ -> LinearInterp(NoBC()), Val(N))
@inline _gridded_methods(itp::ConstantInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    map(s -> ConstantInterp(s, NoBC()), itp.sides)
@inline _gridded_methods(itp::HeteroInterpolantND) = itp.methods

@inline _gridded_eval_itp_methods!(
    out::AbstractArray{<:Any, N},
    itp::AbstractInterpolantND{Tg, Tv, N},
    gq::GriddedQuery,
    ops,
    extraps,
    ::_NoGriddedMethods
) where {Tg, Tv, N} = false

@inline function _gridded_eval_itp_methods!(
        out::AbstractArray{<:Any, N},
        itp::AbstractInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        ops,
        extraps,
        methods
    ) where {Tg, Tv, N}
    return _gridded_eval_methods!(out, itp.grids, itp.data, gq.axes, methods, ops, extraps)
end

# Default (non-separable) arm: first ask the shared method dispatcher whether
# this interpolant can use a separable GriddedQuery kernel. If not, use the
# shared point-wise batch core writing through a flat, aliasing view of the N-D
# output (GriddedQuery unravels column-major, so linear-index fills land at the
# right N-D position).
@inline function _gridded_eval_itp!(
        out::AbstractArray{<:Any, N},
        itp::AbstractInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        ops,
        extraps,
        search,
        hint
    ) where {Tg, Tv, N}
    _gridded_eval_itp_methods!(out, itp, gq, ops, extraps, _gridded_methods(itp)) && return out
    return _nd_batch_pointwise!(vec(out), itp, gq, ops, extraps, search, hint)
end
