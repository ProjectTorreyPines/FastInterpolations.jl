# ============================================================================
# GriddedQuery — constant (nearest-node) separable evaluation
# ============================================================================
#
# Constant interpolation degenerates the separable machinery to a pure GATHER:
# each axis independently resolves ONE source node (side offset folded into the
# anchor's `idx` at build), so the whole resize is
#   out[j₁,…] = data[sel₁[j₁], …] * (one(dL₁) * …)     — no blend, no passes.
# A single fused loop is strictly optimal (a multi-pass would only re-copy),
# so there is no strategy selection. The trailing carrier chain and the
# any-deriv `* 0` short-circuit mirror `_constant_nd_evaluate` exactly.

# Gather generator over N. The ops tuple type picks the point-wise formula at
# compile time: all-EvalValue → plain gather × carrier; any deriv → `* 0`
# (a constant's derivative is zero everywhere — same promote, zero value).

# ---- anchor payload + resolution (constant) ----------------------------------
# Constant: the physical interval is stored; `select_right` picks the node at
# gather time. Carrier `one(Tq)` is reconstructed from the type param (op-agnostic
# — the derivative `* 0` is a gather-kernel concern, not a payload one).
struct _ConstantValuePayload{Tq} <: _AbstractAnchorPayload
    select_right::Bool
end
@inline _carrier(::_ConstantValuePayload{Tq}) where {Tq} = one(Tq)

@inline function _axis_anchor_type(
        m::ConstantInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    Tone = promote_type(eltype(grid), eltype(targets))
    return _AxisAnchor{_interval_type(grid), _ConstantValuePayload{Tone}}
end

# The side offset selects the node at gather time (`select_right`); the anchor
# stores the physical interval, not a folded index. Clamp/Fill folds the
# coordinate into the boundary cell (`dL ∈ [0, h]`) — the point-wise ND surface
# clamps the query COORDINATE before its kernel, so an OOB-left raw `dL < 0`
# would flip `RightSide`'s `iszero` test to the wrong node. Wrap folds in
# `_anchor_loc`, Fill's slabs are a post-pass, NoExtrap throws in the loop.
@inline function _resolve_anchor(
        m::ConstantInterp,
        ::Type{_AxisAnchor{I, _ConstantValuePayload{Tone}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tone}
    h = _get_h(grid, idxL, xL, xR)
    dL = xq - xL
    if extrap isa Union{ClampExtrap, FillExtrap}
        dL = clamp(dL, zero(dL), oftype(dL, h))
    end
    select_right = _compute_single_offset(m.side, h, dL) == 1
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _ConstantValuePayload{Tone}}(interval, _ConstantValuePayload{Tone}(select_right))
end

@generated function _constant_gridded_gather!(
        out::AbstractArray{<:Any, N},
        data::AbstractArray{<:Any, N},
        anchors::Tuple{Vararg{Vector, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}}
    ) where {N}
    js = [Symbol(:j_, d) for d in 1:N]
    sels = [Symbol(:sel_, d) for d in 1:N]
    ones_ = [Symbol(:one_, d) for d in 1:N]
    carrier = Expr(:call, :*, ones_...)
    value = :(data[$(sels...)] * $carrier)
    ops <: NTuple{N, EvalValue} || (value = :($value * 0))
    body = :(out[$(js...)] = $value)
    for d in 1:N   # d = 1 built first → innermost loop (stride-1 writes)
        a = Symbol(:a_, d)
        body = quote
            for $(js[d]) in eachindex(anchors[$d])
                $a = anchors[$d][$(js[d])]
                $(sels[d]) = ifelse($a.select_right, $a.idxR, $a.idxL)
                $(ones_[d]) = _carrier($a.payload)
                $body
            end
        end
    end
    return quote
        @inbounds $body
        return out
    end
end

# ---- core ---------------------------------------------------------------------
# Anchors build (firing NoExtrap's O(ΣM_d) validation) BEFORE any output write;
# Fill's OOB slabs reuse the shared post-pass.
@with_pool pool function _constant_gridded_eval!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{Vararg{ConstantInterp, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {Tv, N}
    out_size = map(length, targets)
    size(out) == out_size || throw(
        DimensionMismatch("output size $(size(out)) != query size $out_size")
    )
    anchors = _axis_anchors_all(pool, methods, grids, targets, extraps, ops, Tv, Tv, Val(N))
    any(iszero, out_size) && return out
    _constant_gridded_gather!(out, data, anchors, ops)
    return _gridded_fill_oob!(out, grids, data, targets, extraps, ops)
end

# ---- one-shot API ---------------------------------------------------------------
# Thin named spelling over `interp`/`interp!`. The generic allocating path uses
# the method tuple's `_select_op` shape witness, so constant keeps its natural
# selection eltype while sharing the same gridded fast-path gate as every method.
"""
    constant_interp(grids, data, gq::GriddedQuery; bc, side, extrap, deriv)
    constant_interp!(out, grids, data, gq::GriddedQuery; bc, side, extrap, deriv)

One-shot N-D constant (nearest-node) interpolation at every combination of
`gq.axes` coordinates (rectilinear resize): per-axis node selections are
resolved once and the output is a single gather — no per-point search.
"""
function constant_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {N}
    return interp(grids, data, gq; method = _constant_gridded_methods(side, bc, Val(N)), extrap = extrap, deriv = deriv)
end

function constant_interp!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{Vararg{AbstractSide}}} = NearestSide(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {N}
    return interp!(out, grids, data, gq; method = _constant_gridded_methods(side, bc, Val(N)), extrap = extrap, deriv = deriv)
end

# N = 1 disambiguation: a flat vector IS the 1-D output array, but it also
# matches the generic batch entry `constant_interp!(output::AbstractVector,
# grids, data, queries)` — pin the intersection to the gridded arm.
function constant_interp!(
        out::AbstractVector,
        grids::Tuple{AbstractVector},
        data::AbstractVector{Tv},
        gq::GriddedQuery{<:Tuple{Any}};
        bc::Union{AbstractBC, Tuple{AbstractBC}} = NoBC(),
        side::Union{AbstractSide, Tuple{AbstractSide}} = NearestSide(),
        extrap::Union{AbstractExtrap, Tuple{AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{DerivOp}} = EvalValue()
    ) where {Tv}
    return interp!(out, grids, data, gq; method = _constant_gridded_methods(side, bc, Val(1)), extrap = extrap, deriv = deriv)
end

@inline function _constant_gridded_methods(side, bc, ::Val{N}) where {N}
    bcs = _resolve_bcs_nd(bc, Val(N))
    sides = _resolve_side_nd(side, Val(N))
    return map(ConstantInterp, sides, bcs)
end

# ---- gridded method dispatch (constant separable arm) -------------------------
# Prepared constant gridded evaluation: both one-shot `interp(..., gq;
# method = ConstantInterp())` and persistent `itp(gq)` arrive here after their
# grids/extraps/method tuple have been resolved.
@inline function _gridded_eval_methods!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{ConstantInterp{<:AbstractSide, NoBC}, Vararg{ConstantInterp{<:AbstractSide, NoBC}}},
        ops,
        extraps
    ) where {Tv, N}
    _constant_gridded_eval!(out, grids, data, targets, methods, ops, extraps)
    return true
end

# ---- unified `interp` fast-path -------------------------------------------------
# All-constant methods with NoBC (periodic seam cells have `idx_R ≠ idx + 1`
# and fall through to the point-wise batch). Deriv ops are supported — the
# gather's compile-time `* 0` arm mirrors `_constant_nd_evaluate`.
@inline function _try_gridded_oneshot_methods!(
        out_nd::AbstractArray{<:Any, N},
        grids,
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{ConstantInterp{<:AbstractSide, NoBC}, Vararg{ConstantInterp{<:AbstractSide, NoBC}}},
        extrap,
        deriv
    ) where {Tv, N}
    grids_typed, _, _ = _nd_promote_grids_raw(grids, data)
    _validate_nd_grids(grids_typed, data)
    bcs = map(m -> m.bc, methods)
    ops = _resolve_deriv_nd(deriv, Val(N))
    extraps = _resolve_extrap(extrap, bcs, Val(N), Tv)
    return _gridded_eval_methods!(out_nd, grids_typed, data, targets, methods, ops, extraps)
end
