# ============================================================================
# _AxisAnchor — shared per-axis anchor backbone for gridded (separable) eval
# ============================================================================
#
# Shared RESOLUTION scaffold (searcher selection + hint chaining, InBounds lean
# arm, NoExtrap throwing BEFORE any output-sized allocation). Each method
# defines its own concrete named payload via `_resolve_anchor`, and consuming
# kernels stay per-family.
#
# `interval::I` is the physical search cell (shared `_AbstractIndices{2}` layer,
# same as `_AnchorLoc`): ordinary axes store one index (`_ContiguousIndices{2}`,
# right tap derived), exclusive-periodic axes store both (`_ExplicitIndices{2}`,
# the seam wraps). `payload::P` is a concrete named type — its identity is the
# method/op tag, so no phantom method parameter is needed.

struct _AxisAnchor{I <: _AbstractIndices{2}, P}
    interval::I
    payload::P
end

# Virtual properties: `idxL`/`idxR` read through the interval; every other
# symbol forwards to the named payload's field (`alpha`, `inv_h`, `dL`, `h`,
# `select_right`, …). Val-dispatch keeps each access a single folded `getfield`.
@inline Base.getproperty(a::_AxisAnchor, s::Symbol) = _get_axis_anchor_property(a, Val(s))
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:interval}) = getfield(a, :interval)
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:payload}) = getfield(a, :payload)
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:idxL}) = getfield(a, :interval)[Val(1)]
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{:idxR}) = getfield(a, :interval)[Val(2)]
@inline _get_axis_anchor_property(a::_AxisAnchor, ::Val{s}) where {s} = getproperty(getfield(a, :payload), s)

# ---- named per-method payloads -----------------------------------------------
# Concrete, distinct even when fields coincide (the semantic identity is the
# dispatch/routing tag). Phantom arithmetic type params carry the query carrier
# (`one(Tq)` reconstructed without storing it) for promotion/AD fidelity.

# Constant: the physical interval is stored; `select_right` picks the node at
# gather time. Carrier `one(Tq)` is reconstructed from the type param.
struct _ConstantValuePayload{Tq}
    select_right::Bool
end
@inline _carrier(::_ConstantValuePayload{Tq}) where {Tq} = one(Tq)

# Linear op-minimal payloads: EvalValue keeps only `alpha`; EvalDeriv1 keeps
# only `inv_h` (the carrier arithmetic type is retained as `Talpha` so the
# kernel can reconstruct `one(Talpha)` without storing `alpha`); every higher
# derivative of a degree-1 polynomial is zero → a zero-size payload. The per-op
# payload type is chosen once in `_axis_anchor_type`; `_resolve_anchor` and the
# consuming kernel then dispatch on it.
struct _LinearValuePayload{Talpha}
    alpha::Talpha
end
struct _LinearDeriv1Payload{Talpha, Tinv}
    inv_h::Tinv
end
struct _LinearZeroPayload{Talpha} end

@inline _linear_payload_type(::EvalValue, ::Type{Tα}, ::Type{Tinv}) where {Tα, Tinv} = _LinearValuePayload{Tα}
@inline _linear_payload_type(::EvalDeriv1, ::Type{Tα}, ::Type{Tinv}) where {Tα, Tinv} = _LinearDeriv1Payload{Tα, Tinv}
@inline _linear_payload_type(::AbstractEvalOp, ::Type{Tα}, ::Type{Tinv}) where {Tα, Tinv} = _LinearZeroPayload{Tα}

# Op-minimal gridded combine: the fused/fullbuffer kernels pass the whole anchor
# so each op reads only the field its payload carries. Results match the
# point-wise 1D `_linear_kernel(op, yL, yR, inv_h, α)` exactly (value blend /
# slope / carrier zero); the carrier `one(Tα)` is reconstructed from the payload
# type parameter rather than stored. (These live here, beside the payloads, so
# the `_AxisAnchor` signatures are in scope — gridded_query.jl is included
# first.)
@inline _linear_kernel(::EvalValue, yL::Tv, yR::Tv, a::_AxisAnchor{<:Any, <:_LinearValuePayload}) where {Tv} =
    _linear_value_blend(a.alpha, yL, yR)

@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, a::_AxisAnchor{<:Any, _LinearDeriv1Payload{Tα, Tinv}}) where {Tv, Tα, Tinv}
    Tc = _promote_eltype(_coeff_op, Tinv, Tv)
    return _fielddiff(Tc, yR, yL) * a.inv_h * one(Tα)
end

@inline _linear_kernel(::AbstractEvalOp, yL::Tv, yR::Tv, a::_AxisAnchor{<:Any, _LinearZeroPayload{Tα}}) where {Tv, Tα} =
    (0 * yL + 0 * yR) * one(Tα)

# Local Hermite: geometry-only `(dL, h, inv_h)`; slopes are data-dependent and
# computed in-pass. All fields consumed by the Hermite basis for every op.
struct _LocalHermitePayload{Tdl, Th, Tinv}
    dL::Tdl
    h::Th
    inv_h::Tinv
end

# Cubic / quadratic nodal-partials cell geometry. Distinct names though the
# fields currently coincide.
struct _CubicPartialsPayload{Tdl, Th, Tinv}
    dL::Tdl
    h::Th
    inv_h::Tinv
end

struct _QuadraticPartialsPayload{Tdl, Th, Tinv}
    dL::Tdl
    h::Th
    inv_h::Tinv
end

# Interval representation per axis type — reuses the `_AnchorLoc` selector's
# invariant (contiguous ordinary / explicit exclusive-periodic).
@inline _interval_type(::AbstractVector) = _ContiguousIndices{2}
@inline _interval_type(::_ExclusivePeriodicAxis) = _ExplicitIndices{2}

# Anchor vector eltype per (method, axis types, pass value type). `Tvals` is
# the eltype of the array the consuming pass READS (raw data for pass 1 /
# raw-local methods; the promoted intermediate for later Hermite passes) —
# it feeds the value-matched width type exactly like point-wise eval.
# `op` selects op-minimal payloads (only Linear specializes; the others keep one
# payload per method and ignore it).
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

# One method covers ordinary AND exclusive-periodic Linear: the interval type
# selector carries the right-tap distinction, so only the op-minimal payload
# type varies (value → `alpha`, deriv1 → `inv_h`, higher → zero-size).
@inline function _axis_anchor_type(
        m::LinearInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tinv = _promote_eltype(_inv_op, Tw)
    Tα = promote_type(eltype(grid), eltype(targets), Tinv)
    return _AxisAnchor{_interval_type(grid), _linear_payload_type(op, Tα, Tinv)}
end

@inline function _axis_anchor_type(
        m::AbstractLocalHermiteInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tdl = promote_type(eltype(grid), eltype(targets), Tw)
    return _AxisAnchor{_interval_type(grid), _LocalHermitePayload{Tdl, Tw, Tw}}
end

# ---- per-method payload resolution -------------------------------------------
# Constant: the side offset selects the node at gather time (`select_right`);
# the anchor stores the physical interval, not a folded index. Clamp/Fill folds
# the coordinate into the boundary cell (`dL ∈ [0, h]`) — the point-wise ND
# surface clamps the query COORDINATE before its kernel, so an OOB-left raw
# `dL < 0` would flip `RightSide`'s `iszero` test to the wrong node. Wrap folds
# in `_anchor_loc`, Fill's slabs are a post-pass, NoExtrap throws in the loop.
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

# EvalValue: only the in-cell weight `alpha` (Clamp/Fill fold it into [0, 1]).
@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{I, _LinearValuePayload{Tα}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tα}
    inv_h = _get_inv_h(grid, idxL, xL, xR)
    α = Tα(_alpha_of(xq, xL, inv_h))
    if extrap isa Union{ClampExtrap, FillExtrap}
        α = clamp(α, zero(Tα), one(Tα))
    end
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LinearValuePayload{Tα}}(interval, _LinearValuePayload{Tα}(α))
end

# EvalDeriv1: only the reciprocal width. Clamp/Fill needs no fold — the boundary
# cell's slope is exactly this cell's `inv_h` (the coordinate already resolved
# to the boundary cell during search).
@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{I, _LinearDeriv1Payload{Tα, Tinv}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tα, Tinv}
    inv_h = Tinv(_get_inv_h(grid, idxL, xL, xR))
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LinearDeriv1Payload{Tα, Tinv}}(interval, _LinearDeriv1Payload{Tα, Tinv}(inv_h))
end

# EvalDeriv2 and higher: the derivative is zero everywhere → no stored geometry.
@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{I, _LinearZeroPayload{Tα}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tα}
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LinearZeroPayload{Tα}}(interval, _LinearZeroPayload{Tα}())
end

# Local Hermite: location-only payload `(dL, h, inv_h)` — slopes are
# data-dependent (and nonlinear), computed in-pass; geometry runs at the
# value-matched `Tw` exactly like the point-wise 1D arm. Clamp/Fill folds the
# coordinate into the cell (`dL ∈ [0, h]`), matching the point-wise ND
# surface, which clamps the query coordinate and still evaluates the kernel
# (a Clamp-OOB derivative is the boundary cell polynomial's slope): t = 0 is
# basis-exact; t = 1 is `h·inv_h` (≤ ~1 ulp).
@inline function _resolve_anchor(
        m::AbstractLocalHermiteInterp,
        ::Type{_AxisAnchor{I, _LocalHermitePayload{Tdl, Tw, Tw2}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tdl, Tw, Tw2}
    h = _get_h(Tw, grid, idxL)
    inv_h = _get_inv_h(Tw, grid, idxL)
    dL = Tdl(xq - xL)
    if extrap isa Union{ClampExtrap, FillExtrap}
        dL = clamp(dL, zero(Tdl), Tdl(h))
    end
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LocalHermitePayload{Tdl, Tw, Tw2}}(interval, _LocalHermitePayload{Tdl, Tw, Tw2}(dL, Tw(h), Tw2(inv_h)))
end

# ---- shared resolution loop ---------------------------------------------------
# Mirrors `_gridded_anchors_loop!` (see its notes): the searcher union splits at
# this call; @inline keeps the hinted arm's fresh `RefHint` inside the caller's
# compiled unit (allocation-elided). NoExtrap throws the canonical axis-named
# error on the FIRST OOB coordinate — before any output-sized buffer exists.
@inline function _axis_anchors_loop!(
        anchors::Vector{A},
        m::AbstractInterpMethod,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        searcher::Searcher
    ) where {A <: _AxisAnchor}
    k = 0
    @inbounds for xq in targets
        k += 1
        if extrap isa InBounds
            # caller-asserted in-domain: lean search, same call the default
            # path reaches after its `_oob_state` check (bit-identical anchors).
            idx, idxR, xL, xR = search_interval(searcher, grid, xq, extrap)
            anchors[k] = _resolve_anchor(m, A, grid, idx, idxR, xq, xL, xR, extrap)
            continue
        end
        loc = _anchor_loc(grid, xq, extrap isa WrapExtrap, searcher)
        if extrap isa NoExtrap && loc.state != IN_DOMAIN
            _throw_domain_error(xq, grid, dim)
        end
        anchors[k] = _resolve_anchor(m, A, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
    end
    return anchors
end

function _axis_anchors_pooled(
        pool,
        m::AbstractInterpMethod,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    A = _axis_anchor_type(m, grid, targets, op, Tvals)
    anchors = acquire!(pool, A, length(targets))
    return _axis_anchors_loop!(anchors, m, grid, targets, extrap, dim, _gridded_build_searcher(grid, targets))
end

# All-axes anchor tuple via literal-index construction — an ntuple closure over
# the heterogeneous (methods, grids, targets) tuples boxes on some type
# combinations (the `_gridded_hulls` lesson). `T1`/`Trest` split the pass value
# type: Hermite's pass 1 reads raw data (`Tv`) while later passes read the
# promoted intermediate (`Tmid`); raw-local single-pass methods pass the same
# type twice.
@generated function _axis_anchors_all(
        pool,
        methods,
        grids,
        targets,
        extraps,
        ops,
        ::Type{T1},
        ::Type{Trest},
        ::Val{N}
    ) where {T1, Trest, N}
    exprs = [
        :(
                _axis_anchors_pooled(
                    pool, methods[$d], grids[$d], targets[$d], extraps[$d], $d,
                    ops[$d], $(d == 1 ? :T1 : :Trest)
                )
            )
            for d in 1:N
    ]
    return :(($(exprs...),))
end
