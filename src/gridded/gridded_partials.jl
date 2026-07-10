# ============================================================================
# GriddedQuery — cubic/quadratic fused-anchor evaluation over nodal partials
# ============================================================================

# Cubic and quadratic ND interpolants already precompute a leading partials
# tensor. This path keeps that build unchanged and only replaces the query loop:
# anchors resolve each target axis once, then a fused output loop calls the
# existing scalar cell kernels without per-output interval search.

# Cubic/quadratic partials cell geometry: distinct named payloads (byte-equal
# fields today, but the quadratic cell kernel does not consume `h`). One method
# per method type so each carries its own payload identity.
@inline function _axis_anchor_type(
        m::CubicInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tdl = promote_type(eltype(grid), eltype(targets), Tw)
    return _AxisAnchor{_interval_type(grid), _CubicPartialsPayload{Tdl, Tw, Tw}}
end

@inline function _axis_anchor_type(
        m::QuadraticInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tdl = promote_type(eltype(grid), eltype(targets), Tw)
    return _AxisAnchor{_interval_type(grid), _QuadraticPartialsPayload{Tdl, Tw, Tw}}
end

@inline function _resolve_anchor(
        m::CubicInterp,
        ::Type{_AxisAnchor{I, _CubicPartialsPayload{Tdl, Tw, Tw2}}},
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
    return _AxisAnchor{I, _CubicPartialsPayload{Tdl, Tw, Tw2}}(interval, _CubicPartialsPayload{Tdl, Tw, Tw2}(dL, Tw(h), Tw2(inv_h)))
end

@inline function _resolve_anchor(
        m::QuadraticInterp,
        ::Type{_AxisAnchor{I, _QuadraticPartialsPayload{Tdl, Tw, Tw2}}},
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
    return _AxisAnchor{I, _QuadraticPartialsPayload{Tdl, Tw, Tw2}}(interval, _QuadraticPartialsPayload{Tdl, Tw, Tw2}(dL, Tw(h), Tw2(inv_h)))
end

@generated function _gridded_fused_partials!(
        out::AbstractArray{<:Any, N},
        partials::AbstractArray{<:Any, NP1},
        anchors::NTuple{N, Vector{<:_AxisAnchor}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        ::Val{Kind}
    ) where {N, NP1, Kind}
    kernel = if Kind === :cubic
        :_eval_nd_cell
    elseif Kind === :quadratic
        :_eval_nd_quad_cell
    else
        error("unsupported gridded partials kind $Kind")
    end
    js = [Symbol(:j_, d) for d in 1:N]
    idxs = [Symbol(:idx_, d) for d in 1:N]
    dls = [Symbol(:dL_, d) for d in 1:N]
    hs = [Symbol(:h_, d) for d in 1:N]
    invhs = [Symbol(:invh_, d) for d in 1:N]
    body = :(
        out[$(js...)] = $kernel(
            partials,
            ($(idxs...),),
            ($(hs...),),
            ($(invhs...),),
            ($(dls...),),
            ops,
        )
    )
    for d in 1:N
        a = Symbol(:a_, d)
        body = quote
            for $(js[d]) in eachindex(anchors[$d])
                $a = anchors[$d][$(js[d])]
                $(idxs[d]) = $a.idxL
                $(dls[d]) = $a.dL
                $(hs[d]) = $a.h
                $(invhs[d]) = $a.inv_h
                $body
            end
        end
    end
    return quote
        @inbounds $body
        return out
    end
end

@with_pool pool function _gridded_eval_cubic_partials!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        partials::AbstractArray{Tp, NP1},
        targets::Tuple,
        methods::Tuple{Vararg{CubicInterp, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        sample
    ) where {Tp, N, NP1}
    out_size = map(length, targets)
    size(out) == out_size || throw(
        DimensionMismatch("output size $(size(out)) != query size $out_size")
    )
    anchors = _axis_anchors_all(pool, methods, grids, targets, extraps, ops, Tp, Tp, Val(N))
    if !any(iszero, out_size)
        _gridded_fused_partials!(out, partials, anchors, ops, Val(:cubic))
    end
    return _gridded_fill_oob_sample!(out, grids, sample, targets, extraps, ops)
end

@with_pool pool function _gridded_eval_quadratic_partials!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        partials::AbstractArray{Tp, NP1},
        targets::Tuple,
        methods::Tuple{Vararg{QuadraticInterp, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        sample
    ) where {Tp, N, NP1}
    out_size = map(length, targets)
    size(out) == out_size || throw(
        DimensionMismatch("output size $(size(out)) != query size $out_size")
    )
    anchors = _axis_anchors_all(pool, methods, grids, targets, extraps, ops, Tp, Tp, Val(N))
    if !any(iszero, out_size)
        _gridded_fused_partials!(out, partials, anchors, ops, Val(:quadratic))
    end
    return _gridded_fill_oob_sample!(out, grids, sample, targets, extraps, ops)
end

# ---- persistent functor adapters --------------------------------------------

@inline _gridded_methods(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    map(CubicInterp, itp.bcs)
@inline _gridded_methods(itp::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    map(QuadraticInterp, itp.bcs)

@inline function _gridded_eval_itp_methods!(
        out::AbstractArray{<:Any, N},
        itp::CubicInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        ops,
        extraps,
        methods::Tuple{Vararg{CubicInterp, N}}
    ) where {Tg, Tv, N}
    _gridded_eval_cubic_partials!(
        out, itp.grids, itp.nodal_derivs.partials, gq.axes, methods, ops, extraps, _sample_data(itp)
    )
    return true
end

@inline function _gridded_eval_itp_methods!(
        out::AbstractArray{<:Any, N},
        itp::QuadraticInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        ops,
        extraps,
        methods::Tuple{Vararg{QuadraticInterp, N}}
    ) where {Tg, Tv, N}
    _gridded_eval_quadratic_partials!(
        out, itp.grids, itp.nodal_derivs.partials, gq.axes, methods, ops, extraps, _sample_data(itp)
    )
    return true
end

# ---- one-shot GriddedQuery hooks --------------------------------------------

@inline function _try_gridded_oneshot_methods!(
        out_nd::AbstractArray{<:Any, N},
        grids,
        data::AbstractArray{Tv, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}},
        methods::Tuple{Vararg{CubicInterp, N}},
        extrap,
        deriv,
        coeffs
    ) where {Tv, N}
    coeffs isa OnTheFly && return false
    _cubic_gridded_oneshot_fused!(out_nd, grids, data, gq.axes, methods, extrap, deriv)
    return true
end

@with_pool pool function _cubic_gridded_oneshot_fused!(
        out_nd::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{Vararg{CubicInterp, N}},
        extrap,
        deriv
    ) where {Tv, N}
    grids_typed, Tg, Tv_p, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)
    bcs = map(m -> m.bc, methods)
    _validate_nd_bcs!(grids_typed, bcs, data, Val(N))
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv_p)
    ops = _resolve_deriv_nd(deriv, Val(N))

    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids_typed, data, bcs)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_p)
    Tz = _promote_eltype(_coeff_op, Tg, Tv)
    partials = acquire!(pool, Tz, (1 << N, size(data_p)...))
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)
    return _gridded_eval_cubic_partials!(
        out_nd, grids_p, partials, targets, map(CubicInterp, bcs_p), ops, extraps_eff, @inbounds(first(data_p))
    )
end

@inline function _try_gridded_oneshot_methods!(
        out_nd::AbstractArray{<:Any, N},
        grids,
        data::AbstractArray{Tv, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}},
        methods::Tuple{Vararg{QuadraticInterp, N}},
        extrap,
        deriv,
        coeffs
    ) where {Tv, N}
    coeffs isa OnTheFly && return false
    _quadratic_gridded_oneshot_fused!(out_nd, grids, data, gq.axes, methods, extrap, deriv)
    return true
end

@with_pool pool function _quadratic_gridded_oneshot_fused!(
        out_nd::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{Vararg{QuadraticInterp, N}},
        extrap,
        deriv
    ) where {Tv, N}
    grids_typed, Tg, _, _ = _nd_promote_grids(grids, data)
    _validate_nd_grids(grids_typed, data)
    bcs = map(m -> m.bc, methods)
    _validate_quadratic_bcs_nd(bcs)
    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))

    grids_c = map(g -> _cache_axis_pooled(pool, g), grids_typed)
    extraps_eff = map(_resolve_extrap, extraps_val, grids_c)
    Tz = _promote_eltype(_coeff_op, Tg, Tv)
    partials = acquire!(pool, Tz, (1 << N, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids_c, data, bcs)
    return _gridded_eval_quadratic_partials!(
        out_nd, grids_c, partials, targets, map(QuadraticInterp, bcs), ops, extraps_eff, @inbounds(first(data))
    )
end
