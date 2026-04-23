# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                CONSTANT ONE-SHOT SERIES INTERPOLATION                    ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → constant_anchor.jl → constant_oneshot_series.jl → ...
# Shared anchor eval: _constant_eval_at_anchor(y, x_last, aq, op, side, extrap) in constant_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 INTERNAL: PERIODIC CORE                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline @with_pool pool function _constant_oneshot_series_periodic!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::PeriodicBC,
        op::AbstractEvalOp,
        side::AbstractSide,
        searcher
    ) where {Tg}
    vecs = _series_vectors(s)
    n = length(x)
    x_p, y_p_first, extrap_p = _periodic_extend_1d_pooled!(pool, x, first(vecs), bc, WrapExtrap())
    n_p = length(x_p)
    x_last = eltype(x_p)(last(x_p))
    aq = _anchor_query(x_p, xq, Val(:constant), true, searcher)
    @inbounds output[1] = _constant_eval_at_anchor(y_p_first, x_last, aq, op, side, extrap_p)

    K = length(output)
    if K > 1
        if bc isa PeriodicBC{:exclusive}
            Tv_buf = eltype(y_p_first)
            y_p = acquire!(pool, Tv_buf, n_p)
            @inbounds for k in 2:K
                copyto!(y_p, 1, vecs[k], 1, n)
                y_p[n + 1] = vecs[k][1]
                output[k] = _constant_eval_at_anchor(y_p, x_last, aq, op, side, extrap_p)
            end
        else
            @inbounds for k in 2:K
                _check_periodic_endpoints(bc, vecs[k])
                output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, op, side, extrap_p)
            end
        end
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

@inline function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, Tg)
    K = n_series(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    output = Vector{Tv_out}(undef, K)
    if _is_periodic_bc(bc)
        searcher = _resolve_search(x, xq, search, hint)
        return _constant_oneshot_series_periodic!(output, x, s, xq, bc, deriv, side, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    @inbounds for k in 1:K
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
    end
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline function constant_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    x = _to_float(x, Tg)
    if _is_periodic_bc(bc)
        searcher = _resolve_search(x, xq, search, hint)
        return _constant_oneshot_series_periodic!(output, x, s, xq, bc, deriv, side, searcher)
    end
    _check_domain(x, xq, extrap)
    searcher = _resolve_search(x, xq, search, hint)
    aq = _anchor_query(x, xq, Val(:constant), extrap isa WrapExtrap, searcher)
    x_last = Tg(last(x))
    vecs = _series_vectors(s)
    @inbounds for k in eachindex(output)
        output[k] = _constant_eval_at_anchor(vecs[k], x_last, aq, deriv, side, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@with_pool pool function constant_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    _validate_series_lengths(s, length(x))
    x = _to_float(x, Tg)
    K = n_series(s)
    _validate_series_outputs(outputs, K, length(xqs))
    vecs = _series_vectors(s)

    if _is_periodic_bc(bc)
        x_p, y_p_first, extrap_p = _periodic_extend_1d_pooled!(pool, x, first(vecs), bc, WrapExtrap())
        Tg_p = eltype(x_p)
        x_last = Tg_p(last(x_p))
        searcher = _resolve_search(x_p, xqs, search, nothing)
        aq_vec = acquire!(pool, _ConstantAnchoredQuery{Tg_p, Tg_p}, length(xqs))
        _fill_anchors!(aq_vec, x_p, xqs, Val(:constant), true, searcher)
        @inbounds for j in eachindex(xqs)
            outputs[1][j] = _constant_eval_at_anchor(y_p_first, x_last, aq_vec[j], deriv, side, extrap_p)
        end
        if K > 1
            if bc isa PeriodicBC{:exclusive}
                n = length(x)
                Tv_buf = eltype(y_p_first)
                y_p = acquire!(pool, Tv_buf, length(x_p))
                @inbounds for k in 2:K
                    copyto!(y_p, 1, vecs[k], 1, n)
                    y_p[n + 1] = vecs[k][1]
                    for j in eachindex(xqs)
                        outputs[k][j] = _constant_eval_at_anchor(y_p, x_last, aq_vec[j], deriv, side, extrap_p)
                    end
                end
            else
                @inbounds for k in 2:K
                    _check_periodic_endpoints(bc, vecs[k])
                    for j in eachindex(xqs)
                        outputs[k][j] = _constant_eval_at_anchor(vecs[k], x_last, aq_vec[j], deriv, side, extrap_p)
                    end
                end
            end
        end
        return outputs
    end

    # Domain check: NoExtrap → throws if OOB, returns InBounds(); others → pass-through
    extrap_eff = _check_domain(x, xqs, extrap)
    searcher = _resolve_search(x, xqs, search, nothing)
    wrap = extrap_eff isa WrapExtrap
    x_last = Tg(last(x))
    # Pre-compute anchors via pool, then K outer × Q inner for cache locality
    aq_vec = acquire!(pool, _ConstantAnchoredQuery{Tg, Tg}, length(xqs))
    _fill_anchors!(aq_vec, x, xqs, Val(:constant), wrap, searcher)
    @inbounds for k in 1:K
        for j in eachindex(xqs)
            outputs[k][j] = _constant_eval_at_anchor(vecs[k], x_last, aq_vec[j], deriv, side, extrap_eff)
        end
    end
    return outputs
end

function constant_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Real}
    K = n_series(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    outputs = [Vector{Tv_out}(undef, length(xqs)) for _ in 1:K]
    constant_interp!(outputs, x, s, xqs; bc, side, extrap, deriv, search)
    return outputs
end

# NOTE: the former Real type promotion wrappers (Tg <: Real) have been removed.
# The hot-path methods above now use unconstrained Tg, and _to_float handles
# grid normalization for all types (Int, Float, Dual, etc.), preventing
# infinite recursion on duck types like ForwardDiff.Dual <: Real.
