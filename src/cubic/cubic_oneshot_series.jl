# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CUBIC ONE-SHOT SERIES INTERPOLATION                     ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides `cubic_interp(x, Series(y1,y2,...), xq; bc=...)` without constructing
# a CubicSeriesInterpolant. Uses pool allocation for z-buffer reuse.
#
# Include order: ... → cubic_anchor.jl → cubic_oneshot_series.jl → ...

# ─── Decoupled anchor kernel (reads raw y, z vectors) ───────────────────────

# EvalValue: 4-term dot product
@inline function _cubic_series_eval_kernel(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalValue
    )
    wyL, wyR, wzL, wzR = aq.w0
    @inbounds return muladd(
        wyR, y[aq.idx + 1], muladd(
            wyL, y[aq.idx],
            muladd(wzR, z[aq.idx + 1], wzL * z[aq.idx])
        )
    )
end

# EvalDeriv1: 4-term with w1
@inline function _cubic_series_eval_kernel(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalDeriv1
    )
    wyL, wyR, wzL, wzR = aq.w1
    @inbounds return muladd(
        wyR, y[aq.idx + 1], muladd(
            wyL, y[aq.idx],
            muladd(wzR, z[aq.idx + 1], wzL * z[aq.idx])
        )
    )
end

# EvalDeriv2: 2-term with w2 (z-only)
@inline function _cubic_series_eval_kernel(
        ::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalDeriv2
    )
    wzL, wzR = aq.w2
    @inbounds return muladd(wzR, z[aq.idx + 1], wzL * z[aq.idx])
end

# EvalDeriv3: 2-term with w3 (z-only)
@inline function _cubic_series_eval_kernel(
        ::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalDeriv3
    )
    wzL, wzR = aq.w3
    @inbounds return muladd(wzR, z[aq.idx + 1], wzL * z[aq.idx])
end

# DerivOp{N≥4}: zero
@inline function _cubic_series_eval_kernel(
        y::AbstractVector, ::AbstractVector,
        aq::_CubicAnchoredQuery, ::DerivOp{N}
    ) where {N}
    return 0 * (@inbounds y[aq.idx])
end

# ─── Decoupled anchor dispatch with extrap ───────────────────────────────────

# Inside domain, ExtendExtrap, WrapExtrap → just kernel
@inline function _cubic_series_eval_at_anchor(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, op::AbstractEvalOp, ::AbstractExtrap
    )
    return _cubic_series_eval_kernel(y, z, aq, op)
end

# NoExtrap → throw if OOB
@inline function _cubic_series_eval_at_anchor(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, op::AbstractEvalOp, ::NoExtrap
    )
    if aq.side != 0x00
        throw(DomainError(aq.xq, "query point outside domain"))
    end
    return _cubic_series_eval_kernel(y, z, aq, op)
end

# ClampExtrap / FillExtrap → boundary value if OOB
@inline function _cubic_series_eval_at_anchor(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, op::AbstractEvalOp, extrap::_ClampOrFill
    )
    if aq.side != 0x00
        y_bnd = aq.side == 0x01 ? first(y) : last(y)
        return _eval_extrapolation(op, y_bnd, extrap, aq.xq)
    end
    return _cubic_series_eval_kernel(y, z, aq, op)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      INTERNAL: NON-PERIODIC CORE                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Core BCPair scalar: solve per y-vector reusing z buffer
@inline @with_pool pool function _cubic_oneshot_series_bcpair!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::BCPair,
        extrap::AbstractExtrap,
        autocache::Bool,
        op::AbstractEvalOp,
        searcher
    ) where {Tg <: AbstractFloat}
    cache = _get_cubic_cache(x, bc, autocache)
    aq = _anchor_query(cache.x, xq, Val(:cubic), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    z = similar!(pool, first(vecs))
    @inbounds for k in eachindex(output)
        _solve_system!(z, cache, vecs[k], bc)
        output[k] = _cubic_series_eval_at_anchor(vecs[k], z, aq, op, extrap)
    end
    return output
end

# Core BCPair NTuple: solve per y-vector reusing z buffer
@inline @with_pool pool function _cubic_oneshot_series_bcpair_ntuple(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq,
        bc::BCPair,
        extrap::AbstractExtrap,
        autocache::Bool,
        op::AbstractEvalOp,
        searcher
    ) where {Tg <: AbstractFloat}
    cache = _get_cubic_cache(x, bc, autocache)
    aq = _anchor_query(cache.x, xq, Val(:cubic), extrap isa WrapExtrap, searcher)
    vecs = _series_vectors(s)
    z = similar!(pool, first(vecs))
    K = n_series(s)
    return ntuple(Val(K)) do k
        _solve_system!(z, cache, vecs[k], bc)
        _cubic_series_eval_at_anchor(vecs[k], z, aq, op, extrap)
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      INTERNAL: PERIODIC CORE                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Periodic: reuse _cubic_periodic_solve! per y-vector.
# Each call extends exclusive→inclusive grid, solves, returns (cache, y_p, z).
@inline @with_pool pool function _cubic_oneshot_series_periodic!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq,
        bc::PeriodicBC,
        op::AbstractEvalOp,
        autocache::Bool,
        searcher
    ) where {Tg <: AbstractFloat}
    vecs = _series_vectors(s)
    # Solve first series to get cache and extended grid
    cache, y_p_first, z_first = _cubic_periodic_solve!(pool, x, first(vecs), bc, autocache)
    # Build anchor on the extended (inclusive) grid
    aq = _anchor_query(cache.x, xq, Val(:cubic), true, searcher)
    output[1] = _cubic_series_eval_kernel(y_p_first, z_first, aq, op)

    # Solve remaining series reusing same cache
    for k in 2:length(output)
        _, y_p_k, z_k = _cubic_periodic_solve!(pool, x, vecs[k], bc, autocache)
        @inbounds output[k] = _cubic_series_eval_kernel(y_p_k, z_k, aq, op)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    cubic_interp(x, Series(y1, y2, ...), xq; bc, extrap, autocache, deriv, search, hint) → NTuple

One-shot cubic spline interpolation of multiple y-series at a single query point.
Returns `NTuple{K}` when `Series` is constructed from varargs.

# Strategy
Build cache once → anchor once → solve+eval per y-vector with z-buffer reuse.
"""
@inline function cubic_interp(
        x::AbstractVector{Tg},
        s::Series{<:Tuple},
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        # Periodic: use vector path and convert to tuple
        K = n_series(s)
        output = Vector{promote_type(_series_eltype(s), Tq)}(undef, K)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return ntuple(k -> @inbounds(output[k]), Val(K))
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
    return _cubic_oneshot_series_bcpair_ntuple(x, s, xq, bc_pair, extrap, autocache, deriv, searcher)
end

"""
    cubic_interp(x, Series(Y::Matrix), xq; ...) → Vector

One-shot cubic spline interpolation of multiple y-series at a single query point.
Returns `Vector` when `Series` wraps a matrix or vector-of-vectors.
"""
@inline function cubic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    output = Vector{promote_type(_series_eltype(s), Tq)}(undef, K)
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return output
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
    _cubic_oneshot_series_bcpair!(output, x, s, xq, bc_pair, extrap, autocache, deriv, searcher)
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    @assert length(output) == n_series(s) "output length must match number of series"
    searcher = _resolve_search(x, xq, search, hint)
    if _is_periodic_bc(bc)
        _cubic_oneshot_series_periodic!(output, x, s, xq, bc, deriv, autocache, searcher)
        return output
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
    _cubic_oneshot_series_bcpair!(output, x, s, xq, bc_pair, extrap, autocache, deriv, searcher)
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@with_pool pool function cubic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    x = _to_float(x, Tg)
    _validate_series_lengths(s, length(x))
    K = n_series(s)
    @assert length(outputs) == K "outputs length must match number of series"
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xqs, search, nothing)

    # Solve z for each y-vector upfront (need all z for the query loop)
    zs = [similar!(pool, v) for v in vecs]
    if _is_periodic_bc(bc)
        throw(ArgumentError("Vector query with PeriodicBC not yet supported for one-shot Series. Use pre-built interpolant."))
    end
    bc_pair = _normalize_bc(bc, _series_eltype(s))
    cache = _get_cubic_cache(x, bc_pair, autocache)
    for k in 1:K
        _solve_system!(zs[k], cache, vecs[k], bc_pair)
    end

    wrap = extrap isa WrapExtrap
    @inbounds for j in eachindex(xqs)
        aq = _anchor_query(cache.x, xqs[j], Val(:cubic), wrap, searcher)
        for k in 1:K
            outputs[k][j] = _cubic_series_eval_at_anchor(vecs[k], zs[k], aq, deriv, extrap)
        end
    end
    return outputs
end

function cubic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tq <: Real}
    K = n_series(s)
    Tv = promote_type(_series_eltype(s), Tq)
    outputs = [Vector{Tv}(undef, length(xqs)) for _ in 1:K]
    cubic_interp!(outputs, x, s, xqs; bc, extrap, autocache, deriv, search)
    return outputs
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     REAL TYPE PROMOTION WRAPPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@inline function cubic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    bc_promoted = _promote_bc(bc, Tg_float)
    return cubic_interp(x_typed, s, xq; bc = bc_promoted, extrap, autocache, deriv, search, hint)
end

@inline function cubic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    bc_promoted = _promote_bc(bc, Tg_float)
    return cubic_interp!(output, x_typed, s, xq; bc = bc_promoted, extrap, autocache, deriv, search, hint)
end

function cubic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    xqs_typed = _to_float(xqs, Tg_float)
    bc_promoted = _promote_bc(bc, Tg_float)
    return cubic_interp!(outputs, x_typed, s, xqs_typed; bc = bc_promoted, extrap, autocache, deriv, search)
end

function cubic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: Real, Tq <: Real}
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    x_typed = _to_float(x, Tg_float)
    xqs_typed = _to_float(xqs, Tg_float)
    bc_promoted = _promote_bc(bc, Tg_float)
    return cubic_interp(x_typed, s, xqs_typed; bc = bc_promoted, extrap, autocache, deriv, search)
end
