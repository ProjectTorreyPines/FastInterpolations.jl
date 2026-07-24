# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               QUADRATIC ONE-SHOT SERIES INTERPOLATION                    ║
# ║     Search once → anchor once → loop kernel per y-vector (Strategy B)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → quadratic_anchor.jl → quadratic_oneshot_series.jl → ...
# Shared anchor eval: _quadratic_eval_at_anchor(y, a, d, aq, op, extrap) in quadratic_anchor.jl

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         SCALAR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── Scalar Series → Vector return (consistent with SeriesInterpolant) ───────

@inline @with_pool pool function quadratic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Number}
    _validate_series_lengths(s, length(x))
    # Pool-backed cache: K-series loop reuses h/inv_h across all K calls.
    x = _cache_axis_pooled(pool, x, _promote_grid_float(Tg, _series_eltype(s)))
    _check_domain(x, xq, extrap)
    nx = length(x)
    K = n_series(s)
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xq, search, hint)
    A_lean = _quadratic_series_anchor_type(extrap, x, _coord_eltype(Tq, eltype(x)))
    anchor = _build_series_anchor(QuadraticInterp(), A_lean, x, xq, extrap, extrap isa WrapExtrap, searcher)
    Tv_out = _value_type(_series_eltype(s), Tg)
    Tg_actual = eltype(x)
    Tcoeff = _promote_eltype(_coeff_op, Tg_actual, _series_eltype(s))
    output = Vector{_promote_eltype(_interp_op, Tg_actual, _series_eltype(s), Tq)}(undef, K)
    d = acquire!(pool, Tcoeff, nx)
    a = acquire!(pool, Tcoeff, nx - 1)
    y_buf = acquire!(pool, Tv_out, nx)
    @inbounds for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(d, a, x, y_buf, bc_promoted)
        output[k] = _quadratic_series_eval(y_buf, a, d, anchor, deriv, extrap)
    end
    return output
end

# ─── In-place scalar ─────────────────────────────────────────────────────────

@inline @with_pool pool function quadratic_interp!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        s::Series,
        xq::Tq;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tq <: Number}
    _validate_series_lengths(s, length(x))
    length(output) == n_series(s) || _throw_series_dim_mismatch(length(output), n_series(s))
    # Pool-backed cache: K-series loop reuses h/inv_h.
    x = _cache_axis_pooled(pool, x, _promote_grid_float(Tg, _series_eltype(s)))
    _check_domain(x, xq, extrap)
    nx = length(x)
    vecs = _series_vectors(s)
    searcher = _resolve_search(x, xq, search, hint)
    A_lean = _quadratic_series_anchor_type(extrap, x, _coord_eltype(Tq, eltype(x)))
    anchor = _build_series_anchor(QuadraticInterp(), A_lean, x, xq, extrap, extrap isa WrapExtrap, searcher)
    Tv_out = _value_type(_series_eltype(s), Tg)
    Tg_actual = eltype(x)
    Tcoeff = _promote_eltype(_coeff_op, Tg_actual, _series_eltype(s))
    d = acquire!(pool, Tcoeff, nx)
    a = acquire!(pool, Tcoeff, nx - 1)
    y_buf = acquire!(pool, Tv_out, nx)
    @inbounds for k in eachindex(output)
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(d, a, x, y_buf, bc_promoted)
        output[k] = _quadratic_series_eval(y_buf, a, d, anchor, deriv, extrap)
    end
    return output
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         VECTOR ONE-SHOT API                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Series-outer loop: pre-compute anchors once, then solve+eval per series.
# O(n + Q) memory (single d/a buffer + anchor vector) — search done once for all series.
@with_pool pool function quadratic_interp!(
        outputs::AbstractVector{<:AbstractVector},
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Number}
    _validate_series_lengths(s, length(x))
    # Pool-backed cache: K-series × Q-query loop reuses h/inv_h.
    x = _cache_axis_pooled(pool, x, _promote_grid_float(Tg, _series_eltype(s)))
    K = n_series(s)
    _validate_series_outputs(outputs, K, length(xqs))
    # Domain check: NoExtrap → throws if OOB, returns InBounds(); others → pass-through
    extrap_eff = _check_domain(x, xqs, extrap)
    nx = length(x)
    vecs = _series_vectors(s)
    Tv_out = _value_type(_series_eltype(s), Tg)
    Tg_actual = eltype(x)
    Tcoeff = _promote_eltype(_coeff_op, Tg_actual, _series_eltype(s))

    # Pre-compute lean anchors once (search Q times, not K×Q; op-independent)
    A_lean = _quadratic_series_anchor_type(extrap_eff, x, _coord_eltype(Tq, Tg_actual))
    anchors = acquire!(pool, A_lean, length(xqs))
    _fill_series_anchors_resolved!(QuadraticInterp(), anchors, x, xqs, extrap_eff, extrap_eff isa WrapExtrap, search, nothing)

    d = acquire!(pool, Tcoeff, nx)
    a = acquire!(pool, Tcoeff, nx - 1)
    y_buf = acquire!(pool, Tv_out, nx)
    @inbounds for k in 1:K
        copyto!(y_buf, 1, vecs[k], 1, nx)
        bc_promoted = _normalize_bc(bc, first(y_buf))
        _compute_quadratic_coeffs!(d, a, x, y_buf, bc_promoted)
        for j in eachindex(xqs)
            outputs[k][j] = _quadratic_series_eval(vecs[k], a, d, anchors[j], deriv, extrap_eff)
        end
    end
    return outputs
end

function quadratic_interp(
        x::AbstractVector{Tg},
        s::Series,
        xqs::AbstractVector{Tq};
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tq <: Number}
    K = n_series(s)
    Tg_float = _promote_grid_float(Tg, _series_eltype(s))
    Tv_out = _promote_eltype(_interp_op, Tg_float, _series_eltype(s), Tq)
    outputs = _alloc_series_batch_outputs(Tv_out, K, length(xqs))
    quadratic_interp!(outputs, x, s, xqs; bc, extrap, deriv, search)
    return outputs
end
