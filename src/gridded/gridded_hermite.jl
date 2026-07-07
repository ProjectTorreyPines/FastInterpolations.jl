# ============================================================================
# GriddedQuery — local Hermite (PCHIP / Cardinal / Akima) separable evaluation
# ============================================================================
#
# The contract is the method's own point-wise ND semantics: `_collapse_dims`
# (sequential 1D one-shots per fiber, raw grids, no mixed partials). Three
# consequences shape this file:
#
#   * FIXED pass order 1..N — the slope estimators are nonlinear (PCHIP
#     sign-clamp, Akima abs-weights), so per-axis interpolation operators do
#     not commute; axis-1-first is exactly `_collapse_dims`' collapse order.
#     (No cost-based reordering, unlike the linear fullbuffer.)
#   * RAW grids + per-pass width type — point-wise hands raw grids to the 1D
#     one-shots; pass d runs at `Tw = _promote_grid_float(eltype(grid_d),
#     eltype(src))` where `src` is that pass's input (data for pass 1, the
#     `Tmid` intermediate after — the same eltype point-wise's later collapses
#     see in their pool buffers).
#   * Clamp = coordinate fold, kernel still evaluates — the point-wise ND
#     surface clamps each query coordinate into the domain BEFORE the collapse
#     (unlike the 1D surface's `_eval_extrapolation` early return), so a
#     Clamp-OOB derivative is the boundary cell polynomial's slope, not zero.
#     The anchor's `dL ∈ [0, h]` fold reproduces this for every op.
#
# Slopes are computed IN-PASS from the full source fiber (2 per output
# element via `_local_slope`); full-axis node/boundary formulas are equivalent
# to the point-wise `_axis_window` slices by the validated window-trait
# contract. Only a fullbuffer strategy exists — a "fused" Hermite would redo
# the whole windowed collapse per output (≈ point-wise minus the search).

# Method → slope-method tag (NoBC only on this path; the hook constrains BC).
@inline _gridded_slope_method(::PchipInterp) = PchipSlopes()
@inline _gridded_slope_method(m::CardinalInterp) = CardinalSlopes(m.tension)
@inline _gridded_slope_method(::AkimaInterp) = AkimaSlopes()

# One separable pass: resolve axis `D` through the Hermite combine, pass every
# other axis through (`ranges[e]` bounds; `ranges[D]` ignored). Loop nest and
# static-dispatch mirror `_gridded_pass!` — axis 1 innermost (stride-1), the
# axis-D anchor destructured at its own nesting level so the basis math is
# loop-invariant for D ≥ 2 passes. The fiber view (all indices fixed but axis
# D) feeds `_local_slope` exactly like point-wise's per-fiber 1D one-shot.
@generated function _hermite_gridded_pass!(
        dest::AbstractArray{<:Any, N},
        src::AbstractArray{Ts, N},
        anchors::Vector{<:_AxisAnchor},
        m::AbstractLocalHermiteInterp,
        grid::AbstractVector{Tgr},
        op::AbstractEvalOp,
        ::Val{D},
        ranges::NTuple{N, UnitRange{Int}}
    ) where {Ts, Tgr, N, D}
    js = [Symbol(:j_, d) for d in 1:N]
    fib_idx = Any[js[d] for d in 1:N]
    fib_idx[D] = :(Colon())
    lo = Any[js[d] for d in 1:N]
    lo[D] = :il
    hi = copy(lo)
    hi[D] = :(il + 1)
    body = quote
        fib = view(src, $(fib_idx...))
        dyL = _local_slope(Tw, sm, grid, fib, il, n)
        dyR = _local_slope(Tw, sm, grid, fib, il + 1, n)
        dest[$(js...)] = _hermite_kernel_1d(op, src[$(lo...)], src[$(hi...)], dyL, dyR, h_a, invh_a, dL_a)
    end
    for d in 1:N   # d = 1 built first → innermost loop
        body = if d == D
            quote
                for $(js[d]) in eachindex(anchors)
                    a = anchors[$(js[d])]
                    il = a.idx
                    dL_a, h_a, invh_a = a.payload
                    $body
                end
            end
        else
            quote
                for $(js[d]) in ranges[$d]
                    $body
                end
            end
        end
    end
    return quote
        Tw = _promote_grid_float(Tgr, Ts)
        sm = _gridded_slope_method(m)
        n = _data_length(grid)
        @inbounds $body
        return dest
    end
end

# Static dispatch over the runtime axis id (same rationale as
# `_gridded_pass_dim!`): a generated if-chain turns the pass call into N
# static call sites with literal anchors/methods/grids/ops selections.
@generated function _hermite_pass_dim!(
        dest::AbstractArray{<:Any, N},
        src::AbstractArray{<:Any, N},
        anchors::Tuple{Vararg{Vector, N}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        d::Int,
        ranges::NTuple{N, UnitRange{Int}}
    ) where {N}
    ex = :(throw(ArgumentError("pass axis out of range")))
    for k in N:-1:1
        ex = :(
            d == $k ?
                _hermite_gridded_pass!(dest, src, anchors[$k], methods[$k], grids[$k], ops[$k], Val($k), ranges) : $ex
        )
    end
    return ex
end

# Multi-pass core through pooled `Tmid` intermediates, FIXED order 1..N (see
# header). Pass-through bounds: unresolved axes keep their full source extent
# (no tap-hull tightening for the k-tap window — a future zoom optimization),
# resolved axes their full target extent.
@with_pool pool function _hermite_gridded_fullbuffer!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        anchors::Tuple{Vararg{Vector, N}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        ::Type{Tmid}
    ) where {N, Tmid}
    out_size = map(length, anchors)
    cur_size = size(data)
    ranges = map(n -> 1:n, cur_size)
    src = data
    for d in 1:N
        cur_size = Base.setindex(cur_size, out_size[d], d)
        dest = d == N ? out : acquire!(pool, Tmid, cur_size...)
        _hermite_pass_dim!(dest, src, anchors, methods, grids, ops, d, ranges)
        ranges = Base.setindex(ranges, 1:out_size[d], d)
        src = dest
    end
    return out
end

# ---- core ---------------------------------------------------------------------
# Anchors build (firing NoExtrap's O(ΣM_d) validation) BEFORE any output write.
# Pass-1 anchors resolve against `Tv` (raw data), later axes against `Tmid`
# (the intermediate those passes read) — matching each point-wise collapse
# stage's fiber eltype.
@with_pool pool function _hermite_gridded_eval!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{Vararg{AbstractLocalHermiteInterp, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tmid}
    ) where {Tv, N, Tmid}
    out_size = map(length, targets)
    size(out) == out_size || throw(
        DimensionMismatch("output size $(size(out)) != query size $out_size")
    )
    anchors = _axis_anchors_all(pool, methods, grids, targets, extraps, Tv, Tmid, Val(N))
    if !any(iszero, out_size)
        _hermite_gridded_fullbuffer!(out, grids, data, anchors, methods, ops, Tmid)
    end
    return _gridded_fill_oob!(out, grids, data, targets, extraps, ops)
end

# ---- gridded method dispatch (local-Hermite separable arm) --------------------
# All axes local-Hermite with NoBC (mixed flavors compose per axis; PeriodicBC
# slope stencils need the wrap-aware machinery and fall through to point-wise).
# `Tmid` mirrors point-wise `_promote_query_eltype(Tv, q_eval)` — the pool
# buffer type of the collapse intermediates. This one prepared arm is shared by
# one-shot `interp(..., gq; method = PchipInterp())` and persistent `itp(gq)`.
@inline function _gridded_eval_methods!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{AbstractLocalHermiteInterp{NoBC}, Vararg{AbstractLocalHermiteInterp{NoBC}}},
        ops,
        extraps
    ) where {Tv, N}
    Tmid = promote_type(Tv, promote_type(map(eltype, targets)...))
    _hermite_gridded_eval!(out, grids, data, targets, methods, ops, extraps, Tmid)
    return true
end

# ---- unified `interp` fast-path -------------------------------------------------
@inline function _try_gridded_oneshot_methods!(
        out_nd::AbstractArray{<:Any, N},
        grids,
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{AbstractLocalHermiteInterp{NoBC}, Vararg{AbstractLocalHermiteInterp{NoBC}}},
        extrap,
        deriv
    ) where {Tv, N}
    _validate_nd_grids(grids, data)
    bcs = map(m -> m.bc, methods)
    ops = _resolve_deriv_nd(deriv, Val(N))
    extraps = _resolve_extrap(extrap, bcs, Val(N), Tv)
    return _gridded_eval_methods!(out_nd, grids, data, targets, methods, ops, extraps)
end
