# ========================================
# HeteroInterpolantND — Evaluation
# ========================================
# On-the-fly tensor product via sequential 1D one-shot interpolation.
#
# Algorithm: For query (q₁, q₂, ..., qₙ), collapse dimensions sequentially:
#   1. Along dim 1: for each fiber data[:, i₂, ...], one-shot eval at q₁
#   2. Along dim 2: for each fiber of the intermediate result, eval at q₂
#   3. Continue until scalar result.
#
# Type stability: achieved via recursive Base.tail dispatch (not a loop).

# ========================================
# Hint Peeling Helpers
# ========================================
# hints can be `nothing` (no hints) or a tuple of per-axis hints.
# These helpers allow _collapse_dims to peel hints with Base.tail uniformly.

@inline _first_hint(::Nothing) = nothing
@inline _first_hint(hints::Tuple) = first(hints)
@inline _tail_hints(::Nothing) = nothing
@inline _tail_hints(hints::Tuple) = Base.tail(hints)

# ========================================
# 1D Fiber One-Shot Helpers
# ========================================
# Each method type dispatches to the corresponding 1D one-shot API.
# No intermediate interpolant object is created — direct grid + data + query eval.
# search and hint are forwarded per-axis from the user-specified values.

@inline function _oneshot_eval_1d(m::CubicInterp, grid, fiber, extrap, q, op, search, hint)
    return cubic_interp(grid, fiber, q; bc = m.bc, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(::LinearInterp, grid, fiber, extrap, q, op, search, hint)
    return linear_interp(grid, fiber, q; extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::QuadraticInterp, grid, fiber, extrap, q, op, search, hint)
    return quadratic_interp(grid, fiber, q; bc = m.bc, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::ConstantInterp, grid, fiber, extrap, q, op, search, hint)
    return constant_interp(grid, fiber, q; side = m.side, extrap = extrap, deriv = op, search = search, hint = hint)
end

# Hermite family: local slope methods (PCHIP, Cardinal, Akima)
@inline function _oneshot_eval_1d(m::PchipInterp, grid, fiber, extrap, q, op, search, hint)
    return pchip_interp(grid, fiber, q; bc = m.bc, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::CardinalInterp, grid, fiber, extrap, q, op, search, hint)
    return cardinal_interp(grid, fiber, q; bc = m.bc, tension = m.tension, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::AkimaInterp, grid, fiber, extrap, q, op, search, hint)
    return akima_interp(grid, fiber, q; bc = m.bc, extrap = extrap, deriv = op, search = search, hint = hint)
end

# ========================================
# Sequential Dimension Collapse (Pool-Based)
# ========================================
# Recursive type-stable dispatch: each step removes dim 1 and recurses
# with Base.tail of all tuples. Julia infers concrete types at each level.
# Intermediate arrays are pool-allocated via @with_pool (zero heap alloc after warmup).
#
# The first argument `::Type{Tr}` is the promoted result type, computed once at the
# entry point via `_promote_query_eltype(Tv, q_eval)`. This is the buffer type for
# intermediate results — for plain Float64 queries it typically equals the data
# element type (preserving zero-alloc), while for AD (ForwardDiff.Dual) queries it
# promotes to the Dual-compatible type so the query-dependent intermediate values
# fit into the pool buffer. `Tr` is plumbed unchanged through the recursion.

# Base case: 1D data → one-shot eval final dimension (Tr is carried but unused —
# the 1D scalar eval returns a scalar directly, no buffer needed).
#
# `windows[1]` indexes axis 1 of `data`. For non-fancy callers it's `Base.OneTo`
# (or `UnitRange`) and `view(data, windows[1])` is a zero-cost identity-shaped
# SubArray; for the OnTheFly wrap-aware periodic path it is a `Vector{Int}` of
# wrapped indices that select the cell-local fiber directly out of full data.
@inline function _collapse_dims(
        ::Type,
        data::AbstractVector,
        grids::Tuple{AbstractVector},
        methods::Tuple{AbstractInterpMethod},
        extraps::Tuple{AbstractExtrap},
        q_eval::Tuple{Real},
        ops::Tuple{AbstractEvalOp},
        searches::Tuple{AbstractSearchPolicy},
        hints,
        windows::Tuple{AbstractVector{Int}},
    )
    return _oneshot_eval_1d(
        methods[1], grids[1], view(data, windows[1]),
        extraps[1], q_eval[1], ops[1], searches[1], _first_hint(hints)
    )
end

# Recursive case: collapse dim 1 → (M-1)D array, then recurse.
#
# Generic windows: each `windows[d]` is an `AbstractVector{Int}` selecting which
# entries of axis d of `data` participate. Existing callers pre-slice `data` and
# pass `Base.OneTo` per axis (identity), so this generalization preserves their
# zero-alloc behavior. The new OnTheFly wrap-aware periodic path passes the full
# data plus per-axis index vectors (UnitRange for non-periodic, pool-allocated
# `Vector{Int}` for periodic) — fibers are built by single-level direct indexing
# into `data`, avoiding the nested-SubArray allocation that `view(view(data, vec, vec), :, idx)`
# produces.
@inline @with_pool pool function _collapse_dims(
        ::Type{Tr},
        data::AbstractArray{<:Any, M},
        grids::Tuple{AbstractVector, Vararg{AbstractVector}},
        methods::Tuple{AbstractInterpMethod, Vararg{AbstractInterpMethod}},
        extraps::Tuple{AbstractExtrap, Vararg{AbstractExtrap}},
        q_eval::Tuple{Real, Vararg{Real}},
        ops::Tuple{AbstractEvalOp, Vararg{AbstractEvalOp}},
        searches::Tuple{AbstractSearchPolicy, Vararg{AbstractSearchPolicy}},
        hints,
        windows::NTuple{M, AbstractVector{Int}},
    ) where {Tr, M}
    # Result shape derives from windows (not size(data)) — `data` may be the
    # full array with windows[d] selecting a sub-range per axis.
    remaining_size = ntuple(d -> length(@inbounds windows[d + 1]), Val(M - 1))
    result = acquire!(pool, Tr, remaining_size)
    hint_1 = _first_hint(hints)

    # Build axis-1 fiber by direct indexing: data[windows[1], windows[2][idx[1]], …].
    # Single-level view → no nested-SubArray alloc even when windows[1] is Vector{Int}.
    for idx in CartesianIndices(remaining_size)
        tail_scalars = ntuple(d -> @inbounds(windows[d + 1][idx[d]]), Val(M - 1))
        fiber = view(data, windows[1], tail_scalars...)
        result[idx] = _oneshot_eval_1d(
            first(methods), first(grids), fiber,
            first(extraps), first(q_eval), first(ops), first(searches), hint_1
        )
    end

    # Recurse with regular pool buffer (no fancy windowing on `result`).
    new_windows = map(Base.OneTo, remaining_size)
    return _collapse_dims(
        Tr, result, Base.tail(grids), Base.tail(methods),
        Base.tail(extraps), Base.tail(q_eval), Base.tail(ops),
        Base.tail(searches), _tail_hints(hints), new_windows
    )
end

# ========================================
# Cell-local window builder
# ========================================
#
# Shared helper for the windowing path: pre-searches the query cell, builds
# per-axis windows, and slices `data`/`grids` accordingly. Returns the tuple
# `(data_local, grids_local, rel_windows)` that the windowed `_collapse_dims`
# kernel needs. Used by both `_eval_hetero_nd` (immediate evaluation) and
# `_locate_cell` (vector-calculus cell caching).
#
# `@inline` is load-bearing: both call sites are hot paths, and the helper
# threads concrete types (itp.methods, itp.grids, itp.data) through the
# windowing logic. Inlining lets the compiler specialize on the interpolant's
# concrete type and unroll the per-axis `map` calls into straight-line code.
@inline function _build_windowed_cell(itp, q_eval, extraps, policies, hints, mono)
    # Pre-search via per-axis adaptive function barriers (hint state mutated in-place).
    # Thread `extraps` so an InBounds range axis leans the WINDOW-location search too (bit-identical
    # index). Inner `_collapse_dims` still re-promotes per 1D fiber via `itp.extraps`.
    indices, _, _ = _search_all_intervals(q_eval, itp.grids, policies, hints, mono, extraps)
    # Per-axis window: cell-local for windowable methods, full axis for global-solve.
    # `map` over heterogeneous tuples is unrolled by the compiler with no closure
    # capture, which is more allocation-robust than `ntuple(d -> ..., Val(N))` for
    # this kind of zipped per-axis dispatch (see hetero_window.jl `_axis_window`).
    windows = map(_axis_window, itp.methods, indices, map(_data_length, itp.grids))
    data_local = view(itp.data, windows...)
    grids_local = map(view, itp.grids, windows)
    rel_windows = map(Base.OneTo ∘ length, windows)
    return data_local, grids_local, rel_windows
end

# ========================================
# Core Eval Entry Point
# ========================================

# OnTheFly path: sequential 1D one-shot interpolation per query.
#
# Windowing strategy (Phase 3):
#   - If any axis is a local method (PCHIP/Cardinal/Akima), pre-search the cell once
#     using the user's hint (which mutates them to absolute indices, preserving the
#     existing contract), build per-axis cell-local windows via `_axis_window`, and
#     slice both `data` and `grids` to those windows. The inner `_collapse_dims` then
#     iterates only over `stencil^(N-1)` fibers — O(1) per query, independent of N.
#   - Otherwise (pure global-solve tuples like (CubicInterp, CubicInterp)), skip the
#     pre-search entirely and use full windows. The kernel sees the original data and
#     grids — bit-for-bit identical to the pre-Phase-3 behavior, zero regression.
@inline function _eval_hetero_nd(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array},
        query::Tuple{Vararg{Number, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N, G, M, E, P}
    # `extraps` is the domain-checked, per-axis InBounds-promoted tuple. `_handle_all_extraps`
    # folds Clamp/Wrap/Fill (unchanged by promotion) and no-ops on InBounds/NoExtrap. The inner
    # `_collapse_dims` keeps the ORIGINAL `itp.extraps` so each 1D fiber promotes for itself.
    q_eval = _handle_all_extraps(query, itp.grids, extraps)
    Tr = _promote_eltype(Tv, Tg, typeof.(q_eval)...)

    # Wrap-aware path: routed only when at least one axis is a periodic local
    # Hermite method. Pool scope (and the wrap-aware buffers) live entirely
    # inside `_eval_hetero_nd_wrap_aware` — the NoBC branch below stays
    # pool-free and identical to its pre-Phase-2 behavior.
    if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query))
        return _eval_hetero_nd_wrap_aware(itp, q_eval, Tr, ops, policies, hints, mono)
    end

    if _has_any_windowable_method(itp.methods) && !_has_grididx(typeof(query))
        data_local, grids_local, rel_windows = _build_windowed_cell(itp, q_eval, extraps, policies, hints, mono)
        # Inner kernel uses policies for fiber re-search on sliced grids.
        # Pass `nothing` as hints — tiny inner search on 2–6 point fibers is negligible.
        return _collapse_dims(
            Tr, data_local, grids_local, itp.methods, itp.extraps,
            q_eval, ops, policies, nothing, rel_windows,
        )
    end

    # Global-solve fallback (no pre-search, no slicing).
    # Pass nothing for hints — full-fiber 1D searches are O(n) anyway,
    # hint write-back is negligible and avoids Ref allocation on scalar calls.
    full_windows = map(Base.OneTo, size(itp.data))
    return _collapse_dims(
        Tr, itp.data, itp.grids, itp.methods, itp.extraps,
        q_eval, ops, policies, nothing, full_windows,
    )
end

# Wrap-aware persistent path: same shape as the OnTheFly oneshot wrap-aware
# branch in `_interp_nd_oneshot_onthefly`. Pool scope is local to this function
# so the NoBC `_eval_hetero_nd` branch never enters a `@with_pool` setup.
#
# `mono` is intentionally unused: BC-aware search via `_search_all_axis_intervals`
# is required for `PeriodicBC{:exclusive}` seam queries (`q ≥ x[n]` must return
# `idx_L = n`, not the clamped `n-1` that the non-BC `_search_all_intervals` would
# produce). The stencil search resolves `Searcher{...,<:PeriodicBC{:exclusive}}`
# per axis and handles seam wrap directly; mono-aware LinearBinarySearch hint
# walking still works inside the resolved Searcher when the user opts in.
@inline @with_pool pool function _eval_hetero_nd_wrap_aware(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Number, N}},
        ::Type{Tr},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        ::NTuple{N, Bool},
    ) where {Tg, Tv, N, G, M, E, P, Tr}
    windows, grids_local, methods_inner, extraps_inner =
        _build_wrap_aware_cell_components(pool, itp, q_eval, policies, hints)
    return _collapse_dims(
        Tr, itp.data, grids_local, methods_inner, extraps_inner,
        q_eval, ops, policies, nothing, windows,
    )
end

# Pool-allocated wrap-aware cell builder. Used by `_eval_hetero_nd_wrap_aware`
# (OnTheFly persistent operator) where the cell components are consumed
# inside the same `@with_pool` scope as their construction.
# Vector-calculus paths (gradient/hessian/laplacian), which call
# `_eval_at_cell` many times across separate scopes, instead use the
# heap-allocated counterpart `_build_wrap_aware_cell_heap` below.
@inline function _build_wrap_aware_cell_components(
        pool,
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Number, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, M, E, P}
    intervals, _, _ = _search_all_axis_intervals(q_eval, itp.grids, policies, hints)
    indices = map(first, intervals)
    windows = map((m, x, ix) -> _axis_window_pooled(pool, m, x, ix), itp.methods, itp.grids, indices)
    grids_local = map((m, x, w, ix) -> _axis_grid_pooled(pool, m, x, w, ix), itp.methods, itp.grids, windows, indices)
    methods_inner = map(_strip_periodic_bc, itp.methods)
    extraps_inner = map(_strip_wrap_extrap, itp.extraps, itp.methods)
    return windows, grids_local, methods_inner, extraps_inner
end

# PreCompute path: precomputed partials + local kernel eval (O(1) per query)
@inline function _eval_hetero_nd(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:_HeteroPartials},
        query::Tuple{Vararg{Number, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N, G, M, E, P}
    # Direct ND kernel (no inner 1D fibers) → thread the promoted `extraps` into the ND search.
    return _eval_hetero_precomputed(
        itp.data, itp.grids, itp.methods, extraps,
        query, ops, policies, hints, mono,
    )
end

# Heap-allocated wrap-aware cell builder. Used by the periodic branch in the
# OnTheFly persistent `_locate_cell` for vector-calculus calls (gradient/
# hessian/laplacian), where the cell components must outlive multiple
# `_eval_at_cell` invocations. The OnTheFly oneshot path
# (`_eval_hetero_nd_wrap_aware`) and the persistent `(itp)(query)` operator
# both still use the pool variant (`_build_wrap_aware_cell_components`) since
# their cell lifetime fits cleanly inside a single `@with_pool` scope.
@inline function _build_wrap_aware_cell_heap(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Number, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, M, E, P}
    intervals, _, _ = _search_all_axis_intervals(q_eval, itp.grids, policies, hints)
    indices = map(first, intervals)
    windows = map(_axis_window_heap, itp.methods, itp.grids, indices)
    grids_local = map(_axis_grid_heap, itp.methods, itp.grids, windows, indices)
    methods_inner = map(_strip_periodic_bc, itp.methods)
    extraps_inner = map(_strip_wrap_extrap, itp.extraps, itp.methods)
    return windows, grids_local, methods_inner, extraps_inner
end

@inline _axis_window_heap(m::AbstractInterpMethod, x::AbstractVector, ix::Int) =
    _axis_window(m, ix, _data_length(x))
@inline _axis_window_heap(m::AbstractLocalHermiteInterp{<:PeriodicBC}, x::AbstractVector, ix::Int) =
    _fill_periodic_window!(Vector{Int}(undef, _fixed_window_size(m)), m, ix, _data_length(x))

@inline _axis_grid_heap(::AbstractInterpMethod, x::AbstractVector, w::AbstractVector{Int}, ::Int) =
    view(x, w)
@inline _axis_grid_heap(m::AbstractLocalHermiteInterp{<:PeriodicBC}, x::AbstractVector{Tg}, ::AbstractVector{Int}, ix::Int) where {Tg} =
    _fill_periodic_grid!(Vector{Tg}(undef, _fixed_window_size(m)), m, x, ix)

# ========================================
# Callable Interface
# ========================================

# Tuple query form — unified entry for Real and GridIdx (GridIdx <: Real).
# Resolves GridIdx at entry (no-op for plain Real), then routes:
# - NoInterp in methods → _eval_nointerp (pre-slice strategy)
# - Normal → standard hetero eval (GridIdx search short-circuits via dispatch)
@inline function (itp::HeteroInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Number, N}};
        deriv = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search = itp.searches,
        hint = nothing,
    ) where {Tg, Tv, N}
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    # Call-time extrap override: `nothing` keeps the stored per-axis extraps;
    # `InBounds` opts into the in-domain fast-path (broadcast or per-axis).
    # Resolved via the generic `AbstractInterpolantND` method; other modes throw.
    extraps0 = _resolve_extrap_override_nd(itp, extrap)
    if _has_nointerp_method(typeof(itp.methods))
        # InBounds override intentionally not threaded into the generated `_eval_nointerp`
        # (it reads `itp.extraps`): value-correct, no fast-path — as for the OTF fibers.
        _validate_nointerp_grididx(itp.methods, resolved)
        search_tuple = _resolve_search_nd(search, Val(N))
        return _eval_nointerp(itp, resolved, ops, search_tuple, hint)
    end
    # GridIdx auto-promotion on persistent path — mirrors the scalar one-shot
    # `interp` GridIdx auto-promotion via `_promote_grididx_to_nointerp`. Without
    # this, a GridIdx query on a Hermite persistent interpolant would fall
    # through to the full-fiber path and run ~20-100× slower than the same query
    # via one-shot `interp()`. Promoting the GridIdx axis to NoInterp and
    # re-routing through `_interp_nointerp_oneshot` pre-slices data/grids and
    # delegates to the pool-backed one-shot fast path.
    #
    # Restrictions:
    #   (a) Only for the OnTheFly/AutoCoeffs layout where `itp.data` is a plain
    #       N-dim array. PreCompute interpolants store `_HeteroPartials` (an
    #       (N+1)-dim cache of precomputed slopes/curvatures) which doesn't
    #       satisfy `_interp_nointerp_oneshot`'s array-rank signature — and the
    #       PreCompute solve is already amortized, so there's no speedup to
    #       capture anyway.
    #   (b) Only for EvalValue. Deriv queries (gradient/hessian) fall through to
    #       `_eval_hetero_nd`, where the windowing gate keeps them correct via
    #       the full-fiber fallback. (Option C territory.)
    # The `isa` check folds at compile time because `typeof(itp.data)` is
    # statically known from the concrete interpolant type.
    #
    # IMPLICIT COUPLING: `_interp_nointerp_oneshot` recurses into the public
    # `interp(grids_r, data_r, query_r; method=methods_r, ...)` without
    # passing `coeffs`, relying on `_resolve_coeffs_nd_oneshot(AutoCoeffs(),
    # scalar_query, ...)` to return `OnTheFly()`. That holds today for every
    # scalar one-shot path. If the AutoCoeffs resolver is ever changed to
    # return PreCompute on some scalar case, GridIdx persistent queries will
    # silently start building partials per call (heap alloc + perf regression)
    # — pass `coeffs=OnTheFly()` explicitly here to harden against that.
    if itp.data isa AbstractArray{<:Any, N} &&
            _has_grididx(typeof(resolved)) && _all_eval_value(ops)
        promoted = _promote_grididx_to_nointerp(itp.methods, resolved)
        return _interp_nointerp_oneshot(
            itp.grids, itp.data, resolved, promoted,
            deriv, extraps0, search, hint,
        )
    end
    # Promote each axis query to Tc before validate / fill so the OOB/fill VALUE
    # carries the grid carrier (Dual grid → Dual), matching the OnTheFly collapse.
    # Identity on Float64; Int grids stay Int. (GridIdx branch above returns early.)
    qc = map(_promote_coord, resolved, map(eltype, itp.grids))
    # Validate + per-axis promote (in-domain NoExtrap → InBounds for the search), mirroring the
    # homogeneous ND scalar path. PreCompute threads `extraps_eff` to its ND search; the OnTheFly
    # collapse keeps promoting transitively inside each 1D fiber (see `_locate_cell`).
    extraps_eff = _validate_nd_domain(itp.grids, qc, extraps0)
    oob_result = _try_fill_oob(qc, itp.grids, extraps_eff, ops, _sample_data(itp))
    oob_result !== nothing && return oob_result
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))
    return _eval_hetero_nd(itp, qc, extraps_eff, ops, policies, hints, mono)
end

# Vararg form: itp(0.5, 0.3) or itp(0.5, GridIdx(3)) → itp((0.5, ...))
# GridIdx <: Real, so Vararg{Number, N} matches both.
@inline function (itp::HeteroInterpolantND{Tg, Tv, N})(
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return itp(q; kw...)
end

# ========================================
# _locate_cell / _eval_at_cell Protocol
# ========================================
# Enables vector_calculus.jl functions (gradient, hessian, laplacian).

# OnTheFly: cell stores everything needed for re-collapse (including searches + hints + windows).
#
# Phase 4: when at least one axis is a local method, compute cell-local windows ONCE here
# and cache them in the cell tuple. The vector_calculus path (gradient/hessian/laplacian)
# calls `_eval_at_cell` N or N² times per query — windowing once amortizes the search +
# `_axis_window` cost across all those evaluations. Pure global-solve tuples skip the
# pre-search and store full windows (zero regression).
@inline function _locate_cell(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array},
        query::Tuple{Vararg{Number, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N, G, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, extraps)

    # Periodic wrap-aware path: build the wrap-aware cell ONCE here so the
    # generic `_gradient_generic` / `_hessian_generic` / `_laplacian_generic`
    # protocol (locate-once + N or N(N+1)/2 `_eval_at_cell` calls) flows through
    # unmodified. The `Vector{Int}` / `Vector{Tg}` buffers are heap-allocated
    # (one set per gradient/hessian/laplacian call) — vector-calculus paths
    # don't have a zero-alloc test, so the small heap churn (~200 B per call)
    # is preferable to the multi-helper pool dance the previous design needed.
    # Persistent itp `(itp)(q)` eval (which DOES require zero alloc) goes
    # through `_eval_hetero_nd_wrap_aware` and stays on the pool path.
    if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query))
        windows, grids_local, methods_inner, extraps_inner =
            _build_wrap_aware_cell_heap(itp, q_eval, policies, hints)
        return (itp.data, grids_local, methods_inner, extraps_inner, q_eval, policies, nothing, windows)
    end

    if _has_any_windowable_method(itp.methods) && !_has_grididx(typeof(query))
        data_local, grids_local, rel_windows = _build_windowed_cell(itp, q_eval, extraps, policies, hints, mono)
        # Inner kernel uses policies for fiber re-search on sliced grids.
        # Pass user-facing `itp.extraps` here (not InBounds-promoted `extraps`):
        # the recursive 1D collapse runs its own per-axis `_check_domain` and
        # benefits from the 1D-level InBounds promotion — promotion happens
        # naturally inside each 1D oneshot call.
        return (data_local, grids_local, itp.methods, itp.extraps, q_eval, policies, nothing, rel_windows)
    end

    # Full-fiber fallback: pass hints for persistent hint update in per-fiber 1D search.
    # (Unlike _eval_hetero_nd which passes nothing — that's the scalar path where
    # auto-created Refs are throwaway. Here in _locate_cell, batch callers provide
    # persistent Refs via _ensure_hint_nd.)
    full_windows = map(Base.OneTo, size(itp.data))
    return (itp.data, itp.grids, itp.methods, itp.extraps, q_eval, policies, hints, full_windows)
end

@inline function _eval_at_cell(
        ::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, M, E, P}
    data, grids, methods, extraps, q_eval, searches, hints, windows = cell
    # Tr promotes data eltype with grid + query eltypes → Dual-safe pool buffers for AD.
    Tr = _promote_eltype(Tv, Tg, typeof.(q_eval)...)
    return _collapse_dims(Tr, data, grids, methods, extraps, q_eval, ops, searches, hints, windows)
end

# PreCompute: cell stores precomputed cell location (locate-once optimization)
@inline function _locate_cell(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:_HeteroPartials},
        query::Tuple{Vararg{Number, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N, G, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, extraps)
    # 6-arg search: per-axis `extraps` let an InBounds range axis take the lean direct
    # search. Per-axis dispatch (no 1D-style shared core), so ExtendExtrap axes clamp.
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, policies, hints, mono, extraps)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, itp.grids, indices, Ls)
    return (itp.data.partials, indices, hs, inv_hs, dLs)
end

@inline function _eval_at_cell(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:_HeteroPartials},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, M, E, P}
    partials, indices, hs, inv_hs, dLs = cell
    return _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, itp.methods)
end

# ========================================
# Required Traits
# ========================================

@inline _sample_data(itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:Array}) where {Tg, Tv, N, G, M, E, P} =
    @inbounds first(itp.data)
@inline _sample_data(itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, <:_HeteroPartials}) where {Tg, Tv, N, G, M, E, P} =
    @inbounds itp.data.partials[1]
