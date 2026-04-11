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
@inline function _oneshot_eval_1d(::PchipInterp, grid, fiber, extrap, q, op, search, hint)
    return pchip_interp(grid, fiber, q; extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(m::CardinalInterp, grid, fiber, extrap, q, op, search, hint)
    return cardinal_interp(grid, fiber, q; tension = m.tension, extrap = extrap, deriv = op, search = search, hint = hint)
end

@inline function _oneshot_eval_1d(::AkimaInterp, grid, fiber, extrap, q, op, search, hint)
    return akima_interp(grid, fiber, q; extrap = extrap, deriv = op, search = search, hint = hint)
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
# `windows` parameter: per the recursion-alignment invariant, `data` and `grids[1]` are
# already aligned at this layer (the top-level call site pre-sliced them). `windows[1]`
# is `1:length(data)` here — passed through for symmetry only.
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
        windows::Tuple{AbstractUnitRange{Int}},
    )
    return _oneshot_eval_1d(
        methods[1], grids[1], data, extraps[1], q_eval[1], ops[1], searches[1], _first_hint(hints)
    )
end

# Recursive case: collapse dim 1 → (M-1)D array, then recurse
#
# `windows` parameter: see hetero_window.jl for how the top-level entry computes per-axis
# windows. Inside this recursion, `windows[d] == 1:size(data, d)` always (the call site
# pre-slices `data` and `grids` so they're aligned with the actual cell-local stencil).
# We thread `windows` through unchanged — the kernel iterates `Base.tail(size(data))`,
# which equals `length.(Base.tail(windows))` by the alignment invariant.
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
        windows::NTuple{M, AbstractUnitRange{Int}},
    ) where {Tr, M}
    remaining_size = Base.tail(size(data))
    result = acquire!(pool, Tr, remaining_size)
    hint_1 = _first_hint(hints)

    # Collapse first dimension: for each fiber along dim 1, one-shot eval
    for idx in CartesianIndices(remaining_size)
        fiber = view(data, :, idx)  # column-major contiguous
        result[idx] = _oneshot_eval_1d(
            first(methods), first(grids), fiber,
            first(extraps), first(q_eval), first(ops), first(searches), hint_1
        )
    end

    # Recurse with remaining dimensions (Tr unchanged — all levels share one buffer type).
    # The result buffer is sized to `remaining_size`, so its windows are 1-based.
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
@inline function _build_windowed_cell(itp, q_eval, search_tuple, hints)
    # Pre-search uses user hints (mutates them to absolute indices, preserving contract).
    indices, _, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple, hints)
    # Per-axis window: cell-local for windowable methods, full axis for global-solve.
    # `map` over heterogeneous tuples is unrolled by the compiler with no closure
    # capture, which is more allocation-robust than `ntuple(d -> ..., Val(N))` for
    # this kind of zipped per-axis dispatch (see hetero_window.jl `_axis_window`).
    windows = map(_axis_window, itp.methods, indices, map(length, itp.grids))
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
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        searches::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    # Tr promotes data eltype with grid + query eltypes → Dual-safe pool buffers for AD.
    # Grid eltype included: when grid is Dual, 1D oneshot returns Dual-typed results
    # that must fit into _collapse_dims intermediate buffers.
    Tr = _output_eltype(Tv, Tg, typeof.(q_eval)...)

    # Persistent-path gate: use `_has_any_windowable_method` (strict superset of
    # `_has_any_local_method`) because `itp.spacings` is pre-computed at
    # construction — no per-call allocation even when the windowable method is
    # Linear/Constant (which would be a net loss in the scalar oneshot path,
    # hence the asymmetry — see hetero_window.jl for the detailed rationale).
    #
    # GridIdx safety gate: `GridIdx(k)` means "evaluate exactly at grid[k]", i.e.
    # the index is ABSOLUTE into the original grid. The windowed path slices the
    # grid to a cell-local range (e.g. `view(grid, 7:10)`), at which point
    # `GridIdx(8)` is out of bounds in the windowed view. The whole cell-local
    # window is semantically pointless for a GridIdx axis (the kernel's search
    # is O(1) direct lookup anyway), so we simply fall through to the full-fiber
    # path whenever ANY query element is a GridIdx. `_has_grididx` is a compile-
    # time type-level check (zero runtime cost on the common no-GridIdx path).
    if _has_any_windowable_method(itp.methods) && !_has_grididx(typeof(query))
        data_local, grids_local, rel_windows = _build_windowed_cell(itp, q_eval, searches, hints)
        # Inner kernel sees the pre-sliced data + grids. Pass `nothing` as hints — the
        # tiny inner search on a 2–6 point fiber is negligible, and this prevents any
        # relative-coordinate hint from leaking back to the user.
        return _collapse_dims(
            Tr, data_local, grids_local, itp.methods, itp.extraps,
            q_eval, ops, searches, nothing, rel_windows,
        )
    end

    # Global-solve fallback (no pre-search, no slicing — bit-for-bit current behavior).
    full_windows = map(Base.OneTo, size(itp.data))
    return _collapse_dims(
        Tr, itp.data, itp.grids, itp.methods, itp.extraps,
        q_eval, ops, searches, hints, full_windows,
    )
end

# PreCompute path: precomputed partials + local kernel eval (O(1) per query)
@inline function _eval_hetero_nd(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:_HeteroPartials},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
        searches::NTuple{N, AbstractSearchPolicy},
        hints,
    ) where {Tg, Tv, N, G, S, M, E, P}
    return _eval_hetero_precomputed(
        itp.data, itp.grids, itp.spacings, itp.methods, itp.extraps,
        query, ops, searches, hints
    )
end

# ========================================
# Callable Interface
# ========================================

# Tuple query form — unified entry for Real and GridIdx (GridIdx <: Real).
# Resolves GridIdx at entry (no-op for plain Real), then routes:
# - NoInterp in methods → _eval_nointerp (pre-slice strategy)
# - Normal → standard hetero eval (GridIdx search short-circuits via dispatch)
@inline function (itp::HeteroInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv = EvalValue(),
        search = itp.searches,
        hint = nothing,
    ) where {Tg, Tv, N}
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    if _has_nointerp_method(typeof(itp.methods))
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
            deriv, itp.extraps, search, hint,
        )
    end
    _validate_nd_domain(itp.grids, resolved, itp.extraps)
    oob_result = _try_fill_oob(resolved, itp.grids, itp.extraps, ops, _zero_ref(itp))
    oob_result !== nothing && return oob_result
    search_tuple = _resolve_search_nd(search, Val(N), resolved)
    return _eval_hetero_nd(itp, resolved, ops, search_tuple, hint)
end

# Vararg form: itp(0.5, 0.3) or itp(0.5, GridIdx(3)) → itp((0.5, ...))
# GridIdx <: Real, so Vararg{Real, N} matches both.
@inline function (itp::HeteroInterpolantND{Tg, Tv, N})(
        q::Vararg{Real, N};
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
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        search_tuple::NTuple{N, AbstractSearchPolicy},
        hints = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)

    # Persistent-path gate — same asymmetric rule as `_eval_hetero_nd` above,
    # plus the same GridIdx safety gate (windowing would alias GridIdx absolute
    # indices into a sliced view).
    if _has_any_windowable_method(itp.methods) && !_has_grididx(typeof(query))
        data_local, grids_local, rel_windows = _build_windowed_cell(itp, q_eval, search_tuple, hints)
        # Cell tuple: pre-sliced data + grids + relative windows + nothing hints
        # (the inner kernel must not see relative-coord hints — see hetero_window.jl invariant 2).
        return (data_local, grids_local, itp.methods, itp.extraps, q_eval, search_tuple, nothing, rel_windows)
    end

    full_windows = map(Base.OneTo, size(itp.data))
    return (itp.data, itp.grids, itp.methods, itp.extraps, q_eval, search_tuple, hints, full_windows)
end

@inline function _eval_at_cell(
        ::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, S, M, E, P}
    data, grids, methods, extraps, q_eval, searches, hints, windows = cell
    # Tr promotes data eltype with query eltypes → Dual-safe pool buffers for AD.
    Tr = _promote_query_eltype(Tv, q_eval)
    return _collapse_dims(Tr, data, grids, methods, extraps, q_eval, ops, searches, hints, windows)
end

# PreCompute: cell stores precomputed cell location (locate-once optimization)
@inline function _locate_cell(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:_HeteroPartials},
        query::Tuple{Vararg{Real, N}},
        search_tuple::NTuple{N, AbstractSearchPolicy},
        hints = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query, itp.grids, itp.extraps)
    indices, Ls, _ = _search_all_intervals(q_eval, itp.grids, itp.spacings, search_tuple, hints)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, itp.spacings, indices, Ls)
    return (itp.data.partials, indices, hs, inv_hs, dLs)
end

@inline function _eval_at_cell(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:_HeteroPartials},
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N, G, S, M, E, P}
    partials, indices, hs, inv_hs, dLs = cell
    return _eval_hetero_nd_cell(partials, indices, hs, inv_hs, dLs, ops, itp.methods)
end

# ========================================
# Required Traits
# ========================================

@inline _zero_ref(itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array}) where {Tg, Tv, N, G, S, M, E, P} =
    @inbounds first(itp.data)
@inline _zero_ref(itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:_HeteroPartials}) where {Tg, Tv, N, G, S, M, E, P} =
    @inbounds itp.data.partials[1]

@inline _deriv_zero_fill(::HeteroInterpolantND, ::NTuple{N, AbstractEvalOp}, ::Val{N}) where {N} = false
