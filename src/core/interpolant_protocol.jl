# ========================================
# Shared Interpolant Callable Interface (1D + ND)
# ========================================
#
# All interpolant callables defined once on AbstractInterpolant1D / AbstractInterpolantND.
# Per-type files only need trait implementations.
#
# ── 1D Subtypes must implement:
#   _itp_eval_scalar(itp, xq, extrap, op, searcher)   — core scalar evaluation
#   _itp_vector_loop!(out, itp, xq, extrap, op, searcher) — core vector loop
#
# ── 1D Subtypes may override (defaults work for Linear/Quadratic/Constant):
#   _itp_grid(itp)    — grid vector (default: itp.x, Cubic overrides to itp.cache.x)
#   _itp_extrap(itp)  — extrap mode (default: itp.extrap)
#   _itp_search(itp)  — search policy (default: itp.search_policy)

# ╔══════════════════════════════════════╗
# ║       1D Interpolant Protocol        ║
# ╚══════════════════════════════════════╝

# ── Trait Defaults ──

@inline _itp_grid(itp::AbstractInterpolant1D) = itp.x
@inline _itp_extrap(itp::AbstractInterpolant1D) = itp.extrap
@inline _itp_search(itp::AbstractInterpolant1D) = itp.search_policy

# ── N=1 tuple-grid collapse: kwarg unwrapping ──
# A 1-axis ND constructor `foo_interp((x,), y; ...)` and the 1D tuple-query shims
# below both forward to the genuine 1D path. Per-axis kwargs arrive ND-style as
# 1-tuples (`extrap=(WrapExtrap(),)`, `deriv=(DerivOp(1),)`); unwrap them to the
# scalar the 1D signature expects. Non-tuple values (`store`, `tension`, a scalar
# `extrap`) pass through untouched.
@inline _unwrap_axis1(x::Tuple{Any}) = x[1]
@inline _unwrap_axis1(x) = x
@inline _unwrap_nd_kwargs(nt::NamedTuple) = map(_unwrap_axis1, nt)

# ── Call-time extrap override (singular: 1D / ND per-axis) ──
# The stored extrap is the construction-time contract; the only per-call override
# is `InBounds` (an in-domain fast-path ASSERTION, not an extrapolation contract —
# it never changes OOB behavior, since the caller guarantees no OOB query). `nothing`
# (the omitted default) keeps the stored contract. Any other extrapolation mode would
# silently change OOB semantics, so it errors. The callables annotate the kwarg
# `Union{Nothing, AbstractExtrap[, Tuple]}`, so scalar junk (`extrap=5`) is cut at
# the call boundary (`TypeError`) — only a valid-but-disallowed mode reaches the
# friendly throw. Dispatch on the override type folds away at specialization.
@inline _resolve_extrap_override(stored, ::Nothing) = stored
@inline _resolve_extrap_override(_stored, over::InBounds) = over
@noinline _resolve_extrap_override(stored, over::AbstractExtrap) =
    _throw_extrap_not_overridable(stored, over)
# Per-axis tuple elements reuse this resolver (via the `map` below). The tuple kwarg
# annotation cannot check element types, so a junk element (e.g. `(5, nothing)`) is
# cut here as a `TypeError` — matching the scalar boundary rather than surfacing an
# internal `MethodError`.
@noinline _resolve_extrap_override(_stored, over) =
    throw(TypeError(:extrap, "per-axis override tuple element", Union{Nothing, AbstractExtrap}, over))

@noinline _throw_extrap_not_overridable(stored, over) = throw(
    ArgumentError(
        "call-time `extrap` override must be `InBounds(...)` (an in-domain fast-path); got $(over). " *
            "The stored extrapolation mode ($(stored)) is the interpolant's contract and cannot be " *
            "changed per call — rebuild the interpolant to use a different mode."
    )
)

# ── ND per-axis resolution (plural: broadcast / tuple) ──
# Resolve a call-time override against the stored per-axis extraps tuple:
#   nothing  → stored (all axes unchanged)
#   InBounds → broadcast to every axis (fast-path on all axes)
#   Tuple    → per-axis, each element `nothing` (keep stored) or `InBounds`
#             (a disallowed element / single non-InBounds mode routes to the throw)
# The itp-level entry resolves against the interpolant's stored per-axis `extraps`;
# every `AbstractInterpolantND` (tensor-product and Hetero alike) shares this method.
@inline _resolve_extrap_override_nd(itp::AbstractInterpolantND, over) = _resolve_extrap_override_nd(itp.extraps, over)
@inline _resolve_extrap_override_nd(stored::NTuple{N, AbstractExtrap}, ::Nothing) where {N} = stored
@inline _resolve_extrap_override_nd(stored::NTuple{N, AbstractExtrap}, over::InBounds) where {N} =
    ntuple(_ -> over, Val(N))
@inline function _resolve_extrap_override_nd(stored::NTuple{N, AbstractExtrap}, over::Tuple) where {N}
    length(over) == N || _throw_extrap_ndims(N, length(over))
    return map(_resolve_extrap_override, stored, over)
end
@noinline _resolve_extrap_override_nd(stored::NTuple{N, AbstractExtrap}, over::AbstractExtrap) where {N} =
    _throw_extrap_not_overridable(stored, over)   # single non-InBounds mode never broadcasts

@noinline _throw_extrap_ndims(want::Int, got::Int) = throw(
    ArgumentError("call-time `extrap` tuple must have $want elements to match grid dimensions, got $got")
)

# ── Output eltype trait ──
# Default: arithmetic kernels (Linear/Cubic/Quadratic) divide by `h`, so route
# through the shared `_interp_op` for Julia inference. Selection
# kernels (Constant) and Hermite-family (with `dy`) override this default.
@inline _promote_eltype(::AbstractInterpolant1D{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq} =
    _promote_eltype(_interp_op, Tg, Tv, Tq)

# ========================================
# 1D Scalar Call — Hot Path
# ========================================
# @inline for broadcast fusion. `xq::Number` is the numeric-coordinate admission
# boundary (Real, Unitful, Dual for AD); GriddedQuery/anchored queries carry their
# own more-specific callables. @boundscheck normalized: always here, not per-type eval.

@inline function (itp::AbstractInterpolant1D{Tg, Tv})(
        xq::Number;
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv}
    grid = _itp_grid(itp)
    # `nothing` → stored extrap; `InBounds` → per-call fast-path; else errors.
    extrap_eff = _resolve_extrap_override(_itp_extrap(itp), extrap)
    # Domain check is delegated to the per-method `_eval_*_at_point` callable
    # below — NoExtrap throws via that path's `@boundscheck _check_domain`,
    # while Clamp/Fill/Wrap handle OOB inside their specialized eval methods
    # without ever calling `_check_domain`. Outer redundancy removed.
    searcher = _resolve_search(grid, xq, search, hint)
    return _itp_eval_scalar(itp, xq, extrap_eff, deriv, searcher)
end

# ========================================
# 1D Vector Call — Allocating
# ========================================
# Buffer eltype comes from the `_promote_eltype(itp, Tq)` trait — generic for
# arithmetic kernels via `_interp_op` (Int→Float upgrade) and
# `_select_op` for selection kernels (Constant). Duck
# `SVector × Dual` resolves via the trait's `Base.promote_op` fallback.

# ── Deriv-aware buffer eltype (3-arg sibling of the 2-arg trait) ──
# Value eltype from the per-method 2-arg trait (Constant/Hermite overrides honored),
# then scaled into value/coordᴺ space — else a unit-grid deriv batch sizes in value
# units and the store throws. A lone DerivOp broadcasts to all axes; a Tuple is per-axis.
@inline _promote_eltype(itp::AbstractInterpolant1D{Tg}, ::Type{Tq}, op::DerivOp) where {Tg, Tq} =
    _deriv_eltype(_promote_eltype(itp, Tq), Tg, op)
@inline _promote_eltype(itp::AbstractInterpolantND{Tg, Tv, N}, ::Type{Tq}, op::DerivOp) where {Tg, Tv, N, Tq} =
    _promote_eltype(itp, Tq, ntuple(_ -> op, Val(N)))
@inline _promote_eltype(itp::AbstractInterpolantND, ::Type{Tq}, ops::Tuple) where {Tq} =
    _deriv_eltype_nd(_promote_eltype(itp, Tq), itp.grids, ops)

# Fold the `_deriv1_op` witness one order at a time (never `h^-N` — type-unstable for
# units); `DerivOp{0}` is the identity. ND folds the single-axis helper across axes.
@inline _deriv_eltype(::Type{Tr}, ::Type{Tg}, ::DerivOp{0}) where {Tr, Tg} = Tr
@inline _deriv_eltype(::Type{Tr}, ::Type{Tg}, ::DerivOp{N}) where {Tr, Tg, N} =
    _deriv_eltype(_promote_eltype(_deriv1_op, Tr, Tg), Tg, DerivOp{N - 1}())
@inline _deriv_eltype_nd(::Type{Tr}, ::Tuple{}, ::Tuple{}) where {Tr} = Tr
@inline _deriv_eltype_nd(::Type{Tr}, grids::Tuple, ops::Tuple) where {Tr} =
    _deriv_eltype_nd(_deriv_eltype(Tr, eltype(first(grids)), first(ops)), Base.tail(grids), Base.tail(ops))

function (itp::AbstractInterpolant1D{Tg, Tv})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Number}
    grid = _itp_grid(itp)
    extrap_eff = _resolve_extrap_override(_itp_extrap(itp), extrap)
    output = Vector{_promote_eltype(itp, Tq, deriv)}(undef, length(xq))
    searcher = _resolve_search(grid, xq, search, hint)
    _itp_vector_loop!(output, itp, xq, extrap_eff, deriv, searcher)
    return output
end

# ========================================
# 1D Vector Call — In-Place
# ========================================

function (itp::AbstractInterpolant1D{Tg, Tv})(
        output::AbstractVector,
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Number}
    @assert length(output) == length(xq) "output length must match xq length"
    grid = _itp_grid(itp)
    extrap_eff = _resolve_extrap_override(_itp_extrap(itp), extrap)
    searcher = _resolve_search(grid, xq, search, hint)
    _itp_vector_loop!(output, itp, xq, extrap_eff, deriv, searcher)
    return output
end

# ========================================
# 1D Shaped Array Call — Allocating + In-Place
# ========================================
# A real array query (Matrix, 3-D, noncontiguous view) yields an array of the same
# shape; the `AbstractVector` methods above stay more specific, so the vector hot path
# is unchanged. Both reuse the extrap/searcher resolution and the (array-widened) family
# loop, which fills by `eachindex(xq, output)` — column-major for same-shape arrays.

function (itp::AbstractInterpolant1D{Tg, Tv})(
        q::AbstractArray{Tq};
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Number}
    grid = _itp_grid(itp)
    extrap_eff = _resolve_extrap_override(_itp_extrap(itp), extrap)
    output = _alloc_query_output(_promote_eltype(itp, Tq, deriv), q)
    searcher = _resolve_search(grid, q, search, hint)
    _itp_vector_loop!(output, itp, q, extrap_eff, deriv, searcher)
    return output
end

function (itp::AbstractInterpolant1D{Tg, Tv})(
        output::AbstractArray,
        q::AbstractArray{Tq};
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Number}
    _check_query_output_size(output, q)
    grid = _itp_grid(itp)
    extrap_eff = _resolve_extrap_override(_itp_extrap(itp), extrap)
    searcher = _resolve_search(grid, q, search, hint)
    _itp_vector_loop!(output, itp, q, extrap_eff, deriv, searcher)
    return output
end

# ── ND-style tuple queries on a 1D interpolant ──
# A 1-axis grid collapses to the 1D path (see the `*_interp((x,), y)` forwarders),
# so generic tensor code that issues tuple queries stays transparent: `itp((x,))`
# (scalar), `itp(out, (xv,))` / `itp((xv,))` (SoA batch). Per-axis kwargs given as
# 1-tuples (`extrap=(WrapExtrap(),)`, `deriv=(DerivOp(1),)`) unwrap to scalar. Each
# just strips the 1-tuple and forwards to the positional 1D form above.
@inline (itp::AbstractInterpolant1D)(q::Tuple{Real}; kwargs...) =
    itp(q[1]; _unwrap_nd_kwargs(values(kwargs))...)
@inline (itp::AbstractInterpolant1D)(output::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    itp(output, q[1]; _unwrap_nd_kwargs(values(kwargs))...)
@inline (itp::AbstractInterpolant1D)(q::Tuple{AbstractVector}; kwargs...) =
    itp(q[1]; _unwrap_nd_kwargs(values(kwargs))...)
# Shaped single-axis SoA: `(q_matrix,)` collapses to the shaped 1D path (vector
# forms above stay more specific), so generic tensor code that issues `(q,)` is
# transparent for array queries too.
@inline (itp::AbstractInterpolant1D)(output::AbstractArray, q::Tuple{AbstractArray}; kwargs...) =
    itp(output, q[1]; _unwrap_nd_kwargs(values(kwargs))...)
@inline (itp::AbstractInterpolant1D)(q::Tuple{AbstractArray}; kwargs...) =
    itp(q[1]; _unwrap_nd_kwargs(values(kwargs))...)

# ╔══════════════════════════════════════╗
# ║       ND Interpolant Protocol        ║
# ╚══════════════════════════════════════╝

# ── Output eltype trait (ND) — mirrors the 1D version. ──
@inline _promote_eltype(::AbstractInterpolantND{Tg, Tv, N}, ::Type{Tq}) where {Tg, Tv, N, Tq} =
    _promote_eltype(_interp_op, Tg, Tv, Tq)

# ========================================
# Unified Batch Interpolant Evaluation (Generic ND)
# ========================================
#
# Single batch loop for all AbstractInterpolantND subtypes.
# Query extraction dispatches via _query_extract on query container type.

@inline function _interp_nd_batch!(
        output::AbstractArray,
        itp::AbstractInterpolantND{Tg, Tv, N},
        queries,
        extraps_eff::Tuple{Vararg{AbstractExtrap, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    zref = _sample_data(itp)
    @inbounds for k in 1:_query_length(queries)
        query_k = _extract_query_point(queries, k, Val(N))
        # `extraps_eff` carries per-axis `InBounds()` from `_validate_nd_domain`
        # when all batch queries on that axis are in-bounds (1D `_check_domain`
        # union-split per axis via the heterogeneous `map`). `_try_fill_oob` and
        # `_locate_cell` compile away the wrap/clamp/fill per-query branches on
        # the InBounds axes — see `_handle_axis_extrap(::InBounds) = q` and
        # `_is_fill_oob`'s @generated `fill_dims` filtering.
        oob_val = _try_fill_oob(query_k, itp.grids, extraps_eff, ops, zref)
        if oob_val !== nothing
            output[k] = oob_val
            continue
        end
        cell = _locate_cell(itp, query_k, extraps_eff, policies, hints, mono)
        output[k] = _eval_at_cell(itp, cell, ops)
    end
    return output
end

# Scalar-path forwarder: vector_calculus and per-method scalar paths call
# the 5-arg form; we inject `itp.extraps` so the canonical 6-arg
# `_locate_cell` works uniformly. The 6-arg form is the source of truth —
# batch callers (`_interp_nd_batch!`) and the windowed/hetero paths can pass
# a different `extraps_eff` (e.g., InBounds-promoted) without affecting
# scalar callers. Pure `getfield` body — Julia elides this trivially.
@inline _locate_cell(itp::AbstractInterpolantND, q, policies, hints, mono) =
    _locate_cell(itp, q, itp.extraps, policies, hints, mono)

# ========================================
# Unified Scalar Interpolant Evaluation (Generic ND)
# ========================================
#
# Single scalar entry point for AbstractInterpolantND subtypes whose eval
# structure matches `validate → try_fill_oob → locate → eval` (Cubic /
# Linear / Constant / Quadratic). The per-family callables delegate to
# `_eval_nd_scalar_query` (below), which resolves search/hints/ops then calls here.
#
# Hetero is *not* routed through here — its callable has GridIdx/NoInterp
# branches and `_eval_hetero_nd` uses a non-`_locate_cell` path (recursive
# `_collapse_dims` / `_eval_hetero_precomputed`).
@inline function _eval_nd_at_point(
        itp::AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Number, N}},
        ops::NTuple{N, AbstractEvalOp},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
    ) where {Tg, Tv, N}
    # Promote each axis query to its coordinate type Tc ONCE at the eval surface,
    # before validate / try_fill_oob. This (a) makes the OOB/fill VALUE carry the
    # coordinate carrier (Dual grid → Dual fill, via _promote_extrap_val) so the
    # public return is concrete, and (b) runs the OOB classification on the widened
    # type (Float grid + Int query → Float). Identity on the Float64 hot path; Int
    # grids stay Int. Search remains primal-safe (_oob_state / search extract primal).
    qc = map(_promote_coord, query, map(eltype, itp.grids))
    # Validate (NoExtrap throw must precede the FillExtrap short-circuit) AND promote per axis:
    # an in-domain NoExtrap axis becomes InBounds for the search (lean), mirroring the batch path.
    # `extraps` carries any call-time InBounds override; an InBounds axis skips the domain scan.
    extraps_eff = _validate_nd_domain(itp.grids, qc, extraps)
    oob_result = _try_fill_oob(qc, itp.grids, extraps_eff, ops, _sample_data(itp))
    oob_result !== nothing && return oob_result
    cell = _locate_cell(itp, qc, extraps_eff, policies, hints, mono)
    return _eval_at_cell(itp, cell, ops)
end

# Shared scalar-query entry for the tensor-product ND callables (Linear/Cubic/
# Quadratic/Constant/Hermite): resolve query/ops/search/hints + the call-time
# extrap override once, then delegate. Hetero keeps its own callable.
@inline function _eval_nd_scalar_query(
        itp::AbstractInterpolantND{Tg, Tv, N}, query, deriv, extrap, search, hint
    ) where {Tg, Tv, N}
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))
    extraps = _resolve_extrap_override_nd(itp, extrap)
    return _eval_nd_at_point(itp, resolved, ops, policies, hints, mono, extraps)
end

# ========================================
# ND Scalar: Vector query → tuple conversion
# ========================================
# ForwardDiff/Optim — AbstractVector point query. `{<:Number}` is the scalar-vs-container
# gate: an AoS batch (`Vector{<:Tuple}`) stays on the batch method below, not misread as
# one point. Every ND point path is Number-gated; tuple-izing recovers per-element
# concrete types from an abstract mixed-unit eltype (`Quantity <: Number`).
@inline function (itp::AbstractInterpolantND{Tg, Tv, N})(
        query::AbstractVector{<:Number};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return itp(query_tuple; deriv = deriv, extrap = extrap, search = search, hint = hint)
end

# ========================================
# ND Scalar: Vararg convenience
# ========================================
# Converts vararg calls to tuple form: itp(0.5, GridIdx(3)) → itp((0.5, GridIdx(3)))
# Per-type callables (Cubic, Linear, etc.) accept Tuple{Vararg{Number, N}} directly,
# resolving GridIdx via _resolve_grididx internally. GridIdx <: Real, so Vararg{Number,N} matches.
@inline function (itp::AbstractInterpolantND{Tg, Tv, N})(
        q::Vararg{Number, N};
        kw...,
    ) where {Tg, Tv, N}
    return itp(q; kw...)
end

# ========================================
# ND In-Place Batch — Unified
# ========================================
#
# Single entry point for all batch in-place evaluation.
# Protocol functions (_query_length, _query_extract, _query_eltype) dispatch
# directly on query container type — no normalization needed.

# Shaped in-place batch: `output` must match `_query_size(queries)` EXACTLY — a
# length-only match would silently flatten a matrix query into a vector sink. The
# `AbstractVector` peer keeps the established vector-batch dispatch stable; both route
# through one shape-checking helper. (Column-major linear indexing fills the shape.)
function (itp::AbstractInterpolantND{Tg, Tv, N})(
        output::AbstractArray,
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _nd_batch_inplace!(itp, output, queries, deriv, extrap, search, hint)
end

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        output::AbstractVector,
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    return _nd_batch_inplace!(itp, output, queries, deriv, extrap, search, hint)
end

@inline function _nd_batch_inplace!(
        itp::AbstractInterpolantND{Tg, Tv, N}, output, queries, deriv, extrap, search, hint
    ) where {Tg, Tv, N}
    ops = _resolve_deriv_nd(deriv, Val(N))
    _check_query_output_size(output, queries)
    # `extrap` (nothing → stored; InBounds → fast-path; else errors) resolved per-axis.
    extraps0 = _resolve_extrap_override_nd(itp, extrap)
    return _nd_batch_pointwise!(output, itp, queries, ops, extraps0, search, hint)
end

# Point-wise batch core, shared by the generic in-place callable above and the
# unified `GriddedQuery` functor's fallback branch (a non-separable method's
# gridded eval is this core into a flat view of its N-D output). `extraps0` is
# already override-resolved; `ops` already per-axis. HeteroND rides this too —
# it is an `AbstractInterpolantND`, so the resolver threads its stored state.
@inline function _nd_batch_pointwise!(
        output::AbstractArray,
        itp::AbstractInterpolantND{Tg, Tv, N},
        queries,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps0,
        search,
        hint
    ) where {Tg, Tv, N}
    _query_validate(queries)
    # Validate + batch-level InBounds promotion: throws on OOB NoExtrap and returns per-axis
    # `InBounds()` for SoA queries all in-bounds on an axis (AoS/generic stays original).
    extraps_eff = _validate_nd_domain(itp.grids, queries, extraps0)
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _check_mono_nd(policies, queries)
    _interp_nd_batch!(output, itp, queries, extraps_eff, ops, policies, hints, mono)
    return output
end

# ========================================
# ND Allocating Batch — Unified
# ========================================
# Buffer eltype from `_promote_eltype(itp, Tq)` (see 1D variant for kernel-
# type rationale); delegates to in-place — no sample-first eval.

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    Tq = _query_eltype(queries)
    output = _alloc_query_output(_promote_eltype(itp, Tq, deriv), queries)
    return itp(output, queries; deriv = deriv, extrap = extrap, search = search, hint = hint)
end
