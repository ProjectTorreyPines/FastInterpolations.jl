# ============================================================================
# GriddedQuery — rectilinear (tensor-product) evaluation
# ============================================================================
#
# A `GriddedQuery` bundles one target-coordinate vector per axis. Evaluating an
# interpolant at it computes the interpolant at every combination (outer/tensor
# product) of those coordinates, returning an N-D array sized `map(length, axes)`.
#
#   itp(GriddedQuery(x1d, y1d))  ==  [itp((x1d[i], y1d[j])) for i, j]   (M×N)
#
# The ONE idea this path adds over point-wise evaluation is SEPARABILITY:
# each axis's (interval index, weight, cell width) is resolved ONCE per axis —
# O(ΣM_d) anchors — and reused across all O(∏M_d) outputs.
#
# This file holds the `GriddedQuery` type + ND-query protocol and the pieces
# shared by every separable method: the anchor-build searcher, the FillExtrap
# post-pass, and the `_try_gridded_separable!` dispatch hook. Each method's own
# machinery lives in its file (gridded_linear.jl for the default linear path,
# gridded_constant.jl, gridded_hermite.jl, gridded_partials.jl).

"""
    GriddedQuery(x1d, y1d, ...)
    GriddedQuery(axes::Tuple{Vararg{AbstractVector}})

A rectilinear (tensor-product) query: one coordinate vector per axis. Evaluating
an interpolant at a `GriddedQuery` returns the interpolant sampled at every
combination of the per-axis coordinates - an `N`-D array of size
`map(length, axes)`.

The coordinate axes are arbitrary `AbstractVector`s. `AbstractRange` axes work
directly; do not `collect` them unless you really need a `Vector`.

# Examples

```julia
data = rand(10, 20)
itp = linear_interp((1:10, 1:20), data)
gq = GriddedQuery(range(1, 10, 40), [5, 6, 7])

A = itp(gq)                    # size(A) == (40, 3)
B = linear_interp((1:10, 1:20), data, gq)
out = similar(A)
itp(out, gq)                   # writes into the shaped output array
```

This is different from the usual ND batch query `(xqs, yqs)`, which evaluates
pairwise points `(xqs[i], yqs[i])` and returns a vector. `GriddedQuery(xqs,
yqs)` evaluates all combinations and returns a matrix.

# Collection interface

A `GriddedQuery` is a read-only, immutable `AbstractArray{E, N}` of its
Cartesian-product coordinate points — the lazy, non-materialized grid of points,
analogous to `CartesianIndices` but yielding coordinate tuples instead of index
tuples. `E` is the point tuple type (`Tuple` of per-axis element types), `N` the
number of axes. Its column-major order (axis 1 varies fastest) is exactly the
interpolation output order, so `itp(gq)[k] == itp(gq[k])` (up to the method's
normal floating-point contract). It has no `setindex!` — it is never mutated.

```julia
gq = GriddedQuery(([10, 20, 30], [1.5, 2.5]))   # 3×2 grid of points

size(gq)            # (3, 2)
eltype(gq)          # Tuple{Int, Float64}  — the coordinate point type
gq[2]               # (20, 1.5)   — k-th point, column-major (linear index)
gq[1, 2]            # (10, 2.5)   — shaped index (also gq[CartesianIndex(1, 2)])
gq[1:2, 2]          # [(10, 2.5), (20, 2.5)]  — slicing (scalar dim dropped)
gq[:, 1]            # [(10, 1.5), (20, 1.5), (30, 1.5)]  — a column of points
first.(gq)          # 3×2 Matrix{Int} — broadcasting works
collect(gq)         # 3×2 Matrix of point tuples
map(itp, gq)        # 3×2 Matrix — the shaped point-wise reference for itp(gq)
```
"""
struct GriddedQuery{T <: Tuple, E, N} <: AbstractArray{E, N}
    axes::T
end
# `E` (point tuple type) and `N` (axis count) are derived from the axes tuple
# type `T`, so callers only ever pass the axes.
@inline _gq_point_eltype(::Type{T}) where {T <: Tuple} =
    Tuple{ntuple(d -> eltype(fieldtype(T, d)), Val(fieldcount(T)))...}
function GriddedQuery(axes::Tuple{Vararg{AbstractVector}})
    T = typeof(axes)
    return GriddedQuery{T, _gq_point_eltype(T), fieldcount(T)}(axes)
end
GriddedQuery(axes::AbstractVector...) = GriddedQuery(axes)

Base.size(gq::GriddedQuery) = map(length, gq.axes)

# ---- ND query protocol ------------------------------------------------------
# A GriddedQuery is a batch of ∏M_d cartesian-product points in COLUMN-MAJOR
# order, so the generic `interp`/batch path consumes it like any other query
# container. Its `_query_size` is the N-D `size(gq)` the GriddedQuery contract
# promises, so the allocating path returns that array directly and in-place
# callers pass a matching one. Point k unravels via `CartesianIndices` (axis 1
# fastest), so linear-indexed writes land at the right N-D position.
# `Val(fieldcount(T))` keeps the ntuple over the (heterogeneous) axes tuple
# type-stable (a runtime axis index would box).
@inline _query_length(gq::GriddedQuery) = prod(size(gq))
@inline _query_size(gq::GriddedQuery) = size(gq)
@inline _query_eltype(gq::GriddedQuery) = promote_type(map(eltype, gq.axes)...)
# Shared UNCHECKED logical-point core: unravel linear index `k` (column-major)
# and gather one coordinate per axis. The `@inbounds` are written INSIDE the
# `ntuple` closure on purpose — `@inbounds`/`@propagate_inbounds` do NOT
# propagate across a closure boundary, so a caller-side `@inbounds` would leave
# the per-axis `gq.axes[d][ci[d]]` checks in place. Keeping them lexical makes
# this core codegen-identical to the pre-collection `_query_extract`.
# `Val(fieldcount(T))` keeps the ntuple over the heterogeneous axes tuple
# type-stable (a runtime axis index would box). Callers guarantee `k` is in
# range: `_query_extract` by protocol contract, `getindex` by its own
# `@boundscheck` (a valid linear `k` ⟹ every 1-based per-axis access is valid).
@inline function _gridded_point(gq::GriddedQuery{T}, k) where {T}
    ci = @inbounds CartesianIndices(size(gq))[k]
    return ntuple(d -> @inbounds(gq.axes[d][ci[d]]), Val(fieldcount(T)))
end
@inline _query_extract(gq::GriddedQuery, k) = _gridded_point(gq, k)

# ---- AbstractArray interface -------------------------------------------------
# `GriddedQuery` is an IndexCartesian `AbstractArray`, so `size` + the per-axis
# scalar `getindex` below are the whole required surface. Linear indexing
# `gq[k]`, broadcasting, `collect`/`map`, `first`/`last`, `eachindex`/`keys`,
# bounds checking, and the `HasShape`/`HasEltype` iterator traits come from
# Base's generic machinery. Cartesian *slicing* is intercepted below to stay
# lazy (a sub-`GriddedQuery`). No `setindex!` → read-only (like a `Range`).
Base.IndexStyle(::Type{<:GriddedQuery}) = IndexCartesian()

# Per-axis scalar accessor. The `@inbounds` are LEXICAL inside the ntuple closure:
# `@inbounds` does not cross the closure boundary (same reason as `_gridded_point`),
# so a caller-side `@inbounds gq[i, j]` still needs them here to elide the per-axis
# checks. `checkbounds` (from AbstractArray) validates the indices against `size(gq)`.
Base.@propagate_inbounds function Base.getindex(gq::GriddedQuery{T, E, N}, I::Vararg{Int, N}) where {T, E, N}
    @boundscheck checkbounds(gq, I...)
    return ntuple(d -> @inbounds(gq.axes[d][I[d]]), Val(N))
end

# Lazy Cartesian slicing: when EVERY index is a vector/range/colon (no scalar
# drop, no linear span), return a sub-`GriddedQuery` over per-axis sub-selections
# instead of materializing a `Matrix{E}` — the value-valued analogue of
# `CartesianIndices[1:2, 3:4]` staying a `CartesianIndices`. It preserves eltype
# and shape, re-enters the separable fast path, and copies NO coordinate data:
# `view` is 0-copy for both range axes (stays a range) and vector axes
# (`SubArray`); a `Colon` returns the parent axis object untouched. Mixed
# scalar+vector (dim drop) and single-arg linear ranges keep Base's materializing
# behavior (arity ≠ N ⇒ this method doesn't match).
@inline _sub_axis(ax, I) = view(ax, I)
@inline _sub_axis(ax, ::Colon) = ax
Base.@propagate_inbounds function Base.getindex(
        gq::GriddedQuery{T, E, N}, I::Vararg{Union{AbstractVector, Colon}, N}
    ) where {T, E, N}
    @boundscheck checkbounds(gq, I...)
    return GriddedQuery(ntuple(d -> _sub_axis(gq.axes[d], I[d]), Val(N)))
end

# N=0 Aqua disambiguator: for a 0-D grid, `gq[]` matches both `Vararg{Int,0}` and
# `Vararg{Union{AbstractVector,Colon},0}`; the explicit 0-arg accessor dominates.
Base.getindex(gq::GriddedQuery{T, E, 0}) where {T, E} = _gridded_point(gq, 1)

# Custom `iterate` (a specialization of the AbstractArray default): forwarding to
# `Iterators.product(gq.axes...)` traverses in the same column-major order but is
# measurably faster for a full sweep than the getindex-per-step default, and stays
# 0-alloc. `collect`/`map`/`enumerate` still preserve `size(gq)` via `HasShape`.
@inline Base.iterate(gq::GriddedQuery) = iterate(Iterators.product(gq.axes...))
@inline Base.iterate(gq::GriddedQuery, state) = iterate(Iterators.product(gq.axes...), state)

# ---- per-axis anchor resolution --------------------------------------------
# Anchor-build searcher. Range grids resolve to the O(1) direct arm as usual.
# For search-based (Vector) grids, CLUSTERED targets chain a hint instead
# (LinearBinarySearch, walk window 8): each search starts at the previous
# anchor's interval. The winning regime is mean consecutive-target stride
# within a few cells — measured on a n=2048 Vector grid (sorted targets):
# 6.9x at 0.25-cell stride, 4.3x at 2, 1.2x at 6, LOSES at 8 (walk exhaust
# then binary fallback) — threshold 4 leaves margin for non-uniform cells.
# The stride estimate uses first/last (exact for sorted targets; an unsorted
# misestimate degrades gracefully — a bad hint just falls back to binary).
@inline function _gridded_build_searcher(grid::AbstractVector, targets::AbstractVector)
    s = _resolve_searcher_for_grid(grid, DEFAULT_SEARCHER)
    s isa Searcher{BinarySearch, NoHint} || return s
    M = length(targets)
    M >= 2 || return s
    span_t = abs(float(last(targets)) - float(first(targets)))
    span_g = float(last(grid)) - float(first(grid))
    span_t * (length(grid) - 1) <= 4 * span_g * (M - 1) || return s
    return _to_searcher(LinearBinarySearch())
end

# ---- FillExtrap post-pass ----------------------------------------------------
# Mirrors point-wise `_try_fill_oob`: ANY Fill-axis OOB coordinate fills its
# whole output slab with the FIRST Fill axis's value (EvalValue) or a carrier
# zero (any derivative op). Classification re-scans the query axis with
# `_oob_state` (the same widened bracket the anchor build used); compile-time
# no-op unless some axis is FillExtrap.
@inline _gridded_has_fill(::Tuple{}) = false
@inline _gridded_has_fill(extraps::Tuple) =
    extraps[1] isa FillExtrap || _gridded_has_fill(Base.tail(extraps))
@inline _gridded_first_fill_value(extraps::Tuple) =
    extraps[1] isa FillExtrap ? extraps[1].fill_value :
    _gridded_first_fill_value(Base.tail(extraps))

function _gridded_fill_oob!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}}
    ) where {N}
    return _gridded_fill_oob_sample!(out, grids, @inbounds(first(data)), targets, extraps, ops)
end

function _gridded_fill_oob_sample!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data_sample,
        targets::Tuple,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ops::Tuple{Vararg{AbstractEvalOp, N}}
    ) where {N}
    _gridded_has_fill(extraps) || return out
    fill_value = _gridded_first_fill_value(extraps)
    for d in 1:N
        extraps[d] isa FillExtrap || continue
        k = 0
        for xq in targets[d]
            k += 1
            if _oob_state(grids[d], xq) != IN_DOMAIN
                fill!(selectdim(out, d, k), _fill_extrap_result(ops, fill_value, data_sample, xq, grids))
            end
        end
    end
    return out
end

# ---- unified `interp` fast-path ----------------------------------------------
# `_try_gridded_separable!` is the `interp!` fast-path hook. A `GriddedQuery`
# reaches one method-tuple dispatcher with an already shaped N-D output;
# supported method families prepare raw one-shot inputs and then call
# `_gridded_eval_methods!`. Unsupported tuples return `false`, so `interp!`
# falls through to its point-wise batch.
@inline _try_gridded_separable!(output, grids, data, queries, methods, extrap, deriv) =
    _try_gridded_separable!(output, grids, data, queries, methods, extrap, deriv, AutoCoeffs())
@inline _try_gridded_separable!(output, grids, data, queries, methods, extrap, deriv, coeffs) = false
@inline function _try_gridded_separable!(
        output::AbstractArray{<:Any, N},
        grids,
        data::AbstractArray{Tv, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap,
        deriv,
        coeffs
    ) where {Tv, N}
    return _try_gridded_oneshot_methods!(output, grids, data, gq, methods, extrap, deriv, coeffs)
end

# Method-tuple dispatcher scaffolding (shared): the generic no-op fallback, the
# `GriddedQuery` → `gq.axes` bridge, and the empty-method N=0 case. Each separable
# family adds its own concrete `_try_gridded_oneshot_methods!` overload in its
# file; anything unmatched falls through here to `false` (point-wise batch).
@inline _try_gridded_oneshot_methods!(out, grids, data, targets, methods, extrap, deriv) = false
@inline _try_gridded_oneshot_methods!(out, grids, data, gq::GriddedQuery, methods, extrap, deriv, _coeffs) =
    _try_gridded_oneshot_methods!(out, grids, data, gq.axes, methods, extrap, deriv)
@inline function _try_gridded_oneshot_methods!(
        out_nd::AbstractArray{<:Any, 0},
        grids,
        data::AbstractArray{Tv, 0},
        gq::GriddedQuery{<:Tuple{}},
        methods::Tuple{},
        extrap,
        deriv,
        coeffs
    ) where {Tv}
    return false
end

function interp!(
        output::AbstractArray{Tout, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint = nothing,
    ) where {Tout, N}
    size(output) == size(gq) || throw(
        DimensionMismatch("output size $(size(output)) != query size $(size(gq))")
    )
    return _interp_nd_oneshot_batch_public!(
        output, grids, data, gq, method, coeffs, deriv, extrap, search, hint
    )
end

function interp!(
        output::AbstractArray,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint = nothing,
    ) where {N}
    throw(DimensionMismatch("output size $(size(output)) != query size $(size(gq))"))
end
