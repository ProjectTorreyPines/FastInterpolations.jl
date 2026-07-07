# ============================================================================
# GriddedQuery — rectilinear (tensor-product) evaluation
# ============================================================================
#
# A `GriddedQuery` bundles one target-coordinate vector per axis. Evaluating an
# interpolant at it computes the interpolant at every combination (outer/tensor
# product) of those coordinates, returning an N-D array sized `map(length, axes)`.
#
#   itp(GriddedQuery((x1d, y1d)))  ==  [itp((x1d[i], y1d[j])) for i, j]   (M×N)
#
# The ONE idea this path adds over point-wise evaluation is SEPARABILITY:
# each axis's (interval index, weight, cell width) is resolved ONCE per axis —
# O(ΣM_d) anchors — and reused across all O(∏M_d) outputs, instead of
# re-searching and re-weighting every axis at every output point.
#
# Two execution strategies consume the same anchors (measured on a fixed-source
# ratio sweep):
#   * FUSED      — straight (2^N − 1)-blend per output from the 2^N corners,
#                  no intermediate buffer. Wins pure downsampling.
#   * FULLBUFFER — classic separable multi-pass through pooled intermediates,
#                  one axis per pass. Wins same-size / upsampling / mixed
#                  (contiguous passes).
# Both are generated over N; per-axis derivative ops ride on the shared 1D
# kernel `_linear_kernel(op, yL, yR, inv_h, α)`, exactly like point-wise ND.

"""
    GriddedQuery(axes::Tuple)
    GriddedQuery(x1d, y1d, ...)

A rectilinear (tensor-product) query: one coordinate vector per axis. Evaluating
an interpolant at a `GriddedQuery` returns the interpolant sampled at every
combination of the per-axis coordinates — an `N`-D array of size
`map(length, axes)`.

The coordinate vectors are arbitrary `AbstractVector`s (not restricted to
ranges); uniform (`AbstractRange`) axes are the image-resize special case.
"""
struct GriddedQuery{T <: Tuple}
    axes::T
end
GriddedQuery(axes::AbstractVector...) = GriddedQuery(axes)

Base.size(gq::GriddedQuery) = map(length, gq.axes)
Base.ndims(::GriddedQuery{T}) where {T} = fieldcount(T)

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
@inline function _query_extract(gq::GriddedQuery{T}, k) where {T}
    ci = @inbounds CartesianIndices(size(gq))[k]
    return ntuple(d -> @inbounds(gq.axes[d][ci[d]]), Val(fieldcount(T)))
end

# ---- per-axis anchor --------------------------------------------------------
# The retained per-axis resolution artifact: interval index + linear weight +
# cell reciprocal width, with extrapolation already folded into the weight.
# 24 B isbits; one Vector per query axis is the whole precomputation. `inv_h`
# feeds the derivative kernels (DCE'd by value-only evaluation).
struct _GriddedAnchor{T <: Real}
    idx::Int    # left node index (right node = idx + 1)
    alpha::T    # normalized in-cell weight; may leave [0, 1] under ExtendExtrap
    inv_h::T    # reciprocal cell width 1/(x[idx+1] - x[idx])
end

"""
    _gridded_anchors!(anchors, grid, targets, extrap, dim) -> anchors

Resolve every target coordinate in `targets` against `grid` once — the O(M)
work that both strategies reuse O(∏M_d) times. Extrapolation is folded here:
`WrapExtrap` folds the coordinate into the domain, `ClampExtrap`/`FillExtrap`
clamp the weight to the boundary node (Fill's OOB slabs are overwritten by the
[`_gridded_fill_oob!`](@ref) post-pass), `ExtendExtrap` keeps the out-of-range
weight (linear extrapolation), and `NoExtrap` throws on the first
out-of-domain coordinate — BEFORE any output-sized buffer exists. `dim` names
the axis in that error.
"""
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

function _gridded_anchors!(
        anchors::Vector{_GriddedAnchor{T}},
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int
    ) where {T}
    length(anchors) == length(targets) || throw(
        DimensionMismatch("anchor buffer length $(length(anchors)) != query length $(length(targets))")
    )
    return _gridded_anchors_loop!(anchors, grid, targets, extrap, dim, _gridded_build_searcher(grid, targets))
end

# The searcher union (direct / binary / hinted) splits at this call, keeping
# the fill loop fully specialized per arm. @inline is LOAD-BEARING: it keeps
# the hinted arm's fresh `RefHint` inside the caller's compiled unit, so
# escape analysis elides the `Ref(1)` heap allocation — the same mechanism
# that keeps `_ensure_hint_nd`'s per-call Refs allocation-free on the scalar
# ND path. A non-inlined barrier here measured 32 B/axis.
@inline function _gridded_anchors_loop!(
        anchors::Vector{_GriddedAnchor{T}},
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        searcher::Searcher
    ) where {T}
    k = 0
    @inbounds for xq in targets
        k += 1
        if extrap isa InBounds
            # caller-asserted in-domain (call-time override): skip the domain
            # classification and take the lean search directly — the same call
            # the default path reaches after its `_oob_state` check, so
            # in-domain anchors are bit-identical to the default path's.
            idx, _, xL, xR = search_interval(searcher, grid, xq, extrap)
            inv_h = _get_inv_h(grid, idx, xL, xR)
            anchors[k] = _GriddedAnchor{T}(idx, T(_alpha_of(xq, xL, inv_h)), T(inv_h))
            continue
        end
        loc = _anchor_loc(grid, xq, extrap isa WrapExtrap, searcher)
        if extrap isa NoExtrap && loc.state != IN_DOMAIN
            _throw_domain_error(xq, grid, dim)   # canonical message, axis-named
        end
        inv_h = _get_inv_h(grid, loc.idx, loc.xL, loc.xR)
        α = T(_alpha_of(loc.xq, loc.xL, inv_h))
        if extrap isa Union{ClampExtrap, FillExtrap}
            α = clamp(α, zero(T), one(T))
        end
        anchors[k] = _GriddedAnchor{T}(loc.idx, α, T(inv_h))
    end
    return anchors
end

# Weight type: float promotion of grid × query (Float32 grid+query stays Float32).
@inline _gridded_alpha_type(grid::AbstractVector, targets::AbstractVector) =
    float(promote_type(eltype(grid), eltype(targets)))

function _gridded_anchors(
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int
    )
    T = _gridded_alpha_type(grid, targets)
    return _gridded_anchors!(
        Vector{_GriddedAnchor{T}}(undef, length(targets)), grid, targets, extrap, dim
    )
end

function _gridded_anchors_pooled(
        pool,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int
    )
    T = _gridded_alpha_type(grid, targets)
    return _gridded_anchors!(
        acquire!(pool, _GriddedAnchor{T}, length(targets)), grid, targets, extrap, dim
    )
end

# ---- FUSED strategy ---------------------------------------------------------
# One straight-line (2^N − 1)-blend per output from the 2^N corners of the
# anchor cell — no intermediate buffer. Generated over N: loop `d` loads
# axis-d's anchor at its own nesting level (outer-axis loads amortize), axis 1
# is innermost (stride-1 writes). The corner loads have data-dependent
# addresses, so the loop does not SIMD-vectorize; throughput comes from the
# branch-free body pipelining across iterations (2D measured load-port-bound,
# ~2.3 cycles/output).

# Expression builder for the generated kernel (defined before the generator;
# the generator body assembles expressions only): nested corner collapse where
# the axis-d combine wraps the two axis-d corners and axes d+1..N are
# collapsed inside — for N = 2 this is exactly lo/hi = axis-2 combines, then
# the axis-1 combine. The combine is the shared 1D kernel
# `_linear_kernel(op, yL, yR, inv_h, α)`, so the per-axis op selects value
# blend / slope / carrier-zero exactly as point-wise ND eval does.
function _fused_corner_expr(N::Int, d::Int, idxs::Vector{Any})
    d > N && return Expr(:ref, :A, idxs...)
    lo = copy(idxs)
    lo[d] = Symbol(:il_, d)
    hi = copy(idxs)
    hi[d] = :($(Symbol(:il_, d)) + 1)
    return :(
        _linear_kernel(
            ops[$d],
            $(_fused_corner_expr(N, d + 1, lo)),
            $(_fused_corner_expr(N, d + 1, hi)),
            $(Symbol(:invh_, d)),
            $(Symbol(:alpha_, d)),
        )
    )
end

@generated function _gridded_fused!(
        out::AbstractArray{<:Any, N},
        A::AbstractArray{<:Any, N},
        anchors::NTuple{N, Vector{<:_GriddedAnchor}},
        ops::Tuple{Vararg{AbstractEvalOp, N}}
    ) where {N}
    js = [Symbol(:j_, d) for d in 1:N]
    body = :(out[$(js...)] = $(_fused_corner_expr(N, 1, Any[:_ for _ in 1:N])))
    for d in 1:N   # d = 1 built first → innermost loop
        a = Symbol(:a_, d)
        body = quote
            for $(js[d]) in eachindex(anchors[$d])
                $a = anchors[$d][$(js[d])]
                $(Symbol(:il_, d)) = $a.idx
                $(Symbol(:alpha_, d)) = $a.alpha
                $(Symbol(:invh_, d)) = $a.inv_h
                $body
            end
        end
    end
    return quote
        @inbounds $body
        return out
    end
end

_gridded_fused!(
    out::AbstractArray{<:Any, N},
    A::AbstractArray{<:Any, N},
    anchors::NTuple{N, Vector{<:_GriddedAnchor}}
) where {N} = _gridded_fused!(out, A, anchors, ntuple(_ -> EvalValue(), Val(N)))

# ---- FULLBUFFER strategy ----------------------------------------------------
# Classic separable multi-pass: each pass resolves ONE axis (`D`) through the
# anchor combine and passes every other axis through. Generated over (N, D)
# with axis 1 innermost — for D ≥ 2 both loops are stride-1 (contiguous
# blend), for D = 1 the two taps (idx, idx+1) are memory-adjacent loads
# (gather). `ranges[e]` bounds each pass-through axis `e` (`ranges[D]` is
# ignored): the multi-pass core passes tap HULLS there so slabs no
# downstream pass will ever read are never computed (a pure loop-bounds
# tightening — identical arithmetic on the entries that are computed).
@generated function _gridded_pass!(
        dest::AbstractArray{<:Any, N},
        src::AbstractArray{<:Any, N},
        anchors::Vector{<:_GriddedAnchor},
        op::AbstractEvalOp,
        ::Val{D},
        ranges::NTuple{N, UnitRange{Int}}
    ) where {N, D}
    js = [Symbol(:j_, d) for d in 1:N]
    lo = Any[js[d] for d in 1:N]
    lo[D] = :il
    hi = copy(lo)
    hi[D] = :(il + 1)
    body = :(
        dest[$(js...)] = _linear_kernel(op, src[$(lo...)], src[$(hi...)], a.inv_h, a.alpha)
    )
    for d in 1:N   # d = 1 built first → innermost loop
        body = if d == D
            quote
                for $(js[d]) in eachindex(anchors)
                    a = anchors[$(js[d])]
                    il = a.idx
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
        @inbounds $body
        return dest
    end
end

# Full-range convenience form (explicit compositions, tests).
_gridded_pass!(
    dest::AbstractArray{<:Any, N},
    src::AbstractArray{<:Any, N},
    anchors::Vector{<:_GriddedAnchor},
    op::AbstractEvalOp,
    v::Val
) where {N} = _gridded_pass!(dest, src, anchors, op, v, map(UnitRange{Int}, axes(src)))

# Static dispatch over the runtime axis id: a generated if-chain on `d`
# turns the pass call into N STATIC call sites (literal Val / anchors[k] /
# ops[k]), so isbits arguments — the ranges tuple — cross in registers. A
# single runtime-Val(d) dynamic call here boxed the tuple (96 B/call).
@generated function _gridded_pass_dim!(
        dest::AbstractArray{<:Any, N},
        src::AbstractArray{<:Any, N},
        anchors::NTuple{N, Vector{<:_GriddedAnchor}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        d::Int,
        ranges::NTuple{N, UnitRange{Int}}
    ) where {N}
    ex = :(throw(ArgumentError("pass axis out of range")))
    for k in N:-1:1
        ex = :(
            d == $k ? _gridded_pass!(dest, src, anchors[$k], ops[$k], Val($k), ranges) : $ex
        )
    end
    return ex
end

# Tap hull of one query axis: the only source/intermediate slab range any
# pass consuming that axis's anchors will ever read. Full-domain queries give
# the whole axis; clustered (zoom) queries shrink upstream pass work to the
# window. Computed as a coordinate `extrema` (contiguous SIMD reduction over
# the raw targets — an anchor-vector idx scan is a strided 24 B walk) mapped
# through two locates: coordinate → interval index is monotone for every
# non-wrapping extrap (OOB resolves to a boundary cell), so the coordinate
# extrema ARE the tap extrema. Wrap folds coordinates and breaks that
# monotonicity — it keeps the full axis.
_gridded_hull(grid::AbstractVector, targets::AbstractVector, ::WrapExtrap) =
    1:length(grid)
function _gridded_hull(grid::AbstractVector, targets::AbstractVector, ::AbstractExtrap)
    # separate minimum/maximum: each SIMD-vectorizes (~0.15 ns/elem measured);
    # Base.extrema's tuple-mapreduce is ~34x slower and was end-to-end visible
    lo = minimum(targets)
    hi = maximum(targets)
    searcher = _resolve_searcher_for_grid(grid, DEFAULT_SEARCHER)
    return _anchor_loc(grid, lo, false, searcher).idx:(_anchor_loc(grid, hi, false, searcher).idx + 1)
end

# Pass order: adjacent-exchange comparator on total cost with the measured
# per-element pass costs (gather along axis 1 ≈ 0.65 ns, contiguous
# otherwise ≈ 0.2 ns). Putting axis `d` before `e` is cheaper iff
#   c_d·M_d·(n_e − M_e) ≤ c_e·M_e·(n_d − M_d)
# (shrink-most-first, cost-weighted). For N = 2 this reduces exactly to the
# old dim2-first model.
@inline _gridded_pass_cost(d::Int) = d == 1 ? 0.65 : 0.2
@inline function _gridded_pass_before(d::Int, e::Int, src_size::Dims, out_size::Dims)
    return _gridded_pass_cost(d) * out_size[d] * (src_size[e] - out_size[e]) <
        _gridded_pass_cost(e) * out_size[e] * (src_size[d] - out_size[d])
end

# Multi-pass core through pooled intermediates held in `Tmid` (the promoted
# output eltype — narrowing into a caller-supplied narrower `out` happens
# exactly once, on the last pass). Passes run in the comparator order above;
# each `_gridded_pass!` call is a function barrier (the running array type
# alternates data → pooled intermediate → out).
@with_pool pool function _gridded_fullbuffer!(
        out::AbstractArray{<:Any, N},
        A::AbstractArray{<:Any, N},
        anchors::NTuple{N, Vector{<:_GriddedAnchor}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        hulls::NTuple{N, UnitRange{Int}},
        ::Type{Tmid}
    ) where {N, Tmid}
    src_size = size(A)
    out_size = map(length, anchors)
    order = ntuple(identity, Val(N))
    @inbounds for k in 2:N   # insertion sort by the pairwise cost comparator
        j = k
        while j > 1 && _gridded_pass_before(order[j], order[j - 1], src_size, out_size)
            oj = order[j]
            order = Base.setindex(Base.setindex(order, order[j - 1], j), oj, j - 1)
            j -= 1
        end
    end
    # Pass-through bounds: an unresolved axis only needs its tap hull (zoom
    # queries shrink upstream passes to the window); once axis d is resolved
    # its full target extent is needed downstream.
    ranges = hulls
    cur_size = src_size
    src = A
    @inbounds for k in 1:N
        d = order[k]
        cur_size = Base.setindex(cur_size, out_size[d], d)
        dest = k == N ? out : acquire!(pool, Tmid, cur_size...)
        _gridded_pass_dim!(dest, src, anchors, ops, d, ranges)
        ranges = Base.setindex(ranges, 1:out_size[d], d)
        src = dest
    end
    return out
end

# ---- strategy selection ------------------------------------------------------
# Measured map (fixed-source ratio sweep): fused wins pure downsampling (its
# per-output work shrinks with the output), fullbuffer wins same-size /
# upsampling / mixed (contiguous passes amortize per-node work).
function _gridded_apply!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        anchors::NTuple{N, Vector{<:_GriddedAnchor}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tmid}
    ) where {N, Tmid}
    A = data
    out_size = map(length, anchors)
    size(out) == out_size || throw(
        DimensionMismatch("output size $(size(out)) != query size $out_size")
    )
    any(iszero, out_size) && return out
    if all(map(<, out_size, size(A)))
        _gridded_fused!(out, A, anchors, ops)
    else
        _gridded_fullbuffer!(out, A, anchors, ops, _gridded_hulls(grids, targets, extraps, Val(N)), Tmid)
    end
    return out
end

# Per-axis hull tuple via literal-index construction (no capturing closure). An
# `ntuple(d -> _gridded_hull(grids[d], targets[d], extraps[d]), Val(N))` boxes the
# heterogeneous `targets` tuple into its closure on some axis-type combinations
# (e.g. a `UnitRange{Int}` query axis beside a `Vector` — 3 allocs/call); the
# @generated form assembles the tuple directly and stays allocation-free.
@generated function _gridded_hulls(grids, targets, extraps, ::Val{N}) where {N}
    exprs = [:(_gridded_hull(grids[$d], targets[$d], extraps[$d])) for d in 1:N]
    return :(($(exprs...),))
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
    _gridded_has_fill(extraps) || return out
    fill_value = _gridded_first_fill_value(extraps)
    data_sample = @inbounds first(data)
    for d in 1:N
        extraps[d] isa FillExtrap || continue
        k = 0
        for xq in targets[d]
            k += 1
            if _oob_state(grids[d], xq) != IN_DOMAIN
                fill!(selectdim(out, d, k), _fill_extrap_result(ops, fill_value, data_sample, xq))
            end
        end
    end
    return out
end

# ---- entry points ------------------------------------------------------------
# Output eltype: matches point-wise scalar eval's promotion (pinned by test).
# `Tg`/`Tv` are the value-matched grid float and value type; both the persistent
# functor and the one-shot front resolve them the same way before this call.
@inline _gridded_out_eltype(::Type{Tg}, ::Type{Tv}, targets::Tuple) where {Tg, Tv} =
    _promote_eltype(_interp_op, Tg, Tv, promote_type(map(eltype, targets)...))

# `grids` here are already resolved to the value-matched `Tg` (persistent: the
# interpolant's cached axes; one-shot: `_cache_axis_pooled`) — the eval reads
# them as a plain `(grids, data)` pair, decoupled from any interpolant struct.
# Anchors are built (firing NoExtrap's O(ΣM_d) validation) BEFORE the
# output-sized array is allocated.
@with_pool pool function _gridded_eval(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tout}
    ) where {N, Tout}
    anchors = ntuple(
        d -> _gridded_anchors_pooled(pool, grids[d], targets[d], extraps[d], d),
        Val(N)
    )
    out = Array{Tout, N}(undef, map(length, targets))
    _gridded_apply!(out, grids, data, targets, anchors, ops, extraps, Tout)
    return _gridded_fill_oob!(out, grids, data, targets, extraps, ops)
end

@with_pool pool function _gridded_eval!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tout}
    ) where {N, Tout}
    anchors = ntuple(
        d -> _gridded_anchors_pooled(pool, grids[d], targets[d], extraps[d], d),
        Val(N)
    )
    _gridded_apply!(out, grids, data, targets, anchors, ops, extraps, Tout)
    return _gridded_fill_oob!(out, grids, data, targets, extraps, ops)
end

# ---- dispatch ----------------------------------------------------------------
"""
    (itp::LinearInterpolantND{Tg,Tv,N})(gq::GriddedQuery; deriv = EvalValue(), extrap = nothing)
    (itp::LinearInterpolantND{Tg,Tv,N})(out::AbstractArray, gq::GriddedQuery; deriv = EvalValue(), extrap = nothing)

Evaluate an N-D linear interpolant at every combination of `gq.axes`
coordinates. Each axis's anchors are resolved once (pooled, O(ΣM_d)) and
reused across the grid; the strategy is picked per call — fused (no
intermediate) for pure downsampling, separable multi-pass full buffer
otherwise. `deriv` takes a per-axis op tuple or a single op, and `extrap`
the call-time `InBounds` override (single or per-axis tuple), exactly like
point-wise ND eval. The 2-arg form writes into `out` (FI batch convention).
"""
function (itp::LinearInterpolantND{Tg, Tv, N})(
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing
    ) where {Tg, Tv, N}
    targets = gq.axes
    ops = _resolve_deriv_nd(deriv, Val(N))
    extraps = _resolve_extrap_override_nd(itp, extrap)
    Tout = _gridded_out_eltype(Tg, Tv, targets)
    return _gridded_eval(itp.grids, itp.data, targets, ops, extraps, Tout)
end
function (itp::LinearInterpolantND{Tg, Tv, N})(
        out::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing
    ) where {Tg, Tv, N}
    targets = gq.axes
    ops = _resolve_deriv_nd(deriv, Val(N))
    extraps = _resolve_extrap_override_nd(itp, extrap)
    Tout = _gridded_out_eltype(Tg, Tv, targets)
    return _gridded_eval!(out, itp.grids, itp.data, targets, ops, extraps, Tout)
end

# ---- one-shot API ------------------------------------------------------------
# `linear_interp(grids, data, gq)` mirrors the persistent gridded functor with
# no interpolant kept alive. Raw grids are resolved to the value-matched `Tg` in
# a POOL-backed `_CachedVector`/`_CachedRange` (`_cache_axis_pooled`), released
# at call scope — so a `Vector` grid pays no permanent cache allocation and the
# gridded machinery sees exactly the axis type the persistent path feeds it. The
# grid resolution shares its `@with_pool` frame with the eval; `_gridded_eval`'s
# own nested pool holds the anchors + fullbuffer scratch (fullbuffer already
# nests a pool, so this is an established pattern).
@with_pool pool function _linear_gridded_oneshot(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tg},
        ::Type{Tout}
    ) where {N, Tg, Tout}
    grids_c = map(g -> _cache_axis_pooled(pool, g, Tg), grids)
    extraps_c = map(_resolve_extrap, extraps, grids_c)
    return _gridded_eval(grids_c, data, targets, ops, extraps_c, Tout)
end

@with_pool pool function _linear_gridded_oneshot!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tg},
        ::Type{Tout}
    ) where {N, Tg, Tout}
    grids_c = map(g -> _cache_axis_pooled(pool, g, Tg), grids)
    extraps_c = map(_resolve_extrap, extraps, grids_c)
    return _gridded_eval!(out, grids_c, data, targets, ops, extraps_c, Tout)
end

"""
    linear_interp(grids, data, gq::GriddedQuery; bc, extrap, deriv)
    linear_interp!(out, grids, data, gq::GriddedQuery; bc, extrap, deriv)

One-shot N-D linear interpolation at every combination of `gq.axes` coordinates
(rectilinear resize). Builds no persistent interpolant: grids are pool-cached at
the value-matched precision for the duration of the call, so the warm path is
zero-allocation apart from the output array. `extrap` is the per-axis
extrapolation policy (`NoExtrap`/`ClampExtrap`/`FillExtrap`/`ExtrapExtend`) — the
same `WrapExtrap`/periodic support as the persistent gridded functor is reached
by building the interpolant explicitly.
"""
function linear_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {Tv, N}
    _validate_nd_grids(grids, data)
    targets = gq.axes
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    Tout = _gridded_out_eltype(Tg, Tv, targets)
    return _linear_gridded_oneshot(grids, data, targets, ops, extraps, Tg, Tout)
end

function linear_interp!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {Tv, N}
    _validate_nd_grids(grids, data)
    targets = gq.axes
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    Tout = _gridded_out_eltype(Tg, Tv, targets)
    return _linear_gridded_oneshot!(out, grids, data, targets, ops, extraps, Tg, Tout)
end

# ---- unified `interp` fast-path ----------------------------------------------
# `_try_gridded_separable!` is the `interp!` fast-path hook: for a (query, method)
# that has a separable gridded evaluator it does the write and returns `true`; the
# generic default returns `false`, so `interp!` falls through to its pointwise
# batch. It runs BEFORE `interp!` flattens the output, so the N-D array reaches
# the separable kernel with no reshape (zero extra allocation). Today only a
# GriddedQuery on an all-`LinearInterp` method qualifies (per-axis anchors resolved
# once, reused across ∏M_d outputs — 3–9× faster than pointwise, more in higher
# dimensions). Adding a separable evaluator for another method (constant, cubic, …)
# later is one more method here — no change to `interp!`.
@inline _try_gridded_separable!(output, grids, data, queries, methods, extrap, deriv) = false
function _try_gridded_separable!(
        output, grids, data, gq::GriddedQuery,
        methods::Tuple{LinearInterp, Vararg{LinearInterp}}, extrap, deriv
    )
    # already-N-D output is written directly; a flat vector (caller's choice) is
    # reshaped (aliasing) to the query shape for the N-D kernel.
    out_nd = output isa AbstractVector ? reshape(output, size(gq)) : output
    linear_interp!(out_nd, grids, data, gq; bc = map(m -> m.bc, methods), extrap = extrap, deriv = deriv)
    return true
end
