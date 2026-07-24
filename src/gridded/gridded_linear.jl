# ============================================================================
# GriddedQuery — linear separable evaluation
# ============================================================================
#
# The default separable method: per-axis anchors (interval + op-minimal payload)
# resolved once via the shared backbone, then reused across every output by one
# of two strategies:
#   * FUSED      — straight (2^N − 1)-blend per output from the 2^N cell corners,
#                  no intermediate buffer. Wins pure downsampling.
#   * FULLBUFFER — classic separable multi-pass through pooled intermediates,
#                  one axis per pass. Wins same-size / upsampling / mixed.
# Both are generated over N and ride the shared 1D kernel `_linear_kernel`, so
# per-axis derivative ops behave exactly like point-wise ND eval.
#
# This file owns the whole Linear method: op-minimal payloads + `_linear_kernel`
# overloads + `_axis_anchor_type`/`_resolve_anchor`, the anchor build, both
# strategy kernels, the eval entry points, and the `linear_interp` one-shot API.
# `GriddedQuery` itself + shared helpers (searcher, Fill post-pass,
# `_try_gridded_separable!`) live in gridded_query.jl.

# Weight type: float promotion of grid × query (Float32 grid+query stays Float32).
@inline _gridded_alpha_type(grid::AbstractVector, targets::AbstractVector) =
    float(promote_type(eltype(grid), eltype(targets)))

@inline _linear_gridded_anchor_method() = LinearInterp(NoBC())

# ---- op-minimal anchor payloads + resolution (linear) ------------------------
# EvalValue keeps only `alpha`; EvalDeriv1 keeps only `inv_h` (the carrier
# arithmetic type is retained as `Talpha` so the kernel reconstructs `one(Talpha)`
# without storing `alpha`); every higher derivative of a degree-1 polynomial is
# zero → a zero-size payload. The per-op payload type is chosen once in
# `_axis_anchor_type`; `_resolve_anchor` and the fused/fullbuffer kernels then
# dispatch on it.
struct _LinearValuePayload{Talpha} <: _AbstractAnchorPayload
    alpha::Talpha
end
struct _LinearDeriv1Payload{Talpha, Tinv} <: _AbstractAnchorPayload
    inv_h::Tinv
end
struct _LinearZeroPayload{Talpha} <: _AbstractAnchorPayload end

@inline _linear_payload_type(::EvalValue, ::Type{Tα}, ::Type{Tinv}) where {Tα, Tinv} = _LinearValuePayload{Tα}
@inline _linear_payload_type(::EvalDeriv1, ::Type{Tα}, ::Type{Tinv}) where {Tα, Tinv} = _LinearDeriv1Payload{Tα, Tinv}
@inline _linear_payload_type(::AbstractEvalOp, ::Type{Tα}, ::Type{Tinv}) where {Tα, Tinv} = _LinearZeroPayload{Tα}

# Op-minimal gridded combine: the fused/fullbuffer kernels pass the whole anchor
# so each op reads only the field its payload carries. Results match the
# point-wise 1D `_linear_kernel(op, yL, yR, inv_h, α)` exactly (value blend /
# slope / carrier zero); the carrier `one(Tα)` is reconstructed from the payload
# type parameter rather than stored.
@inline _linear_kernel(::EvalValue, yL::Tv, yR::Tv, a::_AxisAnchor{<:Any, <:_LinearValuePayload}) where {Tv} =
    _linear_value_blend(a.alpha, yL, yR)

@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, a::_AxisAnchor{<:Any, _LinearDeriv1Payload{Tα, Tinv}}) where {Tv, Tα, Tinv}
    # `_fielddiff` types the VALUE difference `yR - yL` (dimension Y, e.g. m), NOT
    # the slope output (Y/X) — the `* inv_h` supplies the 1/X. Feeding `Tinv` to
    # `_coeff_op` mistook the reciprocal spacing for the spacing (→ Y·X = m·s) and
    # threw on unit grids; recover the value-diff witness so Real stays Float-widened.
    Tg = _promote_eltype(_inv_op, Tinv)
    Tc = _promote_eltype(_interp_op, Tg, Tv, Tg)
    return _fielddiff(Tc, yR, yL) * a.inv_h * one(Tα)
end

@inline _linear_kernel(::AbstractEvalOp, yL::Tv, yR::Tv, a::_AxisAnchor{<:Any, _LinearZeroPayload{Tα}}) where {Tv, Tα} =
    (0 * yL + 0 * yR) * one(Tα)

# One method covers ordinary AND exclusive-periodic Linear: the interval type
# selector carries the right-tap distinction, so only the op-minimal payload
# type varies.
@inline function _axis_anchor_type(
        m::LinearInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tinv = _promote_eltype(_inv_op, Tw)
    # α is the in-cell fraction `(q - L)·inv_h` — dimensionless for unit grids
    # (s·s⁻¹). A raw `promote_type` of the coordinate/inverse types collapses to
    # an abstract `Quantity{Float64}` (Real: still Float64); the op-witness keeps
    # it the concrete dimensionless carrier the payload converts α into.
    Tα = _promote_eltype(_alpha_of, eltype(targets), eltype(grid), Tinv)
    return _AxisAnchor{_interval_type(grid), _linear_payload_type(op, Tα, Tinv)}
end

# EvalValue: only the in-cell weight `alpha` (Clamp/Fill fold it into [0, 1]).
@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{I, _LinearValuePayload{Tα}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tα}
    inv_h = _get_inv_h(grid, idxL, xL, xR)
    α = Tα(_alpha_of(xq, xL, inv_h))
    if extrap isa Union{ClampExtrap, FillExtrap}
        α = clamp(α, zero(Tα), one(Tα))
    end
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LinearValuePayload{Tα}}(interval, _LinearValuePayload{Tα}(α))
end

# EvalDeriv1: only the reciprocal width. Clamp/Fill needs no fold — the boundary
# cell's slope is exactly this cell's `inv_h` (the coordinate already resolved
# to the boundary cell during search).
@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{I, _LinearDeriv1Payload{Tα, Tinv}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tα, Tinv}
    inv_h = Tinv(_get_inv_h(grid, idxL, xL, xR))
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LinearDeriv1Payload{Tα, Tinv}}(interval, _LinearDeriv1Payload{Tα, Tinv}(inv_h))
end

# EvalDeriv2 and higher: the derivative is zero everywhere → no stored geometry.
@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{I, _LinearZeroPayload{Tα}}},
        grid::AbstractVector,
        idxL::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {I, Tα}
    interval = _interval_indices(grid, idxL, idxR)
    return _AxisAnchor{I, _LinearZeroPayload{Tα}}(interval, _LinearZeroPayload{Tα}())
end

# `op` selects the op-minimal payload (value → alpha, deriv1 → inv_h, higher →
# zero-size); it defaults to EvalValue for the resize case and direct/test use.
function _gridded_anchors(
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        op::AbstractEvalOp = EvalValue()
    )
    return _gridded_anchors(grid, targets, extrap, dim, op, _gridded_alpha_type(grid, targets))
end

function _gridded_anchors(
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    m = _linear_gridded_anchor_method()
    A = _axis_anchor_type(m, grid, targets, op, Tvals)
    anchors = Vector{A}(undef, length(targets))
    return _axis_anchors_loop!(anchors, m, grid, targets, extrap, dim, _gridded_build_searcher(grid, targets))
end

function _gridded_anchors_pooled(
        pool,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    return _axis_anchors_pooled(pool, _linear_gridded_anchor_method(), grid, targets, extrap, dim, op, Tvals)
end

function _gridded_anchors_pooled(
        pool,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        op::AbstractEvalOp = EvalValue()
    )
    return _gridded_anchors_pooled(pool, grid, targets, extrap, dim, op, _gridded_alpha_type(grid, targets))
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
# blend / slope / carrier-zero exactly as point-wise ND eval does. The left/
# right taps come straight from the anchor's interval (`a.idxL`/`a.idxR`) — the
# ordinary cell derives the right tap, the exclusive-periodic seam stores it.

function _fused_corner_expr(N::Int, d::Int, idxs::Vector{Any})
    d > N && return Expr(:ref, :A, idxs...)
    lo = copy(idxs)
    lo[d] = Symbol(:il_, d)
    hi = copy(idxs)
    hi[d] = Symbol(:ir_, d)
    # The op-minimal anchor supplies whichever geometry `ops[d]` needs (alpha for
    # value, inv_h for deriv1, nothing for higher) — passed whole so the kernel
    # reads only the field its payload carries.
    return :(
        _linear_kernel(
            ops[$d],
            $(_fused_corner_expr(N, d + 1, lo)),
            $(_fused_corner_expr(N, d + 1, hi)),
            $(Symbol(:a_, d)),
        )
    )
end

@generated function _gridded_fused!(
        out::AbstractArray{<:Any, N},
        A::AbstractArray{<:Any, N},
        anchors::NTuple{N, Vector},
        ops::Tuple{Vararg{AbstractEvalOp, N}}
    ) where {N}
    js = [Symbol(:j_, d) for d in 1:N]
    body = :(out[$(js...)] = $(_fused_corner_expr(N, 1, Any[:_ for _ in 1:N])))
    for d in 1:N   # d = 1 built first → innermost loop
        a = Symbol(:a_, d)
        body = quote
            for $(js[d]) in eachindex(anchors[$d])
                $a = anchors[$d][$(js[d])]
                $(Symbol(:il_, d)) = $a.idxL
                $(Symbol(:ir_, d)) = $a.idxR
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
    anchors::NTuple{N, Vector}
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
        anchors::Vector,
        op::AbstractEvalOp,
        ::Val{D},
        ranges::NTuple{N, UnitRange{Int}}
    ) where {N, D}
    js = [Symbol(:j_, d) for d in 1:N]
    lo = Any[js[d] for d in 1:N]
    lo[D] = :il
    hi = copy(lo)
    hi[D] = :ir
    body = :(
        dest[$(js...)] = _linear_kernel(op, src[$(lo...)], src[$(hi...)], a)
    )
    for d in 1:N   # d = 1 built first → innermost loop
        body = if d == D
            quote
                for $(js[d]) in eachindex(anchors)
                    a = anchors[$(js[d])]
                    il = a.idxL
                    ir = a.idxR
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
    anchors::Vector,
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
        anchors::NTuple{N, Vector},
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
    1:_data_length(grid)
function _gridded_hull(grid::AbstractVector, targets::AbstractVector, ::AbstractExtrap)
    # separate minimum/maximum: each SIMD-vectorizes (~0.15 ns/elem measured);
    # Base.extrema's tuple-mapreduce is ~34x slower and was end-to-end visible
    lo = minimum(targets)
    hi = maximum(targets)
    searcher = _resolve_searcher_for_grid(grid, DEFAULT_SEARCHER)
    return _anchor_loc(grid, lo, false, searcher).idxL:(_anchor_loc(grid, hi, false, searcher).idxL + 1)
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
        anchors::NTuple{N, Vector},
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
        anchors::NTuple{N, Vector},
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

# ---- entry points ------------------------------------------------------------
# Output eltype: matches point-wise scalar eval's promotion (pinned by test).
# `Tg`/`Tv` are the value-matched grid float and value type; both the persistent
# functor and the one-shot front resolve them the same way before this call.
# Value eltype, then folded per-axis into derivative space (Y/Xᴺ) so a unit-grid
# gridded derivative sizes in ∂-units (e.g. W/s), not value units. Value ops
# (DerivOp{0}) leave it unchanged → the value fast path stays bit-identical.
@inline _gridded_out_eltype(::Type{Tg}, ::Type{Tv}, targets::Tuple, grids::Tuple, ops::Tuple) where {Tg, Tv} =
    _deriv_eltype_nd(_promote_eltype(_interp_op, Tg, Tv, promote_type(map(eltype, targets)...)), grids, ops)

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
        d -> _gridded_anchors_pooled(pool, grids[d], targets[d], extraps[d], d, ops[d], Tout),
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
        d -> _gridded_anchors_pooled(pool, grids[d], targets[d], extraps[d], d, ops[d], Tout),
        Val(N)
    )
    _gridded_apply!(out, grids, data, targets, anchors, ops, extraps, Tout)
    return _gridded_fill_oob!(out, grids, data, targets, extraps, ops)
end

# ---- gridded method dispatch (linear separable arm) --------------------------
# Prepared linear gridded evaluation: callers have already resolved/cached grids
# and extrapolation modes. This is the single linear kernel entry used by both
# one-shot `interp(..., gq; method = LinearInterp())` and persistent `itp(gq)`.
@inline function _gridded_eval_methods!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        targets::Tuple,
        ::Tuple{LinearInterp, Vararg{LinearInterp}},
        ops,
        extraps
    ) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    _gridded_eval!(out, grids, data, targets, ops, extraps, _gridded_out_eltype(Tg, Tv, targets, grids, ops))
    return true
end

# Allocating override: `_gridded_eval` builds anchors (firing NoExtrap's throw)
# BEFORE allocating the output, so a throwing query never leaves the full
# O(∏M_d) matrix behind (pinned in test_gridded_query.jl).
@inline function _gridded_alloc_itp(
        itp::LinearInterpolantND{Tg, Tv, N},
        gq::GriddedQuery,
        ops,
        extraps,
        _search,
        _hint
    ) where {Tg, Tv, N}
    return _gridded_eval(itp.grids, itp.data, gq.axes, ops, extraps, _gridded_out_eltype(Tg, Tv, gq.axes, itp.grids, ops))
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
@with_pool pool function _linear_gridded_oneshot!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        targets::Tuple,
        bcs::NTuple{N, AbstractBC},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        ::Type{Tg},
        ::Type{Tout}
    ) where {N, Tg, Tout}
    grids_c = map((g, bc) -> _linear_gridded_axis_pooled(pool, g, bc, Tg), grids, bcs)
    extraps_c = _resolve_extrap(extraps, bcs, grids_c, data, Val(N))
    _gridded_eval_methods!(
        out, grids_c, data, targets, ntuple(_ -> LinearInterp(NoBC()), Val(N)), ops, extraps_c
    )
    return out
end

@inline _linear_gridded_axis_pooled(pool, grid, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _cache_axis_pooled(pool, grid, Tg)
@inline _linear_gridded_axis_pooled(pool, grid, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} =
    _cache_axis(_cache_axis_pooled(pool, grid, Tg), bc)

"""
    linear_interp(grids, data, gq::GriddedQuery; bc, extrap, deriv)
    linear_interp!(out, grids, data, gq::GriddedQuery; bc, extrap, deriv)

One-shot N-D linear interpolation at every combination of `gq.axes` coordinates
(rectilinear resize), returning an N-D `size(gq)` array. Thin named spelling of
`interp(grids, data, gq; method = LinearInterp(bc))` — identical output (linear
is arithmetic, so its float-forcing eltype matches the generic path) and the
same separable fast path via the `_try_gridded_separable!` hook. `extrap` is the
per-axis policy (`NoExtrap`/`ClampExtrap`/`FillExtrap`/`ExtendExtrap`).
"""
function linear_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {N}
    return interp(grids, data, gq; method = _linear_gridded_methods(bc, Val(N)), extrap = extrap, deriv = deriv)
end

function linear_interp!(
        out::AbstractArray{<:Any, N},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {N}
    return interp!(out, grids, data, gq; method = _linear_gridded_methods(bc, Val(N)), extrap = extrap, deriv = deriv)
end

# N = 1 disambiguation: a flat vector IS the 1-D output array, but it also
# matches the batch-protocol entry `linear_interp!(output::AbstractVector,
# grids, data, queries)` — pin the intersection to the gridded arm.
function linear_interp!(
        out::AbstractVector,
        grids::Tuple{AbstractVector},
        data::AbstractVector,
        gq::GriddedQuery{<:Tuple{Any}};
        kwargs...
    )
    return linear_interp!(out, only(grids), data, only(gq.axes); _unwrap_nd_kwargs(values(kwargs))...)
end

# N = 1 Aqua disambiguation (allocating): a 1-axis `GriddedQuery` is an
# `AbstractVector`, so this gridded one-shot overlaps the 1-D batch one-shot
# `linear_interp(grids::Tuple{AbstractVector}, data::AbstractVector,
# q::AbstractVector{<:Real})`. Latent (a GriddedQuery never has Real elements);
# pin the intersection to the 1-D arm over the single axis.
function linear_interp(
        grids::Tuple{AbstractVector},
        data::AbstractVector,
        gq::GriddedQuery{<:Tuple{Any}};
        kwargs...
    )
    return linear_interp(only(grids), data, only(gq.axes); _unwrap_nd_kwargs(values(kwargs))...)
end

# `bc` (single or per-axis) → the per-axis `LinearInterp` method tuple `interp` wants.
@inline _linear_gridded_methods(bc::AbstractBC, ::Val{N}) where {N} = ntuple(_ -> LinearInterp(bc), Val(N))
@inline _linear_gridded_methods(bc::NTuple{N, AbstractBC}, ::Val{N}) where {N} = map(LinearInterp, bc)

@inline function _try_gridded_oneshot_methods!(
        out_nd::AbstractArray{<:Any, N},
        grids,
        data::AbstractArray{Tv, N},
        targets::Tuple,
        methods::Tuple{Vararg{LinearInterp, N}},
        extrap,
        deriv
    ) where {Tv, N}
    # Single resolution site for all linear-gridded one-shots — the named
    # `linear_interp`/`linear_interp!` forward here through `interp`. Owns the
    # arg resolution (was in `linear_interp!`) and calls the in-place core
    # directly, so the forward chain doesn't re-enter the named function.
    _validate_nd_grids(grids, data)
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv)
    bcs = map(m -> m.bc, methods)
    extraps = _resolve_extrap(extrap, bcs, Val(N), Tv)
    ops = _resolve_deriv_nd(deriv, Val(N))
    Tout = _gridded_out_eltype(Tg, Tv, targets, grids, ops)
    _linear_gridded_oneshot!(out_nd, grids, data, targets, bcs, ops, extraps, Tg, Tout)
    return true
end
