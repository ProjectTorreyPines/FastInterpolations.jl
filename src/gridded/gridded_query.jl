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
    _gridded_anchors!(anchors, g, t, ex, dim) -> anchors

Resolve every target coordinate in `t` against grid `g` once — the O(M) work
that both strategies reuse O(∏M_d) times. Extrapolation is folded here:
`WrapExtrap` folds the coordinate into the domain, `ClampExtrap`/`FillExtrap`
clamp the weight to the boundary node (Fill's OOB slabs are overwritten by the
[`_gridded_fill_oob!`](@ref) post-pass), `ExtendExtrap` keeps the out-of-range
weight (linear extrapolation), and `NoExtrap` throws on the first
out-of-domain coordinate — BEFORE any output-sized buffer exists. `dim` names
the axis in that error.
"""
function _gridded_anchors!(
        anchors::Vector{_GriddedAnchor{T}},
        g::AbstractVector,
        t::AbstractVector,
        ex::AbstractExtrap,
        dim::Int
    ) where {T}
    length(anchors) == length(t) || throw(
        DimensionMismatch("anchor buffer length $(length(anchors)) != query length $(length(t))")
    )
    searcher = _resolve_searcher_for_grid(g, DEFAULT_SEARCHER)
    k = 0
    @inbounds for xq in t
        k += 1
        loc = _anchor_loc(g, xq, ex isa WrapExtrap, searcher)
        if ex isa NoExtrap && loc.state != IN_DOMAIN
            throw(DomainError(xq, "GriddedQuery axis $dim: coordinate outside grid domain (NoExtrap)"))
        end
        inv_h = _get_inv_h(g, loc.idx, loc.xL, loc.xR)
        α = T(_alpha_of(loc.xq, loc.xL, inv_h))
        if ex isa Union{ClampExtrap, FillExtrap}
            α = clamp(α, zero(T), one(T))
        end
        anchors[k] = _GriddedAnchor{T}(loc.idx, α, T(inv_h))
    end
    return anchors
end

# Weight type: float promotion of grid × query (Float32 grid+query stays Float32).
@inline _gridded_alpha_type(g::AbstractVector, t::AbstractVector) =
    float(promote_type(eltype(g), eltype(t)))

function _gridded_anchors(g::AbstractVector, t::AbstractVector, ex::AbstractExtrap, dim::Int)
    T = _gridded_alpha_type(g, t)
    return _gridded_anchors!(Vector{_GriddedAnchor{T}}(undef, length(t)), g, t, ex, dim)
end

function _gridded_anchors_pooled(pool, g::AbstractVector, t::AbstractVector, ex::AbstractExtrap, dim::Int)
    T = _gridded_alpha_type(g, t)
    return _gridded_anchors!(acquire!(pool, _GriddedAnchor{T}, length(t)), g, t, ex, dim)
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
# (gather).
@generated function _gridded_pass!(
        dest::AbstractArray{<:Any, N},
        src::AbstractArray{<:Any, N},
        anchors::Vector{<:_GriddedAnchor},
        op::AbstractEvalOp,
        ::Val{D}
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
                for $(js[d]) in axes(src, $d)
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

# Pass order: adjacent-exchange comparator on total cost with the measured
# per-element pass costs (gather along axis 1 ≈ 0.65 ns, contiguous
# otherwise ≈ 0.2 ns). Putting axis `d` before `e` is cheaper iff
#   c_d·M_d·(n_e − M_e) ≤ c_e·M_e·(n_d − M_d)
# (shrink-most-first, cost-weighted). For N = 2 this reduces exactly to the
# old dim2-first model.
@inline _gridded_pass_cost(d::Int) = d == 1 ? 0.65 : 0.2
@inline function _gridded_pass_before(d::Int, e::Int, nsz::Dims, msz::Dims)
    return _gridded_pass_cost(d) * msz[d] * (nsz[e] - msz[e]) <
        _gridded_pass_cost(e) * msz[e] * (nsz[d] - msz[d])
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
        ::Type{Tmid}
    ) where {N, Tmid}
    nsz = size(A)
    msz = map(length, anchors)
    order = acquire!(pool, Int, N)
    @inbounds for d in 1:N
        order[d] = d
    end
    @inbounds for k in 2:N   # insertion sort by the pairwise cost comparator
        j = k
        while j > 1 && _gridded_pass_before(order[j], order[j - 1], nsz, msz)
            order[j], order[j - 1] = order[j - 1], order[j]
            j -= 1
        end
    end
    sz = nsz
    src = A
    @inbounds for k in 1:N
        d = order[k]
        sz = Base.setindex(sz, msz[d], d)
        dest = k == N ? out : acquire!(pool, Tmid, sz...)
        _gridded_pass!(dest, src, anchors[d], ops[d], Val(d))
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
        itp::LinearInterpolantND{Tg, Tv, N},
        anchors::NTuple{N, Vector{<:_GriddedAnchor}},
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        ::Type{Tmid}
    ) where {Tg, Tv, N, Tmid}
    A = itp.data
    msz = map(length, anchors)
    size(out) == msz || throw(
        DimensionMismatch("output size $(size(out)) != query size $msz")
    )
    any(iszero, msz) && return out
    if all(map(<, msz, size(A)))
        _gridded_fused!(out, A, anchors, ops)
    else
        _gridded_fullbuffer!(out, A, anchors, ops, Tmid)
    end
    return out
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
        itp::LinearInterpolantND{Tg, Tv, N},
        ts::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}}
    ) where {Tg, Tv, N}
    _gridded_has_fill(itp.extraps) || return out
    fv = _gridded_first_fill_value(itp.extraps)
    zref = _sample_data(itp)
    for d in 1:N
        itp.extraps[d] isa FillExtrap || continue
        k = 0
        for xq in ts[d]
            k += 1
            if _oob_state(itp.grids[d], xq) != IN_DOMAIN
                fill!(selectdim(out, d, k), _fill_extrap_result(ops, fv, zref, xq))
            end
        end
    end
    return out
end

# ---- entry points ------------------------------------------------------------
# Output eltype: matches point-wise scalar eval's promotion (pinned by test).
@inline _gridded_out_eltype(::LinearInterpolantND{Tg, Tv, N}, ts::Tuple) where {Tg, Tv, N} =
    _promote_eltype(_interp_op, Tg, Tv, promote_type(map(eltype, ts)...))

# Anchors are built (firing NoExtrap's O(ΣM_d) validation) BEFORE the
# output-sized array is allocated.
@with_pool pool function _gridded_eval(
        itp::LinearInterpolantND{Tg, Tv, N},
        ts::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        ::Type{Tout}
    ) where {Tg, Tv, N, Tout}
    anchors = ntuple(
        d -> _gridded_anchors_pooled(pool, itp.grids[d], ts[d], itp.extraps[d], d),
        Val(N)
    )
    out = Array{Tout, N}(undef, map(length, ts))
    _gridded_apply!(out, itp, anchors, ops, Tout)
    return _gridded_fill_oob!(out, itp, ts, ops)
end

@with_pool pool function _gridded_eval!(
        out::AbstractArray{<:Any, N},
        itp::LinearInterpolantND{Tg, Tv, N},
        ts::Tuple,
        ops::Tuple{Vararg{AbstractEvalOp, N}},
        ::Type{Tout}
    ) where {Tg, Tv, N, Tout}
    anchors = ntuple(
        d -> _gridded_anchors_pooled(pool, itp.grids[d], ts[d], itp.extraps[d], d),
        Val(N)
    )
    _gridded_apply!(out, itp, anchors, ops, Tout)
    return _gridded_fill_oob!(out, itp, ts, ops)
end

# ---- dispatch ----------------------------------------------------------------
"""
    (itp::LinearInterpolantND{Tg,Tv,N})(gq::GriddedQuery; deriv = EvalValue())
    (itp::LinearInterpolantND{Tg,Tv,N})(out::AbstractArray, gq::GriddedQuery; deriv = EvalValue())

Evaluate an N-D linear interpolant at every combination of `gq.axes`
coordinates. Each axis's anchors are resolved once (pooled, O(ΣM_d)) and
reused across the grid; the strategy is picked per call — fused (no
intermediate) for pure downsampling, separable multi-pass full buffer
otherwise. `deriv` takes a per-axis op tuple or a single op, exactly like
point-wise ND eval. The 2-arg form writes into `out` (FI batch convention).
"""
function (itp::LinearInterpolantND{Tg, Tv, N})(
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {Tg, Tv, N}
    ts = gq.axes
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _gridded_eval(itp, ts, ops, _gridded_out_eltype(itp, ts))
end
function (itp::LinearInterpolantND{Tg, Tv, N})(
        out::AbstractArray{<:Any, N},
        gq::GriddedQuery{<:Tuple{Vararg{Any, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue()
    ) where {Tg, Tv, N}
    ts = gq.axes
    ops = _resolve_deriv_nd(deriv, Val(N))
    return _gridded_eval!(out, itp, ts, ops, _gridded_out_eltype(itp, ts))
end
