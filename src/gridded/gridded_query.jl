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
# each axis's (interval index, weight) pair is resolved ONCE per axis — O(M+N)
# anchors — and reused across all O(M·N) outputs, instead of re-searching and
# re-weighting both axes at every output point.
#
# Two execution strategies consume the same anchors (measured on a fixed-source
# ratio sweep):
#   * FUSED      — straight 3-blend per output from the 4 corners, no
#                  intermediate buffer. Wins all pure downsampling.
#   * FULLBUFFER — classic separable two-pass through one pooled intermediate.
#                  Wins same-size / upsampling / mixed (contiguous passes).

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
# The retained per-axis resolution artifact: interval index + linear weight,
# with extrapolation already folded into the weight. 16 B isbits; one Vector
# per query axis is the whole precomputation.
struct _GriddedAnchor{T <: Real}
    idx::Int    # left node index (right node = idx + 1)
    alpha::T    # normalized in-cell weight; may leave [0, 1] under ExtendExtrap
end

"""
    _gridded_anchors!(anchors, g, t, ex, dim) -> anchors

Resolve every target coordinate in `t` against grid `g` once — the O(M) work
that both strategies reuse O(M·N) times. Extrapolation is folded here:
`ClampExtrap` clamps the weight to the boundary node, `ExtendExtrap` keeps the
out-of-range weight (linear extrapolation), and `NoExtrap` throws on the first
out-of-domain coordinate — BEFORE any O(M·N) buffer exists. `dim` names the
axis in that error.
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
        loc = _anchor_loc(g, xq, false, searcher)
        if ex isa NoExtrap && loc.state != IN_DOMAIN
            throw(DomainError(xq, "GriddedQuery axis $dim: coordinate outside grid domain (NoExtrap)"))
        end
        α = T(_alpha_of(loc.xq, loc.xL, loc.xR, g))
        if ex isa ClampExtrap
            α = clamp(α, zero(T), one(T))
        end
        anchors[k] = _GriddedAnchor{T}(loc.idx, α)
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
# the axis-d blend wraps the two axis-d corners and axes d+1..N are collapsed
# inside — for N = 2 this is exactly lo/hi = axis-2 blends, then the axis-1
# blend.
function _fused_corner_expr(N::Int, d::Int, idxs::Vector{Any})
    d > N && return Expr(:ref, :A, idxs...)
    lo = copy(idxs)
    lo[d] = Symbol(:il_, d)
    hi = copy(idxs)
    hi[d] = :($(Symbol(:il_, d)) + 1)
    return :(
        _linear_value_blend(
            $(Symbol(:alpha_, d)),
            $(_fused_corner_expr(N, d + 1, lo)),
            $(_fused_corner_expr(N, d + 1, hi)),
        )
    )
end

@generated function _gridded_fused!(
        out::AbstractArray{<:Any, N},
        A::AbstractArray{<:Any, N},
        anchors::NTuple{N, Vector{<:_GriddedAnchor}}
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
                $body
            end
        end
    end
    return quote
        @inbounds $body
        return out
    end
end

# ---- FULLBUFFER strategy ----------------------------------------------------
# Classic separable two-pass; each pass is a 1D resolve of one axis.

# Contiguous column blend resolving axis 2: stride-1 in both src and dest.
function _gridded_blend_dim2!(dest::AbstractMatrix, src::AbstractMatrix, ay::Vector{<:_GriddedAnchor})
    n1 = size(src, 1)
    @inbounds for j in eachindex(ay)
        a = ay[j]
        jl = a.idx
        jr = jl + 1
        for r in 1:n1
            dest[r, j] = _linear_value_blend(a.alpha, src[r, jl], src[r, jr])
        end
    end
    return dest
end

# Gather resolving axis 1: j outer keeps src columns hot; the two taps
# (idx, idx+1) are memory-adjacent loads.
function _gridded_gather_dim1!(dest::AbstractMatrix, src::AbstractMatrix, ax::Vector{<:_GriddedAnchor})
    @inbounds for j in 1:size(src, 2)
        for i in eachindex(ax)
            a = ax[i]
            il = a.idx
            dest[i, j] = _linear_value_blend(a.alpha, src[il, j], src[il + 1, j])
        end
    end
    return dest
end

# Two-pass core through one pooled intermediate held in `Tmid` (the promoted
# output eltype — narrowing into a caller-supplied narrower `out` happens
# exactly once). `dim2_first` picks the fold order: true → blend dim2 first
# (n1×N mid), false → gather dim1 first (M×n2 mid). Both orders are
# mathematically equivalent (machine-eps, not bit-identical).
@with_pool pool function _gridded_fullbuffer!(
        out::AbstractMatrix,
        A::AbstractMatrix,
        ax::Vector{<:_GriddedAnchor},
        ay::Vector{<:_GriddedAnchor},
        ::Type{Tmid},
        dim2_first::Bool
    ) where {Tmid}
    if dim2_first
        B = acquire!(pool, Tmid, size(A, 1), length(ay))
        _gridded_blend_dim2!(B, A, ay)
        _gridded_gather_dim1!(out, B, ax)
    else
        B = acquire!(pool, Tmid, length(ax), size(A, 2))
        _gridded_gather_dim1!(B, A, ax)
        _gridded_blend_dim2!(out, B, ay)
    end
    return out
end

# ---- strategy selection ------------------------------------------------------
# Measured map (fixed-source ratio sweep): fused wins all pure downsampling
# (its per-output work shrinks with the output), fullbuffer wins same-size /
# upsampling / mixed (contiguous passes amortize per-node work). Fold order
# from measured per-element pass costs: contiguous blend ≈ 0.2 ns,
# gather ≈ 0.65 ns.
function _gridded_apply!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        ax::Vector{<:_GriddedAnchor},
        ay::Vector{<:_GriddedAnchor},
        ::Type{Tmid}
    ) where {Tg, Tv, Tmid}
    A = itp.data
    n1, n2 = size(A)
    M = length(ax)
    N = length(ay)
    size(out) == (M, N) || throw(
        DimensionMismatch("output size $(size(out)) != query size ($M, $N)")
    )
    (M == 0 || N == 0) && return out
    if M < n1 && N < n2
        _gridded_fused!(out, A, (ax, ay))
    else
        dim2_first = 0.2 * (n1 * N) + 0.65 * (M * N) <= 0.65 * (M * n2) + 0.2 * (M * N)
        _gridded_fullbuffer!(out, A, ax, ay, Tmid, dim2_first)
    end
    return out
end

# ---- entry points ------------------------------------------------------------
# Output eltype: matches point-wise scalar eval's promotion (pinned by test).
@inline _gridded_out_eltype(::LinearInterpolantND{Tg, Tv, 2}, tx, ty) where {Tg, Tv} =
    _promote_eltype(_interp_op, Tg, Tv, promote_type(eltype(tx), eltype(ty)))

# Wrap needs seam-aware anchors; Fill needs per-axis OOB masks — both are
# roadmap items, rejected loudly rather than silently mis-evaluated.
@inline function _gridded_check_extraps(extraps::Tuple)
    foreach(extraps) do ex
        ex isa Union{ClampExtrap, ExtendExtrap, NoExtrap} || throw(
            ArgumentError(
                "$(typeof(ex)) is not yet supported for GriddedQuery " *
                    "(supported: ClampExtrap, ExtendExtrap, NoExtrap)"
            )
        )
    end
    return nothing
end

# Anchors are built (firing NoExtrap's O(M+N) validation) BEFORE the O(M·N)
# output is allocated.
@with_pool pool function _gridded_eval(
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tout}
    ) where {Tg, Tv, Tout}
    ax = _gridded_anchors_pooled(pool, itp.grids[1], tx, itp.extraps[1], 1)
    ay = _gridded_anchors_pooled(pool, itp.grids[2], ty, itp.extraps[2], 2)
    out = Matrix{Tout}(undef, length(tx), length(ty))
    return _gridded_apply!(out, itp, ax, ay, Tout)
end

@with_pool pool function _gridded_eval!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tout}
    ) where {Tg, Tv, Tout}
    ax = _gridded_anchors_pooled(pool, itp.grids[1], tx, itp.extraps[1], 1)
    ay = _gridded_anchors_pooled(pool, itp.grids[2], ty, itp.extraps[2], 2)
    return _gridded_apply!(out, itp, ax, ay, Tout)
end

# ---- dispatch ----------------------------------------------------------------
"""
    (itp::LinearInterpolantND{Tg,Tv,2})(gq::GriddedQuery)
    (itp::LinearInterpolantND{Tg,Tv,2})(out::AbstractMatrix, gq::GriddedQuery)

Evaluate a 2-D linear interpolant at every combination of `gq.axes`
coordinates. Each axis's anchors are resolved once (pooled, O(M+N)) and
reused across the grid; the strategy is picked per call — fused (no
intermediate) for pure downsampling, two-pass full buffer otherwise. The
2-arg form writes into `out` (FI batch convention).
"""
function (itp::LinearInterpolantND{Tg, Tv, 2})(gq::GriddedQuery{<:Tuple{Any, Any}}) where {Tg, Tv}
    _gridded_check_extraps(itp.extraps)
    tx, ty = gq.axes
    return _gridded_eval(itp, tx, ty, _gridded_out_eltype(itp, tx, ty))
end
function (itp::LinearInterpolantND{Tg, Tv, 2})(
        out::AbstractMatrix,
        gq::GriddedQuery{<:Tuple{Any, Any}}
    ) where {Tg, Tv}
    _gridded_check_extraps(itp.extraps)
    tx, ty = gq.axes
    return _gridded_eval!(out, itp, tx, ty, _gridded_out_eltype(itp, tx, ty))
end
