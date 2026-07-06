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
# Why a dedicated path — SEPARABILITY:
# Point-by-point evaluation recomputes each axis's search + weight redundantly
# across the whole grid, and reads 2^N corners per output point. A gridded query
# resolves each axis's anchors (interval index + weight) ONCE per axis and reuses
# them across the grid, building the result axis-by-axis. Each pass is a 1D
# interpolation that reuses the shared 1D kernel — the classic separable resize,
# here generalized to arbitrary rectilinear grids (the target vectors need not be
# ranges) and, in time, to any tensor-product method (linear/cubic/quadratic).
#
# This first cut: LINEAR, 2D. It is built from FI internals so the method kernel
# is a drop-in swap point later:
#   * `_anchor_loc`        — OOB-safe interval location (shared by all methods)
#   * `_alpha_of`          — normalized in-cell position (the linear weight)
#   * `_linear_value_blend`— the shared 1D linear value kernel
# Swapping the anchor stencil (2→4 tap) and the kernel gives cubic, unchanged
# structure.

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

# ---- per-axis anchor resolution -------------------------------------------
# ClampExtrap freezes an OOB query at the boundary node — clamp the weight to
# [0,1] (the `_anchor_loc` boundary interval + α∈[0,1] reproduces the clamp).
# ExtendExtrap keeps the (out-of-range) weight → linear extrapolation.
@inline _resolve_alpha(α, ::ClampExtrap) = clamp(α, zero(α), one(α))
@inline _resolve_alpha(α, ::ExtendExtrap) = α
@inline _resolve_alpha(α, ::AbstractExtrap) = α

"""
    _axis_anchors(g, t, ex, [searcher]) -> (idx::Vector{Int}, α::Vector)

Resolve every target coordinate in `t` against grid `g` under extrap `ex` to a
linear anchor `(idx, α)` with right node `idx+1`. Computed once per axis; reused
across the whole grid.
"""
function _axis_anchors(
        g::AbstractVector, t::AbstractVector, ex::AbstractExtrap,
        searcher::Searcher = DEFAULT_SEARCHER
    )
    M = length(t)
    Tα = float(promote_type(eltype(g), eltype(t)))
    idxs = Vector{Int}(undef, M)
    αs = Vector{Tα}(undef, M)
    @inbounds for k in 1:M
        loc = _anchor_loc(g, t[k], false, searcher)
        α = _alpha_of(loc.xq, loc.xL, loc.xR, g)
        idxs[k] = loc.idx
        αs[k] = _resolve_alpha(Tα(α), ex)
    end
    return idxs, αs
end

# ---- 2D linear separable core ---------------------------------------------
# pass 1 resizes dim2 (columns): contiguous-column AXPY, converts eltype once.
# pass 2 resizes dim1 (rows): reuses the shared 1D kernel per output element.
# Intermediate `B` is kept in the promoted (float) eltype so the (optional) final
# narrowing happens once — matching a single-quantization tensor eval.
function _gridded_linear_2d(itp::LinearInterpolantND{Tg, Tv, 2}, tx, ty) where {Tg, Tv}
    A = itp.data
    n1 = size(A, 1)
    M = length(tx)
    N = length(ty)
    ix, αx = _axis_anchors(itp.grids[1], tx, itp.extraps[1])
    iy, αy = _axis_anchors(itp.grids[2], ty, itp.extraps[2])
    Tout = _promote_eltype(_interp_op, Tg, Tv, eltype(tx))

    B = Matrix{Tout}(undef, n1, N)                      # dim2-resized (float mid)
    @inbounds for j in 1:N
        jl = iy[j]; jr = jl + 1; a = αy[j]
        for r in 1:n1
            B[r, j] = _linear_value_blend(a, A[r, jl], A[r, jr])
        end
    end

    C = Matrix{Tout}(undef, M, N)                       # dim1-resized (final)
    @inbounds for j in 1:N
        for i in 1:M
            il = ix[i]; ir = il + 1; a = αx[i]
            C[i, j] = _linear_value_blend(a, B[il, j], B[ir, j])
        end
    end
    return C
end

# ---- dispatch --------------------------------------------------------------
"""
    (itp::LinearInterpolantND{Tg,Tv,2})(gq::GriddedQuery)

Evaluate a 2-D linear interpolant on a rectilinear grid of target coordinates,
returning an `(length(gq.axes[1]) × length(gq.axes[2]))` array. Separable: each
axis's weights are resolved once and reused across the grid.
"""
(itp::LinearInterpolantND{Tg, Tv, 2})(gq::GriddedQuery{<:Tuple{Any, Any}}) where {Tg, Tv} =
    _gridded_linear_2d(itp, gq.axes[1], gq.axes[2])
