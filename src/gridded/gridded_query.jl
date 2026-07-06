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

# Output eltype: matches point-wise scalar eval's promotion (pinned by test).
@inline _gridded_out_eltype(::LinearInterpolantND{Tg, Tv, 2}, tx, ty) where {Tg, Tv} =
    _promote_eltype(_interp_op, Tg, Tv, promote_type(eltype(tx), eltype(ty)))

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
    p1 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[1], tx, itp.extraps[1], 1)
    p2 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[2], ty, itp.extraps[2], 2)
    Tout = _gridded_out_eltype(itp, tx, ty)

    B = Matrix{Tout}(undef, n1, N)                      # dim2-resolved (float mid)
    @inbounds for j in 1:N
        a = p2.anchors[j]
        jl = a.idx; jr = jl + 1
        for r in 1:n1
            B[r, j] = _eval_anchor(a, A[r, jl], A[r, jr])
        end
    end

    C = Matrix{Tout}(undef, M, N)                       # dim1-resolved (final)
    @inbounds for j in 1:N
        for i in 1:M
            a = p1.anchors[i]
            il = a.idx
            C[i, j] = _eval_anchor(a, B[il, j], B[il + 1, j])
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
