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

# ---- pass kernels (pool-agnostic pure functions; hetero convention) --------
# Contiguous column blend resolving axis 2: pure AXPY over stride-1 columns.
function _pass_blend_dim2!(dest::AbstractMatrix, src::AbstractMatrix, plan::_AxisAnchorBatch)
    n1 = size(src, 1)
    anchors = plan.anchors
    @inbounds for j in eachindex(anchors)
        a = anchors[j]
        jl = a.idx; jr = jl + 1
        for r in 1:n1
            dest[r, j] = _eval_anchor(a, src[r, jl], src[r, jr])
        end
    end
    return dest
end

# Gather resolving axis 1: j outer keeps src columns hot; the two taps
# (idx, idx+1) are memory-adjacent → load-pair on ARM, no hardware gather needed.
function _pass_gather_dim1!(dest::AbstractMatrix, src::AbstractMatrix, plan::_AxisAnchorBatch)
    anchors = plan.anchors
    @inbounds for j in 1:size(src, 2)
        for i in eachindex(anchors)
            a = anchors[i]
            il = a.idx
            dest[i, j] = _eval_anchor(a, src[il, j], src[il + 1, j])
        end
    end
    return dest
end

# ---- pass-order cost model --------------------------------------------------
# Measured per-element seeds (Apple Silicon, predecessor investigation); the
# benchmark suite re-calibrates. Order A = blend dim2 first (n1×N mid);
# order B = gather dim1 first (M×n2 mid).
const _GRIDDED_C_BLEND = 0.2
const _GRIDDED_C_GATHER = 0.65
@inline function _gridded_dim2_first(n1::Int, n2::Int, M::Int, N::Int)
    cost_a = n1 * N * _GRIDDED_C_BLEND + M * N * _GRIDDED_C_GATHER
    cost_b = M * n2 * _GRIDDED_C_GATHER + M * N * _GRIDDED_C_BLEND
    return cost_a <= cost_b
end

# ---- 2D linear separable core ---------------------------------------------
# Each pass resolves one axis's anchors once and reuses them across the grid.
# The cost model picks which axis to fold first — the intermediate's shape
# (n1×N vs M×n2) drives total work; both orders are mathematically equivalent.
function _gridded_linear_2d(itp::LinearInterpolantND{Tg, Tv, 2}, tx, ty) where {Tg, Tv}
    A = itp.data
    n1, n2 = size(A)
    M = length(tx)
    N = length(ty)
    p1 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[1], tx, itp.extraps[1], 1)
    p2 = _axis_anchors(LinearInterp(), EvalValue(), itp.grids[2], ty, itp.extraps[2], 2)
    Tout = _gridded_out_eltype(itp, tx, ty)
    C = Matrix{Tout}(undef, M, N)
    if _gridded_dim2_first(n1, n2, M, N)
        B = Matrix{Tout}(undef, n1, N)
        _pass_blend_dim2!(B, A, p2)
        _pass_gather_dim1!(C, B, p1)
    else
        B = Matrix{Tout}(undef, M, n2)
        _pass_gather_dim1!(B, A, p1)
        _pass_blend_dim2!(C, B, p2)
    end
    return C
end

# ---- supported-extrap gate ---------------------------------------------------
# Wrap needs seam-aware _IdxStencil{2} anchors; Fill needs per-axis OOB masks —
# both are roadmap items, rejected loudly rather than silently mis-evaluated.
@inline _gridded_extrap_guard1(::Union{ClampExtrap, ExtendExtrap, NoExtrap}) = nothing
@inline _gridded_extrap_guard1(ex::AbstractExtrap) = throw(
    ArgumentError(
        "$(typeof(ex)) is not yet supported for GriddedQuery " *
            "(supported: ClampExtrap, ExtendExtrap, NoExtrap)"
    )
)
@inline function _gridded_extrap_guard(extraps::Tuple)
    foreach(_gridded_extrap_guard1, extraps)
    return nothing
end

# ---- dispatch --------------------------------------------------------------
"""
    (itp::LinearInterpolantND{Tg,Tv,2})(gq::GriddedQuery)

Evaluate a 2-D linear interpolant on a rectilinear grid of target coordinates,
returning an `(length(gq.axes[1]) × length(gq.axes[2]))` array. Separable: each
axis's weights are resolved once and reused across the grid.
"""
function (itp::LinearInterpolantND{Tg, Tv, 2})(gq::GriddedQuery{<:Tuple{Any, Any}}) where {Tg, Tv}
    _gridded_extrap_guard(itp.extraps)
    return _gridded_linear_2d(itp, gq.axes[1], gq.axes[2])
end
