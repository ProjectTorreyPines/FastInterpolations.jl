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
function _pass_blend_dim2!(dest::AbstractMatrix, src::AbstractMatrix, plan::_AbstractAxisPlan)
    n1 = size(src, 1)
    @inbounds for j in 1:length(plan)
        a = _axis_anchor_at(plan, j)
        jl = a.idx; jr = jl + 1
        α = a.alpha
        if iszero(α)                       # exact node hit → memcpy column jl
            copyto!(view(dest, :, j), view(src, :, jl))
        elseif isone(α)                    # exact right node → memcpy column jr
            copyto!(view(dest, :, j), view(src, :, jr))
        else
            for r in 1:n1
                dest[r, j] = _eval_anchor(a, src[r, jl], src[r, jr])
            end
        end
    end
    return dest
end

# Gather resolving axis 1: j outer keeps src columns hot; the two taps
# (idx, idx+1) are memory-adjacent → load-pair on ARM, no hardware gather needed.
function _pass_gather_dim1!(dest::AbstractMatrix, src::AbstractMatrix, plan::_AbstractAxisPlan)
    @inbounds for j in 1:size(src, 2)
        for i in 1:length(plan)
            a = _axis_anchor_at(plan, i)
            il = a.idx
            dest[i, j] = _eval_anchor(a, src[il, j], src[il + 1, j])
        end
    end
    return dest
end

# Fused anchored tensor core: no O(M·N)-proportional intermediate. Straight-line
# 3-blend per output (2 axis-2 blends + 1 axis-1 blend from the 4 corners) —
# search/weight construction is amortized to the per-axis plans, and the fixed
# branch-free body lets LLVM vectorize the whole inner loop. A same-cell
# run-reuse variant (share row_lo/row_hi across outputs in one source cell) was
# measured 26–61% SLOWER at every ratio: the data-dependent run scan plus
# tiny variable-length SIMD loops cost more than the two recomputed blends.
function _pass_fused_linear_2d!(dest::AbstractMatrix, src::AbstractMatrix, p1::_AxisPlan, p2::_AxisPlan)
    ax = p1.anchors
    ay = p2.anchors
    @inbounds for j in eachindex(ay)
        a2 = ay[j]
        jl = a2.idx
        jr = jl + 1
        @simd for i in eachindex(ax)
            a1 = ax[i]
            il = a1.idx
            row_lo = _eval_anchor(a2, src[il, jl], src[il, jr])
            row_hi = _eval_anchor(a2, src[il + 1, jl], src[il + 1, jr])
            dest[i, j] = _eval_anchor(a1, row_lo, row_hi)
        end
    end
    return dest
end

# ---- 2D linear separable core ---------------------------------------------
# Each pass resolves one axis's anchors once and reuses them across the grid.
# NO automatic strategy/order selection lives here: callers name the strategy
# entry point explicitly, and two-pass entries take the fold order as an
# explicit argument — pure kernel behavior, no selector overhead. (The measured
# strategy map, fixed 512×384 sweep: fused wins ALL pure downsampling,
# two-pass wins same-size/upsampling/mixed; pass-order/orientation seeds from
# the retired auto models: blend ≈ 0.2, gather ≈ 0.65 ns/elem.)
# Plan builder: both axis plans, pooled, built BEFORE any output allocation —
# this is where NoExtrap's O(M)/O(N) domain validation fires
# (`_guard_axis_state`), so the throw happens before the caller's O(M·N)
# buffer is allocated.
@inline function _gridded_plans_pooled(pool, itp::LinearInterpolantND{Tg, Tv, 2}, tx, ty) where {Tg, Tv}
    m = LinearInterp()
    op = EvalValue()
    p1 = _axis_anchors_pooled(pool, m, op, itp.grids[1], tx, itp.extraps[1], 1)
    p2 = _axis_anchors_pooled(pool, m, op, itp.grids[2], ty, itp.extraps[2], 2)
    return p1, p2
end

# Pooled two-pass core. `@with_pool` scopes the intermediate; the pass kernels
# stay pool-agnostic (hetero `_axis_window_pooled` convention). The
# intermediate is kept in `Tmid` (the promoted eltype, matching the
# allocating path's `out` — NOT necessarily `eltype(out)`, since a caller may
# pass a narrower `out` in the in-place form) so narrowing into `out` happens
# exactly once (single-quantization contract). Plans are ALREADY built (and
# thus already validated) by the caller — this core never touches
# `itp.grids`/`itp.extraps` directly, so it never revalidates or reallocates
# a plan.
# `dim2_first` is the caller's EXPLICIT fold order (no internal cost model):
# true → blend dim2 first (n1×N mid), false → gather dim1 first (M×n2 mid).
# Both orders are mathematically equivalent (machine-eps, not bit-identical).
@with_pool pool function _gridded_apply_fullbuffer!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        p1::_AxisPlan,
        p2::_AxisPlan,
        ::Type{Tmid},
        dim2_first::Bool
    ) where {Tg, Tv, Tmid}
    A = itp.data
    n1, n2 = size(A)
    M = size(out, 1)
    N = size(out, 2)
    size(out) == (length(p1), length(p2)) || throw(
        DimensionMismatch("output size $(size(out)) != query size ($(length(p1)), $(length(p2)))")
    )
    (M == 0 || N == 0) && return out
    if p1.identity && p2.identity
        copyto!(out, A)                                   # M == n1, N == n2 by identity
    elseif p2.identity
        _pass_gather_dim1!(out, A, p1)                    # N == n2: single pass, no mid
    elseif p1.identity
        _pass_blend_dim2!(out, A, p2)                     # M == n1: single pass, no mid
    elseif dim2_first
        B = acquire!(pool, Tmid, n1, N)
        _pass_blend_dim2!(B, A, p2)
        _pass_gather_dim1!(out, B, p1)
    else
        B = acquire!(pool, Tmid, M, n2)
        _pass_gather_dim1!(B, A, p1)
        _pass_blend_dim2!(out, B, p2)
    end
    return out
end

function _gridded_apply_fused!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        p1::_AbstractAxisPlan,
        p2::_AbstractAxisPlan,
        ::Type{Tmid}
    ) where {Tg, Tv, Tmid}
    A = itp.data
    M = size(out, 1)
    N = size(out, 2)
    size(out) == (length(p1), length(p2)) || throw(
        DimensionMismatch("output size $(size(out)) != query size ($(length(p1)), $(length(p2)))")
    )
    (M == 0 || N == 0) && return out
    if _axis_identity(p1) && _axis_identity(p2)
        copyto!(out, A)
    elseif _axis_identity(p2)
        _pass_gather_dim1!(out, A, p1)
    elseif _axis_identity(p1)
        _pass_blend_dim2!(out, A, p2)
    else
        _pass_fused_linear_2d!(out, A, p1, p2)
    end
    return out
end

@with_pool pool function _gridded_apply_fused_pooled!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tmid}
    ) where {Tg, Tv, Tmid}
    p1, p2 = _gridded_plans_pooled(pool, itp, tx, ty)
    return _gridded_apply_fused!(out, itp, p1, p2, Tmid)
end

# Full-buffer strategy behind pooled plans: the anchor buffers come from the
# same pool scope that outlives the apply, so the whole candidate is
# allocation-free after pool warmup (the O(n1·N / M·n2) intermediate is pooled
# inside `_gridded_apply_fullbuffer!`'s own nested scope).
@with_pool pool function _gridded_apply_fullbuffer_pooled!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tmid},
        dim2_first::Bool
    ) where {Tg, Tv, Tmid}
    p1, p2 = _gridded_plans_pooled(pool, itp, tx, ty)
    return _gridded_apply_fullbuffer!(out, itp, p1, p2, Tmid, dim2_first)
end

# Public default strategy: FUSED anchors. Chosen for robustness, not a
# selector: flat ~0.7–1.6 ns/out across the whole ratio sweep, wins all pure
# downsampling outright, bounded ≤3× behind two-pass at strong upsampling,
# and needs no O(mid)/O(line) scratch — only the O(M+N) plans. Callers who
# know their shape call a strategy entry explicitly.
@with_pool pool function _gridded_apply!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tmid}
    ) where {Tg, Tv, Tmid}
    p1, p2 = _gridded_plans_pooled(pool, itp, tx, ty)
    return _gridded_apply_fused!(out, itp, p1, p2, Tmid)
end

@with_pool pool function _gridded_eval(
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tmid}
    ) where {Tg, Tv, Tmid}
    p1, p2 = _gridded_plans_pooled(pool, itp, tx, ty)
    out = Matrix{Tmid}(undef, length(tx), length(ty))
    return _gridded_apply_fused!(out, itp, p1, p2, Tmid)
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
    (itp::LinearInterpolantND{Tg,Tv,2})(out::AbstractMatrix, gq::GriddedQuery)

Evaluate a 2-D linear interpolant at every combination of `gq.axes`
coordinates. Each axis's anchors are resolved once (pooled, O(M+N)) and
reused across the grid; the default strategy is the FUSED anchored tensor
core (no intermediate buffer). The 2-arg form writes into `out` (FI batch
convention). Explicit strategy entries expose the alternatives with
caller-chosen order: `_gridded_apply_fused_pooled!` and
`_gridded_apply_fullbuffer_pooled!` here; the sliding-window and lazy-plan
candidates live in gridded_experimental.jl.
"""
function (itp::LinearInterpolantND{Tg, Tv, 2})(gq::GriddedQuery{<:Tuple{Any, Any}}) where {Tg, Tv}
    _gridded_extrap_guard(itp.extraps)
    tx, ty = gq.axes
    Tmid = _gridded_out_eltype(itp, tx, ty)
    return _gridded_eval(itp, tx, ty, Tmid)       # validates (NoExtrap) before output alloc
end
function (itp::LinearInterpolantND{Tg, Tv, 2})(
        out::AbstractMatrix,
        gq::GriddedQuery{<:Tuple{Any, Any}}
    ) where {Tg, Tv}
    _gridded_extrap_guard(itp.extraps)
    tx, ty = gq.axes
    Tmid = _gridded_out_eltype(itp, tx, ty)
    return _gridded_apply!(out, itp, tx, ty, Tmid) # validates (NoExtrap) before output writes
end
