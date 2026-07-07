# ============================================================================
# GriddedQuery — EXPERIMENTAL strategy candidates (NOT on the public path)
# ============================================================================
#
# Alternative plan representations and pass strategies from the strategy
# investigation rounds, kept compilable for A/B benchmarking and diagnostics.
# The production path (axis_anchor.jl + gridded_query.jl) uses materialized
# pooled anchor plans with the FUSED kernel as the public default and the
# two-pass full-buffer as the explicit alternative.
#
# Measured verdicts (fixed 512×384 Float64 source ratio sweep, ns/out):
#
# * SLIDING WINDOW (`_gridded_apply_windowed!`): two cached resolved lines,
#   O(M or N) scratch. Never uniquely wins after the plain fused kernel —
#   fused takes all pure downsampling, full-buffer ties it at same/upsampling.
#   Remaining niche: very large outputs where full-buffer's intermediate
#   exceeds L2 (e.g. 2048×1536→1024×768 down2: window 1.18 vs fullbuf 1.37).
#
# * LAZY RANGE PLAN (`_LazyAxisPlan`): preserves the Range query type and
#   synthesizes anchors on demand through the full `_anchor_loc` path.
#   Catastrophic when the lazy axis sits in an inner loop (~9.5× vs
#   materialized: the anchor is recomputed once per OUTPUT instead of once
#   per axis element); cost-neutral for the outer axis (one anchor per
#   column). Register-affine synthesis (the fast_imresize trick) was also
#   probed and LOSES 24–64% to materialized loads except at tiny outputs
#   (−8% at 64×48, where the O(M+N) plan build is the visible cost).
#
# * GENERIC FUSED (`_pass_fused_linear_2d!` on `_AbstractAxisPlan`): the
#   plan-agnostic fallback the lazy entries dispatch to; the production
#   batch×batch specialization compiles to the same loop shape with direct
#   anchor loads.

# ---- lazy Range axis plan ---------------------------------------------------
struct _LazyAxisPlan{
        A <: _AbstractAxisAnchor,
        M <: AbstractInterpMethod,
        O <: AbstractEvalOp,
        G <: AbstractVector,
        T <: AbstractRange,
        E <: AbstractExtrap,
        S <: Searcher,
    } <: _AbstractAxisPlan
    method::M
    op::O
    grid::G
    targets::T
    extrap::E
    dim::Int
    searcher::S
    identity::Bool
end
Base.length(p::_LazyAxisPlan) = length(p.targets)

@inline _axis_identity(p::_LazyAxisPlan) = p.identity

@inline function _axis_anchor_at(p::_LazyAxisPlan{A}, k::Int) where {A}
    xq = @inbounds p.targets[k]
    loc = _anchor_loc(p.grid, xq, false, p.searcher)
    return _axis_anchor(p.method, p.op, loc, p.grid, p.extrap, _anchor_scalar_type(A))
end

@inline function _axis_range_identity(g::AbstractVector, t::AbstractRange)
    n = length(t)
    n == length(g) || return false
    n == 0 && return true
    first(t) == first(g) || return false
    last(t) == last(g) || return false
    n <= 1 && return true
    return (@inbounds t[2] - t[1]) == (@inbounds g[2] - g[1])
end

@inline _validate_lazy_axis(::AbstractExtrap, g, t, dim::Int, searcher::Searcher) = nothing
function _validate_lazy_axis(ex::NoExtrap, g, t, dim::Int, searcher::Searcher)
    @inbounds for tk in eachindex(t)
        loc = _anchor_loc(g, t[tk], false, searcher)
        _guard_axis_state(ex, loc.state, t[tk], dim)
    end
    return nothing
end

function _axis_plan(
        m::LinearInterp,
        op::EvalValue,
        g::AbstractVector,
        t::AbstractRange,
        ex::AbstractExtrap,
        dim::Int,
        searcher::Searcher = DEFAULT_SEARCHER
    )
    A = _axis_anchor_type(m, op, g, t)
    resolved = _resolve_searcher_for_grid(g, searcher)
    _validate_lazy_axis(ex, g, t, dim, resolved)
    return _LazyAxisPlan{A, typeof(m), typeof(op), typeof(g), typeof(t), typeof(ex), typeof(resolved)}(
        m, op, g, t, ex, dim, resolved, _axis_range_identity(g, t)
    )
end

function _axis_plan(
        m::LinearInterp,
        op::EvalValue,
        g::AbstractVector,
        t::AbstractVector,
        ex::AbstractExtrap,
        dim::Int,
        searcher::Searcher = DEFAULT_SEARCHER
    )
    return _axis_anchors(m, op, g, t, ex, dim, searcher)
end

function _axis_plan_pooled(
        pool,
        m::LinearInterp,
        op::EvalValue,
        g::AbstractVector,
        t::AbstractRange,
        ex::AbstractExtrap,
        dim::Int,
        searcher::Searcher = DEFAULT_SEARCHER
    )
    return _axis_plan(m, op, g, t, ex, dim, searcher)
end

function _axis_plan_pooled(
        pool,
        m::LinearInterp,
        op::EvalValue,
        g::AbstractVector,
        t::AbstractVector,
        ex::AbstractExtrap,
        dim::Int,
        searcher::Searcher = DEFAULT_SEARCHER
    )
    return _axis_anchors_pooled(pool, m, op, g, t, ex, dim, searcher)
end

@inline function _gridded_lazy_plans_pooled(pool, itp::LinearInterpolantND{Tg, Tv, 2}, tx, ty) where {Tg, Tv}
    m = LinearInterp()
    op = EvalValue()
    p1 = _axis_plan_pooled(pool, m, op, itp.grids[1], tx, itp.extraps[1], 1)
    p2 = _axis_plan_pooled(pool, m, op, itp.grids[2], ty, itp.extraps[2], 2)
    return p1, p2
end

# ---- generic fused core (plan-agnostic; lazy/mixed plans dispatch here) -----
function _pass_fused_linear_2d!(dest::AbstractMatrix, src::AbstractMatrix, p1::_AbstractAxisPlan, p2::_AbstractAxisPlan)
    M = length(p1)
    @inbounds for j in 1:length(p2)
        a2 = _axis_anchor_at(p2, j)
        jl = a2.idx
        jr = jl + 1
        for i in 1:M
            a1 = _axis_anchor_at(p1, i)
            il = a1.idx
            row_lo = _eval_anchor(a2, src[il, jl], src[il, jr])
            row_hi = _eval_anchor(a2, src[il + 1, jl], src[il + 1, jr])
            dest[i, j] = _eval_anchor(a1, row_lo, row_hi)
        end
    end
    return dest
end

@with_pool pool function _gridded_apply_fused_lazy_pooled!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tmid}
    ) where {Tg, Tv, Tmid}
    p1, p2 = _gridded_lazy_plans_pooled(pool, itp, tx, ty)
    return _gridded_apply_fused!(out, itp, p1, p2, Tmid)
end

# ---- sliding-window tensor cores ---------------------------------------------
# A source column is first resolved along axis 1 into a stride-1 temporary,
# and axis 2 then blends two cached columns into the stride-1 output column.
# The cache size is the linear tap count (2), so memory is O(M) instead of
# O(M*N), while both hot loops remain SIMD-friendly and use the same matched
# anchor kernel.
function _fill_axis1_column!(
        dest::AbstractVector,
        src::AbstractMatrix,
        p1::_AbstractAxisPlan,
        col::Int
    )
    if _axis_identity(p1)
        @inbounds @simd for i in 1:length(p1)
            dest[i] = src[i, col]
        end
    else
        @inbounds @simd for i in 1:length(p1)
            a1 = _axis_anchor_at(p1, i)
            il = a1.idx
            dest[i] = _eval_anchor(a1, src[il, col], src[il + 1, col])
        end
    end
    return dest
end

function _pass_sliding_dim2!(
        dest::AbstractMatrix,
        src::AbstractMatrix,
        p1::_AbstractAxisPlan,
        p2::_AbstractAxisPlan,
        win1::AbstractVector,
        win2::AbstractVector
    )
    M = length(p1)
    length(win1) == M || throw(DimensionMismatch("window length $(length(win1)) != axis-1 query length $M"))
    length(win2) == M || throw(DimensionMismatch("window length $(length(win2)) != axis-1 query length $M"))
    key1 = 0
    key2 = 0
    @inbounds for j in 1:length(p2)
        a2 = _axis_anchor_at(p2, j)
        jl = a2.idx
        jr = jl + 1

        if key1 == jl
            lo = win1
            lo_slot = 1
        elseif key2 == jl
            lo = win2
            lo_slot = 2
        else
            key1 = jl
            _fill_axis1_column!(win1, src, p1, jl)
            lo = win1
            lo_slot = 1
        end

        if key1 == jr
            hi = win1
        elseif key2 == jr
            hi = win2
        elseif lo_slot == 1
            key2 = jr
            _fill_axis1_column!(win2, src, p1, jr)
            hi = win2
        else
            key1 = jr
            _fill_axis1_column!(win1, src, p1, jr)
            hi = win1
        end

        @simd for i in 1:M
            dest[i, j] = _eval_anchor(a2, lo[i], hi[i])
        end
    end
    return dest
end

function _fill_axis2_row!(
        dest::AbstractVector,
        src::AbstractMatrix,
        p2::_AbstractAxisPlan,
        row::Int
    )
    if _axis_identity(p2)
        @inbounds @simd for j in 1:length(p2)
            dest[j] = src[row, j]
        end
    else
        @inbounds @simd for j in 1:length(p2)
            a2 = _axis_anchor_at(p2, j)
            jl = a2.idx
            dest[j] = _eval_anchor(a2, src[row, jl], src[row, jl + 1])
        end
    end
    return dest
end

function _pass_sliding_dim1!(
        dest::AbstractMatrix,
        src::AbstractMatrix,
        p1::_AbstractAxisPlan,
        p2::_AbstractAxisPlan,
        win1::AbstractVector,
        win2::AbstractVector
    )
    N = length(p2)
    length(win1) == N || throw(DimensionMismatch("window length $(length(win1)) != axis-2 query length $N"))
    length(win2) == N || throw(DimensionMismatch("window length $(length(win2)) != axis-2 query length $N"))
    key1 = 0
    key2 = 0
    @inbounds for i in 1:length(p1)
        a1 = _axis_anchor_at(p1, i)
        il = a1.idx
        ir = il + 1

        if key1 == il
            lo = win1
            lo_slot = 1
        elseif key2 == il
            lo = win2
            lo_slot = 2
        else
            key1 = il
            _fill_axis2_row!(win1, src, p2, il)
            lo = win1
            lo_slot = 1
        end

        if key1 == ir
            hi = win1
        elseif key2 == ir
            hi = win2
        elseif lo_slot == 1
            key2 = ir
            _fill_axis2_row!(win2, src, p2, ir)
            hi = win2
        else
            key1 = ir
            _fill_axis2_row!(win1, src, p2, ir)
            hi = win1
        end

        @simd for j in 1:N
            dest[i, j] = _eval_anchor(a1, lo[j], hi[j])
        end
    end
    return dest
end

# `slide_dim2` is the caller's EXPLICIT window orientation (no internal
# heuristic): true → cache axis-1-resolved COLUMNS (contiguous fills, O(M)
# scratch), false → cache axis-2-resolved ROWS (strided fills, O(N) scratch).
@with_pool pool function _gridded_apply_windowed!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        p1::_AbstractAxisPlan,
        p2::_AbstractAxisPlan,
        ::Type{Tmid},
        slide_dim2::Bool
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
    elseif slide_dim2
        win1 = acquire!(pool, Tmid, M)
        win2 = acquire!(pool, Tmid, M)
        _pass_sliding_dim2!(out, A, p1, p2, win1, win2)
    else
        win1 = acquire!(pool, Tmid, N)
        win2 = acquire!(pool, Tmid, N)
        _pass_sliding_dim1!(out, A, p1, p2, win1, win2)
    end
    return out
end

# Sliding-window strategy behind pooled plans.
@with_pool pool function _gridded_apply_windowed_pooled!(
        out::AbstractMatrix,
        itp::LinearInterpolantND{Tg, Tv, 2},
        tx,
        ty,
        ::Type{Tmid},
        slide_dim2::Bool
    ) where {Tg, Tv, Tmid}
    p1, p2 = _gridded_plans_pooled(pool, itp, tx, ty)
    return _gridded_apply_windowed!(out, itp, p1, p2, Tmid, slide_dim2)
end
