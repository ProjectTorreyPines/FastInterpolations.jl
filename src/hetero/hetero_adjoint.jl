# ========================================
# ND Heterogeneous Adjoint: Implementation
# ========================================
#
# Computes f̄ = Wᵀ · ȳ for N-dimensional heterogeneous interpolation.
#
# Pipeline (adj(y_bar) → f_bar):
#   Step 0: Scatter — y_bar → partials_bar (mixed-radix compact, per-axis weights)
#   Step 1: Build adjoint (d=N..1, reverse order, compact strides):
#           a. CubicInterp axes: moments_to_deriv adjoint + Thomas transpose solve + RHS adjoint
#           b. QuadraticInterp axes: recurrence adjoint + secant adjoint
#           c. LinearInterp/ConstantInterp axes: SKIP (sizes[d]=1, no derivative partial)
#   Step 2: Extract f_bar = partials_bar[1, ...]
#
# Dependencies (included before this file):
# - _NDAdjointAnchor (core/nd_adjoint_scatter.jl)
# - _HeteroAdjointAnchor weight functions (hetero_adjoint_types.jl)
# - _adjoint_axis_pair!, _adjoint_axis_pair_periodic! (cubic/nd/cubic_nd_adjoint.jl)
# - _adjoint_axis_pair_quadratic! (quadratic/nd/quadratic_nd_adjoint.jl)
# - _deriv_size, _deriv_sizes (hetero_build.jl)
# - _get_effective_bc, _get_effective_bc_quadratic (cubic_nd_build.jl, hetero_build.jl)
# - _get_cubic_cache (cubic_cache_pool.jl)

# ========================================
# Anchor Baking
# ========================================

"""
    _bake_hetero_nd_anchors(grids, spacings, queries, extraps, methods)

Precompute cell indices and per-axis 4-tuple weights for each query point.
Dispatches to per-method weight functions: cubic, quadratic, linear, or constant.

OOB handling per axis (baked at construction):
- `FillExtrap`: zero ALL weights (fill value independent of f)
- `ClampExtrap`: zero derivative weights w1-w3 (clamped value IS function of f)
"""
function _bake_hetero_nd_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}}
    ) where {N, Tg}
    return _bake_nd_anchors_generic(
        grids, spacings, queries, extraps,
        (d, t, h, inv_h, dL) -> _compute_hetero_anchor_weights(t, h, inv_h, dL, methods[d])
    )
end

# ========================================
# Mixed-Radix Scatter Codegen
# ========================================

"""
    _scatter_hetero_nd_codegen(N, sizes, idx_syms, w_syms) -> Vector{Expr}

Emit compact scatter accumulation statements using mixed-radix partial indexing.

For each partial index p ∈ 1:prod(sizes) and each corner c ∈ 0:2^N-1:
  partials_bar[p, idx₁+c₁, ..., idxₙ+cₙ] += yb * w₁ⱼ * w₂ⱼ * ... * wₙⱼ

Partial index uses column-major mixed-radix decomposition:
  p = 1 + d₁ + sizes[1]*(d₂ + sizes[2]*(d₃ + ...))
  where dₖ ∈ {0, ..., sizes[k]-1}

Per axis d:
- If sizes[d]=2 (derivative method): corner_d ∈ {0,1}, deriv_d ∈ {0,1}
  → weight index = 1 + corner_d + 2*deriv_d (4 entries)
- If sizes[d]=1 (non-derivative method): corner_d ∈ {0,1}, deriv_d = 0
  → weight index = 1 + corner_d (2 entries only)
"""
function _scatter_hetero_nd_codegen(N::Int, sizes::NTuple, idx_syms, w_syms)
    stmts = Expr[]
    NP = prod(sizes)  # compact partial count
    NC = 1 << N       # 2^N corners (spatial, always)

    for p_flat in 0:(NP - 1)
        # Decompose p_flat into per-axis derivative indices (mixed-radix)
        deriv_indices = Vector{Int}(undef, N)
        remainder = p_flat
        for d in 1:N
            deriv_indices[d] = remainder % sizes[d]
            remainder = remainder ÷ sizes[d]
        end

        for c in 0:(NC - 1)
            weight_factors = Symbol[]
            for d in 1:N
                corner_d = (c >> (d - 1)) & 1
                deriv_d = deriv_indices[d]
                # Weight index: 1 + corner + 2*deriv (same convention as _scatter_nd!)
                w_idx = 1 + corner_d + 2 * deriv_d
                push!(weight_factors, w_syms[d, w_idx])
            end
            # Product of per-axis weights
            wp_expr = weight_factors[1]
            for i in 2:length(weight_factors)
                wp_expr = :($wp_expr * $(weight_factors[i]))
            end
            # Corner offsets
            offsets = _corner_offset_expr(c, N)
            idx_exprs = [:($(idx_syms[d]) + $(offsets[d])) for d in 1:N]
            # Mixed-radix partial index (1-based)
            p_idx = p_flat + 1
            lhs = :(partials_bar[$p_idx, $(idx_exprs...)])
            push!(stmts, :($lhs += yb * $wp_expr))
        end
    end
    return stmts
end

# ========================================
# @generated Scatter Kernel
# ========================================

"""
    _scatter_hetero_nd!(partials_bar, yb, anchor, ops, methods)

@generated tensor-product scatter with mixed-radix partial indexing and per-axis DerivOp dispatch.
Uses compact partials_bar first dimension (prod(sizes) instead of 2^N).
"""
@inline @generated function _scatter_hetero_nd!(
        partials_bar::AbstractArray{Tv, NP1},
        yb,
        anchor::_NDAdjointAnchor{Tg, N},
        ops::OPS,
        ::M
    ) where {Tv, Tg, N, NP1, OPS <: NTuple{N, AbstractEvalOp}, M <: Tuple{Vararg{AbstractInterpMethod, N}}}
    sizes = _deriv_sizes(M)
    NP1_expected = N + 1
    NP1 == NP1_expected || error("NP1 must equal N+1, got NP1=$NP1, N=$N")

    stmts = Expr[]

    # Unpack indices
    idx_syms = ntuple(d -> Symbol("idx_", d), N)
    push!(stmts, :($(Expr(:tuple, idx_syms...)) = anchor.indices))

    # Per-axis: select weight field based on DerivOp order from type parameter
    w_field_names = [:w0, :w1, :w2, :w3]
    w_syms = Matrix{Symbol}(undef, N, 4)
    for d in 1:N
        k_d = deriv_order(fieldtype(OPS, d))
        w_syms[d, 1] = Symbol("w_", d, "_fL")
        w_syms[d, 2] = Symbol("w_", d, "_fR")
        w_syms[d, 3] = Symbol("w_", d, "_dyL")
        w_syms[d, 4] = Symbol("w_", d, "_dyR")
        if k_d > 3
            # DerivOp{4+}: zero kernel → zero adjoint contribution
            for j in 1:4
                push!(stmts, :($(w_syms[d, j]) = zero(Tg)))
            end
        else
            wf = w_field_names[k_d + 1]
            lhs = Expr(:tuple, w_syms[d, 1], w_syms[d, 2], w_syms[d, 3], w_syms[d, 4])
            push!(stmts, :($lhs = anchor.$wf[$d]))
        end
    end

    # Emit mixed-radix scatter
    append!(stmts, _scatter_hetero_nd_codegen(N, sizes, idx_syms, w_syms))

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
        return nothing
    end
end

# ========================================
# Scatter Dispatch Wrappers
# ========================================

@inline function _adjoint_scatter_hetero_nd!(partials_bar, anchors, y_bar::Real, ops, methods)
    return @inbounds _scatter_hetero_nd!(partials_bar, y_bar, anchors[1], ops, methods)
end

@inline function _adjoint_scatter_hetero_nd!(partials_bar, anchors, y_bar, ops, methods)
    return @inbounds for q in eachindex(y_bar)
        _scatter_hetero_nd!(partials_bar, y_bar[q], anchors[q], ops, methods)
    end
end

# ========================================
# Build Adjoint — Per-Method Function Barriers
# ========================================

# CubicInterp: dispatch to existing cubic adjoint axis pair (non-periodic or periodic)
@inline function _hetero_axis_adjoint!(
        src_3d, dst_3d,
        ::CubicInterp,
        p_src::Int,
        cache_d, mixed_cache_d,
        spacing_d::AbstractGridSpacing{Tg},
        bc_d, mixed_bc_d,
        grid_d::AbstractVector{Tg},
        shape_before::Int, n_d::Int, shape_after::Int,
        z_bar::AbstractVector{Tv},
        f_contrib::AbstractVector{Tv},
        dy_bar_slice::AbstractVector{Tv},
        q_t,   # periodic solve precomputation (or nothing)
        ::Tg   # mincurv_C (unused for cubic)
    ) where {Tv, Tg}
    is_periodic_d = _is_periodic_bc(bc_d)
    if is_periodic_d
        _adjoint_axis_pair_periodic!(
            src_3d, dst_3d, cache_d, spacing_d,
            shape_before, n_d, shape_after,
            z_bar, f_contrib, dy_bar_slice, q_t
        )
    elseif p_src == 1
        _adjoint_axis_pair!(
            src_3d, dst_3d, cache_d, spacing_d,
            bc_d, grid_d,
            shape_before, n_d, shape_after,
            z_bar, f_contrib, dy_bar_slice
        )
    else
        _adjoint_axis_pair!(
            src_3d, dst_3d, mixed_cache_d, spacing_d,
            mixed_bc_d, grid_d,
            shape_before, n_d, shape_after,
            z_bar, f_contrib, dy_bar_slice
        )
    end
    return nothing
end

# QuadraticInterp: dispatch to existing quadratic adjoint axis pair
@inline function _hetero_axis_adjoint!(
        src_3d, dst_3d,
        m::QuadraticInterp,
        p_src::Int,
        cache_d, mixed_cache_d,
        spacing_d::AbstractGridSpacing{Tg},
        bc_d, mixed_bc_d,
        grid_d::AbstractVector{Tg},
        shape_before::Int, n_d::Int, shape_after::Int,
        z_bar::AbstractVector{Tv},  # reused as s_bar (size n_d-1 needed, we use a view)
        f_contrib::AbstractVector{Tv},
        dy_bar_slice::AbstractVector{Tv},  # reused as d_bar_work
        ::Any,    # q_t (unused for quadratic)
        mincurv_C_d::Tg
    ) where {Tv, Tg}
    eff_bc = _get_effective_bc_quadratic(bc_d, p_src, grid_d)
    bc_q = _to_quadratic_bc_adjoint(eff_bc, Tg)

    # z_bar is size n_d — reuse first n_d-1 elements as s_bar
    s_bar = view(z_bar, 1:(n_d - 1))
    _adjoint_axis_pair_quadratic!(
        src_3d, dst_3d, spacing_d, bc_q, grid_d,
        shape_before, n_d, shape_after,
        s_bar, dy_bar_slice, f_contrib, mincurv_C_d
    )
    return nothing
end

# ========================================
# Build Adjoint — ND Reverse-Axis Iteration
# ========================================

"""
    _build_adjoint_nd_hetero!(partials_bar, adj, sizes, pool, ::Val{d})

Apply the adjoint of the ND heterogeneous build pipeline.
Processes axes in reverse order (d=N..1), using compact strides.

Uses `Val(d)` recursive dispatch so that `methods[d]`, `caches[d]`, etc.
are indexed at compile time — avoiding Union-type boxing from heterogeneous
tuple indexing with a runtime `d`.

Non-derivative axes (sizes[d]=1) are skipped entirely.
Derivative axes dispatch to per-method function barriers.
Work buffers are pool-acquired per-axis (exact `n_d` size required by inner functions).
"""
@inline function _build_adjoint_nd_hetero!(
        ::AbstractArray{Tv}, ::HeteroAdjointND{Tg, N},
        ::NTuple{N, Int}, pool, ::Val{0}
    ) where {Tv, Tg, N}
    return nothing
end

@inline function _build_adjoint_nd_hetero!(
        partials_bar::AbstractArray{Tv},
        adj::HeteroAdjointND{Tg, N},
        sizes::NTuple{N, Int},
        pool,
        ::Val{d}
    ) where {Tv, Tg, N, d}
    grid_size = adj.grid_size

    if sizes[d] != 1
        # Compact stride: product of sizes for all axes before d
        stride_d = d == 1 ? 1 : prod(sizes[1:(d - 1)])
        n_d = grid_size[d]
        method_d = adj.methods[d]   # d is compile-time → concrete type
        spacing_d = adj.spacings[d]

        # Compute reshape dimensions for axis d
        shape_before = 1
        for k in 1:(d - 1)
            shape_before *= grid_size[k]
        end
        shape_after = 1
        for k in (d + 1):N
            shape_after *= grid_size[k]
        end

        # Per-axis work buffers (exact n_d size required by inner functions)
        z_bar = acquire!(pool, Tv, n_d)
        f_contrib = acquire!(pool, Tv, n_d)
        dy_bar_slice = acquire!(pool, Tv, n_d)

        # Precompute q_t for periodic cubic axis
        q_t = nothing
        if method_d isa CubicInterp && _is_periodic_bc(adj.bcs[d])
            n_intervals = n_d - 1
            q_t = acquire!(pool, Tg, n_intervals)
            fill!(q_t, zero(Tg))
            @inbounds q_t[1] = one(Tg)
            @inbounds q_t[n_intervals] = one(Tg)
            _ldiv_tridiagonal_transpose!(q_t, adj.caches[d].thomas)
        end

        for p_src_offset in 0:(stride_d - 1)
            p_src = p_src_offset + 1
            p_dst = p_src_offset + stride_d + 1

            # N-dim views via selectdim, then reshape to 3D
            src_3d = reshape(selectdim(partials_bar, 1, p_src), shape_before, n_d, shape_after)
            dst_3d = reshape(selectdim(partials_bar, 1, p_dst), shape_before, n_d, shape_after)

            _hetero_axis_adjoint!(
                src_3d, dst_3d, method_d, p_src,
                adj.caches[d], adj.mixed_caches[d],
                spacing_d, adj.bcs[d], adj.mixed_bcs[d], adj.grids[d],
                shape_before, n_d, shape_after,
                z_bar, f_contrib, dy_bar_slice,
                q_t, adj.mincurv_Cs[d]
            )
        end
    end

    # Recurse to lower axis (d-1 → ... → 0 base case)
    _build_adjoint_nd_hetero!(partials_bar, adj, sizes, pool, Val(d - 1))
    return nothing
end

# ========================================
# Core Apply Pipeline
# ========================================

@with_pool pool function _hetero_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::HeteroAdjointND{Tg, N, M},
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N, M}
    sizes = map(_deriv_size, adj.methods)
    NP = prod(sizes)

    # Pool-allocate compact partials_bar
    partials_bar = zeros!(pool, Tv, NP, adj.grid_size...)

    # Step 0: Eval adjoint scatter (mixed-radix)
    _adjoint_scatter_hetero_nd!(partials_bar, adj.anchors, y_bar, ops, adj.methods)

    # Step 1: Build adjoint (reverse axis order, compact strides)
    # Uses Val(d) recursive dispatch for zero-alloc heterogeneous tuple indexing.
    # Pool is passed so each axis acquires exact-size work buffers (inner functions use length()).
    if NP > 1  # Skip build if all axes are non-derivative
        _build_adjoint_nd_hetero!(partials_bar, adj, sizes, pool, Val(N))
    end

    # Step 2: Extract f_bar = partials_bar[1, ...]
    src = selectdim(partials_bar, 1, 1)
    f_bar .+= src

    return f_bar
end

# ========================================
# Graceful-error helper
# ========================================

@noinline function _throw_hetero_adjoint_hermite_unsupported(methods)
    local_names = String[]
    for m in methods
        if m isa PchipInterp || m isa CardinalInterp || m isa AkimaInterp
            push!(local_names, string(typeof(m)))
        end
    end
    throw(
        ArgumentError(
            "hetero_adjoint / ND reverse-mode AD is not yet implemented for " *
                "Hermite family methods (found $(join(unique(local_names), ", "))). " *
                "This also affects `Zygote.gradient` / `ChainRulesCore.rrule` on " *
                "`HeteroInterpolantND` built with PCHIP / Cardinal / Akima axes. " *
                "Tracking: claudedocs/TODO/hermite_onthefly_integrate_and_nd_adjoint.md (Task 2). " *
                "Workarounds: (1) switch Hermite axes to `CubicInterp` for AD support; " *
                "(2) use `ForwardDiff.gradient` on a scalar-query closure, which works " *
                "through the forward OnTheFly path without needing an adjoint; " *
                "(3) for 1D, `pchip_adjoint(x, y, xq)` / `cardinal_adjoint(x, xq)` / " *
                "`akima_adjoint(x, y, xq)` are fully supported."
        )
    )
end

# ========================================
# Constructor
# ========================================

"""
    hetero_adjoint(grids::NTuple{N}, queries; methods, extrap=NoExtrap())

Construct an N-dimensional heterogeneous adjoint operator.

# Arguments
- `grids`: N-tuple of grid vectors, one per dimension
- `queries`: Query points (SoA tuple of vectors, AoS, single tuple, SVector, etc.)
- `methods`: Per-axis interpolation method tuple (e.g., `(CubicInterp(), LinearInterp())`)
- `extrap`: Extrapolation mode (single or per-axis tuple)

# Returns
`HeteroAdjointND` operator that can be called as `adj(y_bar)` or `adj(f_bar, y_bar)`.

# Example
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 15)
xq = rand(100)
yq = rand(100)

adj = hetero_adjoint((x, y), (xq, yq); methods=(CubicInterp(), LinearInterp()))
f_bar = adj(y_bar)   # returns 20×15 matrix
```
"""
function hetero_adjoint(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector, Vararg{AbstractVector}};
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(queries) == N || _throw_ndims_mismatch("query vectors", N, length(queries))
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    # 5-arg `_resolve_extrap` (no BC): expand + promote + per-axis 2-arg materialize.
    extraps = _resolve_extrap(extrap, nothing, grids_typed, Val(N), Tg)
    return _build_hetero_nd_adjoint(grids_typed, queries, methods, extraps)
end

# Single-tuple query: hetero_adjoint((x, y), (0.5, 0.5); methods=...)
function hetero_adjoint(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}};
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    return hetero_adjoint(grids, (query,); methods = methods, extrap = extrap)
end

# Single-vector query: hetero_adjoint((x, y), SVector(0.5, 0.5); methods=...)
function hetero_adjoint(
        grids::NTuple{N, AbstractVector},
        query::AbstractVector{<:Real};
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return hetero_adjoint(grids, (query_tuple,); methods = methods, extrap = extrap)
end

# Generic query fallback
function hetero_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    _query_check_ndims(queries, Val(N))
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)
    # 5-arg `_resolve_extrap` (no BC): expand + promote + per-axis 2-arg materialize.
    extraps = _resolve_extrap(extrap, nothing, grids_typed, Val(N), Tg)
    return _build_hetero_nd_adjoint(grids_typed, queries, methods, extraps)
end

# ========================================
# Internal Builder
# ========================================

"""
    _build_hetero_nd_adjoint(grids, queries, methods, extraps)

Internal builder for `HeteroAdjointND`. Separated from the public API so that
`Tg` is bound via the argument type, making the return type fully inferrable.

Per-axis cache construction:
- CubicInterp: CubicSplineCache (user BC) + CubicSplineCache (mixed-partial BC)
- QuadraticInterp: compute MinCurvFit constant, no cache needed
- LinearInterp/ConstantInterp: nothing (no build step)
"""
function _build_hetero_nd_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    # Reject Hermite family methods upfront with a clear message. Without this
    # guard, the scatter path below would fail deep in the per-axis weight
    # computation with an obscure MethodError. Single point of entry for both
    # direct `hetero_adjoint(...)` calls and the Zygote/ChainRules rrule
    # (see `_adjoint_func_from_itp(::HeteroInterpolantND) = hetero_adjoint`).
    _has_any_local_method(methods) && _throw_hetero_adjoint_hermite_unsupported(methods)
    # Validate grid size and cubic PolyFit BC support requirements
    @inbounds for d in 1:N
        len_d = length(grids[d])
        len_d >= 2 || _throw_adjoint_grid_too_small(d, len_d)
        if methods[d] isa CubicInterp
            bc_d = methods[d].bc
            if !(bc_d isa PeriodicBC)
                deg = get_polyfit_degree(bc_d)
                if deg > 0 && len_d < deg + 1
                    throw(
                        ArgumentError(
                            "PolyFit BC on dimension $d requires at least $(deg + 1) grid points, got $len_d"
                        )
                    )
                end
            end
        end
    end

    # Extend exclusive periodic grids → inclusive form (cubic axes only)
    grids_ext = map(grids, methods) do grid_d, method_d
        method_d isa CubicInterp || return grid_d
        bc_d = method_d.bc
        bc_d isa PeriodicBC{:exclusive} || return grid_d
        period = _resolve_exclusive_period(grid_d, bc_d)
        x_end = first(grid_d) + Tg(period)
        if grid_d isa AbstractRange
            _to_float_adding_endpoint(grid_d, Tg)
        else
            vcat(grid_d, x_end)
        end
    end

    spacings = _create_spacings_typed(grids_ext)

    # Per-axis BC normalization (for derivative methods only)
    # Non-BC methods (Linear/Constant) use `nothing` sentinel — never matched
    # against PeriodicBC or BCPair in protocol functions.
    bcs = map(methods) do method_d
        if method_d isa CubicInterp
            bc_d = method_d.bc
            _is_periodic_bc(bc_d) ? bc_d : _normalize_bc(bc_d)
        elseif method_d isa QuadraticInterp
            bc_d = method_d.bc
            _normalize_bc(bc_d)
        else
            nothing
        end
    end

    # Mixed-partial BCs (for derivative methods, p_src > 1)
    mixed_bcs = map(methods, grids_ext) do method_d, grid_d
        if method_d isa CubicInterp
            mixed_bc = _get_effective_bc(method_d.bc, 2, grid_d)
            _is_periodic_bc(mixed_bc) ? mixed_bc : _normalize_bc(mixed_bc)
        elseif method_d isa QuadraticInterp
            mixed_bc = _get_effective_bc_quadratic(method_d.bc, 2, grid_d)
            _normalize_bc(mixed_bc)
        else
            nothing
        end
    end

    # Per-axis caches (cubic only)
    # _effective_autocache: bypass cache pool for Dual grids (ephemeral, hit rate ≈ 0%)
    ac = _effective_autocache(true, Tg)
    caches = map(methods, grids_ext, bcs) do method_d, grid_d, bp_d
        method_d isa CubicInterp || return nothing
        if _is_periodic_bc(bp_d)
            _get_cubic_cache(grid_d, PeriodicBC(), ac)
        else
            _get_cubic_cache(grid_d, bp_d, ac)
        end
    end

    mixed_caches = map(methods, grids_ext, mixed_bcs) do method_d, grid_d, mbp_d
        method_d isa CubicInterp || return nothing
        if _is_periodic_bc(mbp_d)
            _get_cubic_cache(grid_d, PeriodicBC(), ac)
        else
            _get_cubic_cache(grid_d, mbp_d, ac)
        end
    end

    # Per-axis MinCurvFit constants (quadratic only)
    mincurv_Cs = map(methods, spacings, grids_ext) do method_d, spacing_d, grid_d
        if method_d isa QuadraticInterp
            bc_d = method_d.bc
            _mincurv_C_for_bc(bc_d, spacing_d, length(grid_d))
        else
            zero(Tg)
        end
    end

    # Bake per-query anchors
    anchors = _bake_hetero_nd_anchors(grids_ext, spacings, queries, extraps, methods)

    grid_size = ntuple(d -> length(grids_ext[d]), Val(N))

    return HeteroAdjointND{
        Tg, N,
        typeof(methods), typeof(grids_ext), typeof(spacings),
        typeof(caches), typeof(mixed_caches),
        typeof(bcs), typeof(mixed_bcs),
    }(
        methods, grids_ext, spacings,
        caches, mixed_caches, bcs, mixed_bcs,
        anchors, grid_size, mincurv_Cs
    )
end
