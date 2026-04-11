# ========================================
# Shared ND Adjoint Scatter Infrastructure
# ========================================
#
# Generic types and functions reused by all ND adjoint operators
# (cubic, quadratic, and future interpolant types).
#
# Contains:
#   - _NDAdjointAnchor{Tg, N}    (pre-baked per-query weights)
#   - _scatter_nd_codegen          (codegen helper for 4^N scatter)
#   - _scatter_nd!                 (@generated tensor-product scatter)
#   - _adjoint_scatter_nd!         (scalar/vector dispatch wrapper)

# ========================================
# Per-Query Anchor (Baked Weights)
# ========================================

"""
    _NDAdjointAnchor{Tg, N}

Precomputed per-query data for ND adjoint scatter.
Stores cell location and per-axis basis weights for all derivative orders.

All 4 weight sets are pre-baked at construction time. The adjoint is constructed
once and called repeatedly with different `ȳ` vectors, so pre-computing is
optimal (pay once, reuse many times).

# Fields
- `indices`: Per-axis cell index (left node of containing interval)
- `w0`: EvalValue weights per axis — 4-tuple per axis
- `w1`: EvalDeriv1 weights per axis — 4-tuple per axis
- `w2`: EvalDeriv2 weights per axis — 4-tuple per axis
- `w3`: EvalDeriv3 weights per axis — 4-tuple per axis

# Weight Convention
For each axis d:
- Each `wk[d]` is a 4-tuple of weights for the k-th derivative
- Forward: `dᵏP/dxᵏ = Σ_j wk_j · stencil_j` (tensor-product per axis)
- Cubic uses all 4 entries: `(w_fL, w_fR, w_dyL, w_dyR)`
- Quadratic uses 3 entries: `(w_fL, w_fR, w_dfL, 0)` — 4th always zero
"""
struct _NDAdjointAnchor{Tg, N}
    indices::NTuple{N, Int}
    w0::NTuple{N, NTuple{4, Tg}}
    w1::NTuple{N, NTuple{4, Tg}}
    w2::NTuple{N, NTuple{4, Tg}}
    w3::NTuple{N, NTuple{4, Tg}}
end

# ========================================
# Codegen Helper
# ========================================

"""
    _scatter_nd_codegen(N, idx_syms, w_syms) -> Vector{Expr}

Emit 4^N scatter accumulation statements from weight symbols.
Each statement: `partials_bar[p+1, idx₁+c₁, ..., idxₙ+cₙ] += yb * w₁ⱼ * w₂ⱼ * ... * wₙⱼ`

Weight indexing per axis d:
  index = 1 + corner_d + 2*deriv_d
  (corner=0,deriv=0) → 1, (1,0) → 2, (0,1) → 3, (1,1) → 4
"""
function _scatter_nd_codegen(N, idx_syms, w_syms)
    stmts = Expr[]
    NP = 1 << N
    NC = 1 << N
    for p in 0:(NP - 1)
        for c in 0:(NC - 1)
            weight_factors = Symbol[]
            for d in 1:N
                corner_d = (c >> (d - 1)) & 1
                deriv_d = (p >> (d - 1)) & 1
                w_idx = 1 + corner_d + 2 * deriv_d
                push!(weight_factors, w_syms[d, w_idx])
            end
            wp_expr = weight_factors[1]
            for i in 2:length(weight_factors)
                wp_expr = :($wp_expr * $(weight_factors[i]))
            end
            offsets = _corner_offset_expr(c, N)
            idx_exprs = [:($(idx_syms[d]) + $(offsets[d])) for d in 1:N]
            p_idx = _partial_index(p)
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
    _scatter_nd!(partials_bar, yb, anchor, ops)

@generated tensor-product scatter with per-axis DerivOp dispatch.
Selects `anchor.w0[d]`, `w1[d]`, `w2[d]`, or `w3[d]` per axis at compile time
based on the concrete DerivOp types in `ops`.
"""
@inline @generated function _scatter_nd!(
        partials_bar::AbstractArray{Tv, NP1},
        yb,
        anchor::_NDAdjointAnchor{Tg, N},
        ops::OPS
    ) where {Tv, Tg, N, NP1, OPS <: NTuple{N, AbstractEvalOp}}
    NP1 == N + 1 || error("NP1 must equal N+1, got NP1=$NP1, N=$N")

    stmts = Expr[]

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

    append!(stmts, _scatter_nd_codegen(N, idx_syms, w_syms))

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
        return nothing
    end
end

# ========================================
# Dispatch Wrappers (scalar / vector y_bar)
# ========================================

@inline function _adjoint_scatter_nd!(partials_bar, anchors, y_bar::Real, ops)
    return @inbounds _scatter_nd!(partials_bar, y_bar, anchors[1], ops)
end

@inline function _adjoint_scatter_nd!(partials_bar, anchors, y_bar, ops)
    return @inbounds for q in eachindex(y_bar)
        _scatter_nd!(partials_bar, y_bar[q], anchors[q], ops)
    end
end

# ========================================
# Generic ND Anchor Baking
# ========================================
# Shared skeleton for cubic, quadratic, and hetero ND adjoint anchor construction.
# The only per-type difference is the weight computation function.

"""
    _bake_nd_anchors_generic(grids, spacings, queries, extraps, weight_fn)

Generic anchor baking for ND adjoint operators that use `_NDAdjointAnchor{Tg, N}`.
Shared by cubic, quadratic, and hetero adjoint types.

`weight_fn(d, t, h, inv_h, dL)` must return `(w0, w1, w2, w3)` — four 4-tuple weight
sets for derivative orders 0–3.

OOB handling baked into weights at construction time:
- `FillExtrap`: zeros ALL 4 weight tuples (fill value independent of f)
- `ClampExtrap`: zeros w1-w3 only (clamped value IS function of f)
"""
function _bake_nd_anchors_generic(
        grids::NTuple{N, AbstractVector{Tg}},
        spacings::NTuple{N, AbstractGridSpacing{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        weight_fn
    ) where {N, Tg}
    nq = _query_length(queries)
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps)

    anchors = Vector{_NDAdjointAnchor{Tg, N}}(undef, nq)
    @inbounds for q in 1:nq
        query_q = _extract_query_point(queries, q, Val(N))
        idx_and_weights = ntuple(Val(N)) do d
            xq_raw = Tg(query_q[d])
            xq_d = _extrap_axis(xq_raw, grids[d], extraps[d])
            idx, xL, _ = search_interval(DEFAULT_SEARCHER, grids[d], spacings[d], xq_d)
            h = _get_h(spacings[d], idx)
            inv_h = _get_inv_h(spacings[d], idx)
            t = (xq_d - xL) * inv_h
            dL = xq_d - xL
            is_oob = xq_raw < first(grids[d]) || xq_raw > last(grids[d])
            return (idx, weight_fn(d, t, h, inv_h, dL), is_oob)
        end
        indices = ntuple(d -> idx_and_weights[d][1], Val(N))
        w0 = ntuple(d -> idx_and_weights[d][2][1], Val(N))
        w1 = ntuple(d -> idx_and_weights[d][2][2], Val(N))
        w2 = ntuple(d -> idx_and_weights[d][2][3], Val(N))
        w3 = ntuple(d -> idx_and_weights[d][2][4], Val(N))

        # Per-axis OOB weight fixup
        for d in 1:N
            idx_and_weights[d][3] || continue  # skip in-bounds axes
            ext_d = extraps[d]
            zw = (zero(Tg), zero(Tg), zero(Tg), zero(Tg))
            if ext_d isa FillExtrap
                w0 = Base.setindex(w0, zw, d)
                w1 = Base.setindex(w1, zw, d)
                w2 = Base.setindex(w2, zw, d)
                w3 = Base.setindex(w3, zw, d)
            elseif ext_d isa ClampExtrap
                w1 = Base.setindex(w1, zw, d)
                w2 = Base.setindex(w2, zw, d)
                w3 = Base.setindex(w3, zw, d)
            end
        end

        anchors[q] = _NDAdjointAnchor{Tg, N}(indices, w0, w1, w2, w3)
    end
    return anchors
end
