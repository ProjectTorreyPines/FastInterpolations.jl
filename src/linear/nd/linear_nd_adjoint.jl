# ========================================
# ND Linear Adjoint: Implementation
# ========================================
#
# Computes f̄ = Wᵀ · ȳ for N-dimensional linear interpolation.
#
# Pipeline (adj(y_bar) → f_bar):
#   Step 0: Check for 2nd+ derivative → early zero return
#   Step 1: Scatter — y_bar → f_bar (2^N corners per query)
#
# No build adjoint step (unlike cubic): linear has no tridiagonal solve.

# ========================================
# Anchor Baking
# ========================================

"""
    _bake_linear_nd_anchors(grids, queries, extraps)

Precompute cell indices and linear interpolation weights for each query point.

Reads `_get_h(grids[d], idx)` / `_get_inv_h(grids[d], idx)` directly from the
wrapped axes (`_CachedRange` / `_CachedVector` / `_ExclusivePeriodicAxis`).
For `_ExclusivePeriodicAxis`, `search_interval` returns `idx_R = 1` for the
seam cell and `_get_h` returns the seam-cell width — periodic-aware bake
falls out of the wrapper-level dispatch with no method-specific branching.

Per-axis weights:
- `w_value = (1-α, α)` for EvalValue scatter
- `w_deriv = (-inv_h, inv_h)` for EvalDeriv1 scatter

OOB handling per axis (baked at construction):
- `FillExtrap`: zero both w_value and w_deriv (fill value independent of f)
- `ClampExtrap`: keep w_value (clamped boundary value IS function of f), zero w_deriv
"""
function _bake_linear_nd_anchors(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    nq = _query_length(queries)
    _query_validate(queries)
    _validate_nd_domain(grids, queries, extraps)

    anchors = Vector{_LinearNDAdjointAnchor{Tg, N}}(undef, nq)
    @inbounds for q in 1:nq
        query_q = _extract_query_point(queries, q, Val(N))
        idx_and_weights = ntuple(Val(N)) do d
            xq_raw = Tg(query_q[d])
            xq_d = _extrap_axis(xq_raw, grids[d], extraps[d])
            idx, _, xL, _ = search_interval(DEFAULT_SEARCHER, grids[d], xq_d)
            h = _get_h(grids[d], idx)
            inv_h = _get_inv_h(grids[d], idx)
            α = (xq_d - xL) * inv_h
            is_oob = xq_raw < first(grids[d]) || xq_raw > last(grids[d])
            w_val = (one(Tg) - α, α)
            w_der = (-inv_h, inv_h)
            return (idx, w_val, w_der, is_oob)
        end

        indices = ntuple(d -> idx_and_weights[d][1], Val(N))
        w_value = ntuple(d -> idx_and_weights[d][2], Val(N))
        w_deriv = ntuple(d -> idx_and_weights[d][3], Val(N))

        # Per-axis OOB weight fixup
        for d in 1:N
            idx_and_weights[d][4] || continue  # skip in-bounds axes
            ext_d = extraps[d]
            zw = (zero(Tg), zero(Tg))
            if ext_d isa FillExtrap
                w_value = Base.setindex(w_value, zw, d)
                w_deriv = Base.setindex(w_deriv, zw, d)
            elseif ext_d isa ClampExtrap
                w_deriv = Base.setindex(w_deriv, zw, d)
            end
        end

        anchors[q] = _LinearNDAdjointAnchor{Tg, N}(indices, w_value, w_deriv)
    end
    return anchors
end

# ========================================
# Scatter (@generated, any N)
# ========================================
#
# Adjoint of `_multilinear_sum` (linear_nd_eval.jl). For each query, scatters
# y_bar[q] into 2^N corners using tensor-product weight products.
#
# Weight selection per axis d:
#   EvalValue → anchor.w_value[d]
#   EvalDeriv1 → anchor.w_deriv[d]
#   (bit=0) → w[1], (bit=1) → w[2]

"""
    _scatter_linear_nd!(f_bar, yb, anchor, ops)

@generated tensor-product scatter for ND linear adjoint.
Selects `anchor.w_value[d]` or `anchor.w_deriv[d]` per axis at compile time
based on the concrete DerivOp types in `ops`.
"""
@inline @generated function _scatter_linear_nd!(
        f_bar::AbstractArray{Tv, N},
        yb,
        anchor::_LinearNDAdjointAnchor{Tg, N},
        ops::OPS
    ) where {Tv, Tg, N, OPS <: NTuple{N, AbstractEvalOp}}
    num_corners = 1 << N  # 2^N

    stmts = Expr[]

    # Unpack indices
    idx_syms = ntuple(d -> Symbol("idx_", d), N)
    push!(stmts, :($(Expr(:tuple, idx_syms...)) = anchor.indices))

    # Per-axis: select weight field based on DerivOp order, unpack 2-tuple
    w_syms = Matrix{Symbol}(undef, N, 2)  # [d, bit+1]
    for d in 1:N
        k_d = deriv_order(fieldtype(OPS, d))
        wf = k_d == 0 ? :w_value : :w_deriv  # 0→value, 1→deriv (2/3 handled by early return)
        w_syms[d, 1] = Symbol("w_", d, "_L")
        w_syms[d, 2] = Symbol("w_", d, "_R")
        lhs = Expr(:tuple, w_syms[d, 1], w_syms[d, 2])
        push!(stmts, :($lhs = anchor.$wf[$d]))
    end

    # Generate 2^N scatter accumulations
    for corner in 0:(num_corners - 1)
        bits = ntuple(d -> (corner >> (d - 1)) & 1, N)

        # Index expression: idx_d + bit_d
        idx_exprs = [:($(idx_syms[d]) + $(bits[d])) for d in 1:N]

        # Weight expression: product of per-axis weights
        weight_exprs = [w_syms[d, bits[d] + 1] for d in 1:N]
        weight_expr = foldl((a, b) -> :($a * $b), weight_exprs)

        push!(stmts, :(f_bar[$(idx_exprs...)] += yb * $weight_expr))
    end

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
        return nothing
    end
end

# ========================================
# Core Apply Pipeline
# ========================================

# Scalar y_bar: single query, direct scatter (no eachindex on Real)
function _linear_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::LinearAdjointND{Tg, N},
        y_bar::Real,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
    _has_second_or_higher_derivative(ops, Val(N)) && return nothing
    @inbounds _scatter_linear_nd!(f_bar, y_bar, adj.anchors[1], ops)
    return nothing
end

# Vector/Tuple y_bar: loop over elements
function _linear_adjoint_nd_apply!(
        f_bar::AbstractArray{Tv, N},
        adj::LinearAdjointND{Tg, N},
        y_bar,
        ops::NTuple{N, AbstractEvalOp}
    ) where {Tv, Tg, N}
    # 2nd+ derivative → f_bar stays zero (already zero-filled by caller)
    _has_second_or_higher_derivative(ops, Val(N)) && return nothing

    @inbounds for q in eachindex(y_bar)
        _scatter_linear_nd!(f_bar, y_bar[q], adj.anchors[q], ops)
    end
    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    linear_adjoint(grids::NTuple{N}, queries; bc=NoBC(), extrap=NoExtrap())

Construct an N-dimensional linear adjoint operator.

# Arguments
- `grids`: N-tuple of grid vectors, one per dimension
- `queries`: Query points (SoA tuple of vectors, AoS, single tuple, SVector, etc.)
- `bc`: Boundary condition (single `AbstractBC` or per-axis tuple). Use
  `PeriodicBC()` for `:inclusive` periodicity (closed grid with `f[1] == f[end]`)
  or `PeriodicBC(endpoint=:exclusive, period=...)` for `:exclusive` periodicity
  (half-open grid). Default `NoBC()`.
- `extrap`: Extrapolation mode (single or per-axis tuple). Auto-promoted to
  `WrapExtrap` on periodic axes.

# Returns
`LinearAdjointND` operator that can be called as `adj(y_bar)` or `adj(f_bar, y_bar)`.
For `:exclusive` periodic axes, the output `f_bar` shape matches the user-supplied
grid (length `n` along that axis), with the seam contribution folded into index 1.

# Example
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 15)
xq = rand(100)
yq = rand(100)

adj = linear_adjoint((x, y), (xq, yq))
f_bar = adj(y_bar)   # returns 20×15 matrix

# Periodic example (:exclusive on axis 1)
x_excl = collect(range(0.0, step = 1.0/20, length = 20))   # half-open
adj_p = linear_adjoint((x_excl, y), (xq, yq);
    bc = (PeriodicBC(endpoint = :exclusive, period = 1.0), NoBC()))
```
"""
function linear_adjoint(
        grids::NTuple{N, AbstractVector},
        queries::Tuple{AbstractVector, Vararg{AbstractVector}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(queries) == N || _throw_ndims_mismatch("query vectors", N, length(queries))
    return _linear_adjoint_dispatch(grids, queries, bc, extrap)
end

# Single-tuple query: linear_adjoint((x, y), (0.5, 0.5); ...)
function linear_adjoint(
        grids::NTuple{N, AbstractVector},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    return linear_adjoint(grids, (query,); bc = bc, extrap = extrap)
end

# Single-vector query: linear_adjoint((x, y), SVector(0.5, 0.5); ...)
function linear_adjoint(
        grids::NTuple{N, AbstractVector},
        query::AbstractVector{<:Real};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    length(query) == N || _throw_ndims_mismatch("query elements", N, length(query))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return linear_adjoint(grids, (query_tuple,); bc = bc, extrap = extrap)
end

# Generic query fallback
function linear_adjoint(
        grids::NTuple{N, AbstractVector},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        _extra...
    ) where {N}
    _query_check_ndims(queries, Val(N))
    return _linear_adjoint_dispatch(grids, queries, bc, extrap)
end

# Shared resolution path (called by every public overload above).
@inline function _linear_adjoint_dispatch(
        grids::NTuple{N, AbstractVector}, queries, bc, extrap
    ) where {N}
    Tg = _promote_grid_eltype(grids)
    Tg = float(Tg)
    grids_typed = _convert_grids_typed(grids, Tg)

    bcs = _resolve_bcs_nd(bc, Val(N))
    # BC-aware axis wrap: `:exclusive` → `_ExclusivePeriodicAxis(length+1 logical)`.
    # The wrapper exposes virtual seam point `inner[1] + period` for searches
    # near `xq >= inner[end]`, returning `idx_R = 1` so the scatter writes the
    # seam contribution to index 1 directly.
    #
    # Use the 2-arg `_cache_axis` (no Tg closure) — Tg promotion was already
    # applied by `_convert_grids_typed`, and per-element dispatch via `map`
    # gives concrete return types for `@inferred` stability. Mirrors the
    # forward `LinearInterpolantND` pattern at `linear_nd_interpolant.jl:82`.
    grids_typed = map(_cache_axis, grids_typed, bcs)

    # 5-arg `_resolve_extrap` with bcs: validates extrap/BC compat, auto-
    # promotes `WrapExtrap` on periodic axes, materializes per-axis.
    extraps = _resolve_extrap(extrap, bcs, grids_typed, Val(N), Tg)
    return _build_linear_nd_adjoint(grids_typed, queries, bcs, extraps)
end

"""
    _build_linear_nd_adjoint(grids, queries, bcs, extraps)

Internal builder for `LinearAdjointND`. Separated from the public API so that
`Tg` is bound via the argument type, making the return type fully inferrable.

`grids` here are already wrapped via `_cache_axis(g, bc, Tg)`. For `:exclusive`
periodic axes, `length(grids[d]) == n + 1` (virtual seam point is included
in the wrapper's logical length), and `grid_size` reflects that internal size.
The ND adjoint protocol's `_adjoint_output_size` and `_adjoint_apply_exclusive_nd!`
trim back to `n` for the user-visible output.
"""
function _build_linear_nd_adjoint(
        grids::NTuple{N, AbstractVector{Tg}},
        queries,
        bcs::NTuple{N, AbstractBC},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N, Tg}
    # Validate all axes have at least 2 points (scatter writes to idx and idx+1).
    # For wrapped axes this counts the virtual seam too — `_ExclusivePeriodicAxis`
    # exposes `length(inner) + 1`, so a 1-point user input would still fail here.
    @inbounds for d in 1:N
        length(grids[d]) >= 2 || _throw_adjoint_grid_too_small(d, length(grids[d]))
    end

    anchors = _bake_linear_nd_anchors(grids, queries, extraps)
    grid_size = ntuple(d -> length(grids[d]), Val(N))

    return LinearAdjointND(grids, bcs, extraps, anchors, grid_size)
end

# Matrix materialization inherited from AbstractAdjointND (nd_adjoint_protocol.jl)
