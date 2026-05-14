# ========================================
# Cubic Anchored Query
# ========================================
# Precomputed geometry weights for ultra-fast cubic spline evaluation
# at a fixed query point. Enables 2-4x speedup by eliminating interval
# search and geometry setup for repeated evaluations.
#
# Include order: ops.jl → ... → cubic_types.jl → cubic_anchor.jl → cubic_interpolant.jl
#
# ========================================
# _CubicAnchoredQuery Type (Internal)
# ========================================

"""
    _CubicAnchoredQuery{Tg, Tq}

Precomputed weights for ultra-fast cubic spline evaluation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `Tq`: Query type (Float or ForwardDiff.Dual for AD support)

# Fields
- `stencil`: `_IdxStencil{2}` carrying the corner pair `(idxL, idxR)`.
  `idxR == idxL + 1` for non-seam cells; `idxR == 1` at periodic-exclusive
  seam cells (wrap). Virtual properties `aq.idx` (= `idxL`), `aq.idxL`,
  and `aq.idxR` are exposed via `getproperty` for ergonomics.
- `xq`: Original query point (or wrapped value for periodic)
- `state`: Domain state (`IN_DOMAIN`, `OOB_LEFT`, or `OOB_RIGHT`)
- `w0`: Precomputed weights for value (wyL, wyR, wzL, wzR)
- `w1`: Precomputed weights for first derivative (wyL, wyR, wzL, wzR)
- `w2`: Precomputed weights for second derivative (wzL, wzR) - optimized, no y-weights
- `w3`: Precomputed weights for third derivative (wzL, wzR) - optimized, no y-weights

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:cubic))

itp1 = cubic_interp(x, sin.(2π .* x))
itp2 = cubic_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=DerivOp(1))  # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is 2-4x faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search and geometry setup.

# Memory Optimization
w2 and w3 store only (wzL, wzR) since second and third derivatives
depend only on z-values, not y-values. This reduces anchor size by 16 bytes
per query for Float64.

# AD Support
When `xq` is a `ForwardDiff.Dual`, the anchor preserves the Dual type
in both `xq` and weight fields, enabling automatic differentiation through
series interpolant evaluation.
"""
struct _CubicAnchoredQuery{Tg, Tq <: Real}
    # Corner-index stencil: `stencil[1]` is the left index (idxL), `stencil[2]`
    # is the right index (idxR). For non-periodic cells `idxR == idxL + 1`; for
    # periodic-exclusive seam cells `idxR == 1` (wrap). Mirrors `_LinearAnchoredQuery`.
    # Legacy `aq.idx` accessor is preserved via `getproperty` (= idxL).
    stencil::_IdxStencil{2}
    xq::Tq                     # query point (possibly wrapped), Float or Dual
    state::UInt8               # IN_DOMAIN / OOB_LEFT / OOB_RIGHT
    w0::NTuple{4, Tq}           # (wyL, wyR, wzL, wzR) for value
    w1::NTuple{4, Tq}           # (wyL, wyR, wzL, wzR) for first deriv
    w2::NTuple{2, Tq}           # (wzL, wzR) for second deriv - optimized
    w3::NTuple{2, Tq}           # (wzL, wzR) for third deriv - optimized
end

# Virtual property accessors — legacy `aq.idx` ergonomics + new `aq.idxL`/`aq.idxR`.
# Val-dispatch (rather than a single `getproperty` with `===` branches) forces
# per-property compile-time specialization so each lookup inlines to a single
# `getfield` (+ tuple index for the stencil paths). Avoids the union-typed
# return that would otherwise box weights/state in hot loops.
@inline Base.getproperty(aq::_CubicAnchoredQuery, s::Symbol) = _get_cub_prop(aq, Val(s))
@inline _get_cub_prop(aq::_CubicAnchoredQuery, ::Val{:idx}) = getfield(aq, :stencil)[1]
@inline _get_cub_prop(aq::_CubicAnchoredQuery, ::Val{:idxL}) = getfield(aq, :stencil)[1]
@inline _get_cub_prop(aq::_CubicAnchoredQuery, ::Val{:idxR}) = getfield(aq, :stencil)[2]
@inline _get_cub_prop(aq::_CubicAnchoredQuery, ::Val{s}) where {s} = getfield(aq, s)
@inline Base.propertynames(::_CubicAnchoredQuery) =
    (:stencil, :idx, :idxL, :idxR, :xq, :state, :w0, :w1, :w2, :w3)

# Outer constructor: infer Tq from weight element type (not from input xq).
# When grid is Dual, weights are Dual even if xq is Float64.
# Widens xq to match weight type for struct consistency.
@inline function _CubicAnchoredQuery(stencil::_IdxStencil{2}, xq, state::UInt8, w0::NTuple{4, Tw}, w1::NTuple{4, Tw}, w2::NTuple{2, Tw}, w3::NTuple{2, Tw}, ::Type{Tg}) where {Tg, Tw}
    xq_p = convert(Tw, xq)
    return _CubicAnchoredQuery{Tg, Tw}(stencil, xq_p, state, w0, w1, w2, w3)
end

# ========================================
# Weight Computation
# ========================================

"""
    _compute_anchor_weights(::EvalValue, h, inv_h, dL, dR) -> NTuple{4,Tq}

Compute weights for cubic spline value evaluation.

Weights satisfy: S(xq) = wyL*yL + wyR*yR + wzL*zL + wzR*zR

# AD Support
When dL/dR are ForwardDiff.Dual (from xq - xL), the output tuple
preserves Dual type through arithmetic operations.
"""
@inline function _compute_anchor_weights(::EvalValue, h::Tg, inv_h::Tg, dL::Tq, dR::Tq) where {Tg, Tq}
    wyL = dR * inv_h
    wyR = dL * inv_h
    # wzL = (inv_h * dR^3 - h * dR) / 6
    # wzR = (inv_h * dL^3 - h * dL) / 6
    div6 = inv(Tg(6))
    wzL = (inv_h * dR^3 - h * dR) * div6
    wzR = (inv_h * dL^3 - h * dL) * div6
    return (wyL, wyR, wzL, wzR)
end

"""
    _compute_anchor_weights(::EvalDeriv1, h, inv_h, dL, dR) -> NTuple{4,Tq}

Compute weights for cubic spline first derivative evaluation.

Weights satisfy: S'(xq) = wyL*yL + wyR*yR + wzL*zL + wzR*zR
"""
@inline function _compute_anchor_weights(::EvalDeriv1, h::Tg, inv_h::Tg, dL::Tq, dR::Tq) where {Tg, Tq}
    # Promote to Tq for AD support (zero-overhead when Tg === Tq)
    wyL = oftype(dL, -inv_h)
    wyR = oftype(dL, inv_h)
    inv_2h = inv_h * inv(Tg(2))
    h_div6 = h * inv(Tg(6))
    wzL = -dR^2 * inv_2h + h_div6
    wzR = dL^2 * inv_2h - h_div6
    return (wyL, wyR, wzL, wzR)
end

"""
    _compute_anchor_weights(::EvalDeriv2, h, inv_h, dL, dR) -> NTuple{2,Tq}

Compute weights for cubic spline second derivative evaluation.

Weights satisfy: S''(xq) = wzL*zL + wzR*zR (no y contribution)

Returns (wzL, wzR) - optimized to exclude zero y-weights.
"""
@inline function _compute_anchor_weights(::EvalDeriv2, ::Tg, inv_h::Tg, dL::Tq, dR::Tq) where {Tg, Tq}
    wzL = dR * inv_h
    wzR = dL * inv_h
    return (wzL, wzR)
end

"""
    _compute_anchor_weights(::EvalDeriv3, h, inv_h, dL, dR) -> NTuple{2,Tq}

Compute weights for cubic spline third derivative evaluation.

# Formula
S'''(x) = (zR - zL) / h = -inv_h * zL + inv_h * zR
        = wzL*zL + wzR*zR

Weights: wzL=-inv_h, wzR=inv_h

Returns (wzL, wzR) - optimized to exclude zero y-weights.

# Note
Third derivative is constant within each interval (independent of dL, dR).
The weights are converted to Tq type for struct consistency, though they
don't carry AD derivative information.
"""
@inline function _compute_anchor_weights(::EvalDeriv3, ::Tg, inv_h::Tg, dL::Tq, ::Tq) where {Tg, Tq}
    # Promote to Tq for AD support (zero-overhead when Tg === Tq)
    wzL = oftype(dL, -inv_h)
    wzR = oftype(dL, inv_h)
    return (wzL, wzR)
end

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{Tg}, xq::Tq, ::Val{:cubic}; wrap::Bool=false) -> _CubicAnchoredQuery{Tg, Tq}

Create an anchored query for ultra-fast cubic spline evaluation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar, can be Float or ForwardDiff.Dual for AD)
- `::Val{:cubic}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=WrapExtrap()` mode. Distinct from `PeriodicBC` (boundary condition).

# Returns
`_CubicAnchoredQuery{Tg, Tq}` with precomputed geometry weights for value and derivatives.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = cubic_interp(collect(x), sin.(2π .* x))
itp2 = cubic_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:cubic))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=DerivOp(1))  # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is 2-4x faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search and geometry setup.

# AD Support
When `xq` is a ForwardDiff.Dual, the anchor preserves the Dual type
in `xq` and weight fields, enabling automatic differentiation.
"""
@inline function _anchor_query(
        x::AbstractVector{Tg},
        xq::Tq,
        ::Val{:cubic},
        wrap::Bool = false,
        searcher::P = DEFAULT_SEARCHER
    ) where {Tg, Tq <: Real, P <: Searcher}
    # Promote query for anchor: preserve Dual, promote Int/Rational to grid type
    # Cubic anchors store weight tuples with complex arithmetic that requires Float
    xq_promoted = _promote_for_anchor(xq, Tg)
    return _anchor_query_impl(x, xq_promoted, wrap, _resolve_searcher_for_grid(x, searcher))
end

"""
    _anchor_query(x::AbstractVector{T}, xq::AbstractVector, ::Val{:cubic}; wrap::Bool=false) -> Vector{_CubicAnchoredQuery{T,T}}

Create anchored queries for multiple query points.

Internal API: No runtime grid validation. Caller must ensure `x` matches
the grid used for interpolant construction.

# Arguments
- `x`: Grid points (must match interpolant's grid)
- `xq`: Query points (any Real type, auto-promoted to T)
- `::Val{:cubic}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap query points to domain [x[1], x[end]) before anchoring.
          Used for `extrap=WrapExtrap()` mode. Distinct from `PeriodicBC` (boundary condition).

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:cubic))

itp1 = cubic_interp(x, sin.(2π .* x))
itp2 = cubic_interp(x, cos.(2π .* x))

vals1 = itp1(aq_vec)  # Batch evaluation
vals2 = itp2(aq_vec)  # Reuse same anchors
```

# Note
For vector queries, `xq` elements are promoted to grid type `T` for pool compatibility.
AD is not supported for vector queries (use scalar queries for ForwardDiff).
"""
function _anchor_query(
        x::AbstractVector{T},
        xq::AbstractVector{S},
        ::Val{:cubic},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {T, S <: Real, P <: Searcher}
    isempty(xq) && return _CubicAnchoredQuery{T, T}[]
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    # First anchor determines concrete element type (Tq may widen for duck-typed grids)
    aq1 = _anchor_query_impl(x, _promote_for_anchor(xq[1], T), wrap, searcher_resolved)
    output = Vector{typeof(aq1)}(undef, length(xq))
    @inbounds output[1] = aq1
    @inbounds for k in 2:length(xq)
        output[k] = _anchor_query_impl(x, _promote_for_anchor(xq[k], T), wrap, searcher_resolved)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:cubic}; wrap=false) -> buffer

Fill a pre-allocated buffer with anchored queries for cubic spline evaluation.
In-place version of `_anchor_query(x, xq, Val(:cubic))` for zero-allocation pooled usage.

# Arguments
- `buffer::AbstractVector{_CubicAnchoredQuery{Tg,Tq}}`: Pre-allocated buffer (length >= length(xq))
- `x::AbstractVector{Tg}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector{Tq}`: Query points (must match buffer's query type)
- `::Val{:cubic}`: Type tag for cubic interpolation
- `wrap::Bool=false`: If true, wrap query points to domain [x[1], x[end])

# Returns
The same `buffer` object, filled with anchored queries.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
xq = [0.15, 0.35, 0.75]
buffer = Vector{_CubicAnchoredQuery{Float64,Float64}}(undef, length(xq))
_fill_anchors!(buffer, x, xq, Val(:cubic))
```
"""
@inline function _fill_anchors!(
        buffer::AbstractVector{_CubicAnchoredQuery{Tg, Tq}},
        x::AbstractVector{Tg},
        xq::AbstractVector{S},
        ::Val{:cubic},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {Tg, Tq <: Real, S <: Real, P <: Searcher}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)

    @inbounds for k in eachindex(xq)
        buffer[k] = _anchor_query_impl(x, _promote_for_anchor(xq[k], Tg), wrap, searcher_resolved)
    end
    return buffer
end

"""
    _anchor_query_impl(x, xq, wrap, policy) -> _CubicAnchoredQuery

Internal implementation of _anchor_query.

# Arguments
- `x`: Grid points (AbstractFloat type)
- `xq`: Query point (Real type - can be Float or ForwardDiff.Dual)
- `wrap`: Whether to wrap query point to domain
- `policy`: Search policy for interval search (default: DEFAULT_SEARCHER)

# AD Support
When `xq` is a ForwardDiff.Dual, the returned anchor preserves the Dual type
in `xq` and weight fields. The interval search uses `_extract_primal(xq)` for comparisons
while preserving the full Dual value for weight computation.
"""
@inline function _anchor_query_impl(
        x::AbstractVector{Tg},
        xq::Tq,
        wrap::Bool,
        policy::P = DEFAULT_SEARCHER
    ) where {Tg, Tq <: Real, P <: Searcher}
    loc = _anchor_loc(x, xq, wrap, policy)

    # Compute geometry (cubic-internal concern)
    # h and inv_h are Tg (grid type)
    # dL and dR preserve Dual type for AD (via loc.xq)
    h = _get_h(x, loc.idx, loc.xL, loc.xR)
    inv_h = _get_inv_h(x, loc.idx, loc.xL, loc.xR)
    dL = loc.xq - loc.xL
    dR = loc.xR - loc.xq

    # Compute weights for value and derivatives
    w0 = _compute_anchor_weights(EvalValue(), h, inv_h, dL, dR)
    w1 = _compute_anchor_weights(EvalDeriv1(), h, inv_h, dL, dR)
    w2 = _compute_anchor_weights(EvalDeriv2(), h, inv_h, dL, dR)
    w3 = _compute_anchor_weights(EvalDeriv3(), h, inv_h, dL, dR)

    # `_anchor_loc` discards `idx_R` from `search_interval`'s 4-tuple, so this
    # path always assumes `idxR = idxL + 1`. Valid only for:
    #   - non-periodic queries (no seam dispatch in the searcher),
    #   - `WrapExtrap` queries (wrap maps into `[first(x), last(x))`, no seam),
    #   - periodic queries on a *post-extension* (n+1) grid (idxL+1 ≤ n+1).
    # Periodic-exclusive callers on a raw n-size grid MUST bypass this path
    # and build the anchor from `search_interval`'s 4-tuple to preserve the
    # seam pair `(n, 1)` (see `_cubic_oneshot_series_periodic!` /
    # `_cubic_oneshot_series_periodic_vec!`). The proper long-term fix is the
    # `_anchor_loc` 4-tuple refactor tracked as MEMORY.md "search_interval
    # 4-value refactor (PR A)", which would let this path read `idx_R`
    # directly and eliminate the bypass.
    return _CubicAnchoredQuery(_IdxPair(loc.idx, loc.idx + 1), loc.xq, loc.state, w0, w1, w2, w3, Tg)
end

# ========================================
# Shared Raw-Vector Anchor Eval
# ========================================
# Canonical kernel + extrap dispatch that takes raw y, z vectors.
# Used by both CubicInterpolant anchor dispatch AND cubic series one-shot.

# ─── Kernel: dispatches on DerivOp, returns scalar ───────────────────────────

# EvalValue: Full 4-term dot product. `aq.idxR` carries seam wrap (== 1 at the
# periodic-exclusive seam, otherwise == idxL + 1) — kernels are oblivious.
@inline function _cubic_eval_kernel(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalValue
    )
    wyL, wyR, wzL, wzR = aq.w0
    idxL = aq.idxL
    idxR = aq.idxR
    @inbounds return muladd(
        wyR, y[idxR], muladd(
            wyL, y[idxL],
            muladd(wzR, z[idxR], wzL * z[idxL])
        )
    )
end

# EvalDeriv1: Full 4-term with w1
@inline function _cubic_eval_kernel(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalDeriv1
    )
    wyL, wyR, wzL, wzR = aq.w1
    idxL = aq.idxL
    idxR = aq.idxR
    @inbounds return muladd(
        wyR, y[idxR], muladd(
            wyL, y[idxL],
            muladd(wzR, z[idxR], wzL * z[idxL])
        )
    )
end

# EvalDeriv2: Optimized 2-term (z-only, no y-loads)
@inline function _cubic_eval_kernel(
        ::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalDeriv2
    )
    wzL, wzR = aq.w2
    @inbounds return muladd(wzR, z[aq.idxR], wzL * z[aq.idxL])
end

# EvalDeriv3: Optimized 2-term (z-only, no y-loads)
@inline function _cubic_eval_kernel(
        ::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, ::EvalDeriv3
    )
    wzL, wzR = aq.w3
    @inbounds return muladd(wzR, z[aq.idxR], wzL * z[aq.idxL])
end

# DerivOp{N≥4}: zero (N-th derivative of cubic is zero for N ≥ 4)
@inline function _cubic_eval_kernel(
        y::AbstractVector, ::AbstractVector,
        aq::_CubicAnchoredQuery, ::DerivOp{N}
    ) where {N}
    return 0 * (@inbounds y[aq.idxL])
end

# ─── Extrap dispatch: handles OOB logic ──────────────────────────────────────

# Default (ExtendExtrap, WrapExtrap, InBounds): just kernel
@inline function _cubic_eval_at_anchor(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, op::AbstractEvalOp, ::AbstractExtrap
    )
    return _cubic_eval_kernel(y, z, aq, op)
end

# NoExtrap: throw if OOB
@inline function _cubic_eval_at_anchor(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, op::AbstractEvalOp, ::NoExtrap
    )
    aq.state != IN_DOMAIN && throw(DomainError(aq.xq, "query point outside domain"))
    return _cubic_eval_kernel(y, z, aq, op)
end

# ClampExtrap / FillExtrap: boundary value if OOB
@inline function _cubic_eval_at_anchor(
        y::AbstractVector, z::AbstractVector,
        aq::_CubicAnchoredQuery, op::AbstractEvalOp, extrap::_ClampOrFill
    )
    if aq.state != IN_DOMAIN
        y_bnd = aq.state == OOB_LEFT ? first(y) : last(y)
        return _eval_extrapolation(op, y_bnd, extrap, aq.xq)
    end
    return _cubic_eval_kernel(y, z, aq, op)
end
