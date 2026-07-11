# ========================================
# Cubic Adjoint Anchor
# ========================================
# Precomputed geometry weights baked at a fixed query point, consumed by the
# 1D cubic adjoint (cubic_adjoint.jl) to select the per-DerivOp weight field in
# the reverse pass without repeating the interval search or geometry setup.
#
# Include order: ops.jl → ... → cubic_types.jl → cubic_anchor.jl → cubic_interpolant.jl
#
# ========================================
# _CubicAdjointAnchor Type (Internal)
# ========================================

"""
    _CubicAdjointAnchor{Tg, Tq, I}

Precomputed cubic weights baked at a fixed query point, consumed by the cubic
adjoint's reverse pass (not a forward-eval entry point; see Usage).
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `Tq`: Query type (Float or ForwardDiff.Dual for AD support)
- `I <: _AbstractIndices{2}`: Interval representation — `_ContiguousIndices{2}` (one Int) for
  ordinary grids, `_ExplicitIndices{2}` (two Int) for periodic-exclusive seam cells

# Fields
- `interval::I`: Physical cell interval carrying the corner pair `(idxL, idxR)`.
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
Consumed only by the 1D cubic adjoint (`src/cubic/cubic_adjoint.jl`), which
bakes an anchor per query at construction via `_anchor_query(x, xq, Val(:cubic))`
and reuses the precomputed weights across the reverse pass. Not a forward-eval
entry point — forward evaluation goes through `itp(xq)`.

# Performance
Baking all four op weights once lets the adjoint select the right field
(`w0`..`w3`) per `DerivOp{N}` in the pullback without repeating the O(log n)
interval search or geometry setup.

# Memory Optimization
w2 and w3 store only (wzL, wzR) since second and third derivatives
depend only on z-values, not y-values. This reduces anchor size by 16 bytes
per query for Float64.

# AD Support
When `xq` is a `ForwardDiff.Dual`, the anchor preserves the Dual type
in both `xq` and weight fields, enabling automatic differentiation through
series interpolant evaluation.
"""
struct _CubicAdjointAnchor{Tg, Tq <: Real, I <: _AbstractIndices{2}}
    # Physical cell interval: `interval[1]` is the left index (idxL), `interval[2]`
    # is the right index (idxR). For non-periodic cells `idxR == idxL + 1`; for
    # periodic-exclusive seam cells `idxR == 1` (wrap). Ordinary grids store the
    # compact `_ContiguousIndices{2}` (one Int); periodic-exclusive seam cells
    # store `_ExplicitIndices{2}` (src/core/axis_indices.jl). Mirrors `_LinearAnchoredQuery`.
    # Legacy `aq.idx` accessor is preserved via `getproperty` (= idxL).
    interval::I
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
# `getfield` (+ tuple index for the interval paths). Avoids the union-typed
# return that would otherwise box weights/state in hot loops.
@inline Base.getproperty(aq::_CubicAdjointAnchor, s::Symbol) = _get_cub_prop(aq, Val(s))
@inline _get_cub_prop(aq::_CubicAdjointAnchor, ::Val{:idx}) = getfield(aq, :interval)[Val(1)]
@inline _get_cub_prop(aq::_CubicAdjointAnchor, ::Val{:idxL}) = getfield(aq, :interval)[Val(1)]
@inline _get_cub_prop(aq::_CubicAdjointAnchor, ::Val{:idxR}) = getfield(aq, :interval)[Val(2)]
@inline _get_cub_prop(aq::_CubicAdjointAnchor, ::Val{s}) where {s} = getfield(aq, s)
@inline Base.propertynames(::_CubicAdjointAnchor) =
    (:interval, :idx, :idxL, :idxR, :xq, :state, :w0, :w1, :w2, :w3)

# Outer constructor: infer Tq from weight element type (not from input xq).
# When grid is Dual, weights are Dual even if xq is Float64. `I` is inferred
# from the caller's `interval` (contiguous for ordinary grids, explicit at
# periodic-exclusive seams — chosen local to the call site).
# Widens xq to match weight type for struct consistency.
@inline function _CubicAdjointAnchor(interval::I, xq, state::UInt8, w0::NTuple{4, Tw}, w1::NTuple{4, Tw}, w2::NTuple{2, Tw}, w3::NTuple{2, Tw}, ::Type{Tg}) where {Tg, Tw, I <: _AbstractIndices{2}}
    xq_p = convert(Tw, xq)
    return _CubicAdjointAnchor{Tg, Tw, I}(interval, xq_p, state, w0, w1, w2, w3)
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
    _anchor_query(x::AbstractVector{Tg}, xq::Tq, ::Val{:cubic}; wrap::Bool=false) -> _CubicAdjointAnchor{Tg, Tq}

Create an anchored query at a fixed point, baked once and reused by the cubic
adjoint's reverse pass.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar, can be Float or ForwardDiff.Dual for AD)
- `::Val{:cubic}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap `xq` to closed domain [x[1], x[end]] before anchoring.
          Used for `extrap=WrapExtrap()` mode. Distinct from `PeriodicBC` (boundary condition).

# Returns
`_CubicAdjointAnchor{Tg, Tq}` with precomputed geometry weights for value and derivatives.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:cubic))  # baked and stored by the cubic adjoint
```

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
    xq_promoted = _promote_coord(xq, Tg)
    return _anchor_query_impl(x, xq_promoted, wrap, _resolve_searcher_for_grid(x, searcher))
end

"""
    _anchor_query(x::AbstractVector{T}, xq::AbstractVector, ::Val{:cubic}; wrap::Bool=false) -> Vector{_CubicAdjointAnchor{T,T}}

Create anchored queries for multiple query points.

Internal API: No runtime grid validation. Caller must ensure `x` matches
the grid used for interpolant construction.

# Arguments
- `x`: Grid points (must match interpolant's grid)
- `xq`: Query points (any Real type, auto-promoted to T)
- `::Val{:cubic}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap query points to closed domain [x[1], x[end]] before anchoring.
          Used for `extrap=WrapExtrap()` mode. Distinct from `PeriodicBC` (boundary condition).

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:cubic))  # anchor per query for the adjoint pullback
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
    isempty(xq) && return _CubicAdjointAnchor{T, T, _interval_type(x)}[]
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    # First anchor determines concrete element type (Tq may widen for duck-typed grids)
    aq1 = _anchor_query_impl(x, _promote_coord(xq[1], T), wrap, searcher_resolved)
    output = Vector{typeof(aq1)}(undef, length(xq))
    @inbounds output[1] = aq1
    @inbounds for k in 2:length(xq)
        output[k] = _anchor_query_impl(x, _promote_coord(xq[k], T), wrap, searcher_resolved)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:cubic}; wrap=false) -> buffer

Fill a pre-allocated buffer with anchored queries for cubic spline evaluation.
In-place version of `_anchor_query(x, xq, Val(:cubic))` for zero-allocation pooled usage.

# Arguments
- `buffer::AbstractVector{<:_CubicAdjointAnchor{Tg,Tq}}`: Pre-allocated buffer (length >= length(xq))
- `x::AbstractVector{Tg}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector{Tq}`: Query points (must match buffer's query type)
- `::Val{:cubic}`: Type tag for cubic interpolation
- `wrap::Bool=false`: If true, wrap query points to closed domain [x[1], x[end]]

# Returns
The same `buffer` object, filled with anchored queries.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
xq = [0.15, 0.35, 0.75]
buffer = Vector{_CubicAdjointAnchor{Float64,Float64,_ContiguousIndices{2}}}(undef, length(xq))
_fill_anchors!(buffer, x, xq, Val(:cubic))
```
"""
@inline function _fill_anchors!(
        buffer::AbstractVector{<:_CubicAdjointAnchor{Tg, Tq}},
        x::AbstractVector{Tg},
        xq::AbstractVector{S},
        ::Val{:cubic},
        wrap::Bool = false,
        searcher::P = _to_searcher(LinearBinarySearch())
    ) where {Tg, Tq <: Real, S <: Real, P <: Searcher}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)

    @inbounds for k in eachindex(xq)
        buffer[k] = _anchor_query_impl(x, _promote_coord(xq[k], Tg), wrap, searcher_resolved)
    end
    return buffer
end

"""
    _anchor_query_impl(x, xq, wrap, policy) -> _CubicAdjointAnchor

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
    h = _get_h(x, loc.idxL, loc.xL, loc.xR)
    inv_h = _get_inv_h(x, loc.idxL, loc.xL, loc.xR)
    dL = loc.xq - loc.xL
    dR = loc.xR - loc.xq

    # Compute weights for value and derivatives
    w0 = _compute_anchor_weights(EvalValue(), h, inv_h, dL, dR)
    w1 = _compute_anchor_weights(EvalDeriv1(), h, inv_h, dL, dR)
    w2 = _compute_anchor_weights(EvalDeriv2(), h, inv_h, dL, dR)
    w3 = _compute_anchor_weights(EvalDeriv3(), h, inv_h, dL, dR)

    return _CubicAdjointAnchor(loc.interval, loc.xq, loc.state, w0, w1, w2, w3, Tg)
end
