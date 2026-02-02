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
- `idx`: Interval index where xq falls
- `xq`: Original query point (or wrapped value for periodic)
- `side`: Domain position (0=inside, 1=left, 2=right)
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
itp2(aq; deriv=1)     # Reuses same anchor for derivative
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
struct _CubicAnchoredQuery{Tg<:AbstractFloat, Tq<:Real}
    idx::Int                   # interval index
    xq::Tq                     # query point (possibly wrapped), Float or Dual
    side::UInt8                # 0=inside, 1=below_min, 2=above_max
    w0::NTuple{4,Tq}           # (wyL, wyR, wzL, wzR) for value
    w1::NTuple{4,Tq}           # (wyL, wyR, wzL, wzR) for first deriv
    w2::NTuple{2,Tq}           # (wzL, wzR) for second deriv - optimized
    w3::NTuple{2,Tq}           # (wzL, wzR) for third deriv - optimized
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
    # Convert to Tq for type consistency (important for AD support with Dual types)
    wyL = -inv_h + zero(dL)
    wyR =  inv_h + zero(dL)
    inv_2h = inv_h * inv(Tg(2))
    h_div6 = h * inv(Tg(6))
    wzL = -dR^2 * inv_2h + h_div6
    wzR =  dL^2 * inv_2h - h_div6
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
    # Convert to Tq for type consistency with other weight tuples
    # Using zero(dL) ensures correct Tq type even when Tq is Dual
    wzL = -inv_h + zero(dL)
    wzR =  inv_h + zero(dL)
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
          Used for `extrap=:wrap` mode. Distinct from `PeriodicBC` (boundary condition).

# Returns
`_CubicAnchoredQuery{Tg, Tq}` with precomputed geometry weights for value and derivatives.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = cubic_interp(collect(x), sin.(2π .* x))
itp2 = cubic_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:cubic))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=1)     # Reuses same anchor for derivative
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
    ::Val{:cubic};
    wrap::Bool=false,
    searcher::Searcher=DEFAULT_SEARCHER
) where {Tg<:AbstractFloat, Tq<:Real}
    # Promote query for anchor: preserve Dual, promote Int/Rational to grid type
    # Cubic anchors store weight tuples with complex arithmetic that requires Float
    xq_promoted = _promote_for_anchor(xq, Tg)
    return _anchor_query_impl(x, xq_promoted, wrap, searcher)
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
          Used for `extrap=:wrap` mode. Distinct from `PeriodicBC` (boundary condition).

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
    ::Val{:cubic};
    wrap::Bool=false,
    searcher::Searcher=_to_searcher(LinearBinary())
) where {T<:AbstractFloat, S<:Real}
    output = Vector{_CubicAnchoredQuery{T,T}}(undef, length(xq))

    @inbounds for k in eachindex(xq)
        output[k] = _anchor_query_impl(x, T(xq[k]), wrap, searcher)
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
    buffer::AbstractVector{_CubicAnchoredQuery{Tg,Tq}},
    x::AbstractVector{Tg},
    xq::AbstractVector{Tq},
    ::Val{:cubic};
    wrap::Bool=false,
    searcher::Searcher=_to_searcher(LinearBinary())
) where {Tg<:AbstractFloat, Tq<:Real}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"

    # Use original xq[k] directly (no conversion) to preserve precision in weights
    @inbounds for k in eachindex(xq)
        buffer[k] = _anchor_query_impl(x, xq[k], wrap, searcher)
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
    policy::P=DEFAULT_SEARCHER
) where {Tg<:AbstractFloat, Tq<:Real, P<:Searcher}
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Handle wrapping (for extrap=:wrap mode)
    # Generic _wrap_to_domain handles AD primal extraction and returns Tg
    if wrap && (xq_primal < x_min || xq_primal >= x_max)
        xq = _wrap_to_domain(xq, x_min, x_max)
        xq_primal = xq  # xq is now Tg, no need for _extract_primal
    end

    # Determine side (domain position)
    side = if xq_primal < x_min
        0x01  # below min
    elseif xq_primal > x_max
        0x02  # above max
    else
        0x00  # inside
    end

    # Find interval and compute geometry
    # For outside-domain points, use boundary intervals for weight computation
    # Note: Convert primal to Tg for search_interval (requires matching types)
    idx, xL, xR = if xq_primal < x_min
        # Below domain: use first interval
        @inbounds (1, x[1], x[2])
    elseif xq_primal > x_max
        # Above domain: use last interval
        n = length(x)
        @inbounds (n - 1, x[n-1], x[n])
    else
        # Inside domain: use policy-based interval search
        search_interval(policy, x, xq_primal)
    end

    # Compute geometry
    # h and inv_h are Tg (grid type)
    # dL and dR are Tq (preserves Dual type for AD)
    h = xR - xL
    inv_h = one(Tg) / h
    dL = xq - xL  # distance from Left endpoint (Tq type)
    dR = xR - xq  # distance from Right endpoint (Tq type)

    # Compute weights for value and derivatives
    # Weights will be Tq type (preserves Dual for AD)
    w0 = _compute_anchor_weights(EvalValue(), h, inv_h, dL, dR)
    w1 = _compute_anchor_weights(EvalDeriv1(), h, inv_h, dL, dR)
    w2 = _compute_anchor_weights(EvalDeriv2(), h, inv_h, dL, dR)
    w3 = _compute_anchor_weights(EvalDeriv3(), h, inv_h, dL, dR)

    return _CubicAnchoredQuery{Tg,Tq}(idx, xq, side, w0, w1, w2, w3)
end
