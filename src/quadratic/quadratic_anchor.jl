# ========================================
# Quadratic Anchored Query
# ========================================
# Precomputed geometry for ultra-fast quadratic interpolation evaluation
# at a fixed query point. Enables speedup by eliminating interval search
# for repeated evaluations.
#
# Include order: ops.jl → utils.jl → quadratic_kernels.jl → quadratic_interp.jl → quadratic_anchor.jl
#
# ========================================
# _QuadraticAnchoredQuery Type (Internal)
# ========================================

"""
    _QuadraticAnchoredQuery{Tg, Tq}

Precomputed geometry for ultra-fast quadratic interpolation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `Tq`: Query type (Float or ForwardDiff.Dual for AD support)

# Fields
- `idx`: Interval index where xq falls
- `xq`: Original query point (or wrapped value for periodic)
- `side`: Domain position (0=inside, 1=below, 2=above)
- `dL`: Offset from interval start (xq - x_i)

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:quadratic))

itp1 = quadratic_interp(x, sin.(2π .* x))
itp2 = quadratic_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=DerivOp(1))  # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search.

# AD Support
When `xq` is a `ForwardDiff.Dual`, the anchor preserves the Dual type
in both `xq` and `dL` fields, enabling automatic differentiation through
series interpolant evaluation.
"""
struct _QuadraticAnchoredQuery{Tg<:AbstractFloat, Tq<:Real}
    idx::Int                   # interval index
    xq::Tq                     # query point (possibly wrapped), Float or Dual
    side::UInt8                # 0=inside, 1=below_min, 2=above_max
    dL::Tq                     # offset from interval start, Float or Dual
end

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{Tg}, xq::Tq, ::Val{:quadratic}; wrap::Bool=false) -> _QuadraticAnchoredQuery

Create an anchored query for ultra-fast quadratic interpolation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar, can be Float or ForwardDiff.Dual for AD)
- `::Val{:quadratic}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=WrapExtrap()` mode.

# Returns
`_QuadraticAnchoredQuery{Tg, Tq}` with precomputed geometry.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = quadratic_interp(collect(x), sin.(2π .* x))
itp2 = quadratic_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:quadratic))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=DerivOp(1))  # Reuses same anchor for derivative
```
"""
@inline function _anchor_query(
    x::AbstractVector{Tg},
    xq::Tq,
    ::Val{:quadratic},
    wrap::Bool=false,
    searcher::P=DEFAULT_SEARCHER
) where {Tg<:AbstractFloat, Tq<:Real, P<:Searcher}
    return _quadratic_anchor_query_impl(x, xq, wrap, searcher)
end

"""
    _anchor_query(x::AbstractVector{T}, xq::AbstractVector, ::Val{:quadratic}; wrap::Bool=false) -> Vector{_QuadraticAnchoredQuery{T,T}}

Create anchored queries for multiple query points.

Internal API: No runtime grid validation. Caller must ensure `x` matches
the grid used for interpolant construction.

# Arguments
- `x`: Grid points (must match interpolant's grid)
- `xq`: Query points (any Real type, auto-promoted to T)
- `::Val{:quadratic}`: Type tag
- `wrap`: If true, wrap query points to domain [x[1], x[end]) before anchoring.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:quadratic))

itp1 = quadratic_interp(x, sin.(2π .* x))
itp2 = quadratic_interp(x, cos.(2π .* x))

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
    ::Val{:quadratic},
    wrap::Bool=false,
    searcher::P=_to_searcher(LinearBinary())
) where {T<:AbstractFloat, S<:Real, P<:Searcher}
    output = Vector{_QuadraticAnchoredQuery{T,T}}(undef, length(xq))

    @inbounds for k in eachindex(xq)
        output[k] = _quadratic_anchor_query_impl(x, T(xq[k]), wrap, searcher)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:quadratic}; wrap=false) -> buffer

Fill a pre-allocated buffer with anchored queries for quadratic interpolation.
In-place version of `_anchor_query(x, xq, Val(:quadratic))` for zero-allocation pooled usage.

# Arguments
- `buffer::Vector{_QuadraticAnchoredQuery{T,T}}`: Pre-allocated buffer (length >= length(xq))
- `x::AbstractVector{T}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector`: Query points (any Real type, auto-promoted to T)
- `::Val{:quadratic}`: Type tag for quadratic interpolation
- `wrap::Bool=false`: If true, wrap query points to domain [x[1], x[end])

# Returns
The same `buffer` object, filled with anchored queries.

# Note
When buffer element type is `{Tg, Tq}` and `xq` element type is `S`:
- If `Tq === S`: uses `xq[k]` directly (preserves precision)
- Otherwise: uses `_promote_for_anchor(xq[k], Tg)` for lossless promotion
"""
@inline function _fill_anchors!(
    buffer::AbstractVector{_QuadraticAnchoredQuery{Tg, Tq}},
    x::AbstractVector{Tg},
    xq::AbstractVector{S},
    ::Val{:quadratic},
    wrap::Bool=false,
    searcher::P=_to_searcher(LinearBinary())
) where {Tg<:AbstractFloat, Tq<:Real, S<:Real, P<:Searcher}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"

    @inbounds for k in eachindex(xq)
        # Promote query point: preserves precision when S is wider than Tg
        xq_promoted = _promote_for_anchor(xq[k], Tg)
        buffer[k] = _quadratic_anchor_query_impl(x, xq_promoted, wrap, searcher)
    end
    return buffer
end

"""
    _quadratic_anchor_query_impl(x, xq, wrap, policy) -> _QuadraticAnchoredQuery

Internal implementation of _anchor_query for quadratic interpolation.

# Arguments
- `x`: Grid points (AbstractFloat type)
- `xq`: Query point (Real type - can be Float or ForwardDiff.Dual)
- `wrap`: Whether to wrap query point to domain
- `policy`: Search policy for interval search (default: DEFAULT_SEARCHER)

# AD Support
When `xq` is a ForwardDiff.Dual, the returned anchor preserves the Dual type
in `xq` and `dL` fields. The interval search uses `_extract_primal(xq)` for comparisons
while preserving the full Dual value for `dL` computation.
"""
@inline function _quadratic_anchor_query_impl(
    x::AbstractVector{Tg},
    xq::Tq,
    wrap::Bool,
    policy::P=DEFAULT_SEARCHER
) where {Tg<:AbstractFloat, Tq<:Real, P<:Searcher}
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Handle wrapping (for extrap=WrapExtrap() mode)
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
    # For outside-domain points, use boundary intervals
    # Note: Convert primal to Tg for search_interval (requires matching types)
    idx, xL, _ = if xq_primal < x_min
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

    # Compute dL: offset from interval start
    # This preserves Dual type when xq is Dual
    dL = xq - xL

    return _QuadraticAnchoredQuery{Tg,Tq}(idx, xq, side, dL)
end

# ========================================
# QuadraticInterpolant Evaluation with Anchor
# ========================================

"""
    (itp::QuadraticInterpolant)(aq::_QuadraticAnchoredQuery; deriv::DerivOp=EvalValue())

Evaluate quadratic interpolant at anchored query point. Ultra-fast path that
skips interval search.

# Arguments
- `aq`: Pre-computed anchored query from `_anchor_query`
- `deriv`: Derivative order (0=value, 1=first derivative, 2=second derivative)

# Example
```julia
itp = quadratic_interp(x, y)
aq = _anchor_query(x, 0.5, Val(:quadratic))
val = itp(aq)                    # Value
d1 = itp(aq; deriv=DerivOp(1))   # First derivative
d2 = itp(aq; deriv=DerivOp(2))   # Second derivative
```
"""
@inline function (itp::QuadraticInterpolant{T})(aq::_QuadraticAnchoredQuery{T,Tq}; deriv::DerivOp=EvalValue()) where {T<:AbstractFloat, Tq<:Real}
    _quadratic_eval_with_anchor(itp, aq, deriv)
end

@inline function _quadratic_eval_with_anchor(
    itp::QuadraticInterpolant{T},
    aq::_QuadraticAnchoredQuery{T,Tq},
    op::O
) where {T<:AbstractFloat, Tq<:Real, O<:AbstractEvalOp}
    # Handle extrapolation based on mode and side
    return _quadratic_anchor_dispatch(itp, aq, op, itp.extrap)
end

# No extrapolation: throw DomainError if outside domain
@inline function _quadratic_anchor_dispatch(
    itp::QuadraticInterpolant{T},
    aq::_QuadraticAnchoredQuery{T,Tq},
    op::O,
    ::NoExtrap
) where {T<:AbstractFloat, Tq<:Real, O<:AbstractEvalOp}
    if aq.side != 0x00  # outside domain
        x_min, x_max = first(itp.x), last(itp.x)
        throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
    end
    @inbounds return _quadratic_kernel(op, itp.a[aq.idx], itp.d[aq.idx], itp.y[aq.idx], aq.dL)
end

# Inside domain or extension mode: use interpolation
@inline function _quadratic_anchor_dispatch(
    itp::QuadraticInterpolant{T},
    aq::_QuadraticAnchoredQuery{T,Tq},
    op::O,
    ::AbstractExtrap
) where {T<:AbstractFloat, Tq<:Real, O<:AbstractEvalOp}
    @inbounds return _quadratic_kernel(op, itp.a[aq.idx], itp.d[aq.idx], itp.y[aq.idx], aq.dL)
end

# Constant extrapolation: special handling for outside-domain
@inline function _quadratic_anchor_dispatch(
    itp::QuadraticInterpolant{T},
    aq::_QuadraticAnchoredQuery{T,Tq},
    op::O,
    ::ConstExtrap
) where {T<:AbstractFloat, Tq<:Real, O<:AbstractEvalOp}
    if aq.side == 0x01  # below domain
        return _constant_extrap_result(op, @inbounds itp.y[1])
    elseif aq.side == 0x02  # above domain
        return _constant_extrap_result(op, @inbounds itp.y[end])
    else
        @inbounds return _quadratic_kernel(op, itp.a[aq.idx], itp.d[aq.idx], itp.y[aq.idx], aq.dL)
    end
end

# ========================================
# Vector Evaluation with Anchors
# ========================================

"""
    (itp::QuadraticInterpolant)(aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{T}}; deriv::DerivOp=EvalValue())

Evaluate quadratic interpolant at multiple anchored query points.
Returns newly allocated vector.
"""
function (itp::QuadraticInterpolant{T})(
    aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{T}};
    deriv::DerivOp=EvalValue()
) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(aq_vec))
    @inbounds for i in eachindex(aq_vec)
        output[i] = _quadratic_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end

"""
    (itp::QuadraticInterpolant)(output::AbstractVector{T}, aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{T}}; deriv::DerivOp=EvalValue())

In-place evaluation at multiple anchored query points. Zero allocation.
"""
function (itp::QuadraticInterpolant{T})(
    output::AbstractVector{T},
    aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{T}};
    deriv::DerivOp=EvalValue()
) where {T<:AbstractFloat}
    @assert length(output) == length(aq_vec) "output length must match aq_vec length"
    @inbounds for i in eachindex(aq_vec)
        output[i] = _quadratic_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end
