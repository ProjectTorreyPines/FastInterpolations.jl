# ========================================
# Linear Anchored Query
# ========================================
# Precomputed geometry for ultra-fast linear interpolation evaluation
# at a fixed query point. Enables speedup by eliminating interval search
# for repeated evaluations.
#
# Include order: ops.jl → utils.jl → linear_kernels.jl → linear_interp.jl → linear_anchor.jl
#
# ========================================
# _LinearAnchoredQuery Type (Internal)
# ========================================

"""
    _LinearAnchoredQuery{Tg, Tq}

Precomputed geometry for ultra-fast linear interpolation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `Tg`: Grid type (Float32 or Float64)
- `Tq`: Query type (can differ from Tg for precision preservation)

# Fields
- `idx`: Interval index where xq falls
- `xq`: Original query point (or wrapped value for periodic), preserves original precision
- `side`: Domain position (0=inside, 1=below, 2=above)
- `h`: Interval width (xR - xL)
- `inv_h`: Precomputed reciprocal (1/h) for fast derivative computation
- `alpha`: Normalized position within interval: (xq - xL) / h, preserves precision

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:linear))

itp1 = linear_interp(x, sin.(2π .* x))
itp2 = linear_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=1)     # Reuses same anchor for derivative
```

# Performance
Anchored evaluation is faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search.

# Efficiency
- `alpha` for EvalValue: `muladd(alpha, yR - yL, yL)` (no division)
- `inv_h` for EvalDeriv1: `(yR - yL) * inv_h` (no division)
"""
struct _LinearAnchoredQuery{Tg<:AbstractFloat, Tq<:Real}
    idx::Int                   # interval index
    xq::Tq                     # query point (possibly wrapped), original precision
    side::UInt8                # 0=inside, 1=below_min, 2=above_max
    h::Tg                      # interval width
    inv_h::Tg                  # precomputed 1/h
    alpha::Tq                  # normalized position: (xq - xL) / h, preserves precision
end

# ========================================
# Anchored Kernel Overloads
# ========================================
# These _linear_kernel methods receive the anchor directly and extract
# the appropriate precomputed value (alpha or inv_h) based on the operation.
# Leverages Julia's dispatch system for clean, unified API.

"""
    _linear_kernel(::EvalValue, yL, yR, aq::_LinearAnchoredQuery)

Evaluate linear interpolation using anchor's precomputed alpha.
No division: uses muladd(alpha, yR - yL, yL).
"""
@inline function _linear_kernel(::EvalValue, yL::Tv, yR::Tv, aq::_LinearAnchoredQuery) where {Tv}
    return muladd(aq.alpha, yR - yL, yL)
end

"""
    _linear_kernel(::EvalDeriv1, yL, yR, aq::_LinearAnchoredQuery)

Evaluate first derivative using anchor's precomputed inv_h.
No division: uses (yR - yL) * inv_h.
"""
@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, aq::_LinearAnchoredQuery{Tg}) where {Tg<:AbstractFloat, Tv}
    return (yR - yL) * aq.inv_h
end

"""Second derivative of linear is always zero."""
@inline function _linear_kernel(::EvalDeriv2, yL::Tv, ::Tv, ::_LinearAnchoredQuery{Tg}) where {Tg<:AbstractFloat, Tv}
    return zero(promote_type(Tv, Tg))
end

"""Third derivative of linear is always zero."""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, ::Tv, ::_LinearAnchoredQuery{Tg}) where {Tg<:AbstractFloat, Tv}
    return zero(promote_type(Tv, Tg))
end

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{T}, xq::T, ::Val{:linear}; wrap::Bool=false) -> _LinearAnchoredQuery

Create an anchored query for ultra-fast linear interpolation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar)
- `::Val{:linear}`: Type tag to distinguish from cubic anchor
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=:wrap` mode.

# Returns
`_LinearAnchoredQuery{T}` with precomputed geometry.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = linear_interp(collect(x), sin.(2π .* x))
itp2 = linear_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:linear))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq; deriv=1)     # Reuses same anchor for derivative
```
"""
@inline function _anchor_query(
    x::AbstractVector{T},
    xq::T,
    ::Val{:linear};
    wrap::Bool=false,
    searcher::Searcher=DEFAULT_SEARCHER
) where {T<:AbstractFloat}
    return _linear_anchor_query_impl(x, xq, wrap, searcher)
end

# Real wrapper for convenience (scalar) - preserves precision
@inline function _anchor_query(
    x::AbstractVector{Tg},
    xq::S,
    tag::Val{:linear};
    wrap::Bool=false,
    searcher::Searcher=DEFAULT_SEARCHER
) where {Tg<:AbstractFloat, S<:Real}
    xq_promoted = _promote_for_anchor(xq, Tg)
    return _linear_anchor_query_impl(x, xq_promoted, wrap, searcher)
end

"""
    _anchor_query(x::AbstractVector{Tg}, xq::AbstractVector, ::Val{:linear}; wrap::Bool=false)

Create anchored queries for multiple query points with precision preservation.

# Precision Preservation
Uses `_promote_for_anchor` to preserve wider precision when `S` differs from `Tg`.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:linear))

itp1 = linear_interp(x, sin.(2π .* x))
itp2 = linear_interp(x, cos.(2π .* x))

vals1 = itp1(aq_vec)  # Batch evaluation
vals2 = itp2(aq_vec)  # Reuse same anchors
```
"""
function _anchor_query(
    x::AbstractVector{Tg},
    xq::AbstractVector{S},
    ::Val{:linear};
    wrap::Bool=false,
    searcher::Searcher=_to_searcher(LinearBinary())
) where {Tg<:AbstractFloat, S<:Real}
    Tq = promote_type(S, Tg)
    output = Vector{_LinearAnchoredQuery{Tg, Tq}}(undef, length(xq))

    @inbounds for k in eachindex(xq)
        xq_promoted = _promote_for_anchor(xq[k], Tg)
        output[k] = _linear_anchor_query_impl(x, xq_promoted, wrap, searcher)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:linear}; wrap=false) -> buffer

Fill a pre-allocated buffer with anchored queries for linear interpolation.
In-place version of `_anchor_query(x, xq, Val(:linear))` for zero-allocation pooled usage.

# Arguments
- `buffer::Vector{_LinearAnchoredQuery{Tg,Tq}}`: Pre-allocated buffer (length >= length(xq))
- `x::AbstractVector{Tg}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector`: Query points (any Real type)
- `::Val{:linear}`: Type tag for linear interpolation
- `wrap::Bool=false`: If true, wrap query points to domain [x[1], x[end])

# Precision Preservation
Uses `_promote_for_anchor` to preserve wider precision when `S` differs from `Tg`.
"""
@inline function _fill_anchors!(
    buffer::AbstractVector{_LinearAnchoredQuery{Tg, Tq}},
    x::AbstractVector{Tg},
    xq::AbstractVector{S},
    ::Val{:linear};
    wrap::Bool=false,
    searcher::Searcher=_to_searcher(LinearBinary())
) where {Tg<:AbstractFloat, Tq<:Real, S<:Real}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"

    @inbounds for k in eachindex(xq)
        # Promote query point: preserves precision when S is wider than Tg
        xq_promoted = _promote_for_anchor(xq[k], Tg)
        buffer[k] = _linear_anchor_query_impl(x, xq_promoted, wrap, searcher)
    end
    return buffer
end

"""
    _linear_anchor_query_impl(x, xq, wrap, policy) -> _LinearAnchoredQuery

Internal implementation of _anchor_query for linear interpolation.

# Arguments
- `x`: Grid points (type Tg)
- `xq`: Query point (type Tq, can differ from Tg for precision preservation)
- `wrap`: Whether to wrap query point to domain
- `policy`: Search policy for interval search (default: DEFAULT_SEARCHER)

# AD Support
When `xq` is a ForwardDiff.Dual, the returned anchor preserves the Dual type
in `xq` and `alpha` fields. The interval search uses `_extract_primal(xq)` for comparisons.
"""
@inline function _linear_anchor_query_impl(
    x::AbstractVector{Tg},
    xq::Tq,
    wrap::Bool,
    policy::P=DEFAULT_SEARCHER
) where {Tg<:AbstractFloat, Tq<:Real, P<:Searcher}
    x_min, x_max = first(x), last(x)

    # Use primal value for comparisons (supports ForwardDiff.Dual)
    xq_primal = _extract_primal(xq)

    # Handle wrapping (for extrap=:wrap mode)
    if wrap && (xq_primal < x_min || xq_primal >= x_max)
        xq = _wrap_to_domain(xq, x_min, x_max)
        xq_primal = _extract_primal(xq)
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
    h = xR - xL
    inv_h = one(Tg) / h
    # alpha preserves Dual type when xq is Dual
    alpha = (xq - xL) / h

    return _LinearAnchoredQuery{Tg, Tq}(idx, xq, side, h, inv_h, alpha)
end

# ========================================
# LinearInterpolant Evaluation with Anchor
# ========================================

"""
    (itp::LinearInterpolant)(aq::_LinearAnchoredQuery; deriv::Int=0)

Evaluate linear interpolant at anchored query point. Ultra-fast path that
skips interval search.

# Arguments
- `aq`: Pre-computed anchored query from `_anchor_query`
- `deriv`: Derivative order (0=value, 1=first derivative)

# Example
```julia
itp = linear_interp(x, y)
aq = _anchor_query(x, 0.5, Val(:linear))
val = itp(aq)           # Value
d1 = itp(aq; deriv=1)   # First derivative
```
"""
@inline function (itp::LinearInterpolant{Tg})(aq::_LinearAnchoredQuery{Tg}; deriv::Int=0) where {Tg<:AbstractFloat}
    @_dispatch_deriv deriv => op begin
        _linear_eval_with_anchor(itp, aq, op)
    end
end

@inline function _linear_eval_with_anchor(
    itp::LinearInterpolant{Tg},
    aq::_LinearAnchoredQuery{Tg},
    op::O
) where {Tg<:AbstractFloat, O<:AbstractEvalOp}
    # Handle extrapolation based on mode and side
    return _linear_anchor_dispatch(itp, aq, op, itp.extrap)
end

# Default case (extension, wrap): direct anchored kernel evaluation
@inline function _linear_anchor_dispatch(
    itp::LinearInterpolant{Tg},
    aq::_LinearAnchoredQuery{Tg, Tq},
    op::AbstractEvalOp,
    ::Val
) where {Tg<:AbstractFloat, Tq<:Real}
    @inbounds begin
        yL = itp.y[aq.idx]
        yR = itp.y[aq.idx + 1]
    end
    return _linear_kernel(op, yL, yR, aq)
end

# No extrapolation: throw DomainError if outside domain
@inline function _linear_anchor_dispatch(
    itp::LinearInterpolant{Tg},
    aq::_LinearAnchoredQuery{Tg, Tq},
    op::AbstractEvalOp,
    ::Val{:none}
) where {Tg<:AbstractFloat, Tq<:Real}
    if aq.side != 0x00  # outside domain
        x_min, x_max = first(itp.x), last(itp.x)
        throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
    end
    @inbounds begin
        yL = itp.y[aq.idx]
        yR = itp.y[aq.idx + 1]
    end
    return _linear_kernel(op, yL, yR, aq)
end

# Constant extrapolation: boundary handling
@inline function _linear_anchor_dispatch(
    itp::LinearInterpolant{Tg},
    aq::_LinearAnchoredQuery{Tg, Tq},
    op::AbstractEvalOp,
    ::Val{:constant}
) where {Tg<:AbstractFloat, Tq<:Real}
    if aq.side != 0x00  # outside domain
        return _linear_eval_constant_extrap(itp.y, aq.side == 0x01, op)
    end
    @inbounds begin
        yL = itp.y[aq.idx]
        yR = itp.y[aq.idx + 1]
    end
    return _linear_kernel(op, yL, yR, aq)
end

# ========================================
# Vector Evaluation with Anchors
# ========================================

"""
    (itp::LinearInterpolant)(aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}}; deriv::Int=0)

Evaluate linear interpolant at multiple anchored query points.
Returns newly allocated vector with output type promoted from Tv and Tq.
"""
function (itp::LinearInterpolant{Tg,Tv})(
    aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg,Tq}};
    deriv::Int=0
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    T_out = promote_type(Tv, Tq)
    output = Vector{T_out}(undef, length(aq_vec))
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(aq_vec)
            output[i] = _linear_eval_with_anchor(itp, aq_vec[i], op)
        end
    end
    return output
end

"""
    (itp::LinearInterpolant)(output::AbstractVector, aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}}; deriv::Int=0)

In-place evaluation at multiple anchored query points. Zero allocation.
"""
function (itp::LinearInterpolant{Tg})(
    output::AbstractVector,
    aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}};
    deriv::Int=0
) where {Tg<:AbstractFloat}
    @assert length(output) == length(aq_vec) "output length must match aq_vec length"
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(aq_vec)
            output[i] = _linear_eval_with_anchor(itp, aq_vec[i], op)
        end
    end
    return output
end
