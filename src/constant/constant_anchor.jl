# ========================================
# Constant Anchored Query
# ========================================
# Precomputed geometry for ultra-fast constant interpolation evaluation
# at a fixed query point. Enables speedup by eliminating interval search
# for repeated evaluations.
#
# Include order: ops.jl → utils.jl → constant_kernels.jl → constant_interp.jl → constant_anchor.jl
#
# ========================================
# _ConstantAnchoredQuery Type (Internal)
# ========================================

"""
    _ConstantAnchoredQuery{T}

Precomputed geometry for ultra-fast constant interpolation at a fixed query point.
Internal API: no runtime grid validation; callers must ensure the anchor
matches the interpolant grid.

# Type Parameters
- `T`: Float type (Float32 or Float64)

# Fields
- `idx`: Interval index where xq falls
- `xq`: Original query point (or wrapped value for periodic)
- `side`: Domain position (0=inside, 1=below, 2=above)
- `h`: Interval width (for :nearest comparison)
- `dL`: Offset from left boundary (for :nearest comparison)

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
aq = _anchor_query(x, 0.35, Val(:constant))

itp1 = constant_interp(x, sin.(2π .* x))
itp2 = constant_interp(x, cos.(2π .* x))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq)              # Reuses same anchor
```

# Performance
Anchored evaluation is faster than `itp(xq)` for non-uniform grids,
as it eliminates O(log n) binary search.
"""
struct _ConstantAnchoredQuery{T<:AbstractFloat}
    idx::Int                   # interval index
    xq::T                      # query point (possibly wrapped)
    side::UInt8                # 0=inside, 1=below_min, 2=above_max
    h::T                       # interval width
    dL::T                      # offset from left boundary
end

# ========================================
# Anchor Construction
# ========================================

"""
    _anchor_query(x::AbstractVector{T}, xq::T, ::Val{:constant}; wrap::Bool=false) -> _ConstantAnchoredQuery

Create an anchored query for ultra-fast constant interpolation at a fixed point.

# Arguments
- `x`: Grid points (must match grid used for interpolant construction)
- `xq`: Query point (scalar)
- `::Val{:constant}`: Type tag to distinguish from other anchor types
- `wrap`: If true, wrap `xq` to domain [x[1], x[end]) before anchoring.
          Used for `extrap=WrapExtrap()` mode.

# Returns
`_ConstantAnchoredQuery{T}` with precomputed geometry.

# Example
```julia
x = range(0.0, 1.0, 101)
itp1 = constant_interp(collect(x), sin.(2π .* x))
itp2 = constant_interp(collect(x), cos.(2π .* x))

aq = _anchor_query(collect(x), 0.35, Val(:constant))

itp1(aq)              # Ultra-fast: skips interval search
itp2(aq)              # Reuses same anchor
```
"""
@inline function _anchor_query(
    x::AbstractVector{T},
    xq::T,
    ::Val{:constant},
    wrap::Bool=false,
    searcher::P=DEFAULT_SEARCHER
) where {T<:AbstractFloat, P<:Searcher}
    return _constant_anchor_query_impl(x, xq, wrap, _resolve_searcher_for_grid(x, searcher))
end

# Real wrapper for convenience (scalar)
@inline function _anchor_query(
    x::AbstractVector{T},
    xq::Tq,
    tag::Val{:constant},
    wrap::Bool=false,
    searcher::P=DEFAULT_SEARCHER
) where {T<:AbstractFloat, Tq<:Real, P<:Searcher}
    _anchor_query(x, T(xq), tag, wrap, searcher)
end

"""
    _anchor_query(x::AbstractVector{T}, xq::AbstractVector, ::Val{:constant}; wrap::Bool=false) -> Vector{_ConstantAnchoredQuery{T}}

Create anchored queries for multiple query points.

Internal API: No runtime grid validation. Caller must ensure `x` matches
the grid used for interpolant construction.

# Arguments
- `x`: Grid points (must match interpolant's grid)
- `xq`: Query points (any Real type, auto-promoted to T)
- `::Val{:constant}`: Type tag
- `wrap`: If true, wrap query points to domain [x[1], x[end]) before anchoring.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.75], Val(:constant))

itp1 = constant_interp(x, sin.(2π .* x))
itp2 = constant_interp(x, cos.(2π .* x))

vals1 = itp1(aq_vec)  # Batch evaluation
vals2 = itp2(aq_vec)  # Reuse same anchors
```
"""
function _anchor_query(
    x::AbstractVector{T},
    xq::AbstractVector{S},
    ::Val{:constant},
    wrap::Bool=false,
    searcher::P=_to_searcher(LinearBinary())
) where {T<:AbstractFloat, S<:Real, P<:Searcher}
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)
    output = Vector{_ConstantAnchoredQuery{T}}(undef, length(xq))

    @inbounds for k in eachindex(xq)
        output[k] = _constant_anchor_query_impl(x, T(xq[k]), wrap, searcher_resolved)
    end
    return output
end

"""
    _fill_anchors!(buffer, x, xq, ::Val{:constant}; wrap=false) -> buffer

Fill a pre-allocated buffer with anchored queries for constant interpolation.
In-place version of `_anchor_query(x, xq, Val(:constant))` for zero-allocation pooled usage.

# Arguments
- `buffer::Vector{_ConstantAnchoredQuery{T}}`: Pre-allocated buffer (length >= length(xq))
- `x::AbstractVector{T}`: Grid points (must match interpolant's grid)
- `xq::AbstractVector`: Query points (any Real type, auto-promoted to T)
- `::Val{:constant}`: Type tag for constant interpolation
- `wrap::Bool=false`: If true, wrap query points to domain [x[1], x[end])

# Returns
The same `buffer` object, filled with anchored queries.
"""
@inline function _fill_anchors!(
    buffer::AbstractVector{_ConstantAnchoredQuery{T}},
    x::AbstractVector{T},
    xq::AbstractVector{S},
    ::Val{:constant},
    wrap::Bool=false,
    searcher::P=_to_searcher(LinearBinary())
) where {T<:AbstractFloat, S<:Real, P<:Searcher}
    @assert length(buffer) >= length(xq) "Buffer too small: $(length(buffer)) < $(length(xq))"
    searcher_resolved = _resolve_searcher_for_grid(x, searcher)

    @inbounds for k in eachindex(xq)
        buffer[k] = _constant_anchor_query_impl(x, T(xq[k]), wrap, searcher_resolved)
    end
    return buffer
end

"""
    _constant_anchor_query_impl(x, xq, wrap, policy) -> _ConstantAnchoredQuery

Internal implementation of _anchor_query for constant interpolation.

# Arguments
- `x`: Grid points
- `xq`: Query point
- `wrap`: Whether to wrap query point to domain
- `policy`: Search policy for interval search (default: DEFAULT_SEARCHER)
"""
@inline function _constant_anchor_query_impl(
    x::AbstractVector{T},
    xq::T,
    wrap::Bool,
    policy::P=DEFAULT_SEARCHER
) where {T<:AbstractFloat, P<:Searcher}
    x_min, x_max = first(x), last(x)

    # Handle wrapping (for extrap=WrapExtrap() mode)
    if wrap && (xq < x_min || xq >= x_max)
        xq = _wrap_to_domain(xq, x_min, x_max)
    end

    # Determine side (domain position)
    side = if xq < x_min
        0x01  # below min
    elseif xq > x_max
        0x02  # above max
    else
        0x00  # inside
    end

    # Find interval and compute geometry
    # For outside-domain points, use boundary intervals
    idx, xL, xR = if xq < x_min
        # Below domain: use first interval
        @inbounds (1, x[1], x[2])
    elseif xq > x_max
        # Above domain: use last interval
        n = length(x)
        @inbounds (n - 1, x[n-1], x[n])
    else
        # Inside domain: use policy-based interval search
        search_interval(policy, x, xq)
    end

    # Compute geometry
    h = xR - xL
    dL = xq - xL

    return _ConstantAnchoredQuery{T}(idx, xq, side, h, dL)
end

# ========================================
# ConstantInterpolant Evaluation with Anchor
# ========================================

"""
    (itp::ConstantInterpolant)(aq::_ConstantAnchoredQuery; deriv::DerivOp=EvalValue())

Evaluate constant interpolant at anchored query point. Ultra-fast path that
skips interval search.

# Arguments
- `aq`: Pre-computed anchored query from `_anchor_query`
- `deriv`: Derivative order (0=value, 1=first derivative, 2=second derivative)
           Note: derivatives are always 0 for constant interpolation.

# Example
```julia
itp = constant_interp(x, y)
aq = _anchor_query(x, 0.5, Val(:constant))
val = itp(aq)           # Value
```
"""
@inline function (itp::ConstantInterpolant{T})(aq::_ConstantAnchoredQuery{T}; deriv::DerivOp=EvalValue()) where {T<:AbstractFloat}
    _constant_eval_with_anchor(itp, aq, deriv)
end

@inline function _constant_eval_with_anchor(
    itp::ConstantInterpolant{T},
    aq::_ConstantAnchoredQuery{T},
    op::O
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    # Handle extrapolation based on mode and side
    return _constant_anchor_dispatch(itp, aq, op, itp.extrap)
end

# No extrapolation: throw DomainError if outside domain
@inline function _constant_anchor_dispatch(
    itp::ConstantInterpolant{T},
    aq::_ConstantAnchoredQuery{T},
    op::O,
    ::NoExtrap
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    if aq.side != 0x00  # outside domain
        x_min, x_max = first(itp.x), last(itp.x)
        throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
    end
    # Inside domain
    if aq.xq == last(itp.x)
        return op isa EvalValue ? (@inbounds itp.y[end]) : zero(T)
    end
    @inbounds begin
        y_left = itp.y[aq.idx]
        y_right = itp.y[aq.idx + 1]
        return _constant_kernel(op, y_left, y_right, aq.h, aq.dL, itp.side)
    end
end

# Inside domain or extension mode: use interpolation
@inline function _constant_anchor_dispatch(
    itp::ConstantInterpolant{T},
    aq::_ConstantAnchoredQuery{T},
    op::O,
    ::AbstractExtrap
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    # Special case: at right boundary (x_max)
    if aq.xq == last(itp.x)
        return op isa EvalValue ? (@inbounds itp.y[end]) : zero(T)
    end
    @inbounds begin
        y_left = itp.y[aq.idx]
        y_right = itp.y[aq.idx + 1]
        return _constant_kernel(op, y_left, y_right, aq.h, aq.dL, itp.side)
    end
end

# Constant extrapolation: special handling for outside-domain
@inline function _constant_anchor_dispatch(
    itp::ConstantInterpolant{T},
    aq::_ConstantAnchoredQuery{T},
    op::O,
    ::ConstExtrap
) where {T<:AbstractFloat, O<:AbstractEvalOp}
    if aq.side == 0x01  # below domain
        return op isa EvalValue ? (@inbounds itp.y[1]) : zero(T)
    elseif aq.side == 0x02  # above domain
        return op isa EvalValue ? (@inbounds itp.y[end]) : zero(T)
    else
        # Inside domain
        if aq.xq == last(itp.x)
            return op isa EvalValue ? (@inbounds itp.y[end]) : zero(T)
        end
        @inbounds begin
            y_left = itp.y[aq.idx]
            y_right = itp.y[aq.idx + 1]
            return _constant_kernel(op, y_left, y_right, aq.h, aq.dL, itp.side)
        end
    end
end

# ========================================
# Vector Evaluation with Anchors
# ========================================

"""
    (itp::ConstantInterpolant)(aq_vec::AbstractVector{_ConstantAnchoredQuery{T}}; deriv::DerivOp=EvalValue())

Evaluate constant interpolant at multiple anchored query points.
Returns newly allocated vector.
"""
function (itp::ConstantInterpolant{T})(
    aq_vec::AbstractVector{_ConstantAnchoredQuery{T}};
    deriv::DerivOp=EvalValue()
) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(aq_vec))
    @inbounds for i in eachindex(aq_vec)
        output[i] = _constant_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end

"""
    (itp::ConstantInterpolant)(output::AbstractVector{T}, aq_vec::AbstractVector{_ConstantAnchoredQuery{T}}; deriv::DerivOp=EvalValue())

In-place evaluation at multiple anchored query points. Zero allocation.
"""
function (itp::ConstantInterpolant{T})(
    output::AbstractVector{T},
    aq_vec::AbstractVector{_ConstantAnchoredQuery{T}};
    deriv::DerivOp=EvalValue()
) where {T<:AbstractFloat}
    @assert length(output) == length(aq_vec) "output length must match aq_vec length"
    @inbounds for i in eachindex(aq_vec)
        output[i] = _constant_eval_with_anchor(itp, aq_vec[i], deriv)
    end
    return output
end
