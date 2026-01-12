# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC FUSED INTERPOLANT KERNELS                         ║
# ║         Evaluation kernels for CubicMultiInterpolantFused                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides callable methods for CubicMultiInterpolantFused:
# - Scalar evaluation: mitp(xq), mitp(out, xq)
# - Vector evaluation: mitp(xq_vec), mitp(outputs, xq_vec) [Phase 7]
# - Anchored evaluation: mitp(out, aq_vec) [Phase 8]
#
# Include order: ... → cubic_fused_interp.jl → cubic_fused_kernels.jl
#

# ========================================
# Helper Functions
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(mitp::CubicMultiInterpolantFused) = mitp.extrap === Val(:wrap)

# ========================================
# Core Evaluation Kernel
# ========================================

"""
    _eval_fused_scalar!(out, mitp, aq, op) -> out

Core SIMD-friendly kernel for fused multi-series evaluation.

Evaluates all series at a single anchored query point using the 4-term dot product:
    S_k(xq) = wyL*y[k,idx] + wyR*y[k,idx+1] + wzL*z[k,idx] + wzR*z[k,idx+1]

# Arguments
- `out::AbstractVector{T}`: Output vector of length n_series
- `mitp::CubicMultiInterpolantFused{T}`: Fused interpolant
- `aq::_CubicAnchoredQuery{T}`: Anchored query with precomputed weights
- `op::AbstractEvalOp`: Evaluation operation (EvalValue, EvalDeriv1, EvalDeriv2)

# Performance
Uses `@inbounds @simd` for vectorized evaluation across series.
The interleaved memory layout `y[series, point]` ensures contiguous column access.
"""
@inline function _eval_fused_scalar!(
    out::AbstractVector{T},
    mitp::CubicMultiInterpolantFused{T},
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    idx = aq.idx
    wyL, wyR, wzL, wzR = _anchored_weights(aq, op)

    # SIMD loop over series (contiguous column access)
    @inbounds @simd for k in axes(out, 1)
        yL = mitp.y[k, idx]
        yR = mitp.y[k, idx + 1]
        zL = mitp.z[k, idx]
        zR = mitp.z[k, idx + 1]

        # Optimal FMA chain: 3 FMAs + 1 mul
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end

    return out
end

# ========================================
# Scalar API - In-Place
# ========================================

"""
    (mitp::CubicMultiInterpolantFused)(out, xq; deriv=0)

Evaluate fused multi-series interpolant at scalar query point (in-place).

# Arguments
- `out::AbstractVector{T}`: Pre-allocated output vector of length n_series
- `xq::Real`: Query point
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Returns
The same `out` vector, filled with interpolated values.

# Example
```julia
mitp = cubic_interp_fused(x, [y1, y2, y3])
out = zeros(3)
mitp(out, 0.5)  # Fills out with [S1(0.5), S2(0.5), S3(0.5)]
```
"""
function (mitp::CubicMultiInterpolantFused{T})(
    out::AbstractVector{T},
    xq::S;
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    # Validate output length
    if length(out) != mitp.n_series
        throw(DimensionMismatch(
            "output length $(length(out)) must match n_series $(mitp.n_series)"
        ))
    end

    xq_typed = T(xq)

    # Build anchor
    aq = _anchor_query(mitp.x, xq_typed; wrap=_should_wrap(mitp))

    # Dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        # Handle inside vs outside domain
        if aq.side == 0x00
            # Fast path: inside domain
            _eval_fused_scalar!(out, mitp, aq, op)
        else
            # Outside domain: use extrapolation handler
            _eval_fused_with_extrap!(out, mitp, aq, op)
        end
    end

    return out
end

# ========================================
# Scalar API - Out-of-Place
# ========================================

"""
    (mitp::CubicMultiInterpolantFused)(xq; deriv=0)

Evaluate fused multi-series interpolant at scalar query point (allocating).

# Arguments
- `xq::Real`: Query point
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Returns
Vector{T} of length n_series with interpolated values.

# Example
```julia
mitp = cubic_interp_fused(x, [y1, y2, y3])
vals = mitp(0.5)  # Returns [S1(0.5), S2(0.5), S3(0.5)]
```
"""
function (mitp::CubicMultiInterpolantFused{T})(xq::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    out = Vector{T}(undef, mitp.n_series)
    return mitp(out, xq; deriv=deriv)
end

# ========================================
# Extrapolation Handling
# ========================================

"""
    _eval_fused_with_extrap!(out, mitp, aq, op)

Dispatch extrapolation based on mitp.extrap mode.
Called when aq.side != 0x00 (outside domain).
"""
@inline function _eval_fused_with_extrap!(
    out::AbstractVector{T},
    mitp::CubicMultiInterpolantFused{T},
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    _eval_fused_extrap!(out, mitp, aq, op, mitp.extrap, aq.side)
end

# :none - throw DomainError
@inline function _eval_fused_extrap!(
    ::AbstractVector{T},
    mitp::CubicMultiInterpolantFused{T},
    aq::_CubicAnchoredQuery{T},
    ::AbstractEvalOp,
    ::Val{:none},
    ::UInt8
) where {T<:AbstractFloat}
    x_min, x_max = first(mitp.x), last(mitp.x)
    throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
end

# :constant for value - return boundary values
@inline function _eval_fused_extrap!(
    out::AbstractVector{T},
    mitp::CubicMultiInterpolantFused{T},
    aq::_CubicAnchoredQuery{T},
    ::EvalValue,
    ::Val{:constant},
    side::UInt8
) where {T<:AbstractFloat}
    col = side == 0x01 ? 1 : mitp.n_points
    @inbounds @simd for k in axes(out, 1)
        out[k] = mitp.y[k, col]
    end
    return out
end

# :constant for derivatives - return zeros
@inline function _eval_fused_extrap!(
    out::AbstractVector{T},
    ::CubicMultiInterpolantFused{T},
    ::_CubicAnchoredQuery{T},
    ::EvalDeriv1,
    ::Val{:constant},
    ::UInt8
) where {T<:AbstractFloat}
    fill!(out, zero(T))
    return out
end

@inline function _eval_fused_extrap!(
    out::AbstractVector{T},
    ::CubicMultiInterpolantFused{T},
    ::_CubicAnchoredQuery{T},
    ::EvalDeriv2,
    ::Val{:constant},
    ::UInt8
) where {T<:AbstractFloat}
    fill!(out, zero(T))
    return out
end

# :extension - use precomputed weights (polynomial extrapolation from boundary)
@inline function _eval_fused_extrap!(
    out::AbstractVector{T},
    mitp::CubicMultiInterpolantFused{T},
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp,
    ::Val{:extension},
    ::UInt8
) where {T<:AbstractFloat}
    # Anchor already has clamped idx and computed weights
    return _eval_fused_scalar!(out, mitp, aq, op)
end

# :wrap - use precomputed weights (already wrapped at anchor construction)
@inline function _eval_fused_extrap!(
    out::AbstractVector{T},
    mitp::CubicMultiInterpolantFused{T},
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp,
    ::Val{:wrap},
    ::UInt8
) where {T<:AbstractFloat}
    # Anchor already wrapped coordinates and has correct weights
    return _eval_fused_scalar!(out, mitp, aq, op)
end

# ========================================
# Vector API - Multiple Query Points
# ========================================

"""
    (mitp::CubicMultiInterpolantFused)(outputs, xq_vec; deriv=0)

Evaluate fused multi-series interpolant at multiple query points (in-place).

# Arguments
- `outputs::AbstractVector{<:AbstractVector{T}}`: Pre-allocated output vectors
  - Length must equal n_series
  - Each vector must have length equal to length(xq_vec)
- `xq_vec::AbstractVector`: Query points
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Output Layout
`outputs[k][j]` = value of series k at query point xq_vec[j] (series-first layout)

# Returns
The same `outputs` vector of vectors.

# Example
```julia
mitp = cubic_interp_fused(x, [y1, y2, y3])
xq = [0.1, 0.3, 0.5, 0.7, 0.9]
outputs = [zeros(5) for _ in 1:3]
mitp(outputs, xq)  # Fills outputs with interpolated values
```
"""
@with_pool pool function (mitp::CubicMultiInterpolantFused{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(xq)
    n_series = mitp.n_series

    # Validate dimensions
    if length(outputs) != n_series
        throw(DimensionMismatch(
            "outputs length $(length(outputs)) must match n_series $n_series"
        ))
    end
    for (k, out_k) in enumerate(outputs)
        if length(out_k) != n_query
            throw(DimensionMismatch(
                "outputs[$k] length $(length(out_k)) must match n_query $n_query"
            ))
        end
    end

    # Acquire temporary buffer for scalar evaluation
    tmp = acquire!(pool, T, n_series)

    wrap = _should_wrap(mitp)

    @_dispatch_deriv deriv => op begin
        @inbounds for j in 1:n_query
            # Build anchor for this query point
            aq = _anchor_query(mitp.x, xq[j]; wrap=wrap)

            # Evaluate all series at this query
            if aq.side == 0x00
                _eval_fused_scalar!(tmp, mitp, aq, op)
            else
                _eval_fused_with_extrap!(tmp, mitp, aq, op)
            end

            # Scatter to outputs: tmp[k] → outputs[k][j]
            for k in 1:n_series
                outputs[k][j] = tmp[k]
            end
        end
    end

    return outputs
end

"""
    (mitp::CubicMultiInterpolantFused)(xq_vec; deriv=0)

Evaluate fused multi-series interpolant at multiple query points (allocating).

# Arguments
- `xq_vec::AbstractVector`: Query points
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Returns
`Vector{Vector{T}}` with layout `outputs[k][j]` = series k at xq_vec[j]

# Example
```julia
mitp = cubic_interp_fused(x, [y1, y2, y3])
outputs = mitp([0.1, 0.3, 0.5])  # Returns 3 vectors, each of length 3
```
"""
function (mitp::CubicMultiInterpolantFused{T})(
    xq::AbstractVector{T};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(xq)
    outputs = [Vector{T}(undef, n_query) for _ in 1:mitp.n_series]
    return mitp(outputs, xq; deriv=deriv)
end

# Real type wrapper for vector queries
function (mitp::CubicMultiInterpolantFused{T})(
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = T.(xq)
    return mitp(xq_typed; deriv=deriv)
end

# ========================================
# Pre-built Anchor API
# ========================================

"""
    (mitp::CubicMultiInterpolantFused)(outputs, aq_vec; deriv=0)

Evaluate fused multi-series interpolant using pre-built anchored queries (in-place).

This is the zero-allocation hot-loop API. Build anchors once with `_fill_anchors!`,
then reuse across multiple evaluation calls.

# Arguments
- `outputs::AbstractVector{<:AbstractVector{T}}`: Pre-allocated output vectors
  - Length must equal n_series
  - Each vector must have length equal to length(aq_vec)
- `aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}}`: Pre-built anchored queries
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Output Layout
`outputs[k][j]` = value of series k at anchored query aq_vec[j]

# Returns
The same `outputs` vector of vectors.

# Zero-Allocation Pattern
```julia
mitp = cubic_interp_fused(x, [y1, y2, y3])
xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]

# Pre-allocate anchors (once)
aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
FastInterpolations._fill_anchors!(aq_vec, mitp.x, xq_vec; wrap=false)

# Pre-allocate outputs (once)
outputs = [zeros(5) for _ in 1:3]

# Hot loop: zero allocations after warmup
for iteration in 1:1000
    mitp(outputs, aq_vec)  # Reuses pre-built anchors
end
```
"""
@with_pool pool function (mitp::CubicMultiInterpolantFused{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(aq_vec)
    n_series = mitp.n_series

    # Validate dimensions
    if length(outputs) != n_series
        throw(DimensionMismatch(
            "outputs length $(length(outputs)) must match n_series $n_series"
        ))
    end
    for (k, out_k) in enumerate(outputs)
        if length(out_k) != n_query
            throw(DimensionMismatch(
                "outputs[$k] length $(length(out_k)) must match n_query $n_query"
            ))
        end
    end

    # Acquire temporary buffer for scalar evaluation
    tmp = acquire!(pool, T, n_series)

    @_dispatch_deriv deriv => op begin
        @inbounds for j in 1:n_query
            aq = aq_vec[j]

            # Evaluate all series at this anchor
            if aq.side == 0x00
                _eval_fused_scalar!(tmp, mitp, aq, op)
            else
                _eval_fused_with_extrap!(tmp, mitp, aq, op)
            end

            # Scatter to outputs: tmp[k] → outputs[k][j]
            for k in 1:n_series
                outputs[k][j] = tmp[k]
            end
        end
    end

    return outputs
end
