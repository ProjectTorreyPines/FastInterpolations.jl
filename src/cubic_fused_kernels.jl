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
