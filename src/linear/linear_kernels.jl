# ========================================
# Linear Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for linear interpolation.
# No dependencies - can be tested independently.
#
# Unified signature: _linear_kernel(op, yL, yR, h, dL)
# - h = xR - xL (interval width, always Tg)
# - dL = xq - xL (offset from left boundary, can be Tg or Dual{Tg} for AD)
# - α = dL / h (computed internally for Value)
#
# TYPE PARAMETERS:
# - Tg: Grid type (AbstractFloat) - for h (interval width)
# - Tv: Value type - for yL, yR (can be Complex{Tg}, Tg, etc.)
# - dL type: Left unconstrained to support AD (Dual{Tg})
#
# The kernel uses Julia's natural type promotion:
# - α = dL / h: promotes to Dual if dL is Dual
# - result = yL + α*(yR-yL): promotes correctly for Complex*Float, Dual*Float, etc.

"""
    _linear_kernel(::EvalValue, yL::Tv, yR::Tv, h::Tg, dL) where {Tg, Tv}

Evaluate linear interpolation value at offset dL from left boundary.
Returns: yL * (1 - α) + yR * α where α = dL / h

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (for h)
- `Tv`: Value type (for yL, yR) - can be Complex{Tg}, Tg, etc.
- `dL`: Unconstrained - can be Tg or Dual{Tg} for AD support

# Return Type
promote_type(Tv, typeof(α)) - handles Complex*Float and Dual*Float correctly
"""
@inline function _linear_kernel(::EvalValue, yL::Tv, yR::Tv, h::Tg, dL) where {Tg<:AbstractFloat, Tv}
    α = dL / h
    return muladd(α, yR - yL, yL)  # yL + α*(yR-yL)
end

"""
    _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, h::Tg, dL) where {Tg, Tv}

Evaluate first derivative (slope) of linear interpolation.
Returns constant slope: (yR - yL) / h

For Complex values, returns the complex derivative.
For AD (Dual) queries, the derivative is still just the slope (no chain rule needed).
"""
@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, h::Tg, dL) where {Tg<:AbstractFloat, Tv}
    return (yR - yL) / h
end

"""
    _linear_kernel(::EvalDeriv2, yL::Tv, yR::Tv, h::Tg, dL) where {Tg, Tv}

Evaluate second derivative of linear interpolation.
Always returns zero (linear function has no curvature).

Note: Mathematically, the second derivative is a Dirac delta at knots,
but we return zero everywhere as a practical approximation.
"""
@inline function _linear_kernel(::EvalDeriv2, yL::Tv, ::Tv, h::Tg, dL) where {Tg<:AbstractFloat, Tv}
    # Return zero in the appropriate output type
    # For Complex values, zero(Complex{Tg}) = 0.0 + 0.0im
    return zero(promote_type(Tv, Tg))
end

"""
    _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, h::Tg, dL) where {Tg, Tv}

Third derivative of linear interpolation is always zero.
Linear functions have constant first derivative (slope), zero second and third derivatives.
"""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, ::Tv, h::Tg, dL) where {Tg<:AbstractFloat, Tv}
    return zero(promote_type(Tv, Tg))
end

# ========================================
# Alpha-based Kernels (Pre-normalized)
# ========================================
# These kernels take `alpha` directly (already normalized: α = dL/h).
# More efficient when alpha is precomputed in anchor, avoids division.

"""
    _linear_kernel_alpha(::EvalValue, yL, yR, alpha)

Evaluate linear interpolation using pre-normalized alpha.
Avoids division since alpha = dL/h is precomputed.
"""
@inline function _linear_kernel_alpha(::EvalValue, yL::Tv, yR::Tv, alpha::Tq) where {Tv, Tq<:Real}
    return muladd(alpha, yR - yL, yL)  # yL + α*(yR-yL)
end

"""
    _linear_kernel_alpha(::EvalDeriv1, yL, yR, h)

Evaluate first derivative (slope). Needs h, not alpha.
"""
@inline function _linear_kernel_alpha(::EvalDeriv1, yL::Tv, yR::Tv, h::Tg) where {Tg<:AbstractFloat, Tv}
    return (yR - yL) / h
end

"""Second/third derivatives are zero for linear."""
@inline function _linear_kernel_alpha(::EvalDeriv2, yL::Tv, ::Tv, ::Any) where {Tv}
    return zero(Tv)
end

@inline function _linear_kernel_alpha(::EvalDeriv3, yL::Tv, ::Tv, ::Any) where {Tv}
    return zero(Tv)
end
