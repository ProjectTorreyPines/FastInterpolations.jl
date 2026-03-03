# ========================================
# Linear Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for linear interpolation.
# No dependencies - can be tested independently.
#
# Unified API with two signatures:
# 1. Standard: _linear_kernel(op, yL, yR, h, dL)
#    - Used for oneshot and non-anchored evaluation
#    - Computes α = dL/h internally
#
# 2. Anchored: _linear_kernel(op, yL, yR, aq::_LinearAnchoredQuery)
#    - Defined in linear_anchor.jl (after anchor type)
#    - Uses precomputed alpha/inv_h from anchor
#
# TYPE PARAMETERS:
# - Tg: Grid type (AbstractFloat) - for h (interval width)
# - Tv: Value type - for yL, yR (can be Complex{Tg}, Tg, etc.)
# - Td: Offset type - for dL (can be Tg or Dual{Tg} for AD support)

# ========================================
# Standard Kernel (h, dL signature)
# ========================================

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
    return 0 * yL
end

"""
    _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, h::Tg, dL) where {Tg, Tv}

Third derivative of linear interpolation is always zero.
Linear functions have constant first derivative (slope), zero second and third derivatives.
"""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, ::Tv, h::Tg, dL) where {Tg<:AbstractFloat, Tv}
    return 0 * yL
end

