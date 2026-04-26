# ========================================
# Linear Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for linear interpolation.
# No dependencies - can be tested independently.
#
# Unified API with two signatures:
# 1. Standard: _linear_kernel(op, yL, yR, inv_h, α)
#    - Caller computes α via grid-type-dispatched `_alpha_of(q, L, R, grid)`
#      (cached `*inv_h` for `_CachedRange`, direct `(q-L)/(R-L)` for plain
#      `AbstractVector`) and pulls `inv_h` separately. EvalValue uses α only;
#      EvalDeriv1 uses inv_h only — LLVM DCE removes the unused side per op.
#
# 2. Anchored: _linear_kernel(op, yL, yR, aq::_LinearAnchoredQuery)
#    - Defined in linear_anchor.jl (after anchor type)
#    - Uses precomputed alpha/inv_h from anchor
#
# TYPE PARAMETERS:
# - Tg: Grid type (AbstractFloat) - for inv_h (inverse interval width)
# - Tv: Value type - for yL, yR (can be Complex{Tg}, Tg, etc.)
# - α: Unconstrained — can be Tg or Dual{Tg} for AD support

# ========================================
# Standard Kernel (inv_h, dL signature)
# ========================================

"""
    _linear_kernel(::EvalValue, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Evaluate linear interpolation value at normalized cell coordinate α ∈ [0, 1].
Returns: yL + α*(yR - yL). `inv_h` is unused for value eval — DCE'd by LLVM
when the caller's `inv_h` extraction has no other live use.
"""
@inline function _linear_kernel(::EvalValue, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    return muladd(α, yR - yL, yL)  # yL + α*(yR-yL)
end

"""
    _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Evaluate first derivative (slope) of linear interpolation.
Returns constant slope: (yR - yL) * inv_h. `α` is unused — DCE'd.
"""
@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    return (yR - yL) * inv_h
end

"""
    _linear_kernel(::EvalDeriv2, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Evaluate second derivative of linear interpolation.
Always returns zero (linear function has no curvature).

Note: Mathematically, the second derivative is a Dirac delta at knots,
but we return zero everywhere as a practical approximation.
"""
@inline function _linear_kernel(::EvalDeriv2, yL::Tv, ::Tv, inv_h::Tg, α) where {Tg, Tv}
    return 0 * yL
end

"""
    _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Third derivative of linear interpolation is always zero.
Linear functions have constant first derivative (slope), zero second and third derivatives.
"""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, ::Tv, inv_h::Tg, α) where {Tg, Tv}
    return 0 * yL
end

"""Generic fallback: N-th derivative of degree-1 polynomial is zero for N ≥ 2."""
@inline function _linear_kernel(::DerivOp{N}, yL::Tv, ::Tv, ::Tg, α) where {N, Tg, Tv}
    return 0 * yL
end
