# ========================================
# Method Traits
# ========================================
# Trait dispatch for mapping between interpolation methods, constructors, and adjoint operators.
# Used by: AD extensions (ChainRulesCore, Enzyme), interp_nd auto-dispatch.

# Union of all 4 one-shot interpolation functions
const _InterpMethod = Union{
    typeof(linear_interp), typeof(quadratic_interp),
    typeof(constant_interp), typeof(cubic_interp),
}

# One-shot function → adjoint constructor
_adjoint_func(::typeof(linear_interp)) = linear_adjoint
_adjoint_func(::typeof(quadratic_interp)) = quadratic_adjoint
_adjoint_func(::typeof(constant_interp)) = constant_adjoint
_adjoint_func(::typeof(cubic_interp)) = cubic_adjoint

# Interpolant struct → adjoint constructor
_adjoint_func_from_itp(::CubicInterpolantND) = cubic_adjoint
_adjoint_func_from_itp(::QuadraticInterpolantND) = quadratic_adjoint
_adjoint_func_from_itp(::LinearInterpolantND) = linear_adjoint
_adjoint_func_from_itp(::ConstantInterpolantND) = constant_adjoint

# Interpolant struct → adjoint kwargs (bc/extrap/side differ per type)
_adjoint_kwargs_from_itp(itp::CubicInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::QuadraticInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::LinearInterpolantND) = (extrap = itp.extraps,)
_adjoint_kwargs_from_itp(itp::ConstantInterpolantND) = (side = itp.sides, extrap = itp.extraps)
