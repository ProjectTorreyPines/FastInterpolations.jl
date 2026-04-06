# ========================================
# Method Traits
# ========================================
# Trait dispatch for mapping between interpolation methods, constructors, and adjoint operators.
# Used by: AD extensions (ChainRulesCore, Enzyme), interp auto-dispatch.

# Union of all 4 one-shot interpolation functions
const _InterpMethod = Union{
    typeof(linear_interp), typeof(quadratic_interp),
    typeof(constant_interp), typeof(cubic_interp),
    typeof(cardinal_interp), typeof(pchip_interp), typeof(akima_interp),
}

# One-shot function → adjoint constructor
_adjoint_func(::typeof(linear_interp)) = linear_adjoint
_adjoint_func(::typeof(quadratic_interp)) = quadratic_adjoint
_adjoint_func(::typeof(constant_interp)) = constant_adjoint
_adjoint_func(::typeof(cubic_interp)) = cubic_adjoint
_adjoint_func(::typeof(cardinal_interp)) = cardinal_adjoint
_adjoint_func(::typeof(pchip_interp)) = pchip_adjoint
_adjoint_func(::typeof(akima_interp)) = akima_adjoint
# hermite_interp has 4 data args (x, y, dy, xq) vs generic 3 — NOT in _InterpMethod union.
# Uses a dedicated rrule in ChainRulesCoreExt. This trait is for introspection / future use.
_adjoint_func(::typeof(hermite_interp)) = hermite_adjoint

# Interpolant struct → adjoint constructor
_adjoint_func_from_itp(::CubicInterpolantND) = cubic_adjoint
_adjoint_func_from_itp(::QuadraticInterpolantND) = quadratic_adjoint
_adjoint_func_from_itp(::LinearInterpolantND) = linear_adjoint
_adjoint_func_from_itp(::ConstantInterpolantND) = constant_adjoint
_adjoint_func_from_itp(::HeteroInterpolantND) = hetero_adjoint

# Interpolant struct → adjoint kwargs (bc/extrap/side differ per type)
_adjoint_kwargs_from_itp(itp::CubicInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::QuadraticInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::LinearInterpolantND) = (extrap = itp.extraps,)
_adjoint_kwargs_from_itp(itp::ConstantInterpolantND) = (side = itp.sides, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::HeteroInterpolantND) = (methods = itp.methods, extrap = itp.extraps)
