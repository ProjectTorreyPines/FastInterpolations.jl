# ========================================
# Constant Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for constant (step) interpolation.
# No dependencies - can be tested independently.
#
# Unified signature: _constant_kernel(op, y_left, y_right, h, dL, side)
# - h = x_{i+1} - x_i (interval width, Tg)
# - dL = xq - x_i (offset from left boundary, can be Dual for AD)
# - y_left, y_right = values (Tv, can be Complex)
# - side = NearestSide() | LeftSide() | RightSide()
#
# Type parameters:
# - Tg<:AbstractFloat: Grid type (geometry)
# - Tv: Value type (can be Tg, Complex{Tg}, or other Number)
#
# AD Support:
# - dL can be ForwardDiff.Dual for automatic differentiation
# - Comparisons use _extract_primal(dL) to get Float value
# - Output is always Tv (no AD propagation through constant interp - derivative is 0)
#
# Grid point behavior: When dL == 0 (exactly at grid point),
# all side modes return y_left (the value at that grid point).

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::LeftSide)

Constant interpolation with left-continuous (floor) convention.
Always returns the left boundary value `y_left`.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, ::Tv, ::Tg, dL::Td, ::LeftSide) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return y_left
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::RightSide)

Constant interpolation with right-continuous (ceiling) convention.
Returns `y_left` at grid point (dL == 0), `y_right` otherwise.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, ::Tg, dL::Td, ::RightSide) where {Tv, Tg<:AbstractFloat, Td<:Real}
    # Use primal value for comparison (supports ForwardDiff.Dual)
    dL_primal = _extract_primal(dL)
    return iszero(dL_primal) ? y_left : y_right
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::NearestSide)

Constant interpolation with nearest-neighbor convention and left tie-breaking.
Returns `y_left` if dL <= h/2 (including midpoint), `y_right` otherwise.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, h::Tg, dL::Td, ::NearestSide) where {Tv, Tg<:AbstractFloat, Td<:Real}
    # Use primal value for comparison (supports ForwardDiff.Dual)
    dL_primal = _extract_primal(dL)
    return dL_primal <= h / 2 ? y_left : y_right
end

"""
    _constant_kernel(::EvalDeriv1, y_left, y_right, h, dL, side)

First derivative of constant interpolation.
Always returns zero (constant function has no slope).
Returns `zero(Tv)` to preserve Complex type when applicable.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalDeriv1, y_left::Tv, ::Tv, ::Tg, dL::Td, ::AbstractSide) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return zero(Tv)
end

"""
    _constant_kernel(::EvalDeriv2, y_left, y_right, h, dL, side)

Second derivative of constant interpolation.
Always returns zero (constant function has no curvature).
Returns `zero(Tv)` to preserve Complex type when applicable.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalDeriv2, y_left::Tv, ::Tv, ::Tg, dL::Td, ::AbstractSide) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return zero(Tv)
end

"""
    _constant_kernel(::EvalDeriv3, y_left, y_right, h, dL, side) -> zero(Tv)

Third derivative of constant interpolation is always zero.
Constant functions have all derivatives equal to zero.
Returns `zero(Tv)` to preserve Complex type when applicable.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalDeriv3, y_left::Tv, ::Tv, ::Tg, dL::Td, ::AbstractSide) where {Tv, Tg<:AbstractFloat, Td<:Real}
    return zero(Tv)
end
