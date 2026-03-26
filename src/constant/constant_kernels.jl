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
# - Tv: Value type (unconstrained)
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
@inline function _constant_kernel(::EvalValue, y_left::Tv, ::Tv, ::Tg, dL::Td, ::LeftSide) where {Tv, Tg <: AbstractFloat, Td <: Real}
    return y_left
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::RightSide)

Constant interpolation with right-continuous (ceiling) convention.
Returns `y_left` at grid point (dL == 0), `y_right` otherwise.
dL can be any Real (including ForwardDiff.Dual for AD).
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, ::Tg, dL::Td, ::RightSide) where {Tv, Tg <: AbstractFloat, Td <: Real}
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
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, h::Tg, dL::Td, ::NearestSide) where {Tv, Tg <: AbstractFloat, Td <: Real}
    # Use primal value for comparison (supports ForwardDiff.Dual)
    dL_primal = _extract_primal(dL)
    return dL_primal <= h / 2 ? y_left : y_right
end

"""
    _constant_kernel(::EvalDeriv1, y_left, y_right, h, dL, side)

First derivative of constant interpolation.
Always returns zero (constant function has no slope).
Uses `0 * y_left` for duck-typing support and NaN propagation.
"""
@inline function _constant_kernel(::EvalDeriv1, y_left::Tv, ::Tv, ::Tg, dL::Td, ::AbstractSide) where {Tv, Tg <: AbstractFloat, Td <: Real}
    return 0 * y_left
end

"""
    _constant_kernel(::EvalDeriv2, y_left, y_right, h, dL, side)

Second derivative of constant interpolation.
Always returns zero (constant function has no curvature).
"""
@inline function _constant_kernel(::EvalDeriv2, y_left::Tv, ::Tv, ::Tg, dL::Td, ::AbstractSide) where {Tv, Tg <: AbstractFloat, Td <: Real}
    return 0 * y_left
end

"""
    _constant_kernel(::EvalDeriv3, y_left, y_right, h, dL, side)

Third derivative of constant interpolation is always zero.
"""
@inline function _constant_kernel(::EvalDeriv3, y_left::Tv, ::Tv, ::Tg, dL::Td, ::AbstractSide) where {Tv, Tg <: AbstractFloat, Td <: Real}
    return 0 * y_left
end

"""Generic fallback: N-th derivative of degree-0 (constant) is zero for N ≥ 1."""
@inline function _constant_kernel(::DerivOp{N}, y_left::Tv, ::Tv, ::Tg, ::Td, ::AbstractSide) where {N, Tv, Tg <: AbstractFloat, Td <: Real}
    return 0 * y_left
end

# ========================================
# Side Offset Computation
# ========================================
# Given interval width h and distance from left boundary dL,
# compute the offset (0 or 1) determining which endpoint is selected.
# Used by both 1D adjoint scatter and ND kernel/adjoint.

@inline _compute_single_offset(::LeftSide, h, dL) = 0

@inline function _compute_single_offset(::RightSide, h, dL)
    dL_primal = _extract_primal(dL)
    return iszero(dL_primal) ? 0 : 1
end

@inline function _compute_single_offset(::NearestSide, h, dL)
    dL_primal = _extract_primal(dL)
    return dL_primal <= h / 2 ? 0 : 1
end
