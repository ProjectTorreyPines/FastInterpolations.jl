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
# - Tg: Grid type (geometry, unconstrained)
# - Tv: Value type (unconstrained)
#
# AD Support:
# - dL can be ForwardDiff.Dual for automatic differentiation
# - Comparisons use _extract_primal(dL) to get Float value
# - Multiplying the selection by `one(dL)` propagates `Tq`'s carrier (Dual,
#   Float, …) into the result while keeping the value unchanged — aligns
#   in-domain type with `_promote_extrap_val`'s OOB output. Derivative
#   branches use `0 * y_left * one(dL)` so they only require `*(::Int, ::Tv)`
#   and `*(::Tv, ::Real)` (no `zero(::Tv)` assumption beyond master's).
#
# Grid point behavior: When dL == 0 (exactly at grid point),
# all side modes return y_left (the value at that grid point).

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::LeftSide)

Constant interpolation with left-continuous (floor) convention.
Always returns the left boundary value `y_left`.
dL may be Real, Unitful.Quantity, or ForwardDiff.Dual — its carrier propagates via `one(dL)`.
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, ::Tv, ::Tg, dL::Td, ::LeftSide) where {Tv, Tg, Td}
    return y_left * one(dL)
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::RightSide)

Constant interpolation with right-continuous (ceiling) convention.
Returns `y_left` at grid point (dL == 0), `y_right` otherwise.
dL may be Real, Unitful.Quantity, or ForwardDiff.Dual — its carrier propagates via `one(dL)`.
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, ::Tg, dL::Td, ::RightSide) where {Tv, Tg, Td}
    # Use primal value for comparison (supports ForwardDiff.Dual)
    dL_primal = _extract_primal(dL)
    selected = iszero(dL_primal) ? y_left : y_right
    return selected * one(dL)
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::NearestSide)

Constant interpolation with nearest-neighbor convention and left tie-breaking.
Returns `y_left` if dL <= h/2 (including midpoint), `y_right` otherwise.
dL may be Real, Unitful.Quantity, or ForwardDiff.Dual — its carrier propagates via `one(dL)`.
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, h::Tg, dL::Td, ::NearestSide) where {Tv, Tg, Td}
    # Use primal value for comparison (supports ForwardDiff.Dual)
    dL_primal = _extract_primal(dL)
    selected = dL_primal <= h / 2 ? y_left : y_right
    return selected * one(dL)
end

# N-th derivative of degree-0 is zero for N ≥ 1, in `[value]/[grid]ᴺ` space.
# The unit scale comes from the GRID spacing (`_deriv_oneunit(h, op)`, matching
# the OOB branch) so query units can't flip the return type at the boundary;
# `* one(dL)` keeps the query carrier, `0 * y_left` keeps zero/duck/NaN
# semantics. Int grids float the derivative (Int/Int → Float) — correct, not a regression.
"""
    _constant_kernel(::EvalDeriv1, y_left, y_right, h, dL, side)

First derivative of constant interpolation — zero, in `value/grid` units.
"""
@inline function _constant_kernel(::EvalDeriv1, y_left::Tv, ::Tv, h::Tg, dL::Td, ::AbstractSide) where {Tv, Tg, Td}
    return 0 * y_left * _deriv_oneunit(h, DerivOp(1)) * one(dL)
end

"""
    _constant_kernel(::EvalDeriv2, y_left, y_right, h, dL, side)

Second derivative of constant interpolation — zero, in `value/grid²` units.
"""
@inline function _constant_kernel(::EvalDeriv2, y_left::Tv, ::Tv, h::Tg, dL::Td, ::AbstractSide) where {Tv, Tg, Td}
    return 0 * y_left * _deriv_oneunit(h, DerivOp(2)) * one(dL)
end

"""
    _constant_kernel(::EvalDeriv3, y_left, y_right, h, dL, side)

Third derivative of constant interpolation — zero, in `value/grid³` units.
"""
@inline function _constant_kernel(::EvalDeriv3, y_left::Tv, ::Tv, h::Tg, dL::Td, ::AbstractSide) where {Tv, Tg, Td}
    return 0 * y_left * _deriv_oneunit(h, DerivOp(3)) * one(dL)
end

# Generic fallback (N ≥ 4): `_deriv_oneunit` uses `Base.literal_pow` so the grid⁻ᴺ
# factor stays type-stable for unit grids even when N is a type parameter.
"""Generic fallback: N-th derivative of degree-0 (constant) is zero for N ≥ 1."""
@inline function _constant_kernel(::DerivOp{N}, y_left::Tv, ::Tv, h::Tg, dL::Td, ::AbstractSide) where {N, Tv, Tg, Td}
    return 0 * y_left * _deriv_oneunit(h, DerivOp(N)) * one(dL)
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
