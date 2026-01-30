# ========================================
# Constant Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for constant (step) interpolation.
# No dependencies - can be tested independently.
#
# Unified signature: _constant_kernel(op, y_left, y_right, h, dL, side)
# - h = x_{i+1} - x_i (interval width, Tg)
# - dL = xq - x_i (offset from left boundary, Tg)
# - y_left, y_right = values (Tv, can be Complex)
# - side = Val(:nearest) | Val(:left) | Val(:right)
#
# Type parameters:
# - Tg<:AbstractFloat: Grid type (geometry)
# - Tv: Value type (can be Tg, Complex{Tg}, or other Number)
#
# Grid point behavior: When dL == 0 (exactly at grid point),
# all side modes return y_left (the value at that grid point).

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::Val{:left})

Constant interpolation with left-continuous (floor) convention.
Always returns the left boundary value `y_left`.
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, ::Tv, ::Tg, ::Tg, ::Val{:left}) where {Tv, Tg<:AbstractFloat}
    return y_left
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::Val{:right})

Constant interpolation with right-continuous (ceiling) convention.
Returns `y_left` at grid point (dL == 0), `y_right` otherwise.
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, ::Tg, dL::Tg, ::Val{:right}) where {Tv, Tg<:AbstractFloat}
    return iszero(dL) ? y_left : y_right
end

"""
    _constant_kernel(::EvalValue, y_left, y_right, h, dL, ::Val{:nearest})

Constant interpolation with nearest-neighbor convention and left tie-breaking.
Returns `y_left` if dL <= h/2 (including midpoint), `y_right` otherwise.
"""
@inline function _constant_kernel(::EvalValue, y_left::Tv, y_right::Tv, h::Tg, dL::Tg, ::Val{:nearest}) where {Tv, Tg<:AbstractFloat}
    return dL <= h / 2 ? y_left : y_right
end

"""
    _constant_kernel(::EvalDeriv1, y_left, y_right, h, dL, side)

First derivative of constant interpolation.
Always returns zero (constant function has no slope).
Returns `zero(Tv)` to preserve Complex type when applicable.
"""
@inline function _constant_kernel(::EvalDeriv1, y_left::Tv, ::Tv, ::Tg, ::Tg, ::SideVal) where {Tv, Tg<:AbstractFloat}
    return zero(Tv)
end

"""
    _constant_kernel(::EvalDeriv2, y_left, y_right, h, dL, side)

Second derivative of constant interpolation.
Always returns zero (constant function has no curvature).
Returns `zero(Tv)` to preserve Complex type when applicable.
"""
@inline function _constant_kernel(::EvalDeriv2, y_left::Tv, ::Tv, ::Tg, ::Tg, ::SideVal) where {Tv, Tg<:AbstractFloat}
    return zero(Tv)
end

"""
    _constant_kernel(::EvalDeriv3, y_left, y_right, h, dL, side) -> zero(Tv)

Third derivative of constant interpolation is always zero.
Constant functions have all derivatives equal to zero.
Returns `zero(Tv)` to preserve Complex type when applicable.
"""
@inline function _constant_kernel(::EvalDeriv3, y_left::Tv, ::Tv, ::Tg, ::Tg, ::SideVal) where {Tv, Tg<:AbstractFloat}
    return zero(Tv)
end
