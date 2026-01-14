# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  SERIES KERNEL DISPATCH                                   ║
# ║         Parameterized SIMD kernels for method-specific evaluation         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: ... → series_callable.jl → series_kernels.jl
#
# Design: Val-based method dispatch for compile-time specialization
# - Each interpolation method has a unique kernel formula
# - Kernels are inlined and use muladd for FMA optimization
# - Coefficient gathering is method-specific for optimal memory access
#
# Method characteristics:
#   Cubic:     4 coeffs (yL, yR, zL, zR), 4 weights
#   Quadratic: 3 coeffs (y0, y1, y2), 3 weights
#   Linear:    2 coeffs (yL, yR), 2 weights
#   Constant:  1 coeff (y), 0 weights
#

# ════════════════════════════════════════════════════════════════════════════
# METHOD KIND TRAIT
# ════════════════════════════════════════════════════════════════════════════

"""
    _method_kind(::Type{<:AbstractSeriesInterpolant}) -> Val{:method}

Return the interpolation method kind for compile-time dispatch.

Each concrete SeriesInterpolant type should implement this trait:
- `_method_kind(::Type{<:CubicSeriesInterpolant}) = Val(:cubic)`
- `_method_kind(::Type{<:LinearSeriesInterpolant}) = Val(:linear)`
- etc.
"""
_method_kind(::Type{<:AbstractSeriesInterpolant}) = error("_method_kind not implemented")

# ════════════════════════════════════════════════════════════════════════════
# KERNEL EVALUATION - CUBIC
# ════════════════════════════════════════════════════════════════════════════

"""
    _eval_kernel(coeffs::NTuple{4,T}, weights::NTuple{4,T}, ::Val{:cubic}) -> T

Evaluate cubic spline kernel using 4 coefficients and 4 weights.

# Formula
```
result = wyR*yR + wyL*yL + wzR*zR + wzL*zL
```

Uses muladd for optimal FMA chain (3 FMAs + 1 mul).
"""
@inline function _eval_kernel(
    coeffs::NTuple{4,T},
    weights::NTuple{4,T},
    ::Val{:cubic}
) where {T<:AbstractFloat}
    yL, yR, zL, zR = coeffs
    wyL, wyR, wzL, wzR = weights
    # Optimal FMA chain: 3 FMAs + 1 mul
    return muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
end

# ════════════════════════════════════════════════════════════════════════════
# KERNEL EVALUATION - LINEAR
# ════════════════════════════════════════════════════════════════════════════

"""
    _eval_kernel(coeffs::NTuple{2,T}, weights::NTuple{2,T}, ::Val{:linear}) -> T

Evaluate linear interpolation kernel using 2 coefficients and 2 weights.

# Formula
```
result = wyR*yR + wyL*yL
```
"""
@inline function _eval_kernel(
    coeffs::NTuple{2,T},
    weights::NTuple{2,T},
    ::Val{:linear}
) where {T<:AbstractFloat}
    yL, yR = coeffs
    wyL, wyR = weights
    return muladd(wyR, yR, wyL * yL)
end

# ════════════════════════════════════════════════════════════════════════════
# KERNEL EVALUATION - CONSTANT
# ════════════════════════════════════════════════════════════════════════════

"""
    _eval_kernel(coeffs::NTuple{1,T}, weights::Tuple{}, ::Val{:constant}) -> T

Evaluate constant interpolation kernel - simply returns the looked-up value.

No weights needed for constant interpolation.
"""
@inline function _eval_kernel(
    coeffs::NTuple{1,T},
    ::Tuple{},
    ::Val{:constant}
) where {T<:AbstractFloat}
    return coeffs[1]
end

# ════════════════════════════════════════════════════════════════════════════
# KERNEL EVALUATION - QUADRATIC
# ════════════════════════════════════════════════════════════════════════════

"""
    _eval_kernel(coeffs::NTuple{3,T}, weights::NTuple{3,T}, ::Val{:quadratic}) -> T

Evaluate quadratic interpolation kernel using 3 coefficients and 3 weights.

# Formula
```
result = w0*y0 + w1*y1 + w2*y2
```
"""
@inline function _eval_kernel(
    coeffs::NTuple{3,T},
    weights::NTuple{3,T},
    ::Val{:quadratic}
) where {T<:AbstractFloat}
    y0, y1, y2 = coeffs
    w0, w1, w2 = weights
    return muladd(w2, y2, muladd(w1, y1, w0 * y0))
end

# ════════════════════════════════════════════════════════════════════════════
# COEFFICIENT GATHERING - POINT LAYOUT
# ════════════════════════════════════════════════════════════════════════════

"""
    _gather_coefficients_point(matrices, k::Int, idx::Int, ::Val{:method}) -> NTuple{N,T}

Gather coefficients for series k at index idx from point-contiguous matrices.

Point layout: matrices are (n_series × n_points), so column access is contiguous.
This is optimal for SIMD vectorization over the series dimension.
"""

# --- Cubic: 4 coefficients from 2 matrices ---

@inline function _gather_coefficients_point(
    matrices::Tuple{Matrix{T}, Matrix{T}},
    k::Int,
    idx::Int,
    ::Val{:cubic}
) where {T}
    y_point, z_point = matrices
    idx1 = idx + 1
    @inbounds begin
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
    end
    return (yL, yR, zL, zR)
end

# --- Linear: 2 coefficients from 1 matrix ---

@inline function _gather_coefficients_point(
    matrices::Tuple{Matrix{T}},
    k::Int,
    idx::Int,
    ::Val{:linear}
) where {T}
    y_point = matrices[1]
    @inbounds begin
        yL = y_point[k, idx]
        yR = y_point[k, idx + 1]
    end
    return (yL, yR)
end

# --- Constant: 1 coefficient from 1 matrix ---

@inline function _gather_coefficients_point(
    matrices::Tuple{Matrix{T}},
    k::Int,
    idx::Int,
    ::Val{:constant}
) where {T}
    y_point = matrices[1]
    @inbounds return (y_point[k, idx],)
end

# --- Quadratic: 3 coefficients from 1 matrix ---

@inline function _gather_coefficients_point(
    matrices::Tuple{Matrix{T}},
    k::Int,
    idx::Int,
    ::Val{:quadratic}
) where {T}
    y_point = matrices[1]
    # Quadratic uses 3 points centered at idx (or left-biased for boundaries)
    # For interior: idx-1, idx, idx+1
    # This may need adjustment based on actual quadratic anchor implementation
    @inbounds begin
        y0 = y_point[k, idx]
        y1 = y_point[k, idx + 1]
        y2 = y_point[k, idx + 2]
    end
    return (y0, y1, y2)
end
