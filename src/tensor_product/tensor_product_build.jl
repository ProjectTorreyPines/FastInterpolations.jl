# ========================================
# Heterogeneous ND Partial Derivative Build
# ========================================
# Precomputes all 2^N partial derivatives using the bit-encoding build-up algorithm,
# dispatching per-axis differentiation functions based on the interpolation method.
#
# - CubicInterp axes: Thomas tridiagonal solve (_differentiate_nd_along_dim_batch!)
# - QuadraticInterp axes: slope recurrence (_differentiate_nd_along_dim_quadratic!)
# - LinearInterp / ConstantInterp axes: identity copy (no differentiation needed)

# ========================================
# Per-Axis Differentiation Dispatch
# ========================================

@inline function _differentiate_axis!(dst, src, grid, bc, d, ::CubicInterp)
    return _differentiate_nd_along_dim_batch!(dst, src, grid, bc, d)
end

@inline function _differentiate_axis!(dst, src, grid, bc, d, ::QuadraticInterp)
    return _differentiate_nd_along_dim_quadratic!(dst, src, grid, bc, d)
end

@inline function _differentiate_axis!(dst, src, ::Any, ::Any, ::Int, ::LinearInterp)
    return copyto!(dst, src)
end

@inline function _differentiate_axis!(dst, src, ::Any, ::Any, ::Int, ::ConstantInterp)
    return copyto!(dst, src)
end

# ========================================
# Per-Axis Effective BC Dispatch
# ========================================

@inline _get_effective_bc_hetero(bc, p_src, grid, ::CubicInterp) =
    _get_effective_bc(bc, p_src, grid)

@inline _get_effective_bc_hetero(bc, p_src, grid, ::QuadraticInterp) =
    _get_effective_bc_quadratic(bc, p_src, grid)

@inline _get_effective_bc_hetero(bc, _, _, ::LinearInterp) = bc
@inline _get_effective_bc_hetero(bc, _, _, ::ConstantInterp) = bc

# ========================================
# BC Extraction from Method Types
# ========================================

@inline _extract_bc(m::CubicInterp) = m.bc
@inline _extract_bc(m::QuadraticInterp) = m.bc
@inline _extract_bc(::LinearInterp) = CubicFit()      # placeholder, not used by identity copy
@inline _extract_bc(::ConstantInterp) = CubicFit()     # placeholder, not used by identity copy

@inline _extract_bcs(methods::Tuple{Vararg{AbstractInterpMethod}}) = map(_extract_bc, methods)

# ========================================
# Heterogeneous Build-Up Algorithm
# ========================================
# Same bit-encoding as _build_nd_partials_dim! (cubic) / _build_nd_partials_dim_quadratic!,
# but dispatches per-axis differentiation function based on method type.

@inline _build_nd_partials_dim_hetero!(
    partials::AbstractArray{Tv, NP1},
    grids::NTuple{N, AbstractVector{Tg}},
    methods::Tuple{Vararg{AbstractInterpMethod, N}},
    bcs::Tuple{Vararg{AbstractBC, N}},
    ::Val{N},
) where {Tv, Tg <: AbstractFloat, N, NP1} =
    _build_nd_partials_dim_hetero!(partials, grids, methods, bcs, Val(1), Val(N))

@inline function _build_nd_partials_dim_hetero!(
        partials::AbstractArray{Tv, NP1},
        grids::NTuple{N, AbstractVector{Tg}},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        bcs::Tuple{Vararg{AbstractBC, N}},
        ::Val{D},
        ::Val{N},
    ) where {Tv, Tg <: AbstractFloat, D, N, NP1}
    bit_d = 1 << (D - 1)
    method_d = methods[D]
    @inbounds for p_src in 1:bit_d
        p_dst = p_src + bit_d
        effective_bc = _get_effective_bc_hetero(bcs[D], p_src, grids[D], method_d)
        src_view = selectdim(partials, 1, p_src)
        dst_view = selectdim(partials, 1, p_dst)
        _differentiate_axis!(dst_view, src_view, grids[D], effective_bc, D, method_d)
    end
    if D < N
        _build_nd_partials_dim_hetero!(partials, grids, methods, bcs, Val(D + 1), Val(N))
    end
    return partials
end

# ========================================
# Top-Level: Compute All Heterogeneous Partials
# ========================================

"""
    _compute_nd_partials_hetero!(partials, grids, data, methods, bcs)

Compute all 2^N partial derivatives for heterogeneous ND interpolation.

Uses the bit-encoding build-up algorithm with per-axis dispatch:
- CubicInterp: Thomas tridiagonal solve (global cubic spline slopes)
- QuadraticInterp: slope recurrence (quadratic slopes)
- LinearInterp/ConstantInterp: identity copy (derivative slots = value slots)
"""
function _compute_nd_partials_hetero!(
        partials::AbstractArray{Tv, NP1},
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        bcs::Tuple{Vararg{AbstractBC, N}},
    ) where {Tv, Tg <: AbstractFloat, N, NP1}
    # Validate dimensions
    @boundscheck begin
        NP1 == N + 1 || throw(DimensionMismatch("partials must have N+1 dimensions"))
        n_partials = 1 << N
        size(partials, 1) == n_partials || throw(
            DimensionMismatch(
                "partials first dimension must be 2^N=$(n_partials), got $(size(partials, 1))"
            )
        )
    end

    # Stage 0: Copy f (function values) into partials[1, ...]
    f_partial = selectdim(partials, 1, 1)
    copyto!(f_partial, data)

    # Build up higher-order partials stage by stage
    _build_nd_partials_dim_hetero!(partials, grids, methods, bcs, Val(N))

    return partials
end

"""
    _build_nd_coeffs_hetero(grids, data, methods) -> NodalDerivativesND

Allocate and compute heterogeneous partial derivatives. Returns NodalDerivativesND.
"""
function _build_nd_coeffs_hetero(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
    ) where {Tg <: AbstractFloat, Tv, N}
    bcs = _extract_bcs(methods)

    # Allocate partials array: (2^N, n₁, n₂, ..., nₙ)
    n_partials = 1 << N
    partials_shape = (n_partials, size(data)...)
    partials = Array{Tv, N + 1}(undef, partials_shape)

    # Compute all partial derivatives in-place
    _compute_nd_partials_hetero!(partials, grids, data, methods, bcs)

    return NodalDerivativesND{Tv, N, N + 1}(partials)
end
