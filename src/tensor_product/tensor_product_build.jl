# ========================================
# Heterogeneous ND Partial Derivative Build
# ========================================
# Precomputes partial derivatives using the bit-encoding build-up algorithm,
# dispatching per-axis differentiation functions based on the interpolation method.
#
# Uses compact storage: prod(sizes) partials instead of 2^N, where
# sizes[d] = 2 for derivative axes (Cubic/Quadratic), 1 for others.
# Non-derivative axes are skipped entirely in the build (no identity copy).

# ========================================
# Compile-Time Helpers
# ========================================

# Per-axis derivative sizes: 2 if derivative-based, 1 otherwise
@inline _deriv_size(::CubicInterp) = 2
@inline _deriv_size(::QuadraticInterp) = 2
@inline _deriv_size(::LinearInterp) = 1
@inline _deriv_size(::ConstantInterp) = 1

# For @generated: extract sizes from methods tuple TYPE
_is_deriv_method(::Type{<:CubicInterp}) = true
_is_deriv_method(::Type{<:QuadraticInterp}) = true
_is_deriv_method(::Type) = false

function _deriv_sizes(::Type{M}) where {M <: Tuple{Vararg{AbstractInterpMethod}}}
    N = length(M.parameters)
    return ntuple(d -> _is_deriv_method(M.parameters[d]) ? 2 : 1, N)
end

# ========================================
# Per-Axis Differentiation Dispatch
# ========================================

@inline function _differentiate_axis!(dst, src, grid, bc, d, ::CubicInterp)
    return _differentiate_nd_along_dim_batch!(dst, src, grid, bc, d)
end

@inline function _differentiate_axis!(dst, src, grid, bc, d, ::QuadraticInterp)
    return _differentiate_nd_along_dim_quadratic!(dst, src, grid, bc, d)
end

# LinearInterp / ConstantInterp: no differentiation (not called in compact build)

# ========================================
# Per-Axis Effective BC Dispatch
# ========================================

@inline _get_effective_bc_hetero(bc, p_src, grid, ::CubicInterp) =
    _get_effective_bc(bc, p_src, grid)

@inline _get_effective_bc_hetero(bc, p_src, grid, ::QuadraticInterp) =
    _get_effective_bc_quadratic(bc, p_src, grid)

# ========================================
# BC Extraction from Method Types
# ========================================

@inline _extract_bc(m::CubicInterp) = m.bc
@inline _extract_bc(m::QuadraticInterp) = m.bc
@inline _extract_bc(::LinearInterp) = CubicFit()      # placeholder, not used
@inline _extract_bc(::ConstantInterp) = CubicFit()     # placeholder, not used

@inline _extract_bcs(methods::Tuple{Vararg{AbstractInterpMethod}}) = map(_extract_bc, methods)

# ========================================
# Compact Build-Up Algorithm (Mixed-Radix)
# ========================================
# Uses compact strides: stride_d = prod(sizes[1:d-1]).
# Non-derivative axes (sizes[d]=1) are skipped entirely — no identity copy needed.
# Derivative axes are differentiated using the appropriate per-method solver.

@inline function _build_nd_partials_dim_hetero!(
        partials::AbstractArray{Tv, NP1},
        grids,
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        bcs::Tuple{Vararg{AbstractBC, N}},
        sizes::NTuple{N, Int},
        ::Val{D},
        ::Val{N},
    ) where {Tv, D, N, NP1}
    if sizes[D] == 2
        # Derivative axis: differentiate using compact stride
        stride_d = D == 1 ? 1 : prod(sizes[1:(D - 1)])
        method_d = methods[D]
        @inbounds for p_src_offset in 0:(stride_d - 1)
            p_src = p_src_offset + 1
            p_dst = p_src_offset + stride_d + 1
            effective_bc = _get_effective_bc_hetero(bcs[D], p_src, grids[D], method_d)
            src_view = selectdim(partials, 1, p_src)
            dst_view = selectdim(partials, 1, p_dst)
            _differentiate_axis!(dst_view, src_view, grids[D], effective_bc, D, method_d)
        end
    end
    # Non-derivative axis (sizes[D]=1): skip — no entries to compute

    if D < N
        _build_nd_partials_dim_hetero!(partials, grids, methods, bcs, sizes, Val(D + 1), Val(N))
    end
    return partials
end

# ========================================
# Top-Level: Compute Compact Heterogeneous Partials
# ========================================

"""
    _compute_nd_partials_hetero!(partials, grids, data, methods, bcs, sizes)

Compute compact partial derivatives for heterogeneous ND interpolation.

Uses mixed-radix indexing: `sizes[d] = 2` for derivative axes, `1` for others.
Total entries = `prod(sizes)` ≤ 2^N. Non-derivative axes are skipped entirely.
"""
function _compute_nd_partials_hetero!(
        partials::AbstractArray{Tv, NP1},
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        bcs::Tuple{Vararg{AbstractBC, N}},
        sizes::NTuple{N, Int},
    ) where {Tv, Tg <: AbstractFloat, N, NP1}
    @boundscheck begin
        NP1 == N + 1 || throw(DimensionMismatch("partials must have N+1 dimensions"))
        n_partials = prod(sizes)
        size(partials, 1) == n_partials || throw(
            DimensionMismatch(
                "partials first dimension must be prod(sizes)=$(n_partials), got $(size(partials, 1))"
            )
        )
    end

    # Stage 0: Copy f (function values) into partials[1, ...]
    f_partial = selectdim(partials, 1, 1)
    copyto!(f_partial, data)

    # Build up higher-order partials stage by stage
    _build_nd_partials_dim_hetero!(partials, grids, methods, bcs, sizes, Val(1), Val(N))

    return partials
end

"""
    _build_nd_coeffs_hetero(grids, Tv, data, methods) -> HeteroPartials

Allocate and compute compact heterogeneous partial derivatives.
Data is copied directly into `partials[1, ...]` with type promotion to `Tv` —
no intermediate `data_typed` allocation needed.
Memory = `prod(sizes) × grid_size` instead of `2^N × grid_size`.
"""
function _build_nd_coeffs_hetero(
        grids::NTuple{N, AbstractVector{Tg}},
        ::Type{Tv},
        data::AbstractArray{<:Any, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
    ) where {Tg <: AbstractFloat, Tv, N}
    bcs = _extract_bcs(methods)
    sizes = map(_deriv_size, methods)

    # Single allocation: (prod(sizes), n₁, n₂, ..., nₙ)
    n_partials = prod(sizes)
    partials = Array{Tv, N + 1}(undef, n_partials, size(data)...)

    # Promote data to Tv if needed, then copy into partials[1, ...]
    data_tv = eltype(data) === Tv ? data : Tv.(data)
    _compute_nd_partials_hetero!(partials, grids, data_tv, methods, bcs, sizes)

    return HeteroPartials{Tv, N, N + 1}(partials)
end
