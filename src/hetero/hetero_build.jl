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
@inline _deriv_size(::NoInterp) = 1
@inline _deriv_size(::PchipInterp) = 2
@inline _deriv_size(::CardinalInterp) = 2
@inline _deriv_size(::AkimaInterp) = 2

# For @generated: extract sizes from methods tuple TYPE
_is_deriv_method(::Type{<:CubicInterp}) = true
_is_deriv_method(::Type{<:QuadraticInterp}) = true
_is_deriv_method(::Type{PchipInterp}) = true
_is_deriv_method(::Type{AkimaInterp}) = true
_is_deriv_method(::Type{<:CardinalInterp}) = true
_is_deriv_method(::Type{NoInterp}) = false
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
# Only defined for derivative methods (Cubic/Quadratic).
# Linear/Constant have no BC and are never differentiated (sizes[D]=1).

@inline _extract_bc(m::CubicInterp) = m.bc
@inline _extract_bc(m::QuadraticInterp) = m.bc

# For _prepare_periodic_nd: extract BC to detect exclusive periodic axes.
# Non-BC methods return a non-periodic placeholder (never triggers extension).
@inline _bc_for_periodic_check(m::CubicInterp) = m.bc
@inline _bc_for_periodic_check(m::QuadraticInterp) = m.bc
@inline _bc_for_periodic_check(m::PchipInterp) = m.bc
@inline _bc_for_periodic_check(m::CardinalInterp) = m.bc
@inline _bc_for_periodic_check(m::AkimaInterp) = m.bc
@inline _bc_for_periodic_check(::AbstractInterpMethod) = CubicFit()

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
        sizes::NTuple{N, Int},
        ::Val{D},
        ::Val{N},
    ) where {Tv, D, N, NP1}
    if sizes[D] == 2
        # Derivative axis: differentiate using compact stride
        stride_d = D == 1 ? 1 : prod(sizes[1:(D - 1)])
        method_d = methods[D]
        bc_d = _extract_bc(method_d)
        @inbounds for p_src_offset in 0:(stride_d - 1)
            p_src = p_src_offset + 1
            p_dst = p_src_offset + stride_d + 1
            effective_bc = _get_effective_bc_hetero(bc_d, p_src, grids[D], method_d)
            src_view = selectdim(partials, 1, p_src)
            dst_view = selectdim(partials, 1, p_dst)
            _differentiate_axis!(dst_view, src_view, grids[D], effective_bc, D, method_d)
        end
    end
    # Non-derivative axis (sizes[D]=1): skip — no entries to compute

    if D < N
        _build_nd_partials_dim_hetero!(partials, grids, methods, sizes, Val(D + 1), Val(N))
    end
    return partials
end

# ========================================
# Top-Level: Compute Compact Heterogeneous Partials
# ========================================

"""
    _compute_nd_partials_hetero!(partials, grids, data, methods, sizes)

Compute compact partial derivatives for heterogeneous ND interpolation.

Uses mixed-radix indexing: `sizes[d] = 2` for derivative axes, `1` for others.
Total entries = `prod(sizes)` ≤ 2^N. Non-derivative axes are skipped entirely.
BCs are extracted from method types only for derivative axes (Cubic/Quadratic).
"""
function _compute_nd_partials_hetero!(
        partials::AbstractArray{Tv, NP1},
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{<:Any, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        sizes::NTuple{N, Int},
    ) where {Tv, Tg, N, NP1}
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
    _build_nd_partials_dim_hetero!(partials, grids, methods, sizes, Val(1), Val(N))

    return partials
end

"""
    _build_nd_coeffs_hetero(grids, Tv, data, methods) -> _HeteroPartials

Allocate and compute compact heterogeneous partial derivatives.
Data is copied into `partials[1, ...]` via `copyto!` which handles type promotion —
no intermediate allocation needed.
Memory = `prod(sizes) × grid_size` instead of `2^N × grid_size`.
"""
function _build_nd_coeffs_hetero(
        grids::NTuple{N, AbstractVector{Tg}},
        ::Type{Tv},
        data::AbstractArray{<:Any, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
    ) where {Tg, Tv, N}
    sizes = map(_deriv_size, methods)

    # Single allocation: (prod(sizes), n₁, n₂, ..., nₙ)
    # Tz widens Tv with Tg: when grid is Dual, derivatives = data × inv_h → Dual-typed.
    Tz = _output_eltype(Tv, Tg)
    n_partials = prod(sizes)
    partials = Array{Tz, N + 1}(undef, n_partials, size(data)...)

    # copyto! in _compute_nd_partials_hetero! handles data → Tz promotion
    _compute_nd_partials_hetero!(partials, grids, data, methods, sizes)

    return _HeteroPartials{Tz, N, N + 1}(partials)
end
