# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      MULTI-Y CUBIC INTERPOLATION                          ║
# ║         Multiple y-data series sharing the same x-grid                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase 1: Unified-style matrix storage for optimal performance.
# Key optimization: Adaptive layout with lazy transpose for scalar queries.
#
# Include order: ... → cubic_unified_types.jl → multi_cubic_interp.jl
#
# Note: TransposeSnapshot is defined in cubic_unified_types.jl
#

# ========================================
# Type Definition
# ========================================

"""
    CubicMultiInterpolant{T, C, B}

Container for multiple cubic spline interpolants sharing the same x-grid.
Uses unified matrix storage with adaptive layout for optimal performance.

# Type Parameters
- `T`: Float type (Float32 or Float64)
- `C`: Cache type (`CubicSplineCache{T}`)
- `B`: Boundary condition config type (BCPair or PeriodicData)

# Fields
- `cache::C`: Shared CubicSplineCache with LU factorization
- `bc_for_solve::B`: BC configuration for solving systems
- `y::Matrix{T}`: Function values (n_points × n_series) series-contiguous
- `z::Matrix{T}`: Second derivatives (n_points × n_series) series-contiguous
- `_point_snapshot`: Atomic field for lazy point-contiguous layout
- `extrap::ExtrapVal`: Extrapolation mode

# Memory Layout
Primary storage is series-contiguous (n_points × n_series):
- `y[i, k]` = value of series k at grid point i
- `y[:, k]` is contiguous → optimal for vector queries

Lazy transpose (n_series × n_points) for scalar queries:
- `y_point[:, i]` is contiguous → optimal for SIMD scalar queries

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-x)

mitp = cubic_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = mitp(0.5)            # Returns Vector{Float64} of length 3
mitp(output, 0.5)           # In-place

# Vector evaluation
vals = mitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
mitp([out1, out2, out3], xq)    # In-place (zero allocation)
```

# Performance
- Vector queries use series-contiguous layout directly
- Scalar queries trigger lazy transpose on first call
- All series share same cache (O(1) memory overhead)
"""
mutable struct CubicMultiInterpolant{
    T<:AbstractFloat,
    C<:CubicSplineCache{T},
    B
} <: AbstractMultiInterpolant{T}
    const cache::C                    # Shared cache with LU factorization
    const bc_for_solve::B             # BC config for solving
    const y::Matrix{T}                # Series-contiguous y (n_points × n_series)
    const z::Matrix{T}                # Series-contiguous z (n_points × n_series)
    @atomic _point_snapshot::TransposeSnapshot{T}  # Lazy point-contiguous layout
    const extrap::ExtrapVal           # Extrapolation mode

    function CubicMultiInterpolant(
        cache::C,
        bc_for_solve::B,
        y::Matrix{T},
        z::Matrix{T},
        extrap::ExtrapVal
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, B}
        new{T, C, B}(
            cache, bc_for_solve, y, z,
            TransposeSnapshot{T}(),
            extrap
        )
    end
end

# Backward compatibility alias
const MultiCubicInterpolant = CubicMultiInterpolant

# ========================================
# Helper Functions
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(mitp::CubicMultiInterpolant) = mitp.extrap === Val(:wrap)

"""Number of series in the interpolant."""
@inline n_series(mitp::CubicMultiInterpolant) = size(mitp.y, 2)

"""Number of grid points in the interpolant."""
@inline n_points(mitp::CubicMultiInterpolant) = size(mitp.y, 1)

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(mitp::CubicMultiInterpolant{T}) -> (y_point, z_point)

Ensure point-contiguous layout exists. Thread-safe via atomic snapshot.

RCU-style implementation:
- Fast path: atomic acquire read, return if populated
- Slow path: compute transpose, atomic release publish
"""
@inline function _ensure_point_layout!(mitp::CubicMultiInterpolant{T}) where T
    # Fast path: check if already populated
    snap = @atomic :acquire mitp._point_snapshot
    if snap.y_point !== nothing
        return (snap.y_point::Matrix{T}, snap.z_point::Matrix{T})
    end

    # Slow path: build point-contiguous layout
    y_point = permutedims(mitp.y)
    z_point = permutedims(mitp.z)
    new_snap = TransposeSnapshot{T}(y_point, z_point)

    # Atomic publish
    @atomic :release mitp._point_snapshot = new_snap

    return (y_point, z_point)
end

"""
    precompute_transpose!(mitp::CubicMultiInterpolant) -> mitp

Pre-allocate point-contiguous matrices for scalar queries.
Call before hot loops to avoid first-call latency.
"""
function precompute_transpose!(mitp::CubicMultiInterpolant)
    _ensure_point_layout!(mitp)
    return mitp
end

# ========================================
# SIMD Scalar Evaluation Kernels
# ========================================

"""
    _eval_multi_point!(out, y_point, z_point, aq, op)

SIMD-optimized evaluation for point-contiguous layout (n_series × n_points).
Contiguous column access enables vectorization across series dimension.
"""
@inline function _eval_multi_point!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    idx = aq.idx
    idx1 = idx + 1
    wyL, wyR, wzL, wzR = _anchored_weights(aq, op)

    # SIMD loop over series (contiguous column access)
    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]

        # Optimal FMA chain: 3 FMAs + 1 mul
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end

    return out
end

"""
    _eval_multi_point_with_extrap!(out, y_point, z_point, n_pts, aq, extrap, op)

SIMD evaluation with extrapolation handling for multi-series.
"""
@inline function _eval_multi_point_with_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    aq::_CubicAnchoredQuery{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_multi_point!(out, y_point, z_point, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    _eval_multi_point_extrap!(out, y_point, z_point, n_pts, aq, extrap, op, aq.side)
end

# :none - throw DomainError
@inline function _eval_multi_point_extrap!(
    ::AbstractVector{T},
    ::Matrix{T},
    ::Matrix{T},
    ::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:none},
    ::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    throw(DomainError(aq.xq, "Query point outside domain"))
end

# :constant - clamp to boundary
@inline function _eval_multi_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    ::Matrix{T},
    n_pts::Int,
    ::_CubicAnchoredQuery{T},
    ::Val{:constant},
    ::AbstractEvalOp,
    side::UInt8
) where {T<:AbstractFloat}
    idx = side == 0x01 ? 1 : n_pts
    @inbounds @simd for k in axes(out, 1)
        out[k] = y_point[k, idx]
    end
    return out
end

# :extension - extend polynomial
@inline function _eval_multi_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:extension},
    op::AbstractEvalOp,
    side::UInt8
) where {T<:AbstractFloat}
    # Use boundary interval for extension
    idx = side == 0x01 ? 1 : (n_pts - 1)
    idx1 = idx + 1
    wyL, wyR, wzL, wzR = _anchored_weights(aq, op)

    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end
    return out
end

# :wrap - periodic (anchor already adjusted)
@inline function _eval_multi_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    ::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:wrap},
    op::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    # Anchor was already wrapped, use normal evaluation
    return _eval_multi_point!(out, y_point, z_point, aq, op)
end

# ========================================
# Internal: Coefficient Solver
# ========================================

"""
    _solve_multi_coefficients!(z_mat, y_mat, cache, bc_for_solve)

Solve cubic spline systems for all series using shared LU factorization.
"""
@with_pool pool function _solve_multi_coefficients!(
    z_mat::Matrix{T},
    y_mat::Matrix{T},
    cache::CubicSplineCache{T},
    bc_for_solve
) where {T<:AbstractFloat}
    n_series_count = size(y_mat, 2)

    # Solve each series column
    @inbounds for k in 1:n_series_count
        _solve_system!(@view(z_mat[:, k]), cache, @view(y_mat[:, k]), bc_for_solve)
    end

    return z_mat
end

# ========================================
# Constructors
# ========================================

"""
    cubic_interp(x, ys::AbstractVector{<:AbstractVector}; bc=NaturalBC(), extrap=:none, autocache=true, precompute_transpose=false)

Create a multi-Y cubic spline interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `bc`: Boundary condition (NaturalBC, ClampedBC, PeriodicBC, etc.)
- `extrap`: Extrapolation mode (:none, :constant, :extension, :wrap)
- `autocache`: If true, reuse cached LU factorization (default: true)
- `precompute_transpose`: If true, build point-contiguous layout immediately

# Returns
`CubicMultiInterpolant` object with matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

mitp = cubic_interp(x, [y1, y2, y3])
vals = mitp(0.5)  # [sin(π), cos(π), exp(-0.5)]
```
"""
function cubic_interp(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    precompute_transpose::Bool=false
) where {T<:AbstractFloat}
    # Validate input
    @assert !isempty(ys) "ys must not be empty"

    n_pts = length(x)
    n_series_count = length(ys)

    # Validate all y-series have same length as x
    for (k, y) in enumerate(ys)
        if length(y) != n_pts
            throw(DimensionMismatch(
                "y-series $k has length $(length(y)), expected $n_pts (length of x)"
            ))
        end
    end

    # Build y matrix (n_points × n_series) series-contiguous
    y_mat = Matrix{T}(undef, n_pts, n_series_count)
    @inbounds for k in 1:n_series_count
        y_mat[:, k] .= ys[k]
    end

    # Handle periodic BC separately
    if _is_periodic_bc(bc)
        return _build_multi_periodic(x, y_mat, n_pts, n_series_count, autocache, precompute_transpose)
    end

    # Get cache for derivative BC
    bc_pair = _normalize_bc(bc, T)
    cache = _get_cubic_cache(x, bc_pair, autocache)

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_pts, n_series_count)
    _solve_multi_coefficients!(z_mat, y_mat, cache, bc_pair)

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    mitp = CubicMultiInterpolant(cache, bc_pair, y_mat, z_mat, extrap_val)

    if precompute_transpose
        _ensure_point_layout!(mitp)
    end

    return mitp
end

"""
Internal helper for periodic BC multi-interpolant construction.
"""
function _build_multi_periodic(
    x::AbstractVector{T},
    y_mat::Matrix{T},
    n_pts::Int,
    n_series_count::Int,
    autocache::Bool,
    precompute_transpose::Bool
) where {T<:AbstractFloat}
    # Validate periodic endpoints for all series
    atol = T === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64
    @inbounds for k in 1:n_series_count
        y_first = y_mat[1, k]
        y_last = y_mat[n_pts, k]
        if !isapprox(y_first, y_last; atol=atol)
            throw(ArgumentError(
                "Periodic BC requires y[1] ≈ y[end] for series $k, " *
                "got y[1]=$y_first, y[end]=$y_last (diff=$(abs(y_last-y_first)))"
            ))
        end
    end

    # Get periodic cache
    cache = _get_cubic_cache(x, PeriodicBC(), autocache)

    # Build z matrix
    z_mat = Matrix{T}(undef, n_pts, n_series_count)
    _solve_multi_coefficients!(z_mat, y_mat, cache, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    mitp = CubicMultiInterpolant(cache, cache.bc_config, y_mat, z_mat, Val(:wrap))

    if precompute_transpose
        _ensure_point_layout!(mitp)
    end

    return mitp
end

# Matrix input: columns as y-series
"""
    cubic_interp(x, Y::AbstractMatrix; bc=NaturalBC(), extrap=:none, autocache=true, precompute_transpose=false)

Create a multi-Y cubic spline interpolant from a matrix where each column is a y-series.

# Arguments
- `x::AbstractVector`: x-coordinates (length n)
- `Y::AbstractMatrix`: n×m matrix, each column is a y-series
- `bc`, `extrap`, `autocache`, `precompute_transpose`: Same as vector form

# Example
```julia
x = collect(range(0.0, 1.0, 101))
Y = hcat(sin.(2π .* x), cos.(2π .* x))  # 101×2 matrix

mitp = cubic_interp(x, Y)
```
"""
function cubic_interp(
    x::AbstractVector{T},
    Y::AbstractMatrix{T};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    precompute_transpose::Bool=false
) where {T<:AbstractFloat}
    n_pts = length(x)

    # Validate dimensions
    if size(Y, 1) != n_pts
        throw(DimensionMismatch(
            "Y has $(size(Y, 1)) rows but x has $n_pts points (expected n_points × n_series matrix)"
        ))
    end

    n_series_count = size(Y, 2)

    # Copy to ensure ownership
    y_mat = copy(Y)

    # Handle periodic BC separately
    if _is_periodic_bc(bc)
        return _build_multi_periodic(x, y_mat, n_pts, n_series_count, autocache, precompute_transpose)
    end

    # Get cache for derivative BC
    bc_pair = _normalize_bc(bc, T)
    cache = _get_cubic_cache(x, bc_pair, autocache)

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_pts, n_series_count)
    _solve_multi_coefficients!(z_mat, y_mat, cache, bc_pair)

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    mitp = CubicMultiInterpolant(cache, bc_pair, y_mat, z_mat, extrap_val)

    if precompute_transpose
        _ensure_point_layout!(mitp)
    end

    return mitp
end

# Real type wrappers (auto-promote to Float)
function cubic_interp(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    precompute_transpose::Bool=false
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    ys_float = [_to_float(y, T) for y in ys]
    return cubic_interp(x_float, ys_float; bc=bc, extrap=extrap, autocache=autocache, precompute_transpose=precompute_transpose)
end

function cubic_interp(
    x::AbstractVector{Tx},
    Y::AbstractMatrix{Ty};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true,
    precompute_transpose::Bool=false
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    Y_float = T.(Y)
    return cubic_interp(x_float, Y_float; bc=bc, extrap=extrap, autocache=autocache, precompute_transpose=precompute_transpose)
end

# ========================================
# Scalar Evaluation
# ========================================

"""
    (mitp::MultiCubicInterpolant)(xq::Real; deriv=0)

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.
"""
function (mitp::CubicMultiInterpolant{T})(xq::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    xq_typed = T(xq)

    # Build anchor once
    aq = _anchor_query(mitp.cache.x, xq_typed; wrap=_should_wrap(mitp))

    # Allocate output
    n = n_series(mitp)
    output = Vector{T}(undef, n)

    # Evaluate using scalar kernel
    _eval_multi_scalar!(output, mitp, aq, deriv)
    return output
end

"""
    (mitp::MultiCubicInterpolant)(output::AbstractVector, xq::Real; deriv=0)

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (mitp::CubicMultiInterpolant{T})(
    output::AbstractVector{T},
    xq::S;
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == n_series(mitp) "output length must match number of series"

    xq_typed = T(xq)

    # Build anchor once
    aq = _anchor_query(mitp.cache.x, xq_typed; wrap=_should_wrap(mitp))

    # Evaluate using scalar kernel
    _eval_multi_scalar!(output, mitp, aq, deriv)
    return output
end

"""
Internal scalar evaluation kernel using per-series cubic evaluation.
Uses the series-contiguous layout (y[:, k], z[:, k]) for each series.

Note: Triggers lazy point-layout creation for future SIMD scalar queries.
"""
@inline function _eval_multi_scalar!(
    output::AbstractVector{T},
    mitp::CubicMultiInterpolant{T},
    aq::_CubicAnchoredQuery{T},
    deriv::Int
) where {T<:AbstractFloat}
    # Use point-contiguous layout for SIMD evaluation
    y_point, z_point = _ensure_point_layout!(mitp)

    # SIMD dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        _eval_multi_point_with_extrap!(output, y_point, z_point, n_points(mitp), aq, mitp.extrap, op)
    end

    return output
end

"""
Evaluate a single series using anchored query.
Adapts the CubicInterpolant anchored kernel pattern for matrix column access.
"""
@inline function _eval_multi_anchored_single(
    mitp::CubicMultiInterpolant{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Fast path: inside domain
    if aq.side == 0x00
        return _eval_multi_anchored_kernel(mitp, k, aq, op)
    end

    # Outside domain: dispatch on extrapolation mode
    return _eval_multi_anchored_extrap(mitp, k, aq, mitp.extrap, op)
end

"""
Core anchored kernel for matrix column access.
Computes the 4-term dot product: S(xq) = wyL*yL + wyR*yR + wzL*zL + wzR*zR
"""
@inline function _eval_multi_anchored_kernel(
    mitp::CubicMultiInterpolant{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    @inbounds begin
        yL = mitp.y[aq.idx, k]
        yR = mitp.y[aq.idx + 1, k]
        zL = mitp.z[aq.idx, k]
        zR = mitp.z[aq.idx + 1, k]
    end
    wyL, wyR, wzL, wzR = _anchored_weights(aq, op)
    return muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
end

# Extrapolation handlers for multi-interpolant
@inline function _eval_multi_anchored_extrap(
    mitp::CubicMultiInterpolant{T}, k::Int, aq::_CubicAnchoredQuery{T}, ::Val{:none}, ::AbstractEvalOp
) where {T<:AbstractFloat}
    x_min, x_max = first(mitp.cache.x), last(mitp.cache.x)
    throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
end

@inline function _eval_multi_anchored_extrap(
    mitp::CubicMultiInterpolant{T}, k::Int, aq::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalValue
) where {T<:AbstractFloat}
    return aq.side == 0x01 ? @inbounds(mitp.y[1, k]) : @inbounds(mitp.y[end, k])
end

@inline function _eval_multi_anchored_extrap(
    ::CubicMultiInterpolant{T}, ::Int, ::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalDeriv1
) where {T<:AbstractFloat}
    return zero(T)
end

@inline function _eval_multi_anchored_extrap(
    ::CubicMultiInterpolant{T}, ::Int, ::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalDeriv2
) where {T<:AbstractFloat}
    return zero(T)
end

@inline function _eval_multi_anchored_extrap(
    mitp::CubicMultiInterpolant{T}, k::Int, aq::_CubicAnchoredQuery{T}, ::Val{:extension}, op::AbstractEvalOp
) where {T<:AbstractFloat}
    return _eval_multi_anchored_kernel(mitp, k, aq, op)
end

@inline function _eval_multi_anchored_extrap(
    mitp::CubicMultiInterpolant{T}, k::Int, aq::_CubicAnchoredQuery{T}, ::Val{:wrap}, op::AbstractEvalOp
) where {T<:AbstractFloat}
    return _eval_multi_anchored_kernel(mitp, k, aq, op)
end

# ========================================
# Vector Evaluation
# ========================================

"""
    (mitp::MultiCubicInterpolant)(xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
"""
function (mitp::CubicMultiInterpolant{T})(
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)

    # Build anchors once
    aq_vec = _anchor_query(mitp.cache.x, xq_typed; wrap=_should_wrap(mitp))

    # Allocate outputs
    n = n_series(mitp)
    outputs = [Vector{T}(undef, length(xq_typed)) for _ in 1:n]

    # Extract matrices for argument-passing pattern
    y, z = mitp.y, mitp.z
    n_pts = n_points(mitp)
    extrap = mitp.extrap

    # Evaluate all series
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_multi_vector_series!(outputs[k], y, z, n_pts, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

"""
    (mitp::MultiCubicInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points
- `deriv`: Derivative order (0, 1, or 2)

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
Uses task-local pool for anchor vector to achieve zero allocation after warmup.
"""
@with_pool pool function (mitp::CubicMultiInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0
) where {T<:AbstractFloat}
    @assert length(outputs) == n_series(mitp) "outputs length must match number of series"
    @assert all(out -> length(out) == length(xq), outputs) "all output buffers must match xq length"

    # Build anchors from pool (zero allocation after warmup)
    aq_vec = acquire!(pool, _CubicAnchoredQuery{T}, length(xq))
    _fill_anchors!(aq_vec, mitp.cache.x, xq; wrap=_should_wrap(mitp))

    # Extract matrices for argument-passing pattern
    y, z = mitp.y, mitp.z
    n_pts = n_points(mitp)
    n = n_series(mitp)
    extrap = mitp.extrap

    # Evaluate all series
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_multi_vector_series!(outputs[k], y, z, n_pts, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

# Real type wrapper for in-place vector
function (mitp::CubicMultiInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)
    return mitp(outputs, xq_typed; deriv=deriv)
end

"""
    (mitp::MultiCubicInterpolant)(outputs, aq_vec::AbstractVector{<:_CubicAnchoredQuery}; deriv=0)

Evaluate multi-Y interpolant with pre-built anchors (TRUE zero-allocation).

For maximum performance in hot loops, pre-build anchors once and reuse:
```julia
x = ...
mitp = cubic_interp(x, [y1, y2, y3])
xq = [0.1, 0.2, 0.3, ...]

# Pre-build anchors (allocates once)
aq_vec = FastInterpolations._anchor_query(x, xq)

# Zero-allocation loop
outputs = [similar(xq) for _ in 1:3]
for _ in 1:1000
    mitp(outputs, aq_vec)  # Zero allocation!
end
```
"""
function (mitp::CubicMultiInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    @assert length(outputs) == n_series(mitp) "outputs length must match number of series"
    @assert all(out -> length(out) == length(aq_vec), outputs) "all output buffers must match aq_vec length"

    # Extract matrices for argument-passing pattern
    y, z = mitp.y, mitp.z
    n_pts = n_points(mitp)
    n = n_series(mitp)
    extrap = mitp.extrap

    # Evaluate all series
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_multi_vector_series!(outputs[k], y, z, n_pts, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance (avoids struct field access in loop).
"""
@inline function _eval_multi_vector_series!(
    out::AbstractVector{T},
    y::Matrix{T},
    z::Matrix{T},
    n_pts::Int,
    k::Int,
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    @inbounds for j in eachindex(out, aq_vec)
        out[j] = _eval_multi_series_with_extrap(y, z, n_pts, k, aq_vec[j], extrap, op)
    end
    return out
end

"""
Internal: Evaluate single series at single query point with extrapolation handling.
Takes matrices as arguments for optimal performance.
"""
@inline function _eval_multi_series_with_extrap(
    y::Matrix{T},
    z::Matrix{T},
    n_pts::Int,
    k::Int,
    aq::_CubicAnchoredQuery{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_multi_series_anchored(y, z, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap === Val(:extension) || extrap === Val(:wrap)
        return _eval_multi_series_anchored(y, z, k, aq, op)
    elseif extrap === Val(:constant)
        if op isa EvalValue
            pt_idx = aq.side == 0x01 ? 1 : n_pts
            @inbounds return y[pt_idx, k]
        else
            return zero(T)
        end
    else
        throw(DomainError(aq.xq, "Query point outside domain"))
    end
end

"""
Internal: Core cubic evaluation for series k at anchored query point.
Direct matrix access for optimal performance.
"""
@inline function _eval_multi_series_anchored(
    y::Matrix{T},
    z::Matrix{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    idx = aq.idx
    wyL, wyR, wzL, wzR = _anchored_weights(aq, op)
    @inbounds begin
        yL = y[idx, k]
        yR = y[idx + 1, k]
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
end
