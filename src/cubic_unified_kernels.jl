# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC UNIFIED INTERPOLANT KERNELS                      ║
# ║         Evaluation kernels for CubicMultiInterpolantUnified               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Provides callable methods for CubicMultiInterpolantUnified:
# - Scalar evaluation: mitp(xq), mitp(out, xq)
#   Uses point-contiguous layout (n_series × n_points) for SIMD
# - Vector evaluation: mitp(xq_vec), mitp(outputs, xq_vec)
#   Uses series-contiguous layout (n_points × n_series) for cache locality
#
# Include order: ... → cubic_unified_interp.jl → cubic_unified_kernels.jl
#

# ========================================
# Point-Contiguous Kernels (Scalar Queries)
# ========================================

"""
    _eval_unified_point!(out, y_point, z_point, aq, op)

Core SIMD-friendly kernel for point-contiguous evaluation.

Evaluates all series at a single anchored query point using the 4-term dot product:
    S_k(xq) = wyL*y[k,idx] + wyR*y[k,idx+1] + wzL*z[k,idx] + wzR*z[k,idx+1]

# Arguments
- `out::AbstractVector{T}`: Output vector of length n_series
- `y_point::Matrix{T}`: Point-contiguous y values (n_series × n_points)
- `z_point::Matrix{T}`: Point-contiguous z values (n_series × n_points)
- `aq::_CubicAnchoredQuery{T}`: Anchored query with precomputed weights
- `op::AbstractEvalOp`: Evaluation operation (EvalValue, EvalDeriv1, EvalDeriv2)

# Performance
Uses `@inbounds @simd` for vectorized evaluation across series.
The point-contiguous layout ensures `y_point[:, idx]` is contiguous in memory.
"""
@inline function _eval_unified_point!(
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
    _eval_unified_point_with_extrap!(out, y_point, z_point, n_points, aq, extrap, op)

Evaluate point-contiguous layout with extrapolation handling.

Called when aq.side may be non-zero (boundary or outside domain).
"""
@inline function _eval_unified_point_with_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_points::Int,
    aq::_CubicAnchoredQuery{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_unified_point!(out, y_point, z_point, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    _eval_unified_point_extrap!(out, y_point, z_point, n_points, aq, extrap, op, aq.side)
end

# :none - throw DomainError
@inline function _eval_unified_point_extrap!(
    ::AbstractVector{T},
    y_point::Matrix{T},
    ::Matrix{T},
    ::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:none},
    ::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    x_min = y_point[1, 1]  # Not used - just need to throw
    throw(DomainError(aq.xq, "Query point outside domain"))
end

# :constant for value - return boundary values
@inline function _eval_unified_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    ::Matrix{T},
    n_points::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:constant},
    ::EvalValue,
    side::UInt8
) where {T<:AbstractFloat}
    col = side == 0x01 ? 1 : n_points
    @inbounds @simd for k in axes(out, 1)
        out[k] = y_point[k, col]
    end
    return out
end

# :constant for derivatives - return zeros
@inline function _eval_unified_point_extrap!(
    out::AbstractVector{T},
    ::Matrix{T},
    ::Matrix{T},
    ::Int,
    ::_CubicAnchoredQuery{T},
    ::Val{:constant},
    ::EvalDeriv1,
    ::UInt8
) where {T<:AbstractFloat}
    fill!(out, zero(T))
    return out
end

@inline function _eval_unified_point_extrap!(
    out::AbstractVector{T},
    ::Matrix{T},
    ::Matrix{T},
    ::Int,
    ::_CubicAnchoredQuery{T},
    ::Val{:constant},
    ::EvalDeriv2,
    ::UInt8
) where {T<:AbstractFloat}
    fill!(out, zero(T))
    return out
end

# :extension - use precomputed weights (polynomial extrapolation from boundary)
@inline function _eval_unified_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    ::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:extension},
    op::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    # Anchor already has clamped idx and computed weights
    return _eval_unified_point!(out, y_point, z_point, aq, op)
end

# :wrap - use precomputed weights (already wrapped at anchor construction)
@inline function _eval_unified_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    ::Int,
    aq::_CubicAnchoredQuery{T},
    ::Val{:wrap},
    op::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    # Anchor already wrapped coordinates and has correct weights
    return _eval_unified_point!(out, y_point, z_point, aq, op)
end

# ========================================
# Series-Contiguous Kernels (Vector Queries)
# ========================================

"""
    _eval_unified_series_anchored(y, z, k, aq, op)

Evaluate single series k at anchored query point.

Uses series-contiguous layout where y[:, k] and z[:, k] are contiguous.
"""
@inline function _eval_unified_series_anchored(
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

"""
    _eval_unified_series_with_extrap(y, z, n_points, k, aq, extrap, op)

Evaluate single series with extrapolation handling.
"""
@inline function _eval_unified_series_with_extrap(
    y::Matrix{T},
    z::Matrix{T},
    n_points::Int,
    k::Int,
    aq::_CubicAnchoredQuery{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_unified_series_anchored(y, z, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap === Val(:extension) || extrap === Val(:wrap)
        return _eval_unified_series_anchored(y, z, k, aq, op)
    elseif extrap === Val(:constant)
        if op isa EvalValue
            pt_idx = aq.side == 0x01 ? 1 : n_points
            @inbounds return y[pt_idx, k]
        else
            return zero(T)
        end
    else
        throw(DomainError(aq.xq, "Query point outside domain"))
    end
end

"""
    _eval_unified_series_vector!(outputs, y, z, x_grid, xq_vec, extrap, op)

Evaluate using series-contiguous layout for vector queries.

For each series k, all grid points y[:, k] are contiguous → cache-friendly.
Series-major loop ensures each series' data stays in L1 cache while processing
all query points.

# Arguments
- `outputs::AbstractVector{<:AbstractVector{T}}`: Output vectors (one per series)
- `y::Matrix{T}`: Series-contiguous y values (n_points × n_series)
- `z::Matrix{T}`: Series-contiguous z values (n_points × n_series)
- `x_grid`: Grid points for anchor construction
- `xq_vec::AbstractVector{T}`: Query points
- `extrap::ExtrapVal`: Extrapolation mode
- `op::AbstractEvalOp`: Evaluation operation
"""
@with_pool pool function _eval_unified_series_vector!(
    outputs::AbstractVector{<:AbstractVector{T}},
    y::Matrix{T},
    z::Matrix{T},
    x_grid,
    xq_vec::AbstractVector{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    n_query = length(xq_vec)
    n_series_count = size(y, 2)
    n_points = size(y, 1)

    # Pre-build anchors for all query points (from pool)
    wrap = extrap === Val(:wrap)
    anchors = acquire!(pool, _CubicAnchoredQuery{T}, n_query)
    _fill_anchors!(anchors, x_grid, xq_vec; wrap=wrap)

    # Series-major loop: k-th series's all queries processed together
    # y[:, k] and z[:, k] are contiguous → L1 cache-friendly
    @inbounds for k in 1:n_series_count
        out_k = outputs[k]

        for j in 1:n_query
            aq = anchors[j]
            out_k[j] = _eval_unified_series_with_extrap(y, z, n_points, k, aq, extrap, op)
        end
    end

    return outputs
end

"""
    _eval_unified_series_vector_anchored!(outputs, y, z, n_points, aq_vec, extrap, op)

Evaluate using pre-built anchors (zero-allocation hot-loop API).
"""
function _eval_unified_series_vector_anchored!(
    outputs::AbstractVector{<:AbstractVector{T}},
    y::Matrix{T},
    z::Matrix{T},
    n_points::Int,
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    n_query = length(aq_vec)
    n_series_count = size(y, 2)

    # Series-major loop
    @inbounds for k in 1:n_series_count
        out_k = outputs[k]

        for j in 1:n_query
            aq = aq_vec[j]
            out_k[j] = _eval_unified_series_with_extrap(y, z, n_points, k, aq, extrap, op)
        end
    end

    return outputs
end

# ========================================
# Scalar API - In-Place
# ========================================

"""
    (mitp::CubicMultiInterpolantUnified)(out, xq; deriv=0)

Evaluate unified multi-series interpolant at scalar query point (in-place).

Uses point-contiguous layout (lazily created on first call) for SIMD vectorization.

# Arguments
- `out::AbstractVector{T}`: Pre-allocated output vector of length n_series
- `xq::Real`: Query point
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Returns
The same `out` vector, filled with interpolated values.

# Example
```julia
mitp = cubic_interp_unified(x, [y1, y2, y3])
out = zeros(3)
mitp(out, 0.5)  # Fills out with [S1(0.5), S2(0.5), S3(0.5)]
```
"""
function (mitp::CubicMultiInterpolantUnified{T})(
    out::AbstractVector{T},
    xq::S;
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    n_ser = n_series(mitp)

    # Validate output length
    if length(out) != n_ser
        throw(DimensionMismatch(
            "output length $(length(out)) must match n_series $n_ser"
        ))
    end

    xq_typed = T(xq)

    # Use point-contiguous layout for scalar queries
    y_point, z_point = _ensure_point_layout!(mitp)

    # Build anchor
    aq = _anchor_query(mitp.cache.x, xq_typed; wrap=_should_wrap(mitp))

    # Dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        _eval_unified_point_with_extrap!(out, y_point, z_point, n_points(mitp), aq, mitp.extrap, op)
    end

    return out
end

# ========================================
# Scalar API - Out-of-Place
# ========================================

"""
    (mitp::CubicMultiInterpolantUnified)(xq; deriv=0)

Evaluate unified multi-series interpolant at scalar query point (allocating).

# Arguments
- `xq::Real`: Query point
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Returns
Vector{T} of length n_series with interpolated values.

# Example
```julia
mitp = cubic_interp_unified(x, [y1, y2, y3])
vals = mitp(0.5)  # Returns [S1(0.5), S2(0.5), S3(0.5)]
```
"""
function (mitp::CubicMultiInterpolantUnified{T})(xq::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    out = Vector{T}(undef, n_series(mitp))
    return mitp(out, xq; deriv=deriv)
end

# ========================================
# Vector API - In-Place
# ========================================

"""
    (mitp::CubicMultiInterpolantUnified)(outputs, xq_vec; deriv=0)

Evaluate unified multi-series interpolant at multiple query points (in-place).

Uses series-contiguous layout for cache-friendly access pattern.

# Arguments
- `outputs::AbstractVector{<:AbstractVector{T}}`: Pre-allocated output vectors
  - Length must equal n_series
  - Each vector must have length equal to length(xq_vec)
- `xq_vec::AbstractVector`: Query points
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Output Layout
`outputs[k][j]` = value of series k at query point xq_vec[j] (series-first layout)

# Returns
The same `outputs` vector of vectors.

# Example
```julia
mitp = cubic_interp_unified(x, [y1, y2, y3])
xq = [0.1, 0.3, 0.5, 0.7, 0.9]
outputs = [zeros(5) for _ in 1:3]
mitp(outputs, xq)  # Fills outputs with interpolated values
```
"""
function (mitp::CubicMultiInterpolantUnified{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(xq)
    n_ser = n_series(mitp)

    # Validate dimensions
    if length(outputs) != n_ser
        throw(DimensionMismatch(
            "outputs length $(length(outputs)) must match n_series $n_ser"
        ))
    end
    for (k, out_k) in enumerate(outputs)
        if length(out_k) != n_query
            throw(DimensionMismatch(
                "outputs[$k] length $(length(out_k)) must match n_query $n_query"
            ))
        end
    end

    # Dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        _eval_unified_series_vector!(outputs, mitp.y, mitp.z, mitp.cache.x, xq, mitp.extrap, op)
    end

    return outputs
end

# Real type wrapper for vector in-place
function (mitp::CubicMultiInterpolantUnified{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = T.(xq)
    return mitp(outputs, xq_typed; deriv=deriv)
end

# ========================================
# Vector API - Out-of-Place
# ========================================

"""
    (mitp::CubicMultiInterpolantUnified)(xq_vec; deriv=0)

Evaluate unified multi-series interpolant at multiple query points (allocating).

# Arguments
- `xq_vec::AbstractVector`: Query points
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Returns
`Vector{Vector{T}}` with layout `outputs[k][j]` = series k at xq_vec[j]

# Example
```julia
mitp = cubic_interp_unified(x, [y1, y2, y3])
outputs = mitp([0.1, 0.3, 0.5])  # Returns 3 vectors, each of length 3
```
"""
function (mitp::CubicMultiInterpolantUnified{T})(
    xq::AbstractVector{T};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(xq)
    outputs = [Vector{T}(undef, n_query) for _ in 1:n_series(mitp)]
    return mitp(outputs, xq; deriv=deriv)
end

# Real type wrapper for vector out-of-place
function (mitp::CubicMultiInterpolantUnified{T})(
    xq::AbstractVector{S};
    deriv::Int=0
) where {T<:AbstractFloat, S<:Real}
    xq_typed = T.(xq)
    return mitp(xq_typed; deriv=deriv)
end

# ========================================
# Pre-built Anchor API (Vector)
# ========================================

"""
    (mitp::CubicMultiInterpolantUnified)(outputs, aq_vec; deriv=0)

Evaluate unified multi-series interpolant using pre-built anchored queries (in-place).

This is the zero-allocation hot-loop API. Build anchors once with `_fill_anchors!`,
then reuse across multiple evaluation calls.

# Arguments
- `outputs::AbstractVector{<:AbstractVector{T}}`: Pre-allocated output vectors
  - Length must equal n_series
  - Each vector must have length equal to length(aq_vec)
- `aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}}`: Pre-built anchored queries
- `deriv::Int`: Derivative order (0=value, 1=first, 2=second)

# Output Layout
`outputs[k][j]` = value of series k at anchored query aq_vec[j]

# Returns
The same `outputs` vector of vectors.

# Zero-Allocation Pattern
```julia
mitp = cubic_interp_unified(x, [y1, y2, y3])
xq_vec = [0.1, 0.3, 0.5, 0.7, 0.9]

# Pre-allocate anchors (once)
aq_vec = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, length(xq_vec))
FastInterpolations._fill_anchors!(aq_vec, mitp.cache.x, xq_vec; wrap=false)

# Pre-allocate outputs (once)
outputs = [zeros(5) for _ in 1:3]

# Hot loop: zero allocations after warmup
for iteration in 1:1000
    mitp(outputs, aq_vec)  # Reuses pre-built anchors
end
```
"""
function (mitp::CubicMultiInterpolantUnified{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(aq_vec)
    n_ser = n_series(mitp)

    # Validate dimensions
    if length(outputs) != n_ser
        throw(DimensionMismatch(
            "outputs length $(length(outputs)) must match n_series $n_ser"
        ))
    end
    for (k, out_k) in enumerate(outputs)
        if length(out_k) != n_query
            throw(DimensionMismatch(
                "outputs[$k] length $(length(out_k)) must match n_query $n_query"
            ))
        end
    end

    # Dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        _eval_unified_series_vector_anchored!(outputs, mitp.y, mitp.z, n_points(mitp), aq_vec, mitp.extrap, op)
    end

    return outputs
end
