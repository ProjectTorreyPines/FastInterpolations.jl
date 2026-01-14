# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    SERIES INTERPOLANT UTILITIES                           ║
# ║              Validation helpers for AbstractSeriesInterpolant             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: abstract_types.jl → ... → series_utils.jl
#

# ========================================
# Input Validation
# ========================================

"""
    _validate_series_inputs(x, ys)

Validate input y-series dimensions match x-grid.

# Throws
- `ArgumentError` if `ys` is empty
- `DimensionMismatch` if any y-series has different length than `x`
"""
function _validate_series_inputs(
    x::AbstractVector,
    ys::AbstractVector{<:AbstractVector}
)
    isempty(ys) && throw(ArgumentError("ys must not be empty"))
    n = length(x)
    for (k, y) in enumerate(ys)
        length(y) != n && throw(DimensionMismatch(
            "y-series $k has length $(length(y)), expected $n"
        ))
    end
    return nothing
end

# ========================================
# Output Validation
# ========================================

"""
    _validate_series_outputs(outputs, n_series, n_query)

Validate output buffer dimensions for vector evaluation.

# Throws
- `DimensionMismatch` if `outputs` length != `n_series`
- `DimensionMismatch` if any output buffer length != `n_query`
"""
function _validate_series_outputs(
    outputs::AbstractVector{<:AbstractVector},
    n_series::Int,
    n_query::Int
)
    length(outputs) != n_series && throw(DimensionMismatch(
        "outputs length $(length(outputs)) must match n_series $n_series"
    ))
    for (k, out) in enumerate(outputs)
        length(out) != n_query && throw(DimensionMismatch(
            "outputs[$k] length $(length(out)) must match n_query $n_query"
        ))
    end
    return nothing
end

"""
    _validate_scalar_output(output, n_series)

Validate scalar output buffer dimension.

# Throws
- `DimensionMismatch` if `output` length != `n_series`
"""
@inline function _validate_scalar_output(
    output::AbstractVector,
    n_series::Int
)
    length(output) == n_series || throw(DimensionMismatch(
        "output length $(length(output)) must match n_series $n_series"
    ))
    return nothing
end

# ========================================
# Extrapolation Helpers
# ========================================
# Pure data-only helpers for constant extrapolation.
# Used by Linear, Cubic, Constant, and Quadratic series interpolants.

"""
    _boundary_point_index(side::UInt8, n_pts::Int) -> Int

Return the boundary point index for constant extrapolation.

# Arguments
- `side::UInt8`: `0x01` for left boundary, `0x02` for right boundary
- `n_pts::Int`: Total number of grid points

# Returns
- `1` if `side == 0x01` (left boundary)
- `n_pts` if `side == 0x02` (right boundary)
"""
@inline _boundary_point_index(side::UInt8, n_pts::Int) = side == 0x01 ? 1 : n_pts

"""
    _throw_extrap_domain_error(xq::T, x_min::T, x_max::T)

Throw a DomainError for query points outside the interpolation domain.

This function is marked `@noinline` to keep the error path cold and reduce
code size in the hot interpolation loops.

# Throws
- `DomainError` with message indicating the query point and valid domain
"""
@noinline function _throw_extrap_domain_error(xq::T, x_min::T, x_max::T) where {T<:AbstractFloat}
    throw(DomainError(xq, "query point outside domain [$x_min, $x_max]"))
end

"""
    _constant_extrap_boundary_value(y, side, n_pts, k, op) -> T

Get the boundary value for constant extrapolation in vector evaluation path.

For `EvalValue`, returns the boundary y-value. For derivatives (`EvalDeriv1`,
`EvalDeriv2`), returns `zero(T)` since constant extrapolation has zero slope.

# Arguments
- `y::Matrix{T}`: y-values matrix (n_points × n_series)
- `side::UInt8`: `0x01` for left, `0x02` for right boundary
- `n_pts::Int`: Number of grid points
- `k::Int`: Series index
- `op::AbstractEvalOp`: Evaluation operation type

# Returns
- Boundary value for `EvalValue`
- `zero(T)` for `EvalDeriv1` or `EvalDeriv2`
"""
@inline function _constant_extrap_boundary_value(
    y::Matrix{T}, side::UInt8, n_pts::Int, k::Int, ::EvalValue
) where {T<:AbstractFloat}
    @inbounds return y[_boundary_point_index(side, n_pts), k]
end

@inline function _constant_extrap_boundary_value(
    ::Matrix{T}, ::UInt8, ::Int, ::Int, ::Union{EvalDeriv1, EvalDeriv2}
) where {T<:AbstractFloat}
    return zero(T)
end

"""
    _fill_constant_extrap_simd!(out, y_point, side, n_pts, op) -> out

Fill output vector with boundary values for constant extrapolation (SIMD path).

For `EvalValue`, fills with boundary y-values. For derivatives, fills with zeros.
Uses `@simd` for vectorization of the fill loop.

# Arguments
- `out::AbstractVector{T}`: Output vector to fill
- `y_point::Matrix{T}`: y-values in SIMD layout (n_series × n_points), accessed as `y_point[series, point]`
- `side::UInt8`: `0x01` for left, `0x02` for right boundary
- `n_pts::Int`: Number of grid points
- `op::AbstractEvalOp`: Evaluation operation type

# Returns
- `out` (modified in-place)

# Note
Matrix layout uses `y_point[k, idx]` convention (series first, point second)
to match the transposed SIMD layout used across all series interpolants.
"""
@inline function _fill_constant_extrap_simd!(
    out::AbstractVector{T}, y_point::Matrix{T}, side::UInt8, n_pts::Int, ::EvalValue
) where {T<:AbstractFloat}
    idx = _boundary_point_index(side, n_pts)
    @inbounds @simd for k in axes(out, 1)
        out[k] = y_point[k, idx]
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
    out::AbstractVector{T}, ::Matrix{T}, ::UInt8, ::Int, ::Union{EvalDeriv1, EvalDeriv2}
) where {T<:AbstractFloat}
    @inbounds @simd for k in axes(out, 1)
        out[k] = zero(T)
    end
    return out
end
