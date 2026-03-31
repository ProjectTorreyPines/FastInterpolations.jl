# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    SERIES INTERPOLANT UTILITIES                           ║
# ║              Validation helpers for AbstractSeriesInterpolant             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: abstract_types.jl → ... → series_utils.jl
#

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
    length(outputs) != n_series && throw(
        DimensionMismatch(
            "outputs length $(length(outputs)) must match n_series $n_series"
        )
    )
    for (k, out) in enumerate(outputs)
        length(out) != n_query && throw(
            DimensionMismatch(
                "outputs[$k] length $(length(out)) must match n_query $n_query"
            )
        )
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
    length(output) == n_series || throw(
        DimensionMismatch(
            "output length $(length(output)) must match n_series $n_series"
        )
    )
    return nothing
end

# ========================================
# Extrapolation Helpers
# ========================================
# Pure data-only helpers for constant extrapolation.
# Used by Linear, Cubic, Constant, and Quadratic series interpolants.

"""
    _boundary_point_index(state::UInt8, n_pts::Int) -> Int

Return the boundary point index for constant extrapolation.

# Arguments
- `state::UInt8`: `OOB_LEFT` for left boundary, `OOB_RIGHT` for right boundary
- `n_pts::Int`: Total number of grid points

# Returns
- `1` if `state == OOB_LEFT` (left boundary)
- `n_pts` if `state == OOB_RIGHT` (right boundary)
"""
@inline _boundary_point_index(state::UInt8, n_pts::Int) = state == OOB_LEFT ? 1 : n_pts

"""
    _throw_extrap_domain_error(xq::T, x_min::T, x_max::T)

Throw a DomainError for query points outside the interpolation domain.

This function is marked `@noinline` to keep the error path cold and reduce
code size in the hot interpolation loops.

# Throws
- `DomainError` with message indicating the query point and valid domain
"""
@noinline function _throw_extrap_domain_error(xq::T, x_min::T, x_max::T) where {T <: AbstractFloat}
    throw(DomainError(xq, "query point outside domain [$x_min, $x_max]"))
end

"""
    _constant_extrap_boundary_value(y, side, n_pts, k, op, extrap) -> T

Get the boundary value for constant/fill extrapolation in scalar series evaluation path.

For `EvalValue` + `ClampExtrap`, returns the boundary y-value.
For `EvalValue` + `FillExtrap`, returns the fill value.
For derivatives, returns zero via `0 * y` (duck-typing compatible).
"""
@inline function _constant_extrap_boundary_value(
        y::Matrix{Tv}, side::UInt8, n_pts::Int, k::Int, ::EvalValue, ::ClampExtrap
    ) where {Tv}
    @inbounds return y[_boundary_point_index(side, n_pts), k]
end

@inline function _constant_extrap_boundary_value(
        ::Matrix{Tv}, ::UInt8, ::Int, ::Int, ::EvalValue, e::FillExtrap
    ) where {Tv}
    return e.fill_value
end

@inline function _constant_extrap_boundary_value(
        y::Matrix{Tv}, ::UInt8, ::Int, ::Int, ::Union{EvalDeriv1, EvalDeriv2, EvalDeriv3}, ::_ClampOrFill
    ) where {Tv}
    return 0 * first(y)
end

@inline function _constant_extrap_boundary_value(
        y::Matrix{Tv}, ::UInt8, ::Int, ::Int, ::DerivOp{N}, ::_ClampOrFill
    ) where {Tv, N}
    return 0 * first(y)
end

"""
    _fill_constant_extrap_simd!(out, y_point, side, n_pts, op, extrap) -> out

Fill output vector with boundary/fill values for constant/fill extrapolation (SIMD path).

For `EvalValue` + `ClampExtrap`, fills with boundary y-values.
For `EvalValue` + `FillExtrap`, fills with the fill value.
For derivatives, fills with zeros.
"""
@inline function _fill_constant_extrap_simd!(
        out::AbstractVector{Tv}, y_point::Matrix{Tv}, side::UInt8, n_pts::Int, ::EvalValue, ::ClampExtrap
    ) where {Tv}
    idx = _boundary_point_index(side, n_pts)
    @inbounds @simd for k in axes(out, 1)
        out[k] = y_point[k, idx]
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
        out::AbstractVector{Tv}, ::Matrix{Tv}, ::UInt8, ::Int, ::EvalValue, e::FillExtrap
    ) where {Tv}
    @inbounds @simd for k in axes(out, 1)
        out[k] = e.fill_value
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
        out::AbstractVector{Tv}, y::Matrix{Tv}, ::UInt8, ::Int, ::Union{EvalDeriv1, EvalDeriv2, EvalDeriv3}, ::_ClampOrFill
    ) where {Tv}
    z = 0 * first(y)
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
        out::AbstractVector{Tv}, y::Matrix{Tv}, ::UInt8, ::Int, ::DerivOp{N}, ::_ClampOrFill
    ) where {Tv, N}
    z = 0 * first(y)
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
    end
    return out
end
