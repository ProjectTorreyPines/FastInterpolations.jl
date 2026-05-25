# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    SERIES INTERPOLANT UTILITIES                           ║
# ║              Validation helpers for AbstractSeriesInterpolant             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Include order: abstract_types.jl → ... → series_utils.jl
#

# ========================================
# Series one-shot batch loop-order selection
# ========================================
# Vector-batch `*_interp!(outs, x, ::Series, xqs)` picks its loop order
# adaptively from `(NQ, K) = (length(xqs), n_series(s))`:
#
#   Q outer × K inner with stack-resident anchor per query.
#       Tight loop, no pool. Wins on tiny batches (no scratch buffer).
#
#   K outer × Q inner with pool-acquired anchor vector.
#       Inner loop streams one `outputs[k]` Vector — LLVM auto-SIMDs
#       and avoids the K-cache-line write jump per query of the Q×K
#       shape. Wins once NQ or K is large.
#
# The crossover is K-dependent (thresholds empirically tuned on Linear /
# Constant Series oneshot batch microbenchmarks):
#   - K ≤ ~100: NQ ≈ 16 is the cleanest break (Q×K wins below, K×Q above).
#   - K ≥ ~256: Q×K loses at every NQ — the K-inner loop can't SIMD across
#     K separate output Vectors and the K-cache-line write thrash per
#     query dominates. K×Q wins outright.
#
# `_series_use_kq_loop(NQ, K)` codifies both axes: route to the K×Q (pool)
# path if either NQ exceeds the per-K threshold OR K is large enough that
# Q×K can never beat the SIMD-friendly inner stream.
const _SERIES_BATCH_NQ_THRESHOLD = 16
const _SERIES_BATCH_K_THRESHOLD = 256

@inline _series_use_kq_loop(NQ::Int, K::Int) =
    NQ > _SERIES_BATCH_NQ_THRESHOLD || K >= _SERIES_BATCH_K_THRESHOLD

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
    # NOTE: broadcast `0 * first(y)` rather than per-k cell-local
    # `0 * y[_boundary_point_index(side, n_pts), k]`. The batch series eval
    # calls this K×Q times — per-k indexed load adds ~1ns/call (=50μs on a
    # 1000-series × 50-query batch, +94% on this hot path). Future fix:
    # NaN-presence detection at construction.
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
        out::AbstractVector{Tv}, y_point::Matrix{Tv}, ::UInt8, ::Int, ::Union{EvalDeriv1, EvalDeriv2, EvalDeriv3}, ::_ClampOrFill
    ) where {Tv}
    # NOTE: broadcast `z = 0 * first(y_point)` rather than per-k cell-local
    # `0 * y_point[k, idx]`. Per-series cell-local NaN propagation is desirable
    # in principle (consistency with `_eval_constant_series_anchored`) but the
    # per-k indexed load doubles cost on this hot batch-OOB-deriv path
    # (~+94% in 1000-series × 50-query benchmark) for a niche correctness
    # benefit. Future: detect NaN presence at construction and branch.
    z = 0 * first(y_point)
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
        out::AbstractVector{Tv}, y_point::Matrix{Tv}, ::UInt8, ::Int, ::DerivOp{N}, ::_ClampOrFill
    ) where {Tv, N}
    z = 0 * first(y_point)
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
    end
    return out
end
