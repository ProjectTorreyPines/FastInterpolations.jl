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

# Build the Vector{Vector{Tv}} container for a Series batch-allocating one-shot.
# Lives behind a typed barrier because comprehensions that close over a locally-
# bound `Tv = _promote_eltype(...)` lose concreteness in Julia 1.10's inference
# (the type doesn't propagate from the local binding into the closure body).
# Passing `Tv` as a `Type{Tv}` argument restores concrete `Vector{Vector{Tv}}`
# inference on 1.10 LTS while remaining a no-cost @inline on 1.11+/1.12+.
@inline _alloc_series_batch_outputs(::Type{Tv}, K::Int, NQ::Int) where {Tv} =
    [Vector{Tv}(undef, NQ) for _ in 1:K]

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
    _constant_extrap_boundary_value(y, side, n_pts, k, op, extrap, ::Type{Tq}) -> T

Boundary value for constant/fill extrapolation in the series-contiguous (batch)
OOB path. `EvalValue` returns the boundary y (or fill value); derivatives return
`0 * oob_data`. Threaded through `* one(Tq)` for the query carrier — `Tq` is the
type only, since the lean stateful anchors store no `xq`. The deriv source is the
OOB-cell data via `_extrap_oob_data` (ClampExtrap → boundary y, preserving
cell-local NaN; FillExtrap → fill_value). The OOB pins in
test/test_cubic_series_oob_pins.jl guard the shared contract.
"""
@inline function _constant_extrap_boundary_value(
        y::Matrix{Tv}, side::UInt8, n_pts::Int, k::Int, ::EvalValue, ::ClampExtrap, ::Type{Tq}, _
    ) where {Tv, Tq <: Real}
    @inbounds return y[_boundary_point_index(side, n_pts), k] * one(Tq)
end

@inline function _constant_extrap_boundary_value(
        ::Matrix{Tv}, ::UInt8, ::Int, ::Int, ::EvalValue, e::FillExtrap, ::Type{Tq}, _
    ) where {Tv, Tq <: Real}
    return e.fill_value * one(Tq)
end

@inline function _constant_extrap_boundary_value(
        y::Matrix{Tv}, side::UInt8, n_pts::Int, k::Int, ::AbstractEvalOp, ext::_ClampOrFill, ::Type{Tq}, deriv_oneunit
    ) where {Tv, Tq <: Real}
    src = _extrap_oob_data(ext, @inbounds(y[_boundary_point_index(side, n_pts), k]))
    return 0 * src * one(Tq) * deriv_oneunit
end

"""
    _fill_constant_extrap_simd!(out, y_point, side, n_pts, op, extrap, ::Type{Tq}) -> out

Fill `out` with boundary/fill values for constant/fill extrapolation (SIMD path,
point-contiguous OOB). Threaded through `* one(Tq)` — `Tq` is the type only,
since the lean stateful anchors store no `xq`; the carrier is loop-invariant so
LLVM hoists it and the loop stays a single load/mul per `k`. The deriv source is
per-`k` cell-local via `_extrap_oob_data` (ClampExtrap → `y_point[k, idx]`,
NaN-preserving; FillExtrap → `e.fill_value`). The shared-`Tv` constraint on
`out`/`y_point` is intentionally dropped: mixed-precision and Dual OOB queries
need `out` to carry a wider type than the stored `y_point`.
"""
@inline function _fill_constant_extrap_simd!(
        out::AbstractVector, y_point::Matrix, side::UInt8, n_pts::Int, ::EvalValue, ::ClampExtrap, ::Type{Tq}, _
    ) where {Tq <: Real}
    idx = _boundary_point_index(side, n_pts)
    xq_carrier = one(Tq)
    @inbounds @simd for k in axes(out, 1)
        out[k] = y_point[k, idx] * xq_carrier
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
        out::AbstractVector, ::Matrix, ::UInt8, ::Int, ::EvalValue, e::FillExtrap, ::Type{Tq}, _
    ) where {Tq <: Real}
    z = e.fill_value * one(Tq)
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
    end
    return out
end

@inline function _fill_constant_extrap_simd!(
        out::AbstractVector, y_point::Matrix, side::UInt8, n_pts::Int, ::AbstractEvalOp, ext::_ClampOrFill, ::Type{Tq}, deriv_oneunit
    ) where {Tq <: Real}
    idx = _boundary_point_index(side, n_pts)
    xq_carrier = one(Tq)
    @inbounds @simd for k in axes(out, 1)
        out[k] = 0 * _extrap_oob_data(ext, y_point[k, idx]) * xq_carrier * deriv_oneunit
    end
    return out
end
