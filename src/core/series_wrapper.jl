# ============================================================================
# Series input wrapper — dispatch disambiguator for multi-series interpolation
# ============================================================================
# Series{D, S} eliminates dispatch ambiguity between:
#   - Vector{<:AbstractVector} as series input (multiple y-series sharing a grid)
#   - Vector{<:AbstractVector} as vector-valued data (e.g., Vector{SVector{3,Float64}})
#
# The wrapper is consumed at construction time and never stored in the interpolant.
# The strategy parameter S controls evaluation strategy for one-shot vs interpolant paths.

# ─── Strategy types ──────────────────────────────────────────────────────────
# Zero-size singletons for dispatch.
# AutoStrategy: one-shot → loop-kernel, interpolant → SIMD matrix (default)
# LoopStrategy: force loop-kernel in all paths
# SIMDStrategy: force matrix SIMD in all paths (future)

abstract type AbstractSeriesStrategy end

"""Default strategy: loop-kernel for one-shot evaluation, SIMD matrix for interpolant construction."""
struct AutoStrategy <: AbstractSeriesStrategy end

"""Force loop-kernel strategy: search once, evaluate kernel per y-vector."""
struct LoopStrategy <: AbstractSeriesStrategy end

"""Force SIMD matrix strategy: transpose to point-contiguous layout. Reserved for future use."""
struct SIMDStrategy <: AbstractSeriesStrategy end

# ─── Series type ─────────────────────────────────────────────────────────────

"""
    Series(y1, y2, ...)                     # varargs
    Series([y1, y2, ...])                   # vector of vectors
    Series(Y::AbstractMatrix)               # matrix (columns = series)
    Series(...; strategy=AutoStrategy())    # explicit strategy

Input wrapper for multi-series interpolation. Wraps series data to disambiguate
from vector-valued interpolation inputs.

The wrapped data is not copied or promoted — type promotion is handled by the
interpolation constructor.

# Strategy
- `AutoStrategy()` (default): loop-kernel for one-shot, SIMD matrix for interpolant
- `LoopStrategy()`: force loop-kernel strategy
- `SIMDStrategy()`: force SIMD matrix strategy (reserved for future use)

# Examples
```julia
# One-shot evaluation (Strategy B: search once, kernel per y)
linear_interp(x, Series(y_sin, y_cos), 0.5)  # → (sin(0.5), cos(0.5))

# Interpolant construction (SIMD matrix path)
itp = linear_interp(x, Series(y_sin, y_cos))
itp(0.5)  # → [sin(0.5), cos(0.5)]

# Explicit strategy override
cubic_interp(x, Series(y1, y2; strategy=LoopStrategy()), 0.5)
```
"""
struct Series{D, S <: AbstractSeriesStrategy}
    data::D
    strategy::S
end

# Single vector: Series(y1) — single series (no element type constraint for duck-typing)
function Series(y::AbstractVector; strategy::AbstractSeriesStrategy = AutoStrategy())
    return Series{typeof((y,)), typeof(strategy)}((y,), strategy)
end

# Varargs: Series(y1, y2, y3)
function Series(y1::AbstractVector, y2::AbstractVector, rest::AbstractVector...; strategy::AbstractSeriesStrategy = AutoStrategy())
    data = (y1, y2, rest...)
    return Series{typeof(data), typeof(strategy)}(data, strategy)
end

# Vector of vectors: Series([y1, y2, y3])
function Series(ys::AbstractVector{<:AbstractVector}; strategy::AbstractSeriesStrategy = AutoStrategy())
    return Series{typeof(ys), typeof(strategy)}(ys, strategy)
end

# Matrix: Series(Y) where columns = series
function Series(Y::AbstractMatrix; strategy::AbstractSeriesStrategy = AutoStrategy())
    return Series{typeof(Y), typeof(strategy)}(Y, strategy)
end

# ─── Series data access helpers ───────────────────────────────────────────────
#
# _series_vectors:  iterable of column vectors (zero-alloc for all forms)
# n_series:         number of series (extends existing n_series for interpolants)
# _series_eltype:   element type of series data (for type promotion)
# _build_series_mat: build owned Matrix{Tv_out}(n_pts, n_ser) from any form

"""
    _series_vectors(s::Series) → iterable of vectors

Return an iterable over the individual y-series vectors.
Zero-allocation for all forms (Tuple, Vector-of-Vectors, Matrix via `eachcol`).
"""
@inline _series_vectors(s::Series{<:Tuple}) = s.data
@inline _series_vectors(s::Series{<:AbstractVector{<:AbstractVector}}) = s.data
@inline _series_vectors(s::Series{<:AbstractMatrix}) = eachcol(s.data)

"""Number of series in a Series wrapper."""
@inline n_series(s::Series{<:Tuple}) = length(s.data)
@inline n_series(s::Series{<:AbstractVector{<:AbstractVector}}) = length(s.data)
@inline n_series(s::Series{<:AbstractMatrix}) = size(s.data, 2)

"""Element type of the series data (for type promotion at construction time)."""
@inline _series_eltype(::Series{<:AbstractMatrix{Tv}}) where {Tv} = Tv
@inline _series_eltype(::Series{<:AbstractVector{<:AbstractVector{Tv}}}) where {Tv} = Tv
@inline _series_eltype(s::Series{<:AbstractVector{<:AbstractVector}}) =
    promote_type(map(eltype, s.data)...)
@inline _series_eltype(s::Series{<:Tuple}) = promote_type(map(eltype, s.data)...)

# ─── Validation helper for one-shot paths ─────────────────────────────────────

"""
    _validate_series_lengths(s::Series, n_pts::Int)

Validate that all series vectors have the expected length (matching the grid).
Throws `DimensionMismatch` on failure.
"""
@inline function _validate_series_lengths(s::Series, n_pts::Int)
    for (k, v) in enumerate(_series_vectors(s))
        length(v) == n_pts || _throw_series_length_mismatch(k, length(v), n_pts)
    end
    return nothing
end

@noinline function _throw_series_length_mismatch(k::Int, got::Int, expected::Int)
    throw(DimensionMismatch(
        "Series vector $k has length $got, expected $expected (length of x)"
    ))
end

# ─── @noinline throw helpers (keep cold error paths out of hot code) ──────────

@noinline function _throw_series_dim_mismatch(got::Int, expected::Int)
    throw(DimensionMismatch("output length $got must match number of series $expected"))
end

# ─── Core builder: Series → owned Matrix ──────────────────────────────────────

"""
    _build_series_mat(s::Series, n_pts::Int, ::Type{Tv_out}) → (Matrix{Tv_out}, n_ser::Int)

Build an owned `Matrix{Tv_out}(n_pts, n_ser)` from any `Series` form.

- **Matrix**: `copy(Y)` or type-promoted copy (single memcpy)
- **Tuple / Vector-of-Vectors**: allocate + in-place column fill (single allocation)

Validates that each series has exactly `n_pts` elements.
Returns the matrix and the number of series.
"""
function _build_series_mat(s::Series{<:AbstractMatrix}, n_pts::Int, ::Type{Tv_out}) where {Tv_out}
    Y = s.data
    size(Y, 1) == n_pts || throw(
        DimensionMismatch(
            "Matrix has $(size(Y, 1)) rows but grid has $n_pts points (expected n_points × n_series)"
        )
    )
    n_ser = size(Y, 2)
    n_ser > 0 || throw(ArgumentError("Series data must contain at least one series"))
    y_mat = eltype(Y) === Tv_out ? copy(Y) : Tv_out.(Y)
    return y_mat, n_ser
end

function _build_series_mat(s::Series, n_pts::Int, ::Type{Tv_out}) where {Tv_out}
    vecs = _series_vectors(s)
    n_ser = n_series(s)
    n_ser > 0 || throw(ArgumentError("Series data must contain at least one series"))
    y_mat = Matrix{Tv_out}(undef, n_pts, n_ser)
    @inbounds for (k, v) in enumerate(vecs)
        length(v) == n_pts || throw(
            DimensionMismatch(
                "Series vector $k has length $(length(v)), expected $n_pts (length of x)"
            )
        )
        y_mat[:, k] .= v
    end
    return y_mat, n_ser
end
