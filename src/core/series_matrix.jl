# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  SERIES MATRIX INFRASTRUCTURE                             ║
# ║         Lazy transpose and matrix storage helpers for SeriesInterpolants  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Extracted from CubicSeriesInterpolant for reuse across all series types.
#
# Key pattern: Series-contiguous storage (n_points × n_series) for vector queries,
# with lazy transpose to point-contiguous (n_series × n_points) for scalar SIMD.
#
# Include order: ... → series_interface.jl → series_matrix.jl
#

# ════════════════════════════════════════════════════════════════════════════
# LAZY TRANSPOSE - Single Matrix
# ════════════════════════════════════════════════════════════════════════════

"""
    LazyTranspose{T}

Thread-safe lazy transpose holder for a single matrix.

Used by Linear, Constant, and Quadratic series interpolants that store only y values.
The transpose is computed on first scalar query and cached for subsequent calls.

# Thread Safety
Uses atomic acquire/release semantics (RCU pattern) for safe concurrent access.
Multiple threads may race to compute the transpose, but all will get the same result.
"""
mutable struct LazyTranspose{T<:AbstractFloat}
    @atomic snapshot::Union{Nothing, Matrix{T}}

    LazyTranspose{T}() where {T} = new{T}(nothing)
end

"""
    _get_snapshot(lt::LazyTranspose) -> Union{Nothing, Matrix}

Get current snapshot (for testing). Returns nothing if not yet computed.
"""
@inline _get_snapshot(lt::LazyTranspose) = @atomic :acquire lt.snapshot

"""
    _ensure_transpose!(lt::LazyTranspose{T}, src::Matrix{T}) -> Matrix{T}

Ensure point-contiguous transpose exists. Thread-safe via atomic RCU pattern.

# Arguments
- `lt`: LazyTranspose holder
- `src`: Source matrix (n_points × n_series), series-contiguous

# Returns
Transposed matrix (n_series × n_points), point-contiguous for SIMD scalar evaluation.

# Performance
- Fast path: atomic read, return cached transpose
- Slow path: compute permutedims, atomic publish
"""
@inline function _ensure_transpose!(lt::LazyTranspose{T}, src::Matrix{T}) where {T<:AbstractFloat}
    # Fast path: check if already populated
    snap = @atomic :acquire lt.snapshot
    snap !== nothing && return snap::Matrix{T}

    # Slow path: compute transpose
    transposed = permutedims(src)

    # Atomic publish (racing threads will compute same result)
    @atomic :release lt.snapshot = transposed

    return transposed
end

# ════════════════════════════════════════════════════════════════════════════
# LAZY TRANSPOSE PAIR - Dual Matrices (for Cubic)
# ════════════════════════════════════════════════════════════════════════════

"""
    LazyTransposePair{T}

Thread-safe lazy transpose holder for paired matrices (y and z).

Used by CubicSeriesInterpolant which stores both y values and z (second derivatives).
Both transposes are computed together on first scalar query.

# Thread Safety
Same RCU pattern as LazyTranspose.
"""
mutable struct LazyTransposePair{T<:AbstractFloat}
    @atomic snapshot::Union{Nothing, Tuple{Matrix{T}, Matrix{T}}}

    LazyTransposePair{T}() where {T} = new{T}(nothing)
end

"""
    _get_snapshot(ltp::LazyTransposePair) -> Union{Nothing, Tuple{Matrix, Matrix}}

Get current snapshot pair (for testing). Returns nothing if not yet computed.
"""
@inline _get_snapshot(ltp::LazyTransposePair) = @atomic :acquire ltp.snapshot

"""
    _ensure_transpose_pair!(ltp::LazyTransposePair{T}, y::Matrix{T}, z::Matrix{T}) -> (Matrix{T}, Matrix{T})

Ensure point-contiguous transpose pair exists. Thread-safe via atomic RCU pattern.

# Arguments
- `ltp`: LazyTransposePair holder
- `y`: Y-values matrix (n_points × n_series)
- `z`: Z-values matrix (n_points × n_series), typically second derivatives

# Returns
Tuple of transposed matrices (y_point, z_point), each (n_series × n_points).
"""
@inline function _ensure_transpose_pair!(
    ltp::LazyTransposePair{T},
    y::Matrix{T},
    z::Matrix{T}
) where {T<:AbstractFloat}
    # Fast path: check if already populated
    snap = @atomic :acquire ltp.snapshot
    snap !== nothing && return snap::Tuple{Matrix{T}, Matrix{T}}

    # Slow path: compute both transposes
    y_point = permutedims(y)
    z_point = permutedims(z)
    pair = (y_point, z_point)

    # Atomic publish
    @atomic :release ltp.snapshot = pair

    return pair
end

# ════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE HELPERS
# ════════════════════════════════════════════════════════════════════════════

"""
    precompute_transpose!(lt::LazyTranspose, src::Matrix) -> LazyTranspose

Pre-compute transpose before hot loops. Returns the holder for chaining.
"""
function precompute_transpose!(lt::LazyTranspose{T}, src::Matrix{T}) where {T}
    _ensure_transpose!(lt, src)
    return lt
end

"""
    precompute_transpose!(ltp::LazyTransposePair, y::Matrix, z::Matrix) -> LazyTransposePair

Pre-compute transpose pair before hot loops. Returns the holder for chaining.
"""
function precompute_transpose!(ltp::LazyTransposePair{T}, y::Matrix{T}, z::Matrix{T}) where {T}
    _ensure_transpose_pair!(ltp, y, z)
    return ltp
end
