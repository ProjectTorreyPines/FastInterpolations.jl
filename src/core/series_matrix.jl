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
    LazyTranspose{Tv}

Thread-safe lazy transpose holder for a single matrix.

Used by Linear, Constant, and Quadratic series interpolants that store only y values.
The transpose is computed on first scalar query and cached for subsequent calls.

# Type Parameter
- `Tv`: Value type (unconstrained)

# Thread Safety
Uses atomic acquire/release semantics (RCU pattern) for safe concurrent access.
Multiple threads may race to compute the transpose, but all will get the same result.
"""
mutable struct LazyTranspose{Tv}
    @atomic snapshot::Union{Nothing, Matrix{Tv}}

    LazyTranspose{Tv}() where {Tv} = new{Tv}(nothing)
end

"""
    _get_snapshot(lt::LazyTranspose) -> Union{Nothing, Matrix}

Get current snapshot (for testing). Returns nothing if not yet computed.
"""
@inline _get_snapshot(lt::LazyTranspose) = @atomic :acquire lt.snapshot

"""
    _ensure_transpose!(lt::LazyTranspose{Tv}, src::Matrix{Tv}) -> Matrix{Tv}

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
@inline function _ensure_transpose!(lt::LazyTranspose{Tv}, src::Matrix{Tv}) where {Tv}
    # Fast path: check if already populated
    snap = @atomic :acquire lt.snapshot
    snap !== nothing && return snap::Matrix{Tv}

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
    LazyTransposePair{Tv}

Thread-safe lazy transpose holder for paired matrices (y and z).

Used by CubicSeriesInterpolant which stores both y values and z (second derivatives).
Both transposes are computed together on first scalar query.

# Type Parameter
- `Tv`: Value type (unconstrained)

# Thread Safety
Same RCU pattern as LazyTranspose.
"""
mutable struct LazyTransposePair{Tv, Tz}
    @atomic snapshot::Union{Nothing, Tuple{Matrix{Tv}, Matrix{Tz}}}

    LazyTransposePair{Tv, Tz}() where {Tv, Tz} = new{Tv, Tz}(nothing)
    LazyTransposePair{Tv}() where {Tv} = new{Tv, Tv}(nothing)  # backward compat
end

"""
    _get_snapshot(ltp::LazyTransposePair) -> Union{Nothing, Tuple{Matrix, Matrix}}

Get current snapshot pair (for testing). Returns nothing if not yet computed.
"""
@inline _get_snapshot(ltp::LazyTransposePair) = @atomic :acquire ltp.snapshot

"""
    _ensure_transpose_pair!(ltp::LazyTransposePair{Tv}, y::Matrix{Tv}, z::Matrix{Tv}) -> (Matrix{Tv}, Matrix{Tv})

Ensure point-contiguous transpose pair exists. Thread-safe via atomic RCU pattern.

# Arguments
- `ltp`: LazyTransposePair holder
- `y`: Y-values matrix (n_points × n_series)
- `z`: Z-values matrix (n_points × n_series), typically second derivatives

# Returns
Tuple of transposed matrices (y_point, z_point), each (n_series × n_points).
"""
@inline function _ensure_transpose_pair!(
        ltp::LazyTransposePair{Tv, Tz},
        y::Matrix{Tv},
        z::Matrix{Tz}
    ) where {Tv, Tz}
    # Fast path: check if already populated
    snap = @atomic :acquire ltp.snapshot
    snap !== nothing && return snap::Tuple{Matrix{Tv}, Matrix{Tz}}

    # Slow path: compute both transposes
    y_point = permutedims(y)
    z_point = permutedims(z)
    pair = (y_point, z_point)

    # Atomic publish
    @atomic :release ltp.snapshot = pair

    return pair
end

# ════════════════════════════════════════════════════════════════════════════
# LAZY TRANSPOSE TRIPLE - Three Matrices (for Quadratic)
# ════════════════════════════════════════════════════════════════════════════

"""
    LazyTransposeTriple{Tv}

Thread-safe lazy transpose holder for three matrices (y, a, d).

Used by QuadraticSeriesInterpolant which stores y values plus a and d coefficients.
All three transposes are computed together on first scalar query.

# Type Parameter
- `Tv`: Value type (unconstrained)

# Thread Safety
Same RCU pattern as LazyTranspose.
"""
mutable struct LazyTransposeTriple{Tv, Tc}
    @atomic snapshot::Union{Nothing, Tuple{Matrix{Tv}, Matrix{Tc}, Matrix{Tc}}}

    LazyTransposeTriple{Tv, Tc}() where {Tv, Tc} = new{Tv, Tc}(nothing)
    LazyTransposeTriple{Tv}() where {Tv} = new{Tv, Tv}(nothing)  # backward compat
end

"""
    _get_snapshot(ltt::LazyTransposeTriple) -> Union{Nothing, Tuple{Matrix, Matrix, Matrix}}

Get current snapshot triple (for testing). Returns nothing if not yet computed.
"""
@inline _get_snapshot(ltt::LazyTransposeTriple) = @atomic :acquire ltt.snapshot

"""
    _ensure_transpose_triple!(ltt::LazyTransposeTriple{Tv}, y::Matrix{Tv}, a::Matrix{Tv}, d::Matrix{Tv}) -> (Matrix{Tv}, Matrix{Tv}, Matrix{Tv})

Ensure point-contiguous transpose triple exists. Thread-safe via atomic RCU pattern.

# Arguments
- `ltt`: LazyTransposeTriple holder
- `y`: Y-values matrix (n_points × n_series)
- `a`: Quadratic coefficients matrix (n_points × n_series)
- `d`: Slope coefficients matrix (n_points × n_series)

# Returns
Tuple of transposed matrices (y_point, a_point, d_point), each (n_series × n_points).
"""
@inline function _ensure_transpose_triple!(
        ltt::LazyTransposeTriple{Tv, Tc},
        y::Matrix{Tv},
        a::Matrix{Tc},
        d::Matrix{Tc}
    ) where {Tv, Tc}
    # Fast path: check if already populated
    snap = @atomic :acquire ltt.snapshot
    snap !== nothing && return snap::Tuple{Matrix{Tv}, Matrix{Tc}, Matrix{Tc}}

    # Slow path: compute all three transposes
    y_point = permutedims(y)
    a_point = permutedims(a)
    d_point = permutedims(d)
    triple = (y_point, a_point, d_point)

    # Atomic publish
    @atomic :release ltt.snapshot = triple

    return triple
end

# ════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE HELPERS
# ════════════════════════════════════════════════════════════════════════════

"""
    precompute_transpose!(lt::LazyTranspose, src::Matrix) -> LazyTranspose

Pre-compute transpose before hot loops. Returns the holder for chaining.
"""
function precompute_transpose!(lt::LazyTranspose{Tv}, src::Matrix{Tv}) where {Tv}
    _ensure_transpose!(lt, src)
    return lt
end

"""
    precompute_transpose!(ltp::LazyTransposePair, y::Matrix, z::Matrix) -> LazyTransposePair

Pre-compute transpose pair before hot loops. Returns the holder for chaining.
"""
function precompute_transpose!(ltp::LazyTransposePair{Tv, Tz}, y::Matrix{Tv}, z::Matrix{Tz}) where {Tv, Tz}
    _ensure_transpose_pair!(ltp, y, z)
    return ltp
end

"""
    precompute_transpose!(ltt::LazyTransposeTriple, y::Matrix, a::Matrix, d::Matrix) -> LazyTransposeTriple

Pre-compute transpose triple before hot loops. Returns the holder for chaining.
"""
function precompute_transpose!(ltt::LazyTransposeTriple{Tv, Tc}, y::Matrix{Tv}, a::Matrix{Tc}, d::Matrix{Tc}) where {Tv, Tc}
    _ensure_transpose_triple!(ltt, y, a, d)
    return ltt
end
