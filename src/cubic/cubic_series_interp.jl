# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CUBIC SERIES INTERPOLATION                             ║
# ║         Multiple y-data series sharing the same x-grid                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Unified matrix storage for optimal performance.
# Key optimization: Adaptive layout with lazy transpose for scalar queries.
#
# Include order: ... → cubic_types.jl → ... → cubic_series_interp.jl
#
# Note: TransposeSnapshot is defined in cubic_types.jl
#

# ========================================
# Type Definition
# ========================================

"""
    CubicSeriesInterpolant{T, C, B}

Multi-series cubic spline interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

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

sitp = cubic_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = sitp(0.5)            # Returns Vector{Float64} of length 3
sitp(output, 0.5)           # In-place

# Vector evaluation
vals = sitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
sitp([out1, out2, out3], xq)    # In-place (zero allocation)
```

# Performance
- Vector queries use series-contiguous layout directly
- Scalar queries trigger lazy transpose on first call
- All series share same cache (O(1) memory overhead)

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. The `const` annotation ensures fields
cannot be reassigned while allowing heap allocation. This pattern provides:
- Stable memory addresses for the compiler to optimize field access
- Better inlining of field accesses compared to plain immutable structs
- Compatibility with mutable inner types (LazyTransposePair with @atomic)
Benchmarks show ~15% regression when using plain `struct` instead.
"""
mutable struct CubicSeriesInterpolant{
    T<:AbstractFloat,
    C<:CubicSplineCache{T},
    B
} <: AbstractSeriesInterpolant{T}
    const cache::C                    # Shared cache with LU factorization
    const bc_for_solve::B             # BC config for solving
    const y::Matrix{T}                # Series-contiguous y (n_points × n_series)
    const z::Matrix{T}                # Series-contiguous z (n_points × n_series)
    const _transpose::LazyTransposePair{T}  # Lazy point-contiguous layout (shared infra)
    const extrap::ExtrapVal           # Extrapolation mode

    function CubicSeriesInterpolant(
        cache::C,
        bc_for_solve::B,
        y::Matrix{T},
        z::Matrix{T},
        extrap::ExtrapVal
    ) where {T<:AbstractFloat, C<:CubicSplineCache{T}, B}
        new{T, C, B}(
            cache, bc_for_solve, y, z,
            LazyTransposePair{T}(),
            extrap
        )
    end
end

# ========================================
# Helper Functions
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(sitp::CubicSeriesInterpolant) = sitp.extrap === Val(:wrap)

"""Number of series in the interpolant."""
@inline n_series(sitp::CubicSeriesInterpolant) = size(sitp.y, 2)

"""Number of grid points in the interpolant."""
@inline n_points(sitp::CubicSeriesInterpolant) = size(sitp.y, 1)

"""Return the x-grid vector."""
@inline _get_grid(sitp::CubicSeriesInterpolant) = sitp.cache.x

"""Return the extrapolation mode."""
@inline _get_extrap(sitp::CubicSeriesInterpolant) = sitp.extrap

# ========================================
# Series Interface Traits
# ========================================

"""Return the interpolation method kind for kernel dispatch."""
@inline _method_kind(::Type{<:CubicSeriesInterpolant}) = Val(:cubic)

"""
    _make_anchor(sitp::CubicSeriesInterpolant, xq::T) -> _CubicAnchoredQuery{T}

Build anchor for a query point. Required trait for AbstractSeriesInterpolant.
"""
@inline function _make_anchor(sitp::CubicSeriesInterpolant{T}, xq::T, searcher::Searcher=DEFAULT_SEARCHER) where T
    return _anchor_query(sitp.cache.x, xq; wrap=_should_wrap(sitp), searcher=searcher)
end

"""
    _eval_series_at_anchor!(output, sitp::CubicSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.
"""
@inline function _eval_series_at_anchor!(
    output::AbstractVector{T},
    sitp::CubicSeriesInterpolant{T},
    aq::_CubicAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    y_point, z_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = T(first(sitp.cache.x)), T(last(sitp.cache.x))

    _eval_series_point_with_extrap!(output, y_point, z_point, n_pts, x_min, x_max, aq, sitp.extrap, op)
    return output
end

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::CubicSeriesInterpolant{T}) -> (y_point, z_point)

Ensure point-contiguous layout exists. Delegates to shared LazyTransposePair infrastructure.
"""
@inline function _ensure_point_layout!(sitp::CubicSeriesInterpolant{T}) where T
    return _ensure_transpose_pair!(sitp._transpose, sitp.y, sitp.z)
end

"""
    precompute_transpose!(sitp::CubicSeriesInterpolant) -> sitp

Pre-allocate point-contiguous matrices for scalar queries.
Call before hot loops to avoid first-call latency.
"""
function precompute_transpose!(sitp::CubicSeriesInterpolant)
    _ensure_point_layout!(sitp)
    return sitp
end

# ========================================
# SIMD Scalar Evaluation Kernels
# ========================================

"""
    _eval_series_point!(out, y_point, z_point, aq, op)

SIMD-optimized evaluation for point-contiguous layout (n_series × n_points).
Contiguous column access enables vectorization across series dimension.

Dispatches on concrete EvalOp for optimal performance:
- EvalValue, EvalDeriv1: Full 4-term evaluation
- EvalDeriv2, EvalDeriv3: Optimized 2-term evaluation (no y-loads)
"""
# EvalValue: Full 4-term evaluation
@inline function _eval_series_point!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    aq::_CubicAnchoredQuery{T},
    ::EvalValue
) where {T<:AbstractFloat}
    idx = aq.idx
    idx1 = idx + 1
    wyL, wyR, wzL, wzR = aq.w0

    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end
    return out
end

# EvalDeriv1: Full 4-term evaluation
@inline function _eval_series_point!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    aq::_CubicAnchoredQuery{T},
    ::EvalDeriv1
) where {T<:AbstractFloat}
    idx = aq.idx
    idx1 = idx + 1
    wyL, wyR, wzL, wzR = aq.w1

    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end
    return out
end

# EvalDeriv2: Optimized 2-term evaluation (no y-loads)
@inline function _eval_series_point!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    aq::_CubicAnchoredQuery{T},
    ::EvalDeriv2
) where {T<:AbstractFloat}
    idx = aq.idx
    idx1 = idx + 1
    wzL, wzR = aq.w2

    @inbounds @simd for k in axes(out, 1)
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wzR, zR, wzL * zL)
    end
    return out
end

# EvalDeriv3: Optimized 2-term evaluation (no y-loads)
@inline function _eval_series_point!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    aq::_CubicAnchoredQuery{T},
    ::EvalDeriv3
) where {T<:AbstractFloat}
    idx = aq.idx
    idx1 = idx + 1
    wzL, wzR = aq.w3

    @inbounds @simd for k in axes(out, 1)
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wzR, zR, wzL * zL)
    end
    return out
end

"""
    _eval_series_point_with_extrap!(out, y_point, z_point, n_pts, x_min, x_max, aq, extrap, op)

SIMD evaluation with extrapolation handling for multi-series.
"""
@inline function _eval_series_point_with_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_CubicAnchoredQuery{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_series_point!(out, y_point, z_point, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    _eval_series_point_extrap!(out, y_point, z_point, n_pts, x_min, x_max, aq, extrap, op, aq.side)
end

# :none - throw DomainError
@inline function _eval_series_point_extrap!(
    ::AbstractVector{T},
    ::Matrix{T},
    ::Matrix{T},
    ::Int,
    x_min::T,
    x_max::T,
    aq::_CubicAnchoredQuery{T},
    ::Val{:none},
    ::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# :constant - clamp to boundary (value only, derivatives are zero)
@inline function _eval_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    ::Matrix{T},
    n_pts::Int,
    ::T,
    ::T,
    ::_CubicAnchoredQuery{T},
    ::Val{:constant},
    op::AbstractEvalOp,
    side::UInt8
) where {T<:AbstractFloat}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op)
end

# :extension - extend polynomial (EvalValue)
@inline function _eval_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    ::T,
    ::T,
    aq::_CubicAnchoredQuery{T},
    ::Val{:extension},
    ::EvalValue,
    side::UInt8
) where {T<:AbstractFloat}
    idx = side == 0x01 ? 1 : (n_pts - 1)
    idx1 = idx + 1
    wyL, wyR, wzL, wzR = aq.w0

    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end
    return out
end

# :extension - extend polynomial (EvalDeriv1)
@inline function _eval_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    ::T,
    ::T,
    aq::_CubicAnchoredQuery{T},
    ::Val{:extension},
    ::EvalDeriv1,
    side::UInt8
) where {T<:AbstractFloat}
    idx = side == 0x01 ? 1 : (n_pts - 1)
    idx1 = idx + 1
    wyL, wyR, wzL, wzR = aq.w1

    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
    end
    return out
end

# :extension - extend polynomial (EvalDeriv2) - optimized, no y-loads
@inline function _eval_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    ::T,
    ::T,
    aq::_CubicAnchoredQuery{T},
    ::Val{:extension},
    ::EvalDeriv2,
    side::UInt8
) where {T<:AbstractFloat}
    idx = side == 0x01 ? 1 : (n_pts - 1)
    idx1 = idx + 1
    wzL, wzR = aq.w2

    @inbounds @simd for k in axes(out, 1)
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wzR, zR, wzL * zL)
    end
    return out
end

# :extension - extend polynomial (EvalDeriv3) - optimized, no y-loads
@inline function _eval_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    z_point::Matrix{T},
    n_pts::Int,
    ::T,
    ::T,
    aq::_CubicAnchoredQuery{T},
    ::Val{:extension},
    ::EvalDeriv3,
    side::UInt8
) where {T<:AbstractFloat}
    idx = side == 0x01 ? 1 : (n_pts - 1)
    idx1 = idx + 1
    wzL, wzR = aq.w3

    @inbounds @simd for k in axes(out, 1)
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wzR, zR, wzL * zL)
    end
    return out
end

# ========================================
# Internal: Coefficient Solver
# ========================================

"""
    _solve_series_coefficients!(z_mat, y_mat, cache, bc_for_solve)

Solve cubic spline systems for all series using shared LU factorization.
"""
@with_pool pool function _solve_series_coefficients!(
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

"""
    _solve_series_with_bc_array!(z_mat, y_mat, x, bc_array, autocache)

Solve cubic spline systems for series with per-series boundary conditions.
Groups series by BC type for cache efficiency.

# Arguments
- `z_mat`: Output matrix for second derivatives (n_points × n_series)
- `y_mat`: Input y-values matrix (n_points × n_series)
- `x`: x-grid vector
- `bc_array`: Vector of BCPair, one per series
- `autocache`: Whether to use cache pool
"""
@with_pool pool function _solve_series_with_bc_array!(
    z_mat::Matrix{T},
    y_mat::Matrix{T},
    x::AbstractVector{T},
    bc_array::AbstractVector{<:BCPair{T}},
    autocache::Bool
) where {T<:AbstractFloat}
    n_series = size(y_mat, 2)

    # Group series by BC type for cache reuse
    # Dict: typeof(bc) => (cache, indices)
    type_groups = Dict{DataType, Tuple{CubicSplineCache{T}, Vector{Int}}}()

    for k in 1:n_series
        bc = bc_array[k]
        bc_type = typeof(bc)

        if haskey(type_groups, bc_type)
            # Cache already exists for this BC type → just add index
            push!(type_groups[bc_type][2], k)
        else
            # New BC type → get cache from pool
            cache = _get_cubic_cache(x, bc, autocache)
            type_groups[bc_type] = (cache, [k])
        end
    end

    # Solve each group using its cache
    for (_, (cache, indices)) in type_groups
        for k in indices
            _solve_system!(@view(z_mat[:, k]), cache, @view(y_mat[:, k]), bc_array[k])
        end
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
- `bc`: Boundary condition (NaturalBC, ClampedBC, PeriodicBC, or Vector of BC for per-series)
- `extrap`: Extrapolation mode (:none, :constant, :extension, :wrap)
- `autocache`: If true, reuse cached LU factorization (default: true)
- `precompute_transpose`: If true, build point-contiguous layout immediately

# Returns
`CubicSeriesInterpolant` object with matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

# Uniform BC for all series
sitp = cubic_interp(x, [y1, y2, y3])
vals = sitp(0.5)  # [sin(π), cos(π), exp(-0.5)]

# Per-series BC (each series can have different BC)
sitp = cubic_interp(x, [y1, y2, y3]; bc=[
    NaturalBC(),
    BCPair(Deriv1(2.0), Deriv1(0.0)),
    BCPair(Deriv2(0.0), Deriv3(5.0)),
])
```
"""
function cubic_interp(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
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

    # Handle periodic BC separately (only for scalar BC)
    if bc isa AbstractBC && _is_periodic_bc(bc)
        return _build_series_periodic(x, y_mat, n_pts, n_series_count, autocache, precompute_transpose)
    end

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_pts, n_series_count)

    if bc isa AbstractVector
        # Per-series BC array
        bc_array = _normalize_bc_array(bc, T, n_series_count)
        _solve_series_with_bc_array!(z_mat, y_mat, x, bc_array, autocache)
        # All per-series caches share the same x-grid, so any BC's cache is valid here.
        # We store the first BC as a "representative" for struct compatibility and
        # x-grid access via _get_grid(cache); the specific BC chosen does not affect evaluation.
        bc_representative = bc_array[1]
        cache = _get_cubic_cache(x, bc_representative, autocache)
    else
        # Uniform BC (original path)
        bc_pair = _normalize_bc(bc, T)
        cache = _get_cubic_cache(x, bc_pair, autocache)
        _solve_series_coefficients!(z_mat, y_mat, cache, bc_pair)
        bc_representative = bc_pair
    end

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    sitp = CubicSeriesInterpolant(cache, bc_representative, y_mat, z_mat, extrap_val)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

"""
Internal helper for periodic BC multi-interpolant construction.
"""
function _build_series_periodic(
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
    _solve_series_coefficients!(z_mat, y_mat, cache, cache.bc_config)

    # Periodic BC always uses :wrap extrapolation
    sitp = CubicSeriesInterpolant(cache, cache.bc_config, y_mat, z_mat, Val(:wrap))

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
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

sitp = cubic_interp(x, Y)
```
"""
function cubic_interp(
    x::AbstractVector{T},
    Y::AbstractMatrix{T};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
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

    # Handle periodic BC separately (only for scalar BC)
    if bc isa AbstractBC && _is_periodic_bc(bc)
        return _build_series_periodic(x, y_mat, n_pts, n_series_count, autocache, precompute_transpose)
    end

    # Build z matrix by solving systems
    z_mat = Matrix{T}(undef, n_pts, n_series_count)

    if bc isa AbstractVector
        # Per-series BC array
        bc_array = _normalize_bc_array(bc, T, n_series_count)
        _solve_series_with_bc_array!(z_mat, y_mat, x, bc_array, autocache)
        # All per-series caches share the same x-grid, so any BC's cache is valid here.
        # We store the first BC as a "representative" for struct compatibility and
        # x-grid access via _get_grid(cache); the specific BC chosen does not affect evaluation.
        bc_representative = bc_array[1]
        cache = _get_cubic_cache(x, bc_representative, autocache)
    else
        # Uniform BC (original path)
        bc_pair = _normalize_bc(bc, T)
        cache = _get_cubic_cache(x, bc_pair, autocache)
        _solve_series_coefficients!(z_mat, y_mat, cache, bc_pair)
        bc_representative = bc_pair
    end

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    sitp = CubicSeriesInterpolant(cache, bc_representative, y_mat, z_mat, extrap_val)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

# Real type wrappers (auto-promote to Float)
function cubic_interp(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
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
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
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
    (sitp::CubicSeriesInterpolant)(xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.
"""
function (sitp::CubicSeriesInterpolant{T})(xq::S; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat, S<:Real}
    out = Vector{T}(undef, n_series(sitp))
    return sitp(out, xq; deriv=deriv, search=search)
end

"""
    (sitp::CubicSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (sitp::CubicSeriesInterpolant{T})(
    output::AbstractVector{T},
    xq::S;
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:AbstractFloat, S<:Real}
    _validate_scalar_output(output, n_series(sitp))

    xq_typed = T(xq)

    # Build anchor using trait
    aq = _make_anchor(sitp, xq_typed, _to_searcher(search))

    # Dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        _eval_series_at_anchor!(output, sitp, aq, op)
    end
    return output
end

# ========================================
# Vector Evaluation
# ========================================

"""
    (sitp::CubicSeriesInterpolant)(xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
"""
function (sitp::CubicSeriesInterpolant{T})(
    xq::AbstractVector{S};
    deriv::Int=0,
    search::AbstractSearchPolicy=LinearBounded()
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)
    n_query = length(xq_typed)

    outputs = [Vector{T}(undef, n_query) for _ in 1:n_series(sitp)]
    sitp(outputs, xq_typed; deriv=deriv, search=search)

    return outputs
end

"""
    (sitp::CubicSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points
- `deriv`: Derivative order (0, 1, or 2)

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
Uses task-local pool for anchor vector to achieve zero allocation after warmup.
"""
@with_pool pool function (sitp::CubicSeriesInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0,
    search::AbstractSearchPolicy=LinearBounded()
) where {T<:AbstractFloat}
    n_query = length(xq)
    n_ser = n_series(sitp)

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

    # Build anchors from pool (zero allocation after warmup)
    aq_vec = acquire!(pool, _CubicAnchoredQuery{T}, length(xq))
    _fill_anchors!(aq_vec, sitp.cache.x, xq; wrap=_should_wrap(sitp), searcher=_to_searcher(search))

    # Extract matrices for argument-passing pattern
    y, z = sitp.y, sitp.z
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    x_min, x_max = T(first(sitp.cache.x)), T(last(sitp.cache.x))

    # Evaluate all series
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_series_vector!(outputs[k], y, z, n_pts, x_min, x_max, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

# Real type wrapper for in-place vector
function (sitp::CubicSeriesInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S};
    deriv::Int=0,
    search::AbstractSearchPolicy=LinearBounded()
) where {T<:AbstractFloat, S<:Real}
    xq_typed = _to_float(xq, T)
    return sitp(outputs, xq_typed; deriv=deriv, search=search)
end

"""
    (sitp::CubicSeriesInterpolant)(outputs, aq_vec::AbstractVector{<:_CubicAnchoredQuery}; deriv=0)

Evaluate multi-Y interpolant with pre-built anchors (TRUE zero-allocation).

For maximum performance in hot loops, pre-build anchors once and reuse:
```julia
x = ...
sitp = cubic_interp(x, [y1, y2, y3])
xq = [0.1, 0.2, 0.3, ...]

# Pre-build anchors (allocates once)
aq_vec = FastInterpolations._anchor_query(x, xq)

# Zero-allocation loop
outputs = [similar(xq) for _ in 1:3]
for _ in 1:1000
    sitp(outputs, aq_vec)  # Zero allocation!
end
```
"""
function (sitp::CubicSeriesInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    n_query = length(aq_vec)
    n_ser = n_series(sitp)

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

    # Extract matrices for argument-passing pattern
    y, z = sitp.y, sitp.z
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    x_min, x_max = T(first(sitp.cache.x)), T(last(sitp.cache.x))

    # Evaluate all series
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_series_vector!(outputs[k], y, z, n_pts, x_min, x_max, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance (avoids struct field access in loop).
"""
@inline function _eval_series_vector!(
    out::AbstractVector{T},
    y::Matrix{T},
    z::Matrix{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    k::Int,
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{T}},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    @inbounds for j in eachindex(out, aq_vec)
        out[j] = _eval_series_with_extrap(y, z, n_pts, x_min, x_max, k, aq_vec[j], extrap, op)
    end
    return out
end

"""
Internal: Evaluate single series at single query point with extrapolation handling.
Takes matrices as arguments for optimal performance.
"""
@inline function _eval_series_with_extrap(
    y::Matrix{T},
    z::Matrix{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    k::Int,
    aq::_CubicAnchoredQuery{T},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_series_anchored(y, z, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap === Val(:extension) || extrap === Val(:wrap)
        return _eval_series_anchored(y, z, k, aq, op)
    elseif extrap === Val(:constant)
        return _constant_extrap_boundary_value(y, aq.side, n_pts, k, op)
    else
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
end

"""
Internal: Core cubic evaluation for series k at anchored query point.
Direct matrix access for optimal performance.

Dispatches on concrete EvalOp for optimal performance:
- EvalValue, EvalDeriv1: Full 4-term evaluation
- EvalDeriv2, EvalDeriv3: Optimized 2-term evaluation (no y-loads)
"""
# EvalValue: Full 4-term evaluation
@inline function _eval_series_anchored(
    y::Matrix{T},
    z::Matrix{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    ::EvalValue
) where {T<:AbstractFloat}
    idx = aq.idx
    wyL, wyR, wzL, wzR = aq.w0
    @inbounds begin
        yL = y[idx, k]
        yR = y[idx + 1, k]
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
end

# EvalDeriv1: Full 4-term evaluation
@inline function _eval_series_anchored(
    y::Matrix{T},
    z::Matrix{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    ::EvalDeriv1
) where {T<:AbstractFloat}
    idx = aq.idx
    wyL, wyR, wzL, wzR = aq.w1
    @inbounds begin
        yL = y[idx, k]
        yR = y[idx + 1, k]
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
end

# EvalDeriv2: Optimized 2-term evaluation (no y-loads)
@inline function _eval_series_anchored(
    y::Matrix{T},
    z::Matrix{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    ::EvalDeriv2
) where {T<:AbstractFloat}
    idx = aq.idx
    wzL, wzR = aq.w2
    @inbounds begin
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wzR, zR, wzL * zL)
end

# EvalDeriv3: Optimized 2-term evaluation (no y-loads)
@inline function _eval_series_anchored(
    y::Matrix{T},
    z::Matrix{T},
    k::Int,
    aq::_CubicAnchoredQuery{T},
    ::EvalDeriv3
) where {T<:AbstractFloat}
    idx = aq.idx
    wzL, wzR = aq.w3
    @inbounds begin
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wzR, zR, wzL * zL)
end
