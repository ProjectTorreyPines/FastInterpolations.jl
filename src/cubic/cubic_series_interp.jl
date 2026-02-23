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
    CubicSeriesInterpolant{Tg, Tv, C, B}

Multi-series cubic spline interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `Tg`: Grid coordinate type (Float32 or Float64) - always real
- `Tv`: Value type (real or Complex{Tg})
- `C`: Cache type (`CubicSplineCache{Tg}`)
- `B`: Boundary condition config type (BCPair or PeriodicData)

# Fields
- `cache::C`: Shared CubicSplineCache with LU factorization
- `bc_for_solve::B`: BC configuration for solving systems
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `z::Matrix{Tv}`: Second derivatives (n_points × n_series) series-contiguous
- `_point_snapshot`: Atomic field for lazy point-contiguous layout
- `extrap::E`: Extrapolation mode (compile-time specialized via type parameter)

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

# Complex values
y_complex = exp.(2im .* π .* x)
sitp_c = cubic_interp(x, [y_complex])  # CubicSeriesInterpolant{Float64, ComplexF64, ...}
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
    Tg<:AbstractFloat,
    Tv,
    C<:CubicSplineCache{Tg},
    B,
    E<:AbstractExtrap,
    P<:AbstractSearchPolicy
} <: AbstractSeriesInterpolant{Tg, Tv}
    const cache::C                    # Shared cache with LU factorization
    const bc_for_solve::B             # BC config for solving
    const y::Matrix{Tv}               # Series-contiguous y (n_points × n_series)
    const z::Matrix{Tv}               # Series-contiguous z (n_points × n_series)
    const _transpose::LazyTransposePair{Tv}  # Lazy point-contiguous layout (shared infra)
    const extrap::E                   # Extrapolation mode (compile-time specialized)
    const search_policy::P            # Default search policy (immutable, thread-safe)

    function CubicSeriesInterpolant(
        cache::C,
        bc_for_solve::B,
        y::Matrix{Tv},
        z::Matrix{Tv},
        extrap::E,
        search::P=Binary()
    ) where {Tg<:AbstractFloat, Tv, C<:CubicSplineCache{Tg}, B, E<:AbstractExtrap, P<:AbstractSearchPolicy}
        new{Tg, Tv, C, B, E, P}(
            cache, bc_for_solve, y, z,
            LazyTransposePair{Tv}(),
            extrap, search
        )
    end
end

# ========================================
# Helper Functions
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(sitp::CubicSeriesInterpolant) = sitp.extrap isa WrapExtrap

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
    _make_anchor(sitp::CubicSeriesInterpolant, xq::Tq) -> _CubicAnchoredQuery{Tg, Tq}

Build anchor for a query point. Required trait for AbstractSeriesInterpolant.

# AD Support
When `xq` is a ForwardDiff.Dual, the returned anchor preserves the Dual type.
"""
@inline function _make_anchor(sitp::CubicSeriesInterpolant{Tg}, xq::Tq, searcher::Searcher=DEFAULT_SEARCHER) where {Tg, Tq<:Real}
    return _anchor_query(sitp.cache.x, xq, Val(:cubic); wrap=_should_wrap(sitp), searcher=searcher)
end

"""
    _eval_series_at_anchor!(output, sitp::CubicSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.

# AD Support
When `aq` has Dual type weights (from Dual query), the output will have promoted type.
"""
@inline function _eval_series_at_anchor!(
    output::AbstractVector,
    sitp::CubicSeriesInterpolant{Tg,Tv},
    aq::_CubicAnchoredQuery{Tg,Tq},
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    y_point, z_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))

    _eval_series_point_with_extrap!(output, y_point, z_point, n_pts, x_min, x_max, aq, sitp.extrap, op)
    return output
end

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::CubicSeriesInterpolant{Tg,Tv}) -> (y_point, z_point)

Ensure point-contiguous layout exists. Delegates to shared LazyTransposePair infrastructure.
"""
@inline function _ensure_point_layout!(sitp::CubicSeriesInterpolant{Tg,Tv}) where {Tg, Tv}
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
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::EvalValue
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::EvalDeriv1
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::EvalDeriv2
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::EvalDeriv3
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_CubicAnchoredQuery{Tg,Tq},
    extrap::AbstractExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_series_point!(out, y_point, z_point, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    _eval_series_point_extrap!(out, y_point, z_point, n_pts, x_min, x_max, aq, extrap, op, aq.side)
end

# NoExtrap - throw DomainError
@inline function _eval_series_point_extrap!(
    ::AbstractVector,
    ::Matrix{Tv},
    ::Matrix{Tv},
    ::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::NoExtrap,
    ::AbstractEvalOp,
    ::UInt8
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# ConstExtrap - clamp to boundary (value only, derivatives are zero)
@inline function _eval_series_point_extrap!(
    out::AbstractVector,
    y_point::Matrix{Tv},
    ::Matrix{Tv},
    n_pts::Int,
    ::Tg,
    ::Tg,
    ::_CubicAnchoredQuery{Tg,Tq},
    ::ConstExtrap,
    op::AbstractEvalOp,
    side::UInt8
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op)
end

# ExtendExtrap - extend polynomial (EvalValue)
@inline function _eval_series_point_extrap!(
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    n_pts::Int,
    ::Tg,
    ::Tg,
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::ExtendExtrap,
    ::EvalValue,
    side::UInt8
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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

# ExtendExtrap - extend polynomial (EvalDeriv1)
@inline function _eval_series_point_extrap!(
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    n_pts::Int,
    ::Tg,
    ::Tg,
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::ExtendExtrap,
    ::EvalDeriv1,
    side::UInt8
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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

# ExtendExtrap - extend polynomial (EvalDeriv2) - optimized, no y-loads
@inline function _eval_series_point_extrap!(
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    n_pts::Int,
    ::Tg,
    ::Tg,
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::ExtendExtrap,
    ::EvalDeriv2,
    side::UInt8
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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

# ExtendExtrap - extend polynomial (EvalDeriv3) - optimized, no y-loads
@inline function _eval_series_point_extrap!(
    out::AbstractVector,
    y_point::Matrix{Tv},
    z_point::Matrix{Tv},
    n_pts::Int,
    ::Tg,
    ::Tg,
    aq::_CubicAnchoredQuery{Tg,Tq},
    ::ExtendExtrap,
    ::EvalDeriv3,
    side::UInt8
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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
    z_mat::Matrix{Tv},
    y_mat::Matrix{Tv},
    cache::CubicSplineCache{Tg},
    bc_for_solve
) where {Tg<:AbstractFloat, Tv}
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
    z_mat::Matrix{Tv},
    y_mat::Matrix{Tv},
    x::AbstractVector{Tg},
    bc_array::AbstractVector{<:BCPair},
    autocache::Bool
) where {Tg<:AbstractFloat, Tv}
    n_series = size(y_mat, 2)

    # Group series by BC type for cache reuse
    # Dict: typeof(bc) => (cache, indices)
    type_groups = Dict{DataType, Tuple{CubicSplineCache{Tg}, Vector{Int}}}()

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
    cubic_interp(x, ys::AbstractVector{<:AbstractVector}; bc=NaturalBC(), extrap=NoExtrap(), autocache=true, precompute_transpose=false)

Create a multi-Y cubic spline interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `bc`: Boundary condition (NaturalBC, ClampedBC, PeriodicBC, or Vector of BC for per-series)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ConstExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
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
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
    extrap::AbstractExtrap=NoExtrap(),
    autocache::Bool=true,
    precompute_transpose::Bool=false,
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
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
    y_mat = Matrix{Tv}(undef, n_pts, n_series_count)
    @inbounds for k in 1:n_series_count
        y_mat[:, k] .= ys[k]
    end

    # Handle periodic BC separately (only for scalar BC)
    if bc isa AbstractBC && _is_periodic_bc(bc)
        return _build_series_periodic(x, y_mat, bc, n_pts, n_series_count, autocache, precompute_transpose, search)
    end

    # Build z matrix by solving systems
    z_mat = Matrix{Tv}(undef, n_pts, n_series_count)

    if bc isa AbstractVector
        # Per-series BC array
        bc_array = _normalize_bc_array(bc, Tg, n_series_count)
        _solve_series_with_bc_array!(z_mat, y_mat, x, bc_array, autocache)
        # All per-series caches share the same x-grid, so any BC's cache is valid here.
        # We store the first BC as a "representative" for struct compatibility and
        # x-grid access via _get_grid(cache); the specific BC chosen does not affect evaluation.
        bc_representative = bc_array[1]
        cache = _get_cubic_cache(x, bc_representative, autocache)
    else
        # Uniform BC (original path)
        bc_pair = _normalize_bc(bc, Tg)
        cache = _get_cubic_cache(x, bc_pair, autocache)
        _solve_series_coefficients!(z_mat, y_mat, cache, bc_pair)
        bc_representative = bc_pair
    end

    sitp = CubicSeriesInterpolant(cache, bc_representative, y_mat, z_mat, extrap, search)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

"""
Internal helper for periodic BC multi-interpolant construction.
"""
function _build_series_periodic(
    x::AbstractVector{Tg},
    y_mat::Matrix{Tv},
    bc::PeriodicBC,
    n_pts::Int,
    n_series_count::Int,
    autocache::Bool,
    precompute_transpose::Bool,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    # Extend data for exclusive endpoint
    x, y_mat = _prepare_periodic(x, y_mat, bc)
    n_pts = size(y_mat, 1)

    # Validate periodic endpoints for all series
    atol = Tg === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64
    @inbounds for k in 1:n_series_count
        y_first = y_mat[1, k]
        y_last = y_mat[n_pts, k]
        if !isapprox(y_first, y_last; atol=atol)
            throw(ArgumentError(
                "PeriodicBC (inclusive endpoint) requires y[1] ≈ y[end] for series $k, " *
                "got y[1]=$y_first, y[end]=$y_last (diff=$(abs(y_last-y_first))). " *
                "If your data does not repeat the first point, use " *
                "PeriodicBC(endpoint=:exclusive) instead."
            ))
        end
    end

    # Get periodic cache
    cache = _get_cubic_cache(x, PeriodicBC(), autocache)

    # Build z matrix
    z_mat = Matrix{Tv}(undef, n_pts, n_series_count)
    _solve_series_coefficients!(z_mat, y_mat, cache, cache.bc_config)

    # Periodic BC always uses wrap extrapolation
    sitp = CubicSeriesInterpolant(cache, cache.bc_config, y_mat, z_mat, WrapExtrap(), search)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

# Matrix input: columns as y-series
"""
    cubic_interp(x, Y::AbstractMatrix; bc=NaturalBC(), extrap=NoExtrap(), autocache=true, precompute_transpose=false)

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
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
    extrap::AbstractExtrap=NoExtrap(),
    autocache::Bool=true,
    precompute_transpose::Bool=false,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
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
        return _build_series_periodic(x, y_mat, bc, n_pts, n_series_count, autocache, precompute_transpose, search)
    end

    # Build z matrix by solving systems
    z_mat = Matrix{Tv}(undef, n_pts, n_series_count)

    if bc isa AbstractVector
        # Per-series BC array
        bc_array = _normalize_bc_array(bc, Tg, n_series_count)
        _solve_series_with_bc_array!(z_mat, y_mat, x, bc_array, autocache)
        # All per-series caches share the same x-grid, so any BC's cache is valid here.
        # We store the first BC as a "representative" for struct compatibility and
        # x-grid access via _get_grid(cache); the specific BC chosen does not affect evaluation.
        bc_representative = bc_array[1]
        cache = _get_cubic_cache(x, bc_representative, autocache)
    else
        # Uniform BC (original path)
        bc_pair = _normalize_bc(bc, Tg)
        cache = _get_cubic_cache(x, bc_pair, autocache)
        _solve_series_coefficients!(z_mat, y_mat, cache, bc_pair)
        bc_representative = bc_pair
    end

    sitp = CubicSeriesInterpolant(cache, bc_representative, y_mat, z_mat, extrap, search)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

# Type promotion wrappers (auto-promote to Float, handle Complex)
function cubic_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
    extrap::AbstractExtrap=NoExtrap(),
    autocache::Bool=true,
    precompute_transpose::Bool=false,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv}
    # Compute promoted grid type (Tg may be Int, promotes to Float)
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    Tv_float = _value_type(Tv, Tg_float)
    x_typed = _to_float(x, Tg_float)
    ys_typed = [Tv_float.(y) for y in ys]
    bc_typed = _promote_bc(bc, Tg_float)
    return cubic_interp(x_typed, ys_typed; bc=bc_typed, extrap=extrap, autocache=autocache, precompute_transpose=precompute_transpose, search=search)
end

function cubic_interp(
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    bc::Union{AbstractBC, AbstractVector{<:AbstractBC}}=NaturalBC(),
    extrap::AbstractExtrap=NoExtrap(),
    autocache::Bool=true,
    precompute_transpose::Bool=false,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv}
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    Tv_float = _value_type(Tv, Tg_float)
    x_typed = _to_float(x, Tg_float)
    Y_typed = Tv_float.(Y)
    bc_typed = _promote_bc(bc, Tg_float)
    return cubic_interp(x_typed, Y_typed; bc=bc_typed, extrap=extrap, autocache=autocache, precompute_transpose=precompute_transpose, search=search)
end

# ========================================
# Scalar Evaluation
# ========================================

"""
    (sitp::CubicSeriesInterpolant)(xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.

# AD Support
When `xq` is a ForwardDiff.Dual, the output type is promoted to preserve
derivatives. Output type is `promote_type(Tv, Tq)`.
"""
function (sitp::CubicSeriesInterpolant{Tg,Tv})(
    xq::Tq;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual (no-op for Float/Float-backed Dual)
    xq_promoted = _promote_for_anchor(xq, Tg)
    T_out = promote_type(Tv, typeof(xq_promoted))
    output = Vector{T_out}(undef, n_series(sitp))

    # Build anchor preserving Dual type in xq
    aq = _make_anchor(sitp, xq_promoted, _to_searcher(search, hint))

    # Dispatch on derivative order
    @_dispatch_deriv deriv => op begin
        _eval_series_at_anchor!(output, sitp, aq, op)
    end
    return output
end

"""
    (sitp::CubicSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (in-place).

Note: For AD support with ForwardDiff.Dual, use the out-of-place version
which automatically promotes the output type.
"""
function (sitp::CubicSeriesInterpolant{Tg,Tv})(
    output::AbstractVector,  # Relaxed: accepts any element type for lossless promotion
    xq::Tq;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    _validate_scalar_output(output, n_series(sitp))

    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual
    xq_promoted = _promote_for_anchor(xq, Tg)

    # Build anchor preserving Dual type in xq (for AD)
    aq = _make_anchor(sitp, xq_promoted, _to_searcher(search, hint))

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

# Precision Preservation
For mixed-type queries (e.g., Float64 queries on Float32 grid), output type is
`promote_type(Tv, Tq)` to preserve precision and match scalar/broadcast semantics.
"""
function (sitp::CubicSeriesInterpolant{Tg,Tv})(
    xq::AbstractVector{Tq};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    n_query = length(xq)
    n_ser = n_series(sitp)
    T_out = promote_type(Tv, Tq)  # Lossless: wider type to avoid precision loss

    # Explicit Vector{Vector{T_out}} for type stability on Julia LTS
    outputs = Vector{Vector{T_out}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{T_out}(undef, n_query)
    end
    # Delegate to in-place (handles precision preservation via anchor building)
    sitp(outputs, xq; deriv=deriv, search=search, hint=hint)

    return outputs
end

"""
    (sitp::CubicSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector{Tq}; deriv=0, search=sitp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tq<:Real}

Evaluate multi-Y interpolant at multiple query points (in-place).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points (any Real type)
- `deriv`: Derivative order (0, 1, 2, or 3)

# Zero Allocation (Hot Path)
When `eltype(xq) === Tg`, uses task-local pool for anchors → zero allocation after warmup.
When `eltype(xq) !== Tg`, allocates anchor vector with precision-preserving weights.

# Precision Preservation
Builds anchors from original `xq` (preserving precision in weights) for scalar/vector symmetry.
"""
@with_pool pool function (sitp::CubicSeriesInterpolant{Tg,Tv})(
    outputs::AbstractVector{<:AbstractVector},
    xq::AbstractVector{Tq};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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

    # Build anchors - pool handles both Tq===Tg and mixed-type cases
    # Each unique type combination gets its own pool slot
    aq_vec = acquire!(pool, _CubicAnchoredQuery{Tg,Tq}, n_query)
    _fill_anchors!(aq_vec, sitp.cache.x, xq, Val(:cubic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    # Extract matrices for argument-passing pattern (series-contiguous layout)
    # This is faster than point-contiguous for vector queries because:
    # - outputs[k] is contiguous, writes are sequential
    # - y[:, k] is contiguous, reads are sequential
    # - No temp buffer or scatter loop needed
    y, z = sitp.y, sitp.z
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))

    # Evaluate all series using series-contiguous layout
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_series_vector!(outputs[k], y, z, n_pts, x_min, x_max, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

"""
    (sitp::CubicSeriesInterpolant)(outputs, aq_vec::AbstractVector{<:_CubicAnchoredQuery{Tg,Tq}}; deriv=0) where {Tg<:AbstractFloat, Tq<:Real}

Evaluate multi-Y interpolant with pre-built anchors (TRUE zero-allocation).

For maximum performance in hot loops, pre-build anchors once and reuse:
```julia
x = ...
sitp = cubic_interp(x, [y1, y2, y3])
xq = [0.1, 0.2, 0.3, ...]

# Pre-build anchors (allocates once)
aq_vec = FastInterpolations._anchor_query(x, xq, Val(:cubic))

# Zero-allocation loop
outputs = [similar(xq) for _ in 1:3]
for _ in 1:1000
    sitp(outputs, aq_vec)  # Zero allocation!
end
```
"""
function (sitp::CubicSeriesInterpolant{Tg,Tv})(
    outputs::AbstractVector{<:AbstractVector{Tv}},
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{Tg}};
    deriv::Int=0
) where {Tg<:AbstractFloat, Tv}
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
    x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))

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
    out::AbstractVector{Tv},
    y::Matrix{Tv},
    z::Matrix{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    k::Int,
    aq_vec::AbstractVector{<:_CubicAnchoredQuery{Tg}},
    extrap::AbstractExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
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
    y::Matrix{Tv},
    z::Matrix{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    k::Int,
    aq::_CubicAnchoredQuery{Tg},
    extrap::AbstractExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_series_anchored(y, z, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap isa ExtendExtrap || extrap isa WrapExtrap
        return _eval_series_anchored(y, z, k, aq, op)
    elseif extrap isa ConstExtrap
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
    y::Matrix{Tv},
    z::Matrix{Tv},
    k::Int,
    aq::_CubicAnchoredQuery{Tg},
    ::EvalValue
) where {Tg<:AbstractFloat, Tv}
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
    y::Matrix{Tv},
    z::Matrix{Tv},
    k::Int,
    aq::_CubicAnchoredQuery{Tg},
    ::EvalDeriv1
) where {Tg<:AbstractFloat, Tv}
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
    y::Matrix{Tv},
    z::Matrix{Tv},
    k::Int,
    aq::_CubicAnchoredQuery{Tg},
    ::EvalDeriv2
) where {Tg<:AbstractFloat, Tv}
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
    y::Matrix{Tv},
    z::Matrix{Tv},
    k::Int,
    aq::_CubicAnchoredQuery{Tg},
    ::EvalDeriv3
) where {Tg<:AbstractFloat, Tv}
    idx = aq.idx
    wzL, wzR = aq.w3
    @inbounds begin
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wzR, zR, wzL * zL)
end
