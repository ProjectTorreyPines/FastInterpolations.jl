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
- `Tv`: Value type (unconstrained)
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
        Tg,
        Tv,
        C <: CubicSplineCache{Tg},
        B,
        E <: AbstractExtrap,
        P <: AbstractSearchPolicy,
        Tz,
    } <: AbstractSeriesInterpolant{Tg, Tv}
    const cache::C                    # Shared cache with LU factorization
    const bc_for_solve::B             # BC config for solving
    const y::Matrix{Tv}               # Series-contiguous y (n_points × n_series)
    const z::Matrix{Tz}               # Series-contiguous z: Tz = _output_eltype(Tv, Tg)
    const _transpose::LazyTransposePair{Tv, Tz}  # Lazy point-contiguous layout
    const extrap::E                   # Extrapolation mode (compile-time specialized)
    const search_policy::P            # Default search policy (immutable, thread-safe)

    function CubicSeriesInterpolant(
            cache::C,
            bc_for_solve::B,
            y::Matrix{Tv},
            z::Matrix,
            extrap::E,
            search::P = AutoSearch()
        ) where {Tg, Tv, C <: CubicSplineCache{Tg}, B, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        Tz = eltype(z)
        # y/z are NOT copied here — factory function provides owned matrices.
        return new{Tg, Tv, C, B, E, P, Tz}(
            cache, bc_for_solve, y, z,
            LazyTransposePair{Tv, Tz}(),
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
@inline function _make_anchor(sitp::CubicSeriesInterpolant{Tg}, xq::Tq, searcher::P = DEFAULT_SEARCHER) where {Tg, Tq <: Real, P <: Searcher}
    return _anchor_query(sitp.cache.x, xq, Val(:cubic), _should_wrap(sitp), searcher)
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
        sitp::CubicSeriesInterpolant{Tg, Tv},
        aq::_CubicAnchoredQuery{Tg, Tq},
        op::AbstractEvalOp
    ) where {Tg, Tv, Tq <: Real}
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
@inline function _ensure_point_layout!(sitp::CubicSeriesInterpolant{Tg, Tv}) where {Tg, Tv}
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
        z_point::Matrix{Tz},
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::EvalValue
    ) where {Tg, Tv, Tz, Tq <: Real}
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
        z_point::Matrix{Tz},
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::EvalDeriv1
    ) where {Tg, Tv, Tz, Tq <: Real}
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
        z_point::Matrix{Tz},
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::EvalDeriv2
    ) where {Tg, Tv, Tz, Tq <: Real}
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
        z_point::Matrix{Tz},
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::EvalDeriv3
    ) where {Tg, Tv, Tz, Tq <: Real}
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

@inline function _eval_series_point!(
        out::AbstractVector,
        y_point::Matrix{Tv},
        z_point::Matrix{Tz},
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::DerivOp{N}
    ) where {Tg, Tv, Tz, Tq <: Real, N}
    z = 0 * (@inbounds y_point[1, aq.idx])
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
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
        z_point::Matrix{Tz},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_CubicAnchoredQuery{Tg, Tq},
        extrap::AbstractExtrap,
        op::AbstractEvalOp
    ) where {Tg, Tv, Tz, Tq <: Real}
    # Inside domain: normal evaluation
    if aq.state == IN_DOMAIN
        return _eval_series_point!(out, y_point, z_point, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    return _eval_series_point_extrap!(out, y_point, z_point, n_pts, x_min, x_max, aq, extrap, op, aq.state)
end

# NoExtrap - throw DomainError
@inline function _eval_series_point_extrap!(
        ::AbstractVector,
        ::Matrix{Tv},
        ::Matrix{Tv},
        ::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::NoExtrap,
        ::AbstractEvalOp,
        ::UInt8
    ) where {Tg, Tv, Tq <: Real}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# ClampExtrap - clamp to boundary (value only, derivatives are zero)
@inline function _eval_series_point_extrap!(
        out::AbstractVector,
        y_point::Matrix{Tv},
        ::Matrix{Tv},
        n_pts::Int,
        ::Tg,
        ::Tg,
        ::_CubicAnchoredQuery{Tg, Tq},
        extrap::_ClampOrFill,
        op::AbstractEvalOp,
        side::UInt8
    ) where {Tg, Tv, Tq <: Real}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op, extrap)
end

# ExtendExtrap - extend polynomial (EvalValue)
@inline function _eval_series_point_extrap!(
        out::AbstractVector,
        y_point::Matrix{Tv},
        z_point::Matrix{Tz},
        n_pts::Int,
        ::Tg,
        ::Tg,
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::ExtendExtrap,
        ::EvalValue,
        side::UInt8
    ) where {Tg, Tv, Tz, Tq <: Real}
    idx = side == OOB_LEFT ? 1 : (n_pts - 1)
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
        z_point::Matrix{Tz},
        n_pts::Int,
        ::Tg,
        ::Tg,
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::ExtendExtrap,
        ::EvalDeriv1,
        side::UInt8
    ) where {Tg, Tv, Tz, Tq <: Real}
    idx = side == OOB_LEFT ? 1 : (n_pts - 1)
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
        z_point::Matrix{Tz},
        n_pts::Int,
        ::Tg,
        ::Tg,
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::ExtendExtrap,
        ::EvalDeriv2,
        side::UInt8
    ) where {Tg, Tv, Tz, Tq <: Real}
    idx = side == OOB_LEFT ? 1 : (n_pts - 1)
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
        z_point::Matrix{Tz},
        n_pts::Int,
        ::Tg,
        ::Tg,
        aq::_CubicAnchoredQuery{Tg, Tq},
        ::ExtendExtrap,
        ::EvalDeriv3,
        side::UInt8
    ) where {Tg, Tv, Tz, Tq <: Real}
    idx = side == OOB_LEFT ? 1 : (n_pts - 1)
    idx1 = idx + 1
    wzL, wzR = aq.w3

    @inbounds @simd for k in axes(out, 1)
        zL = z_point[k, idx]
        zR = z_point[k, idx1]
        out[k] = muladd(wzR, zR, wzL * zL)
    end
    return out
end

# ExtendExtrap - 4th+ derivative of cubic is zero → fill zeros
@inline function _eval_series_point_extrap!(
        out::AbstractVector,
        y_point::Matrix{Tv},
        ::Matrix{Tv},
        ::Int,
        ::Tg,
        ::Tg,
        ::_CubicAnchoredQuery{Tg, Tq},
        ::ExtendExtrap,
        ::DerivOp{N},
        ::UInt8
    ) where {Tg, Tv, Tq <: Real, N}
    z = 0 * first(y_point)
    @inbounds @simd for k in axes(out, 1)
        out[k] = z
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
        z_mat::Matrix{Tz},
        y_mat::Matrix{Tv},
        cache::CubicSplineCache{Tg},
        bc_for_solve
    ) where {Tz, Tv, Tg}
    n_series_count = size(y_mat, 2)

    # Solve each series column
    @inbounds for k in 1:n_series_count
        _solve_system!(@view(z_mat[:, k]), cache, @view(y_mat[:, k]), bc_for_solve)
    end

    return z_mat
end

"""
    _solve_series_with_bc_array!(z_mat, y_mat, x, bc_cache_array, bc_solve_array, autocache)

Solve cubic spline systems for series with per-series boundary conditions.
Groups series by BC type for cache efficiency.

# Arguments
- `z_mat`: Output matrix for second derivatives (n_points × n_series)
- `y_mat`: Input y-values matrix (n_points × n_series)
- `x`: x-grid vector
- `bc_cache_array`: Vector of BCPair (Tg-typed), used for cache matrix construction
- `bc_solve_array`: Vector of BCPair (Tv-typed), used for RHS computation
- `autocache`: Whether to use cache pool
"""
@with_pool pool function _solve_series_with_bc_array!(
        z_mat::Matrix{Tz},
        y_mat::Matrix{Tv},
        x::AbstractVector{Tg},
        bc_cache_array::AbstractVector{<:BCPair},
        bc_solve_array::AbstractVector{<:BCPair},
        autocache::Bool
    ) where {Tz, Tv, Tg}
    n_series = size(y_mat, 2)

    # Group series by BC type for cache reuse (using Tg-typed BCs for matrix structure)
    # Dict: typeof(bc) => (cache, indices)
    type_groups = Dict{DataType, Tuple{CubicSplineCache{Tg}, Vector{Int}}}()

    for k in 1:n_series
        bc = bc_cache_array[k]
        bc_type = typeof(bc)

        if haskey(type_groups, bc_type)
            # Cache already exists for this BC type → just add index
            push!(type_groups[bc_type][2], k)
        else
            # New BC type → get cache from pool
            cache = _get_cubic_cache(x, bc, _effective_autocache(autocache, eltype(x)))
            type_groups[bc_type] = (cache, [k])
        end
    end

    # Solve each group using its cache (using Tv-typed BCs for RHS computation)
    for (_, (cache, indices)) in type_groups
        for k in indices
            _solve_system!(@view(z_mat[:, k]), cache, @view(y_mat[:, k]), bc_solve_array[k])
        end
    end

    return z_mat
end

# ========================================
# Series Constructor (canonical entry point)
# ========================================

"""
    cubic_interp(x, Series(y1, y2, ...); bc=CubicFit(), extrap=NoExtrap(), autocache=true, precompute_transpose=false)
    cubic_interp(x, Series([y1, y2, ...]); ...)
    cubic_interp(x, Series(Y::AbstractMatrix); ...)

Create a multi-Y cubic spline interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `s::Series`: Wrapped series data (varargs, vector-of-vectors, or matrix)
- `bc`: Boundary condition (CubicFit, ZeroCurvBC, ZeroSlopeBC, PeriodicBC, or Vector of BC for per-series)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `autocache`: If true, reuse cached LU factorization (default: true)
- `precompute_transpose`: If true, build point-contiguous layout immediately
- `search::AbstractSearchPolicy`: Search policy for interval lookup

# Returns
`CubicSeriesInterpolant` object with matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

sitp = cubic_interp(x, Series(y1, y2, y3))
vals = sitp(0.5)  # [sin(π), cos(π), exp(-0.5)]

# Per-series BC (each series can have different BC)
sitp = cubic_interp(x, Series(y1, y2, y3); bc=[
    ZeroCurvBC(),
    BCPair(Deriv1(2.0), Deriv1(0.0)),
    BCPair(Deriv2(0.0), Deriv3(5.0)),
])

# Matrix form
Y = hcat(y1, y2)
sitp = cubic_interp(x, Series(Y))
```
"""
function cubic_interp(
        x::AbstractVector{Tg},
        s::Series;
        bc::Union{AbstractBC, AbstractVector{<:AbstractBC}} = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        precompute_transpose::Bool = false,
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    # Type promotion: widen grid if y's float base is wider than Tg
    Tv = _series_eltype(s)
    Tg_new = _promote_grid_float(Tg, Tv)
    if Tg_new !== Tg
        return cubic_interp(
            _to_float(x, Tg_new), s;
            bc = _promote_bc(bc, Tg_new), extrap, autocache, precompute_transpose, search
        )
    end

    # Normalize early: Range → _CachedRange, Vector → identity.
    # All downstream code sees only _CachedRange{Tg} or Vector{Tg}.
    x = _to_float(x, Tg)

    n_pts = length(x)
    Tv_out = _value_type(Tv, Tg)
    y_mat, n_ser = _build_series_mat(s, n_pts, Tv_out)

    # Handle periodic BC separately (only for scalar BC)
    if bc isa AbstractBC && _is_periodic_bc(bc)
        return _build_series_periodic(x, y_mat, bc, n_pts, n_ser, autocache, precompute_transpose, search)
    end

    # Build z matrix by solving tridiagonal systems
    # z coefficients mix y (Tv_out) with grid spacing (Tg) → Dual when grid is Dual
    Tz = _output_eltype(Tv_out, Tg)
    z_mat = Matrix{Tz}(undef, n_pts, n_ser)

    if bc isa AbstractVector
        # Per-series BC array: Tg-typed for cache matrix, Tv-typed for RHS
        bc_cache_array = _normalize_bc_array(bc, Tg, n_ser)
        bc_solve_array = _normalize_bc_array(bc, Tv_out, n_ser)
        _solve_series_with_bc_array!(z_mat, y_mat, x, bc_cache_array, bc_solve_array, autocache)
        bc_representative = bc_cache_array[1]
        cache = _get_cubic_cache(x, bc_representative, _effective_autocache(autocache, eltype(x)))
    else
        # Uniform BC: Tg-typed for cache matrix, Tv-typed for RHS
        bc_for_cache = _normalize_bc(bc, Tg)
        bc_for_solve = _normalize_bc(bc, first(y_mat))
        cache = _get_cubic_cache(x, bc_for_cache, _effective_autocache(autocache, eltype(x)))
        _solve_series_coefficients!(z_mat, y_mat, cache, bc_for_solve)
        bc_representative = bc_for_cache
    end

    extrap_p = _promote_extrap(extrap, eltype(y_mat))
    sitp = CubicSeriesInterpolant(cache, bc_representative, y_mat, z_mat, extrap_p, search)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

# Real grid promotion (Int, etc.) → convert to float and delegate

# Note: Real wrapper (Tg <: Real) removed — typed method above handles
# all grid types including ForwardDiff.Dual via _to_float + _promote_grid_float.

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
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    # Extend data for exclusive endpoint
    x, y_mat = _prepare_periodic(x, y_mat, bc)
    n_pts = size(y_mat, 1)

    # Validate periodic endpoints for all series (strict == equality)
    @inbounds for k in 1:n_series_count
        y_first = y_mat[1, k]
        y_last = y_mat[n_pts, k]
        y_first == y_last || _throw_periodic_series_error(k, y_first, y_last)
    end

    # Get periodic cache
    cache = _get_cubic_cache(x, PeriodicBC(), _effective_autocache(autocache, eltype(x)))

    # Build z matrix (Dual when grid is Dual)
    Tz = _output_eltype(Tv, eltype(cache.x))
    z_mat = Matrix{Tz}(undef, n_pts, n_series_count)
    _solve_series_coefficients!(z_mat, y_mat, cache, cache.bc_config)

    # Periodic BC always uses wrap extrapolation
    sitp = CubicSeriesInterpolant(cache, cache.bc_config, y_mat, z_mat, WrapExtrap(), search)

    if precompute_transpose
        _ensure_point_layout!(sitp)
    end

    return sitp
end

# ========================================
# Scalar Evaluation
# ========================================

"""
    (sitp::CubicSeriesInterpolant)(xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.

# AD Support
When `xq` is a ForwardDiff.Dual, the output type is promoted to preserve
derivatives. Output type is `promote_type(Tv, Tq)`.
"""
function (sitp::CubicSeriesInterpolant{Tg, Tv})(
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual (no-op for Float/Float-backed Dual)
    xq_promoted = _promote_for_anchor(xq, Tg)
    T_out = _series_output_type(_output_eltype(Tv, Tg), typeof(xq_promoted))
    output = Vector{T_out}(undef, n_series(sitp))

    # Build anchor preserving Dual type in xq
    aq = _make_anchor(sitp, xq_promoted, _resolve_search(sitp.cache.x, xq, search, hint))

    _eval_series_at_anchor!(output, sitp, aq, deriv)
    return output
end

"""
    (sitp::CubicSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (in-place).

Note: For AD support with ForwardDiff.Dual, use the out-of-place version
which automatically promotes the output type.
"""
function (sitp::CubicSeriesInterpolant{Tg, Tv})(
        output::AbstractVector,  # Relaxed: accepts any element type for lossless promotion
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    _validate_scalar_output(output, n_series(sitp))

    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual
    xq_promoted = _promote_for_anchor(xq, Tg)

    # Build anchor preserving Dual type in xq (for AD)
    aq = _make_anchor(sitp, xq_promoted, _resolve_search(sitp.cache.x, xq, search, hint))

    _eval_series_at_anchor!(output, sitp, aq, deriv)
    return output
end

# ========================================
# Vector Evaluation
# ========================================

"""
    (sitp::CubicSeriesInterpolant)(xq::AbstractVector; deriv=EvalValue())

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.

# Precision Preservation
For mixed-type queries (e.g., Float64 queries on Float32 grid), output type is
`promote_type(Tv, Tq)` to preserve precision and match scalar/broadcast semantics.
"""
function (sitp::CubicSeriesInterpolant{Tg, Tv})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    n_query = length(xq)
    n_ser = n_series(sitp)
    T_out = _series_output_type(_output_eltype(Tv, Tg), Tq)

    # Explicit Vector{Vector{T_out}} for type stability on Julia LTS
    outputs = Vector{Vector{T_out}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{T_out}(undef, n_query)
    end
    # Delegate to in-place (handles precision preservation via anchor building)
    sitp(outputs, xq; deriv = deriv, search = search, hint = hint)

    return outputs
end

"""
    (sitp::CubicSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector{Tq}; deriv=EvalValue(), search=sitp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tq<:Real}

Evaluate multi-Y interpolant at multiple query points (in-place).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points (any Real type)
- `deriv`: Derivative order (`EvalValue()`, `DerivOp(1)`, `DerivOp(2)`, or `DerivOp(3)`)

# Zero Allocation (Hot Path)
When `eltype(xq) === Tg`, uses task-local pool for anchors → zero allocation after warmup.
When `eltype(xq) !== Tg`, allocates anchor vector with precision-preserving weights.

# Precision Preservation
Builds anchors from original `xq` (preserving precision in weights) for scalar/vector symmetry.
"""
@with_pool pool function (sitp::CubicSeriesInterpolant{Tg, Tv})(
        outputs::AbstractVector{<:AbstractVector},
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    n_query = length(xq)
    n_ser = n_series(sitp)

    # Validate dimensions
    if length(outputs) != n_ser
        throw(
            DimensionMismatch(
                "outputs length $(length(outputs)) must match n_series $n_ser"
            )
        )
    end
    for (k, out_k) in enumerate(outputs)
        if length(out_k) != n_query
            throw(
                DimensionMismatch(
                    "outputs[$k] length $(length(out_k)) must match n_query $n_query"
                )
            )
        end
    end

    # Build anchors — Tq widens via promote_type (Float32 on Float64 grid → Float64)
    Tq_w = promote_type(Tq, Tg)
    aq_vec = acquire!(pool, _CubicAnchoredQuery{Tg, Tq_w}, n_query)
    searcher = _resolve_search(sitp.cache.x, xq, search, hint)
    _fill_anchors!(aq_vec, sitp.cache.x, xq, Val(:cubic), _should_wrap(sitp), searcher)

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
    @inbounds for k in 1:n
        _eval_series_vector!(outputs[k], y, z, n_pts, x_min, x_max, k, aq_vec, extrap, deriv)
    end
    return outputs
end

"""
    (sitp::CubicSeriesInterpolant)(outputs, aq_vec::AbstractVector{<:_CubicAnchoredQuery{Tg,Tq}}; deriv=EvalValue()) where {Tg, Tq<:Real}

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
function (sitp::CubicSeriesInterpolant{Tg, Tv})(
        outputs::AbstractVector{<:AbstractVector{Tv}},
        aq_vec::AbstractVector{<:_CubicAnchoredQuery{Tg}};
        deriv::DerivOp = EvalValue()
    ) where {Tg, Tv}
    n_query = length(aq_vec)
    n_ser = n_series(sitp)

    # Validate dimensions
    if length(outputs) != n_ser
        throw(
            DimensionMismatch(
                "outputs length $(length(outputs)) must match n_series $n_ser"
            )
        )
    end
    for (k, out_k) in enumerate(outputs)
        if length(out_k) != n_query
            throw(
                DimensionMismatch(
                    "outputs[$k] length $(length(out_k)) must match n_query $n_query"
                )
            )
        end
    end

    # Extract matrices for argument-passing pattern
    y, z = sitp.y, sitp.z
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    x_min, x_max = Tg(first(sitp.cache.x)), Tg(last(sitp.cache.x))

    # Evaluate all series
    @inbounds for k in 1:n
        _eval_series_vector!(outputs[k], y, z, n_pts, x_min, x_max, k, aq_vec, extrap, deriv)
    end
    return outputs
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance (avoids struct field access in loop).
"""
@inline function _eval_series_vector!(
        out::AbstractVector,
        y::Matrix{Tv},
        z::Matrix{Tz},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        k::Int,
        aq_vec::AbstractVector{<:_CubicAnchoredQuery{Tg}},
        extrap::AbstractExtrap,
        op::AbstractEvalOp
    ) where {Tg, Tv, Tz}
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
        z::Matrix{Tz},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        k::Int,
        aq::_CubicAnchoredQuery{Tg},
        extrap::AbstractExtrap,
        op::AbstractEvalOp
    ) where {Tg, Tv, Tz}
    # Inside domain: normal evaluation
    if aq.state == IN_DOMAIN
        return _eval_series_anchored(y, z, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap isa ExtendExtrap || extrap isa WrapExtrap
        return _eval_series_anchored(y, z, k, aq, op)
    elseif extrap isa _ClampOrFill
        return _constant_extrap_boundary_value(y, aq.state, n_pts, k, op, extrap)
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
        z::Matrix{Tz},
        k::Int,
        aq::_CubicAnchoredQuery{Tg},
        ::EvalValue
    ) where {Tg, Tv, Tz}
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
        z::Matrix{Tz},
        k::Int,
        aq::_CubicAnchoredQuery{Tg},
        ::EvalDeriv1
    ) where {Tg, Tv, Tz}
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
        z::Matrix{Tz},
        k::Int,
        aq::_CubicAnchoredQuery{Tg},
        ::EvalDeriv2
    ) where {Tg, Tv, Tz}
    idx = aq.idx
    wzL, wzR = aq.w2
    @inbounds begin
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wzR, zR, wzL * zL)
end

@inline function _eval_series_anchored(
        y::Matrix{Tv},
        z::Matrix{Tz},
        k::Int,
        aq::_CubicAnchoredQuery{Tg},
        ::DerivOp{N}
    ) where {Tg, Tv, Tz, N}
    @inbounds yL = y[aq.idx, k]
    return 0 * yL
end

# EvalDeriv3: Optimized 2-term evaluation (no y-loads)
@inline function _eval_series_anchored(
        y::Matrix{Tv},
        z::Matrix{Tz},
        k::Int,
        aq::_CubicAnchoredQuery{Tg},
        ::EvalDeriv3
    ) where {Tg, Tv, Tz}
    idx = aq.idx
    wzL, wzR = aq.w3
    @inbounds begin
        zL = z[idx, k]
        zR = z[idx + 1, k]
    end
    return muladd(wzR, zR, wzL * zL)
end
