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
    const z::Matrix{Tz}               # Series-contiguous z: Tz = _promote_eltype(_coeff_op, Tg, Tv)
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
    Tz = _promote_eltype(_coeff_op, Tg, Tv_out)
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
        bc_for_cache = _normalize_bc(bc)
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
    Tz = _promote_eltype(_coeff_op, eltype(cache.x), Tv)
    z_mat = Matrix{Tz}(undef, n_pts, n_series_count)
    _solve_series_coefficients!(z_mat, y_mat, cache, cache.bc)

    # Periodic BC always uses wrap extrapolation.
    sitp = CubicSeriesInterpolant(cache, cache.bc, y_mat, z_mat, WrapExtrap(), search)

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
    xq_promoted = _promote_coord(xq, Tg)
    T_out = _promote_eltype(_interp_op, Tg, Tv, typeof(xq_promoted))
    output = Vector{T_out}(undef, n_series(sitp))
    # Delegate to in-place (holds the shared lean anchor build + point-layout eval)
    sitp(output, xq; deriv = deriv, search = search, hint = hint)
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

    # One lean op/extrap-aware anchor, built from the statically-typed `sitp.extrap`
    # (same helpers as the batch path). NoExtrap throws OOB inside the build.
    A = _cubic_series_anchor_type(deriv, sitp.extrap, sitp.cache.x, _coord_eltype(Tq, Tg))
    searcher = _resolve_search(sitp.cache.x, xq, search, hint)
    a = @_narrow_searcher searcher _build_series_anchor(
        CubicInterp(), A, sitp.cache.x, xq, sitp.extrap, _should_wrap(sitp), searcher
    )

    # Point-contiguous layout: `out[k]` streams across the K series (SIMD).
    y_point, z_point = _ensure_point_layout!(sitp)
    _cubic_series_eval!(output, y_point, z_point, a, sitp.extrap)
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
    T_out = _promote_eltype(_interp_op, Tg, Tv, Tq)

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

    # Build lean op/extrap-aware anchors — Tq widens via _coord_eltype
    # (Float32 on Float64 grid → Float64). Payload carries only this op's
    # weights; Clamp/Fill select the stateful wrapper. NoExtrap throws during
    # this build (before any output is written) with a mixed-precision-safe
    # DomainError.
    Tq_w = _coord_eltype(Tq, Tg)
    A = _cubic_series_anchor_type(deriv, sitp.extrap, sitp.cache.x, Tq_w)
    anchors = acquire!(pool, A, n_query)
    searcher = _resolve_search(sitp.cache.x, xq, search, hint)
    @_narrow_searcher searcher _fill_series_anchors!(
        CubicInterp(), anchors, sitp.cache.x, xq, sitp.extrap, _should_wrap(sitp), searcher
    )

    # Extract matrices for argument-passing pattern (series-contiguous layout)
    # This is faster than point-contiguous for vector queries because:
    # - outputs[k] is contiguous, writes are sequential
    # - y[:, k] is contiguous, reads are sequential
    # - No temp buffer or scatter loop needed
    y, z = sitp.y, sitp.z
    n = n_series(sitp)
    extrap = sitp.extrap

    # Evaluate all series using series-contiguous layout
    @inbounds for k in 1:n
        out_k = outputs[k]
        for j in eachindex(out_k, anchors)
            out_k[j] = _cubic_series_eval(y, z, k, anchors[j], extrap)
        end
    end
    return outputs
end
