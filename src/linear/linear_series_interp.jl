# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    LINEAR SERIES INTERPOLATION                            ║
# ║         Multiple y-data series sharing the same x-grid                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase E.1: Unified matrix storage for optimal performance.
# Key optimization: Adaptive layout with lazy transpose for scalar queries.
#
# Include order: ... → linear_anchor.jl → linear_series_interp.jl
#

# ========================================
# Type Definition
# ========================================

"""
    LinearSeriesInterpolant{Tg, Tv, E, P, X}

Multi-series linear interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `Tg`: Grid type (Float32 or Float64)
- `Tv`: Value type (unconstrained)
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type
- `X<:AbstractVector{Tg}`: Grid container type — `_CachedRange{Tg}` for Range input,
  `_CachedVector{Tg,Tinv}` for Vector input (carries cached `h`/`inv_h`).

# Fields
- `x::X`: Shared x-grid (wrapped — `_CachedVector`/`_CachedRange`)
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `_transpose::LazyTranspose{Tv}`: Lazy point-contiguous layout for scalar SIMD
- `extrap::E`: Extrapolation mode (compile-time specialized via type parameter)
- `search_policy::P`: Default search policy for interval lookup (immutable, thread-safe)

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

sitp = linear_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = sitp(0.5)            # Returns Vector{Float64} of length 3
sitp(output, 0.5)           # In-place

# Vector evaluation
vals = sitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
sitp([out1, out2, out3], xq)    # In-place (zero allocation)

# Complex values are also supported
y_complex = [exp.(2im * π * x), (1.0+2.0im) .* x]
sitp_complex = linear_interp(x, y_complex)
```

# Performance
- Vector queries use series-contiguous layout directly
- Scalar queries trigger lazy transpose on first call

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct LinearSeriesInterpolant{Tg, Tv, E <: AbstractExtrap, P <: AbstractSearchPolicy, X <: AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (wrapped — `_CachedVector` for Vector, `_CachedRange` for Range; carries cached `h`/`inv_h`)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const extrap::E                        # Extrapolation mode (compile-time specialized)
    const search_policy::P                # Default search policy (immutable, thread-safe)

    function LinearSeriesInterpolant(
            x::AbstractVector{Tg},
            y::Matrix{Tv},
            extrap::E,
            search::P = AutoSearch();
            bc::AbstractBC = NoBC()
        ) where {Tg, Tv, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        # `_cache_axis` (insurance) + `_convert_copy` (ownership). Series
        # factory pre-extends `:exclusive` to `:inclusive` form, so `bc` here
        # is normally `NoBC` or `:inclusive`. y is owned by `_build_series_mat`.
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        return new{Tg, Tv, E, P, typeof(xc)}(xc, y, LazyTranspose{Tv}(), extrap, search)
    end
end

# ========================================
# Required Trait Implementations
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(sitp::LinearSeriesInterpolant) = sitp.extrap isa WrapExtrap

"""Number of series in the interpolant."""
@inline n_series(sitp::LinearSeriesInterpolant) = size(sitp.y, 2)

"""Number of grid points in the interpolant."""
@inline n_points(sitp::LinearSeriesInterpolant) = size(sitp.y, 1)

"""Return the x-grid vector."""
@inline _get_grid(sitp::LinearSeriesInterpolant) = sitp.x

"""Return the extrapolation mode."""
@inline _get_extrap(sitp::LinearSeriesInterpolant) = sitp.extrap

# ========================================
# Series Interface Traits
# ========================================

"""Return the interpolation method kind for kernel dispatch."""
@inline _method_kind(::Type{<:LinearSeriesInterpolant}) = Val(:linear)

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::LinearSeriesInterpolant) -> y_point

Ensure point-contiguous layout exists. Delegates to shared LazyTranspose infrastructure.
"""
@inline function _ensure_point_layout!(sitp::LinearSeriesInterpolant)
    return _ensure_transpose!(sitp._transpose, sitp.y)
end

"""
    precompute_transpose!(sitp::LinearSeriesInterpolant) -> sitp

Pre-allocate point-contiguous matrix for scalar queries.
Call before hot loops to avoid first-call latency.
"""
function precompute_transpose!(sitp::LinearSeriesInterpolant)
    _ensure_point_layout!(sitp)
    return sitp
end

# ========================================
# Constructors
# ========================================

# ========================================
# Series Constructor (canonical entry point)
# ========================================

"""
    linear_interp(x, Series(y1, y2, ...); extrap=NoExtrap(), search=AutoSearch())
    linear_interp(x, Series([y1, y2, ...]); ...)
    linear_interp(x, Series(Y::AbstractMatrix); ...)

Create a multi-Y linear interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `s::Series`: Wrapped series data (varargs, vector-of-vectors, or matrix)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `search::AbstractSearchPolicy`: Search policy for interval lookup

# Returns
`LinearSeriesInterpolant` object with matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)

sitp = linear_interp(x, Series(y1, y2))
vals = sitp(0.5)  # [sin(π), cos(π)]

# Matrix form
Y = hcat(y1, y2)
sitp = linear_interp(x, Series(Y))

# Complex values
y_complex = exp.(2im * π * x)
sitp = linear_interp(x, Series(y_complex, y1))
```
"""
# Unified Series constructor. Handles grid promotion internally via
# _promote_grid_float + _to_float, same as the scalar/vector API.
# The former Real wrapper is absorbed to prevent infinite recursion on duck grids.
function linear_interp(
        x::AbstractVector{Tg},
        s::Series;
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    Tv = _series_eltype(s)
    Tg_new = _promote_grid_float(Tg, Tv)
    x_typed = _to_float(x, Tg_new)

    n_pts = length(x_typed)
    # For duck grids (Dual), Tg_new is not AbstractFloat → _value_type duck fallback
    # returns Tv unchanged (no widening to grid type).
    Tv_out = Tg_new <: AbstractFloat ? _value_type(Tv, Tg_new) : Tv
    y_mat, _ = _build_series_mat(s, n_pts, Tv_out)

    # Periodic path: extend x + y_mat to `:inclusive` form, normalize BC label
    # so the inner ctor sees a self-consistent (x, y, bc) triple.
    if _is_periodic_bc(bc)
        x_typed, y_mat = _prepare_periodic(x_typed, y_mat, bc)
        _validate_series_endpoints(bc, y_mat)
        bc_inner = _bc_after_extend(bc)
        # Materialize against the extended grid on both branches — duck grids
        # keep their eltype, Float grids go through value-type promotion.
        extrap_p = Tg_new <: AbstractFloat ? _promote_extrap(WrapExtrap(), Tv_out) : WrapExtrap()
        # Caching wrap (zero-copy of buffer); ownership copy in inner ctor.
        # Thread `Tg_new` so `Int` ranges + `Float32` data become
        # `_CachedRange{Float32}` (not silently widened to Float64).
        x_eff = _cache_axis(x_typed, bc_inner, Tg_new)
        return LinearSeriesInterpolant(x_eff, y_mat, extrap_p, search; bc = bc_inner)
    end

    extrap_p = Tg_new <: AbstractFloat ? _promote_extrap(extrap, Tv_out) : extrap
    x_eff = _cache_axis(x_typed, bc, Tg_new)
    return LinearSeriesInterpolant(x_eff, y_mat, extrap_p, search; bc = bc)
end

# ========================================
# Scalar Evaluation (Override default for debugging)
# ========================================

"""
    (sitp::LinearSeriesInterpolant)(xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.
Supports ForwardDiff.Dual input: output type is promoted to include Dual.
"""
function (sitp::LinearSeriesInterpolant{Tg, Tv, P})(
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    T_out = _promote_eltype(_interp_op, Tg, Tv, Tq)
    out = Vector{T_out}(undef, n_series(sitp))
    return sitp(out, xq; deriv = deriv, search = search, hint = hint)
end

"""
    (sitp::LinearSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (in-place).

# AD Support
Supports ForwardDiff.Dual input for automatic differentiation.
The anchor is built from primal value, but original xq is used for arithmetic.
"""
function (sitp::LinearSeriesInterpolant{Tg, Tv, P})(
        output::AbstractVector,  # Relaxed: allows Dual vector
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    _validate_scalar_output(output, n_series(sitp))

    # One lean op/extrap-aware anchor (shared build; NoExtrap throws OOB inside),
    # then the point-contiguous SIMD eval streaming across the K series.
    A = _linear_series_anchor_type(deriv, sitp.extrap, sitp.x, _coord_eltype(Tq, Tg))
    a = _build_series_anchor(
        LinearInterp(), A, sitp.x, xq, sitp.extrap, _should_wrap(sitp),
        _resolve_search(sitp.x, xq, search, hint)
    )
    y_point = _ensure_point_layout!(sitp)
    _linear_series_eval!(output, y_point, a, sitp.extrap)
    return output
end

# ========================================
# Vector Evaluation
# ========================================

"""
    (sitp::LinearSeriesInterpolant)(xq::AbstractVector; deriv=EvalValue())

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
Output type is promoted to wider type for precision preservation.
"""
function (sitp::LinearSeriesInterpolant{Tg, Tv, P})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    n_query = length(xq)
    n_ser = n_series(sitp)
    T_out = _promote_eltype(_interp_op, Tg, Tv, Tq)

    # Explicit Vector{Vector{T_out}} for type stability on Julia LTS
    outputs = Vector{Vector{T_out}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{T_out}(undef, n_query)
    end
    # Delegate to in-place (unified path handles precision preservation)
    sitp(outputs, xq; deriv = deriv, search = search, hint = hint)

    return outputs
end

"""
    (sitp::LinearSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=EvalValue())

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points (any Real type, auto-promoted for search)
- `deriv`: Derivative order (0 or 1)

Zero-alloc by construction (Q outer × K inner): anchor is built once per
query on the stack and reused for all K series in an inner loop, staying in
registers across the K evals. No pool, no `aq_vec` scratch.

# Precision Preservation
Per-query anchor is typed `_LinearAnchoredQuery{Tg, promote_type(Tq, Tg)}`
via the outer constructor's alpha-type inference.
"""
function (sitp::LinearSeriesInterpolant{Tg, Tv, P})(
        outputs::AbstractVector{<:AbstractVector},
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    _validate_series_outputs(outputs, n_series(sitp), length(xq))
    searcher = _resolve_search(sitp.x, xq, search, hint)
    return _linear_series_inplace_kernel!(outputs, sitp, xq, searcher, deriv)
end

# Q×K — small-NQ fast path. Stack-resident anchor per query, no pool.
@inline function _linear_series_qk!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::LinearSeriesInterpolant{Tg},
        xq::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    wrap = _should_wrap(sitp)
    y = sitp.y
    x_grid = sitp.x
    n_ser = n_series(sitp)
    extrap = sitp.extrap
    A = _linear_series_anchor_type(deriv, extrap, x_grid, _coord_eltype(eltype(xq), Tg))
    @inbounds for j in eachindex(xq)
        a = _build_series_anchor(LinearInterp(), A, x_grid, xq[j], extrap, wrap, searcher)
        for k in 1:n_ser
            outputs[k][j] = _linear_series_eval(y, k, a, extrap)
        end
    end
    return outputs
end

# K×Q — large-NQ fast path. Pool-acquired anchor vector, sequential `outputs[k]`
# inner loop (SIMD-able).
@inline @with_pool pool function _linear_series_kq!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::LinearSeriesInterpolant{Tg},
        xq::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    wrap = _should_wrap(sitp)
    y = sitp.y
    x_grid = sitp.x
    n_ser = n_series(sitp)
    extrap = sitp.extrap
    NQ = length(xq)
    A = _linear_series_anchor_type(deriv, extrap, x_grid, _coord_eltype(eltype(xq), Tg))
    anchors = acquire!(pool, A, NQ)
    _fill_series_anchors!(LinearInterp(), anchors, x_grid, xq, extrap, wrap, searcher)
    @inbounds for k in 1:n_ser
        for j in 1:NQ
            outputs[k][j] = _linear_series_eval(y, k, anchors[j], extrap)
        end
    end
    return outputs
end

# Thin function barrier: ensures the resolved `searcher` (and any hint Ref it
# may carry) is consumed inside a fresh stack frame, preventing escape-analysis
# spillover into a 16-byte heap box.
#
# Adaptive: `length(xq)` and `n_series(sitp)` feed `_series_use_kq_loop` to
# select the loop order — see `src/core/series_utils.jl`.
#
# Note `@inline` is intentional: the barrier benefit comes from this being a
# *separate named function* (a clean specialization point for the compiler),
# not from preventing inlining. Empirically `@noinline` regresses to ~32 B
# because the forced function-call frame adds arg-passing overhead.
@inline function _linear_series_inplace_kernel!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::LinearSeriesInterpolant{Tg},
        xq::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    if _series_use_kq_loop(length(xq), n_series(sitp))
        return _linear_series_kq!(outputs, sitp, xq, searcher, deriv)
    else
        return _linear_series_qk!(outputs, sitp, xq, searcher, deriv)
    end
end
