# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CONSTANT SERIES INTERPOLATION                          ║
# ║         Multiple y-data series sharing the same x-grid                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Unified matrix storage with adaptive layout — lazy transpose for scalar queries.
#
# Include order: ... → constant_anchor.jl → constant_series_interp.jl
#

# ========================================
# Type Definition
# ========================================

"""
    ConstantSeriesInterpolant{Tg, Tv, E, SD, P, X}

Multi-series constant (step) interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `Tg`: Grid type (unconstrained — supports duck types like ForwardDiff.Dual)
- `Tv`: Value type (unconstrained)
- `E<:AbstractExtrap`: Extrapolation mode type (compile-time specialized)
- `SD<:AbstractSide`: Side selection type (compile-time specialized)
- `P<:AbstractSearchPolicy`: Search policy type
- `X<:AbstractVector{Tg}`: Grid container type — `_CachedRange{Tg}` for Range input,
  `_CachedVector{Tg,Tinv}` for Vector input (carries cached `h`/`inv_h`).

# Fields
- `x::X`: Shared x-grid (wrapped — `_CachedVector`/`_CachedRange`)
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `_transpose::LazyTranspose{Tv}`: Lazy point-contiguous layout for scalar SIMD
- `extrap::E`: Extrapolation mode (compile-time specialized)
- `side::SD`: Side selection (NearestSide(), LeftSide(), RightSide())
- `search_policy::P`: Default search policy

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

sitp = constant_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = sitp(0.5)            # Returns Vector{Float64} of length 3
sitp(output, 0.5)           # In-place

# Vector evaluation
vals = sitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
sitp([out1, out2, out3], xq)    # In-place (zero allocation)

# Complex values are also supported
y_complex = [exp.(2im * π * x), (1.0+2.0im) .* x]
sitp_complex = constant_interp(x, y_complex)
```

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct ConstantSeriesInterpolant{Tg, Tv, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy, X <: AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (wrapped — `_CachedVector`/`_CachedRange` carrying cached `h`/`inv_h`)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const extrap::E                        # Extrapolation mode (compile-time specialized)
    const side::SD                        # Side selection (compile-time specialized)
    const search_policy::P                # Default search policy

    function ConstantSeriesInterpolant(
            x::AbstractVector{Tg},
            y::Matrix{Tv},
            extrap::E,
            side::SD,
            search::P = AutoSearch();
            bc::AbstractBC = NoBC()
        ) where {Tg, Tv, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy}
        # `_cache_axis` (insurance) + `_convert_copy` (ownership). Series
        # factory pre-extends `:exclusive` to `:inclusive` form, so `bc` here
        # is normally `NoBC` or `:inclusive`. y is owned by `_build_series_mat`.
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        return new{Tg, Tv, E, SD, P, typeof(xc)}(xc, y, LazyTranspose{Tv}(), extrap, side, search)
    end
end

# ========================================
# Required Trait Implementations
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(sitp::ConstantSeriesInterpolant) = sitp.extrap isa WrapExtrap

"""Number of series in the interpolant."""
@inline n_series(sitp::ConstantSeriesInterpolant) = size(sitp.y, 2)

"""Number of grid points in the interpolant."""
@inline n_points(sitp::ConstantSeriesInterpolant) = size(sitp.y, 1)

"""Return the x-grid vector."""
@inline _get_grid(sitp::ConstantSeriesInterpolant) = sitp.x

"""Return the extrapolation mode."""
@inline _get_extrap(sitp::ConstantSeriesInterpolant) = sitp.extrap

# ========================================
# Series Interface Traits
# ========================================

"""Return the interpolation method kind for kernel dispatch."""
@inline _method_kind(::Type{<:ConstantSeriesInterpolant}) = Val(:constant)

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::ConstantSeriesInterpolant) -> y_point

Ensure point-contiguous layout exists. Delegates to shared LazyTranspose infrastructure.
"""
@inline function _ensure_point_layout!(sitp::ConstantSeriesInterpolant)
    return _ensure_transpose!(sitp._transpose, sitp.y)
end

"""
    precompute_transpose!(sitp::ConstantSeriesInterpolant) -> sitp

Pre-allocate point-contiguous matrix for scalar queries.
Call before hot loops to avoid first-call latency.
"""
function precompute_transpose!(sitp::ConstantSeriesInterpolant)
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
    constant_interp(x, Series(y1, y2, ...); side=NearestSide(), extrap=NoExtrap(), search=AutoSearch())
    constant_interp(x, Series([y1, y2, ...]); ...)
    constant_interp(x, Series(Y::AbstractMatrix); ...)

Create a multi-Y constant (step) interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `s::Series`: Wrapped series data (varargs, vector-of-vectors, or matrix)
- `side::AbstractSide`: `NearestSide()`, `LeftSide()`, or `RightSide()`
- `extrap::AbstractExtrap`: `NoExtrap()`, `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `search::AbstractSearchPolicy`: Search policy for interval lookup

# Example
```julia
x = collect(range(0.0, 1.0, 101))
sitp = constant_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))
```
"""
function constant_interp(
        x::AbstractVector{Tg},
        s::Series;
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    Tv_out = _series_eltype(s)
    n_pts = length(x)
    y_mat, _ = _build_series_mat(s, n_pts, Tv_out)

    # Periodic path: extend x + y_mat to `:inclusive` form, normalize BC label
    # so the inner ctor sees a self-consistent (x, y, bc) triple.
    if _is_periodic_bc(bc)
        x_ext, y_mat_ext = _prepare_periodic(x, y_mat, bc)
        _validate_series_endpoints(bc, y_mat_ext)
        bc_inner = _bc_after_extend(bc)
        # `WrapExtrap` is a tag struct — wrap domain is read from the extended
        # axis at query time. `_promote_extrap` only handles `FillExtrap` value
        # promotion; passthrough for `WrapExtrap`.
        extrap_p = _promote_extrap(WrapExtrap(), Tv_out)
        # Caching wrap (zero-copy of buffer); ownership copy in inner ctor.
        # Thread `Tg` so `Int` ranges + `Float32` data become
        # `_CachedRange{Float32}` (not silently widened to Float64).
        x_eff = _cache_axis(x_ext, bc_inner, Tg)
        return ConstantSeriesInterpolant(x_eff, y_mat_ext, extrap_p, side, search; bc = bc_inner)
    end

    extrap_p = _promote_extrap(extrap, Tv_out)
    x_eff = _cache_axis(x, bc, Tg)
    return ConstantSeriesInterpolant(x_eff, y_mat, extrap_p, side, search; bc = bc)
end

# NOTE: the former Real grid promotion wrapper (Tg <: Real) has been removed.
# The constructor above now uses unconstrained Tg and handles promotion internally.

# ========================================
# Scalar Evaluation
# ========================================

"""
    (sitp::ConstantSeriesInterpolant)(xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.

# Derivative support
- `deriv=EvalValue()`: Returns function values
- `deriv=DerivOp(1),DerivOp(2)`: Returns zeros (step function derivative is zero everywhere)

# AD Support
Output eltype routes through the kernel-shape trait
`_promote_eltype(_select_op, Tg, Tv, Tq)`. `Base.promote_op` on
`yv * one(xq - xL)` jointly considers grid (Tg), value (Tv) and query (Tq),
so duck carriers on either Tg (e.g. `Dual` grid + `Float` xq) or Tq (`Float`
grid + `Dual` xq) propagate uniformly. Carriers like `SVector × Dual`
resolve correctly instead of collapsing to `Vector{Any}`.
"""
function (sitp::ConstantSeriesInterpolant{Tg, Tv, P})(
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Number}
    T_out = _deriv_eltype(_promote_eltype(_select_op, Tg, Tv, Tq), Tg, deriv)
    out = Vector{T_out}(undef, n_series(sitp))
    return sitp(out, xq; deriv = deriv, search = search, hint = hint)
end

"""
    (sitp::ConstantSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (sitp::ConstantSeriesInterpolant{Tg, Tv, P})(
        output::AbstractVector,  # Relaxed: accepts any element type for lossless promotion
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Number}
    _validate_scalar_output(output, n_series(sitp))

    # One lean op/side/extrap-aware gather anchor (shared build; NoExtrap throws
    # OOB inside), then the point-contiguous SIMD gather across the K series.
    # `ConstantInterp(sitp.side)` threads the side into `select_right` at build.
    A = _constant_series_anchor_type(deriv, sitp.extrap, sitp.x, _coord_eltype(Tq, Tg))
    a = _build_series_anchor(
        ConstantInterp(sitp.side), A, sitp.x, xq, sitp.extrap, _should_wrap(sitp),
        _resolve_search(sitp.x, xq, search, hint)
    )
    y_point = _ensure_point_layout!(sitp)
    _constant_series_eval!(output, y_point, a, sitp.extrap)
    return output
end

# ========================================
# Vector Evaluation
# ========================================

"""
    (sitp::ConstantSeriesInterpolant)(xq::AbstractVector; deriv=EvalValue())

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
"""
function (sitp::ConstantSeriesInterpolant{Tg, Tv, P})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Number}
    # Normalize queries to the grid's base float type (not Tg itself, which may be Dual)
    xq_typed = _promote_query_typed(xq, Tg)
    n_query = length(xq_typed)
    n_ser = n_series(sitp)

    T_out = _deriv_eltype(_promote_eltype(_select_op, Tg, Tv, Tq), Tg, deriv)
    outputs = Vector{Vector{T_out}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{T_out}(undef, n_query)
    end
    sitp(outputs, xq_typed; deriv = deriv, search = search, hint = hint)

    return outputs
end

"""
    (sitp::ConstantSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=EvalValue())

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points
- `deriv`: Derivative order (0, 1, or 2)

Zero-alloc by construction (Q outer × K inner): anchor is built once per
query on the stack and reused for all K series, staying in registers across
the K evals. No pool, no `aq_vec` scratch.
"""
function (sitp::ConstantSeriesInterpolant{Tg, Tv, P})(
        outputs::AbstractVector{<:AbstractVector},
        xq::AbstractVector{<:Number};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P}
    # Normalize queries to the grid's base float type (not Tg itself, which may be Dual)
    xq_typed = _promote_query_typed(xq, Tg)
    _validate_series_outputs(outputs, n_series(sitp), length(xq_typed))
    searcher = _resolve_search(sitp.x, xq_typed, search, hint)
    return _constant_series_inplace_kernel!(outputs, sitp, xq_typed, searcher, deriv)
end

# Q×K — small-NQ fast path. Stack-resident anchor per query, no pool.
@inline function _constant_series_qk!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::ConstantSeriesInterpolant{Tg},
        xq_typed::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    wrap = _should_wrap(sitp)
    y = sitp.y
    x_grid = sitp.x
    n_ser = n_series(sitp)
    extrap = sitp.extrap
    m = ConstantInterp(sitp.side)
    A = _constant_series_anchor_type(deriv, extrap, x_grid, _coord_eltype(eltype(xq_typed), Tg))
    @inbounds for j in eachindex(xq_typed)
        a = _build_series_anchor(m, A, x_grid, xq_typed[j], extrap, wrap, searcher)
        for k in 1:n_ser
            outputs[k][j] = _constant_series_eval(y, k, a, extrap)
        end
    end
    return outputs
end

# K×Q — large-NQ fast path. Pool-acquired anchor vector, sequential `outputs[k]`
# inner loop (SIMD-able).
@inline @with_pool pool function _constant_series_kq!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::ConstantSeriesInterpolant{Tg},
        xq_typed::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    wrap = _should_wrap(sitp)
    y = sitp.y
    x_grid = sitp.x
    n_ser = n_series(sitp)
    extrap = sitp.extrap
    NQ = length(xq_typed)
    A = _constant_series_anchor_type(deriv, extrap, x_grid, _coord_eltype(eltype(xq_typed), Tg))
    anchors = acquire!(pool, A, NQ)
    _fill_series_anchors!(ConstantInterp(sitp.side), anchors, x_grid, xq_typed, extrap, wrap, searcher)
    @inbounds for k in 1:n_ser
        for j in 1:NQ
            outputs[k][j] = _constant_series_eval(y, k, anchors[j], extrap)
        end
    end
    return outputs
end

# Thin function barrier: see comment on `_linear_series_inplace_kernel!`.
# Adaptive: `length(xq_typed)` and `n_series(sitp)` feed `_series_use_kq_loop`
# to select the loop order — see `src/core/series_utils.jl`.
@inline function _constant_series_inplace_kernel!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::ConstantSeriesInterpolant{Tg},
        xq_typed::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    if _series_use_kq_loop(length(xq_typed), n_series(sitp))
        return _constant_series_kq!(outputs, sitp, xq_typed, searcher, deriv)
    else
        return _constant_series_qk!(outputs, sitp, xq_typed, searcher, deriv)
    end
end
