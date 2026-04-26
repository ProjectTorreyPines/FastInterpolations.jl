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
    LinearSeriesInterpolant{Tg, Tv, P, X}

Multi-series linear interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `Tg`: Grid type (Float32 or Float64)
- `Tv`: Value type (unconstrained)
- `P`: Search policy type
- `X`: Grid container type (Vector or Range)

# Fields
- `x::X`: Shared x-grid (Vector or Range)
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `_transpose::LazyTranspose{Tv}`: Lazy point-contiguous layout for scalar SIMD
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
mutable struct LinearSeriesInterpolant{Tg, Tv, E <: AbstractExtrap, P <: AbstractSearchPolicy, X <: AbstractVector{Tg}, S <: AbstractGridSpacing{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (Range or Vector)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const spacing::S                      # Precomputed grid spacing (ScalarSpacing for Range, VectorSpacing for Vector)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const extrap::E                        # Extrapolation mode (compile-time specialized)
    const search_policy::P                # Default search policy (immutable, thread-safe)

    function LinearSeriesInterpolant(
            x::AbstractVector{Tg},
            y::Matrix{Tv},
            extrap::E,
            search::P = AutoSearch()
        ) where {Tg, Tv, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        # _to_float(copy(x), Tg): Range → _CachedRange (O(1) search + no TwicePrecision overhead);
        # Vector → defensive copy (copy is identity for immutable Range, alloc for Vector).
        # typeof(xc) rebinds X after conversion (view → Vector, TwicePrecision → _CachedRange).
        # y is NOT copied here — _build_series_mat() already provides an owned matrix.
        xc = _to_float(copy(x), Tg)
        spacing = _create_spacing(xc)
        return new{Tg, Tv, E, P, typeof(xc), typeof(spacing)}(xc, y, spacing, LazyTranspose{Tv}(), extrap, search)
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

"""
    _make_anchor(sitp::LinearSeriesInterpolant, xq) -> _LinearAnchoredQuery

Build anchor for a query point. Required trait for AbstractSeriesInterpolant.
Accepts any query type — `_linear_anchor_query_impl` + the outer constructor of
`_LinearAnchoredQuery` handle primal extraction for search and type widening.
"""
@inline function _make_anchor(sitp::LinearSeriesInterpolant{Tg}, xq, searcher::P = DEFAULT_SEARCHER) where {Tg, P <: Searcher}
    return _linear_anchor_query_impl(sitp.x, xq, _should_wrap(sitp), searcher)
end

"""
    _eval_series_at_anchor!(output, sitp::LinearSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.
"""
@inline function _eval_series_at_anchor!(
        output::AbstractVector{Tv},
        sitp::LinearSeriesInterpolant{Tg, Tv},
        aq::_LinearAnchoredQuery{Tg},
        op::AbstractEvalOp
    ) where {Tg, Tv}
    y_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    _eval_linear_series_point_extrap!(output, y_point, sitp.x, n_pts, x_min, x_max, aq, sitp.extrap, op, aq.state)
    return output
end

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
# SIMD Scalar Evaluation Kernels
# ========================================

# NoExtrap - throw DomainError
@inline function _eval_linear_series_point_extrap!(
        ::AbstractVector{Tv},
        ::Matrix{Tv},
        ::AbstractVector{Tg},
        ::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_LinearAnchoredQuery{Tg},
        ::NoExtrap,
        ::AbstractEvalOp,
        ::UInt8
    ) where {Tg, Tv}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# ClampExtrap/FillExtrap - clamp to boundary or fill (value only, derivatives are zero)
@inline function _eval_linear_series_point_extrap!(
        out::AbstractVector{Tv},
        y_point::Matrix{Tv},
        ::AbstractVector{Tg},
        n_pts::Int,
        ::Tg,
        ::Tg,
        ::_LinearAnchoredQuery{Tg},
        extrap::_ClampOrFill,
        op::AbstractEvalOp,
        side::UInt8
    ) where {Tg, Tv}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op, extrap)
end

# ExtendExtrap - extend linear polynomial using boundary interval.
# _get_inv_h(x, xL, xR) dispatches to x.inv_h (_CachedRange) or inv(xR-xL) (Vector).
@inline function _eval_linear_series_point_extrap!(
        out::AbstractVector{Tv},
        y_point::Matrix{Tv},
        x::AbstractVector{Tg},
        n_pts::Int,
        ::Tg,
        ::Tg,
        aq::_LinearAnchoredQuery{Tg},
        ::ExtendExtrap,
        op::AbstractEvalOp,
        side::UInt8
    ) where {Tg, Tv}
    idx = side == OOB_LEFT ? 1 : (n_pts - 1)
    idx1 = idx + 1
    @inbounds begin
        xL = x[idx]
        xR = x[idx1]
    end
    inv_h = _get_inv_h(x, xL, xR)
    α = _alpha_of(aq.xq, xL, xR, x)
    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        out[k] = _linear_kernel(op, yL, yR, inv_h, α)
    end
    return out
end

# ========================================
# Scalar Evaluation Core
# ========================================

"""
    _eval_linear_series_point!(output, sitp, aq, op)

Core scalar evaluation for all series at a single query point.
Uses SIMD-optimized point-contiguous layout for vectorization across series.

Query information (`aq.xq`, `aq.xL`, `aq.inv_h`) is read from the anchor.
For duck grids (Dual), `aq.xq` carries the widened query type (via the outer
constructor), so `dL = aq.xq - aq.xL` correctly propagates grid-side partials.

# Arguments
- `output`: Pre-allocated output vector (length = n_series)
- `sitp`: LinearSeriesInterpolant
- `aq`: Anchor with precomputed indices (`idxL`/`idxR`) and widened `xq`
- `op`: Evaluation operation (value, derivative)

# AD Support
Supports ForwardDiff.Dual input: the anchor's indices come from the primal value,
while `aq.xq` carries the Dual payload so `dL = aq.xq - aq.xL` preserves derivatives.
"""
@inline function _eval_linear_series_point!(
        output::AbstractVector,
        sitp::LinearSeriesInterpolant{Tg, Tv},
        aq::_LinearAnchoredQuery{Tg},
        op::AbstractEvalOp
    ) where {Tg, Tv}
    # Outside domain: delegate to extrapolation handler
    if aq.state != IN_DOMAIN
        return _eval_series_at_anchor!(output, sitp, aq, op)
    end

    # Inside domain: SIMD evaluation with point-contiguous layout
    y_point = _ensure_point_layout!(sitp)
    idxL = aq.idxL
    idxR = aq.idxR

    inv_h = aq.inv_h
    α = aq.alpha  # precomputed by anchor constructor

    @inbounds @simd for k in axes(output, 1)
        yL = y_point[k, idxL]
        yR = y_point[k, idxR]
        output[k] = _linear_kernel(op, yL, yR, inv_h, α)
    end
    return output
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

    # Periodic path: extend x + y_mat once (matrix overload of `_prepare_periodic`),
    # validate `:inclusive` endpoints per series, force WrapExtrap.
    if _is_periodic_bc(bc)
        x_typed, y_mat = _prepare_periodic(x_typed, y_mat, bc)
        _validate_series_endpoints(bc, y_mat)
        # Materialize against the extended grid on both branches — duck grids
        # keep their eltype, Float grids go through value-type promotion.
        extrap_p = Tg_new <: AbstractFloat ? _promote_extrap(WrapExtrap(x_typed), Tv_out) : WrapExtrap(x_typed)
        return LinearSeriesInterpolant(x_typed, y_mat, extrap_p, search)
    end

    extrap_p = Tg_new <: AbstractFloat ? _promote_extrap(extrap, Tv_out) : extrap
    return LinearSeriesInterpolant(x_typed, y_mat, extrap_p, search)
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
    T_out = _series_output_type(_output_eltype(Tv, Tg), Tq)
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
    n_ser = n_series(sitp)

    # Validate output length
    _validate_scalar_output(output, n_ser)

    # Build anchor — _anchor_loc inside handles primal extraction for search,
    # outer constructor widens xq to match grid arithmetic type.
    aq = _make_anchor(sitp, xq, _resolve_search(sitp.x, xq, search, hint))

    _eval_linear_series_point!(output, sitp, aq, deriv)
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
    T_out = _series_output_type(_output_eltype(Tv, Tg), Tq)

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

# Thin function barrier: ensures the resolved `searcher` (and any hint Ref it
# may carry) is consumed inside a fresh stack frame, preventing escape-analysis
# spillover into a 16-byte heap box. No pool needed.
#
# Note `@inline` is intentional: the barrier benefit comes from this being a
# *separate named function* (a clean specialization point for the compiler),
# not from preventing inlining. Empirically `@noinline` regresses to ~32 B
# because the forced function-call frame adds arg-passing overhead — see
# claudedocs/PR_test_infra_refac.md.
@inline function _linear_series_inplace_kernel!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::LinearSeriesInterpolant{Tg},
        xq::AbstractVector,
        searcher::Searcher,
        deriv::DerivOp
    ) where {Tg}
    wrap = _should_wrap(sitp)
    y = sitp.y
    x_grid = sitp.x
    n_pts = n_points(sitp)
    n_ser = n_series(sitp)
    extrap = sitp.extrap
    x_min = Tg(first(sitp.x))
    x_max = Tg(last(sitp.x))

    @inbounds for j in eachindex(xq)
        aq = _anchor_query(x_grid, xq[j], Val(:linear), wrap, searcher)
        for k in 1:n_ser
            outputs[k][j] = _eval_linear_series_with_extrap(
                y, x_grid, n_pts, x_min, x_max, k, aq, extrap, deriv
            )
        end
    end
    return outputs
end

"""
Internal: Evaluate single series at single query point with extrapolation handling.

# Precision Preservation
Uses anchor's precomputed `alpha` for value evaluation (avoids division).
Uses anchor's `xq` for domain error messages.
"""
@inline function _eval_linear_series_with_extrap(
        y::Matrix{Tv},
        x::AbstractVector{Tg},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        k::Int,
        aq::_LinearAnchoredQuery{Tg},
        extrap::AbstractExtrap,
        op::AbstractEvalOp
    ) where {Tg, Tv}
    # Inside domain: normal evaluation
    if aq.state == IN_DOMAIN
        return _eval_linear_series_anchored(y, x, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap isa ExtendExtrap || extrap isa WrapExtrap
        return _eval_linear_series_anchored(y, x, k, aq, op)
    elseif extrap isa _ClampOrFill
        return _constant_extrap_boundary_value(y, aq.state, n_pts, k, op, extrap)
    else
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
end

"""
Internal: Core linear evaluation for series k at anchored query point.

Uses anchor's precomputed values via `_linear_kernel(op, yL, yR, aq)`.
The kernel internally extracts alpha (for EvalValue) or inv_h (for derivatives).
"""
@inline function _eval_linear_series_anchored(
        y::Matrix{Tv},
        ::AbstractVector{Tg},
        k::Int,
        aq::_LinearAnchoredQuery{Tg},
        op::AbstractEvalOp
    ) where {Tg, Tv}
    @inbounds begin
        yL = y[aq.idxL, k]
        yR = y[aq.idxR, k]
    end
    return _linear_kernel(op, yL, yR, aq)
end
