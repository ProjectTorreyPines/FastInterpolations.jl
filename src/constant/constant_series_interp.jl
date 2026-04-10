# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    CONSTANT SERIES INTERPOLATION                          ║
# ║         Multiple y-data series sharing the same x-grid                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase E.2: Unified matrix storage for optimal performance.
# Key optimization: Adaptive layout with lazy transpose for scalar queries.
#
# Include order: ... → constant_anchor.jl → constant_series_interp.jl
#

# ========================================
# Type Definition
# ========================================

"""
    ConstantSeriesInterpolant{Tg, Tv, P, X}

Multi-series constant (step) interpolant with unified matrix storage and SIMD optimization.
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
- `extrap::AbstractExtrap`: Extrapolation mode
- `side::SD`: Side selection (NearestSide(), LeftSide(), RightSide())

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
mutable struct ConstantSeriesInterpolant{Tg, Tv, S <: AbstractGridSpacing{Tg}, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy, X <: AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (Range or Vector)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const spacing::S                      # Grid spacing (ScalarSpacing or VectorSpacing)
    const extrap::E                        # Extrapolation mode (compile-time specialized)
    const side::SD                        # Side selection (compile-time specialized)
    const search_policy::P                # Default search policy

    function ConstantSeriesInterpolant(
            x::AbstractVector{Tg},
            y::Matrix{Tv},
            extrap::E,
            side::SD,
            search::P = AutoSearch()
        ) where {Tg, Tv, E <: AbstractExtrap, SD <: AbstractSide, P <: AbstractSearchPolicy}
        # _to_float(copy(x), Tg): Range → _CachedRange (O(1) search + no TwicePrecision overhead);
        # Vector → defensive copy. copy() on Range is identity (zero alloc).
        # typeof(xc) rebinds X after conversion (view → Vector, TwicePrecision → _CachedRange).
        # y is NOT copied here — _build_series_mat() already provides an owned matrix.
        xc = _to_float(copy(x), Tg)
        spacing = _create_spacing(xc)
        return new{Tg, Tv, typeof(spacing), E, SD, P, typeof(xc)}(xc, y, LazyTranspose{Tv}(), spacing, extrap, side, search)
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

"""
    _make_anchor(sitp::ConstantSeriesInterpolant, xq::Tg, searcher) -> _ConstantAnchoredQuery{Tg}

Build anchor for a query point. Required trait for AbstractSeriesInterpolant.
"""
@inline function _make_anchor(sitp::ConstantSeriesInterpolant{Tg}, xq::Tg, searcher::P = DEFAULT_SEARCHER) where {Tg, P <: Searcher}
    return _constant_anchor_query_impl(sitp.x, xq, _should_wrap(sitp), searcher)
end

"""
    _eval_series_at_anchor!(output, sitp::ConstantSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.
"""
@inline function _eval_series_at_anchor!(
        output::AbstractVector{Tv},
        sitp::ConstantSeriesInterpolant{Tg, Tv},
        aq::_ConstantAnchoredQuery{Tg},
        op::AbstractEvalOp
    ) where {Tg, Tv}
    y_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    _eval_constant_series_point_extrap!(output, y_point, sitp.x, n_pts, x_min, x_max, aq, sitp.extrap, sitp.side, op, aq.state)
    return output
end

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
# SIMD Scalar Evaluation Kernels
# ========================================

"""
    _eval_constant_series_point!(output, sitp, aq, xq, op)

Main entry point for scalar evaluation of multi-series constant interpolation.
Uses anchor for index/side info, but computes dL from original `xq` for AD support.

# AD Support
Supports ForwardDiff.Dual input: anchor provides index/side from primal,
while `xq` is used directly in arithmetic to preserve derivative information.

# Arguments
- `output`: Pre-allocated output vector (length = n_series)
- `sitp`: ConstantSeriesInterpolant
- `aq`: Anchor with precomputed index/side (from primal value)
- `xq`: Original query point (any Real type, including ForwardDiff.Dual)
- `op`: Evaluation operation (value, derivative)

Note: Inside-domain evaluation uses this function directly.
Outside-domain delegates to `_eval_series_at_anchor!` for extrapolation.
"""
@inline function _eval_constant_series_point!(
        output::AbstractVector,  # Relaxed: accepts any element type for lossless promotion
        sitp::ConstantSeriesInterpolant{Tg, Tv},
        aq::_ConstantAnchoredQuery{Tg},
        xq,  # Original xq (any Real, including Dual)
        op::AbstractEvalOp
    ) where {Tg, Tv}
    # Outside domain: delegate to extrapolation handler (trait method)
    if aq.state != IN_DOMAIN
        return _eval_series_at_anchor!(output, sitp, aq, op)
    end

    # Inside domain: SIMD evaluation with point-contiguous layout
    y_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)

    # Special case: at right boundary (use primal for comparison)
    xq_primal = _extract_primal(xq)
    if Tg(xq_primal) == Tg(last(sitp.x))
        if op isa EvalValue
            @inbounds @simd for k in axes(output, 1)
                output[k] = y_point[k, n_pts]
            end
        else
            z = 0 * first(y_point)
            @inbounds @simd for k in axes(output, 1)
                output[k] = z
            end
        end
        return output
    end

    # Inside domain: compute dL from original xq for AD support
    idx = aq.idx
    idx1 = idx + 1
    h = aq.h
    @inbounds xL = sitp.x[idx]
    dL = xq - xL  # Use original xq to preserve Dual for AD

    # SIMD loop over series
    @inbounds @simd for k in axes(output, 1)
        y_left = y_point[k, idx]
        y_right = y_point[k, idx1]
        output[k] = _constant_kernel(op, y_left, y_right, h, dL, sitp.side)
    end

    return output
end

# NoExtrap - throw DomainError
@inline function _eval_constant_series_point_extrap!(
        ::AbstractVector{Tv},
        ::Matrix{Tv},
        ::AbstractVector{Tg},
        ::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_ConstantAnchoredQuery{Tg},
        ::NoExtrap,
        ::AbstractSide,
        ::AbstractEvalOp,
        ::UInt8
    ) where {Tg, Tv}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# ClampExtrap - clamp to boundary
@inline function _eval_constant_series_point_extrap!(
        out::AbstractVector{Tv},
        y_point::Matrix{Tv},
        ::AbstractVector{Tg},
        n_pts::Int,
        ::Tg,
        ::Tg,
        ::_ConstantAnchoredQuery{Tg},
        extrap::_ClampOrFill,
        ::AbstractSide,
        op::AbstractEvalOp,
        side::UInt8
    ) where {Tg, Tv}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op, extrap)
end

# ExtendExtrap - extend using same constant value at boundary interval
@inline function _eval_constant_series_point_extrap!(
        out::AbstractVector{Tv},
        y_point::Matrix{Tv},
        x::AbstractVector{Tg},
        n_pts::Int,
        ::Tg,
        ::Tg,
        aq::_ConstantAnchoredQuery{Tg},
        ::ExtendExtrap,
        side_val::AbstractSide,
        op::AbstractEvalOp,
        ::UInt8
    ) where {Tg, Tv}
    # Use boundary interval for extension (inline evaluation, no xq needed)
    idx = aq.idx
    idx1 = idx + 1
    h = aq.h
    dL = aq.dL

    # SIMD loop over series
    @inbounds @simd for k in axes(out, 1)
        y_left = y_point[k, idx]
        y_right = y_point[k, idx1]
        out[k] = _constant_kernel(op, y_left, y_right, h, dL, side_val)
    end
    return out
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
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    # Type promotion: widen grid if y's float base is wider than Tg
    Tv = _series_eltype(s)
    Tg_new = _promote_grid_float(Tg, Tv)
    if Tg_new !== Tg
        return constant_interp(_to_float(x, Tg_new), s; side, extrap, search)
    end

    n_pts = length(x)
    Tv_out = _value_type(Tv, Tg)
    y_mat, _ = _build_series_mat(s, n_pts, Tv_out)
    extrap_p = _promote_extrap(extrap, Tv_out)

    return ConstantSeriesInterpolant(x, y_mat, extrap_p, side, search)
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
When `xq` is a ForwardDiff.Dual, the output type is promoted to preserve
derivatives. Output type is `promote_type(Tv, Tq)`.
"""
function (sitp::ConstantSeriesInterpolant{Tg, Tv, P})(
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    T_out = _series_output_type(Tv, Tq)
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
    ) where {Tg, Tv, P, Tq <: Real}
    n_ser = n_series(sitp)

    # Validate output length
    _validate_scalar_output(output, n_ser)

    # AD Support: Extract primal for anchor building, pass original xq for AD
    xq_primal = _extract_primal(xq)
    xq_typed = Tg(xq_primal)

    # Build anchor using primal value
    aq = _make_anchor(sitp, xq_typed, _resolve_search(sitp.x, xq, search, hint))

    # Dispatch on derivative order - pass original xq for AD support
    _eval_constant_series_point!(output, sitp, aq, xq, deriv)
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
    ) where {Tg, Tv, P, Tq <: Real}
    xq_typed = _to_float(xq, Tg)
    n_query = length(xq_typed)
    n_ser = n_series(sitp)

    # Explicit Vector{Vector{Tv}} for type stability on Julia LTS
    outputs = Vector{Vector{Tv}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{Tv}(undef, n_query)
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

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
Uses task-local pool for anchor vector to achieve zero allocation after warmup.
"""
@with_pool pool function (sitp::ConstantSeriesInterpolant{Tg, Tv, P})(
        outputs::AbstractVector{<:AbstractVector{Tv}},
        xq::AbstractVector{<:Real};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P}
    xq_typed = _to_float(xq, Tg)
    n_query = length(xq_typed)
    n_ser = n_series(sitp)

    # Validate dimensions
    _validate_series_outputs(outputs, n_ser, n_query)

    # Build anchors from pool (zero allocation after warmup)
    aq_vec = acquire!(pool, _ConstantAnchoredQuery{Tg}, length(xq_typed))
    searcher = _resolve_search(sitp.x, xq_typed, search, hint)
    _fill_anchors!(aq_vec, sitp.x, xq_typed, Val(:constant), _should_wrap(sitp), searcher)

    # Extract matrices for argument-passing pattern
    y = sitp.y
    x_grid = sitp.x
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    side_val = sitp.side
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    # Evaluate all series with derivative dispatch
    @inbounds for k in 1:n
        _eval_constant_series_vector!(outputs[k], y, x_grid, n_pts, x_min, x_max, k, aq_vec, extrap, side_val, deriv)
    end
    return outputs
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance.
"""
@inline function _eval_constant_series_vector!(
        out::AbstractVector{Tv},
        y::Matrix{Tv},
        x::AbstractVector{Tg},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        k::Int,
        aq_vec::AbstractVector{<:_ConstantAnchoredQuery{Tg}},
        extrap::AbstractExtrap,
        side_val::AbstractSide,
        op::AbstractEvalOp
    ) where {Tg, Tv}
    @inbounds for j in eachindex(out, aq_vec)
        out[j] = _eval_constant_series_with_extrap(y, x, n_pts, x_min, x_max, k, aq_vec[j], extrap, side_val, op)
    end
    return out
end

"""
Internal: Evaluate single series at single query point with extrapolation handling.
"""
@inline function _eval_constant_series_with_extrap(
        y::Matrix{Tv},
        x::AbstractVector{Tg},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        k::Int,
        aq::_ConstantAnchoredQuery{Tg},
        extrap::AbstractExtrap,
        side_val::AbstractSide,
        op::AbstractEvalOp
    ) where {Tg, Tv}
    # Special case: at right boundary (MUST be preserved!)
    if aq.xq == x_max
        if op isa EvalValue
            @inbounds return y[n_pts, k]
        else
            return 0 * first(y)  # Derivatives of step function are zero
        end
    end

    # Inside domain: normal evaluation
    if aq.state == IN_DOMAIN
        return _eval_constant_series_anchored(y, k, aq, side_val, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap isa ExtendExtrap || extrap isa WrapExtrap
        return _eval_constant_series_anchored(y, k, aq, side_val, op)
    elseif extrap isa _ClampOrFill
        return _constant_extrap_boundary_value(y, aq.state, n_pts, k, op, extrap)
    else
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
end

"""
Internal: Core constant evaluation for series k at anchored query point.
"""
@inline function _eval_constant_series_anchored(
        y::Matrix{Tv},
        k::Int,
        aq::_ConstantAnchoredQuery{Tg},
        side_val::AbstractSide,
        op::AbstractEvalOp
    ) where {Tg, Tv}
    # Derivatives of constant (step) function are zero
    if !(op isa EvalValue)
        return 0 * first(y)
    end

    idx = aq.idx
    @inbounds begin
        y_left = y[idx, k]
        y_right = y[idx + 1, k]
    end
    return _constant_kernel(EvalValue(), y_left, y_right, aq.h, aq.dL, side_val)
end
