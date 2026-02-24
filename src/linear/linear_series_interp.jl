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
- `Tv`: Value type (Tg for real, Complex{Tg} for complex)
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
mutable struct LinearSeriesInterpolant{Tg<:AbstractFloat, Tv, E<:AbstractExtrap, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (Range or Vector)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const extrap::E                        # Extrapolation mode (compile-time specialized)
    const search_policy::P                # Default search policy (immutable, thread-safe)

    function LinearSeriesInterpolant(
        x::X,
        y::Matrix{Tv},
        extrap::E,
        search::P=AutoSearch()
    ) where {Tg<:AbstractFloat, Tv, E<:AbstractExtrap, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}}
        new{Tg,Tv,E,P,X}(x, y, LazyTranspose{Tv}(), extrap, search)
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
    _make_anchor(sitp::LinearSeriesInterpolant, xq::Tg) -> _LinearAnchoredQuery{Tg}

Build anchor for a query point. Required trait for AbstractSeriesInterpolant.
"""
@inline function _make_anchor(sitp::LinearSeriesInterpolant{Tg}, xq::Tg, searcher::Searcher=DEFAULT_SEARCHER) where Tg
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
) where {Tg<:AbstractFloat, Tv}
    y_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    _eval_linear_series_point_extrap!(output, y_point, sitp.x, n_pts, x_min, x_max, aq, sitp.extrap, op, aq.side)
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
) where {Tg<:AbstractFloat, Tv}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# ConstExtrap - clamp to boundary (value only, derivatives are zero)
@inline function _eval_linear_series_point_extrap!(
    out::AbstractVector{Tv},
    y_point::Matrix{Tv},
    ::AbstractVector{Tg},
    n_pts::Int,
    ::Tg,
    ::Tg,
    ::_LinearAnchoredQuery{Tg},
    ::ConstExtrap,
    op::AbstractEvalOp,
    side::UInt8
) where {Tg<:AbstractFloat, Tv}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op)
end

# ExtendExtrap - extend linear polynomial
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
) where {Tg<:AbstractFloat, Tv}
    # Use boundary interval for extension
    idx = side == 0x01 ? 1 : (n_pts - 1)
    idx1 = idx + 1

    @inbounds begin
        xL = x[idx]
        xR = x[idx1]
    end
    h = xR - xL
    dL = aq.xq - xL

    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        out[k] = _linear_kernel(op, yL, yR, h, dL)
    end
    return out
end

# ========================================
# Scalar Evaluation Core
# ========================================

"""
    _eval_linear_series_point!(output, sitp, aq, xq, op)

Core scalar evaluation for all series at a single query point.
Uses SIMD-optimized point-contiguous layout for vectorization across series.

# Arguments
- `output`: Pre-allocated output vector (length = n_series)
- `sitp`: LinearSeriesInterpolant
- `aq`: Anchor with precomputed index/side (from primal value)
- `xq`: Original query point (any Real type, including ForwardDiff.Dual)
- `op`: Evaluation operation (value, derivative)

# AD Support
Supports ForwardDiff.Dual input: anchor provides index/side from primal,
while `xq` is used directly in arithmetic to preserve derivative information.
"""
@inline function _eval_linear_series_point!(
    output::AbstractVector,
    sitp::LinearSeriesInterpolant{Tg, Tv},
    aq::_LinearAnchoredQuery{Tg},
    xq,  # Original xq (any Real, including Dual)
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    # Outside domain: delegate to extrapolation handler
    if aq.side != 0x00
        return _eval_series_at_anchor!(output, sitp, aq, op)
    end

    # Inside domain: SIMD evaluation with point-contiguous layout
    y_point = _ensure_point_layout!(sitp)
    idx = aq.idx
    idx1 = idx + 1

    @inbounds begin
        xL = sitp.x[idx]
        xR = sitp.x[idx1]
    end
    h = xR - xL
    dL = xq - xL  # Original xq preserves Dual for AD

    @inbounds @simd for k in axes(output, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        output[k] = _linear_kernel(op, yL, yR, h, dL)
    end
    return output
end

# ========================================
# Constructors
# ========================================

"""
    linear_interp(x, ys::AbstractVector{<:AbstractVector}; extrap=NoExtrap())

Create a multi-Y linear interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ConstExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`

# Returns
`LinearSeriesInterpolant` object with matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

sitp = linear_interp(x, [y1, y2, y3])
vals = sitp(0.5)  # [sin(π), cos(π), exp(-0.5)]

# Complex values are also supported
y_complex = [exp.(2im * π * x), (1.0+2.0im) .* x]
sitp_complex = linear_interp(x, y_complex)
```
"""
# Hot path: x is AbstractFloat, ys elements can be Tg or Complex{Tg}
function linear_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    extrap::AbstractExtrap=NoExtrap(),
    search::P=AutoSearch()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Check if Tv's float base requires grid widening (not for Int types)
    # Int-based types (Complex{Int}) are handled by internal _value_type conversion
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = _to_float(x, Tg_new)
        return linear_interp(x_promoted, ys; extrap, search)
    end

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
    # Promote Tv to appropriate type based on Tg
    Tv_out = _value_type(Tv, Tg)
    y_mat = Matrix{Tv_out}(undef, n_pts, n_series_count)
    @inbounds for k in 1:n_series_count
        y_mat[:, k] .= Tv_out.(ys[k])
    end

    return LinearSeriesInterpolant(x, y_mat, extrap, search)
end

# Matrix input: columns as y-series
"""
    linear_interp(x, Y::AbstractMatrix; extrap=NoExtrap())

Create a multi-Y linear interpolant from a matrix where each column is a y-series.

# Arguments
- `x::AbstractVector`: x-coordinates (length n)
- `Y::AbstractMatrix`: n×m matrix, each column is a y-series
- `extrap`: Extrapolation mode

# Example
```julia
x = collect(range(0.0, 1.0, 101))
Y = hcat(sin.(2π .* x), cos.(2π .* x))  # 101×2 matrix

sitp = linear_interp(x, Y)

# Complex matrix also supported
Y_complex = hcat(exp.(2im * π * x), (1.0+2.0im) .* x)
sitp_complex = linear_interp(x, Y_complex)
```
"""
function linear_interp(
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:AbstractFloat, Tv}
    # Check if Tv's float base requires grid widening
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = _to_float(x, Tg_new)
        return linear_interp(x_promoted, Y; extrap, search)
    end

    n_pts = length(x)

    # Validate dimensions
    if size(Y, 1) != n_pts
        throw(DimensionMismatch(
            "Y has $(size(Y, 1)) rows but x has $n_pts points (expected n_points × n_series matrix)"
        ))
    end

    # Promote Tv to appropriate type based on Tg
    Tv_out = _value_type(Tv, Tg)
    y_mat = Tv_out === Tv ? copy(Y) : Tv_out.(Y)

    return LinearSeriesInterpolant(x, y_mat, extrap, search)
end

# ========================================
# Type Promotion Wrappers (Int, mixed types)
# ========================================
# POLICY: Tg is computed from x and real part of y element types

# Vector-of-vectors wrapper for non-AbstractFloat x
function linear_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv}
    # Compute promoted grid type (Tg may be Int, promotes to Float)
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    x_typed = _to_float(x, Tg_float)
    return linear_interp(x_typed, ys; extrap, search)
end

# Matrix wrapper for non-AbstractFloat x
function linear_interp(
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv}
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    x_typed = _to_float(x, Tg_float)
    return linear_interp(x_typed, Y; extrap, search)
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
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    xq::Tq; 
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    T_out = promote_type(Tv, Tq)  # Dual input → Dual output
    out = Vector{T_out}(undef, n_series(sitp))
    return sitp(out, xq; deriv=deriv, search=search, hint=hint)
end

"""
    (sitp::LinearSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate multi-Y interpolant at scalar query point (in-place).

# AD Support
Supports ForwardDiff.Dual input for automatic differentiation.
The anchor is built from primal value, but original xq is used for arithmetic.
"""
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    output::AbstractVector,  # Relaxed: allows Dual vector
    xq::Tq;
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    n_ser = n_series(sitp)

    # Validate output length
    _validate_scalar_output(output, n_ser)

    # Extract primal for anchor (search/comparison needs Float)
    xq_primal = _extract_primal(xq)
    xq_typed = Tg(xq_primal)

    # Build anchor from primal
    resolved = _resolve_search(search, xq)
    aq = _make_anchor(sitp, xq_typed, _to_searcher(resolved, hint))

    # Dispatch on derivative order with Dual-aware evaluation
    _eval_linear_series_point!(output, sitp, aq, xq, deriv)  # Pass original xq
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
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    xq::AbstractVector{Tq};
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    n_query = length(xq)
    n_ser = n_series(sitp)
    T_out = promote_type(Tv, Tq)  # Lossless: wider type to avoid precision loss

    # Explicit Vector{Vector{T_out}} for type stability on Julia LTS
    outputs = Vector{Vector{T_out}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{T_out}(undef, n_query)
    end
    # Delegate to in-place (unified path handles precision preservation)
    sitp(outputs, xq; deriv=deriv, search=search, hint=hint)

    return outputs
end

"""
    (sitp::LinearSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=EvalValue())

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points (any Real type, auto-promoted for search)
- `deriv`: Derivative order (0 or 1)

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
Uses task-local pool for anchor vector to achieve zero allocation after warmup.

# Precision Preservation
Uses pooled anchors with promoted type `promote_type(Tq, Tg)` to preserve precision in alpha.
Pool handles both same-type and mixed-type cases efficiently.
"""
@with_pool pool function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    outputs::AbstractVector{<:AbstractVector},
    xq::AbstractVector{Tq};
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    n_query = length(xq)
    n_ser = n_series(sitp)

    # Validate dimensions
    _validate_series_outputs(outputs, n_ser, n_query)

    # Build anchors - pool handles both same-type and mixed-type cases
    Tq_eff = promote_type(Tq, Tg)
    aq_vec = acquire!(pool, _LinearAnchoredQuery{Tg, Tq_eff}, n_query)
    resolved = _resolve_search(search, xq)
    _fill_anchors!(aq_vec, sitp.x, xq, Val(:linear); wrap=_should_wrap(sitp), searcher=_to_searcher(resolved, hint))

    # Extract matrices for argument-passing pattern
    y = sitp.y
    x_grid = sitp.x
    n_pts = n_points(sitp)
    extrap = sitp.extrap
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    # Evaluate all series - anchor already has correct alpha precision
    @inbounds for k in 1:n_ser
        _eval_linear_series_vector!(outputs[k], y, x_grid, n_pts, x_min, x_max, k, aq_vec, extrap, deriv)
    end
    return outputs
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance.

# Precision Preservation
The anchor's `alpha` field already contains precision-preserving normalized position
computed as `(xq - xL) / h` with the query's original precision.
"""
@inline function _eval_linear_series_vector!(
    out::AbstractVector,
    y::Matrix{Tv},
    x::AbstractVector{Tg},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    k::Int,
    aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}},
    extrap::AbstractExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    @inbounds for j in eachindex(out, aq_vec)
        out[j] = _eval_linear_series_with_extrap(y, x, n_pts, x_min, x_max, k, aq_vec[j], extrap, op)
    end
    return out
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
) where {Tg<:AbstractFloat, Tv}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_linear_series_anchored(y, x, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap isa ExtendExtrap || extrap isa WrapExtrap
        return _eval_linear_series_anchored(y, x, k, aq, op)
    elseif extrap isa ConstExtrap
        return _constant_extrap_boundary_value(y, aq.side, n_pts, k, op)
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
) where {Tg<:AbstractFloat, Tv}
    @inbounds begin
        yL = y[aq.idx, k]
        yR = y[aq.idx + 1, k]
    end
    return _linear_kernel(op, yL, yR, aq)
end
