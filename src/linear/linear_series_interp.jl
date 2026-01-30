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
mutable struct LinearSeriesInterpolant{Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (Range or Vector)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const extrap::ExtrapVal               # Extrapolation mode
    const search_policy::P                # Default search policy (immutable, thread-safe)

    function LinearSeriesInterpolant(
        x::X,
        y::Matrix{Tv},
        extrap::ExtrapVal,
        search::P=Binary()
    ) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}}
        new{Tg,Tv,P,X}(x, y, LazyTranspose{Tv}(), extrap, search)
    end
end

# ========================================
# Required Trait Implementations
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(sitp::LinearSeriesInterpolant) = sitp.extrap === Val(:wrap)

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

    _eval_linear_series_point_with_extrap!(output, y_point, sitp.x, n_pts, x_min, x_max, aq, sitp.extrap, op)
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

"""
    _eval_linear_series_point!(out, y_point, x, aq, op)

SIMD-optimized evaluation for point-contiguous layout (n_series × n_points).
Contiguous column access enables vectorization across series dimension.
"""
@inline function _eval_linear_series_point!(
    out::AbstractVector{Tv},
    y_point::Matrix{Tv},
    x::AbstractVector{Tg},
    aq::_LinearAnchoredQuery{Tg},
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    idx = aq.idx
    idx1 = idx + 1

    # Get interval data
    @inbounds begin
        xL = x[idx]
        xR = x[idx1]
    end
    h = xR - xL
    dL = aq.xq - xL

    # SIMD loop over series (contiguous column access)
    @inbounds @simd for k in axes(out, 1)
        yL = y_point[k, idx]
        yR = y_point[k, idx1]
        out[k] = _linear_kernel(op, yL, yR, h, dL)
    end

    return out
end

"""
    _eval_linear_series_point_with_extrap!(out, y_point, x, n_pts, x_min, x_max, aq, extrap, op)

SIMD evaluation with extrapolation handling for multi-series linear interpolation.
"""
@inline function _eval_linear_series_point_with_extrap!(
    out::AbstractVector{Tv},
    y_point::Matrix{Tv},
    x::AbstractVector{Tg},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_LinearAnchoredQuery{Tg},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_linear_series_point!(out, y_point, x, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    _eval_linear_series_point_extrap!(out, y_point, x, n_pts, x_min, x_max, aq, extrap, op, aq.side)
end

# :none - throw DomainError
@inline function _eval_linear_series_point_extrap!(
    ::AbstractVector{Tv},
    ::Matrix{Tv},
    ::AbstractVector{Tg},
    ::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_LinearAnchoredQuery{Tg},
    ::Val{:none},
    ::AbstractEvalOp,
    ::UInt8
) where {Tg<:AbstractFloat, Tv}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# :constant - clamp to boundary (value only, derivatives are zero)
@inline function _eval_linear_series_point_extrap!(
    out::AbstractVector{Tv},
    y_point::Matrix{Tv},
    ::AbstractVector{Tg},
    n_pts::Int,
    ::Tg,
    ::Tg,
    ::_LinearAnchoredQuery{Tg},
    ::Val{:constant},
    op::AbstractEvalOp,
    side::UInt8
) where {Tg<:AbstractFloat, Tv}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op)
end

# :extension - extend linear polynomial
@inline function _eval_linear_series_point_extrap!(
    out::AbstractVector{Tv},
    y_point::Matrix{Tv},
    x::AbstractVector{Tg},
    n_pts::Int,
    ::Tg,
    ::Tg,
    aq::_LinearAnchoredQuery{Tg},
    ::Val{:extension},
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
# Constructors
# ========================================

"""
    linear_interp(x, ys::AbstractVector{<:AbstractVector}; extrap=:none)

Create a multi-Y linear interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `extrap`: Extrapolation mode (:none, :constant, :extension, :wrap)

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
    extrap::Symbol=:none,
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Check if Tv's real part requires Tg promotion
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        # Promote Tg to match the wider value type
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = Tg_new.(x)
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

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    return LinearSeriesInterpolant(x, y_mat, extrap_val, search)
end

# Matrix input: columns as y-series
"""
    linear_interp(x, Y::AbstractMatrix; extrap=:none)

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
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    # Check if Tv's real part requires Tg promotion
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = Tg_new.(x)
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

    # Convert extrap symbol to Val
    extrap_val = _symbol_to_extrap_val(extrap)

    return LinearSeriesInterpolant(x, y_mat, extrap_val, search)
end

# ========================================
# Type Promotion Wrappers (Int, mixed types)
# ========================================
# POLICY: Tg is computed from x and real part of y element types

# Vector-of-vectors wrapper for non-AbstractFloat x
function linear_interp(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty}
    # Compute Tg from x and real part of y
    Tg = float(promote_type(Tx, _real_eltype(Ty)))
    x_typed = _to_float(x, Tg)
    return linear_interp(x_typed, ys; extrap, search)
end

# Matrix wrapper for non-AbstractFloat x
function linear_interp(
    x::AbstractVector{Tx},
    Y::AbstractMatrix{Ty};
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty}
    Tg = float(promote_type(Tx, _real_eltype(Ty)))
    x_typed = _to_float(x, Tg)
    return linear_interp(x_typed, Y; extrap, search)
end

# ========================================
# Scalar Evaluation (Override default for debugging)
# ========================================

"""
    (sitp::LinearSeriesInterpolant)(xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.
"""
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(xq::S; deriv::Int=0, search=sitp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, P, S<:Real}
    out = Vector{Tv}(undef, n_series(sitp))
    return sitp(out, xq; deriv=deriv, search=search, hint=hint)
end

"""
    (sitp::LinearSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    output::AbstractVector{Tv},
    xq::S;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, S<:Real}
    n_ser = n_series(sitp)

    # Validate output length
    _validate_scalar_output(output, n_ser)

    xq_typed = Tg(xq)

    # Build anchor
    aq = _make_anchor(sitp, xq_typed, _to_searcher(search, hint))

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
    (sitp::LinearSeriesInterpolant)(xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
"""
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    xq::AbstractVector{S};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, S<:Real}
    xq_typed = _to_float(xq, Tg)
    n_query = length(xq_typed)

    outputs = [Vector{Tv}(undef, n_query) for _ in 1:n_series(sitp)]
    sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)

    return outputs
end

"""
    (sitp::LinearSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points
- `deriv`: Derivative order (0 or 1)

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
Uses task-local pool for anchor vector to achieve zero allocation after warmup.
"""
@with_pool pool function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    outputs::AbstractVector{<:AbstractVector{Tv}},
    xq::AbstractVector{Tg};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P}
    n_query = length(xq)
    n_ser = n_series(sitp)

    # Validate dimensions
    _validate_series_outputs(outputs, n_ser, n_query)

    # Build anchors from pool (zero allocation after warmup)
    aq_vec = acquire!(pool, _LinearAnchoredQuery{Tg}, length(xq))
    _fill_anchors!(aq_vec, sitp.x, xq, Val(:linear); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    # Extract matrices for argument-passing pattern
    y = sitp.y
    x_grid = sitp.x
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    # Evaluate all series
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_linear_series_vector!(outputs[k], y, x_grid, n_pts, x_min, x_max, k, aq_vec, extrap, op)
        end
    end
    return outputs
end

# Real type wrapper for in-place vector
function (sitp::LinearSeriesInterpolant{Tg,Tv,P})(
    outputs::AbstractVector{<:AbstractVector{Tv}},
    xq::AbstractVector{S};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, S<:Real}
    xq_typed = _to_float(xq, Tg)
    return sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance.
"""
@inline function _eval_linear_series_vector!(
    out::AbstractVector{Tv},
    y::Matrix{Tv},
    x::AbstractVector{Tg},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    k::Int,
    aq_vec::AbstractVector{<:_LinearAnchoredQuery{Tg}},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    @inbounds for j in eachindex(out, aq_vec)
        out[j] = _eval_linear_series_with_extrap(y, x, n_pts, x_min, x_max, k, aq_vec[j], extrap, op)
    end
    return out
end

"""
Internal: Evaluate single series at single query point with extrapolation handling.
"""
@inline function _eval_linear_series_with_extrap(
    y::Matrix{Tv},
    x::AbstractVector{Tg},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    k::Int,
    aq::_LinearAnchoredQuery{Tg},
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_linear_series_anchored(y, x, k, aq, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap === Val(:extension) || extrap === Val(:wrap)
        return _eval_linear_series_anchored(y, x, k, aq, op)
    elseif extrap === Val(:constant)
        return _constant_extrap_boundary_value(y, aq.side, n_pts, k, op)
    else
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
end

"""
Internal: Core linear evaluation for series k at anchored query point.
"""
@inline function _eval_linear_series_anchored(
    y::Matrix{Tv},
    x::AbstractVector{Tg},
    k::Int,
    aq::_LinearAnchoredQuery{Tg},
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    idx = aq.idx
    @inbounds begin
        yL = y[idx, k]
        yR = y[idx + 1, k]
        xL = x[idx]
        xR = x[idx + 1]
    end
    h = xR - xL
    dL = aq.xq - xL
    return _linear_kernel(op, yL, yR, h, dL)
end
