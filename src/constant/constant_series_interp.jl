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
    ConstantSeriesInterpolant{T}

Multi-series constant (step) interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `T`: Float type (Float32 or Float64)

# Fields
- `x::Vector{T}`: Shared x-grid
- `y::Matrix{T}`: Function values (n_points × n_series) series-contiguous
- `_transpose::LazyTranspose{T}`: Lazy point-contiguous layout for scalar SIMD
- `extrap::ExtrapVal`: Extrapolation mode
- `side::SideVal`: Side selection (:nearest, :left, :right)

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
```

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct ConstantSeriesInterpolant{T<:AbstractFloat, P<:AbstractSearchPolicy} <: AbstractSeriesInterpolant{T, T}
    const x::Vector{T}                    # Shared x-grid
    const y::Matrix{T}                    # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{T}    # Lazy point-contiguous layout
    const extrap::ExtrapVal               # Extrapolation mode
    const side::SideVal                   # Side selection
    const search_policy::P                # Default search policy

    function ConstantSeriesInterpolant(
        x::Vector{T},
        y::Matrix{T},
        extrap::ExtrapVal,
        side::SideVal,
        search::P=Binary()
    ) where {T<:AbstractFloat, P<:AbstractSearchPolicy}
        new{T,P}(x, y, LazyTranspose{T}(), extrap, side, search)
    end
end

# ========================================
# Required Trait Implementations
# ========================================

"""Check if wrap mode is active (for anchor construction)."""
@inline _should_wrap(sitp::ConstantSeriesInterpolant) = sitp.extrap === Val(:wrap)

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
    _make_anchor(sitp::ConstantSeriesInterpolant, xq::T, searcher) -> _ConstantAnchoredQuery{T}

Build anchor for a query point. Required trait for AbstractSeriesInterpolant.
"""
@inline function _make_anchor(sitp::ConstantSeriesInterpolant{T}, xq::T, searcher::Searcher=DEFAULT_SEARCHER) where T
    return _constant_anchor_query_impl(sitp.x, xq, _should_wrap(sitp), searcher)
end

"""
    _eval_series_at_anchor!(output, sitp::ConstantSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.
"""
@inline function _eval_series_at_anchor!(
    output::AbstractVector{T},
    sitp::ConstantSeriesInterpolant{T},
    aq::_ConstantAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    y_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = T(first(sitp.x)), T(last(sitp.x))

    _eval_constant_series_point_with_extrap!(output, y_point, sitp.x, n_pts, x_min, x_max, aq, sitp.extrap, sitp.side, op)
    return output
end

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::ConstantSeriesInterpolant{T}) -> y_point

Ensure point-contiguous layout exists. Delegates to shared LazyTranspose infrastructure.
"""
@inline function _ensure_point_layout!(sitp::ConstantSeriesInterpolant{T}) where T
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
    _eval_constant_series_point!(out, y_point, x, aq, side_val, op)

SIMD-optimized evaluation for point-contiguous layout (n_series × n_points).
Contiguous column access enables vectorization across series dimension.
"""
@inline function _eval_constant_series_point!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    x::Vector{T},
    aq::_ConstantAnchoredQuery{T},
    side_val::SideVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    idx = aq.idx
    idx1 = idx + 1

    # Get interval data
    h = aq.h
    dL = aq.dL

    # SIMD loop over series (contiguous column access)
    @inbounds @simd for k in axes(out, 1)
        y_left = y_point[k, idx]
        y_right = y_point[k, idx1]
        out[k] = _constant_kernel(op, y_left, y_right, h, dL, side_val)
    end

    return out
end

"""
    _eval_constant_series_point_with_extrap!(out, y_point, x, n_pts, x_min, x_max, aq, extrap, side_val, op)

SIMD evaluation with extrapolation handling for multi-series constant interpolation.
"""
@inline function _eval_constant_series_point_with_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    x::Vector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_ConstantAnchoredQuery{T},
    extrap::ExtrapVal,
    side_val::SideVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Special case: at right boundary (x_max)
    if aq.xq == x_max
        if op isa EvalValue
            @inbounds @simd for k in axes(out, 1)
                out[k] = y_point[k, n_pts]
            end
        else
            @inbounds @simd for k in axes(out, 1)
                out[k] = zero(T)
            end
        end
        return out
    end

    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_constant_series_point!(out, y_point, x, aq, side_val, op)
    end

    # Outside domain: dispatch on extrap mode
    _eval_constant_series_point_extrap!(out, y_point, x, n_pts, x_min, x_max, aq, extrap, side_val, op, aq.side)
end

# :none - throw DomainError
@inline function _eval_constant_series_point_extrap!(
    ::AbstractVector{T},
    ::Matrix{T},
    ::Vector{T},
    ::Int,
    x_min::T,
    x_max::T,
    aq::_ConstantAnchoredQuery{T},
    ::Val{:none},
    ::SideVal,
    ::AbstractEvalOp,
    ::UInt8
) where {T<:AbstractFloat}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# :constant - clamp to boundary
@inline function _eval_constant_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    ::Vector{T},
    n_pts::Int,
    ::T,
    ::T,
    ::_ConstantAnchoredQuery{T},
    ::Val{:constant},
    ::SideVal,
    op::AbstractEvalOp,
    side::UInt8
) where {T<:AbstractFloat}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op)
end

# :extension - extend using same constant value
@inline function _eval_constant_series_point_extrap!(
    out::AbstractVector{T},
    y_point::Matrix{T},
    x::Vector{T},
    n_pts::Int,
    ::T,
    ::T,
    aq::_ConstantAnchoredQuery{T},
    ::Val{:extension},
    side_val::SideVal,
    op::AbstractEvalOp,
    side::UInt8
) where {T<:AbstractFloat}
    # Use boundary interval for extension
    return _eval_constant_series_point!(out, y_point, x, aq, side_val, op)
end


# ========================================
# Constructors
# ========================================

"""
    constant_interp(x, ys::AbstractVector{<:AbstractVector}; side=:nearest, extrap=:none)

Create a multi-Y constant interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `side`: Side for discontinuities (:left, :right, :nearest)
- `extrap`: Extrapolation mode (:none, :constant, :extension, :wrap)

# Returns
`ConstantSeriesInterpolant` object with matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

sitp = constant_interp(x, [y1, y2, y3])
vals = sitp(0.5)
```
"""
function constant_interp(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    side::Symbol=:nearest,
    extrap::Symbol=:none,
    search::P=Binary()
) where {T<:AbstractFloat, P<:AbstractSearchPolicy}
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

    # Convert symbols to Val types
    extrap_val = _symbol_to_extrap_val(extrap)

    @_dispatch_side side => side_val begin
        # Copy x to ensure ownership
        x_vec = collect(x)
        return ConstantSeriesInterpolant(x_vec, y_mat, extrap_val, side_val, search)
    end
end

# Matrix input: columns as y-series
"""
    constant_interp(x, Y::AbstractMatrix; side=:nearest, extrap=:none)

Create a multi-Y constant interpolant from a matrix where each column is a y-series.

# Arguments
- `x::AbstractVector`: x-coordinates (length n)
- `Y::AbstractMatrix`: n×m matrix, each column is a y-series
- `side`, `extrap`: Same as vector form

# Example
```julia
x = collect(range(0.0, 1.0, 101))
Y = hcat(sin.(2π .* x), cos.(2π .* x))  # 101×2 matrix

sitp = constant_interp(x, Y)
```
"""
function constant_interp(
    x::AbstractVector{T},
    Y::AbstractMatrix{T};
    side::Symbol=:nearest,
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {T<:AbstractFloat}
    n_pts = length(x)

    # Validate dimensions
    if size(Y, 1) != n_pts
        throw(DimensionMismatch(
            "Y has $(size(Y, 1)) rows but x has $n_pts points (expected n_points × n_series matrix)"
        ))
    end

    # Copy to ensure ownership
    y_mat = copy(Y)
    x_vec = collect(x)

    # Convert symbols to Val types
    extrap_val = _symbol_to_extrap_val(extrap)

    @_dispatch_side side => side_val begin
        return ConstantSeriesInterpolant(x_vec, y_mat, extrap_val, side_val, search)
    end
end

# Real type wrappers (auto-promote to Float)
function constant_interp(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    side::Symbol=:nearest,
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    ys_float = [_to_float(y, T) for y in ys]
    return constant_interp(x_float, ys_float; side=side, extrap=extrap, search=search)
end

function constant_interp(
    x::AbstractVector{Tx},
    Y::AbstractMatrix{Ty};
    side::Symbol=:nearest,
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    Y_float = T.(Y)
    return constant_interp(x_float, Y_float; side=side, extrap=extrap, search=search)
end

# ========================================
# Scalar Evaluation
# ========================================

"""
    (sitp::ConstantSeriesInterpolant)(xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (out-of-place).

Returns a vector of values, one per y-series.

# Derivative support
- `deriv=0`: Returns function values
- `deriv=1,2`: Returns zeros (step function derivative is zero everywhere)
"""
function (sitp::ConstantSeriesInterpolant{T,P})(xq::S; deriv::Int=0, search=sitp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, P, S<:Real}
    out = Vector{T}(undef, n_series(sitp))
    return sitp(out, xq; deriv=deriv, search=search, hint=hint)
end

"""
    (sitp::ConstantSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (sitp::ConstantSeriesInterpolant{T,P})(
    output::AbstractVector{T},
    xq::S;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P, S<:Real}
    n_ser = n_series(sitp)

    # Validate output length
    _validate_scalar_output(output, n_ser)

    xq_typed = T(xq)

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
    (sitp::ConstantSeriesInterpolant)(xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (out-of-place).

Returns a vector of vectors: one vector per y-series, each containing results for all query points.
"""
function (sitp::ConstantSeriesInterpolant{T,P})(
    xq::AbstractVector{S};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P, S<:Real}
    xq_typed = _to_float(xq, T)
    n_query = length(xq_typed)

    outputs = [Vector{T}(undef, n_query) for _ in 1:n_series(sitp)]
    sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)

    return outputs
end

"""
    (sitp::ConstantSeriesInterpolant)(outputs::AbstractVector{<:AbstractVector}, xq::AbstractVector; deriv=0)

Evaluate multi-Y interpolant at multiple query points (in-place, zero allocation).

# Arguments
- `outputs`: Vector of pre-allocated output buffers (one per y-series)
- `xq`: Query points
- `deriv`: Derivative order (0, 1, or 2)

This is the KILLER FEATURE: zero-allocation batch evaluation for hot loops.
Uses task-local pool for anchor vector to achieve zero allocation after warmup.
"""
@with_pool pool function (sitp::ConstantSeriesInterpolant{T,P})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P}
    n_query = length(xq)
    n_ser = n_series(sitp)

    # Validate dimensions
    _validate_series_outputs(outputs, n_ser, n_query)

    # Build anchors from pool (zero allocation after warmup)
    aq_vec = acquire!(pool, _ConstantAnchoredQuery{T}, length(xq))
    _fill_anchors!(aq_vec, sitp.x, xq, Val(:constant); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    # Extract matrices for argument-passing pattern
    y = sitp.y
    x_grid = sitp.x
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    side_val = sitp.side
    x_min, x_max = T(first(sitp.x)), T(last(sitp.x))

    # Evaluate all series with derivative dispatch
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_constant_series_vector!(outputs[k], y, x_grid, n_pts, x_min, x_max, k, aq_vec, extrap, side_val, op)
        end
    end
    return outputs
end

# Real type wrapper for in-place vector
function (sitp::ConstantSeriesInterpolant{T,P})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P, S<:Real}
    xq_typed = _to_float(xq, T)
    return sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)
end

"""
Internal: Evaluate a single series for vector of query points.
Uses argument-passing pattern for optimal performance.
"""
@inline function _eval_constant_series_vector!(
    out::AbstractVector{T},
    y::Matrix{T},
    x::Vector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    k::Int,
    aq_vec::AbstractVector{<:_ConstantAnchoredQuery{T}},
    extrap::ExtrapVal,
    side_val::SideVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    @inbounds for j in eachindex(out, aq_vec)
        out[j] = _eval_constant_series_with_extrap(y, x, n_pts, x_min, x_max, k, aq_vec[j], extrap, side_val, op)
    end
    return out
end

"""
Internal: Evaluate single series at single query point with extrapolation handling.
"""
@inline function _eval_constant_series_with_extrap(
    y::Matrix{T},
    x::Vector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    k::Int,
    aq::_ConstantAnchoredQuery{T},
    extrap::ExtrapVal,
    side_val::SideVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Special case: at right boundary (MUST be preserved!)
    if aq.xq == x_max
        if op isa EvalValue
            @inbounds return y[n_pts, k]
        else
            return zero(T)  # Derivatives of step function are zero
        end
    end

    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_constant_series_anchored(y, k, aq, side_val, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap === Val(:extension) || extrap === Val(:wrap)
        return _eval_constant_series_anchored(y, k, aq, side_val, op)
    elseif extrap === Val(:constant)
        return _constant_extrap_boundary_value(y, aq.side, n_pts, k, op)
    else
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
end

"""
Internal: Core constant evaluation for series k at anchored query point.
"""
@inline function _eval_constant_series_anchored(
    y::Matrix{T},
    k::Int,
    aq::_ConstantAnchoredQuery{T},
    side_val::SideVal,
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    # Derivatives of constant (step) function are zero
    if !(op isa EvalValue)
        return zero(T)
    end

    idx = aq.idx
    @inbounds begin
        y_left = y[idx, k]
        y_right = y[idx + 1, k]
    end
    return _constant_kernel(EvalValue(), y_left, y_right, aq.h, aq.dL, side_val)
end
