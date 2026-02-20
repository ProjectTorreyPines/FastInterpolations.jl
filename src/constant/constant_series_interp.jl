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
- `Tv`: Value type (Tg for real, Complex{Tg} for complex)
- `P`: Search policy type
- `X`: Grid container type (Vector or Range)

# Fields
- `x::X`: Shared x-grid (Vector or Range)
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `_transpose::LazyTranspose{Tv}`: Lazy point-contiguous layout for scalar SIMD
- `extrap::AbstractExtrap`: Extrapolation mode
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

# Complex values are also supported
y_complex = [exp.(2im * π * x), (1.0+2.0im) .* x]
sitp_complex = constant_interp(x, y_complex)
```

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct ConstantSeriesInterpolant{Tg<:AbstractFloat, Tv, E<:AbstractExtrap, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                            # Shared x-grid (Range or Vector)
    const y::Matrix{Tv}                   # Series-contiguous y (n_points × n_series)
    const _transpose::LazyTranspose{Tv}   # Lazy point-contiguous layout
    const extrap::E                        # Extrapolation mode (compile-time specialized)
    const side::SideVal                   # Side selection
    const search_policy::P                # Default search policy

    function ConstantSeriesInterpolant(
        x::X,
        y::Matrix{Tv},
        extrap::E,
        side::SideVal,
        search::P=Binary()
    ) where {Tg<:AbstractFloat, Tv, E<:AbstractExtrap, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}}
        new{Tg,Tv,E,P,X}(x, y, LazyTranspose{Tv}(), extrap, side, search)
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
@inline function _make_anchor(sitp::ConstantSeriesInterpolant{Tg}, xq::Tg, searcher::Searcher=DEFAULT_SEARCHER) where Tg
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
) where {Tg<:AbstractFloat, Tv}
    y_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    _eval_constant_series_point_extrap!(output, y_point, sitp.x, n_pts, x_min, x_max, aq, sitp.extrap, sitp.side, op, aq.side)
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
) where {Tg<:AbstractFloat, Tv}
    # Outside domain: delegate to extrapolation handler (trait method)
    if aq.side != 0x00
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
            @inbounds @simd for k in axes(output, 1)
                output[k] = zero(Tv)
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
    ::SideVal,
    ::AbstractEvalOp,
    ::UInt8
) where {Tg<:AbstractFloat, Tv}
    _throw_extrap_domain_error(aq.xq, x_min, x_max)
end

# ConstExtrap - clamp to boundary
@inline function _eval_constant_series_point_extrap!(
    out::AbstractVector{Tv},
    y_point::Matrix{Tv},
    ::AbstractVector{Tg},
    n_pts::Int,
    ::Tg,
    ::Tg,
    ::_ConstantAnchoredQuery{Tg},
    ::ConstExtrap,
    ::SideVal,
    op::AbstractEvalOp,
    side::UInt8
) where {Tg<:AbstractFloat, Tv}
    return _fill_constant_extrap_simd!(out, y_point, side, n_pts, op)
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
    side_val::SideVal,
    op::AbstractEvalOp,
    ::UInt8
) where {Tg<:AbstractFloat, Tv}
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

"""
    constant_interp(x, ys::AbstractVector{<:AbstractVector}; side=:nearest, extrap=NoExtrap())

Create a multi-Y constant interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `side`: Side for discontinuities (:left, :right, :nearest)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ConstExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`

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
# Hot path: x is AbstractFloat, ys elements can be Tg or Complex{Tg}
function constant_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    side::Symbol=:nearest,
    extrap::AbstractExtrap=NoExtrap(),
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Check if Tv's float base requires grid widening (not for Int types)
    # Int-based types (Complex{Int}) are handled by internal _value_type conversion
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = _to_float(x, Tg_new)
        return constant_interp(x_promoted, ys; side, extrap, search)
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

    @_dispatch_side side => side_val begin
        return ConstantSeriesInterpolant(x, y_mat, extrap, side_val, search)
    end
end

# Matrix input: columns as y-series
"""
    constant_interp(x, Y::AbstractMatrix; side=:nearest, extrap=NoExtrap())

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
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    side::Symbol=:nearest,
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    # Check if Tv's float base requires grid widening
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = _to_float(x, Tg_new)
        return constant_interp(x_promoted, Y; side, extrap, search)
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

    @_dispatch_side side => side_val begin
        return ConstantSeriesInterpolant(x, y_mat, extrap, side_val, search)
    end
end

# ========================================
# Type Promotion Wrappers (Int, mixed types)
# ========================================
# POLICY: Tg is computed from x and real part of y element types

# Vector-of-vectors wrapper for non-AbstractFloat x
function constant_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    side::Symbol=:nearest,
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv}
    # Compute promoted grid type (Tg may be Int, promotes to Float)
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    x_typed = _to_float(x, Tg_float)
    return constant_interp(x_typed, ys; side, extrap, search)
end

# Matrix wrapper for non-AbstractFloat x
function constant_interp(
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    side::Symbol=:nearest,
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=Binary()
) where {Tg<:Real, Tv}
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    x_typed = _to_float(x, Tg_float)
    return constant_interp(x_typed, Y; side, extrap, search)
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

# AD Support
When `xq` is a ForwardDiff.Dual, the output type is promoted to preserve
derivatives. Output type is `promote_type(Tv, Tq)`.
"""
function (sitp::ConstantSeriesInterpolant{Tg,Tv,P})(
    xq::Tq;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    T_out = promote_type(Tv, Tq)  # Lossless: wider type to avoid precision loss
    out = Vector{T_out}(undef, n_series(sitp))
    return sitp(out, xq; deriv=deriv, search=search, hint=hint)
end

"""
    (sitp::ConstantSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0, search=Binary())

Evaluate multi-Y interpolant at scalar query point (in-place).
"""
function (sitp::ConstantSeriesInterpolant{Tg,Tv,P})(
    output::AbstractVector,  # Relaxed: accepts any element type for lossless promotion
    xq::Tq;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    n_ser = n_series(sitp)

    # Validate output length
    _validate_scalar_output(output, n_ser)

    # AD Support: Extract primal for anchor building, pass original xq for AD
    xq_primal = _extract_primal(xq)
    xq_typed = Tg(xq_primal)

    # Build anchor using primal value
    aq = _make_anchor(sitp, xq_typed, _to_searcher(search, hint))

    # Dispatch on derivative order - pass original xq for AD support
    @_dispatch_deriv deriv => op begin
        _eval_constant_series_point!(output, sitp, aq, xq, op)
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
function (sitp::ConstantSeriesInterpolant{Tg,Tv,P})(
    xq::AbstractVector{Tq};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    xq_typed = _to_float(xq, Tg)
    n_query = length(xq_typed)
    n_ser = n_series(sitp)

    # Explicit Vector{Vector{Tv}} for type stability on Julia LTS
    outputs = Vector{Vector{Tv}}(undef, n_ser)
    @inbounds for k in 1:n_ser
        outputs[k] = Vector{Tv}(undef, n_query)
    end
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
@with_pool pool function (sitp::ConstantSeriesInterpolant{Tg,Tv,P})(
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
    aq_vec = acquire!(pool, _ConstantAnchoredQuery{Tg}, length(xq))
    _fill_anchors!(aq_vec, sitp.x, xq, Val(:constant); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    # Extract matrices for argument-passing pattern
    y = sitp.y
    x_grid = sitp.x
    n_pts = n_points(sitp)
    n = n_series(sitp)
    extrap = sitp.extrap
    side_val = sitp.side
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

    # Evaluate all series with derivative dispatch
    @_dispatch_deriv deriv => op begin
        @inbounds for k in 1:n
            _eval_constant_series_vector!(outputs[k], y, x_grid, n_pts, x_min, x_max, k, aq_vec, extrap, side_val, op)
        end
    end
    return outputs
end

# Real type wrapper for in-place vector
function (sitp::ConstantSeriesInterpolant{Tg,Tv,P})(
    outputs::AbstractVector{<:AbstractVector{Tv}},
    xq::AbstractVector{Tq};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    xq_typed = _to_float(xq, Tg)
    return sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)
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
    side_val::SideVal,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
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
    side_val::SideVal,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    # Special case: at right boundary (MUST be preserved!)
    if aq.xq == x_max
        if op isa EvalValue
            @inbounds return y[n_pts, k]
        else
            return zero(Tv)  # Derivatives of step function are zero
        end
    end

    # Inside domain: normal evaluation
    if aq.side == 0x00
        return _eval_constant_series_anchored(y, k, aq, side_val, op)
    end

    # Outside domain: dispatch on extrap mode
    if extrap isa ExtendExtrap || extrap isa WrapExtrap
        return _eval_constant_series_anchored(y, k, aq, side_val, op)
    elseif extrap isa ConstExtrap
        return _constant_extrap_boundary_value(y, aq.side, n_pts, k, op)
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
    side_val::SideVal,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    # Derivatives of constant (step) function are zero
    if !(op isa EvalValue)
        return zero(Tv)
    end

    idx = aq.idx
    @inbounds begin
        y_left = y[idx, k]
        y_right = y[idx + 1, k]
    end
    return _constant_kernel(EvalValue(), y_left, y_right, aq.h, aq.dL, side_val)
end
