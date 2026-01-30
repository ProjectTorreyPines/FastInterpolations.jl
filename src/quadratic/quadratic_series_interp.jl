# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    QUADRATIC SERIES INTERPOLATION                          ║
# ║         Multiple y-data series sharing the same x-grid                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Unified matrix storage for optimal performance.
# Key optimization: Adaptive layout with lazy transpose for scalar queries.
#
# Include order: ... → quadratic_anchor.jl → quadratic_series_interp.jl
#

# ========================================
# Type Definition
# ========================================

"""
    QuadraticSeriesInterpolant{T}

Multi-series quadratic spline interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `T`: Float type (Float32 or Float64)

# Fields
- `x::Vector{T}`: Grid points (sorted)
- `y::Matrix{T}`: Function values (n_points × n_series) series-contiguous
- `a::Matrix{T}`: Quadratic coefficients (n_points × n_series) series-contiguous
- `d::Matrix{T}`: Slope coefficients (n_points × n_series) series-contiguous
- `h::Vector{T}`: Grid spacing (shared across all series)
- `_transpose::LazyTransposeTriple{T}`: Lazy point-contiguous layout for SIMD
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

sitp = quadratic_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = sitp(0.5)            # Returns Vector{Float64} of length 3
sitp(output, 0.5)           # In-place

# Vector evaluation
vals = sitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
sitp([out1, out2, out3], xq)    # In-place (zero allocation)

# Derivatives
d1 = sitp(0.5; deriv=1)     # First derivatives
d2 = sitp(0.5; deriv=2)     # Second derivatives
```

# Performance
- Vector queries use series-contiguous layout directly
- Scalar queries trigger lazy transpose on first call
- All series share same h[] array (O(1) memory overhead)

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct QuadraticSeriesInterpolant{T<:AbstractFloat, P<:AbstractSearchPolicy} <: AbstractSeriesInterpolant{T, T}
    const x::Vector{T}                        # Grid points
    const y::Matrix{T}                        # Series-contiguous y (n_points × n_series)
    const a::Matrix{T}                        # Series-contiguous a (n_points × n_series)
    const d::Matrix{T}                        # Series-contiguous d (n_points × n_series)
    const h::Vector{T}                        # Grid spacing (shared)
    const _transpose::LazyTransposeTriple{T}  # Lazy point-contiguous layout
    const extrap::ExtrapVal                   # Extrapolation mode
    const search_policy::P                    # Default search policy

    function QuadraticSeriesInterpolant(
        x::Vector{T},
        y::Matrix{T},
        a::Matrix{T},
        d::Matrix{T},
        h::Vector{T},
        extrap::ExtrapVal,
        search::P=Binary()
    ) where {T<:AbstractFloat, P<:AbstractSearchPolicy}
        new{T,P}(x, y, a, d, h, LazyTransposeTriple{T}(), extrap, search)
    end
end

# ========================================
# Required Trait Implementations
# ========================================

"""Number of series in the interpolant."""
@inline n_series(sitp::QuadraticSeriesInterpolant) = size(sitp.y, 2)

"""Number of grid points."""
@inline n_points(sitp::QuadraticSeriesInterpolant) = size(sitp.y, 1)

"""Get the grid vector for anchor construction."""
@inline _get_grid(sitp::QuadraticSeriesInterpolant) = sitp.x

"""Get the extrapolation mode."""
@inline _get_extrap(sitp::QuadraticSeriesInterpolant) = sitp.extrap

"""Check if wrap mode is active."""
@inline _should_wrap(sitp::QuadraticSeriesInterpolant) = sitp.extrap === Val(:wrap)

"""Return the interpolation method kind for dispatch."""
@inline _method_kind(::Type{<:QuadraticSeriesInterpolant}) = Val(:quadratic)

# ========================================
# SIMD Evaluation Kernel
# ========================================

"""
    _eval_series_at_anchor!(output, sitp::QuadraticSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.
"""
@inline function _eval_series_at_anchor!(
    output::AbstractVector{T},
    sitp::QuadraticSeriesInterpolant{T},
    aq::_QuadraticAnchoredQuery{T},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    y_point, a_point, d_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = T(first(sitp.x)), T(last(sitp.x))

    _eval_quadratic_series_point_with_extrap!(output, y_point, a_point, d_point, n_pts, x_min, x_max, aq, sitp.extrap, op)
    return output
end

# ========================================
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::QuadraticSeriesInterpolant{T}) -> (y_point, a_point, d_point)

Ensure point-contiguous layout exists. Delegates to shared LazyTransposeTriple infrastructure.
"""
@inline function _ensure_point_layout!(sitp::QuadraticSeriesInterpolant{T}) where T
    return _ensure_transpose_triple!(sitp._transpose, sitp.y, sitp.a, sitp.d)
end

# ========================================
# SIMD Point Evaluation with Extrapolation
# ========================================

"""
    _eval_quadratic_series_point_with_extrap!(output, y_point, a_point, d_point, n_pts, x_min, x_max, aq, extrap, op)

SIMD kernel for evaluating all series at a single anchor point with extrapolation handling.
"""
@inline function _eval_quadratic_series_point_with_extrap!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    a_point::Matrix{T},
    d_point::Matrix{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val{:none},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    if aq.side != 0x00
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
    _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)
end

@inline function _eval_quadratic_series_point_with_extrap!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    a_point::Matrix{T},
    d_point::Matrix{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val{:constant},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    if aq.side != 0x00  # outside domain
        idx = _boundary_point_index(aq.side, n_pts)
        _fill_boundary_values!(output, y_point, idx, op)
    else
        _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)
    end
end

@inline function _eval_quadratic_series_point_with_extrap!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    a_point::Matrix{T},
    d_point::Matrix{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val,  # :extension, :wrap, or anything else
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)
end

# ========================================
# Quadratic Series Point Kernel
# ========================================

"""
    _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)

SIMD kernel for quadratic evaluation at a single anchor point.
Uses point-contiguous layout: y_point[:, idx] gives all series values at point idx.
"""
@inline function _eval_quadratic_series_point_kernel!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    a_point::Matrix{T},
    d_point::Matrix{T},
    aq::_QuadraticAnchoredQuery{T},
    op::EvalValue
) where {T<:AbstractFloat}
    idx = aq.idx
    dL = aq.dL

    @inbounds @simd for k in eachindex(output)
        y_k = y_point[k, idx]
        a_k = a_point[k, idx]
        d_k = d_point[k, idx]
        # Quadratic kernel: a*dL² + d*dL + y
        output[k] = muladd(muladd(a_k, dL, d_k), dL, y_k)
    end
    return output
end

@inline function _eval_quadratic_series_point_kernel!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    a_point::Matrix{T},
    d_point::Matrix{T},
    aq::_QuadraticAnchoredQuery{T},
    op::EvalDeriv1
) where {T<:AbstractFloat}
    idx = aq.idx
    dL = aq.dL

    @inbounds @simd for k in eachindex(output)
        a_k = a_point[k, idx]
        d_k = d_point[k, idx]
        # First derivative: 2*a*dL + d
        output[k] = muladd(T(2)*a_k, dL, d_k)
    end
    return output
end

@inline function _eval_quadratic_series_point_kernel!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    a_point::Matrix{T},
    d_point::Matrix{T},
    aq::_QuadraticAnchoredQuery{T},
    op::EvalDeriv2
) where {T<:AbstractFloat}
    idx = aq.idx

    @inbounds @simd for k in eachindex(output)
        a_k = a_point[k, idx]
        # Second derivative: 2*a (constant within interval)
        output[k] = T(2) * a_k
    end
    return output
end

# ========================================
# Boundary Value Helper
# ========================================

"""Fill output with boundary values for constant extrapolation."""
@inline function _fill_boundary_values!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    idx::Int,
    op::EvalValue
) where {T<:AbstractFloat}
    @inbounds @simd for k in eachindex(output)
        output[k] = y_point[k, idx]
    end
    return output
end

@inline function _fill_boundary_values!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    idx::Int,
    op::EvalDeriv1
) where {T<:AbstractFloat}
    fill!(output, zero(T))
    return output
end

@inline function _fill_boundary_values!(
    output::AbstractVector{T},
    y_point::Matrix{T},
    idx::Int,
    op::EvalDeriv2
) where {T<:AbstractFloat}
    fill!(output, zero(T))
    return output
end

# ========================================
# Constructors
# ========================================

"""
    quadratic_interp(x, ys::AbstractVector{<:AbstractVector}; bc=Left(QuadraticFit()), extrap=:none)

Create a multi-Y quadratic series interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `bc`: Boundary condition (Left/Right with QuadraticFit, Deriv1, Deriv2, MinCurvFit)
- `extrap`: Extrapolation mode (:none, :constant, :extension)

# Returns
`QuadraticSeriesInterpolant` object with unified matrix storage.

# Example
```julia
x = collect(range(0.0, 1.0, 101))
y1 = sin.(2π .* x)
y2 = cos.(2π .* x)
y3 = exp.(-x)

sitp = quadratic_interp(x, [y1, y2, y3])
vals = sitp(0.5)  # Returns [val1, val2, val3]
```
"""
function quadratic_interp(
    x::AbstractVector{T},
    ys::AbstractVector{<:AbstractVector{T}};
    bc::QuadraticBC{T}=Left(QuadraticFit{T}()),
    extrap::Symbol=:none,
    search::P=Binary()
) where {T<:AbstractFloat, P<:AbstractSearchPolicy}
    _validate_series_inputs(x, ys)

    n_pts = length(x)
    n_ser = length(ys)

    # Allocate matrices (n_points × n_series)
    y_mat = Matrix{T}(undef, n_pts, n_ser)
    a_mat = Matrix{T}(undef, n_pts, n_ser)  # Padded to n_pts for uniform size
    d_mat = Matrix{T}(undef, n_pts, n_ser)

    # Shared grid spacing (computed once)
    h = Vector{T}(undef, n_pts - 1)

    # Compute coefficients for each series
    for (k, y_k) in enumerate(ys)
        @inbounds for i in 1:n_pts
            y_mat[i, k] = y_k[i]
        end

        # Compute coefficients (h, d, a) for this series
        h_k, d_k, a_k = _compute_quadratic_coeffs(x, y_k, bc)

        # Store in matrices
        @inbounds for i in 1:n_pts
            d_mat[i, k] = d_k[i]
        end
        @inbounds for i in 1:(n_pts-1)
            a_mat[i, k] = a_k[i]
        end
        # Pad last row of a_mat with zero
        a_mat[n_pts, k] = zero(T)

        # Copy h (same for all series, but computed fresh - just use the last one)
        if k == 1
            copyto!(h, h_k)
        end
    end

    @_dispatch_extrap extrap => ev begin
        return QuadraticSeriesInterpolant(Vector{T}(x), y_mat, a_mat, d_mat, h, ev, search)
    end
end

# Matrix input: columns as y-series
"""
    quadratic_interp(x, Y::AbstractMatrix; bc=Left(QuadraticFit()), extrap=:none)

Create a multi-Y quadratic series interpolant from a matrix where each column is a y-series.

# Arguments
- `x::AbstractVector`: x-coordinates (length n)
- `Y::AbstractMatrix`: n×m matrix, each column is a y-series
- `bc`, `extrap`: Same as vector form

# Example
```julia
x = collect(range(0.0, 1.0, 101))
Y = hcat(sin.(2π .* x), cos.(2π .* x))  # 101×2 matrix

sitp = quadratic_interp(x, Y)
```
"""
function quadratic_interp(
    x::AbstractVector{T},
    Y::AbstractMatrix{T};
    bc::QuadraticBC{T}=Left(QuadraticFit{T}()),
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {T<:AbstractFloat}
    ys = [Y[:, k] for k in axes(Y, 2)]
    return quadratic_interp(x, ys; bc=bc, extrap=extrap, search=search)
end

# Real type wrappers (auto-promote to Float)
function quadratic_interp(
    x::AbstractVector{Tx},
    ys::AbstractVector{<:AbstractVector{Ty}};
    bc=Left(QuadraticFit()),
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    ys_float = [_to_float(y, T) for y in ys]
    bc_typed = _promote_bc(bc, T)
    return quadratic_interp(x_float, ys_float; bc=bc_typed, extrap=extrap, search=search)
end

function quadratic_interp(
    x::AbstractVector{Tx},
    Y::AbstractMatrix{Ty};
    bc=Left(QuadraticFit()),
    extrap::Symbol=:none,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty<:Real}
    T = promote_type(float(Tx), float(Ty))
    x_float = _to_float(x, T)
    Y_float = T.(Y)
    bc_typed = _promote_bc(bc, T)
    return quadratic_interp(x_float, Y_float; bc=bc_typed, extrap=extrap, search=search)
end

# ========================================
# Callable Interface (via Default + Override)
# ========================================

# Scalar evaluation (explicit implementation for deriv keyword support)
"""
    (sitp::QuadraticSeriesInterpolant)(xq::Real; deriv=0, search=Binary())

Evaluate all series at scalar query point (out-of-place).
"""
function (sitp::QuadraticSeriesInterpolant{T,P})(xq::S; deriv::Int=0, search=sitp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, P, S<:Real}
    xq_typed = T(xq)
    aq = _anchor_query(sitp.x, xq_typed, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    output = Vector{T}(undef, n_series(sitp))
    @_dispatch_deriv deriv => op begin
        _eval_series_at_anchor!(output, sitp, aq, op)
    end
    return output
end

"""
    (sitp::QuadraticSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=0, search=Binary())

Evaluate all series at scalar query point (in-place, zero allocation).
"""
function (sitp::QuadraticSeriesInterpolant{T,P})(
    output::AbstractVector{T},
    xq::S;
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P, S<:Real}
    _validate_scalar_output(output, n_series(sitp))

    xq_typed = T(xq)
    aq = _anchor_query(sitp.x, xq_typed, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    @_dispatch_deriv deriv => op begin
        _eval_series_at_anchor!(output, sitp, aq, op)
    end
    return output
end

# ========================================
# Vector Evaluation with Derivatives
# ========================================

"""
    (sitp::QuadraticSeriesInterpolant)(xq::AbstractVector; deriv=0)

Evaluate all series at multiple query points (out-of-place).
Returns a vector of vectors: one vector per y-series.
"""
function (sitp::QuadraticSeriesInterpolant{T,P})(
    xq::AbstractVector{S};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P, S<:Real}
    xq_typed = _to_float(xq, T)

    # Allocate outputs
    outputs = [Vector{T}(undef, length(xq_typed)) for _ in 1:n_series(sitp)]

    sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)
    return outputs
end

"""
    (sitp::QuadraticSeriesInterpolant)(outputs, xq::AbstractVector; deriv=0)

Evaluate all series at multiple query points (in-place, zero allocation).
"""
@with_pool pool function (sitp::QuadraticSeriesInterpolant{T,P})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{T};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P}
    _validate_series_outputs(outputs, n_series(sitp), length(xq))

    # Acquire anchor buffer from pool
    aq_vec = acquire!(pool, _QuadraticAnchoredQuery{T}, length(xq))
    _fill_anchors!(aq_vec, sitp.x, xq, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    @_dispatch_deriv deriv => op begin
        _eval_series_anchored!(outputs, sitp, aq_vec, op)
    end
    return outputs
end

# Real type wrapper for in-place vector
function (sitp::QuadraticSeriesInterpolant{T,P})(
    outputs::AbstractVector{<:AbstractVector{T}},
    xq::AbstractVector{S};
    deriv::Int=0,
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {T<:AbstractFloat, P, S<:Real}
    xq_typed = _to_float(xq, T)
    return sitp(outputs, xq_typed; deriv=deriv, search=search, hint=hint)
end

"""Evaluate all series using pre-built anchors."""
function _eval_series_anchored!(
    outputs::AbstractVector{<:AbstractVector{T}},
    sitp::QuadraticSeriesInterpolant{T},
    aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{T}},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    @inbounds for k in 1:n_series(sitp)
        y_col = view(sitp.y, :, k)
        a_col = view(sitp.a, :, k)
        d_col = view(sitp.d, :, k)
        output_k = outputs[k]

        for (j, aq) in enumerate(aq_vec)
            output_k[j] = _eval_single_quadratic_with_extrap(y_col, a_col, d_col, length(sitp.x),
                                                            T(first(sitp.x)), T(last(sitp.x)),
                                                            aq, sitp.extrap, op)
        end
    end
    return outputs
end

# ========================================
# Single-Series Evaluation Helpers
# ========================================

"""Evaluate single series at anchor with extrapolation handling."""
@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{T},
    a::AbstractVector{T},
    d::AbstractVector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val{:none},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    if aq.side != 0x00
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
    return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], aq.dL)
end

@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{T},
    a::AbstractVector{T},
    d::AbstractVector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val{:constant},
    op::EvalValue
) where {T<:AbstractFloat}
    if aq.side != 0x00  # outside domain
        idx = _boundary_point_index(aq.side, n_pts)
        return @inbounds y[idx]
    else
        return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], aq.dL)
    end
end

@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{T},
    a::AbstractVector{T},
    d::AbstractVector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val{:constant},
    op::Union{EvalDeriv1, EvalDeriv2}
) where {T<:AbstractFloat}
    if aq.side != 0x00  # outside domain
        return zero(T)
    else
        return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], aq.dL)
    end
end

@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{T},
    a::AbstractVector{T},
    d::AbstractVector{T},
    n_pts::Int,
    x_min::T,
    x_max::T,
    aq::_QuadraticAnchoredQuery{T},
    extrap::Val,  # :extension, :wrap, etc.
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], aq.dL)
end

# ========================================
# Pre-built Anchor Evaluation
# ========================================

"""
    (sitp::QuadraticSeriesInterpolant)(outputs, aq_vec::AbstractVector{<:_QuadraticAnchoredQuery}; deriv=0)

Evaluate with pre-built anchors (TRUE zero-allocation).
"""
function (sitp::QuadraticSeriesInterpolant{T})(
    outputs::AbstractVector{<:AbstractVector{T}},
    aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    _validate_series_outputs(outputs, n_series(sitp), length(aq_vec))

    @_dispatch_deriv deriv => op begin
        _eval_series_anchored!(outputs, sitp, aq_vec, op)
    end
    return outputs
end
