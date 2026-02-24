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
    QuadraticSeriesInterpolant{Tg, Tv, E, P, X}

Multi-series quadratic spline interpolant with unified matrix storage and SIMD optimization.
Shares a single x-grid across N y-series for efficient batch evaluation.

# Type Parameters
- `Tg`: Grid type (Float32 or Float64)
- `Tv`: Value type (Tg for real, Complex{Tg} for complex)
- `E`: Extrapolation mode type (compile-time specialized)
- `P`: Search policy type
- `X`: Grid container type (Vector or Range)

# Fields
- `x::X`: Grid points (sorted, Vector or Range)
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `a::Matrix{Tv}`: Quadratic coefficients (n_points × n_series) series-contiguous
- `d::Matrix{Tv}`: Slope coefficients (n_points × n_series) series-contiguous
- `h::Vector{Tg}`: Grid spacing (shared across all series, always real)
- `_transpose::LazyTransposeTriple{Tv}`: Lazy point-contiguous layout for SIMD
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

sitp = quadratic_interp(x, [y1, y2, y3])

# Scalar evaluation
vals = sitp(0.5)            # Returns Vector{Float64} of length 3
sitp(output, 0.5)           # In-place

# Vector evaluation
vals = sitp([0.1, 0.5, 0.9])    # Returns Vector of Vectors
sitp([out1, out2, out3], xq)    # In-place (zero allocation)

# Derivatives
d1 = sitp(0.5; deriv=DerivOp(1))     # First derivatives
d2 = sitp(0.5; deriv=DerivOp(2))     # Second derivatives

# Complex values are also supported
y_complex = [exp.(2im * π * x), (1.0+2.0im) .* x]
sitp_complex = quadratic_interp(x, y_complex)
```

# Performance
- Vector queries use series-contiguous layout directly
- Scalar queries trigger lazy transpose on first call
- All series share same h[] array (O(1) memory overhead)

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct QuadraticSeriesInterpolant{Tg<:AbstractFloat, Tv, E<:AbstractExtrap, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                                # Grid points (Range or Vector)
    const y::Matrix{Tv}                       # Series-contiguous y (n_points × n_series)
    const a::Matrix{Tv}                       # Series-contiguous a (n_points × n_series)
    const d::Matrix{Tv}                       # Series-contiguous d (n_points × n_series)
    const h::Vector{Tg}                       # Grid spacing (shared, always real)
    const _transpose::LazyTransposeTriple{Tv} # Lazy point-contiguous layout
    const extrap::E                           # Extrapolation mode (compile-time specialized)
    const search_policy::P                    # Default search policy

    function QuadraticSeriesInterpolant(
        x::X,
        y::Matrix{Tv},
        a::Matrix{Tv},
        d::Matrix{Tv},
        h::Vector{Tg},
        extrap::E,
        search::P=AutoSearch()
    ) where {Tg<:AbstractFloat, Tv, E<:AbstractExtrap, P<:AbstractSearchPolicy, X<:AbstractVector{Tg}}
        new{Tg,Tv,E,P,X}(x, y, a, d, h, LazyTransposeTriple{Tv}(), extrap, search)
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
@inline _should_wrap(sitp::QuadraticSeriesInterpolant) = sitp.extrap isa WrapExtrap

"""Return the interpolation method kind for dispatch."""
@inline _method_kind(::Type{<:QuadraticSeriesInterpolant}) = Val(:quadratic)

# ========================================
# SIMD Evaluation Kernel
# ========================================

"""
    _eval_series_at_anchor!(output, sitp::QuadraticSeriesInterpolant, aq, op)

Evaluate all series at the given anchor point. Required trait for AbstractSeriesInterpolant.
Uses point-contiguous layout for SIMD optimization.

# AD Support
Anchor can contain ForwardDiff.Dual in `xq` and `dL` fields for AD propagation.
"""
@inline function _eval_series_at_anchor!(
    output::AbstractVector,
    sitp::QuadraticSeriesInterpolant{Tg, Tv},
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    y_point, a_point, d_point = _ensure_point_layout!(sitp)
    n_pts = n_points(sitp)
    x_min, x_max = Tg(first(sitp.x)), Tg(last(sitp.x))

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
    output::AbstractVector,
    y_point::Matrix{Tv},
    a_point::Matrix{Tv},
    d_point::Matrix{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    extrap::NoExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    if aq.side != 0x00
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
    _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)
end

@inline function _eval_quadratic_series_point_with_extrap!(
    output::AbstractVector,
    y_point::Matrix{Tv},
    a_point::Matrix{Tv},
    d_point::Matrix{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    extrap::ConstExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    if aq.side != 0x00  # outside domain
        idx = _boundary_point_index(aq.side, n_pts)
        _fill_boundary_values!(output, y_point, idx, op)
    else
        _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)
    end
end

@inline function _eval_quadratic_series_point_with_extrap!(
    output::AbstractVector,
    y_point::Matrix{Tv},
    a_point::Matrix{Tv},
    d_point::Matrix{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    extrap::AbstractExtrap,  # ExtendExtrap, WrapExtrap, or anything else
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)
end

# ========================================
# Quadratic Series Point Kernel
# ========================================

"""
    _eval_quadratic_series_point_kernel!(output, y_point, a_point, d_point, aq, op)

SIMD kernel for quadratic evaluation at a single anchor point.
Uses point-contiguous layout: y_point[:, idx] gives all series values at point idx.

# AD Support
When `aq.dL` is a ForwardDiff.Dual, the output will also be Dual.
"""
@inline function _eval_quadratic_series_point_kernel!(
    output::AbstractVector,
    y_point::Matrix{Tv},
    a_point::Matrix{Tv},
    d_point::Matrix{Tv},
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    op::EvalValue
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
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
    output::AbstractVector,
    y_point::Matrix{Tv},
    a_point::Matrix{Tv},
    d_point::Matrix{Tv},
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    op::EvalDeriv1
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    idx = aq.idx
    dL = aq.dL

    @inbounds @simd for k in eachindex(output)
        a_k = a_point[k, idx]
        d_k = d_point[k, idx]
        # First derivative: 2*a*dL + d
        output[k] = muladd(Tg(2)*a_k, dL, d_k)
    end
    return output
end

@inline function _eval_quadratic_series_point_kernel!(
    output::AbstractVector,
    y_point::Matrix{Tv},
    a_point::Matrix{Tv},
    d_point::Matrix{Tv},
    aq::_QuadraticAnchoredQuery{Tg, Tq},
    op::EvalDeriv2
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    idx = aq.idx

    @inbounds @simd for k in eachindex(output)
        a_k = a_point[k, idx]
        # Second derivative: 2*a (constant within interval)
        output[k] = Tg(2) * a_k
    end
    return output
end

# ========================================
# Boundary Value Helper
# ========================================

"""Fill output with boundary values for constant extrapolation."""
@inline function _fill_boundary_values!(
    output::AbstractVector{Tv},
    y_point::Matrix{Tv},
    idx::Int,
    op::EvalValue
) where {Tv}
    @inbounds @simd for k in eachindex(output)
        output[k] = y_point[k, idx]
    end
    return output
end

@inline function _fill_boundary_values!(
    output::AbstractVector{Tv},
    y_point::Matrix{Tv},
    idx::Int,
    op::EvalDeriv1
) where {Tv}
    fill!(output, zero(Tv))
    return output
end

@inline function _fill_boundary_values!(
    output::AbstractVector{Tv},
    y_point::Matrix{Tv},
    idx::Int,
    op::EvalDeriv2
) where {Tv}
    fill!(output, zero(Tv))
    return output
end

# ========================================
# Constructors
# ========================================

"""
    quadratic_interp(x, ys::AbstractVector{<:AbstractVector}; bc=Left(QuadraticFit()), extrap=NoExtrap())

Create a multi-Y quadratic series interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `ys`: Vector of y-value vectors (all same length as x)
- `bc`: Boundary condition (Left/Right with QuadraticFit, Deriv1, Deriv2, MinCurvFit)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ConstExtrap()`, or `ExtendExtrap()`

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
# Hot path: x is AbstractFloat, ys elements can be Tg or Complex{Tg}
function quadratic_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    search::P=AutoSearch()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Check if Tv's float base requires grid widening (not for Int types)
    # Int-based types (Complex{Int}) are handled by internal _value_type conversion
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        Tg_new = promote_type(Tg, Tv_real)
        x_promoted = _to_float(x, Tg_new)
        bc_typed = _promote_bc(bc, Tg_new)
        return quadratic_interp(x_promoted, ys; bc=bc_typed, extrap, search)
    end

    _validate_series_inputs(x, ys)

    n_pts = length(x)
    n_ser = length(ys)

    # Promote Tv to appropriate type based on Tg
    Tv_out = _value_type(Tv, Tg)

    # Allocate matrices (n_points × n_series)
    y_mat = Matrix{Tv_out}(undef, n_pts, n_ser)
    a_mat = Matrix{Tv_out}(undef, n_pts, n_ser)  # Padded to n_pts for uniform size
    d_mat = Matrix{Tv_out}(undef, n_pts, n_ser)

    # Shared grid spacing (computed once, always real)
    h = Vector{Tg}(undef, n_pts - 1)

    # Compute coefficients for each series
    for (k, y_k) in enumerate(ys)
        # Convert y values to output type
        y_typed = Tv_out.(y_k)
        @inbounds for i in 1:n_pts
            y_mat[i, k] = y_typed[i]
        end

        # Compute coefficients (h, d, a) for this series
        h_k, d_k, a_k = _compute_quadratic_coeffs(x, y_typed, bc)

        # Store in matrices
        @inbounds for i in 1:n_pts
            d_mat[i, k] = d_k[i]
        end
        @inbounds for i in 1:(n_pts-1)
            a_mat[i, k] = a_k[i]
        end
        # Pad last row of a_mat with zero
        a_mat[n_pts, k] = zero(Tv_out)

        # Copy h (same for all series, but computed fresh - just use the last one)
        if k == 1
            copyto!(h, h_k)
        end
    end

    return QuadraticSeriesInterpolant(x, y_mat, a_mat, d_mat, h, extrap, search)
end

# Matrix input: columns as y-series
"""
    quadratic_interp(x, Y::AbstractMatrix; bc=Left(QuadraticFit()), extrap=NoExtrap())

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
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:AbstractFloat, Tv}
    ys = [Y[:, k] for k in axes(Y, 2)]
    return quadratic_interp(x, ys; bc=bc, extrap=extrap, search=search)
end

# ========================================
# Type Promotion Wrappers (Int, mixed types)
# ========================================
# POLICY: Tg is computed from x and real part of y element types

# Vector-of-vectors wrapper for non-AbstractFloat x
function quadratic_interp(
    x::AbstractVector{Tg},
    ys::AbstractVector{<:AbstractVector{Tv}};
    bc=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv}
    # Compute promoted grid type (Tg may be Int, promotes to Float)
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    x_typed = _to_float(x, Tg_float)
    bc_typed = _promote_bc(bc, Tg_float)
    return quadratic_interp(x_typed, ys; bc=bc_typed, extrap, search)
end

# Matrix wrapper for non-AbstractFloat x
function quadratic_interp(
    x::AbstractVector{Tg},
    Y::AbstractMatrix{Tv};
    bc=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {Tg<:Real, Tv}
    Tg_float = float(promote_type(Tg, _real_eltype(Tv)))
    x_typed = _to_float(x, Tg_float)
    bc_typed = _promote_bc(bc, Tg_float)
    return quadratic_interp(x_typed, Y; bc=bc_typed, extrap, search)
end

# ========================================
# Callable Interface (via Default + Override)
# ========================================

# Scalar evaluation (explicit implementation for deriv keyword support)
"""
    (sitp::QuadraticSeriesInterpolant)(xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate all series at scalar query point (out-of-place).

# AD Support
Supports ForwardDiff.Dual input: output type is promoted to include Dual.
The anchor preserves the Dual type in `xq` and `dL` fields for AD propagation.
"""
function (sitp::QuadraticSeriesInterpolant{Tg,Tv,P})(
    xq::Tq;
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual (no-op for Float/Float-backed Dual)
    xq_promoted = _promote_for_anchor(xq, Tg)
    T_out = promote_type(Tv, typeof(xq_promoted))
    aq = _anchor_query(sitp.x, xq_promoted, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    output = Vector{T_out}(undef, n_series(sitp))
    _eval_series_at_anchor!(output, sitp, aq, deriv)
    return output
end

"""
    (sitp::QuadraticSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate all series at scalar query point (in-place, zero allocation).

# AD Support
Supports ForwardDiff.Dual input for automatic differentiation.
The anchor preserves the Dual type in `xq` and `dL` fields for AD propagation.
"""
function (sitp::QuadraticSeriesInterpolant{Tg,Tv,P})(
    output::AbstractVector,  # Relaxed: allows Dual vector
    xq::Tq;
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    _validate_scalar_output(output, n_series(sitp))

    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual
    xq_promoted = _promote_for_anchor(xq, Tg)

    aq = _anchor_query(sitp.x, xq_promoted, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))

    _eval_series_at_anchor!(output, sitp, aq, deriv)
    return output
end

# ========================================
# Vector Evaluation with Derivatives
# ========================================

"""
    (sitp::QuadraticSeriesInterpolant)(xq::AbstractVector; deriv=EvalValue())

Evaluate all series at multiple query points (out-of-place).
Returns a vector of vectors: one vector per y-series.
Output type is promoted to wider type for precision preservation.
"""
function (sitp::QuadraticSeriesInterpolant{Tg,Tv,P})(
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
    (sitp::QuadraticSeriesInterpolant)(outputs, xq::AbstractVector; deriv=EvalValue())

Evaluate all series at multiple query points (in-place, zero allocation when types match).

# Precision Preservation
Uses pooled anchors with promoted type `promote_type(Tq, Tg)` to preserve precision.
Pool handles both same-type and mixed-type cases efficiently.
"""
@with_pool pool function (sitp::QuadraticSeriesInterpolant{Tg,Tv,P})(
    outputs::AbstractVector{<:AbstractVector},
    xq::AbstractVector{Tq};
    deriv::DerivOp=EvalValue(),
    search=sitp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv, P, Tq<:Real}
    n_query = length(xq)
    _validate_series_outputs(outputs, n_series(sitp), n_query)

    # Build anchors - pool handles both same-type and mixed-type cases
    Tq_eff = promote_type(Tq, Tg)
    aq_vec = acquire!(pool, _QuadraticAnchoredQuery{Tg, Tq_eff}, n_query)
    if Tq === Tg
        _fill_anchors!(aq_vec, sitp.x, xq, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))
    else
        # Mixed type: convert query points to preserve precision
        xq_promoted = _promote_for_anchor.(xq, Tg)
        _fill_anchors!(aq_vec, sitp.x, xq_promoted, Val(:quadratic); wrap=_should_wrap(sitp), searcher=_to_searcher(search, hint))
    end

    # Evaluate all series - anchor already has correct dL precision
    _eval_series_anchored!(outputs, sitp, aq_vec, deriv)
    return outputs
end

"""
Evaluate all series using pre-built anchors.

The anchor's `dL` field already has the correct precision (via `_promote_for_anchor`).
"""
function _eval_series_anchored!(
    outputs::AbstractVector{<:AbstractVector},
    sitp::QuadraticSeriesInterpolant{Tg,Tv},
    aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{Tg}},
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    x_grid = sitp.x
    n_pts = length(x_grid)
    x_min, x_max = Tg(first(x_grid)), Tg(last(x_grid))

    @inbounds for k in 1:n_series(sitp)
        y_col = view(sitp.y, :, k)
        a_col = view(sitp.a, :, k)
        d_col = view(sitp.d, :, k)
        output_k = outputs[k]

        for (j, aq) in enumerate(aq_vec)
            output_k[j] = _eval_single_quadratic_with_extrap(y_col, a_col, d_col, n_pts, x_min, x_max, aq, aq.dL, sitp.extrap, op)
        end
    end
    return outputs
end

# ========================================
# Single-Series Evaluation Helpers
# ========================================

"""
Evaluate single series at anchor with extrapolation handling.

# Precision Preservation
Takes `dL` as a parameter to allow caller to compute with original xq precision.
"""
@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Taq},
    dL::Tq,  # Passed by caller for precision control
    extrap::NoExtrap,
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Taq<:Real, Tq<:Real}
    if aq.side != 0x00
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
    return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
end

@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Taq},
    dL::Tq,
    extrap::ConstExtrap,
    op::EvalValue
) where {Tg<:AbstractFloat, Tv, Taq<:Real, Tq<:Real}
    if aq.side != 0x00  # outside domain
        idx = _boundary_point_index(aq.side, n_pts)
        return @inbounds y[idx]
    else
        return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
    end
end

@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Taq},
    dL::Tq,
    extrap::ConstExtrap,
    op::Union{EvalDeriv1, EvalDeriv2}
) where {Tg<:AbstractFloat, Tv, Taq<:Real, Tq<:Real}
    if aq.side != 0x00  # outside domain
        return zero(Tv)
    else
        return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
    end
end

@inline function _eval_single_quadratic_with_extrap(
    y::AbstractVector{Tv},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    n_pts::Int,
    x_min::Tg,
    x_max::Tg,
    aq::_QuadraticAnchoredQuery{Tg, Taq},
    dL::Tq,
    extrap::AbstractExtrap,  # ExtendExtrap, WrapExtrap, etc.
    op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv, Taq<:Real, Tq<:Real}
    return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
end

# ========================================
# Pre-built Anchor Evaluation
# ========================================

"""
    (sitp::QuadraticSeriesInterpolant)(outputs, aq_vec::AbstractVector{<:_QuadraticAnchoredQuery}; deriv=EvalValue())

Evaluate with pre-built anchors (TRUE zero-allocation).
"""
function (sitp::QuadraticSeriesInterpolant{Tg,Tv})(
    outputs::AbstractVector{<:AbstractVector{Tv}},
    aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{Tg, Tq}};
    deriv::DerivOp=EvalValue()
) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    _validate_series_outputs(outputs, n_series(sitp), length(aq_vec))

    _eval_series_anchored!(outputs, sitp, aq_vec, deriv)
    return outputs
end
