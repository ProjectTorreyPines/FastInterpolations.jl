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
- `Tg`: Grid type (unconstrained — supports duck types like ForwardDiff.Dual)
- `Tv`: Value type (unconstrained)
- `E`: Extrapolation mode type (compile-time specialized)
- `P`: Search policy type
- `X`: Grid container type (Vector or Range)

# Fields
- `x::X`: Grid points (sorted, Vector or Range)
- `y::Matrix{Tv}`: Function values (n_points × n_series) series-contiguous
- `a::Matrix{Tv}`: Quadratic coefficients (n_points × n_series) series-contiguous
- `d::Matrix{Tv}`: Slope coefficients (n_points × n_series) series-contiguous
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
- All series share same grid spacing (O(1) memory overhead)

# Implementation Note: `mutable struct` with `const` fields
This type uses `mutable struct` with all `const` fields (Julia 1.8+) instead of
plain `struct` for performance reasons. See CubicSeriesInterpolant for details.
"""
mutable struct QuadraticSeriesInterpolant{Tg, Tv, E <: AbstractExtrap, P <: AbstractSearchPolicy, X <: AbstractVector{Tg}, Tc} <: AbstractSeriesInterpolant{Tg, Tv}
    const x::X                                 # Wrapped grid (`_CachedRange`/`_CachedVector` carrying cached `h`/`inv_h`)
    const y::Matrix{Tv}                        # Series-contiguous y (n_points × n_series)
    const a::Matrix{Tc}                        # Series-contiguous a: Tc = _promote_eltype(_coeff_op, Tg, Tv)
    const d::Matrix{Tc}                        # Series-contiguous d: Tc = _promote_eltype(_coeff_op, Tg, Tv)
    const _transpose::LazyTransposeTriple{Tv, Tc} # Lazy point-contiguous layout
    const extrap::E                            # Extrapolation mode (compile-time specialized)
    const search_policy::P                     # Default search policy

    # `_cache_axis` (insurance) + `_convert_copy` (ownership).
    # y/a/d are freshly built by the outer factory's solver loop.
    function QuadraticSeriesInterpolant(
            x::AbstractVector{Tg},
            y::Matrix{Tv},
            a::Matrix,
            d::Matrix,
            extrap::E,
            search::P = AutoSearch();
            bc::AbstractBC = NoBC()
        ) where {Tg, Tv, E <: AbstractExtrap, P <: AbstractSearchPolicy}
        Tc = eltype(a)
        xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
        return new{Tg, Tv, E, P, typeof(xc), Tc}(xc, y, a, d, LazyTransposeTriple{Tv, Tc}(), extrap, search)
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
# Lazy Point-Layout Management
# ========================================

"""
    _ensure_point_layout!(sitp::QuadraticSeriesInterpolant{T}) -> (y_point, a_point, d_point)

Ensure point-contiguous layout exists. Delegates to shared LazyTransposeTriple infrastructure.
"""
@inline function _ensure_point_layout!(sitp::QuadraticSeriesInterpolant{T}) where {T}
    return _ensure_transpose_triple!(sitp._transpose, sitp.y, sitp.a, sitp.d)
end

# ========================================
# Constructors
# ========================================

# ========================================
# Series Constructor (canonical entry point)
# ========================================

"""
    quadratic_interp(x, Series(y1, y2, ...); bc=Left(QuadraticFit()), extrap=NoExtrap(), search=AutoSearch())
    quadratic_interp(x, Series([y1, y2, ...]); ...)
    quadratic_interp(x, Series(Y::AbstractMatrix); ...)

Create a multi-Y quadratic series interpolant for multiple y-data series sharing the same x-grid.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `s::Series`: Wrapped series data (varargs, vector-of-vectors, or matrix)
- `bc`: Boundary condition (Left/Right with QuadraticFit, Deriv1, Deriv2, MinCurvFit)
- `extrap::AbstractExtrap`: `NoExtrap()`, `ClampExtrap()`, or `ExtendExtrap()`
- `search::AbstractSearchPolicy`: Search policy for interval lookup

# Example
```julia
x = collect(range(0.0, 1.0, 101))
sitp = quadratic_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))
```
"""
function quadratic_interp(
        x::AbstractVector{Tg},
        s::Series;
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg}
    # Type promotion: widen grid if y's float base is wider than Tg
    Tv = _series_eltype(s)
    Tg_new = _promote_grid_float(Tg, Tv)
    if Tg_new !== Tg
        return quadratic_interp(_to_float(x, Tg_new), s; bc = _normalize_bc(bc), extrap, search)
    end

    # Caching wrap (zero-copy of buffer): Range → `_CachedRange{Tg}`,
    # Vector → `_CachedVector{Tg, Tinv}`. Inner ctor's `_convert_copy` takes
    # ownership.
    x = _cache_axis(x, NoBC(), Tg)

    n_pts = length(x)
    Tv_out = _value_type(Tv, Tg)
    y_mat, n_ser = _build_series_mat(s, n_pts, Tv_out)

    # Allocate coefficient matrices (Dual when grid is Dual)
    Tc = _promote_eltype(_coeff_op, eltype(x), Tv_out)
    a_mat = Matrix{Tc}(undef, n_pts, n_ser)
    d_mat = Matrix{Tc}(undef, n_pts, n_ser)

    # Promote BC values to Tv_out for convert(Tv, bc.val) compatibility.
    # The wrapped axis `x` carries `h`/`inv_h` directly via `_get_h(x, i)`.
    bc_promoted = _normalize_bc(bc, first(y_mat))
    _typed_zero = 0 * y_mat[1]

    # Compute coefficients for each series from y_mat columns
    for k in 1:n_ser
        y_col = @view y_mat[:, k]
        d_k, a_k = _compute_quadratic_coeffs(x, y_col, bc_promoted)

        @inbounds for i in 1:n_pts
            d_mat[i, k] = d_k[i]
        end
        @inbounds for i in 1:(n_pts - 1)
            a_mat[i, k] = a_k[i]
        end
        a_mat[n_pts, k] = _typed_zero
    end

    extrap_p = _promote_extrap(extrap, eltype(y_mat))
    return QuadraticSeriesInterpolant(x, y_mat, a_mat, d_mat, extrap_p, search)
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
function (sitp::QuadraticSeriesInterpolant{Tg, Tv, P})(
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual (no-op for Float/Float-backed Dual)
    xq_promoted = _promote_coord(xq, Tg)
    T_out = _promote_eltype(_interp_op, Tg, Tv, typeof(xq_promoted))
    output = Vector{T_out}(undef, n_series(sitp))

    # One lean dL-baking anchor (shared build; NoExtrap throws OOB inside), then
    # the point-contiguous op-threaded kernel over the stored y/a/d coefficients.
    A = _quadratic_series_anchor_type(sitp.extrap, sitp.x, _coord_eltype(Tq, Tg))
    a = _build_series_anchor(QuadraticInterp(), A, sitp.x, xq_promoted, sitp.extrap, _should_wrap(sitp), _resolve_search(sitp.x, xq, search, hint))
    y_point, a_point, d_point = _ensure_point_layout!(sitp)
    _quadratic_series_eval!(output, y_point, a_point, d_point, a, deriv, sitp.extrap)
    return output
end

"""
    (sitp::QuadraticSeriesInterpolant)(output::AbstractVector, xq::Real; deriv=EvalValue(), search=AutoSearch())

Evaluate all series at scalar query point (in-place, zero allocation).

# AD Support
Supports ForwardDiff.Dual input for automatic differentiation.
The anchor preserves the Dual type in `xq` and `dL` fields for AD propagation.
"""
function (sitp::QuadraticSeriesInterpolant{Tg, Tv, P})(
        output::AbstractVector,  # Relaxed: allows Dual vector
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    _validate_scalar_output(output, n_series(sitp))

    # Promote for anchor: Int→Float, Int-backed Dual→Float-backed Dual
    xq_promoted = _promote_coord(xq, Tg)

    A = _quadratic_series_anchor_type(sitp.extrap, sitp.x, _coord_eltype(Tq, Tg))
    a = _build_series_anchor(QuadraticInterp(), A, sitp.x, xq_promoted, sitp.extrap, _should_wrap(sitp), _resolve_search(sitp.x, xq, search, hint))
    y_point, a_point, d_point = _ensure_point_layout!(sitp)
    _quadratic_series_eval!(output, y_point, a_point, d_point, a, deriv, sitp.extrap)
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
function (sitp::QuadraticSeriesInterpolant{Tg, Tv, P})(
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
    (sitp::QuadraticSeriesInterpolant)(outputs, xq::AbstractVector; deriv=EvalValue())

Evaluate all series at multiple query points (in-place, zero allocation when types match).

# Precision Preservation
Uses pooled anchors with promoted type `_coord_eltype(Tq, Tg)` to preserve precision.
Pool handles both same-type and mixed-type cases efficiently.
"""
@with_pool pool function (sitp::QuadraticSeriesInterpolant{Tg, Tv, P})(
        outputs::AbstractVector{<:AbstractVector},
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = sitp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, P, Tq <: Real}
    n_query = length(xq)
    _validate_series_outputs(outputs, n_series(sitp), n_query)

    # One lean anchor buffer (op-independent; `_build_series_anchor` promotes the
    # query internally), then the series-contiguous kernel per (k, j).
    A = _quadratic_series_anchor_type(sitp.extrap, sitp.x, _coord_eltype(Tq, Tg))
    anchors = acquire!(pool, A, n_query)
    searcher = _resolve_search(sitp.x, xq, search, hint)
    _fill_series_anchors!(QuadraticInterp(), anchors, sitp.x, xq, sitp.extrap, _should_wrap(sitp), searcher)

    y = sitp.y
    a = sitp.a
    d = sitp.d
    extrap = sitp.extrap
    @inbounds for k in 1:n_series(sitp)
        for j in 1:n_query
            outputs[k][j] = _quadratic_series_eval(y, a, d, k, anchors[j], deriv, extrap)
        end
    end
    return outputs
end

"""
Evaluate all series using pre-built anchors.

The anchor's `dL` field already has the correct precision (via `_promote_coord`).
"""
function _eval_series_anchored!(
        outputs::AbstractVector{<:AbstractVector},
        sitp::QuadraticSeriesInterpolant{Tg, Tv},
        aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{Tg}},
        op::AbstractEvalOp
    ) where {Tg, Tv}
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
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_QuadraticAnchoredQuery{Tg, Taq},
        dL::Tq,  # Passed by caller for precision control
        extrap::NoExtrap,
        op::AbstractEvalOp
    ) where {Tg, Tv, Tc, Taq <: Real, Tq <: Real}
    if aq.state != IN_DOMAIN
        _throw_extrap_domain_error(aq.xq, x_min, x_max)
    end
    return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
end

@inline function _eval_single_quadratic_with_extrap(
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_QuadraticAnchoredQuery{Tg, Taq},
        dL::Tq,
        extrap::_ClampOrFill,
        op::EvalValue
    ) where {Tg, Tv, Tc, Taq <: Real, Tq <: Real}
    if aq.state != IN_DOMAIN  # outside domain
        y_bnd = @inbounds y[_boundary_point_index(aq.state, n_pts)]
        return _eval_extrapolation(op, y_bnd, extrap, aq.xq)
    else
        return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
    end
end

@inline function _eval_single_quadratic_with_extrap(
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_QuadraticAnchoredQuery{Tg, Taq},
        dL::Tq,
        extrap::_ClampOrFill,
        op::Union{EvalDeriv1, EvalDeriv2, EvalDeriv3}
    ) where {Tg, Tv, Tc, Taq <: Real, Tq <: Real}
    if aq.state != IN_DOMAIN  # outside domain
        return _eval_extrapolation(op, first(y), extrap, aq.xq)
    else
        return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
    end
end

# DerivOp{N≥4} + ClampOrFill: zero without loading a/d/y
@inline function _eval_single_quadratic_with_extrap(
        y::AbstractVector{Tv},
        ::AbstractVector{Tc},
        ::AbstractVector{Tc},
        ::Int,
        ::Tg,
        ::Tg,
        ::_QuadraticAnchoredQuery{Tg, Taq},
        ::Tq,
        ::_ClampOrFill,
        ::DerivOp{N}
    ) where {Tg, Tv, Tc, Taq <: Real, Tq <: Real, N}
    return 0 * first(y)
end

@inline function _eval_single_quadratic_with_extrap(
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        n_pts::Int,
        x_min::Tg,
        x_max::Tg,
        aq::_QuadraticAnchoredQuery{Tg, Taq},
        dL::Tq,
        extrap::AbstractExtrap,  # ExtendExtrap, WrapExtrap, etc.
        op::AbstractEvalOp
    ) where {Tg, Tv, Tc, Taq <: Real, Tq <: Real}
    return _quadratic_kernel(op, a[aq.idx], d[aq.idx], y[aq.idx], dL)
end

# ========================================
# Pre-built Anchor Evaluation
# ========================================

"""
    (sitp::QuadraticSeriesInterpolant)(outputs, aq_vec::AbstractVector{<:_QuadraticAnchoredQuery}; deriv=EvalValue())

Evaluate with pre-built anchors (TRUE zero-allocation).
"""
function (sitp::QuadraticSeriesInterpolant{Tg, Tv})(
        outputs::AbstractVector{<:AbstractVector{Tv}},
        aq_vec::AbstractVector{<:_QuadraticAnchoredQuery{Tg, Tq}};
        deriv::DerivOp = EvalValue()
    ) where {Tg, Tv, Tq <: Real}
    _validate_series_outputs(outputs, n_series(sitp), length(aq_vec))

    _eval_series_anchored!(outputs, sitp, aq_vec, deriv)
    return outputs
end
