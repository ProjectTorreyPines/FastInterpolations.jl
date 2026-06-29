# ========================================
# Hermite Mathematical Functions (Generic)
# ========================================
#
# Pure mathematical functions for Hermite interpolation shared across all dimensions:
# - Hermite basis functions (h00, h10, h01, h11)
# - 1D Hermite kernel for value and derivatives
# - Moment-to-derivative conversion (1D)
#
# Type-generic: Works with any value type (duck typing) and ForwardDiff.Dual queries
#
# Note: These functions use the OUTPUT type (Tv or promoted type) for
# intermediate calculations to support complex values and AD.

# ========================================
# HERMITE BASIS FUNCTIONS
# ========================================
#
# Cubic Hermite interpolation uses function values AND derivative values
# at interval endpoints to construct a smooth C1-continuous interpolant.
#
# Given interval [xL, xR] with:
#   - yL, yR: function values at endpoints
#   - dyL, dyR: derivative values (dy/dx, unscaled)
#   - normalized position t ∈ [0,1]
#
# The Hermite polynomial is:
#   P(t) = h00(t)*yL + h10(t)*(h*dyL) + h01(t)*yR + h11(t)*(h*dyR)
#
# Basis functions:
#   h00(t) = 2t³ - 3t² + 1   (left endpoint value basis)
#   h10(t) = t³ - 2t² + t    (left endpoint derivative basis)
#   h01(t) = -2t³ + 3t²      (right endpoint value basis)
#   h11(t) = t³ - t²         (right endpoint derivative basis)

"""
    _hermite_h00(t)

Left endpoint value basis function: h00(t) = 2t³ - 3t² + 1
"""
@inline function _hermite_h00(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return muladd(T(2), t3, muladd(T(-3), t2, one(T)))
end

"""
    _hermite_h10(t)

Left endpoint derivative basis function: h10(t) = t³ - 2t² + t
"""
@inline function _hermite_h10(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return muladd(t3, one(T), muladd(T(-2), t2, t))
end

"""
    _hermite_h01(t)

Right endpoint value basis function: h01(t) = -2t³ + 3t²
"""
@inline function _hermite_h01(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return muladd(T(-2), t3, muladd(T(3), t2, zero(T)))
end

"""
    _hermite_h11(t)

Right endpoint derivative basis function: h11(t) = t³ - t²
"""
@inline function _hermite_h11(t::T) where {T}
    t2 = t * t
    t3 = t2 * t
    return t3 - t2
end

# ========================================
# HERMITE 1D EVALUATION KERNELS
# ========================================
# Dispatch on EvalValue, EvalDeriv1, EvalDeriv2, EvalDeriv3
#
# Arguments:
# - yL, yR: Function values at left/right endpoints (type Tv)
# - dyL, dyR: Derivative values at endpoints (type Tv)
# - h: Interval width (type Tg)
# - inv_h: 1/h precomputed (type Tinv; decoupled from Tg)
# - dL: Distance from left endpoint xq - xL (type Tq — query type)

"""
    _hermite_kernel_1d(::EvalValue, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate cubic Hermite interpolation value at a point within interval.

Returns interpolated function value.
"""
@inline function _hermite_kernel_1d(
        ::EvalValue,
        yL, yR, dyL, dyR,
        h::Tg, inv_h::Tinv, dL::Tq
    ) where {Tg, Tinv, Tq}
    # Promote to output type for intermediate calculations
    t = dL * inv_h

    h00_val = _hermite_h00(t)
    h10_val = _hermite_h10(t)
    h01_val = _hermite_h01(t)
    h11_val = _hermite_h11(t)

    value_contrib = muladd(h00_val, yL, h01_val * yR)
    deriv_contrib = muladd(h10_val, dyL, h11_val * dyR) * h

    return value_contrib + deriv_contrib
end

"""
    _hermite_kernel_1d(::EvalDeriv1, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate first derivative: dP/dx = (dP/dt) / h
"""
@inline function _hermite_kernel_1d(
        ::EvalDeriv1,
        yL, yR, dyL, dyR,
        h::Tg, inv_h::Tinv, dL::Tq
    ) where {Tg, Tinv, Tq}
    # Let t remain in coordinate type - basis derivatives stay real
    t = dL * inv_h
    t_sq = t * t

    # Derivatives of basis functions w.r.t. t (real arithmetic)
    dh00_dt = muladd(6, t_sq, -6 * t)                    # 6t² - 6t
    dh10_dt = muladd(3, t_sq, muladd(-4, t, one(t)))     # 3t² - 4t + 1
    dh01_dt = muladd(-6, t_sq, 6 * t)                    # -6t² + 6t
    dh11_dt = muladd(3, t_sq, -2 * t)                    # 3t² - 2t

    # Auto-promote when combining with value types
    value_contrib = muladd(dh00_dt, yL, dh01_dt * yR)
    deriv_contrib = muladd(dh10_dt, dyL, dh11_dt * dyR) * h

    dP_dt = value_contrib + deriv_contrib
    return dP_dt * inv_h
end

"""
    _hermite_kernel_1d(::EvalDeriv2, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate second derivative: d²P/dx² = (d²P/dt²) / h²
"""
@inline function _hermite_kernel_1d(
        ::EvalDeriv2,
        yL, yR, dyL, dyR,
        h::Tg, inv_h::Tinv, dL::Tq
    ) where {Tg, Tinv, Tq}
    # Let t remain in coordinate type - basis derivatives stay real
    t = dL * inv_h

    # Second derivatives of basis functions w.r.t. t (real arithmetic)
    d2h00_dt2 = muladd(12, t, -6)      # 12t - 6
    d2h10_dt2 = muladd(6, t, -4)       # 6t - 4
    d2h01_dt2 = muladd(-12, t, 6)      # -12t + 6
    d2h11_dt2 = muladd(6, t, -2)       # 6t - 2

    # Auto-promote when combining with value types
    value_contrib = muladd(d2h00_dt2, yL, d2h01_dt2 * yR)
    deriv_contrib = muladd(d2h10_dt2, dyL, d2h11_dt2 * dyR) * h

    d2P_dt2 = value_contrib + deriv_contrib
    inv_h2 = inv_h * inv_h
    return d2P_dt2 * inv_h2
end

"""
    _hermite_kernel_1d(::EvalDeriv3, yL, yR, dyL, dyR, h, inv_h, dL)

Evaluate third derivative: d³P/dx³ = (d³P/dt³) / h³ (constant within interval)
"""
@inline function _hermite_kernel_1d(
        ::EvalDeriv3,
        yL, yR, dyL, dyR,
        h::Tg, inv_h::Tinv, dL::Tq
    ) where {Tg, Tinv, Tq}
    # Third derivatives are constants: d³h00/dt³=12, d³h10/dt³=6, d³h01/dt³=-12, d³h11/dt³=6
    # `(yL - yR)` is widened into the moment field via `_fielddiff` so narrow y can't wrap.
    value_contrib = 12 * _fielddiff(_promote_eltype(_coeff_op, Tg, typeof(yL)), yL, yR)
    deriv_contrib = 6 * h * (dyL + dyR)

    d3P_dt3 = value_contrib + deriv_contrib
    inv_h3 = inv_h * inv_h * inv_h
    return d3P_dt3 * inv_h3 * one(dL)
end

"""
    _hermite_kernel_1d(::DerivOp{N}, ...) where {N}

Generic fallback: N-th derivative of cubic Hermite is zero for N ≥ 4.
"""
@inline function _hermite_kernel_1d(
        ::DerivOp{N},
        yL, ::Any, ::Any, ::Any,
        ::Tg, ::Tinv, dL::Tq
    ) where {N, Tg, Tinv, Tq}
    return 0 * yL * one(dL)
end

# ========================================
# MOMENT-TO-DERIVATIVE CONVERSION (1D)
# ========================================
#
# Convert cubic spline moments (second derivatives) to first derivatives
# for use in Hermite interpolation construction.
#
# Mathematical Foundation:
# Cubic splines solve for moments m[i] = f''(x[i]) via tridiagonal systems.
# Given moments, we compute first derivatives dy[i] = f'(x[i]):
#   S'(x[i]) from right = (y[i+1] - y[i])/h - h*(2*m[i] + m[i+1])/6
#   S'(x[i+1]) from left = (y[i+1] - y[i])/h + h*(m[i] + 2*m[i+1])/6

"""
    _moments_to_derivatives_1d!(dydx, m, y, spacing)

Convert cubic spline moments (second derivatives) to first derivatives.
In-place version that modifies `dydx`.

# Arguments
- `dydx::AbstractVector{Tv}`: Output first derivatives (modified in-place)
- `m::AbstractVector{Tv}`: Input moments (second derivatives)
- `y::AbstractVector{Tv}`: Function values at grid points
- `x::AbstractVector{Tg}`: Grid spacing information
"""
function _moments_to_derivatives_1d!(
        dydx::AbstractVector{Tv},
        m::AbstractVector{Tv},
        y::AbstractVector{Tv},
        x::AbstractVector{Tg}
    ) where {Tv, Tg}
    n = length(y)
    @assert length(dydx) == n "dydx length mismatch"
    @assert length(m) == n "m length mismatch"
    @assert n >= 2 "Need at least 2 points"

    inv_6 = inv(Tg(6))

    # First point (using right derivative from first interval)
    @inbounds begin
        h1 = _get_h(x, 1)
        inv_h1 = _get_inv_h(x, 1)
        h_over_6 = h1 * inv_6
        linear_slope = _fielddiff(eltype(dydx), y[2], y[1]) * inv_h1
        moment_sum = muladd(Tg(2), m[1], m[2])
        dydx[1] = muladd(-h_over_6, moment_sum, linear_slope)
    end

    # Interior and last points (using left derivative from each interval)
    @inbounds for i in 1:(n - 1)
        h = _get_h(x, i)
        inv_h = _get_inv_h(x, i)
        h_over_6 = h * inv_6
        linear_slope = _fielddiff(eltype(dydx), y[i + 1], y[i]) * inv_h
        moment_sum = muladd(Tg(2), m[i + 1], m[i])
        dydx[i + 1] = muladd(h_over_6, moment_sum, linear_slope)
    end

    return dydx
end

"""
    _apply_derivative_bc!(dydx, bc, endpoints)

Apply boundary condition corrections to computed derivatives.
Called after _moments_to_derivatives_1d! to enforce specific BC values.
"""
function _apply_derivative_bc!(dydx::AbstractVector{Tv}, bc::BCPair, args...) where {Tv}
    if bc.left isa Deriv1
        @inbounds dydx[1] = bc.left.val
    end
    if bc.right isa Deriv1
        @inbounds dydx[end] = bc.right.val
    end
    return nothing
end

function _apply_derivative_bc!(dydx::AbstractVector{Tv}, bc::PeriodicBC{E}, args...) where {Tv, E}
    # Enforce periodic: dydx[1] == dydx[end]
    # Use a `Real` scalar (`0.5`) rather than `inv(Tv(2))` so duck-typed `Tv`
    # without `inv` (e.g. test `MyDuck` with only `+,-, Real*Tv`) still work.
    # `Real * Tv` is the documented minimum op set for duck values.
    avg = (dydx[1] + dydx[end]) * 0.5
    @inbounds dydx[1] = avg
    @inbounds dydx[end] = avg
    return nothing
end

# Fallback (no-op for other BC types)
function _apply_derivative_bc!(dydx, bc, args...)
    return nothing
end

# ========================================
# 1D DIFFERENTIATION HELPERS
# ========================================
# Unified API: _deriv_1d!(deriv, values, grid, bc)
# BC dispatch: AbstractBC or CubicFit

"""
    _deriv_1d!(deriv, values, grid, bc)

Differentiate 1D vector using cubic splines. BC type determines the method:
- `AbstractBC` (ZeroCurvBC, ZeroSlopeBC, PeriodicBC, etc.): Use specified BC
- `CubicFit`: Estimate endpoint derivatives via polynomial fitting
"""
@with_pool pool function _deriv_1d!(
        deriv::AbstractVector{Tv}, values::AbstractVector{Tv},
        grid::AbstractVector{Tg}, bc::AbstractBC
    ) where {Tg, Tv}
    n = length(values)
    # Cache construction: _get_cubic_cache internally uses _cache_pointbc (duck-safe,
    # converts BC to structural form with zero(Tg) — no convert(Tg, bc.val) needed).
    # Computation: normalize BC values to Tv via value-based _normalize_bc.
    bc_compute = _is_periodic_bc(bc) ? PeriodicBC() : _normalize_bc(bc, first(values))
    cache = _get_cubic_cache(grid, bc, _effective_autocache(true, Tg))
    actual_bc = cache.bc isa PeriodicBC ? cache.bc : bc_compute
    Tz = _promote_eltype(_coeff_op, eltype(cache.x), Tv)
    m = acquire!(pool, Tz, n)
    _solve_system!(m, cache, values, actual_bc)
    _moments_to_derivatives_1d!(deriv, m, values, cache.x)
    _apply_derivative_bc!(deriv, actual_bc)
    return deriv
end

@with_pool pool function _deriv_1d!(
        deriv::AbstractVector{Tv}, values::AbstractVector{Tv},
        grid::AbstractVector{Tg}, ::CubicFit
    ) where {Tg, Tv}
    n = length(values)
    @assert n >= 4 "Need at least 4 points for CubicFit"

    deriv_left = _estimate_endpoint_derivative(grid, values, LeftSide(), CubicFit())
    deriv_right = _estimate_endpoint_derivative(grid, values, RightSide(), CubicFit())

    # BC values are Tv type (can be Complex)
    bc = BCPair(Deriv1(deriv_left), Deriv1(deriv_right))
    # Cache uses grid type Tg for matrix structure
    bc_cache = BCPair(Deriv1(zero(Tg)), Deriv1(zero(Tg)))
    cache = _get_cubic_cache(grid, bc_cache, _effective_autocache(true, Tg))
    Tz = _promote_eltype(_coeff_op, eltype(cache.x), Tv)
    m = acquire!(pool, Tz, n)
    _solve_system!(m, cache, values, bc)
    _moments_to_derivatives_1d!(deriv, m, values, cache.x)
    _apply_derivative_bc!(deriv, bc)
    return deriv
end

# ========================================
# BATCH THOMAS SOLVERS
# ========================================
#
# High-performance batch solvers for ND cubic interpolation.
# Key optimization: Loop transposition for cache-friendly SIMD access.
#
# When solving along axis D (D > 1), we transpose the LOOP ORDER (not data):
# - Outer loop: system step (k = 1:n_sys)
# - Inner loop: axis 1 (i = 1:n_batch) - CONTIGUOUS in column-major!
#
# This enables @simd vectorization over the contiguous dimension.
#
# Used by _differentiate_nd_along_dim_batch! which reshapes ND arrays to
# 2D matrices (shape_before × n_d) for batch processing.

"""
    _ldiv_along_dim_vectorized!(z, thomas)

Batch Thomas solver for systems along axis 2 (rows).
KEY OPTIMIZATION: Outer loop = system step, inner loop = axis 1 (contiguous).
This enables @simd vectorization over the contiguous dimension.

# Type Parameters
- `Tv`: Value type (unconstrained)
- `Tg`: Grid type (AbstractFloat) for ThomasFactorization

# Arguments
- `z::AbstractMatrix{Tv}`: RHS matrix (modified in-place), systems along axis 2
- `thomas::ThomasFactorization{Tg}`: Thomas factorization with dl, du, inv_d
"""
@inline function _ldiv_along_dim_vectorized!(
        z::AbstractMatrix{Tv},
        thomas::ThomasFactorization{Tg, V}
    ) where {Tv, Tg, V <: AbstractVector{Tg}}
    dl = thomas.dl
    du = thomas.du
    inv_d = thomas.inv_d
    n_sys = length(inv_d)   # System size (axis 2 length)
    n_batch = size(z, 1)    # Batch size (axis 1 length, contiguous!)

    # Forward substitution: transposed loop order for contiguous access
    @inbounds for k in 2:n_sys
        factor = -dl[k - 1]
        @simd for i in 1:n_batch
            z[i, k] = muladd(factor, z[i, k - 1], z[i, k])
        end
    end

    # Backward substitution: final column
    inv_d_n = inv_d[n_sys]
    @inbounds @simd for i in 1:n_batch
        z[i, n_sys] *= inv_d_n
    end

    # Backward substitution: remaining columns
    @inbounds for k in (n_sys - 1):-1:1
        u_factor = -du[k]
        d_factor = inv_d[k]
        @simd for i in 1:n_batch
            z[i, k] = muladd(u_factor, z[i, k + 1], z[i, k]) * d_factor
        end
    end
    return z
end

# Dispatch to SIMD-optimized solver (only for D ≥ 2)
# Note: Val(1) error method is at the end of this file
@inline _ldiv_along_dim!(z, thomas, ::Val{D}) where {D} = _ldiv_along_dim_vectorized!(z, thomas)

# ========================================
# HIGH-LEVEL BATCH SOLVER INTERFACE
# ========================================

"""
    solve_along_dim!(out_z, cache, data, bc, ::Val{D})

Compute cubic spline second derivatives (moments) for batch systems along dimension D.
Optimized for memory locality and SIMD execution.

# Use Cases
- ND grids: `Val(2)` for batch processing of reshaped 2D matrix slices
- For axis 1: Use `_solve_system!` in a loop (per-column approach is faster)

# Type Parameters
- `Tv`: Value type (unconstrained)
- `Tg`: Grid type (AbstractFloat) for cache and spacing

# Arguments
- `out_z::AbstractMatrix{Tv}`: Output array (same size as data, modified in-place)
- `cache::CubicSplineCache{Tg}`: Precomputed Thomas factorization and grid info
- `data::AbstractMatrix{Tv}`: Input data array
- `bc::BCPair`: Boundary condition pair
- `::Val{D}`: Dimension along which to solve (D ≥ 2 only)
"""
function solve_along_dim!(
        out_z::AbstractMatrix{Tv},
        cache::CubicSplineCache{Tg, X, F, BC_cache},
        data::AbstractMatrix{Tv},
        bc::BCPair,
        dim::Val{D}
    ) where {Tv, Tg, X, F, BC_cache, D}
    # Step 1: Compute RHS for all systems
    compute_rhs_along_dim!(out_z, data, cache.x, bc, dim)

    # Step 2: Batch solve (SIMD for D≥2)
    _ldiv_along_dim!(out_z, cache.thomas, dim)

    return out_z
end

# ========================================
# BATCH RHS COMPUTATION
# ========================================

"""
    compute_rhs_along_dim!(D, data, x, spacing, bc, ::Val{D})

Compute RHS for batch systems along dimension `D`.

# Arguments
- `D::AbstractMatrix{T}`: Output RHS matrix (modified in-place)
- `data::AbstractMatrix{T}`: Input data matrix
- `x::AbstractVector{T}`: Grid points
- `x::AbstractVector{T}`: Grid spacing object
- `bc::BCPair{T}`: Boundary condition pair
- `::Val{D}`: Dimension along which to compute RHS

Note: Val(1) error method is at the end of this file.
"""
function compute_rhs_along_dim!(
        D::AbstractMatrix{Tv},
        data::AbstractMatrix{Tv},
        x::AbstractVector{Tg},
        bc::BCPair,
        ::Val{2}
    ) where {Tv, Tg}
    n_batch = size(data, 1)
    @inbounds for i in 1:n_batch
        compute_rhs!(view(D, i, :), view(data, i, :), x, bc)
    end
    return D
end

# ========================================
# BATCH MOMENT-TO-DERIVATIVE CONVERSION
# ========================================

"""
    moments_to_derivatives_along_dim!(out, M, data, spacing, bc, ::Val{D})

Convert moments to derivatives for batch systems along dimension D.

# Arguments
- `out::AbstractMatrix{T}`: Output derivatives (modified in-place)
- `M::AbstractMatrix{T}`: Input moments (second derivatives)
- `data::AbstractMatrix{T}`: Original function values
- `x::AbstractVector{T}`: Grid spacing
- `bc`: Boundary condition configuration
- `::Val{D}`: Dimension along which conversion was performed

Note: Val(1) error method is at the end of this file.
"""
function moments_to_derivatives_along_dim!(
        out::AbstractMatrix{Tv},
        M::AbstractMatrix{Tv},
        data::AbstractMatrix{Tv},
        x::AbstractVector{Tg},
        bc,
        ::Val{2}
    ) where {Tv, Tg}
    n_batch = size(data, 1)
    @inbounds for i in 1:n_batch
        _moments_to_derivatives_1d!(
            view(out, i, :), view(M, i, :), view(data, i, :), x
        )
        _apply_derivative_bc!(view(out, i, :), bc)
    end
    return out
end

# ========================================
# UNSUPPORTED BATCH OPERATIONS (Val(1))
# ========================================
#
# These methods explicitly throw errors for batch operations along axis 1.
# Benchmarking showed that per-column approach is faster due to view creation overhead.
# For axis 1 operations, use the per-column alternatives mentioned in error messages.
#
# These are grouped here for clarity and to simplify coverage testing.

"""
    _ldiv_along_dim!(z, thomas, ::Val{1})

Batch solving along axis 1 is explicitly unsupported.
Benchmarking showed per-column approach is faster due to view creation overhead.
Use `_solve_system!` in a loop for axis 1, or use `Val(2)` for SIMD-optimized batch solving.
"""
@noinline function _ldiv_along_dim!(z, thomas, ::Val{1})
    throw(
        ArgumentError(
            "Batch solving along axis 1 (Val(1)) is not supported.\n" *
                "Reason: Per-column approach is faster due to view creation overhead.\n" *
                "Use _solve_system! in a loop for axis 1, or use Val(2) for SIMD-optimized batch solving."
        )
    )
end

"""
    compute_rhs_along_dim!(..., ::Val{1})

Batch RHS computation along axis 1 is explicitly unsupported.
Use `compute_rhs!` in a loop for axis 1, or use `Val(2)` for batch computation.
"""
@noinline function compute_rhs_along_dim!(
        D::AbstractMatrix{Tv},
        data::AbstractMatrix{Tv},
        x::AbstractVector{Tg},
        bc::BCPair,
        ::Val{1}
    ) where {Tv, Tg}
    throw(
        ArgumentError(
            "Batch RHS computation along axis 1 (Val(1)) is not supported.\n" *
                "Use compute_rhs! in a loop for axis 1, or use Val(2) for batch computation."
        )
    )
end

"""
    moments_to_derivatives_along_dim!(..., ::Val{1})

Batch moment-to-derivative conversion along axis 1 is explicitly unsupported.
Use `_moments_to_derivatives_1d!` in a loop for axis 1, or use `Val(2)` for batch conversion.
"""
@noinline function moments_to_derivatives_along_dim!(
        out::AbstractMatrix{Tv},
        M::AbstractMatrix{Tv},
        data::AbstractMatrix{Tv},
        x::AbstractVector,
        bc,
        ::Val{1}
    ) where {Tv}
    throw(
        ArgumentError(
            "Batch moment-to-derivative along axis 1 (Val(1)) is not supported.\n" *
                "Use _moments_to_derivatives_1d! in a loop for axis 1, or use Val(2) for batch conversion."
        )
    )
end
