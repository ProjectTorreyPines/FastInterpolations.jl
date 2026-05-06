# ========================================
# Quadratic Spline Coefficient Computation
# ========================================
# Functions to compute spline coefficients (s, d, a) from grid and values.
#
# Mathematical model:
#   S_i(x) = a_i*(x - x_i)² + d_i*(x - x_i) + y_i
#
# Coefficient computation:
#   1. s[i] = (y[i+1] - y[i]) / h[i]  (secant slopes)
#   2. Fill d[] via BC-dependent recurrence:
#      - Left BC:  d[1] from BC, forward  recurrence d[i+1] = 2*s[i] - d[i]
#      - Right BC: d[n] from BC, backward recurrence d[i] = 2*s[i] - d[i+1]
#   3. a[i] = (s[i] - d[i]) / h[i]  (quadratic coefficients)

# ========================================
# Type Alias for Quadratic BC
# ========================================

"""
Supported boundary conditions for quadratic spline interpolation.
- `Left{B<:PointBC}`: BC at left endpoint (forward recurrence)
- `Right{B<:PointBC}`: BC at right endpoint (backward recurrence)
- `MinCurvFit`: Global curvature minimization (singleton)
"""
const QuadraticBC = Union{Left, Right, MinCurvFit}

# ========================================
# Secant Computation
# ========================================

"""
    _compute_quadratic_secants!(s, y, spacing)

Compute secant slopes: s[i] = (y[i+1] - y[i]) * inv_h[i]

# Arguments
- `s::AbstractVector`: Output secant vector (length n-1)
- `y::AbstractVector`: Values at grid points (length n)
- `axis::AbstractVector{Tg}`: Wrapped or raw grid axis (read via `_get_inv_h(axis, i)`)
"""
@inline function _compute_quadratic_secants!(s::AbstractVector{Tc}, y::AbstractVector, axis::AbstractVector{Tg}) where {Tc, Tg}
    n = length(y) - 1
    @inbounds for i in 1:n
        s[i] = (y[i + 1] - y[i]) * _get_inv_h(axis,i)  # Tv * Tg → Tv
    end
    return s
end

# ========================================
# Recurrence Functions
# ========================================

"""
    _forward_recurrence!(d, s, d1)

Fill slope array using forward recurrence from d[1].
d[i+1] = 2*s[i] - d[i]

# Arguments
- `d::AbstractVector`: Output slope array (length n)
- `s::AbstractVector`: Secant slopes (length n-1)
- `d1`: Initial slope d[1]
"""
@inline function _forward_recurrence!(d::AbstractVector{Tc}, s::AbstractVector{Tc}, d1) where {Tc}
    d[1] = d1
    n = length(d)
    @inbounds for i in 1:(n - 1)
        d[i + 1] = 2 * s[i] - d[i]  # All Tv operations
    end
    return d
end

"""
    _backward_recurrence!(d, s, dn)

Fill slope array using backward recurrence from d[n].
d[i] = 2*s[i] - d[i+1]

# Arguments
- `d::AbstractVector`: Output slope array (length n)
- `s::AbstractVector`: Secant slopes (length n-1)
- `dn`: Final slope d[n]
"""
@inline function _backward_recurrence!(d::AbstractVector{Tc}, s::AbstractVector{Tc}, dn) where {Tc}
    n = length(d)
    d[n] = dn
    @inbounds for i in (n - 1):-1:1
        d[i] = 2 * s[i] - d[i + 1]  # All Tv operations
    end
    return d
end

# ========================================
# Slope Filling (BC-Dispatched)
# ========================================

"""
    _fill_slopes!(d, s, spacing, bc, x, y)

Fill slope array d[] based on boundary condition type.
Dispatches at compile time to use optimal recurrence direction:
- Left BC:  compute d[1], forward recurrence  → O(n)
- Right BC: compute d[n], backward recurrence → O(n)

The `x` and `y` parameters are needed for PolyFit{D} BCs which estimate
derivatives from data. For other BC types, they are ignored.

# Type Parameters
- `Tv`: Value type for d, s (unconstrained)
- `Tg`: Grid type for spacing, x
"""
# Left(Deriv1): d[1] given directly, forward recurrence
# convert() is a no-op when types match (optimized away at compile time)
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Left{<:Deriv1}, ::AbstractVector{Tg}, ::AbstractVector
    ) where {Tc, Tg}
    d1 = convert(Tc, bc.bc.val)
    return _forward_recurrence!(d, s, d1)
end

# Left(Deriv2): d[1] = s[1] - (κ/2)*h[1], forward recurrence
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Left{<:Deriv2}, ::AbstractVector{Tg}, ::AbstractVector
    ) where {Tc, Tg}
    κ = convert(Tc, bc.bc.val)
    d1 = s[1] - κ * (_get_h(axis,1) / 2)  # Tv - Tv*Tg → Tv (no /(Tv,Int) needed)
    return _forward_recurrence!(d, s, d1)
end

# Right(Deriv1): d[n] given directly, backward recurrence
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Right{<:Deriv1}, ::AbstractVector{Tg}, ::AbstractVector
    ) where {Tc, Tg}
    dn = convert(Tc, bc.bc.val)
    return _backward_recurrence!(d, s, dn)
end

# Right(Deriv2): compute d[n] from curvature, backward recurrence
# d[n] = s[n-1] + (κ/2)*h[n-1]  (derived from a[n-1] = κ/2)
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Right{<:Deriv2}, ::AbstractVector{Tg}, ::AbstractVector
    ) where {Tc, Tg}
    κ = convert(Tc, bc.bc.val)
    n_intervals = length(s)
    dn = s[end] + κ * (_get_h(axis,n_intervals) / 2)  # Tv + Tv*Tg → Tv (no /(Tv,Int) needed)
    return _backward_recurrence!(d, s, dn)
end

# MinCurvFit: minimize total curvature via closed-form optimization
"""
    _fill_slopes!(d, s, spacing, ::MinCurvFit, x, y)

Fill slope array using global curvature minimization.

Minimizes total curvature: ∫(S'')² dx = Σ 4*a[i]²*h[i] = Σ (s[i] - d[i])²/h[i]

# Mathematical Derivation
The slope d[i] depends on d[1] via forward recurrence:
- d[i] = α[i] * d[1] + β[i]  where α[i] = (-1)^(i+1) (alternating sign)
- β[1] = 0, β[i+1] = 2*s[i] - β[i]

Setting df/d(d[1]) = 0 gives the closed-form solution:
- d[1]_optimal = [Σ α[i]*(s[i] - β[i])/h[i]] / [Σ 1/h[i]]

For Complex values, the optimization minimizes |s[i] - d[i]|² which works
element-wise on real and imaginary parts.

# Complexity
O(n) time, O(1) extra space (on-the-fly β computation).
"""
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        ::MinCurvFit, ::AbstractVector{Tg}, ::AbstractVector
    ) where {Tc, Tg}
    n = length(d)
    n_intervals = n - 1  # = length(s)

    # Edge case: single segment (n=2)
    # For single segment, minimize a² = (s-d[1])²/h
    # This means d[1] = s[1] (making a = 0, zero curvature)
    if n == 2
        d1 = @inbounds s[1]
        _forward_recurrence!(d, s, d1)
        return d
    end

    # Compute optimal d[1] using closed-form solution
    # d[i] = α[i] * d[1] + β[i]
    # α[i] = (-1)^(i+1): +1, -1, +1, -1, ...
    # β[i+1] = 2*s[i] - β[i], β[1] = 0

    # Objective: minimize Σ (s[i] - d[i])²/h[i]
    # = Σ (s[i] - α[i]*d[1] - β[i])²/h[i]
    # df/d(d[1]) = -2 * Σ α[i]*(s[i] - α[i]*d[1] - β[i])/h[i] = 0
    # Note: α[i]² = 1

    # Rearranging:
    # Σ α[i]*(s[i] - β[i])/h[i] = d[1] * Σ 1/h[i]
    # d[1] = [Σ α[i]*(s[i] - β[i])/h[i]] / [Σ 1/h[i]]

    inv_h_sum = zero(Tg)
    numerator = 0 * first(s)
    β = 0 * first(s)
    sign = one(Tg)  # α[1] = (-1)^(1+1) = +1

    @inbounds for i in 1:n_intervals
        inv_h_i = _get_inv_h(axis,i)  # precomputed — no inv() needed
        inv_h_sum += inv_h_i
        numerator += sign * (s[i] - β) * inv_h_i  # Tg * Tv * Tg → Tv
        β = 2 * s[i] - β  # Tv operations
        sign = -sign  # alternate: +1, -1, +1, ...
    end

    d1_optimal = inv(inv_h_sum) * numerator  # Tg * Tv → Tv (duck-safe: no /(Tv,Tg))
    return _forward_recurrence!(d, s, d1_optimal)
end

# ========================================
# Generic PolyFit{D}: Materialize to Deriv1
# ========================================

"""
    _fill_slopes!(d, s, spacing, bc::Left{PolyFit{D}}, x, y)

Fill slope array using generic polynomial fit at left endpoint.

Materializes PolyFit{D} to Deriv1{Tv} using `materialize_bc`, then uses the
estimated derivative directly. Supports all polynomial degrees: LinearFit (D=1),
QuadraticFit (D=2), CubicFit (D=3), etc.

For Complex y values, materialize_bc returns Deriv1{ComplexF64} naturally.
"""
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, ::AbstractVector{Tg},
        bc::Left{PolyFit{D}}, x::AbstractVector{Tg}, y::AbstractVector
    ) where {D, Tc, Tg}
    # Materialize PolyFit{D} → Deriv1{Tv} using estimated derivative
    concrete_bc = materialize_bc(bc.bc, x, y, LeftSide())
    d1 = concrete_bc.val  # Already Tv type from polynomial fit on y values
    return _forward_recurrence!(d, s, d1)
end

"""
    _fill_slopes!(d, s, spacing, bc::Right{PolyFit{D}}, x, y)

Fill slope array using generic polynomial fit at right endpoint.

Materializes PolyFit{D} to Deriv1{Tv} using `materialize_bc`, then uses the
estimated derivative directly.
"""
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, ::AbstractVector{Tg},
        bc::Right{PolyFit{D}}, x::AbstractVector{Tg}, y::AbstractVector
    ) where {D, Tc, Tg}
    # Materialize PolyFit{D} → Deriv1{Tv} using estimated derivative
    concrete_bc = materialize_bc(bc.bc, x, y, RightSide())
    dn = concrete_bc.val  # Already Tv type from polynomial fit on y values
    return _backward_recurrence!(d, s, dn)
end

# ========================================
# Quadratic Coefficient Computation
# ========================================

"""
    _compute_quadratic_coefficients!(a, d, s, spacing)

Compute quadratic coefficients: a[i] = (s[i] - d[i]) * inv_h[i]

# Arguments
- `a::Vector{Tv}`: Output coefficient array (length n-1, value-derived)
- `d::Vector{Tv}`: Slope array (length n, value-derived)
- `s::Vector{Tv}`: Secant slopes (length n-1, value-derived)
- `axis::AbstractVector{Tg}`: Wrapped or raw grid axis (read via `_get_inv_h(axis, i)`)

# Type Parameters
- `Tv`: Value type (unconstrained)
- `Tg`: Grid type
"""
@inline function _compute_quadratic_coefficients!(a::AbstractVector{Tc}, d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg}) where {Tc, Tg}
    @inbounds for i in eachindex(a)
        a[i] = (s[i] - d[i]) * _get_inv_h(axis,i)  # (Tv - Tv) * Tg → Tv
    end
    return a
end

# ========================================
# Coefficient Computation (In-Place)
# ========================================

"""
    _compute_quadratic_coeffs!(d, a, x, y, bc)

Fill pre-allocated coefficient arrays for quadratic spline.
Uses AdaptiveArrayPools internally for temporary `secant` array.

# Arguments (outputs first, then inputs)
- `d::AbstractVector`: Slope coefficients (length n, value-derived)
- `a::AbstractVector`: Quadratic coefficients (length n-1, value-derived)
- `x::AbstractVector{Tg}`: x-coordinates (length n) — wrapped axis carries
  cached `h`/`inv_h`; raw `Vector` falls back to on-the-fly diff
- `y::AbstractVector`: y-values (length n)
- `bc::QuadraticBC`: Boundary condition (Left, Right, or MinCurvFit)

# Type Parameters
- `Tg`: Grid type
- `Tv`: Value type (unconstrained)
"""
@with_pool pool function _compute_quadratic_coeffs!(
        d::AbstractVector,
        a::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        bc::QuadraticBC
    ) where {Tg}
    nx = length(x)

    Tcoeff = eltype(d)
    secant = acquire!(pool, Tcoeff, nx - 1)

    _compute_quadratic_secants!(secant, y, x)
    _fill_slopes!(d, secant, x, bc, x, y)
    _compute_quadratic_coefficients!(a, d, secant, x)

    return nothing
end

# ========================================
# Coefficient Computation (Allocating)
# ========================================

"""
    _compute_quadratic_coeffs(x, y, bc) -> (d, a)

Compute quadratic spline coefficients (allocating version).
Returns only the arrays needed for evaluation: `d`, `a`.

`x` carries cached `h`/`inv_h` when wrapped (`_CachedRange` / `_CachedVector`);
raw `Vector` falls back to on-the-fly diff via the `_get_h(::AbstractVector, i)`
overload.

# Type Parameters
- `Tg`: Grid type for x
- `Tv`: Value type for y, d, a (unconstrained)

# Returns
- `d::Vector{Tv}`: Slope coefficients (value-derived)
- `a::Vector{Tv}`: Quadratic coefficients (value-derived)

Intermediate `secant` array is handled internally via AdaptiveArrayPools.

For repeated interpolation on the same grid, use `QuadraticInterpolant`
which stores precomputed coefficients.
"""
function _compute_quadratic_coeffs(
        x::AbstractVector{Tg},
        y::AbstractVector,
        bc::QuadraticBC
    ) where {Tg}
    nx = length(x)

    # Allocate arrays — widened type when grid is duck-typed (e.g. Dual)
    Tcoeff = _output_eltype(eltype(y), Tg)
    d = Vector{Tcoeff}(undef, nx)
    a = Vector{Tcoeff}(undef, nx - 1)

    # Fill using in-place version
    _compute_quadratic_coeffs!(d, a, x, y, bc)

    return d, a
end
