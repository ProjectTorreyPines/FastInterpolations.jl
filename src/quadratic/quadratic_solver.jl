# ========================================
# Quadratic Spline Coefficient Computation
# ========================================
# Functions to compute spline coefficients (d, a) from grid and values.
#
# Mathematical model:
#   S_i(x) = a_i*(x - x_i)² + d_i*(x - x_i) + y_i
#
# Coefficient computation — one sweep (MinCurvFit: one extra pass for d[1]):
#   s[i] = (y[i+1] - y[i]) / h[i]  (secant slopes, recomputed on read — no buffer)
#   d[] via BC-dependent recurrence:
#     - Left BC:  d[1] from BC, forward  recurrence d[i+1] = 2*s[i] - d[i]
#     - Right BC: d[n] from BC, backward recurrence d[i] = 2*s[i] - d[i+1]
#   a[i] = (s[i] - d[i]) / h[i]  (written by the same sweep)

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
# Secant Slopes
# ========================================

"""
    _QuadraticSecants{Tc}(y, axis)

Secant slopes `s[i] = (y[i+1] - y[i]) * inv_h[i]` (length n-1) as a lazy vector:
each read recomputes `s[i]`, so the solver needs no secant buffer. A read returns
the value a `Vector{Tc}` would store.

# Arguments
- `y::AbstractVector`: Values at grid points (length n)
- `axis::AbstractVector`: Wrapped or raw grid axis (read via `_get_inv_h(axis, i)`)
"""
struct _QuadraticSecants{Tc, Y <: AbstractVector, X <: AbstractVector} <: AbstractVector{Tc}
    y::Y
    axis::X
end
@inline _QuadraticSecants{Tc}(y::Y, axis::X) where {Tc, Y <: AbstractVector, X <: AbstractVector} =
    _QuadraticSecants{Tc, Y, X}(y, axis)
Base.size(s::_QuadraticSecants) = (length(s.y) - 1,)
Base.IndexStyle(::Type{<:_QuadraticSecants}) = IndexLinear()
@inline Base.@propagate_inbounds function Base.getindex(s::_QuadraticSecants{Tc}, i::Int) where {Tc}
    # Diffs lift into VALUE space (wrap-free for narrow eltypes); the [Y/X]
    # dimension enters via `inv_h`, never by converting y into it.
    Tvs = _value_space_eltype(eltype(s.axis), eltype(s.y))
    return convert(Tc, _fielddiff(Tvs, s.y[i + 1], s.y[i]) * _get_inv_h(s.axis, i))
end

# ========================================
# Recurrence Functions
# ========================================

# Coefficient output of the slope sweep: `a[i] = (s[i] - d[i]) * inv_h[i]`, written in
# the same loop as `d`. `a === nothing` → slopes only (ND partials).
@inline _store_quadratic_coeff!(::Nothing, _, ::Int, _, _) = nothing
@inline Base.@propagate_inbounds function _store_quadratic_coeff!(a::AbstractVector, axis, i::Int, si, di)
    a[i] = (si - di) * _get_inv_h(axis, i)  # [Y/X] * [1/X] → [Y/X²]
    return nothing
end

"""
    _forward_recurrence!(d, s, d1, a=nothing, axis=nothing)

Fill slope array using forward recurrence from d[1].
d[i+1] = 2*s[i] - d[i]
With `a`, the same sweep writes a[i] = (s[i] - d[i]) * inv_h[i].

# Arguments
- `d::AbstractVector`: Output slope array (length n)
- `s::AbstractVector`: Secant slopes (length n-1)
- `d1`: Initial slope d[1]
- `a`: Optional output coefficient array (length n-1)
- `axis`: Grid axis supplying `inv_h` for `a`
"""
@inline function _forward_recurrence!(d::AbstractVector{Tc}, s::AbstractVector{Tc}, d1, a = nothing, axis = nothing) where {Tc}
    n = length(d)
    di = convert(Tc, d1)
    @inbounds for i in 1:(n - 1)
        si = s[i]
        d[i] = di
        _store_quadratic_coeff!(a, axis, i, si, di)
        di = convert(Tc, 2 * si - di)  # All Tv operations
    end
    @inbounds d[n] = di
    return d
end

"""
    _backward_recurrence!(d, s, dn, a=nothing, axis=nothing)

Fill slope array using backward recurrence from d[n].
d[i] = 2*s[i] - d[i+1]
With `a`, the same sweep writes a[i] = (s[i] - d[i]) * inv_h[i].

# Arguments
- `d::AbstractVector`: Output slope array (length n)
- `s::AbstractVector`: Secant slopes (length n-1)
- `dn`: Final slope d[n]
- `a`: Optional output coefficient array (length n-1)
- `axis`: Grid axis supplying `inv_h` for `a`
"""
@inline function _backward_recurrence!(d::AbstractVector{Tc}, s::AbstractVector{Tc}, dn, a = nothing, axis = nothing) where {Tc}
    n = length(d)
    di = convert(Tc, dn)
    @inbounds d[n] = di
    @inbounds for i in (n - 1):-1:1
        si = s[i]
        di = convert(Tc, 2 * si - di)  # All Tv operations
        d[i] = di
        _store_quadratic_coeff!(a, axis, i, si, di)
    end
    return d
end

# ========================================
# Slope Filling (BC-Dispatched)
# ========================================

"""
    _fill_slopes!(d, s, spacing, bc, x, y, a=nothing)

Fill slope array d[] based on boundary condition type.
Dispatches at compile time to use optimal recurrence direction:
- Left BC:  compute d[1], forward recurrence  → O(n)
- Right BC: compute d[n], backward recurrence → O(n)

With `a`, the same sweep also writes the quadratic coefficients
a[i] = (s[i] - d[i]) * inv_h[i].

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
        bc::Left{<:Deriv1}, ::AbstractVector{Tg}, ::AbstractVector, a = nothing
    ) where {Tc, Tg}
    d1 = convert(Tc, bc.bc.val)
    return _forward_recurrence!(d, s, d1, a, axis)
end

# Left(Deriv2): d[1] = s[1] - (κ/2)*h[1], forward recurrence
# κ ∈ [Y/X²] ≠ slope space — never pre-convert; κ·h lands in [Y/X].
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Left{<:Deriv2}, ::AbstractVector{Tg}, ::AbstractVector, a = nothing
    ) where {Tc, Tg}
    d1 = s[1] - bc.bc.val * (_get_h(axis, 1) / 2)
    return _forward_recurrence!(d, s, d1, a, axis)
end

# Right(Deriv1): d[n] given directly, backward recurrence
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Right{<:Deriv1}, ::AbstractVector{Tg}, ::AbstractVector, a = nothing
    ) where {Tc, Tg}
    dn = convert(Tc, bc.bc.val)
    return _backward_recurrence!(d, s, dn, a, axis)
end

# Right(Deriv2): compute d[n] from curvature, backward recurrence
# d[n] = s[n-1] + (κ/2)*h[n-1]  (derived from a[n-1] = κ/2)
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Right{<:Deriv2}, ::AbstractVector{Tg}, ::AbstractVector, a = nothing
    ) where {Tc, Tg}
    n_intervals = length(s)
    dn = s[end] + bc.bc.val * (_get_h(axis, n_intervals) / 2)
    return _backward_recurrence!(d, s, dn, a, axis)
end

# MinCurvFit: minimize total curvature via closed-form optimization
"""
    _fill_slopes!(d, s, spacing, ::MinCurvFit, x, y, a=nothing)

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
O(n) time, O(1) extra space (on-the-fly β computation): one pass for d[1],
then the recurrence sweep.
"""
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        ::MinCurvFit, ::AbstractVector{Tg}, ::AbstractVector, a = nothing
    ) where {Tc, Tg}
    n = length(d)
    n_intervals = n - 1  # = length(s)

    # Edge case: single segment (n=2)
    # For single segment, minimize a² = (s-d[1])²/h
    # This means d[1] = s[1] (making a = 0, zero curvature)
    if n == 2
        d1 = @inbounds s[1]
        _forward_recurrence!(d, s, d1, a, axis)
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

    # Accumulator spaces: inv_h_sum ∈ [1/X], numerator ∈ [Y/X²] (s·inv_h),
    # β ∈ [Y/X] — value-witness zeros (Real: all plain 0.0, bit-identical).
    inv_h_sum = zero(_promote_eltype(_inv_op, Tg))
    numerator = 0 * (first(s) * _get_inv_h(axis, 1))
    β = 0 * first(s)
    sign = one(Tg)  # dimensionless ±1 — `one`, not `oneunit`

    @inbounds for i in 1:n_intervals
        inv_h_i = _get_inv_h(axis, i)  # precomputed — no inv() needed
        inv_h_sum += inv_h_i
        numerator += sign * (s[i] - β) * inv_h_i  # Tg * Tv * Tg → Tv
        β = 2 * s[i] - β  # Tv operations
        sign = -sign  # alternate: +1, -1, +1, ...
    end

    d1_optimal = inv(inv_h_sum) * numerator  # Tg * Tv → Tv (duck-safe: no /(Tv,Tg))
    return _forward_recurrence!(d, s, d1_optimal, a, axis)
end

# ========================================
# Generic PolyFit{D}: Materialize to Deriv1
# ========================================

"""
    _fill_slopes!(d, s, spacing, bc::Left{PolyFit{D}}, x, y, a=nothing)

Fill slope array using generic polynomial fit at left endpoint.

Materializes PolyFit{D} to Deriv1{Tv} using `materialize_bc`, then uses the
estimated derivative directly. Supports all polynomial degrees: LinearFit (D=1),
QuadraticFit (D=2), CubicFit (D=3), etc.

For Complex y values, materialize_bc returns Deriv1{ComplexF64} naturally.
"""
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Left{PolyFit{D}}, x::AbstractVector{Tg}, y::AbstractVector, a = nothing
    ) where {D, Tc, Tg}
    # Materialize PolyFit{D} → Deriv1{Tv} using estimated derivative
    concrete_bc = materialize_bc(bc.bc, x, y, LeftSide())
    d1 = concrete_bc.val  # Already Tv type from polynomial fit on y values
    return _forward_recurrence!(d, s, d1, a, axis)
end

"""
    _fill_slopes!(d, s, spacing, bc::Right{PolyFit{D}}, x, y, a=nothing)

Fill slope array using generic polynomial fit at right endpoint.

Materializes PolyFit{D} to Deriv1{Tv} using `materialize_bc`, then uses the
estimated derivative directly.
"""
@inline function _fill_slopes!(
        d::AbstractVector{Tc}, s::AbstractVector{Tc}, axis::AbstractVector{Tg},
        bc::Right{PolyFit{D}}, x::AbstractVector{Tg}, y::AbstractVector, a = nothing
    ) where {D, Tc, Tg}
    # Materialize PolyFit{D} → Deriv1{Tv} using estimated derivative
    concrete_bc = materialize_bc(bc.bc, x, y, RightSide())
    dn = concrete_bc.val  # Already Tv type from polynomial fit on y values
    return _backward_recurrence!(d, s, dn, a, axis)
end

# ========================================
# Coefficient Computation (In-Place)
# ========================================

"""
    _compute_quadratic_coeffs!(d, a, x, y, bc)

Fill pre-allocated coefficient arrays for quadratic spline in one sweep:
secants are recomputed on read (`_QuadraticSecants`, no buffer) and `a` is
written alongside `d`.

# Arguments (outputs first, then inputs)
- `d::AbstractVector`: Slope coefficients (length n, value-derived)
- `a::AbstractVector`: Quadratic coefficients (length n-1, value-derived; one
  `inv_h` power above `d`, so its eltype may differ)
- `x::AbstractVector{Tg}`: x-coordinates (length n) — wrapped axis carries
  cached `h`/`inv_h`; raw `Vector` falls back to on-the-fly diff
- `y::AbstractVector`: y-values (length n)
- `bc::QuadraticBC`: Boundary condition (Left, Right, or MinCurvFit)

# Type Parameters
- `Tg`: Grid type
- `Tv`: Value type (unconstrained)
"""
function _compute_quadratic_coeffs!(
        d::AbstractVector,
        a::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector,
        bc::QuadraticBC
    ) where {Tg}
    s = _QuadraticSecants{eltype(d)}(y, x)
    _fill_slopes!(d, s, x, bc, x, y, a)
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

Secants are recomputed on read — no intermediate buffer.

For repeated interpolation on the same grid, use `QuadraticInterpolant`
which stores precomputed coefficients.
"""
function _compute_quadratic_coeffs(
        x::AbstractVector{Tg},
        y::AbstractVector,
        bc::QuadraticBC
    ) where {Tg}
    nx = length(x)

    # d is order-1 ([Y/X], `_coeff_op`); a is order-2 ([Y/X²], `_coeff_op2`).
    # Real grids collapse both to one float — the historic single witness.
    Tcoeff = _promote_eltype(_coeff_op, Tg, eltype(y))
    Ta = _promote_eltype(_coeff_op2, Tg, eltype(y))
    d = Vector{Tcoeff}(undef, nx)
    a = Vector{Ta}(undef, nx - 1)

    # Fill using in-place version
    _compute_quadratic_coeffs!(d, a, x, y, bc)

    return d, a
end
