# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    QUADRATIC SPLINE INTERPOLATION API                     ║
# ║              C1 piecewise quadratic with single-endpoint BC               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Internal Evaluation Functions
# ========================================

"""
    _quadratic_eval_at_point(x, y, cache, a, d, xi, extrap, op)

Core quadratic spline evaluation at a single point.
Uses precomputed coefficients (a, d) from BC.
"""
@inline function _quadratic_eval_at_point(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    cache::QuadraticSplineCache{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    xi::FT,
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    @boundscheck _check_domain(x, xi, extrap)

    x_min, x_max = first(x), last(x)

    # Boundary special case: xi == x[end]
    # Use last interval (n-1): a[end], d[end-1], y[end-1], dt=h[end]
    if xi == x_max
        dt = cache.h[end]
        @inbounds return _quadratic_kernel(op, a[end], d[end-1], y[end-1], dt)
    end

    # Extrapolation handling
    if xi < x_min
        return _quadratic_eval_extrap(y, a, d, cache.h, xi, x_min, Val(:left), extrap, op)
    elseif xi > x_max
        return _quadratic_eval_extrap(y, a, d, cache.h, xi, x_max, Val(:right), extrap, op)
    end

    # Normal case: interval search and kernel evaluation
    idx, x0, _ = _find_interval_with_bounds(x, xi)
    dt = xi - x0
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], dt)
end

"""
    _quadratic_eval_extrap(y, a, d, h, xi, x_bound, side, extrap, op)

Handle extrapolation for quadratic spline.
"""
@inline function _quadratic_eval_extrap(
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    h::AbstractVector{FT},
    xi::FT,
    x_bound::FT,
    ::Val{:left},
    ::Val{:constant},
    ::EvalValue
) where {FT<:AbstractFloat}
    @inbounds return y[1]
end

@inline function _quadratic_eval_extrap(
    y::AbstractVector{FT},
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::FT,
    ::FT,
    ::Val{:right},
    ::Val{:constant},
    ::EvalValue
) where {FT<:AbstractFloat}
    @inbounds return y[end]
end

# Constant extrapolation for derivatives - zero
@inline function _quadratic_eval_extrap(
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::FT,
    ::FT,
    ::Val,
    ::Val{:constant},
    ::EvalDeriv1
) where {FT<:AbstractFloat}
    return zero(FT)
end

@inline function _quadratic_eval_extrap(
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::AbstractVector{FT},
    ::FT,
    ::FT,
    ::Val,
    ::Val{:constant},
    ::EvalDeriv2
) where {FT<:AbstractFloat}
    return zero(FT)
end

# Extension extrapolation - continue first/last interval polynomial
@inline function _quadratic_eval_extrap(
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    ::AbstractVector{FT},
    xi::FT,
    x_bound::FT,
    ::Val{:left},
    ::Val{:extension},
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    dt = xi - x_bound  # negative
    @inbounds return _quadratic_kernel(op, a[1], d[1], y[1], dt)
end

@inline function _quadratic_eval_extrap(
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    h::AbstractVector{FT},
    xi::FT,
    x_bound::FT,
    ::Val{:right},
    ::Val{:extension},
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    n = length(y)
    dt = xi - (x_bound - h[end])  # distance from x[n-1]
    @inbounds return _quadratic_kernel(op, a[end], d[end], y[end-1], dt)
end


# ========================================
# Coefficient Computation Wrapper
# ========================================

"""
    _compute_quadratic_coeffs(y, cache, bc)

Compute quadratic spline coefficients (s, d, a) from BC.
Uses AdaptiveArrayPools for temporary arrays.
"""
function _compute_quadratic_coeffs(
    y::AbstractVector{FT},
    cache::QuadraticSplineCache{FT},
    bc::Union{Left{FT}, Right{FT}}
) where {FT<:AbstractFloat}
    n = length(y)
    nm1 = n - 1

    # Allocate coefficient arrays
    s = Vector{FT}(undef, nm1)
    d = Vector{FT}(undef, n)
    a = Vector{FT}(undef, nm1)

    # 1. Compute secants
    _compute_quadratic_secants!(s, y, cache.inv_h)

    # 2. Compute d[1] from BC
    d1 = _compute_d1_from_bc(bc, s, cache.h, n)

    # 3. Forward recurrence to fill d[]
    _forward_recurrence!(d, s, d1)

    # 4. Compute quadratic coefficients a[]
    _compute_quadratic_coefficients!(a, d, s, cache.inv_h)

    return s, d, a
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    quadratic_interp(x, y, xi; bc=Left(Deriv2(0)), extrap=:none, deriv=0)

C1 piecewise quadratic spline interpolation at a single point.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (same length as x)
- `xi::Real`: Query point
- `bc::Union{Left,Right}`: Boundary condition at one endpoint
  - `Left(Deriv1(v))`: First derivative = v at left endpoint
  - `Left(Deriv2(v))`: Second derivative = v at left endpoint (default: v=0)
  - `Right(Deriv1(v))`: First derivative = v at right endpoint
  - `Right(Deriv2(v))`: Second derivative = v at right endpoint
- `extrap::Symbol`: Extrapolation mode
  - `:none` (default): throws DomainError if outside domain
  - `:constant`: clamp to boundary values
  - `:extension`: extend the boundary polynomial
- `deriv::Int`: Derivative order (0, 1, or 2)

# Returns
- Interpolated value (Float type)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2  # [0, 1, 4, 9]

# Default: natural at left (Deriv2=0)
quadratic_interp(x, y, 1.5)  # ≈ 2.25

# With specific BC
quadratic_interp(x, y, 1.5; bc=Left(Deriv1(0.0)))  # zero slope at left
quadratic_interp(x, y, 1.5; bc=Right(Deriv1(6.0))) # slope=6 at right

# Derivatives
quadratic_interp(x, y, 1.5; deriv=1)  # ≈ 3.0 (slope at x=1.5)
quadratic_interp(x, y, 1.5; deriv=2)  # ≈ 2.0 (curvature)
```
"""
@inline function quadratic_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xi::FT;
    bc::Union{Left{FT}, Right{FT}}=Left(Deriv2(zero(FT))),
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))
    @boundscheck length(x) >= 2 || throw(ArgumentError("x must have at least 2 elements"))

    # Get/create cache
    cache = _get_quadratic_cache(x)

    # Compute coefficients from BC
    _, d, a = _compute_quadratic_coeffs(y, cache, bc)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _quadratic_eval_at_point(x, y, cache, a, d, xi, ev, op)
        end
    end
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    quadratic_interp!(output, x, y, x_targets; bc=Left(Deriv2(0)), extrap=:none, deriv=0)

In-place quadratic spline interpolation for multiple query points.

# Arguments
- `output`: Pre-allocated output vector
- Other arguments same as `quadratic_interp`

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2
out = zeros(3)
quadratic_interp!(out, x, y, [0.5, 1.5, 2.5])
# out ≈ [0.25, 2.25, 6.25]
```
"""
function quadratic_interp!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    bc::Union{Left{FT}, Right{FT}}=Left(Deriv2(zero(FT))),
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert length(x) >= 2 "x must have at least 2 elements"

    # Get/create cache
    cache = _get_quadratic_cache(x)

    # Compute coefficients from BC
    _, d, a = _compute_quadratic_coeffs(y, cache, bc)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x, x_targets, ev)
            @inbounds for i in eachindex(x_targets, output)
                output[i] = _quadratic_eval_at_point(x, y, cache, a, d, x_targets[i], ev, op)
            end
        end
    end
    return output
end

# ========================================
# Vector interpolation (allocating)
# ========================================

"""
    quadratic_interp(x, y, x_targets; bc=Left(Deriv2(0)), extrap=:none, deriv=0)

Quadratic spline interpolation for multiple query points (allocating version).

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2
result = quadratic_interp(x, y, [0.5, 1.5, 2.5])
# result ≈ [0.25, 2.25, 6.25]
```
"""
function quadratic_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    bc::Union{Left{FT}, Right{FT}}=Left(Deriv2(zero(FT))),
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    output = Vector{FT}(undef, length(x_targets))
    quadratic_interp!(output, x, y, x_targets; bc, extrap, deriv)
    return output
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GENERIC WRAPPERS - CONVENIENCE                        ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# BC Type Promotion Helper
# ========================================

"""Promote BC to target float type FT"""
@inline function _promote_bc(bc::Left{T, Deriv1{T}}, ::Type{FT}) where {T<:AbstractFloat, FT<:AbstractFloat}
    Left(Deriv1(FT(bc.bc.val)))
end

@inline function _promote_bc(bc::Left{T, Deriv2{T}}, ::Type{FT}) where {T<:AbstractFloat, FT<:AbstractFloat}
    Left(Deriv2(FT(bc.bc.val)))
end

@inline function _promote_bc(bc::Right{T, Deriv1{T}}, ::Type{FT}) where {T<:AbstractFloat, FT<:AbstractFloat}
    Right(Deriv1(FT(bc.bc.val)))
end

@inline function _promote_bc(bc::Right{T, Deriv2{T}}, ::Type{FT}) where {T<:AbstractFloat, FT<:AbstractFloat}
    Right(Deriv2(FT(bc.bc.val)))
end

# ========================================
# Scalar Real → Float wrappers
# ========================================

@inline function quadratic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xi::S;
    bc::Union{Left, Right}=Left(Deriv2(zero(T))),
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    bc_promoted = _promote_bc(bc, FT)
    return quadratic_interp(_to_float(x, FT), _to_float(y, FT), FT(xi); bc=bc_promoted, extrap, deriv)
end

# ========================================
# Vector Real → Float wrappers (allocating)
# ========================================

function quadratic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    bc::Union{Left, Right}=Left(Deriv2(zero(T))),
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    output = Vector{FT}(undef, length(x_targets))
    bc_promoted = _promote_bc(bc, FT)
    quadratic_interp!(output, _to_float(x, FT), _to_float(y, FT), _to_float(x_targets, FT); bc=bc_promoted, extrap, deriv)
    return output
end

# ========================================
# Vector Real → Float wrappers (in-place)
# ========================================

function quadratic_interp!(
    output::AbstractVector,
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    bc::Union{Left, Right}=Left(Deriv2(zero(T))),
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    FT = float(T)
    x_float = _to_float(x, FT)
    y_float = _to_float(y, FT)
    x_targets_float = _to_float(x_targets, FT)
    bc_promoted = _promote_bc(bc, FT)

    # Get/create cache
    cache = _get_quadratic_cache(x_float)

    # Compute coefficients
    _, d, a = _compute_quadratic_coeffs(y_float, cache, bc_promoted)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x_float, x_targets_float, ev)
            @inbounds for i in eachindex(x_targets_float, output)
                output[i] = _quadratic_eval_at_point(x_float, y_float, cache, a, d, x_targets_float[i], ev, op)
            end
        end
    end
    return output
end
