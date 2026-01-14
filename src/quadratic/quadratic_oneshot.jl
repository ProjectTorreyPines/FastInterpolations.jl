# ========================================
# Quadratic Interpolation Oneshot API
# ========================================
# Zero-allocation quadratic spline interpolation functions.
# QuadraticBC type alias and _compute_quadratic_coeffs are in quadratic_solver.jl.
# Type definitions in quadratic_types.jl.
# Callable methods (2-arg form) in quadratic_interpolant.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    QUADRATIC SPLINE INTERPOLATION API                     ║
# ║              C1 piecewise quadratic with single-endpoint BC               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Internal Evaluation Functions
# ========================================

# Note: _constant_extrap_result helper is defined in cubic_eval.jl (shared)

"""
    _quadratic_eval_core(x, y, a, d, xi, op)

Core quadratic spline evaluation at a single point.
Uses interval clamping for extension extrapolation (matches cubic pattern).
"""
@inline function _quadratic_eval_core(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    xi::FT,
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    # _find_interval clamps idx to [1, n-1]
    # This handles both normal evaluation and extension extrapolation
    idx, xL, _ = _find_interval(x, xi)
    dt = xi - xL
    @inbounds return _quadratic_kernel(op, a[idx], d[idx], y[idx], dt)
end

# ========================================
# Extrapolation-aware Evaluation (matches cubic pattern)
# ========================================

"Evaluate with no extrapolation - throws DomainError if outside domain."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    xi::FT,
    ::Val{:none},
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    return _quadratic_eval_core(x, y, a, d, xi, op)
end

"Evaluate with constant extrapolation - returns boundary values outside domain."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    xi::FT,
    ::Val{:constant},
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    xi < first(x) && return _constant_extrap_result(op, @inbounds y[1])
    xi > last(x) && return _constant_extrap_result(op, @inbounds y[end])
    return _quadratic_eval_core(x, y, a, d, xi, op)
end

"Evaluate with extension extrapolation - extends boundary polynomial."
@inline function _quadratic_eval_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    xi::FT,
    ::Val{:extension},
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    # Interval clamping in _find_interval handles extension
    return _quadratic_eval_core(x, y, a, d, xi, op)
end

"""
    _quadratic_eval_at_point(x, y, h, a, d, xi, extrap, op)

Entry point for quadratic spline evaluation with extrapolation dispatch.
Note: `h` parameter kept for API compatibility but not used (interval info from x).
"""
@inline function _quadratic_eval_at_point(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    ::AbstractVector{FT},  # h - unused, kept for API compatibility
    a::AbstractVector{FT},
    d::AbstractVector{FT},
    xi::FT,
    extrap::ExtrapVal,
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    @boundscheck _check_domain(x, xi, extrap)
    return _quadratic_eval_with_extrap(x, y, a, d, xi, extrap, op)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    quadratic_interp(x, y, xi; bc=Left(ParabolaFit()), extrap=:none, deriv=0)

C1 piecewise quadratic spline interpolation at a single point.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (same length as x)
- `xi::Real`: Query point
- `bc`: Boundary condition (one of):
  - `Left(ParabolaFit())`: 3-point parabola fit at left (default, exact for polynomials)
  - `Right(ParabolaFit())`: 3-point parabola fit at right
  - `Left(Deriv1(v))`: First derivative = v at left endpoint
  - `Left(Deriv2(v))`: Second derivative = v at left endpoint
  - `Right(Deriv1(v))`: First derivative = v at right endpoint
  - `Right(Deriv2(v))`: Second derivative = v at right endpoint
  - `MinCurvFit()`: Minimize total curvature (globally smooth)
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

# Default: ParabolaFit (exact for quadratic polynomials)
quadratic_interp(x, y, 1.5)  # ≈ 2.25 (exact)

# With specific BC
quadratic_interp(x, y, 1.5; bc=Left(Deriv1(0.0)))  # zero slope at left
quadratic_interp(x, y, 1.5; bc=MinCurvFit())        # minimize curvature

# Derivatives
quadratic_interp(x, y, 1.5; deriv=1)  # ≈ 3.0 (slope at x=1.5)
quadratic_interp(x, y, 1.5; deriv=2)  # ≈ 2.0 (curvature)
```
"""
@inline @with_pool pool function quadratic_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xi::FT;
    bc::QuadraticBC{FT}=Left(ParabolaFit{FT}()),
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))
    @boundscheck length(x) >= 2 || throw(ArgumentError("x must have at least 2 elements"))

    # Compute coefficients using temporary arrays from pool
    nx = length(x)
    h = acquire!(pool, FT, nx-1)
    d = acquire!(pool, FT, nx)
    a = acquire!(pool, FT, nx-1)
    _compute_quadratic_coeffs!(h, d, a, x, y, bc)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _quadratic_eval_at_point(x, y, h, a, d, xi, ev, op)
        end
    end
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    quadratic_interp!(output, x, y, x_targets; bc=Left(ParabolaFit()), extrap=:none, deriv=0)

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
@with_pool pool function quadratic_interp!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    bc::QuadraticBC{FT}=Left(ParabolaFit{FT}()),
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert length(x) >= 2 "x must have at least 2 elements"

    # Compute coefficients using temporary arrays from pool
    nx = length(x)
    h = acquire!(pool, FT, nx-1)
    d = acquire!(pool, FT, nx)
    a = acquire!(pool, FT, nx-1)
    _compute_quadratic_coeffs!(h, d, a, x, y, bc)


    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x, x_targets, ev)
            @inbounds for i in eachindex(x_targets, output)
                output[i] = _quadratic_eval_at_point(x, y, h, a, d, x_targets[i], ev, op)
            end
        end
    end
    return output
end

# ========================================
# Vector interpolation (allocating)
# ========================================

"""
    quadratic_interp(x, y, x_targets; bc=Left(ParabolaFit()), extrap=:none, deriv=0)

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
    bc::QuadraticBC{FT}=Left(ParabolaFit{FT}()),
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

"""
    _promote_bc(bc::Left/Right, ::Type{FT}) -> Left{FT}/Right{FT}

Promote Left/Right BC wrapper to target float type FT.
Uses _promote_pointbc from bc_types.jl for inner BC promotion.
"""
# Same-type passthrough (zero-cost, no object creation)
@inline _promote_bc(bc::Left{T, <:PointBC{T}}, ::Type{T}) where {T<:AbstractFloat} = bc
@inline _promote_bc(bc::Right{T, <:PointBC{T}}, ::Type{T}) where {T<:AbstractFloat} = bc

# Type conversion (delegates to _promote_pointbc)
@inline function _promote_bc(bc::Left{T, <:PointBC{T}}, ::Type{FT}) where {T<:AbstractFloat, FT<:AbstractFloat}
    Left(_promote_pointbc(bc.bc, FT))
end

@inline function _promote_bc(bc::Right{T, <:PointBC{T}}, ::Type{FT}) where {T<:AbstractFloat, FT<:AbstractFloat}
    Right(_promote_pointbc(bc.bc, FT))
end

# MinCurvFit promotion (not a PointBC, needs explicit handling)
@inline _promote_bc(::MinCurvFit, ::Type{T}) where {T<:AbstractFloat} = MinCurvFit{T}()
@inline _promote_bc(::MinCurvFit{T}, ::Type{T}) where {T<:AbstractFloat} = MinCurvFit{T}()

# Note: ParabolaFit <: PointBC, handled by generic _promote_pointbc in bc_types.jl

# ========================================
# Scalar Real → Float wrappers
# ========================================

@inline function quadratic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xi::S;
    bc::QuadraticBC{<:AbstractFloat}=Left(ParabolaFit{Float64}()),
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
    bc::QuadraticBC{<:AbstractFloat}=Left(ParabolaFit{Float64}()),
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

@with_pool pool function quadratic_interp!(
    output::AbstractVector,
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    bc::QuadraticBC{<:AbstractFloat}=Left(ParabolaFit{Float64}()),
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

    # Compute coefficients using temporary arrays from pool
    nx = length(x)
    h = acquire!(pool, FT, nx-1) 
    d = acquire!(pool, FT, nx)
    a = acquire!(pool, FT, nx-1)
    _compute_quadratic_coeffs!(h, d, a, x_float, y_float, bc_promoted)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x_float, x_targets_float, ev)
            @inbounds for i in eachindex(x_targets_float, output)
                output[i] = _quadratic_eval_at_point(x_float, y_float, h, a, d, x_targets_float[i], ev, op)
            end
        end
    end
    return output
end


