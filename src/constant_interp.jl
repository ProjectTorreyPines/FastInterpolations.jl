# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      CONSTANT (STEP) INTERPOLATION                        ║
# ║              Piecewise constant interpolation with side options           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Internal Evaluation Functions
# ========================================

"""
    _constant_eval_extrap(y, xi, x_min, x_max, extrap, side, op)

Handle extrapolation for constant interpolation.
- EvalValue: returns boundary value (y[1] or y[end])
- EvalDeriv1/EvalDeriv2: returns zero (constant function)

Note: :none and :wrap modes never reach this function.
- :none throws DomainError via _check_domain
- :wrap is handled in _constant_eval_at_point before extrap check
"""
@inline function _constant_eval_extrap(
    y::AbstractVector{FT}, xi::FT, x_min::FT, x_max::FT,
    ::Val{:constant}, ::SideVal, ::EvalValue
) where {FT<:AbstractFloat}
    if xi < x_min
        return @inbounds y[1]
    else  # xi > x_max
        return @inbounds y[end]
    end
end

@inline function _constant_eval_extrap(
    y::AbstractVector{FT}, ::FT, ::FT, ::FT,
    ::Val{:constant}, ::SideVal, ::EvalDeriv1
) where {FT<:AbstractFloat}
    return zero(FT)
end

@inline function _constant_eval_extrap(
    y::AbstractVector{FT}, ::FT, ::FT, ::FT,
    ::Val{:constant}, ::SideVal, ::EvalDeriv2
) where {FT<:AbstractFloat}
    return zero(FT)
end

# :extension delegates to :constant (slope=0 for constant function)
@inline function _constant_eval_extrap(
    y::AbstractVector{FT}, xi::FT, x_min::FT, x_max::FT,
    ::Val{:extension}, side::SideVal, op::AbstractEvalOp
) where {FT<:AbstractFloat}
    return _constant_eval_extrap(y, xi, x_min, x_max, Val(:constant), side, op)
end


"""
    _constant_eval_at_point(x, y, xi, extrap, side, op)

Core constant interpolation evaluation.

Evaluation flow:
1. Domain check (:none mode → DomainError if outside)
2. :wrap mode → wrap to [x_min, x_max) and evaluate
3. Boundary check (xi == x_max → y[end] for non-wrap modes)
4. Extrapolation check (xi outside domain → extrap handling)
5. Interval search → kernel evaluation
"""
@inline function _constant_eval_at_point(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xi::FT,
    extrap::ExtrapVal,
    side::SideVal,
    op::AbstractEvalOp
) where {FT<:AbstractFloat}
    # Domain check for :none mode (throws DomainError)
    @boundscheck _check_domain(x, xi, extrap)

    x_min, x_max = first(x), last(x)

    # :wrap mode handles all cases (inside and outside domain)
    if extrap === Val(:wrap)
        xi_wrapped = _wrap_to_domain(xi, x_min, x_max)
        idx, xL, xR = _find_interval(x, xi_wrapped)
        h = xR - xL
        dL = xi_wrapped - xL
        @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dL, side)
    end

    # Boundary special case: xi == x[end] → y[end] directly
    # (avoids _find_interval returning idx=n-1, dL=h)
    if xi == x_max
        return op isa EvalValue ? (@inbounds y[end]) : zero(FT)
    end

    # Extrapolation handling (:constant, :extension)
    if xi < x_min || xi > x_max
        return _constant_eval_extrap(y, xi, x_min, x_max, extrap, side, op)
    end

    # Normal case: interval search and kernel evaluation
    idx, xL, xR = _find_interval(x, xi)
    h = xR - xL
    dL = xi - xL
    @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dL, side)
end


# ========================================
# ConstantInterpolant Struct
# ========================================

"""
    ConstantInterpolant{T,X,Y}

Lightweight callable interpolant for constant (step) interpolation.
Returned by `constant_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `mode::ExtrapVal`: Extrapolation mode
- `side::SideVal`: Side selection (:nearest, :left, :right)

# Usage
```julia
itp = constant_interp(x, y)  # default: extrap=:none, side=:nearest
val = itp(0.5)               # scalar evaluation
vals = itp.(query_points)    # broadcast

# With options
itp_left = constant_interp(x, y; side=:left)
itp_wrap = constant_interp(x, y; extrap=:wrap, side=:right)
```
"""
struct ConstantInterpolant{T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
    x::X
    y::Y
    mode::ExtrapVal
    side::SideVal

    function ConstantInterpolant(
        x::X, y::Y;
        extrap::Symbol=:none,
        side::Symbol=:nearest
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"

        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                return new{T,X,Y}(x, y, ev, sv)
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# ─────────────────────────────────────────────────────────────
@inline function (itp::ConstantInterpolant{T})(xi::T; deriv::Int=0) where {T<:AbstractFloat}
    @_dispatch_deriv deriv => op begin
        _constant_eval_at_point(itp.x, itp.y, xi, itp.mode, itp.side, op)
    end
end

# Real scalar wrapper - delegates to T method
@inline function (itp::ConstantInterpolant{T})(xi::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); deriv=deriv)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T})(xi::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    xi_typed = _to_float(xi, T)
    output = Vector{T}(undef, length(xi_typed))
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.mode, itp.side, op)
        end
    end
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call (zero allocation)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi, itp.mode)
        @inbounds for i in eachindex(xi, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi[i], itp.mode, itp.side, op)
        end
    end
    return output
end

# In-place with type conversion
function (itp::ConstantInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = _to_float(xi, T)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.mode, itp.side, op)
        end
    end
    return output
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                  All arguments have same FT<:AbstractFloat                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    constant_interp(x, y, xi; extrap=:none, side=:nearest, deriv=0)

Constant (step/piecewise constant) interpolation at a single point.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (same length as x)
- `xi::Real`: Query point
- `extrap::Symbol`: Extrapolation mode
  - `:none` (default): throws DomainError if outside domain
  - `:constant`: clamp to boundary values
  - `:extension`: same as :constant (slope=0)
  - `:wrap`: wrap to [x_min, x_max)
- `side::Symbol`: Side selection
  - `:nearest` (default): nearest neighbor (left tie-breaking at midpoint)
  - `:left`: always use left value
  - `:right`: use right value (except at grid points)
- `deriv::Int`: Derivative order (0, 1, or 2). Derivatives are always 0.

# Returns
- Interpolated value (Float type)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

constant_interp(x, y, 0.5)                    # 10.0 (nearest to left)
constant_interp(x, y, 0.5; side=:left)        # 10.0
constant_interp(x, y, 0.5; side=:right)       # 20.0
constant_interp(x, y, 1.0)                    # 20.0 (grid point)
constant_interp(x, y, -1.0; extrap=:constant) # 10.0 (clamped)
```
"""
@inline function constant_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xi::FT;
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                _constant_eval_at_point(x, y, xi, ev, sv, op)
            end
        end
    end
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    constant_interp!(output, x, y, x_targets; extrap=:none, side=:nearest, deriv=0)

Zero-allocation constant interpolation for multiple query points.

# Arguments
- `output`: Pre-allocated output vector
- `x, y, x_targets`: Grid and query points
- `extrap, side, deriv`: Same as `constant_interp`

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]
out = zeros(3)
constant_interp!(out, x, y, [0.5, 1.5, 2.5])
# out == [10.0, 20.0, 30.0]
```
"""
function constant_interp!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                @boundscheck _check_domain(x, x_targets, ev)
                @inbounds for i in eachindex(x_targets, output)
                    output[i] = _constant_eval_at_point(x, y, x_targets[i], ev, sv, op)
                end
            end
        end
    end
    return output
end

# ========================================
# Vector interpolation (allocating)
# ========================================

"""
    constant_interp(x, y, x_targets; extrap=:none, side=:nearest, deriv=0)

Constant interpolation for multiple query points (allocating version).

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]
result = constant_interp(x, y, [0.5, 1.5, 2.5])
# result == [10.0, 20.0, 30.0]
```
"""
function constant_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0
) where {FT<:AbstractFloat}
    output = Vector{FT}(undef, length(x_targets))
    constant_interp!(output, x, y, x_targets; extrap, side, deriv)
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    constant_interp(x, y; extrap=:none, side=:nearest) -> ConstantInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values
- `extrap::Symbol`: Extrapolation mode
- `side::Symbol`: Side selection

# Returns
`ConstantInterpolant` object for scalar/broadcast evaluation.

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

itp = constant_interp(x, y)
itp(0.5)           # 10.0
itp.([0.5, 1.5])   # [10.0, 20.0]

# Fused broadcast (optimal)
result = @. coef * itp(query)
```
"""
function constant_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT};
    extrap::Symbol=:none,
    side::Symbol=:nearest
) where {FT<:AbstractFloat}
    return ConstantInterpolant(x, y; extrap, side)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GENERIC WRAPPERS - CONVENIENCE                        ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar Real → Float wrappers
# ========================================

@inline function constant_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xi::S;
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    return constant_interp(_to_float(x, FT), _to_float(y, FT), FT(xi); extrap, side, deriv)
end

# ========================================
# Vector Real → Float wrappers (allocating)
# ========================================

function constant_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    output = Vector{FT}(undef, length(x_targets))
    constant_interp!(output, x, y, x_targets; extrap, side, deriv)
    return output
end

# ========================================
# Vector Real → Float wrappers (in-place)
# ========================================

function constant_interp!(
    output::AbstractVector,
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    FT = float(T)
    x_float = _to_float(x, FT)
    y_float = _to_float(y, FT)
    x_targets_float = _to_float(x_targets, FT)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                @boundscheck _check_domain(x_float, x_targets_float, ev)
                @inbounds for i in eachindex(x_targets_float, output)
                    output[i] = _constant_eval_at_point(x_float, y_float, x_targets_float[i], ev, sv, op)
                end
            end
        end
    end
    return output
end

# ========================================
# 2-arg Callable Real → Float wrapper
# ========================================

function constant_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrap::Symbol=:none,
    side::Symbol=:nearest
) where {T<:Real}
    FT = float(T)
    return ConstantInterpolant(_to_float(x, FT), _to_float(y, FT); extrap, side)
end
