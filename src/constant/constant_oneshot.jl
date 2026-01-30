# ========================================
# Constant Interpolation Oneshot API
# ========================================
# Zero-allocation constant interpolation functions.
# Type definitions in constant_types.jl.
# Callable methods (2-arg form) in constant_interpolant.jl.

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

Type parameters:
- Tg: Grid type (AbstractFloat) for xi, x_min, x_max
- Tv: Value type for y (can be Tg, Complex{Tg}, etc.)
"""
@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, xi::Tg, x_min::Tg, x_max::Tg,
    ::Val{:constant}, ::SideVal, ::EvalValue
) where {Tg<:AbstractFloat, Tv}
    if xi < x_min
        return @inbounds y[1]
    else  # xi > x_max
        return @inbounds y[end]
    end
end

@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, ::Tg, ::Tg, ::Tg,
    ::Val{:constant}, ::SideVal, ::EvalDeriv1
) where {Tg<:AbstractFloat, Tv}
    return zero(Tv)
end

@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, ::Tg, ::Tg, ::Tg,
    ::Val{:constant}, ::SideVal, ::EvalDeriv2
) where {Tg<:AbstractFloat, Tv}
    return zero(Tv)
end

@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, ::Tg, ::Tg, ::Tg,
    ::Val{:constant}, ::SideVal, ::EvalDeriv3
) where {Tg<:AbstractFloat, Tv}
    return zero(Tv)
end

# :extension delegates to :constant (slope=0 for constant function)
@inline function _constant_eval_extrap(
    y::AbstractVector{Tv}, xi::Tg, x_min::Tg, x_max::Tg,
    ::Val{:extension}, side::SideVal, op::AbstractEvalOp
) where {Tg<:AbstractFloat, Tv}
    return _constant_eval_extrap(y, xi, x_min, x_max, Val(:constant), side, op)
end


"""
    _constant_eval_at_point(x, y, xi, extrap, side, op, searcher)

Core constant interpolation evaluation with search policy.

Evaluation flow:
1. Domain check (:none mode → DomainError if outside)
2. :wrap mode → wrap to [x_min, x_max) and evaluate
3. Boundary check (xi == x_max → y[end] for non-wrap modes)
4. Extrapolation check (xi outside domain → extrap handling)
5. Interval search → kernel evaluation

Type parameters:
- Tg: Grid type (AbstractFloat) for x and xi
- Tv: Value type for y (can be Tg, Complex{Tg}, etc.)
"""
@inline function _constant_eval_at_point(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xi::Tg,
    extrap::ExtrapVal,
    side::SideVal,
    op::AbstractEvalOp,
    searcher::S
) where {Tg<:AbstractFloat, Tv, S<:Searcher}
    # Domain check for :none mode (throws DomainError)
    @boundscheck _check_domain(x, xi, extrap)

    x_min, x_max = first(x), last(x)

    # :wrap mode handles all cases (inside and outside domain)
    if extrap === Val(:wrap)
        xi_wrapped = _wrap_to_domain(xi, x_min, x_max)
        idx, xL, xR = search_interval(searcher, x, xi_wrapped)
        h = xR - xL
        dL = xi_wrapped - xL
        @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dL, side)
    end

    # Boundary special case: xi == x[end] → y[end] directly
    # (avoids _search_interval returning idx=n-1, dL=h)
    if xi == x_max
        return op isa EvalValue ? (@inbounds y[end]) : zero(Tv)
    end

    # Extrapolation handling (:constant, :extension)
    if xi < x_min || xi > x_max
        return _constant_eval_extrap(y, xi, x_min, x_max, extrap, side, op)
    end

    # Normal case: interval search and kernel evaluation
    idx, xL, xR = search_interval(searcher, x, xi)
    h = xR - xL
    dL = xi - xL
    @inbounds return _constant_kernel(op, y[idx], y[idx+1], h, dL, side)
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         PUBLIC API - HOT PATH                             ║
# ║                 Tg<:AbstractFloat grid, Tv value type                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar interpolation
# ========================================

"""
    constant_interp(x, y, xi; extrap=:none, side=:nearest, deriv=0, search=Binary())

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
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `Binary()` (default): O(log n) binary search, stateless
  - `HintedBinary()`: O(1) if hint valid, O(log n) fallback
  - `LinearBinary(linear_window=8)`: Linear search within window, then binary fallback

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

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = constant_interp(x, y, sorted_queries; search=LinearBinary(linear_window=8))
```
"""
@inline function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xi::Tg;
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0,
    search=Binary(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                _constant_eval_at_point(x, y, xi, ev, sv, op, searcher)
            end
        end
    end
end

# ========================================
# Vector interpolation (in-place)
# ========================================

"""
    constant_interp!(output, x, y, x_targets; extrap=:none, side=:nearest, deriv=0, search=Binary())

Zero-allocation constant interpolation for multiple query points.

# Arguments
- `output`: Pre-allocated output vector
- `x, y, x_targets`: Grid and query points
- `extrap, side, deriv`: Same as `constant_interp`
- `search::AbstractSearchPolicy`: Search algorithm for interval finding

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]
out = zeros(3)
constant_interp!(out, x, y, [0.5, 1.5, 2.5])
# out == [10.0, 20.0, 30.0]

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
output = zeros(1000)
constant_interp!(output, x, y, sorted_queries; search=LinearBinary(linear_window=8))
```
"""
function constant_interp!(
    output::AbstractVector{Tv},
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tg};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                @boundscheck _check_domain(x, x_targets, ev)
                @inbounds for i in eachindex(x_targets, output)
                    output[i] = _constant_eval_at_point(x, y, x_targets[i], ev, sv, op, searcher)
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
    constant_interp(x, y, x_targets; extrap=:none, side=:nearest, deriv=0, search=Binary())

Constant interpolation for multiple query points (allocating version).

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]
result = constant_interp(x, y, [0.5, 1.5, 2.5])
# result == [10.0, 20.0, 30.0]

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = constant_interp(x, y, sorted_queries; search=LinearBinary(linear_window=8))
```
"""
function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    x_targets::AbstractVector{Tg};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tg<:AbstractFloat, Tv}
    output = Vector{Tv}(undef, length(x_targets))
    constant_interp!(output, x, y, x_targets; extrap, side, deriv, search)
    return output
end


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GENERIC WRAPPERS - CONVENIENCE                        ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Scalar Real/Complex → typed wrappers
# ========================================
# Unified wrapper for non-AbstractFloat inputs (Int, mixed types, Complex, etc.)
# POLICY: Tg is computed from x/y ONLY, not from xq

@inline function constant_interp(
    x::AbstractVector{Tx},
    y::AbstractVector{Ty},
    xi::Tq;
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0,
    search=Binary(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tx<:Real, Ty, Tq<:Real}
    # Tg from x/y ONLY (not xq)
    Tg = float(promote_type(Tx, _real_eltype(Ty)))
    x_typed, y_typed = _promote_xy(x, y, Tg)
    return constant_interp(x_typed, y_typed, Tg(xi); extrap, side, deriv, search, hint)
end

# ========================================
# Vector Real/Complex → typed wrappers (allocating)
# ========================================
# POLICY: Tg is computed from x/y ONLY, not from x_targets

function constant_interp(
    x::AbstractVector{Tx},
    y::AbstractVector{Ty},
    x_targets::AbstractVector{Tq};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty, Tq<:Real}
    # Tg from x/y ONLY (not x_targets)
    Tg = float(promote_type(Tx, _real_eltype(Ty)))
    Tv = _value_type(Ty, Tg)
    output = Vector{Tv}(undef, length(x_targets))
    x_typed, y_typed = _promote_xy(x, y, Tg)
    targets_typed = Tg.(x_targets)
    constant_interp!(output, x_typed, y_typed, targets_typed; extrap, side, deriv, search)
    return output
end

# ========================================
# Vector Real/Complex → typed wrappers (in-place)
# ========================================

function constant_interp!(
    output::AbstractVector,
    x::AbstractVector{Tx},
    y::AbstractVector{Ty},
    x_targets::AbstractVector{Tq};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {Tx<:Real, Ty, Tq<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    # Tg from x/y ONLY (not x_targets)
    Tg = float(promote_type(Tx, _real_eltype(Ty)))

    # Determine expected output type and validate
    Tv = _value_type(Ty, Tg)
    Tout = eltype(output)
    if !(Tout >: Tv)
        throw(ArgumentError(
            "output eltype $Tout cannot hold interpolation result type $Tv. " *
            "Use Vector{$Tv} or a wider type (e.g., Vector{Complex{$Tg}} for complex y-values)."
        ))
    end

    x_typed, y_typed = _promote_xy(x, y, Tg)
    targets_typed = Tg.(x_targets)

    constant_interp!(output, x_typed, y_typed, targets_typed; extrap, side, deriv, search)
end
