# ========================================
# Evaluation Operation Types
# ========================================
# Shared types for compile-time dispatch of value/derivative evaluation.
# Must be included first - used by all interpolation files.

"""
    AbstractEvalOp

Abstract type for evaluation operations (value, derivatives).
Used for compile-time dispatch to select appropriate kernel.

Subtypes:
- `EvalValue`: Evaluate function value f(x)
- `EvalDeriv1`: Evaluate first derivative f'(x)
- `EvalDeriv2`: Evaluate second derivative f''(x)
- `EvalDeriv3`: Evaluate third derivative f'''(x)
"""
abstract type AbstractEvalOp end

"""
    EvalValue <: AbstractEvalOp

Singleton type indicating evaluation of function value f(x).
"""
struct EvalValue <: AbstractEvalOp end

"""
    EvalDeriv1 <: AbstractEvalOp

Singleton type indicating evaluation of first derivative f'(x).
"""
struct EvalDeriv1 <: AbstractEvalOp end

"""
    EvalDeriv2 <: AbstractEvalOp

Singleton type indicating evaluation of second derivative f''(x).
"""
struct EvalDeriv2 <: AbstractEvalOp end

"""
    EvalDeriv3 <: AbstractEvalOp

Singleton type indicating evaluation of third derivative f'''(x).

# Note
For cubic splines, S'''(x) is constant within each interval.
Linear/quadratic/constant interpolants always return zero.

# Mathematical Background
The cubic spline third derivative is:
    S'''(x) = (zR - zL) / h

where zL and zR are the second derivative values (moments) at the
interval endpoints, and h is the interval width.
"""
struct EvalDeriv3 <: AbstractEvalOp end

# ========================================
# Typed Extrapolation Mode Tags
# ========================================
#
# Compile-time type tags for extrapolation mode selection.
# Zero-cost type dispatch at the API boundary.
#
# 1D: Structs store E<:AbstractExtrap, dispatch on concrete subtypes
# ND: Structs store NTuple{N,AbstractExtrap}, resolved via _resolve_extrap_nd

"""
    AbstractExtrap

Abstract type for typed extrapolation mode specification.
Use concrete subtypes at the API boundary for compile-time dispatch.

# Concrete subtypes
- [`NoExtrap`](@ref): Throw `DomainError` for out-of-domain queries
- [`ConstExtrap`](@ref): Clamp to nearest boundary value
- [`ExtendExtrap`](@ref): Extend interpolation polynomial beyond domain
- [`WrapExtrap`](@ref): Wrap queries into domain (periodic)

!!! warning "Union-splitting invariant"
    Keep exactly 4 concrete subtypes. Julia's compiler union-splits up to 4 types
    on hot paths; adding a 5th subtype would cause dynamic dispatch and allocation
    in all oneshot/eval code paths.
"""
abstract type AbstractExtrap end

"""
    NoExtrap <: AbstractExtrap

No extrapolation — throws `DomainError` for out-of-domain queries.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=NoExtrap())
```
"""
struct NoExtrap <: AbstractExtrap end

"""
    ConstExtrap <: AbstractExtrap

Constant extrapolation — clamps queries to the nearest boundary value.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=ConstExtrap())
```
"""
struct ConstExtrap <: AbstractExtrap end

"""
    ExtendExtrap <: AbstractExtrap

Extension extrapolation — extends the interpolation polynomial beyond the domain.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=ExtendExtrap())
```
"""
struct ExtendExtrap <: AbstractExtrap end

"""
    WrapExtrap <: AbstractExtrap

Wrap extrapolation — wraps queries into the domain using modular arithmetic.
For periodic data.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=WrapExtrap())
```
"""
struct WrapExtrap <: AbstractExtrap end

# ========================================
# Typed Side Selection Tags (Constant Interpolation)
# ========================================
#
# Compile-time type tags for side mode selection.
# Zero-cost type dispatch — no macros needed with proper types.
#
# 1D: Structs store SD<:AbstractSide, dispatch on concrete subtypes
# ND: Structs store Tuple{Vararg{AbstractSide, N}}, resolved via _resolve_side_nd

"""
    AbstractSide

Abstract type for side selection mode in constant interpolation.
Determines which neighbor value to use at non-grid-point locations.

# Concrete subtypes
- [`NearestSide`](@ref): Nearest neighbor with left tie-breaking at midpoint
- [`LeftSide`](@ref): Always use left (floor) value
- [`RightSide`](@ref): Use right (ceiling) value, except at grid points

!!! info "Union-splitting guarantee"
    With exactly 3 concrete subtypes (< 4 limit), Julia union-splits
    automatically on hot paths. No manual dispatch macros needed.
"""
abstract type AbstractSide end

"""
    NearestSide <: AbstractSide

Nearest-neighbor side selection with left tie-breaking at midpoint.
Returns left value if distance <= h/2, right value otherwise.

# Example
```julia
itp = constant_interp(x, y; side=NearestSide())
```
"""
struct NearestSide <: AbstractSide end

"""
    LeftSide <: AbstractSide

Left-continuous (floor) side selection. Always returns the left boundary value.

# Example
```julia
itp = constant_interp(x, y; side=LeftSide())
```
"""
struct LeftSide <: AbstractSide end

"""
    RightSide <: AbstractSide

Right-continuous (ceiling) side selection.
Returns right value except at grid points (where it returns left value).

# Example
```julia
itp = constant_interp(x, y; side=RightSide())
```
"""
struct RightSide <: AbstractSide end
