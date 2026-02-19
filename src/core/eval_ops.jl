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

"""
    ExtrapVal

Union type for extrapolation mode values.
Using concrete Union enables Julia's union-splitting optimization.
"""
const ExtrapVal = Union{Val{:none}, Val{:constant}, Val{:extension}, Val{:wrap}}

# ========================================
# Typed Extrapolation Mode Tags
# ========================================
#
# Compile-time type tags for extrapolation mode selection.
# These replace runtime Symbol dispatch (:none, :constant, etc.)
# with zero-cost type dispatch at the API boundary.
#
# Flow: User passes NoExtrap() → _extrap_to_val → Val(:none) → internal dispatch

"""
    AbstractExtrapMode

Abstract type for typed extrapolation mode specification.
Use concrete subtypes at the API boundary for compile-time dispatch
instead of runtime Symbol comparison.

# Concrete subtypes
- [`NoExtrap`](@ref): Throw `DomainError` for out-of-domain queries
- [`ConstExtrap`](@ref): Clamp to nearest boundary value
- [`ExtendExtrap`](@ref): Extend interpolation polynomial beyond domain
- [`WrapExtrap`](@ref): Wrap queries into domain (periodic)
"""
abstract type AbstractExtrapMode end

"""
    NoExtrap <: AbstractExtrapMode

No extrapolation — throws `DomainError` for out-of-domain queries.
Replaces `extrap=:none`.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=NoExtrap())
```
"""
struct NoExtrap <: AbstractExtrapMode end

"""
    ConstExtrap <: AbstractExtrapMode

Constant extrapolation — clamps queries to the nearest boundary value.
Replaces `extrap=:constant`.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=ConstExtrap())
```
"""
struct ConstExtrap <: AbstractExtrapMode end

"""
    ExtendExtrap <: AbstractExtrapMode

Extension extrapolation — extends the interpolation polynomial beyond the domain.
Replaces `extrap=:extension`.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=ExtendExtrap())
```
"""
struct ExtendExtrap <: AbstractExtrapMode end

"""
    WrapExtrap <: AbstractExtrapMode

Wrap extrapolation — wraps queries into the domain using modular arithmetic.
For periodic data. Replaces `extrap=:wrap`.

# Example
```julia
itp = cubic_interp((x, y), data; extrap=WrapExtrap())
```
"""
struct WrapExtrap <: AbstractExtrapMode end

"""
    _extrap_to_val(::AbstractExtrapMode) -> Val

Convert an `AbstractExtrapMode` singleton to the corresponding internal `Val` type.
Each method returns a concrete type, enabling compile-time specialization.
"""
@inline _extrap_to_val(::NoExtrap) = Val(:none)
@inline _extrap_to_val(::ConstExtrap) = Val(:constant)
@inline _extrap_to_val(::ExtendExtrap) = Val(:extension)
@inline _extrap_to_val(::WrapExtrap) = Val(:wrap)

"""
    _symbol_to_extrap_mode(extrap::Symbol) -> AbstractExtrapMode

Convert a Symbol extrapolation specifier to the corresponding `AbstractExtrapMode` singleton.
Used in the legacy Symbol → Mode conversion path.
"""
@inline function _symbol_to_extrap_mode(extrap::Symbol)
    extrap === :none && return NoExtrap()
    extrap === :constant && return ConstExtrap()
    extrap === :extension && return ExtendExtrap()
    extrap === :wrap && return WrapExtrap()
    throw(ArgumentError("`extrap` must be :none, :constant, :extension, or :wrap, got :$extrap"))
end

"""
    SideVal

Union type for side selection mode values (constant interpolation).
Using concrete Union enables Julia's union-splitting optimization.

Valid values: `Val(:nearest)`, `Val(:left)`, `Val(:right)`
"""
const SideVal = Union{Val{:nearest}, Val{:left}, Val{:right}}
