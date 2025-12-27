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
    ExtrapVal

Union type for extrapolation mode values.
Using concrete Union enables Julia's union-splitting optimization.
"""
const ExtrapVal = Union{Val{:none}, Val{:constant}, Val{:extension}, Val{:wrap}}
