# Design: Interpolant Derivative API

**Status**: Implemented
**Last Updated**: 2025-12-27

---

## Overview

This document describes the derivative evaluation API for `CubicInterpolant` and `LinearInterpolant`.

### Design Goals

1. **Unified API**: Single pattern `itp(xi; order=N)` for all evaluations
2. **Zero-Allocation**: Scalar evaluations must be allocation-free
3. **Broadcast Support**: Enable `derivative(itp).(xs)` for vector operations
4. **HOF Composition**: Support `find_zero(derivative(itp), a, b)` patterns

---

## API Reference

### Core API: `order` Keyword

The canonical API uses an `order` keyword argument on the interpolant functor:

```julia
itp = cubic_interp(x, y)

# Scalar Evaluation
itp(xi)              # value (order=0, default)
itp(xi; order=1)     # first derivative
itp(xi; order=2)     # second derivative

# Vector Evaluation
itp(xs; order=1)     # first derivatives for all points

# In-place (zero-allocation)
itp(output, xs; order=1)
```

### Broadcast Support: `DerivativeView`

Since Julia doesn't support `itp.(xs; order=1)` syntax, a lightweight wrapper enables broadcast:

```julia
d1 = derivative(itp)   # DerivativeView wrapper
d2 = derivative2(itp)

# Broadcast
d1.(xs)                # equivalent to [itp(x; order=1) for x in xs]
@. coef * d1(xs)       # fused broadcast

# Higher-order functions
find_zero(derivative(itp), a, b)
```

### Complete API Table

| Operation | Scalar | Vector | In-place | Broadcast |
|-----------|--------|--------|----------|-----------|
| Value | `itp(xi)` | `itp(xs)` | `itp(out, xs)` | `itp.(xs)` |
| 1st deriv | `itp(xi; order=1)` | `itp(xs; order=1)` | `itp(out, xs; order=1)` | `derivative(itp).(xs)` |
| 2nd deriv | `itp(xi; order=2)` | `itp(xs; order=2)` | `itp(out, xs; order=2)` | `derivative2(itp).(xs)` |

---

## Design Rationale

### Why `order` Keyword?

The `order` keyword provides a unified interface:
- Consistent with `cubic_interp!(output, cache, y, xs; order=N)`
- Single method to remember for all derivative orders
- Compile-time dispatch via `@_dispatch_order` macro ensures zero-allocation

### Why `DerivativeView` Wrapper?

Julia syntax constraint: `itp.(xs; order=1)` is invalid.

The `DerivativeView` wrapper solves this:
- `derivative(itp)` returns a lightweight callable wrapper
- Wrapper delegates to `parent(xi; order=N)` internally
- Enables idiomatic Julia patterns like `d1.(xs)` and `find_zero(d1, a, b)`

### LinearInterpolant `order=2`

Returns `0.0` because piecewise linear functions have zero curvature everywhere (except Dirac delta impulses at knots, which cannot be represented as finite values).

---

## Implementation Notes

### Type Stability

The `@_dispatch_order` macro converts runtime `order::Int` to compile-time singleton types:

```julia
@_dispatch_order order => op begin
    _eval_with_bc(..., op)  # op is EvalValue(), EvalDeriv1(), or EvalDeriv2()
end
```

### Zero-Allocation Guarantee

Scalar evaluations are zero-allocation because:
- `EvalValue()`, `EvalDeriv1()`, `EvalDeriv2()` are singleton structs (0 bytes)
- `DerivativeView` is immutable with single reference field (stack-allocated)
- All evaluation functions are fully inlined

### Export Policy

```julia
export derivative, derivative2  # Factory functions (public)
# DerivativeView is NOT exported (internal implementation detail)
```

---

## Usage Recommendations

| Scenario | Recommended API |
|----------|-----------------|
| Single point | `itp(xi; order=1)` |
| Many points (performance) | `itp(output, xs; order=1)` |
| Fused broadcast | `d1 = derivative(itp); @. coef * d1(xs)` |
| Root finding | `find_zero(derivative(itp), a, b)` |

---

## File Organization

```
src/
├── cubic_interpolant.jl   # CubicInterpolant with order keyword
├── linear_interp.jl       # LinearInterpolant with order keyword
└── derivative_view.jl     # DerivativeView struct and factories
```

---

## References

- Test Suite: [test/test_derivatives.jl](../../test/test_derivatives.jl)
- Implementation Plan: [docs/plans/PLAN_interpolant-derivative-api.md](../plans/PLAN_interpolant-derivative-api.md)
