# Design: Interpolant Derivative API (Hybrid B+C)

**Status**: Implemented
**Version**: 1.0
**Last Updated**: 2025-12-27

---

## 1. Overview

This document describes the derivative API design for `CubicInterpolant` and `LinearInterpolant` in FastInterpolations.jl. The API follows a "Hybrid B+C" approach that provides both direct evaluation and broadcast support.

### Design Goals

1. **Unified API**: Single pattern `itp(xi; deriv=N)` for all evaluations
2. **Zero-Allocation**: Scalar evaluations must be allocation-free
3. **Broadcast Support**: Enable `derivative(itp).(xs)` for vector operations
4. **HOF Composition**: Support `find_zero(derivative(itp), a, b)` patterns
5. **Julia Idioms**: Follow standard Julia conventions

---

## 2. API Summary

### 2.1 Core API (Option B)

The canonical API uses an `deriv` keyword argument on the interpolant functor:

```julia
# Construction (unchanged)
itp = cubic_interp(x, y; bc=ZeroCurvBC(), extrap=:none)

# Scalar Evaluation
itp(xi)              # value (deriv=0, default)
itp(xi; deriv=1)     # first derivative
itp(xi; deriv=2)     # second derivative

# Vector Evaluation
itp(xs)              # values
itp(xs; deriv=1)     # first derivatives
itp(xs; deriv=2)     # second derivatives

# In-place Vector Evaluation (zero-allocation)
itp(output, xs)              # values
itp(output, xs; deriv=1)     # first derivatives
itp(output, xs; deriv=2)     # second derivatives
```

### 2.2 Broadcast Layer (Option C)

For broadcast and higher-order function support, use `DerivativeView` wrappers:

```julia
# Create callable wrapper
d1 = derivative(itp)   # returns DerivativeView{1, ...}
d2 = derivative2(itp)  # returns DerivativeView{2, ...}

# Scalar call (delegates to parent)
d1(xi)                 # ≡ itp(xi; deriv=1)
d2(xi)                 # ≡ itp(xi; deriv=2)

# Broadcast (main use case)
d1.(xs)                # broadcast over xs
@. coef * d1(xs)       # fused broadcast

# Higher-order function composition
find_zero(derivative(itp), a, b)
optimize(f ∘ derivative(itp), ...)
```

### 2.3 Complete API Table

| Operation | Scalar | Vector | In-place | Broadcast |
|-----------|--------|--------|----------|-----------|
| Value | `itp(xi)` | `itp(xs)` | `itp(out, xs)` | `itp.(xs)` |
| 1st deriv | `itp(xi; deriv=1)` | `itp(xs; deriv=1)` | `itp(out, xs; deriv=1)` | `derivative(itp).(xs)` |
| 2nd deriv | `itp(xi; deriv=2)` | `itp(xs; deriv=2)` | `itp(out, xs; deriv=2)` | `derivative2(itp).(xs)` |

---

## 3. Design Constraints

### 3.1 Zero-Allocation Requirement

All scalar evaluations must maintain zero-allocation (or ≤240 bytes on Julia 1.10-1.11 due to Val dispatch overhead):

```julia
@allocated itp(0.5; deriv=1) == 0  # Julia 1.12+
```

### 3.2 Julia Syntax Constraint

Julia does not support keyword arguments in broadcast syntax:

```julia
itp.(xs; deriv=1)  # ❌ INVALID Julia syntax
```

This constraint is the primary motivation for the Hybrid B+C approach. The `DerivativeView` wrapper solves this by enabling:

```julia
derivative(itp).(xs)  # ✅ Valid Julia syntax
```

### 3.3 Type Stability

The `deriv` parameter is a runtime `Int`, requiring compile-time dispatch for type stability. This is achieved via the `@_dispatch_deriv` macro:

```julia
@_dispatch_deriv deriv => op begin
    _eval_with_bc(cache, y, h, z, xi, extrap, op)
end
```

The macro expands to an if-else chain that binds singleton types (`EvalValue()`, `EvalDeriv1()`, `EvalDeriv2()`) at compile time.

---

## 4. Architecture

### 4.1 Design Philosophy

```
┌─────────────────────────────────────────────────────────────┐
│  Core Layer: Option B                                       │
│  ─────────────────────                                      │
│  itp(xi; deriv=N)  →  @_dispatch_deriv → _eval_with_bc()   │
│  - Proven zero-allocation pattern                           │
│  - API consistency with public functions                    │
│  - Low implementation complexity                            │
├─────────────────────────────────────────────────────────────┤
│  UX Layer: Option C                                         │
│  ──────────────────                                         │
│  derivative(itp)  →  DerivativeView  →  parent(x; deriv=N) │
│  - Broadcast support: derivative(itp).(xs) ✓               │
│  - HOF support: find_zero(derivative(itp), ...)            │
│  - Zero-allocation wrapper creation                         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Type Hierarchy

```
AbstractEvalOp (abstract)
├── EvalValue   # deriv=0: function value
├── EvalDeriv1  # deriv=1: first derivative
└── EvalDeriv2  # deriv=2: second derivative

DerivativeView{Order, ITP}
├── Order = 1: first derivative wrapper
└── Order = 2: second derivative wrapper
```

### 4.3 File Organization

```
src/
├── cubic_interpolant.jl   # CubicInterpolant functor with deriv keyword
├── linear_interp.jl       # LinearInterpolant functor with deriv keyword
└── derivative_view.jl     # DerivativeView struct and factory functions
```

---

## 5. Implementation Details

### 5.1 CubicInterpolant Methods

```julia
# Primary scalar method (zero-allocation hot path)
@inline function (itp::CubicInterpolant{T})(xi::T; deriv::Int=0) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.cache.x, xi, itp.extrap)
    @_dispatch_deriv deriv => op begin
        _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi, itp.extrap, op)
    end
end

# Real wrapper (type conversion)
@inline function (itp::CubicInterpolant{T})(xi::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); deriv=deriv)
end

# Vector call
function (itp::CubicInterpolant{T})(xi::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(xi))
    @_dispatch_deriv deriv => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap, op)
    end
    return output
end

# In-place vector call (zero-allocation)
function (itp::CubicInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat}
    @assert length(output) == length(xi)
    @_dispatch_deriv deriv => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap, op)
    end
    return output
end
```

### 5.2 LinearInterpolant Methods

Same pattern as CubicInterpolant, with linear-specific evaluation:

```julia
# Primary scalar method
@inline function (itp::LinearInterpolant{T})(xi::T; deriv::Int=0) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.x, xi, itp.mode)
    @_dispatch_deriv deriv => op begin
        linear_interp(itp.x, itp.y, xi, itp.mode, op)
    end
end
```

**Note on `deriv=2` for LinearInterpolant**: Returns `0.0` because the mathematical second derivative of a piecewise linear function includes Dirac delta impulses at knots, which cannot be represented as finite floating-point values.

### 5.3 DerivativeView Implementation

```julia
# Internal struct (not exported)
struct DerivativeView{Order, ITP}
    parent::ITP
end

# Factory functions (exported)
@inline derivative(itp::CubicInterpolant) = DerivativeView{1, typeof(itp)}(itp)
@inline derivative2(itp::CubicInterpolant) = DerivativeView{2, typeof(itp)}(itp)
@inline derivative(itp::LinearInterpolant) = DerivativeView{1, typeof(itp)}(itp)
@inline derivative2(itp::LinearInterpolant) = DerivativeView{2, typeof(itp)}(itp)

# Callable methods (scalar only - use broadcast for vectors)
@inline (d::DerivativeView{1, ITP})(xi::Real) where {ITP} = d.parent(xi; deriv=1)
@inline (d::DerivativeView{2, ITP})(xi::Real) where {ITP} = d.parent(xi; deriv=2)
```

**Design Note**: Vector methods are intentionally NOT defined on `DerivativeView`. Users should use broadcast (`d.(xs)`), which follows Julia idiom and enables fused broadcast operations.

---

## 6. Zero-Allocation Analysis

### 6.1 Core Layer (Option B)

The `@_dispatch_deriv` macro ensures zero-allocation by:

1. `EvalValue()`, `EvalDeriv1()`, `EvalDeriv2()` are singleton structs (0 bytes)
2. `let` bindings in the macro don't allocate
3. `_eval_with_bc` is fully inlined
4. When `deriv` is constant, compiler eliminates dead branches

### 6.2 UX Layer (Option C)

```julia
d1 = derivative(itp)  # Creates DerivativeView{1, typeof(itp)}(itp)
d1(xi)                # Calls itp(xi; deriv=1)
```

Zero-allocation because:
- `DerivativeView` is immutable struct with single reference field
- Construction is stack-allocated or inlined (no heap allocation)
- Call directly delegates to parent's zero-allocation path

### 6.3 Broadcast Behavior

```julia
d1.(xs)  # Broadcast over vector
```

Allocates output vector only (expected behavior):
- Wrapper `d1` created once (stack)
- Each element: `d1(xs[i])` → zero-allocation scalar call
- Output: `Vector{T}(undef, length(xs))` allocated (unavoidable)

---

## 7. Export Policy

```julia
# Exported (public API)
export derivative, derivative2

# NOT exported (internal implementation)
# DerivativeView - users interact via derivative(itp), not DerivativeView{1,...}(itp)
```

**Rationale**:
- Users interact via `derivative(itp)`, not `DerivativeView{1, ...}(itp)`
- Keeps public API minimal and clean
- Internal type can evolve without breaking user code

---

## 8. Usage Recommendations

| Scenario | Recommended API | Reason |
|----------|-----------------|--------|
| Single point derivative | `itp(xi; deriv=1)` | Simplest, zero-allocation |
| Many points, performance critical | `itp(output, xs; deriv=1)` | In-place, zero-allocation |
| Fused broadcast | `d1 = derivative(itp); @. coef * d1(xs)` | Bind wrapper first |
| Higher-order functions | `find_zero(derivative(itp), ...)` | Callable wrapper |
| Quick vector evaluation | `derivative(itp).(xs)` | Convenience (allocates output) |

---

## 9. Backward Compatibility

### 9.1 Preserved Behavior

| Existing Code | Status | Notes |
|---------------|--------|-------|
| `itp(xi)` | ✅ Works | Default `deriv=0` |
| `itp.(xs)` | ✅ Works | Value broadcast unchanged |
| `derivative(itp)` | ✅ Works | Returns `DerivativeView` |
| `derivative2(itp)` | ✅ Works | Returns `DerivativeView` |

### 9.2 Removed APIs

| Removed API | Replacement | Reason |
|-------------|-------------|--------|
| `derivative(itp, xi)` | `itp(xi; deriv=1)` | Redundant; single canonical API |
| `derivative2(itp, xi)` | `itp(xi; deriv=2)` | Redundant; single canonical API |
| `derivative(itp, xs)` (vector) | `itp(xs; deriv=1)` | Redundant; single canonical API |
| `derivative2(itp, xs)` (vector) | `itp(xs; deriv=2)` | Redundant; single canonical API |

---

## 10. Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| Hybrid B+C approach | B provides proven zero-allocation core; C fills broadcast/HOF gap |
| `itp.(xs; deriv=1)` invalid | Julia language constraint, not design choice |
| DerivativeView delegates to parent | Single source of truth, easier maintenance |
| `itp.d1` property deferred | Type stability risk with `getproperty`; can add later |
| No vector method in DerivativeView | Follow Julia idiom: use `d.(xs)` broadcast |
| `DerivativeView` not exported | Internal type; users use `derivative(itp)` |
| Linear `deriv=2` returns 0.0 | Practical choice; Dirac delta cannot be finite |
| 2-arg derivative forms removed | Single canonical API (`itp(xi; deriv=N)`) is cleaner |
| In-place `deriv` keyword added | Complete API symmetry with vector calls |

---

## 11. Alternatives Considered

### 11.1 Option A: Construction-time Order (Rejected)

```julia
itp_slope = cubic_interp(x, y; deriv=1)  # Separate interpolant
```

**Why rejected**:
- Storage duplication (same x, y, z stored multiple times)
- Need separate objects for each order
- Inconsistent with established API pattern

### 11.2 Option B Only (Insufficient)

```julia
itp(xi; deriv=1)  # Works for scalar
itp.(xs; deriv=1) # ❌ Invalid syntax
```

**Why insufficient**:
- No broadcast support for derivatives
- No higher-order function composition
- Closure workaround (`x -> itp(x; deriv=1)`) is verbose and non-idiomatic

### 11.3 Option C Only (Unnecessary Complexity)

```julia
derivative(itp)(xi)  # Always through wrapper
```

**Why not standalone**:
- Indirect for simple scalar calls
- Loses API consistency with public functions
- Option B's `deriv` keyword is simpler for scalar evaluation

---

## 12. Testing Strategy

### 12.1 Test Categories

| Category | Description | Location |
|----------|-------------|----------|
| Core deriv keyword | `itp(xi; deriv=N)` functionality | `test/test_derivatives.jl` |
| Type stability | `@inferred` checks | `test/test_derivatives.jl` |
| Allocation tests | Zero-allocation verification | `test/test_derivatives.jl` |
| DerivativeView | Wrapper callable, broadcast | `test/test_derivatives.jl` |
| Boundary conditions | All BC types with deriv keyword | `test/test_derivatives.jl` |
| Edge cases | Small grids, constant functions | `test/test_derivatives.jl` |

### 12.2 Allocation Threshold

```julia
const DERIV_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
```

Julia 1.10-1.11 may have up to 240 bytes due to Val dispatch overhead; Julia 1.12+ should be exactly 0 bytes.

---

## 13. References

- Implementation Plan: [docs/plans/PLAN_interpolant-derivative-api.md](../plans/PLAN_interpolant-derivative-api.md)
- Test Suite: [test/test_derivatives.jl](../../test/test_derivatives.jl)
- Source Files:
  - [src/cubic_interpolant.jl](../../src/cubic_interpolant.jl)
  - [src/linear_interp.jl](../../src/linear_interp.jl)
  - [src/derivative_view.jl](../../src/derivative_view.jl)
