# Design Doc: Analytical Derivatives for Cubic & Linear Splines

## 1. Overview

Adds 1st derivative (slope) and 2nd derivative (curvature) computation to `FastInterpolations.jl`.

> **Naming Note**: "Slope" or "Derivative" is preferred over "Gradient" for 1D interpolation. "Gradient" implies a multi-dimensional partial derivative vector.

**Goals:**
- **Zero-allocation** and **type stability**
- **Consistent API** for `linear_interp` and `cubic_interp`
- Support `order` parameter across all API variants (allocating, in-place, cache-based, interpolant)

---

## 2. Mathematical Basis

### 2.1. Cubic Spline

Cubic spline $S(x)$ on interval $[x_i, x_{i+1}]$ using moments $z$:

$$
S(x) = \frac{z_i (x_{i+1}-x)^3 + z_{i+1} (x-x_i)^3}{6h_i} + \left(\frac{y_{i+1}}{h_i} - \frac{z_{i+1}h_i}{6}\right)(x-x_i) + \left(\frac{y_i}{h_i} - \frac{z_i h_i}{6}\right)(x_{i+1}-x)
$$

where $h_i = x_{i+1} - x_i$, $dt_1 = x - x_i$, $dt_2 = x_{i+1} - x$.

**1st Derivative (Slope):**
$$
S'(x) = \frac{-z_i \cdot dt_2^2 + z_{i+1} \cdot dt_1^2}{2h_i} + \frac{y_{i+1} - y_i}{h_i} + \frac{h_i(z_i - z_{i+1})}{6}
$$

**2nd Derivative (Curvature):**
$$
S''(x) = \frac{z_i \cdot dt_2 + z_{i+1} \cdot dt_1}{h_i}
$$

> Note: 2nd derivative is linear interpolation of $z$ values.

### 2.2. Linear Interpolation

Linear interpolation $L(x)$ on interval $[x_i, x_{i+1}]$:
$$
L(x) = y_i + \frac{y_{i+1}-y_i}{h_i}(x - x_i)
$$

- **1st Derivative (Slope)**: Constant $\frac{y_{i+1}-y_i}{h_i}$ within interval
- **2nd Derivative (Curvature)**: 0 within interval

> **Mathematical Note**: Strictly speaking, 2nd derivative of piecewise-linear is 0 inside intervals and **Dirac delta** at knots. Implementation returns 0 everywhere (regular function approximation).

#### 2.2.1. Knot Behavior (Linear)

1st derivative is discontinuous at knots. Implementation follows **right-continuous** convention:

| Query Point $x$ | Interval Used | Derivative |
|-----------------|---------------|------------|
| $x < x_1$ | Depends on `extrap` | See Section 4 |
| $x_1 \leq x < x_2$ | $[x_1, x_2]$ | $(y_2 - y_1) / h_1$ |
| $x_i \leq x < x_{i+1}$ | $[x_i, x_{i+1}]$ | $(y_{i+1} - y_i) / h_i$ |
| $x = x_n$ (last point) | $[x_{n-1}, x_n]$ | $(y_n - y_{n-1}) / h_{n-1}$ |
| $x > x_n$ | Depends on `extrap` | See Section 4 |

> This matches existing `_find_interval_with_bounds` behavior. Derivative at knot returns **right interval** slope, except at last point.

---

## 3. Architecture

### 3.1. Dispatch Hierarchy (Core Design)

**`order` and `extrap` are runtime values that must be converted to compile-time constants for type stability. Each is dispatched exactly once per call chain.**

**Pattern A: Nested dispatch at public API** (most common)

Both dispatches happen together at the entry point:

```julia
# src/linear_interp.jl, src/cubic_eval.jl
@_dispatch_order order => op begin
    @_dispatch_extrap extrap => ev begin
        # Both op and ev are now concrete types
        _eval_with_bc(cache, y, h, z, xi, ev, op)
    end
end
```

**Pattern B: Separated dispatch** (for complex routing)

`@_dispatch_order` at entry, `@_dispatch_extrap` deeper in call chain:

```julia
# src/cubic_interp.jl - entry point
@_dispatch_order order => op begin
    if _is_periodic_bc(bc)
        return _cubic_interp_periodic!(output, x, y, x_query, autocache, op)
    end
    # BCPair path has extrap dispatch inside
    return _cubic_interp_bcpair!(output, x, y, x_query, bc, extrap, autocache, op)
end

# Internal function handles extrap dispatch
function _cubic_interp_bcpair!(... op::O) where {O<:AbstractEvalOp}
    @_dispatch_extrap extrap => ev begin
        _cubic_vector_loop!(output, cache, y, z, x_query, ev, op)
    end
end
```

**Dispatch flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Public API                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  @_dispatch_order order => op begin                         ││
│  │      # order::Int → EvalValue()/EvalDeriv1()/EvalDeriv2()   ││
│  │                                                             ││
│  │      @_dispatch_extrap extrap => ev begin  ← often nested   ││
│  │          # extrap::Symbol → Val{:none}/Val{:constant}/...   ││
│  │          _eval_impl(..., ev, op)                            ││
│  │      end                                                    ││
│  │  end                                                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Kernel Functions (_cubic_kernel, _linear_kernel)               │
│  - Receives op::O where {O<:AbstractEvalOp} (pass-through)      │
│  - Both op and ev are concrete types → fully specialized        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
1. Each dispatch macro converts runtime value to compile-time constant **once per call chain**
2. Nesting is fine - Julia compiles all combinations (3 orders × 4 extrap modes = 12 specializations)
3. Internal functions receive concrete types via type parameter `op::O where {O<:AbstractEvalOp}`

### 3.2. Module Structure

```julia
# src/FastInterpolations.jl
module FastInterpolations

import LinearAlgebra
using LinearAlgebra: Tridiagonal, lu, ldiv!

# Operation types (must be first - used by all interp files)
include("ops.jl")

# Boundary condition types
include("bc_types.jl")

# Kernel functions (pure math, no dependencies)
include("linear_kernels.jl")
include("cubic_kernels.jl")

# ... existing includes
end
```

**Benefits of separate kernel files:**
- Pure math functions → independent testing/verification
- Future SIMD/GPU optimization as separate modules
- `linear_interp.jl`, `cubic_eval.jl` focus on routing logic

### 3.3. Shared Operation Types (`src/ops.jl`)

Operation traits shared by Linear and Cubic:

```julia
"""
    AbstractEvalOp

Abstract type for evaluation operations (value, derivatives).
Used for compile-time dispatch to select appropriate kernel.
"""
abstract type AbstractEvalOp end

"""Evaluate function value f(x)"""
struct EvalValue <: AbstractEvalOp end

"""Evaluate first derivative f'(x)"""
struct EvalDeriv1 <: AbstractEvalOp end

"""Evaluate second derivative f''(x)"""
struct EvalDeriv2 <: AbstractEvalOp end

"""
    ExtrapVal

Union type for extrapolation mode values.
Concrete Union enables Julia's union-splitting optimization.
"""
const ExtrapVal = Union{Val{:none}, Val{:constant}, Val{:extension}, Val{:wrap}}
```

### 3.4. Dispatch Macro (`src/utils.jl`)

Converts `order::Int` to compile-time constant:

```julia
macro _dispatch_order(pair, body)
    # Parse pair: order => op becomes Expr(:call, :(=>), :order, :op)
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_order expects `order => op`, got: $pair")
    order_expr = pair.args[2]
    op_sym = pair.args[3]
    ord_var = gensym(:order)
    quote
        local $(ord_var) = $(esc(order_expr))
        if $(ord_var) == 0
            let $(esc(op_sym)) = EvalValue()
                $(esc(body))
            end
        elseif $(ord_var) == 1
            let $(esc(op_sym)) = EvalDeriv1()
                $(esc(body))
            end
        elseif $(ord_var) == 2
            let $(esc(op_sym)) = EvalDeriv2()
                $(esc(body))
            end
        else
            throw(ArgumentError("order must be 0, 1, or 2; got $($(ord_var))"))
        end
    end
end
```

**Usage:**
```julia
@_dispatch_order order => op begin
    _cubic_interp_impl(..., op)
end
```

### 3.5. Type Parameter Pattern

**Important**: Using `op::AbstractEvalOp` directly causes **type instability**. Must use type parameter:

```julia
# ❌ BAD: Abstract type directly - type unstable!
@inline function _eval_cubic_at_point(..., op::AbstractEvalOp)
    ...
end

# ✅ GOOD: Type parameter - type stable
@inline function _eval_cubic_at_point(..., op::O) where {O<:AbstractEvalOp}
    ...
end
```

### 3.6. Backward Compatibility with Wrapper Pattern

Add `op` parameter while preserving existing call sites:

```julia
# Wrapper for backward compatibility (op defaults to EvalValue)
@inline _eval_with_bc(cache, y, h, z, xi, ev::Val) =
    _eval_with_bc(cache, y, h, z, xi, ev, EvalValue())

# Full implementation with op
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,BC},
    y, h, z, xi,
    ev::Val,
    op::O
) where {T, X, F, BC, O<:AbstractEvalOp}
    # ... implementation
end
```

This allows:
- Existing code: `_eval_with_bc(cache, y, h, z, xi, ev)` works unchanged
- New code: `_eval_with_bc(cache, y, h, z, xi, ev, op)` available

### 3.7. Kernel Layer

#### 3.7.1. Linear Kernels (`src/linear_kernels.jl`)

**Unified signature**: All ops receive same arguments `(y0, y1, h, dt1)` to eliminate call-site branching.

```julia
# h = x1 - x0 (interval width)
# dt1 = xi - x0 (offset from left)

# Value: linear interpolation
@inline function _linear_kernel(::EvalValue, y0::T, y1::T, h::T, dt1::T) where {T}
    α = dt1 / h
    return y0 * (one(T) - α) + y1 * α
end

# First derivative: constant slope in interval
@inline function _linear_kernel(::EvalDeriv1, y0::T, y1::T, h::T, ::T) where {T}
    return (y1 - y0) / h
end

# Second derivative: always zero
@inline function _linear_kernel(::EvalDeriv2, ::T, ::T, ::T, ::T) where {T}
    return zero(T)
end
```

#### 3.7.2. Cubic Kernels (`src/cubic_kernels.jl`)

```julia
# Value (existing _eval_cubic_at_point logic)
@inline function _cubic_kernel(::EvalValue, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
    I = (z_i * dt2^3 + z_ip1 * dt1^3) / (6 * h_i)
    C = (y_ip1 / h_i - z_ip1 * h_i / 6) * dt1
    D = (y_i / h_i - z_i * h_i / 6) * dt2
    return I + C + D
end

# First derivative
@inline function _cubic_kernel(::EvalDeriv1, z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
    term1 = (-z_i * dt2^2 + z_ip1 * dt1^2) / (2 * h_i)
    term2 = (y_ip1 - y_i) / h_i
    term3 = (z_i - z_ip1) * h_i / 6
    return term1 + term2 + term3
end

# Second derivative (linear interpolation of z)
@inline function _cubic_kernel(::EvalDeriv2, z_i, z_ip1, _, _, h_i, dt1, dt2)
    return (z_i * dt2 + z_ip1 * dt1) / h_i
end
```

### 3.8. Periodic BC Fast-Path Handling

`_cubic_vector_loop!` has a Periodic BC specialization that bypasses `_eval_with_bc`. This path must also propagate `op`:

```julia
# Wrapper for backward compatibility
@inline function _cubic_vector_loop!(output, cache, y, z, x_query, ev::Val)
    _cubic_vector_loop!(output, cache, y, z, x_query, ev, EvalValue())
end

# Periodic BC fast-path with op
@inline function _cubic_vector_loop!(
    output::AbstractVector{T},
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y::AbstractVector{T},
    z::AbstractVector{T},
    x_query::AbstractVector{T},
    ::Val,  # extrap ignored for periodic
    op::O
) where {T<:AbstractFloat, X, F, O<:AbstractEvalOp}
    x_min = first(cache.x)
    x_max = x_min + cache.bc_data.period
    qmin, qmax = minimum(x_query), maximum(x_query)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point(cache.x, y, cache.h, z, xq, op)
        end
    else
        # Slow path: per-element wrap
        period = cache.bc_data.period
        @inbounds for (k, xq) in enumerate(x_query)
            output[k] = _eval_cubic_at_point_periodic(cache.x, y, cache.h, z, xq, period, op)
        end
    end
end
```

---

## 4. Extrapolation Semantics

### 4.1. Behavior Table

| Mode | In Domain | Outside Domain (Value) | Outside Domain (D1) | Outside Domain (D2) |
|------|-----------|------------------------|---------------------|---------------------|
| `:none` | Normal eval | `DomainError` | `DomainError` | `DomainError` |
| `:constant` | Normal eval | $y_1$ or $y_n$ | **0** | **0** |
| `:extension` | Normal eval | Boundary polynomial | Boundary poly deriv | Boundary poly 2nd deriv |
| `:wrap` | Normal eval | $f(x_{wrapped})$ | $f'(x_{wrapped})$ | $f''(x_{wrapped})$ |

### 4.2. Derivative at Boundary Points

For `:constant` extrapolation:
- **Outside domain** (x < x[1] or x > x[n]): Derivative = 0 (constant function)
- **At boundary point** (x = x[1] or x = x[n]): **Right-continuous** rule applies

Same principle as §2.2.1 Linear knot behavior:

| Query Point | Value | Derivative |
|-------------|-------|------------|
| x < x[1] | y[1] | 0 |
| x = x[1] | First interval formula | First interval derivative (right-continuous) |
| x[1] < x < x[n] | Normal eval | Normal eval |
| x = x[n] | Last interval formula | Last interval derivative |
| x > x[n] | y[n] | 0 |

**Implementation:**

```julia
@inline function _eval_cubic_with_extrap(
    x, y, h, z, xi,
    ::Val{:constant},
    op::O
) where {O<:AbstractEvalOp}
    if xi < first(x)
        return _constant_extrap_result(op, y[1])
    elseif xi > last(x)
        return _constant_extrap_result(op, y[end])
    else
        # Inside domain (including boundary points) - normal evaluation
        return _eval_cubic_at_point(x, y, h, z, xi, op)
    end
end

# Constant extrapolation results (outside domain)
@inline _constant_extrap_result(::EvalValue, y_boundary::T) where {T} = y_boundary
@inline _constant_extrap_result(::EvalDeriv1, y_boundary::T) where {T} = zero(T)
@inline _constant_extrap_result(::EvalDeriv2, y_boundary::T) where {T} = zero(T)
```

**Key points:**
- `_find_interval_with_bounds` already follows right-continuous rule
- Boundary points x[1], x[n] are treated as **inside domain** (normal evaluation)
- Only outside domain uses constant extrapolation (derivative = 0)

### 4.3. Periodic BC and Extrapolation

`PeriodicBC` **ignores** `extrap` parameter for derivatives (same as value path):

```julia
@inline function _eval_with_bc(
    cache::CubicSplineCache{T,X,F,PeriodicData{T}},
    y, h, z, xi,
    ::Val,  # extrap ignored
    op::O
) where {T, X, F, O<:AbstractEvalOp}
    _eval_cubic_at_point_periodic(cache.x, y, h, z, xi, cache.bc_data.period, op)
end
```

---

## 5. Public API Design

### 5.1. API Coverage Matrix

`order` parameter supported across **all API variants**:

| API | Scalar | Vector | `order` Support |
|-----|--------|--------|-----------------|
| `linear_interp(x, y, xi)` | Yes | Yes | ✅ |
| `linear_interp!(out, x, y, xi)` | Yes | Yes | ✅ |
| `LinearInterpolant(x, y)(xi)` | Yes | Yes | via `derivative()` |
| `cubic_interp(x, y, xi)` | Yes | Yes | ✅ |
| `cubic_interp!(out, x, y, xi)` | Yes | Yes | ✅ |
| `cubic_interp(cache, y, xi)` | Yes | Yes | ✅ |
| `cubic_interp!(out, cache, y, xi)` | Yes | Yes | ✅ |
| `CubicInterpolant(x, y)(xi)` | Yes | Yes | via `derivative()` |

### 5.2. Example Call Flow

**Cubic spline with order=1:**

```julia
# User calls:
cubic_interp(x, y, xi; order=1, extrap=:constant)

# Expands to:
@_dispatch_order 1 => op begin       # op = EvalDeriv1() (concrete!)
    _cubic_interp_impl(x, y, xi, op; extrap=:constant, ...)
end

# Inside _cubic_interp_impl:
function _cubic_interp_impl(..., op::O; extrap, ...) where {O<:AbstractEvalOp}
    @_dispatch_extrap extrap => ev begin    # ev = Val{:constant}()
        _eval_with_bc(..., ev, op)          # op is already concrete type
    end
end

# Final kernel call - fully specialized:
_cubic_kernel(EvalDeriv1(), z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
```

### 5.3. Interpolant Object APIs

Interpolant objects use `derivative()` function for derivative computation:

> **Ecosystem Note**: `derivative` matches Julia ecosystem conventions (Interpolations.jl, DataInterpolations.jl). `gradient` is for multi-dimensional/vector functions; `derivative` is standard for 1D scalar functions.

```julia
# CubicInterpolant
"""
    derivative(itp::CubicInterpolant, x)

Compute first derivative at point x using pre-computed z coefficients.
Zero-allocation for scalar x.
"""
function derivative(itp::CubicInterpolant{T}, xi::T) where {T}
    _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi, itp.extrap, EvalDeriv1())
end

"""
    derivative2(itp::CubicInterpolant, x)

Compute second derivative at point x.
"""
function derivative2(itp::CubicInterpolant{T}, xi::T) where {T}
    _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi, itp.extrap, EvalDeriv2())
end

# LinearInterpolant
function derivative(itp::LinearInterpolant{T}, xi::T) where {T}
    _linear_with_extrap(itp.x, itp.y, xi, itp.mode, EvalDeriv1())
end

function derivative2(itp::LinearInterpolant{T}, ::T) where {T}
    zero(T)  # Always zero for linear (Dirac delta at knots ignored)
end
```

---

## 6. Testing Strategy

### 6.1. Mathematical Accuracy

> **Note**: Cubic spline reproduces polynomials exactly only when BCs match the actual function.
> Natural BC (z=0) is exact only for functions with f''(boundary)=0. General polynomials require explicit D2 BC.

```julia
@testset "Polynomial exactness - quadratic" begin
    # f(x) = x², f'(x) = 2x, f''(x) = 2
    x = collect(0.0:0.1:1.0)
    y = x.^2

    # Explicit D2 BC: f''(0) = 2, f''(1) = 2
    bc = BCPair(D2(2.0), D2(2.0))
    xi = 0.5

    @test cubic_interp(x, y, xi; bc=bc, order=0) ≈ xi^2 atol=1e-10
    @test cubic_interp(x, y, xi; bc=bc, order=1) ≈ 2*xi atol=1e-10
    @test cubic_interp(x, y, xi; bc=bc, order=2) ≈ 2.0 atol=1e-10
end

@testset "Polynomial exactness - cubic" begin
    # f(x) = x³, f'(x) = 3x², f''(x) = 6x
    x = collect(0.0:0.1:1.0)
    y = x.^3

    # Explicit D2 BC: f''(0) = 0, f''(1) = 6
    bc = BCPair(D2(0.0), D2(6.0))
    xi = 0.5

    @test cubic_interp(x, y, xi; bc=bc, order=0) ≈ xi^3 atol=1e-10
    @test cubic_interp(x, y, xi; bc=bc, order=1) ≈ 3*xi^2 atol=1e-10
    @test cubic_interp(x, y, xi; bc=bc, order=2) ≈ 6*xi atol=1e-10
end
```

### 6.2. Zero-Allocation Verification

> **Note**: Uses `ALLOC_THRESHOLD` defined in `test/runtests.jl`.
> Julia 1.12+ allows 0, earlier versions allow 240 bytes (Val dispatch overhead).

```julia
@testset "Zero allocation - derivatives" begin
    cache = CubicSplineCache(collect(0.0:0.1:1.0))
    y = rand(11)

    # Warm up (JIT compilation)
    cubic_interp(cache, y, 0.5; order=1)
    cubic_interp(cache, y, 0.5; order=2)

    # Test using shared threshold
    @test @allocated(cubic_interp(cache, y, 0.5; order=1)) <= ALLOC_THRESHOLD
    @test @allocated(cubic_interp(cache, y, 0.5; order=2)) <= ALLOC_THRESHOLD
end
```

### 6.3. Constant Extrapolation - Boundary Derivatives

```julia
@testset "Constant extrapolation - boundary derivatives" begin
    x = collect(0.0:0.5:2.0)
    y = x.^2  # f(x) = x², f'(x) = 2x

    # Outside domain: derivative = 0 (constant extrapolation)
    @test cubic_interp(x, y, -1.0; extrap=:constant, order=1) ≈ 0.0
    @test cubic_interp(x, y, 3.0; extrap=:constant, order=1) ≈ 0.0

    # At boundary points: normal evaluation (right-continuous)
    d_at_0 = cubic_interp(x, y, 0.0; order=1)
    d_at_2 = cubic_interp(x, y, 2.0; order=1)
    @test d_at_0 ≈ cubic_interp(x, y, 0.0; extrap=:constant, order=1)
    @test d_at_2 ≈ cubic_interp(x, y, 2.0; extrap=:constant, order=1)

    # 2nd derivative also 0 outside domain
    @test cubic_interp(x, y, -1.0; extrap=:constant, order=2) ≈ 0.0
    @test cubic_interp(x, y, 3.0; extrap=:constant, order=2) ≈ 0.0
end
```

### 6.4. Periodic BC Derivatives

```julia
@testset "Periodic BC derivatives - continuity" begin
    x = range(0, 2π, 101)
    y = sin.(collect(x))
    y[end] = y[1]  # Ensure periodic

    # Derivative should be continuous across boundary
    ε = 1e-6
    d_left = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), order=1)
    d_right = cubic_interp(x, y, ε; bc=PeriodicBC(), order=1)
    @test d_left ≈ d_right atol=1e-4

    # 2nd derivative also continuous
    d2_left = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), order=2)
    d2_right = cubic_interp(x, y, ε; bc=PeriodicBC(), order=2)
    @test d2_left ≈ d2_right atol=1e-3
end
```

---

## 7. FAQ

**Q: Is there overhead from nested `order` and `extrap` dispatch?**

A: Minimal. Julia compiles specialized versions for all combinations (3 orders × 4 extrap modes = 12). The dispatch macros convert runtime values to compile-time constants at the entry point, so all downstream code is fully specialized. Nesting them together or separating them doesn't affect runtime performance.

**Q: Why `op::O where {O<:AbstractEvalOp}` instead of `op::AbstractEvalOp`?**

A: Julia type inference. Abstract type annotation prevents compiler from inferring concrete type → type unstable. Type parameter preserves concrete type through call chain.

**Q: What's the derivative value outside domain with `:constant` extrapolation?**

A: **0**. `:constant` extrapolation maintains boundary value as constant outside domain, so derivative is naturally 0. Boundary points x[1], x[n] are treated as **inside domain** with normal derivative (right-continuous rule).

**Q: Do I need to update all internal function calls?**

A: No. **Wrapper pattern** keeps existing calls (`_eval_with_bc(cache, y, h, z, xi, ev)`) working unchanged while enabling new calls (`_eval_with_bc(..., op)`).

**Q: Why does Periodic BC ignore `extrap`?**

A: By definition of Periodic BC, all coordinates wrap into domain. There is no "outside domain" concept, so `extrap` setting is meaningless.
