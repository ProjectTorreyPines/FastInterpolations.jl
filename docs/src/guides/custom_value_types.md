# Custom Value Types (Duck Typing)

FastInterpolations supports interpolation with arbitrary value types — not just `Real` and `Complex`. Any type that supports the required arithmetic operations can be used as the value type `Tv`.

## Type System

Internally, every interpolant separates two type roles:

* **Grid type (`Tg`)**: Always `<: AbstractFloat` (`Float64`, `Float32`). Used for grid coordinates, spacings, and search. Must be ordered and real-valued.
* **Value type (`Tv`)**: Unconstrained. Used for function values (`y`), derivatives, and coefficients. Can be any type supporting the required operations.

## Quick Start

```julia
using FastInterpolations
using StaticArrays

# Interpolate 3D vector fields with SVector
x = range(0.0, 1.0, 10)
y = [SVector(sin(xi), cos(xi), xi^2) for xi in x]

itp = linear_interp(x, y)
itp(0.5)  # → SVector{3,Float64}

# Works with all methods (default BCs)
itp_cubic = cubic_interp(x, y)
itp_cubic(0.5)  # → SVector{3,Float64}
itp_cubic(0.5; deriv=EvalDeriv1())  # → first derivative, also SVector
```

## Core Operations (7 — sufficient for all methods)

With **default boundary conditions** (CubicFit, QuadraticFit), only 7 operations are needed. These work for constant, linear, quadratic, and cubic interpolation — including ND and Series.

```julia
zero(::Type{Tv}) → Tv          # zero element
+(::Tv, ::Tv) → Tv             # value addition
-(::Tv, ::Tv) → Tv             # value subtraction
*(::Tg, ::Tv) → Tv             # left scalar multiplication
*(::Tv, ::Tg) → Tv             # right scalar multiplication
*(::Int, ::Tv) → Tv            # integer scaling (2*s, 6*d in recurrence)
/(::Tv, ::Tg) → Tv             # scalar division
```

These 7 operations form the **bimodule over the grid ring `Tg`** — the minimal algebraic structure required for polynomial interpolation.

### Per-method breakdown (core ops only)

Not all 7 are needed for every method:

| Operation | Constant | Linear | Quadratic | Cubic |
|:----------|:--------:|:------:|:---------:|:-----:|
| `zero(::Type{Tv})` | required | required | required | required |
| `+(::Tv, ::Tv)` | — | required | required | required |
| `-(::Tv, ::Tv)` | — | required | required | required |
| `*(::Tg, ::Tv)` | — | required | required | required |
| `*(::Tv, ::Tg)` | — | ND only | required | required |
| `*(::Int, ::Tv)` | — | — | required | required |
| `/(::Tv, ::Tg)` | — | deriv only | required | required |

## Conditional Operations (beyond the core 7)

These operations are only needed in specific scenarios. If you only use default BCs and don't need PeriodicBC, the core 7 are sufficient.

### `convert(::Type{Tv}, ::Real)` — explicit Deriv BCs with Real literals

When you pass `Deriv1(0.0)` (a Real literal) as a boundary condition, the constructor promotes the BC value to `Tv` via `convert(Tv, Real)`. This is needed because the solver operates in `Tv` space.

```julia
# Needs convert(MyType, Real):
quadratic_interp(x, y; bc=Left(Deriv1(0.0)))
cubic_interp(x, y; bc=Deriv1(0.0))

# Does NOT need convert — pass Tv values directly:
quadratic_interp(x, y; bc=Left(Deriv1(MyType(0.0))))
cubic_interp(x, y; bc=Deriv1(MyType(0.0)))

# Does NOT need convert — default BC:
quadratic_interp(x, y)  # default = QuadraticFit()
cubic_interp(x, y)       # default = CubicFit()
```

**Tip**: Pass `Deriv1(MyType(val))` instead of `Deriv1(val)` to avoid needing `convert`. Julia's built-in `convert(T, x::T) = x` handles the identity case.

### `/(::Tv, ::Int)` — quadratic with Deriv2 BC only

The quadratic solver's `_fill_slopes!` computes `κ / 2` where `κ` is a `Tv` value. This only triggers with explicit `Deriv2` boundary conditions.

```julia
# Needs /(MyType, Int):
quadratic_interp(x, y; bc=Left(Deriv2(MyType(0.0))))

# Does NOT need it:
quadratic_interp(x, y)                       # default = QuadraticFit()
quadratic_interp(x, y; bc=Left(Deriv1(...))) # Deriv1, not Deriv2
```

### `isapprox(::Tv, ::Tv; atol, rtol)` — PeriodicBC with endpoint=:inclusive

`PeriodicBC()` defaults to `endpoint=:inclusive`, which validates that `y[1] ≈ y[end]`. The check uses `==` first (bitwise equality for immutable structs — always works), then falls back to `isapprox` if `==` returns `false`.

```julia
# Needs isapprox (approximate match):
cubic_interp(x, y; bc=PeriodicBC(endpoint=:inclusive))

# Does NOT need isapprox:
cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))  # skips validation
cubic_interp(x, y; bc=PeriodicBC())  # if y[1] === y[end] exactly
```

### Summary table (including conditional)

| Operation | When needed |
|:----------|:-----------|
| Core 7 (above) | Always, with default BCs |
| `convert(::Type{Tv}, ::Real)` | `Deriv1/2(Real_literal)` BC (avoidable by passing `Deriv(Tv_value)`) |
| `/(::Tv, ::Int)` | Quadratic with explicit `Deriv2` BC |
| `isapprox(::Tv, ::Tv; kw...)` | `PeriodicBC(endpoint=:inclusive)` when `y[1] !== y[end]` |

## What is NOT required

- **`muladd(a, b, c)`**: Julia provides a generic fallback `muladd(x,y,z) = x*y+z`. If `*` and `+` are defined, `muladd` works automatically. Defining it explicitly is optional (enables FMA hardware acceleration).
- **`*(::Tv, ::Tv)`**: No kernel performs value-times-value multiplication.
- **`-(::Tv)` (unary negation)**: Negation is always applied to `Tg` scalars, not `Tv` values.
- **`<:Number`**: Your type does not need to be a `Number` subtype. In fact, staying outside `Number` is recommended — Julia's `muladd(::Number, ::Number, ::Number)` tries type promotion, which fails without `promote_rule`. Outside `Number`, the generic `muladd(x,y,z) = x*y+z` fires automatically.

## Source References

Where each operation is used in the codebase:

| Operation | Used in | Context |
|:----------|:--------|:--------|
| `+(::Tv, ::Tv)` | `_linear_kernel`, `_quadratic_kernel`, `_cubic_kernel` | value accumulation via `muladd` fallback |
| `-(::Tv, ::Tv)` | `_linear_kernel`, `_compute_rhs!` | slope differences `yR - yL`, `y[i+1] - y[i]` |
| `*(::Tg, ::Tv)` | `_linear_kernel`, `_cubic_kernel` | `α * (yR-yL)` scalar interpolation weight |
| `*(::Tv, ::Tg)` | `_quadratic_kernel`, `_thomas_solve!` | `value * inv_h`, backward substitution |
| `*(::Int, ::Tv)` | `_compute_slopes_forward!`, `_compute_rhs!` | `2*s[i]` recurrence, `6*(...)` RHS |
| `/(::Tv, ::Tg)` | `_linear_kernel` (deriv), `_compute_rhs!` | `(yR-yL)/h` derivatives and RHS |
| `convert(Tv, Real)` | `_promote_pointbc` | BC value promotion in constructor |
| `/(::Tv, ::Int)` | `_fill_slopes!` (quadratic Deriv2) | `κ / 2` curvature |
| `isapprox` | `_periodic_match` | PeriodicBC endpoint validation |

## Defining a Custom Type

### Minimal (7 ops — works with all methods using default BCs)

```julia
struct MyValue
    val::Float64
end

# Core 7 operations — sufficient for constant, linear, quadratic, cubic
Base.zero(::Type{MyValue}) = MyValue(0.0)
Base.:+(a::MyValue, b::MyValue) = MyValue(a.val + b.val)
Base.:-(a::MyValue, b::MyValue) = MyValue(a.val - b.val)
Base.:*(a::Float64, b::MyValue) = MyValue(a * b.val)
Base.:*(a::MyValue, b::Float64) = MyValue(a.val * b)
Base.:*(a::Integer, b::MyValue) = MyValue(a * b.val)
Base.:/(a::MyValue, b::Float64) = MyValue(a.val / b)

# Use it — all methods work with default BCs
x = range(0.0, 1.0, 10)
y = [MyValue(sin(xi)) for xi in x]

constant_interp(x, y)(0.5)    # → MyValue(...)
linear_interp(x, y)(0.5)      # → MyValue(...)
quadratic_interp(x, y)(0.5)   # → MyValue(...)
cubic_interp(x, y)(0.5)       # → MyValue(...)
```

### Full (with optional extras for explicit BCs)

```julia
# Add convert for explicit Deriv BCs in quadratic:
Base.convert(::Type{MyValue}, x::Real) = MyValue(Float64(x))

# Add isapprox for PeriodicBC(endpoint=:inclusive):
Base.isapprox(a::MyValue, b::MyValue; kwargs...) = isapprox(a.val, b.val; kwargs...)

# Now these also work:
quadratic_interp(x, y; bc=Left(Deriv1(0.0)))(0.5)
```

## Mathematical Interpretation

The required operations form a **bimodule over the grid ring `Tg`**:

- **Left module**: `*(::Tg, ::Tv) → Tv` (scalar acts from the left)
- **Right module**: `*(::Tv, ::Tg) → Tv` (scalar acts from the right)
- **Z-module**: `*(::Int, ::Tv) → Tv` (integer scaling, implied by repeated addition)
- **Division**: `/(::Tv, ::Tg)` is the right inverse of scalar multiplication

This is the standard algebraic structure of a vector space (or module) over a field (or ring), which is the minimal requirement for polynomial interpolation.

## Known Limitations

### `zero(promote_type(Tv, Tg))`

Linear/constant deriv2/3 paths use `zero(promote_type(Tv, Tg))`. If `promote_type` fails for your type, evaluating higher derivatives of linear interpolation may error. Practical impact is minimal (the result is always zero).

### Series output type

`promote_type(Tv, Tq)` may return `Any` for custom types. The `_series_output_type` helper falls back to `Tv` in this case.
