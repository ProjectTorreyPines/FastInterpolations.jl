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

# Works with all methods and BCs
itp_cubic = cubic_interp(x, y)
itp_cubic(0.5)  # → SVector{3,Float64}
itp_cubic(0.5; deriv=EvalDeriv1())  # → first derivative, also SVector
```

## Core Operations (7 — sufficient for all methods and BCs)

Only 7 operations are needed. These work for constant, linear, quadratic, and cubic interpolation — including all boundary conditions, ND, and Series.

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

## Conditional Operation

The core 7 operations are sufficient for **all methods, all BCs, all modes** — except one edge case:

### `isapprox(::Tv, ::Tv; atol, rtol)` — PeriodicBC with endpoint=:inclusive

`PeriodicBC()` defaults to `endpoint=:inclusive`, which validates that `y[1] ≈ y[end]`. The check uses `==` first (bitwise equality for immutable structs — always works), then falls back to `isapprox` if `==` returns `false`.

```julia
# Needs isapprox (approximate match):
cubic_interp(x, y; bc=PeriodicBC(endpoint=:inclusive))

# Does NOT need isapprox:
cubic_interp(x, y; bc=PeriodicBC(endpoint=:exclusive))  # skips validation
cubic_interp(x, y; bc=PeriodicBC())  # if y[1] === y[end] exactly
```

### Tips

- **Explicit Deriv BCs**: Pass `Deriv1(MyType(val))` instead of `Deriv1(val)` — no `convert` needed.
- **Avoid `<: Number`**: Julia's `muladd(::Number, ::Number, ::Number)` tries type promotion, which fails without `promote_rule`. Outside `Number`, the generic `muladd(x,y,z) = x*y+z` fires automatically.
- **`muladd`**, **`*(::Tv, ::Tv)`**, **`-(::Tv)`**, **`/(::Tv, ::Int)`**, **`convert(::Type{Tv}, ::Real)`** are all NOT required.

## Defining a Custom Type

```julia
struct MyValue
    val::Float64
end

# Core 7 operations — sufficient for ALL methods and ALL boundary conditions
Base.zero(::Type{MyValue}) = MyValue(0.0)
Base.:+(a::MyValue, b::MyValue) = MyValue(a.val + b.val)
Base.:-(a::MyValue, b::MyValue) = MyValue(a.val - b.val)
Base.:*(a::Float64, b::MyValue) = MyValue(a * b.val)
Base.:*(a::MyValue, b::Float64) = MyValue(a.val * b)
Base.:*(a::Integer, b::MyValue) = MyValue(a * b.val)
Base.:/(a::MyValue, b::Float64) = MyValue(a.val / b)

# Use it — all methods and BCs work
x = range(0.0, 1.0, 10)
y = [MyValue(sin(xi)) for xi in x]

constant_interp(x, y)(0.5)                                     # → MyValue(...)
linear_interp(x, y)(0.5)                                       # → MyValue(...)
quadratic_interp(x, y)(0.5)                                    # → MyValue(...)
quadratic_interp(x, y; bc=Left(Deriv1(MyValue(0.0))))(0.5)     # → MyValue(...)
cubic_interp(x, y)(0.5)                                        # → MyValue(...)
cubic_interp(x, y; bc=Deriv1(MyValue(0.0)))(0.5)               # → MyValue(...)
cubic_interp(x, y; bc=ZeroCurvBC())(0.5)                       # → MyValue(...)
```

### Optional: `isapprox` for PeriodicBC

```julia
# Only needed for PeriodicBC(endpoint=:inclusive) with approximate endpoint match:
Base.isapprox(a::MyValue, b::MyValue; kwargs...) = isapprox(a.val, b.val; kwargs...)
```

## Mathematical Interpretation

The required operations form a **bimodule over the grid ring `Tg`**:

- **Left module**: `*(::Tg, ::Tv) → Tv` (scalar acts from the left)
- **Right module**: `*(::Tv, ::Tg) → Tv` (scalar acts from the right)
- **Z-module**: `*(::Int, ::Tv) → Tv` (integer scaling, implied by repeated addition)
- **Division**: `/(::Tv, ::Tg)` is the right inverse of scalar multiplication

This is the standard algebraic structure of a vector space (or module) over a field (or ring), which is the minimal requirement for polynomial interpolation.

## Known Limitations

### Series output type

`promote_type(Tv, Tq)` may return `Any` for custom types. The `_series_output_type` helper falls back to `Tv` in this case.
