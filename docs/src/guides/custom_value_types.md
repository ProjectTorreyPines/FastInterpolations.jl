# Custom Value Types (Duck Typing)

FastInterpolations supports interpolation with arbitrary value types — not just `Real` and `Complex`. Any type that supports the required arithmetic operations can be used as the value type `Tv`.

## Type System

Internally, every interpolant separates two type roles:

* **Grid type (`Tg`)**: Always `<: AbstractFloat` (`Float64`, `Float32`). Used for grid coordinates, spacings, and search. Must be ordered and real-valued.
* **Value type (`Tv`)**: Unconstrained. Used for function values (`y`), derivatives, and coefficients. Can be any type supporting the required operations.

## Quick Example

This is a **path interpolation with `SVector`** example.
Using `SVector{2,Float64}` waypoints, you can interpolate an arbitrary 2D path directly with one-shot APIs (`linear_interp`, `cubic_interp`) and visualize the result.

```@example duck_typing
using FastInterpolations
using StaticArrays
using Plots

# 9 waypoints in 2D (spiral-like path)
t = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8]
pts = SVector{2, Float64}[
    SVector(0.2, 0.0), SVector(0.5, 0.5),
    SVector(0.1, 1.0), SVector(-0.8, 0.9),
    SVector(-1.4, 0.0), SVector(-1.0, -1.1),
    SVector(0.1, -1.8), SVector(1.5, -1.2),
    SVector(2.0, 0.2),
]

tq = range(first(t), last(t), length=200) # dense query points for smooth curves

# Duck typing lets us interpolate Vector{SVector{2,Float64}} directly.
linear_path = linear_interp(t, pts, tq)
cubic_path = cubic_interp(t, pts, tq)

# Split Vector{SVector{2}} into x/y arrays for plotting
linear_x = getindex.(linear_path, 1)
linear_y = getindex.(linear_path, 2)
cubic_x = getindex.(cubic_path, 1)
cubic_y = getindex.(cubic_path, 2)
pts_x = getindex.(pts, 1)
pts_y = getindex.(pts, 2)

p = plot(linear_x, linear_y; label="linear_interp", lw=2, aspect_ratio=:equal);
plot!(p, cubic_x, cubic_y; label="cubic_interp", lw=2);
scatter!(p, pts_x, pts_y; label="control points", ms=6, color=:black);
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

```@example duck_typing
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

# Use it — all methods and BCs work (oneshot API)
x = range(0.0, 1.0, 10)
y = [MyValue(sin(xi)) for xi in x]

constant_interp(x, y, 0.5)                                     # → MyValue(...)
linear_interp(x, y, 0.5)                                       # → MyValue(...)
quadratic_interp(x, y, 0.5)                                    # → MyValue(...)
quadratic_interp(x, y, 0.5; bc=Left(Deriv1(zero(MyValue))))    # → MyValue(...)
cubic_interp(x, y, 0.5)                                        # → MyValue(...)
cubic_interp(x, y, 0.5; bc=Deriv1(zero(MyValue)))              # → MyValue(...)
cubic_interp(x, y, 0.5; bc=ZeroCurvBC())                       # → MyValue(...)
nothing #hide
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
