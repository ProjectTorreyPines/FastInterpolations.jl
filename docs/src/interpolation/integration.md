# Integration

FastInterpolations.jl provides **exact analytical integration** computed directly from spline coefficients. No numerical quadrature is used.

## Basic Usage

The primary function is `integrate(itp, a, b)`, which computes the definite integral $\int_{a}^{b} f(x)\,dx$.

```@example integ
using FastInterpolations

x = range(0.0, 2π, 50)
y = sin.(x)
itp = cubic_interp(x, y)

# Compute ∫₀^π sin(x) dx
integrate(itp, 0.0, π)   # ≈ 2.0
nothing #hide
```

The bounds $a$ and $b$ can be any real numbers within the domain (or outside, if extrapolation is enabled). They do not need to be grid points.

Swapping bounds behaves consistently with calculus rules: $\int_a^b = -\int_b^a$.

```@example integ
integrate(itp, π, 0.0)   # ≈ -2.0
nothing #hide
```

### Keyword Arguments

```julia
integrate(itp, a, b; search=AutoSearch(), hint=nothing)
nothing #hide
```

- **`search`**: Search policy for locating the bounds in the grid. Defaults to `AutoSearch()`, which adapts automatically. See [Search & Hints](@ref search_hints) for details.
- **`hint`**: A `Ref{Int}` to speed up sequential queries by remembering the last search index.

---

## Full-Domain Integration

To integrate over the entire domain $[x_1, x_n]$, simply omit the bounds:

```@example integ
integrate(itp)
nothing #hide
```

!!! tip "Performance"
    `integrate(itp)` is faster than `integrate(itp, x[1], x[end])`. It uses a specialized summation path that avoids search operations and bound checks entirely.

---

## One-Shot Integration

When you need the full-domain integral once and not the interpolant itself, pass the raw data with a `method` directly. This mirrors the unified `interp(x, y; method=…)` API and builds the interpolant internally with no copies and no per-cell axis caches, so the call allocates nothing:

```@example integ
xs = range(0.0, π, 50)
ys = sin.(xs)

integrate(xs, ys; method = LinearInterp())   # trapezoidal (exact for the linear interpolant)
integrate(xs, ys; method = CubicInterp())    # spline quadrature
nothing #hide
```

The `method` keyword is required; its options (`bc`, `side`, `tension`) are forwarded to the underlying interpolant.

---

## Series Interpolants

For multi-channel data (Series Interpolants), integration works channel-wise.

### Scalar Integration
Returns a `Vector` containing the integral for each series.

```@example integ
x = range(0.0, 1.0, 30)
# Data with 2 channels
Y = hcat(sin.(π .* x), cos.(π .* x))

sitp = cubic_interp(x, Series(Y))

# Returns [∫ sin, ∫ cos]
integrate(sitp, 0.0, 1.0)
nothing #hide
```

### Cumulative Integration
`cumulative_integrate(sitp)` returns the prefix sums of integrals at grid points.

- **Scalar Interpolant**: Returns `Vector{T}` (length $N$).
- **Series Interpolant**: Returns `Matrix{T}` (size $N \times K$).

```@example integ
# Result is (30 × 2) matrix, matching logic of sitp.y
cum = cumulative_integrate(sitp)
nothing #hide
```

!!! note "Matrix Output"
    For Series, `cumulative_integrate` returns a `Matrix` rather than a `Vector{Vector}`. This keeps the output memory layout consistent with the input data layout.

---

## Extrapolation Behavior

Integration respects the `extrap` keyword used during interpolant creation.

```@example integ
x = range(0.0, 2π, 50)
y = sin.(x)

# Constant extrapolation means f(x) = f(x₁) for x < x₁
itp_flat = cubic_interp(x, y; extrap=ClampExtrap())

# Integrates the constant value outside the domain
integrate(itp_flat, -1.0, 10.0)
nothing #hide
```

| Mode | Effect on Integration |
|------|-----------------------|
| `NoExtrap()` | Error if bounds are outside domain. |
| `ClampExtrap()` | Linearly adds area (value × distance). |
| `ExtendExtrap()` | Evaluation continues using the polynomial of the boundary cell. |
| `WrapExtrap()` | Periodic integration. |

---

## API Summary

| Function | Description | Return Type |
|----------|-------------|-------------|
| `integrate(itp, a, b)` | Definite integral over $[a, b]$ | Scalar (or Vector for Series) |
| `integrate(itp)` | Full-domain integration | Scalar (or Vector for Series) |
| `integrate(x, y; method)` | One-shot full-domain integral from raw data | Scalar |
| `cumulative_integrate(itp)` | Grid-aligned indefinite integrals | Vector (Scalar) / **Matrix** (Series) |