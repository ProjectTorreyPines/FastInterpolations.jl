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
```

The bounds $a$ and $b$ can be any real numbers within the domain (or outside, if extrapolation is enabled). They do not need to be grid points.

Swapping bounds behaves consistently with calculus rules: $\int_a^b = -\int_b^a$.

```@example integ
integrate(itp, π, 0.0)   # ≈ -2.0
```

### Keyword Arguments

```julia
integrate(itp, a, b; search=LinearBinary(), hint=nothing)
```

- **`search`**: Search policy for locating the bounds in the grid (e.g., `Binary()`, `Hunt()`). Defaults to the interpolant`s policy.
- **`hint`**: A `Ref{Int}` to speed up sequential queries by remembering the last search index.

---

## Full-Domain Integration

To integrate over the entire domain $[x_1, x_n]$, simply omit the bounds:

```@example integ
integrate(itp)
```

!!! tip "Performance"
    `integrate(itp)` is faster than `integrate(itp, x[1], x[end])`. It uses a specialized summation path that avoids search operations and bound checks entirely.

---

## Series Interpolants

For multi-channel data (Series Interpolants), integration works channel-wise.

### Scalar Integration
Returns a `Vector` containing the integral for each series.

```@example integ
x = range(0.0, 1.0, 30)
# Data with 2 channels
Y = hcat(sin.(π .* x), cos.(π .* x))

sitp = cubic_interp(x, Y)

# Returns [∫ sin, ∫ cos]
integrate(sitp, 0.0, 1.0)
```

### Cumulative Integration
`cumulative_integrate(sitp)` returns the prefix sums of integrals at grid points.

- **Scalar Interpolant**: Returns `Vector{T}` (length $N$).
- **Series Interpolant**: Returns `Matrix{T}` (size $N \times K$).

```@example integ
# Result is (30 × 2) matrix, matching logic of sitp.y
cum = cumulative_integrate(sitp)
size(cum)
```

!!! note "Matrix Output"
    For Series, `cumulative_integrate` returns a `Matrix` rather than a `Vector{Vector}`. This keeps the output memory layout consistent with the input data layout.

---

## Extrapolation Behavior

Integration respects the `extrap` keyword used during interpolant creation.

```@example integ
# Constant extrapolation means f(x) = f(x₁) for x < x₁
itp_flat = cubic_interp(x, y; extrap=:constant)

# Integrates the constant value outside the domain
integrate(itp_flat, -1.0, 10.0)
```

| Mode | Effect on Integration |
|------|-----------------------|
| `:none` | Error if bounds are outside domain. |
| `:constant` | Linearly adds area (value × distance). |
| `:extension` | Evaluation continues using the polynomial of the boundary cell. |
| `:wrap` | Periodic integration. |

---

## API Summary

| Function | Description | Return Type |
|----------|-------------|-------------|
| `integrate(itp, a, b)` | Definite integral over $[a, b]$ | Scalar (or Vector for Series) |
| `integrate(itp)` | Full-domain integration | Scalar (or Vector for Series) |
| `cumulative_integrate(itp)` | Grid-aligned indefinite integrals | Vector (Scalar) / **Matrix** (Series) |