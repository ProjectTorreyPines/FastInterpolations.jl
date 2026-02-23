# PointBC: Single-Point Boundary Conditions

`PointBC` types specify derivative conditions at a single endpoint. They are the building blocks used by both **quadratic** and **cubic** splines.

---

## Derivative Types

### Deriv1: First Derivative (Slope)

Specify the slope `S'(endpoint) = v` directly:

```julia
Deriv1(0.0)   # Horizontal tangent
Deriv1(1.0)   # 45° upward slope
Deriv1(-2.0)  # Steep downward slope
```

### Deriv2: Second Derivative (Curvature)

Specify the curvature `S''(endpoint) = v`:

```julia
Deriv2(0.0)   # Zero curvature (inflection point behavior)
Deriv2(1.0)   # Positive curvature (concave up)
Deriv2(-1.0)  # Negative curvature (concave down)
```

### Deriv3: Third Derivative

For advanced use cases, specify `S'''(endpoint) = v`:

```julia
Deriv3(0.0)   # Constant second derivative near endpoint
```

---

## PolyFit: Automatic Derivative Estimation

When you don't know the exact derivative values, `PolyFit{D}` **estimates** them by fitting a degree-D polynomial to the nearest D+1 data points.

### How It Works

`PolyFit{D}` performs these steps:
1. Extract D+1 points from the endpoint (left: first D+1 points, right: last D+1 points)
2. Fit a Lagrange interpolating polynomial through these points
3. Evaluate the polynomial's derivative at the endpoint

This gives a data-driven estimate of the endpoint derivative without requiring manual specification.

### Available Degrees

| Type | Points | Accuracy | Formula (uniform grid, left endpoint) |
|------|--------|----------|--------------------------------------|
| `LinearFit()` = `PolyFit{1}` | 2 | O(h) | `(y₂ - y₁) / h` |
| `QuadraticFit()` = `PolyFit{2}` | 3 | O(h²) | `(-3y₁ + 4y₂ - y₃) / (2h)` |
| `CubicFit()` = `PolyFit{3}` | 4 | O(h³) | `(-11y₁ + 18y₂ - 9y₃ + 2y₄) / (6h)` |
| `PolyFit{N}()` (N > 3) | N+1 | O(hᴺ) | Barycentric interpolation |

!!! tip "Choosing the Right Degree"
    - **LinearFit**: Simplest, first-order accurate
    - **QuadraticFit**: Default for quadratic splines — exact for quadratic polynomials
    - **CubicFit**: Third-order accurate, exact for cubic polynomials
    - **PolyFit{N}**: Arbitrary degree N — uses barycentric interpolation for N > 3

    Match the polynomial degree to your data's smoothness. See the [visual comparison below](#Visual-Comparison) for examples.

!!! warning "Higher Degree ≠ Always Better"
    Higher-degree fits are **not always more accurate**. For oscillatory or noisy data,
    high-degree polynomial fits can produce **overshoot** at boundaries.
    When in doubt, start with `QuadraticFit` or `CubicFit` and compare results visually.

---

## Visual Comparison

### Part 1: Smooth Polynomial Data

When the data near the boundary follows a **smooth polynomial**, higher-degree `PolyFit` estimates the true derivative more accurately.

The plots below show `cubic_interp` applied to `f(x) = 2x³ - 13x` with different boundary conditions.
Notice how `CubicFit()` computes the exact boundary tangent, perfectly reproducing the original cubic.

```@setup polyfit_smooth
using FastInterpolations
using Plots

a, b, c, d = (2, 0, -13,0 )

f(x) = a*x^3 + b*x^2 + c*x + d
f_prime(x) = 3a*x^2 + 2b*x + c
# True cubic polynomial
# f(x) = x^3
# f_prime(x) = 3x^2
# Sparse sampling (5 points)

xs = collect(range(-2.0, 3.0, length=5))
ys = f.(xs)

x_dense = range(minimum(xs), maximum(xs), length=200)
# BCs to compare
bcs = [
    ("ZeroCurvBC (default)", ZeroCurvBC(), :gray),
    ("LinearFit", LinearFit(), :royalblue),
    ("QuadraticFit", QuadraticFit(), :green3),
    ("CubicFit", CubicFit(), :darkorange),
]

plots_list = []
for (title, bc, color) in bcs
    itp = cubic_interp(xs, ys; bc=bc)

    p = plot(title=title, legend=true, titlefontsize=15, dpi=200)

    # True function
    plot!(p, x_dense, f.(x_dense), color=:black,  linewidth=2, label="True f(x)=2x³-13x")

    # Spline
    x_spline = range(xs[1], xs[end], length=150)
    y_spline = [itp(x) for x in x_spline]
    plot!(p, x_spline, y_spline, color=color, linewidth=3, alpha=0.6, label="spline")

    # Data points
    scatter!(p, xs, ys, color=color, alpha=0.6, markerstrokecolor=:black, markersize=8, label=nothing)

    # Tangent lines at boundaries
    # for xi in [xs[1], xs[end]]
        xi=xs[1]
        yi = itp(xi)
        slope_est = itp(xi; deriv=DerivOp(1))

        # Draw estimated tangent
        len = 0.8
        xt = [xi - len, xi + len]
        yt = yi .+ slope_est .* (xt .- xi)
        plot!(p, xt, yt, color=color, linewidth=2, alpha=0.7, linestyle=:dash, label="boundary tangent")

        xi=xs[end]
        yi = itp(xi)
        slope_est = itp(xi; deriv=DerivOp(1))

        # Draw estimated tangent
        xt = [xi - len, xi + len]
        yt = yi .+ slope_est .* (xt .- xi)
        plot!(p, xt, yt, color=color, linewidth=2, alpha=0.7, linestyle=:dash, label=nothing)
    # end

    push!(plots_list, p)
end

# plot(plots_list..., layout=(2, 2), size=(800, 600), xlims=(-2.5, 3.5))
```

```@example polyfit_smooth
plot(plots_list..., layout=(2, 2), size=(800, 600), xlims=(-2.5, 3.5)) # hide
```

**Key observations:**
- **ZeroCurvBC**: Zero-curvature assumption causes deviation from the true cubic
- **LinearFit / QuadraticFit**: Slightly underestimate the boundary slope
- **CubicFit**: Computes the **exact boundary derivative**, enabling perfect reproduction ✓

!!! tip "Polynomial Reproduction Property"
    When the data near the boundary follows a smooth polynomial of degree ≤ D, `PolyFit{D}` computes the exact derivative.

---

### Part 2: Oscillatory Data — When Higher Degree Fails

For **oscillatory data** (e.g., `sin(x)` with sparse sampling), higher-degree `PolyFit` can produce **worse** results due to overshoot at boundaries.

The plots below show the **left boundary region** of a longer interpolation interval. The gray dashed line is the true function. Open circles are all data points, while **filled colored circles** indicate the points used to estimate the left boundary derivative (2 for `LinearFit`, 3 for `QuadraticFit`, 4 for `CubicFit`, etc.).

```@setup polyfit_oscillatory
using FastInterpolations
using Plots

f(x) = sin(3π * x)

xs = collect(range(0.0, 4.0, length=20))
ys = f.(xs)

xlim_range = (-0.05, 2.1)
x_dense = range(minimum(xs), maximum(xlim_range), length=200)
y_true = f.(x_dense)

plots_list = []

# (bc, title, color, n_highlight) - n_highlight is number of points used for derivative estimation
for (bc, title, color, n_highlight) in [
    (ZeroCurvBC(), "ZeroCurvBC (default)", :gray, 0),
    (LinearFit(), "LinearFit (D=1)", :royalblue, 2),
    (QuadraticFit(), "QuadraticFit (D=2)", :green3, 3),
    (CubicFit(), "CubicFit (D=3)", :darkorange, 4),
    (PolyFit{4}(), "PolyFit (D=4)", :mediumorchid, 5),
    (PolyFit{5}(), "PolyFit (D=5)", :mediumorchid, 6),
    (PolyFit{6}(), "PolyFit (D=6)", :mediumorchid, 7),
    (PolyFit{7}(), "PolyFit (D=7)", :mediumorchid, 8),
    (PolyFit{8}(), "PolyFit (D=8)", :mediumorchid, 9),
    (PolyFit{9}(), "PolyFit (D=9)", :mediumorchid, 10)
]
    itp = cubic_interp(xs, ys; bc=bc)

    p = plot(title=title, xlims=xlim_range, ylims=(-1.5, 1.5), legend=false, titlefontsize=15, dpi=200)

    # True function (gray dashed)
    plot!(p, x_dense, y_true, color=:gray, linestyle=:dash, linewidth=2)

    # Cubic spline
    x_spline = x_dense
    y_spline = [itp(x) for x in x_spline]
    plot!(p, x_spline, y_spline, color=color, linewidth=3, alpha=0.7)

    # Data points (white)
    scatter!(p, xs, ys, color=:white, markerstrokecolor=:black, markerstrokewidth=1.5, markersize=6)

    # Highlight points used for PolyFit (colored)
    if n_highlight > 0
        n_fit = min(n_highlight, length(xs))
        scatter!(p, xs[1:n_fit], ys[1:n_fit], color=color, markerstrokecolor=:black, markerstrokewidth=2, markersize=9)
    end

    push!(plots_list, p)
end
```

```@example polyfit_oscillatory
plot(plots_list..., layout=(5, 2), size=(800, 1200)) # hide
```

**Observations by degree:**

| Degree | Behavior |
|--------|----------|
| D=1–2 | Stable with minimal undershoot |
| D=3 | Overshoot begins to appear at the left boundary |
| D=4–6 | **Noticeable overshoot** at the left boundary |
| D=7 | Follows the true function with slight undershoot |
| D=8, 9 | **Severe overshoot with reversed slope** |

!!! danger "Runge-like Phenomenon"
    High-degree polynomial fits on oscillatory or non-polynomial data can produce wild oscillations near boundaries. The fitted polynomial tries to pass through many points but overshoots between them.

!!! tip "Practical Recommendation"
    - Start with **QuadraticFit** or **CubicFit**
    - Only use higher degrees if your data is known to be a smooth polynomial
    - Always **visually inspect** boundary behavior when changing `PolyFit` degree

---

## Usage Examples

### Quadratic Splines

```julia
using FastInterpolations

x = range(0.0, 2π, 15)
y = sin.(x)

# Default: QuadraticFit at left
quadratic_interp(x, y, 1.0)

# CubicFit at right for higher accuracy
quadratic_interp(x, y, 1.0; bc=Right(CubicFit()))

# Known slope at left
quadratic_interp(x, y, 1.0; bc=Left(Deriv1(1.0)))
```

### Cubic Splines

```julia
using FastInterpolations

x = range(0.0, 2π, 15)
y = sin.(x)

# Default: ZeroCurvBC (zero curvature)
cubic_interp(x, y, 1.0)

# Use CubicFit for better polynomial reproduction
cubic_interp(x, y, 1.0; bc=CubicFit())

# Mix PolyFit with known derivative
cubic_interp(x, y, 1.0; bc=BCPair(CubicFit(), Deriv1(0.0)))
```

---

## Mathematical Background

### Lagrange Interpolation

`PolyFit{D}` uses **barycentric Lagrange interpolation** for numerical stability:

```math
p(x) = \frac{\sum_{i=0}^{D} \frac{\beta_i y_i}{x - x_i}}{\sum_{j=0}^{D} \frac{\beta_j}{x - x_j}}
```

where ``\beta_i = \frac{1}{\prod_{j \neq i} (x_i - x_j)}`` are the barycentric weights.

The derivative is computed analytically from this form, avoiding explicit polynomial coefficient computation.

### Accuracy Order

For a smooth function ``f`` sampled on a grid with spacing ``h``:
- `PolyFit{D}` estimates ``f'(x_0)`` with error ``O(h^D)``
- Higher D → higher accuracy, but requires more points

---

## API Reference

See the full API documentation in [Types Reference](../api/types.md):

- [`PointBC`](@ref) — Abstract type for single-point BCs
- [`Deriv1`](@ref) — First derivative specification
- [`Deriv2`](@ref) — Second derivative specification
- [`Deriv3`](@ref) — Third derivative specification
- [`PolyFit`](@ref) — Polynomial fitting BC
- [`LinearFit`](@ref) — Alias for `PolyFit{1}`
- [`QuadraticFit`](@ref) — Alias for `PolyFit{2}`
- [`CubicFit`](@ref) — Alias for `PolyFit{3}`
