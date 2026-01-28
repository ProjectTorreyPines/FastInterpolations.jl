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

!!! tip "Choosing the Right Degree"
    - **LinearFit**: Simplest, but only first-order accurate
    - **QuadraticFit**: Default for quadratic splines — exact for quadratic polynomials
    - **CubicFit**: Best accuracy, exact for cubic polynomials

---

## Visual Comparison

The following visualization shows how different `PolyFit` degrees affect the estimated slope at the boundary:

```@example polyfit_visual
using FastInterpolations
using Plots

# Test function with visible nonlinearity at boundary
f(x) = sin(π * x) + 0.2 * cos(3π * x)
f_prime(x) = π * cos(π * x) - 0.6π * sin(3π * x)

# Sparse data (7 points)
n_points = 7
xs = collect(range(0.0, 1.0, length=n_points))
ys = f.(xs)

true_slope = f_prime(xs[1])

# Estimate slopes with each PolyFit degree
slopes = Dict(
    "LinearFit (D=1)" => FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:left), LinearFit()),
    "QuadraticFit (D=2)" => FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:left), QuadraticFit()),
    "CubicFit (D=3)" => FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:left), CubicFit()),
)

p = plot(
    title="Slope Estimation at Left Boundary",
    xlabel="x", ylabel="y",
    xlims=(-0.1, 0.35), ylims=(-0.15, 0.7),
    legend=:topleft, size=(550, 400)
)

# Boundary point
scatter!(p, [xs[1]], [ys[1]], color=:black, markersize=10, label="Boundary point")

# True tangent
tangent_len = 0.3
x_true = [xs[1] - tangent_len/2, xs[1] + tangent_len/2]
y_true = ys[1] .+ true_slope .* (x_true .- xs[1])
plot!(p, x_true, y_true, color=:black, linewidth=3, label="True slope ($(round(true_slope, digits=2)))")

# Estimated tangents
colors = [:royalblue, :green3, :darkorange]
for (i, (name, slope)) in enumerate(sort(collect(slopes), by=x->x[1]))
    x_est = [xs[1] - tangent_len/2, xs[1] + tangent_len/2]
    y_est = ys[1] .+ slope .* (x_est .- xs[1])
    plot!(p, x_est, y_est, color=colors[i], linewidth=2, linestyle=:dash,
          label="$name ($(round(slope, digits=2)))")
end

p
```

**Key Observation**: Higher polynomial degree → better slope estimate → closer to the true tangent (black line).

---

## Effect on Spline Shape

Different `PolyFit` degrees produce different spline behaviors, especially visible near boundaries:

```@example polyfit_spline
using FastInterpolations
using Plots

f(x) = sin(π * x) + 0.2 * cos(3π * x)

xs = collect(range(0.0, 1.0, length=7))
ys = f.(xs)

xlim_left = (-0.05, 0.55)
x_dense = range(xlim_left[1], xlim_left[2], length=200)
y_true = f.(x_dense)

plots_list = []

for (D, title, color) in [
    (1, "LinearFit (D=1)", :royalblue),
    (2, "QuadraticFit (D=2)", :green3),
    (3, "CubicFit (D=3)", :darkorange)
]
    bc = D == 1 ? LinearFit() : (D == 2 ? QuadraticFit() : CubicFit())
    itp = cubic_interp(xs, ys; bc=bc)

    p = plot(title=title, xlims=xlim_left, legend=false, titlefontsize=10)

    # True function (gray dashed)
    plot!(p, x_dense, y_true, color=:gray, alpha=0.5, linestyle=:dash, linewidth=1)

    # Cubic spline
    x_spline = range(xs[1], min(xs[end], xlim_left[2]), length=150)
    y_spline = [itp(x) for x in x_spline]
    plot!(p, x_spline, y_spline, color=color, linewidth=2.5)

    # Data points
    scatter!(p, xs, ys, color=:white, markerstrokecolor=:black,
             markerstrokewidth=1.5, markersize=6)

    # Highlight points used for fitting
    n_fit = D + 1
    scatter!(p, xs[1:n_fit], ys[1:n_fit], color=color,
             markerstrokecolor=:black, markerstrokewidth=2, markersize=9)

    push!(plots_list, p)
end

plot(plots_list..., layout=(1, 3), size=(900, 300))
```

**Colored points** show the data used by each `PolyFit` to estimate the boundary derivative.

---

## Derivative Estimation Accuracy

How estimation error decreases with polynomial degree:

```@example polyfit_accuracy
using FastInterpolations
using Plots

f(x) = exp(-x) * sin(2π * x)
f_prime(x) = exp(-x) * (2π * cos(2π * x) - sin(2π * x))

xs = collect(range(0.0, 2.0, length=10))
ys = f.(xs)

true_deriv_left = f_prime(xs[1])
true_deriv_right = f_prime(xs[end])

degrees = 1:6
errors_left = Float64[]
errors_right = Float64[]

for D in degrees
    pf = FastInterpolations.PolyFit{D}()
    est_left = FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:left), pf)
    est_right = FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:right), pf)
    push!(errors_left, abs(est_left - true_deriv_left))
    push!(errors_right, abs(est_right - true_deriv_right))
end

p = plot(
    title="Derivative Estimation Error vs PolyFit Degree",
    xlabel="PolyFit Degree (D)",
    ylabel="Absolute Error",
    yscale=:log10,
    legend=:topright,
    size=(550, 350)
)

scatter!(p, degrees, errors_left, color=:blue, markersize=8, label="Left endpoint")
plot!(p, degrees, errors_left, color=:blue, linewidth=1.5, label="")

scatter!(p, degrees, errors_right, color=:red, markersize=8, label="Right endpoint")
plot!(p, degrees, errors_right, color=:red, linewidth=1.5, label="")

p
```

!!! note "Diminishing Returns"
    Beyond D=3-4, accuracy gains diminish and numerical stability may decrease. `CubicFit()` (D=3) is usually the sweet spot.

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

# Default: NaturalBC (zero curvature)
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
