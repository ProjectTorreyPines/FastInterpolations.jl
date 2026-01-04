# Visual Comparison

All examples use the same sparse, non-uniform grid interpolating `sin(x)`:

```julia
using FastInterpolations

x = [0.0, 0.9, 1.5, 2.2, 3.5, 4.5, 5.5, 2π]  # 8 non-uniform points
y = sin.(x)
xq = range(0, 2π, 500)  # query points
```

```@example comparison
using FastInterpolations  # hide
using Plots  # hide
x = [0.0, 0.9, 1.5, 2.2, 3.5, 4.5, 5.5, 2π]  # hide
y = sin.(x)  # hide
xq = range(0, 2π, 500)  # hide
```

---

## Value: ``S(x)``

```julia
constant_interp(x, y, xq)   # step function
linear_interp(x, y, xq)     # piecewise linear
quadratic_interp(x, y, xq)  # C¹ smooth
cubic_interp(x, y, xq)      # C² smooth
```

```@example comparison
p = plot(layout=(2,2), size=(800, 600), legend=:bottomleft, dpi=250)  # hide
methods = [  # hide
    ("Constant", xq -> constant_interp(x, y, xq)),  # hide
    ("Linear", xq -> linear_interp(x, y, xq)),  # hide
    ("Quadratic", xq -> quadratic_interp(x, y, xq)),  # hide
    ("Cubic", xq -> cubic_interp(x, y, xq))  # hide
]  # hide
for (i, (name, interp)) in enumerate(methods)  # hide
    yq = interp(xq)  # hide
    plot!(p[i], xq, yq, label="$name", linewidth=2)  # hide
    plot!(p[i], xq, sin.(xq), label="sin(x)", linestyle=:dash, alpha=0.7, color=:black)  # hide
    scatter!(p[i], x, y, label="data", markersize=6, color=:black)  # hide
    title!(p[i], name)  # hide
    ylims!(p[i], -1.3, 1.3)  # hide
end  # hide
p  # hide
```

---

## First Derivative: ``\frac{dS}{dx}``

```julia
constant_interp(x, y, xq; deriv=1)   # always 0
linear_interp(x, y, xq; deriv=1)     # piecewise constant
quadratic_interp(x, y, xq; deriv=1)  # continuous
cubic_interp(x, y, xq; deriv=1)      # smooth
```

```@example comparison
p = plot(layout=(2,2), size=(800, 600), legend=:bottomleft, dpi=250)  # hide
methods_d1 = [  # hide
    ("Constant", xq -> constant_interp(x, y, xq; deriv=1)),  # hide
    ("Linear", xq -> linear_interp(x, y, xq; deriv=1)),  # hide
    ("Quadratic", xq -> quadratic_interp(x, y, xq; deriv=1)),  # hide
    ("Cubic", xq -> cubic_interp(x, y, xq; deriv=1))  # hide
]  # hide
for (i, (name, interp)) in enumerate(methods_d1)  # hide
    yq = interp(xq)  # hide
    plot!(p[i], xq, yq, label="S'(x)", linewidth=2, color=:red)  # hide
    plot!(p[i], xq, cos.(xq), label="cos(x)", linestyle=:dash, alpha=0.7, color=:black)  # hide
    scatter!(p[i], x, cos.(x), label="cos(xᵢ)", markersize=6, color=:black)  # hide
    hline!(p[i], [0], color=:gray, linestyle=:dot, label=nothing)  # hide
    title!(p[i], "$name")  # hide
    ylims!(p[i], -1.3, 1.3)  # hide
end  # hide
p  # hide
```

---

## Second Derivative: ``\frac{d^2S}{dx^2}``

```julia
constant_interp(x, y, xq; deriv=2)   # always 0
linear_interp(x, y, xq; deriv=2)     # always 0
quadratic_interp(x, y, xq; deriv=2)  # piecewise constant
cubic_interp(x, y, xq; deriv=2)      # continuous
```

```@example comparison
p = plot(layout=(2,2), size=(800, 600), legend=:bottomleft, dpi=250)  # hide
methods_d2 = [  # hide
    ("Constant", xq -> constant_interp(x, y, xq; deriv=2)),  # hide
    ("Linear", xq -> linear_interp(x, y, xq; deriv=2)),  # hide
    ("Quadratic", xq -> quadratic_interp(x, y, xq; deriv=2)),  # hide
    ("Cubic", xq -> cubic_interp(x, y, xq; deriv=2))  # hide
]  # hide
for (i, (name, interp)) in enumerate(methods_d2)  # hide
    yq = interp(xq)  # hide
    plot!(p[i], xq, yq, label="S''(x)", linewidth=2, color=:green)  # hide
    plot!(p[i], xq, -sin.(xq), label="-sin(x)", linestyle=:dash, alpha=0.7, color=:black)  # hide
    scatter!(p[i], x, -sin.(x), label="-sin(xᵢ)", markersize=6, color=:black)  # hide
    hline!(p[i], [0], color=:gray, linestyle=:dot, label=nothing)  # hide
    title!(p[i], "$name")  # hide
    ylims!(p[i], -1.3, 1.3)  # hide
end  # hide
p  # hide
```

---

## Summary

| Method | Value | 1st Derivative | 2nd Derivative |
|--------|-------|----------------|----------------|
| **Constant** | Step function | Always 0 | Always 0 |
| **Linear** | Piecewise linear | Piecewise constant | Always 0 |
| **Quadratic** | C¹ smooth | Continuous | Piecewise constant |
| **Cubic** | C² smooth | Continuous | Continuous |

!!! tip "Choosing a Method"
    - **Speed priority**: Linear (simplest, fastest)
    - **Smooth curves**: Cubic (C² continuity)
    - **Balance**: Quadratic (C¹, simpler BC than cubic)
    - **Discrete states**: Constant (preserves steps)

---

## See Also

- **[Derivatives](derivatives.md)**: Detailed derivative API documentation
- **[Constant](constant.md)** | **[Linear](linear.md)** | **[Quadratic](quadratic.md)** | **[Cubic](cubic.md)**
