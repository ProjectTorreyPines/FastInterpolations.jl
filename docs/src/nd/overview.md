# Multi-Dimensional Interpolation

FastInterpolations.jl supports 2D, 3D, and N-dimensional interpolation on **rectilinear grids**. The API generalizes the 1D case: where 1D takes `x`, ND takes `(x, y, ...)` as a **Tuple**.

!!! tip "Prerequisite"
    This section assumes familiarity with the [1D API](../interpolation/overview.md). Every 1D concept (methods, BCs, extrapolation, derivatives) extends to ND via Tuples.

---

## Quick Start

```julia
using FastInterpolations

# Define a 2D rectilinear grid and data
x = range(0.0, 2π, 20)
y = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]   # non-uniform
data = [sin(xi) * cos(yi) for xi in x, yi in y]

# Interpolant API (recommended)
itp = cubic_interp((x, y), data)
itp((1.0, 0.5))                         # scalar query

# One-shot API
cubic_interp((x, y), data, (1.0, 0.5))  # same result
```

---

## The Tuple Rule

Every 1D argument becomes a Tuple in ND. This applies uniformly:

| Concept | 1D | ND |
|:--------|:---|:---|
| Grid | `x` | `(x, y)` or `(x, y, z)` |
| Query (scalar) | `xq` | `(xq, yq)` |
| Query (batch) | `xqs::Vector` | `(xqs, yqs)` |
| BC | `bc=NaturalBC()` | `bc=(NaturalBC(), PeriodicBC())` |
| Extrap | `extrap=:constant` | `extrap=(:constant, :wrap)` |
| Derivative | `deriv=1` | `deriv=(1, 0)` for ∂f/∂x |
| Search | `search=Binary()` | `search=(Binary(), LinearBinary())` |

**Broadcast rule**: A scalar value is broadcast to all axes. `bc=NaturalBC()` is equivalent to `bc=(NaturalBC(), NaturalBC())` in 2D.

---

## Available Methods

All four interpolation methods support ND:

| Method | Function | BC Required? | Continuity |
|:-------|:---------|:-------------|:-----------|
| Constant | `constant_interp((x,y), data)` | No (`side` only) | C⁻¹ |
| Linear | `linear_interp((x,y), data)` | No | C⁰ |
| Quadratic | `quadratic_interp((x,y), data)` | Yes (1 per axis) | C¹ |
| Cubic | `cubic_interp((x,y), data)` | Yes (2 per axis) | C² |

---

## Grid Types

Each axis can independently be a `Range` (uniform, O(1) lookup) or `Vector` (non-uniform, O(log n) lookup). This allows **heterogeneous grids**:

```julia
x = range(0.0, 1.0, 100)          # uniform → O(1)
y = [0.0, 0.1, 0.4, 0.9, 1.5]    # non-uniform → O(log n)
itp = cubic_interp((x, y), data)   # mixed grid works
```

!!! warning "Data Orientation"
    `data` must satisfy `size(data, d) == length(grids[d])` for each dimension.

---

## Query Modes

### Scalar Query
```julia
itp((0.5, 1.0))  # returns a single value
```

### Batch Query (SoA — Structure of Arrays)
```julia
xqs = range(0.0, 1.0, 50)
yqs = range(0.0, 2.0, 50)
itp((xqs, yqs))  # returns Vector of length 50
```

### Batch Query (AoS — Array of Structures)
```julia
points = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
itp(points)  # returns Vector of length 3
```

---

## Visualization (2D)

2D interpolants have built-in plot recipes:

```julia
using Plots
itp = cubic_interp((x, y), data)
plot(itp)  # heatmap with grid nodes and gridlines
```

Custom options: `show_nodes`, `show_gridlines`, `resolution`, `node_color`, `gridline_style`. Use `help_plot(itp)` to discover all options.

---

## See Also

- **[Boundary Conditions](boundary_conditions.md)** — Per-axis BC configuration
- **[Derivatives](derivatives.md)** — Partial derivatives, gradient, hessian
- **[Extrapolation](extrapolation.md)** — Per-axis extrapolation modes
