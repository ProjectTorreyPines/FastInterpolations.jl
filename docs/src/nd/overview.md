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
itp(1.0, 0.5)                            # scalar query

# One-shot API
cubic_interp((x, y), data, (1.0, 0.5))  # same result
```

---

## The Tuple Rule

Every 1D argument becomes a Tuple in ND. This applies uniformly:

| Concept | 1D | ND |
|:--------|:---|:---|
| Grid | `x` | `(x, y)` or `(x, y, z)` |
| Query (scalar) | `xq` | `(xq, yq)` or `itp(xq, yq)` |
| Query (batch) | `xqs::Vector` | `(xqs, yqs)` |
| BC | `bc=CubicFit()` | `bc=(CubicFit(), PeriodicBC())` |
| Extrap | `extrap=ClampExtrap()` | `extrap=(ClampExtrap(), WrapExtrap())` |
| Derivative | `deriv=DerivOp(1)` | `deriv=DerivOp(1, 0)` for ∂f/∂x |
| Search | `search=AutoSearch()` (default) | `search=AutoSearch()` (default) or per-axis tuple (e.g., `search=(AutoSearch(), AutoSearch())`) |

**Broadcast rule**: A scalar value is broadcast to all axes. `bc=CubicFit()` is equivalent to `bc=(CubicFit(), CubicFit())` in 2D.

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
```@example nd_query_modes
using FastInterpolations # hide
x = range(0.0, 1.0, 100) # hide
y = range(0.0, 2.0, 80) # hide
data = [sin(xi) * cos(yi) for xi in x, yi in y] # hide
itp = linear_interp((x, y), data) # hide
itp((0.5, 1.0))  # explicit tuple form
itp(0.5, 1.0)    # equivalent scalar convenience form
@assert itp((0.5, 1.0)) ≈ itp(0.5, 1.0) # hide
nothing # hide
```

### Batch Query (SoA — Structure of Arrays)
```@example nd_query_modes
xqs = range(0.0, 1.0, 50)
yqs = range(0.0, 2.0, 50)
itp((xqs, yqs))  # returns Vector of length 50
@assert length(itp((xqs, yqs))) == 50 # hide
nothing # hide
```

This form is **pairwise**: it evaluates `(xqs[i], yqs[i])` for each `i`.

### Rectilinear Query (`GriddedQuery`)

Use [`GriddedQuery`](@ref) when you want every combination of one query axis per
dimension, such as image resizing, resampling a coarse field onto a denser grid,
or sampling a surface on a regular output grid:

```@example nd_query_modes
x = 1:10
y = 1:20
data = [sin(xi) + cos(yi) for xi in x, yi in y]
itp = linear_interp((x, y), data)

# This is often what you want for image resize or coarse-to-fine resampling:
naive = [itp((xq, yq)) for xq in 1:10, yq in [5, 6, 7]]

# GriddedQuery computes the same tensor-product output with less repeated work.
gq = GriddedQuery(1:10, [5, 6, 7])     # Range axis + Vector axis

itp(gq)                              # returns Matrix with size (10, 3)
@assert isapprox(itp(gq), naive) # hide
nothing # hide
```

The in-place form writes to an output array with the same dimensionality and
size:

```@example nd_query_modes
out = Matrix{Float64}(undef, size(gq))
itp(out, gq)
@assert isapprox(out, naive) # hide
nothing # hide
```

Named one-shot calls can allocate the shaped output directly, or fill one you
provide:

```@example nd_query_modes
vals = linear_interp((x, y), data, gq)
linear_interp!(out, (x, y), data, gq)
@assert isapprox(vals, naive) && isapprox(out, naive) # hide
nothing # hide
```

For method-selected one-shot calls, use `interp` / `interp!` with the same
`GriddedQuery`:

```@example nd_query_modes
cubic_vals = interp((x, y), data, gq; method = CubicInterp(), extrap = ClampExtrap())
interp!(out, (x, y), data, gq; method = CubicInterp(), extrap = ClampExtrap())
@assert size(cubic_vals) == size(gq) && size(out) == size(gq) # hide
nothing # hide
```

For `N > 1`, `GriddedQuery` does not accept a flat vector output; allocate an
`N`-D array with `size(gq)`.

### Batch Query (AoS — Array of Structures)
```julia
points = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
itp(points)  # returns Vector of length 3
```

### Any Indexable Container (Query Protocol)

ND batch evaluation accepts **any container type** whose elements are indexable points. Types with standard `length`, `getindex`, and `eltype` semantics work zero-config:

```julia
using StaticArrays

# Vector{SVector} — works out of the box
pts = [SVector(0.1, 0.2), SVector(0.3, 0.4)]
itp(pts)  # returns Vector of length 2

# AbstractVector query also works for scalar calls
itp(SVector(0.5, 1.0))  # single-point evaluation
```

For custom containers where `Base` semantics differ (e.g., SoA-style wrappers), override three functions:

```julia
import FastInterpolations: _query_length, _query_extract, _query_eltype

_query_length(q::MyQueries)      = ...   # number of query points
_query_extract(q::MyQueries, k)  = ...   # k-th point (any indexable)
_query_eltype(q::MyQueries)      = ...   # scalar floating type (e.g. Float64)
```

!!! note "Value types vs Query types"
    This is orthogonal to [Custom Value Types (Duck Typing)](../guides/custom_value_types.md), which governs what types can be *interpolated* (`Tv`). The query protocol governs what container types can *hold query points*.

### Shape preservation

Batch evaluation preserves the **shape** of the query container. A vector query returns a `Vector`; a query that carries more than one dimension returns a dense `Array` of the same `size`:

```julia
pts = reshape([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)], 2, 2)
itp(pts)                       # 2×2 Matrix — itp(pts)[i,j] == itp(pts[i,j])

qx = [0.1 0.3; 0.5 0.7]; qy = [0.2 0.4; 0.6 0.8]
itp((qx, qy))                  # 2×2 Matrix (shaped SoA — requires size(qx) == size(qy))
```

In-place calls require the output to match the query shape **exactly** (equal length with a different shape is rejected):

```julia
out = Matrix{Float64}(undef, 2, 2)
itp(out, pts)                  # fills the 2×2 matrix
itp(Vector{Float64}(undef, 4), pts)   # DimensionMismatch — a length-4 vector is not a 2×2 sink
```

Scalar and vector queries are unchanged, and `GriddedQuery` keeps its Cartesian-product shape. Only a query container that already carried more than one dimension changes: it previously returned a flattened vector and now returns a same-shape array. To recover the old flat result, wrap the call: `vec(itp(pts))`.

---

## Visualization (2D)

2D interpolants have built-in plot recipes:

```julia
using Plots
itp = cubic_interp((x, y), data)
plot(itp)  # heatmap with grid nodes and gridlines
```

### Example — Method Comparison

Comparing constant, linear, quadratic, and cubic interpolation on a non-uniform 2D grid:

```@example nd_overview
using FastInterpolations, Plots
using Printf, LinearAlgebra  # hide

f(x, y) = sin(2π * x) * cos(2π * y)

xs = [0.0, 0.1, 0.4, 0.5, 0.82, 1.0]
ys = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
data = [f(xi, yj) for xi in xs, yj in ys]

itp_const = constant_interp((xs, ys), data)
itp_linear   = linear_interp((xs, ys), data)
itp_quad  = quadratic_interp((xs, ys), data; bc=MinCurvFit())
itp_cubic = cubic_interp((xs, ys), data)

kw = (c=:RdBu, clims=(-1,1), ratio=:equal, xlims=(0,1), ylims=(0,1))

itps = (itp_const, itp_linear, itp_quad, itp_cubic)
plot((plot(itp; kw...) for itp in itps)..., layout=(2,2), size=(950, 900))
```
Custom options: `show_nodes`, `show_gridlines`, `resolution`, `node_color`, `gridline_style`. Use `help_plot(itp)` to discover all options.

---

## Unified API: `interp` / `interp!`

The `interp` function provides a **single entry point** for N-dimensional interpolation with per-axis method specification. It supports both homogeneous (all axes same method) and heterogeneous (mixed methods per axis) interpolation.

```julia
# Homogeneous: auto-dispatches to CubicInterpolantND (same as cubic_interp)
itp = interp((x, y), data; method=CubicInterp())

# Heterogeneous: cubic on axis 1, linear on axis 2
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))
itp((0.5, 0.3))  # evaluate
gradient(itp, (0.5, 0.3))  # analytical gradient works
```

### One-Shot (Zero-Allocation)

Evaluate without creating an interpolant — ideal for hot loops with changing data:

```julia
# Scalar one-shot
val = interp((x, y), data, (0.5, 0.3); method=(CubicInterp(), LinearInterp()))

# Batch one-shot
vals = interp((x, y), data, (xqs, yqs); method=CubicInterp())

# In-place batch
interp!(output, (x, y), data, (xqs, yqs); method=CubicInterp())
```

### Per-Axis Options

Each axis can have its own method, boundary condition, and extrapolation:

```julia
itp = interp((x, y, z), data3d;
    method = (CubicInterp(bc=PeriodicBC()), LinearInterp(), QuadraticInterp()),
    extrap = (WrapExtrap(), ClampExtrap(), NoExtrap()),
)
```

For full details, see the dedicated **[Unified API Guide](unified_api.md)**.

---

## See Also

- **[Unified API (`interp`)](unified_api.md)** — Heterogeneous per-axis methods
- **[Boundary Conditions](boundary_conditions.md)** — Per-axis BC configuration
- **[Derivatives](derivatives.md)** — Partial derivatives, gradient, hessian
- **[Extrapolation](extrapolation.md)** — Per-axis extrapolation modes
