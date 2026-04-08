# Interpolation Overview

!!! note "Looking for multi-dimensional interpolation?"
    See [Multi-Dimensional Interpolation](../nd/overview.md) for 2D, 3D, and higher-dimensional interpolation.

FastInterpolations.jl provides five interpolation families with increasing smoothness. All methods support analytical **1st and 2nd derivatives**.

| Method | Continuity | 1st Derivative | 2nd Derivative |
|--------|------------|----------------|----------------|
| Constant | C⁻¹ (discontinuous) | 0 | 0 |
| Linear | C⁰ (continuous) | Piecewise constant | 0 |
| Quadratic | C¹ (smooth) | Continuous | Piecewise constant |
| Cubic | C² (smooth) | Continuous | Continuous |
| Local Cubic Hermite | C¹ (smooth) | Continuous | Piecewise linear |

**Cubic vs Local Cubic Hermite**: `Cubic` is a global C² spline built from a single tridiagonal solve — maximum smoothness, but one changed data point perturbs the entire curve. The Local Cubic Hermite family (`pchip_interp`, `cardinal_interp`, `akima_interp`, `hermite_interp`) determines each cell's slopes from a small 3- or 5-point stencil — C¹ only, but local control, monotonicity preservation (PCHIP), or outlier robustness (Akima).

---

## Grid Types

All interpolation methods support both uniform and non-uniform grids, but performance differs:

| Grid Type | Lookup Complexity | Recommendation |
|-----------|-------------------|----------------|
| `AbstractRange` | **O(1)** (Direct) | Best for uniform grids |
| `AbstractVector` | O(log n) (binary search) | Use only for non-uniform data |

!!! tip "Performance Guide"
    For details on minimizing allocations and choosing the right grid, see [Memory & Allocation](../guides/memory_allocation.md).

---

## API Selection

To balance convenience and maximum performance, three API patterns are available:

- **One-shot**: For dynamic data or simple scripts.
- **Interpolant**: For static lookups (pre-computed).
- **SeriesInterpolant**: For handling multiple series efficiently (10-120× faster scalar queries).

!!! tip "Selection Guide"
    Not sure which one to use? Check the [API Selection Guide](../guides/api_selection.md).

---

## Next Steps

- **[Constant](constant.md)**: Step interpolation with `side` modes
- **[Linear](linear.md)**: Simple and fast
- **[Quadratic](quadratic.md)**: C¹ with single-endpoint BC
- **[Cubic](cubic.md)**: C² global spline with various boundary conditions
- **[Local Cubic Hermite](local_hermite.md)**: C¹ local family — PCHIP, Cardinal, Akima, user-supplied slopes
- **[Visual Comparison](comparison.md)**: Side-by-side plots of all methods
- **[Derivatives](derivatives.md)**: Detailed derivative documentation
- **[Integration](integration.md)**: Analytical integration over splines
