# Interpolation Overview

FastInterpolations.jl provides four interpolation methods with increasing smoothness. All methods support analytical **1st and 2nd derivatives**.

| Method | Continuity | 1st Derivative | 2nd Derivative |
|--------|------------|----------------|----------------|
| Constant | C⁻¹ (discontinuous) | 0 | 0 |
| Linear | C⁰ (continuous) | Piecewise constant | 0 |
| Quadratic | C¹ (smooth) | Continuous | Piecewise constant |
| Cubic | C² (smooth) | Continuous | Continuous |

---

## Grid Types

All interpolation methods support both uniform and non-uniform grids, but performance differs:

| Grid Type | Lookup Complexity | Recommendation |
|-----------|-------------------|----------------|
| `AbstractRange` | **O(1)** (Direct) | Best for uniform grids |
| `AbstractVector` | O(log n) (Binary Search) | Use only for non-uniform data |

!!! tip "Performance Guide"
    For details on minimizing allocations and choosing the right grid, see [Performance Tips](../guides/performance_tips.md).

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
- **[Cubic](cubic.md)**: C² with various boundary conditions
- **[Visual Comparison](comparison.md)**: Side-by-side plots of all methods
- **[Derivatives](derivatives.md)**: Detailed derivative documentation
