# Advanced Usage

Once you're comfortable with basic interpolation, these features help optimize for specific use cases.

## Features

| Feature | Use Case | Description |
|:--------|:---------|:------------|
| [**Series Interpolant**](@ref series_interpolant) | Multiple y-series on shared x-grid | 10-120× faster than separate interpolants |
| [**Search & Hints**](@ref search_hints) | Sequential or streaming queries | O(1) amortized lookup with hints |
| [**Memory & Allocation**](@ref memory_allocation) | Tight loops, real-time systems | Zero-allocation patterns |

## Quick Decision Guide

**Which feature do I need?**

- **"I have multiple outputs for the same input grid"** → [Series Interpolant](@ref series_interpolant)
- **"I query points in sequential order (ODE solvers, streaming)"** → [Search & Hints](@ref search_hints)
- **"I need zero allocations in my hot loop"** → [Memory & Allocation](@ref memory_allocation)

These features can be combined. For example, a Series Interpolant can use `LinearBounded` search policy with external hints for maximum performance in ODE integrator callbacks.
