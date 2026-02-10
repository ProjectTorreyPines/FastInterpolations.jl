# Integration (ND)

ND interpolants support exact analytical integration over hyper-rectangular regions using tensor-product decomposition of the 1D cell integrals.

---

## Basic Usage

```julia
integrate(itp, lower::NTuple{N}, upper::NTuple{N})  # definite integral over hyper-rectangle
integrate(itp)                                      # full-domain integral
```

- **`itp`**: Any ND interpolant (e.g., `CubicInterpolantND`, `LinearInterpolantND`).
- **`lower`**, **`upper`**: Tuples of length `N` specifying the corners of the integration region. Do not need to be grid points — partial-cell integration is handled automatically.

### Example: 2D

```@example nd_integ
using FastInterpolations

x = range(0.0, 1.0, 15)
y = range(0.0, 1.0, 15)
data = [xi * yi for xi in x, yi in y]
itp = cubic_interp((x, y), data)

# ∫∫ xy dx dy over [0,1]×[0,1] = 0.25
integrate(itp, (0.0, 0.0), (1.0, 1.0))
```

---

## Full-Domain Integration

Omit bounds to integrate over the entire domain:

```@example nd_integ
integrate(itp)
```

!!! tip "Performance"
    `integrate(itp)` uses a specialized summation path that avoids search operations entirely — faster than passing explicit bounds.

---

## See Also

- **[1D Integration](../interpolation/integration.md)** — 1D integration reference, Series support, extrapolation behavior
- **[Overview](overview.md)** — ND API introduction
