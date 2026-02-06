# Derivatives (ND)

ND interpolants support analytical partial derivatives via the `deriv` keyword, plus dedicated `gradient`, `hessian`, and `laplacian` functions.

---

## Partial Derivatives with `deriv`

The `deriv` keyword accepts a Tuple specifying the derivative order for each axis:

```julia
itp = cubic_interp((x, y), data)

# Partial derivatives
itp((0.5, 1.0); deriv=(1, 0))   # ∂f/∂x
itp((0.5, 1.0); deriv=(0, 1))   # ∂f/∂y
itp((0.5, 1.0); deriv=(2, 0))   # ∂²f/∂x²
itp((0.5, 1.0); deriv=(1, 1))   # ∂²f/∂x∂y (mixed partial)
itp((0.5, 1.0); deriv=(0, 2))   # ∂²f/∂y²
```

**Shorthand**: `deriv=0` (scalar) evaluates the function value (default). Using `Val` for compile-time dispatch is also supported:

```julia
itp((0.5, 1.0); deriv=Val((1, 0)))  # same as deriv=(1, 0), but type-stable
```

### Maximum Derivative Orders

| Method | Max per axis | Mixed partials? |
|:-------|:-------------|:----------------|
| Constant | 0 (always 0) | N/A |
| Linear | 1 | No |
| Quadratic | 2 | Yes |
| Cubic | 3 | Yes |

---

## Vector Calculus Functions

These functions provide convenient access to common differential operators. They use analytical derivatives internally — **no automatic differentiation**.

### `gradient`

```julia
itp = cubic_interp((x, y, z), data)

gradient(itp, (0.5, 1.0, 0.3))    # → (∂f/∂x, ∂f/∂y, ∂f/∂z) as NTuple
gradient(itp, [0.5, 1.0, 0.3])    # Vector input → Vector output
```

In-place version for zero-allocation loops:

```julia
G = zeros(3)
gradient!(G, itp, (0.5, 1.0, 0.3))  # writes into G
```

### `hessian`

```julia
H = hessian(itp, (0.5, 1.0, 0.3))   # → 3×3 Matrix
# H[i,j] = ∂²f/∂xᵢ∂xⱼ (symmetric, computed efficiently)
```

In-place:

```julia
H = zeros(3, 3)
hessian!(H, itp, (0.5, 1.0, 0.3))
```

### `laplacian`

```julia
∇²f = laplacian(itp, (0.5, 1.0, 0.3))  # ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
```

### Performance

Analytical derivatives are significantly faster than AD:

| Function | Analytical | ForwardDiff | Speedup |
|:---------|:-----------|:------------|:--------|
| `gradient` (2D) | ~50 ns | ~414 ns | **~8×** |
| `hessian` (2D) | ~100 ns | ~1.5 μs | **~15×** |

---

## Optim.jl Integration

The in-place functions are directly compatible with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl):

```julia
using Optim

itp = cubic_interp((x, y), data)
f(x) = itp(Tuple(x))
grad!(G, x) = gradient!(G, itp, x)
hess!(H, x) = hessian!(H, itp, x)

result = optimize(f, grad!, hess!, [0.5, 0.5], NewtonTrustRegion())
```

For details, see [Optimization Guide](../guides/optimization.md).

---

## See Also

- **[1D Derivatives](../interpolation/derivatives.md)** — 1D derivative reference
- **[AD Support (ND)](../guides/autodiff_nd.md)** — ForwardDiff / Zygote integration
- **[Overview](overview.md)** — ND API introduction
