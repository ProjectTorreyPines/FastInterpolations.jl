# Derivatives (ND)

ND interpolants support analytical partial derivatives via the `deriv` keyword, plus dedicated `gradient`, `hessian`, and `laplacian` functions.

---

## Partial Derivatives with `deriv`

The `deriv` keyword accepts a `DerivOp` Tuple specifying the derivative order for each axis:

```julia
itp = cubic_interp((x, y), data)

# Partial derivatives — use DerivOp(n1, n2) convenience constructor
itp((0.5, 1.0); deriv=DerivOp(1, 0))   # ∂f/∂x
itp((0.5, 1.0); deriv=DerivOp(0, 1))   # ∂f/∂y
itp((0.5, 1.0); deriv=DerivOp(2, 0))   # ∂²f/∂x²
itp((0.5, 1.0); deriv=DerivOp(1, 1))   # ∂²f/∂x∂y (mixed partial)
itp((0.5, 1.0); deriv=DerivOp(0, 2))   # ∂²f/∂y²
```

The multi-argument `DerivOp(n1, n2, ...)` constructor returns a Tuple of `DerivOp` singletons. You can also construct the Tuple explicitly:

```julia
# Equivalent forms:
itp((0.5, 1.0); deriv=DerivOp(1, 0))                    # convenience constructor
itp((0.5, 1.0); deriv=(DerivOp(1), DerivOp(0)))          # explicit Tuple
itp((0.5, 1.0); deriv=(EvalDeriv1(), EvalValue()))       # using aliases
```

---

## `DerivativeView` — Callable Derivative Wrapper

[`DerivativeView`](@ref) is a lightweight callable wrapper that fixes the derivative order at construction, enabling broadcast (`dx.(pts)`), dot-call fusion (`@. dx(pts) + dy(pts)`), and higher-order function composition — all zero-allocation. For ND interpolants, create one with `deriv_view` (the 1D shortcuts `deriv1`/`deriv2`/`deriv3` are not supported for ND).

```julia
itp = cubic_interp((x, y), data)

# Create views — order specified per axis as a Tuple
dx  = deriv_view(itp, (1, 0))   # ∂f/∂x
dy  = deriv_view(itp, (0, 1))   # ∂f/∂y
dxx = deriv_view(itp, (2, 0))   # ∂²f/∂x²
dxy = deriv_view(itp, (1, 1))   # ∂²f/∂x∂y

# Evaluate — equivalent to itp((0.5, 1.0); deriv=DerivOp(1, 0))
dx((0.5, 1.0))

# Broadcast over points
points = [(0.5, 0.3), (1.0, 0.5), (1.5, 0.8)]
slopes = dx.(points)

# Fused broadcast — zero temporaries
result = @. dx(points) + dy(points)
```

The derivative order is locked at construction — passing `deriv=...` to a view throws an `ArgumentError`. Create a new view for a different order.

---

## Vector Calculus Functions

These functions provide convenient access to common differential operators. They use analytical derivatives internally — **no automatic differentiation**.

### `gradient`

```julia
itp = cubic_interp((x, y, z), data)

# Out-of-place version (returns vector)
gradient(itp, (0.5, 1.0, 0.3))    # → (∂f/∂x, ∂f/∂y, ∂f/∂z) as NTuple
gradient(itp, [0.5, 1.0, 0.3])    # Vector input → Vector output

# In-place version for zero-allocation loops:
G = zeros(3)
gradient!(G, itp, (0.5, 1.0, 0.3))  # writes into G
```

### `value_gradient`

Computes value and gradient in a single call, performing interval search only **once**. Faster than calling `itp(query)` and `gradient(itp, query)` separately. Ideal for [`Optim.jl`'s `fg!` pattern](../guides/optimization.md).

```julia
val, grad = value_gradient(itp, (0.5, 1.0, 0.3))  # → (f, (∂f/∂x, ∂f/∂y, ∂f/∂z))
val, grad = value_gradient(itp, [0.5, 1.0, 0.3])   # Vector input → Vector gradient
```

### `hessian`

```julia
# Out-of-place version (returns matrix)
H = hessian(itp, (0.5, 1.0, 0.3))   # → 3×3 Matrix
# H[i,j] = ∂²f/∂xᵢ∂xⱼ (symmetric, computed efficiently)

# In-place version for zero-allocation loops:
H = zeros(3, 3)
hessian!(H, itp, (0.5, 1.0, 0.3))
```

### `laplacian`

```julia
∇²f = laplacian(itp, (0.5, 1.0, 0.3))  # ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
```

---

## See Also

- **[1D Derivatives](../interpolation/derivatives.md)** — 1D derivative reference
- **[AD Support (ND)](../guides/autodiff_nd.md)** — ForwardDiff / Zygote / Enzyme integration
- **[Optimization](../guides/optimization.md)** - Seamless Optimization with Optim.jl
- **[Overview](overview.md)** — ND API introduction
