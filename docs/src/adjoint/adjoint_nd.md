# Adjoint Operators (ND)

FastInterpolations provides native adjoint operators for all N-dimensional interpolant types.
Each operator is **query-baked and data-free**: constructed from grids and query points only,
then applied to any sensitivity vector $\bar{\mathbf{y}}$.

| Constructor | Struct | Notes |
|-------------|--------|-------|
| `constant_adjoint(grids, queries; ...)` | [`ConstantAdjointND`](@ref) | Tensor-product scatter |
| `linear_adjoint(grids, queries; ...)` | [`LinearAdjointND`](@ref) | Tensor-product scatter |
| `quadratic_adjoint(grids, queries; ...)` | [`QuadraticAdjointND`](@ref) | Per-axis recurrence adjoint + scatter |
| `cubic_adjoint(grids, queries; ...)` | [`CubicAdjointND`](@ref) | Per-axis transpose solve + scatter |

---

## Quick Start

```@example adjoint_nd
using FastInterpolations
using LinearAlgebra: dot

x = range(0.0, 1.0, 20)
y = range(0.0, 2.0, 15)
xq = range(0.1, 0.9, 10)
yq = range(0.2, 1.8, 10)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# Build the adjoint operator (grids + queries only, no data)
adj = cubic_adjoint((x, y), (xq, yq))
```

```@example adjoint_nd
# Apply to a sensitivity vector
ȳ = randn(10)
f̄ = adj(ȳ)
size(f̄)    # (20, 15) — same shape as data grid
```

### Dot-Product Identity

Verify $\langle W f, \bar{y} \rangle = \langle f, W^\top \bar{y} \rangle$:

```@example adjoint_nd
itp = cubic_interp((x, y), data)
Wf = itp((xq, yq))               # forward: W·f
WTy = adj(ȳ)                      # adjoint: Wᵀ·ȳ

@assert dot(Wf, ȳ) ≈ dot(vec(data), vec(WTy))
println("⟨Wf, ȳ⟩ = ", dot(Wf, ȳ))
println("⟨f, Wᵀȳ⟩ = ", dot(vec(data), vec(WTy)))
```

---

## Constructor Signatures

```julia
constant_adjoint(grids, queries; side=NearestSide(), extrap=NoExtrap())
linear_adjoint(grids, queries; extrap=NoExtrap())
quadratic_adjoint(grids, queries; bc=Left(QuadraticFit()), extrap=NoExtrap())
cubic_adjoint(grids, queries; bc=CubicFit(), extrap=NoExtrap())
```

The ND constructors accept tuples of grids and query vectors.
`bc` and `extrap` accept a single value (broadcast to all axes) or a per-axis N-tuple.

See [Adjoint API Reference](../api/adjoint.md) for full docstrings.

---

## Calling the Operator

### Allocating

```julia
f̄ = adj(ȳ)                                        # value adjoint
f̄ = adj(ȳ; deriv=DerivOp(1))                      # ∂/∂x broadcast to all axes
f̄ = adj(ȳ; deriv=(DerivOp(1), EvalValue()))        # mixed: ∂/∂x₁ only
```

### In-Place (Zero Allocation)

```@example adjoint_nd
∇f_buf = zeros(20, 15)
adj(∇f_buf, ȳ)    # zeros + accumulates into ∇f_buf
∇f_buf[1:3, 1:3]
```

---

## Native API vs AD

### Native Adjoint (Recommended for Performance)

Construct once, apply many times with different $\bar{\mathbf{y}}$:

```@example adjoint_nd
adj = cubic_adjoint((x, y), (xq, yq))

# Gradient of L2 loss: ∂/∂f ||W·f - y_obs||²  =  2·Wᵀ·(W·f - y_obs)
y_obs = randn(10)
residual = itp((xq, yq)) .- y_obs
∇f = adj(2.0 .* residual)
size(∇f)
```

### AD Integration (Zygote / Enzyme)

Zygote and Enzyme use the native adjoint operators internally via registered AD rules.
You get native performance through the standard AD interface:

```julia
using Zygote

# ∂loss/∂data — uses the appropriate adjoint operator internally
∇data = Zygote.gradient(
    d -> sum(abs2, cubic_interp((x, y), d, (xq, yq)) .- y_obs), data
)[1]

# Works with all types:
∇data = Zygote.gradient(
    d -> sum(abs2, linear_interp((x, y), d, (xq, yq)) .- y_obs), data
)[1]
```

```julia
using Enzyme

loss(d, g, q, y_obs) = sum(abs2, cubic_interp(g, d, q) .- y_obs)
df = zeros(20, 15)
Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active,
    Enzyme.Duplicated(copy(data), df),
    Enzyme.Const((x, y)), Enzyme.Const((xq, yq)), Enzyme.Const(y_obs))
# df now contains ∂loss/∂data
```

For full backend compatibility details, see [Adjoint via AD](../guides/adjoint_ad.md).

---

## Matrix Materialization

For verification and debugging, materialize the full weight matrix.
This is $O(n \times m)$ and **not** intended for production use.

```@example adjoint_nd
adj_small = cubic_adjoint(
    (range(0.0, 1.0, 5), range(0.0, 1.0, 4)),
    (rand(3), rand(3))
)
Wᵀ = Matrix(adj_small)
size(Wᵀ)          # (20, 3) — 5×4 grid flattened
```

---

## Performance

- **Construction**: $O(\sum_d n_d + m \log n_d)$ — per-axis factorization + anchor search
- **Apply (per call)**: $O(\sum_d n_d + m \cdot K^N)$ — per-axis transpose solve + tensor-product scatter ($K$ = kernel width: 1 for constant, 2 for linear, 3 for quadratic, 4 for cubic)
- **In-place**: Zero allocation after construction (all types)
- **Cache sharing**: Reuses the same spline caches as forward interpolation

---

## See Also

- **[Adjoint Overview](overview.md)**: Concepts and mathematical background
- **[Adjoint 1D](adjoint_1d.md)**: 1D adjoint operators
- **[Adjoint via AD](../guides/adjoint_ad.md)**: Using AD backends for `∂f/∂y`
- **[Adjoint API Reference](../api/adjoint.md)**: Full docstrings for all adjoint types
