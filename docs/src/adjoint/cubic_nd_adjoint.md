# Cubic Adjoint (ND)

The [`CubicAdjointND`](@ref) operator computes $\bar{\mathbf{f}} = W^\top \bar{\mathbf{y}}$
for N-dimensional cubic spline interpolation — mapping query-space sensitivities back to grid-space.

It is **query-baked and data-free**: constructed from grids and query points only.
The same operator can be applied to any $\bar{\mathbf{y}}$ vector.

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

Verify correctness via the **dot-product identity** $\langle W f, \bar{y} \rangle = \langle f, W^\top \bar{y} \rangle$:

```@example adjoint_nd
itp = cubic_interp((x, y), data)
Wf = itp((xq, yq))               # forward: W·f
WTy = adj(ȳ)                      # adjoint: Wᵀ·ȳ

@assert dot(Wf, ȳ) ≈ dot(vec(data), vec(WTy))
println("⟨Wf, ȳ⟩ = ", dot(Wf, ȳ))
println("⟨f, Wᵀȳ⟩ = ", dot(vec(data), vec(WTy)))
```

---

## Constructor

```julia
cubic_adjoint(grids, queries; bc=CubicFit(), extrap=NoExtrap()) -> CubicAdjointND
```

The constructor accepts the same kwargs as `cubic_interp`.
`bc` and `extrap` accept a single value (broadcast to all axes) or a per-axis N-tuple.

See [Adjoint API Reference](../api/adjoint.md) for full docstrings.

---

## Native API vs AD

There are two ways to compute $\partial L / \partial \mathbf{f}$ through an ND cubic interpolation.
The native API is faster because it avoids AD overhead; AD is convenient when the interpolation
is embedded in a larger differentiable pipeline.

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

In-place version for **zero allocation** in hot loops:

```@example adjoint_nd
∇f_buf = zeros(20, 15)
adj(∇f_buf, 2.0 .* residual)    # zeros + accumulates into ∇f_buf
∇f_buf[1:3, 1:3]
```

Derivative adjoint — adjoint of $\partial^k / \partial x_i^k$ instead of value interpolation:

```julia
f̄ = adj(ȳ; deriv=DerivOp(1))                      # ∂/∂x broadcast to all axes
f̄ = adj(ȳ; deriv=(DerivOp(1), EvalValue()))        # mixed partial ∂/∂x₁ only
```

### AD Integration (Zygote / Enzyme)

Zygote and Enzyme use the native adjoint operators internally via registered AD rules.
This works for **all four interpolant types** — not just cubic. You get native adjoint
performance through the standard AD interface:

```julia
using Zygote

# ∂loss/∂data — uses CubicAdjointND internally
∇data = Zygote.gradient(
    d -> sum(abs2, cubic_interp((x, y), d, (xq, yq)) .- y_obs), data
)[1]

# Works with all types:
∇data = Zygote.gradient(
    d -> sum(abs2, linear_interp((x, y), d, (xq, yq)) .- y_obs), data
)[1]

# Derivative adjoint also works through AD
∇data_d1 = Zygote.gradient(
    d -> sum(cubic_interp((x, y), d, (xq, yq); deriv=DerivOp(1))), data
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

For verification and debugging, you can materialize the full weight matrix.
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

- **Construction**: $O(\sum_d n_d + m \log n_d)$ — per-axis Thomas factorization + anchor search
- **Apply (per call)**: $O(\sum_d n_d + m \cdot 4^N)$ — per-axis transpose solve + tensor-product scatter
- **In-place**: **Zero allocation** after construction
- **Cache sharing**: Reuses the same `CubicSplineCache` per axis as forward interpolation

!!! tip "Dual Cache Architecture"
    `CubicAdjointND` maintains two sets of caches: user-BC caches (for pure derivatives)
    and mixed-partial caches (CubicFit, for cross-axis derivatives). When the user's BC is
    already `CubicFit`, both cache sets are identical and Julia optimizes the branch away.

---

## See Also

- **[Adjoint Overview](overview.md)**: Concepts and mathematical background
- **[Cubic Adjoint (1D)](cubic_1d_adjoint.md)**: 1D adjoint API reference
- **[Adjoint via AD](../guides/adjoint_ad.md)**: Using AD backends for `∂f/∂y`
- **[ND Interpolation](../nd/overview.md)**: Forward ND cubic spline API
- **[ND Derivatives](../nd/derivatives.md)**: Analytical derivatives for ND
