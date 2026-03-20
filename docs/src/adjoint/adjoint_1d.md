# Adjoint Operators (1D)

FastInterpolations provides native adjoint operators for all 1D interpolant types.
Each operator is **query-baked and data-free**: constructed from grid and query points only,
then applied to any sensitivity vector $\bar{\mathbf{y}}$.

| Constructor | Struct | Notes |
|-------------|--------|-------|
| `constant_adjoint(x, xq; ...)` | [`ConstantAdjoint`](@ref) | Pure scatter |
| `linear_adjoint(x, xq; ...)` | [`LinearAdjoint`](@ref) | Pure scatter |
| `quadratic_adjoint(x, xq; ...)` | [`QuadraticAdjoint`](@ref) | Recurrence adjoint + scatter |
| `cubic_adjoint(x, xq; ...)` | [`CubicAdjoint`](@ref) | Transpose Thomas solve + scatter |

---

## Quick Start

```@example adjoint1d
using FastInterpolations
using LinearAlgebra: dot

x = collect(range(0, 1, 50))
xq = [0.1, 0.25, 0.5, 0.75, 0.9]
f = sin.(2π .* x)

# Build adjoint operators (grid + queries only, no data)
adj_const  = constant_adjoint(x, xq)
adj_lin    = linear_adjoint(x, xq)
adj_quad   = quadratic_adjoint(x, xq)
adj_cubic  = cubic_adjoint(x, xq)
```

```@example adjoint1d
# Apply to a sensitivity vector
ȳ = collect(1.0:length(xq))
f̄ = adj_cubic(ȳ)
```

### Dot-Product Identity

Verify $\langle W f, \bar{y} \rangle = \langle f, W^\top \bar{y} \rangle$:

```@example adjoint1d
itp = cubic_interp(x, f)
Wf = itp(xq)                  # forward: W·f
WTy = adj_cubic(ȳ)            # adjoint: Wᵀ·ȳ

@assert dot(Wf, ȳ) ≈ dot(f, WTy)
println("⟨Wf, ȳ⟩ = ", dot(Wf, ȳ))
println("⟨f, Wᵀȳ⟩ = ", dot(f, WTy))
```

---

## Constructor Signatures

```julia
constant_adjoint(x, xq; side=NearestSide(), extrap=NoExtrap())
linear_adjoint(x, xq; extrap=NoExtrap())
quadratic_adjoint(x, xq; bc=Left(QuadraticFit()), extrap=NoExtrap())
cubic_adjoint(x, xq; bc=CubicFit(), extrap=NoExtrap())
```

Each constructor accepts the same keyword arguments as its forward interpolation counterpart.
See [Adjoint API Reference](../api/adjoint.md) for full docstrings.

---

## Calling the Operator

All adjoint operators share a unified calling convention:

### Allocating

```julia
f̄ = adj(ȳ)                          # value adjoint
f̄ = adj(ȳ; deriv=DerivOp(1))        # 1st derivative adjoint
f̄ = adj(ȳ; deriv=DerivOp(2))        # 2nd derivative adjoint
```

### In-Place (Zero Allocation)

```julia
adj(f̄, ȳ)                           # zeros f̄, then accumulates
adj(f̄, ȳ; deriv=DerivOp(1))         # derivative adjoint, in-place
```

```@example adjoint1d
f̄_buf = zeros(length(x))
adj_cubic(f̄_buf, ȳ)
f̄_buf[1:5]
```

### Size

```@example adjoint1d
size(adj_cubic)          # (n_grid, n_query)
```

---

## Matrix Materialization

For verification and debugging, materialize the full weight matrix.
This is $O(n \times m)$ and **not** intended for production use.

```@example adjoint1d
Wᵀ = Matrix(adj_cubic)
size(Wᵀ)          # (n_grid, n_query)
```

```@example adjoint1d
@assert Wᵀ * ȳ ≈ adj_cubic(ȳ)
println("Matrix × ȳ ≈ adj(ȳ): ", isapprox(Wᵀ * ȳ, adj_cubic(ȳ)))
```

The transpose gives the forward interpolation matrix:

```@example adjoint1d
W = Matrix(adj_cubic)'
@assert W * f ≈ itp(xq)
println("W * f ≈ itp(xq): ", isapprox(W * f, itp(xq)))
```

A convenience method builds the forward matrix directly from an interpolant:

```@example adjoint1d
W_direct = Matrix(itp, xq)
@assert W_direct * f ≈ itp(xq)
println("Matrix(itp, xq) * f ≈ itp(xq): ", isapprox(W_direct * f, itp(xq)))
```

!!! tip "Affine BCs"
    For BCs with non-zero values (e.g., `Deriv1(0.5)`), the forward operator is **affine**:
    $y = W f + c$. The adjoint computes $W^\top \bar{y}$ — the constant $c$ does not appear.

---

## Performance

| Method | Construction | Apply (per call) |
|--------|-------------|------------------|
| **Constant / Linear** | $O(m \log n)$ — anchor search | $O(m)$ — pure scatter |
| **Quadratic** | $O(n + m \log n)$ — anchor + slope setup | $O(n + m)$ — recurrence adjoint + scatter |
| **Cubic** | $O(n + m \log n)$ — Thomas factorization + anchor search | $O(n + m)$ — transpose solve + scatter |

- **In-place**: Zero allocation after construction (all types)
- **Cache sharing**: Cubic and quadratic reuse the same spline caches as forward interpolation

---

## See Also

- **[Adjoint Overview](overview.md)**: Concepts and mathematical background
- **[Adjoint ND](adjoint_nd.md)**: N-dimensional adjoint operators
- **[Adjoint via AD](../guides/adjoint_ad.md)**: Using AD backends for `∂f/∂y`
- **[Adjoint API Reference](../api/adjoint.md)**: Full docstrings for all adjoint types
