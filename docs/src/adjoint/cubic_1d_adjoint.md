# Cubic Adjoint (1D)

The [`CubicAdjoint`](@ref) operator computes $\bar{\mathbf{f}} = W^\top \bar{\mathbf{y}}$
for 1D cubic spline interpolation — mapping query-space sensitivities back to grid-space.

It is **query-baked and data-free**: constructed from grid `x` and query points `xq` only.
The same operator can be applied to any $\bar{\mathbf{y}}$ vector.

---

## Quick Start

```@example adjoint1d
using FastInterpolations
using LinearAlgebra: dot

x = collect(range(0, 1, 50))
xq = [0.1, 0.25, 0.5, 0.75, 0.9]
f = sin.(2π .* x)

# Build the adjoint operator (grid + queries only, no data)
adj = cubic_adjoint(x, xq)
```

```@example adjoint1d
# Apply to a sensitivity vector
ȳ = ones(length(xq))
f̄ = adj(ȳ)
```

Verify correctness via the **dot-product identity** $\langle W f, \bar{y} \rangle = \langle f, W^\top \bar{y} \rangle$:

```@example adjoint1d
itp = cubic_interp(x, f)
Wf = itp(xq)                  # forward: W·f
WTy = adj(ȳ)                   # adjoint: Wᵀ·ȳ

@assert dot(Wf, ȳ) ≈ dot(f, WTy)
println("⟨Wf, ȳ⟩ = ", dot(Wf, ȳ))
println("⟨f, Wᵀȳ⟩ = ", dot(f, WTy))
```

---

## Constructor

```@docs
cubic_adjoint
```

**Signature:**
```julia
cubic_adjoint(x, x_query; bc=CubicFit()) -> CubicAdjoint
```

- `x`: Grid points (sorted `AbstractVector`)
- `x_query`: Query points (baked into the operator)
- `bc`: Any [boundary condition](../boundary-conditions/overview.md) — all BC types are supported including `PeriodicBC`

---

## Calling the Operator

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
adj(f̄_buf, ȳ)
f̄_buf[1:5]
```

### Size

```@example adjoint1d
size(adj)          # (n_grid, n_query)
```

---

## Matrix Materialization

For verification and debugging, you can materialize the full weight matrix.
This is $O(n \times m)$ and **not** intended for production use.

```@example adjoint1d
Wᵀ = Matrix(adj)
size(Wᵀ)          # (n_grid, n_query)
```

```@example adjoint1d
@assert Wᵀ * ȳ ≈ adj(ȳ)
println("Matrix × ȳ ≈ adj(ȳ): ", isapprox(Wᵀ * ȳ, adj(ȳ)))
```

The transpose gives the forward interpolation matrix:

```@example adjoint1d
W = Matrix(adj)'
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

- **Construction**: $O(n + m \log n)$ — Thomas factorization + anchor binary search (amortized via `autocache`)
- **Apply (per call)**: $O(n + m)$ — one transpose solve + one scatter pass
- **In-place**: **Zero allocation** after construction
- **Cache sharing**: Reuses the same `CubicSplineCache` as forward interpolation

---

## See Also

- **[Adjoint Overview](overview.md)**: Concepts and mathematical background
- **[Adjoint via AD](../guides/adjoint_ad.md)**: Using AD backends for `∂f/∂y`
- **[Cubic Interpolation](../interpolation/cubic.md)**: Forward cubic spline API
- **[AD Support](../guides/autodiff_support.md)**: General AD support for `∂f/∂x`
