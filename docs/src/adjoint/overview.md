# Adjoint Operators

## What Is an Adjoint?

For fixed grid `x`, query points `xq`, and boundary conditions, spline interpolation is
an **affine operation** on data values `f`:

```math
\mathbf{y} = W \, \mathbf{f} + \mathbf{c}
```

where $W$ is the *interpolation weight matrix* of size $(m \times n)$ — $m$ query points, $n$ grid points —
and $\mathbf{c}$ is a constant offset determined by the boundary condition **values**.

!!! note "Matrix-free implementation"
    $W$ is a mathematical abstraction — FastInterpolations never forms this matrix explicitly.
    Both $W \mathbf{f}$ (forward) and $W^\top \bar{\mathbf{y}}$ (adjoint) are computed via
    **matrix-free** algorithms that exploit the spline's tridiagonal structure.

The **adjoint** (transpose) operator maps query-space sensitivities back to data-space:

```math
\bar{\mathbf{f}} = W^\top \bar{\mathbf{y}}
```

| Direction | Operation | Description |
|-----------|-----------|-------------|
| **Forward** (gather) | $\mathbf{y} = W \mathbf{f} + \mathbf{c}$ | Weighted sum of nearby data + BC offset → interpolated values |
| **Adjoint** (scatter) | $\bar{\mathbf{f}} = W^\top \bar{\mathbf{y}}$ | Distribute query-space sensitivities back to grid nodes |

!!! note "Adjoint ≠ Inverse"
    The adjoint $W^\top$ is the **Jacobian transpose**, not the inverse.
    It maps sensitivities (cotangent vectors) from query-space back to data-space,
    which is exactly what reverse-mode AD computes for the pullback $\partial L / \partial \mathbf{f}$.

## Why Use Adjoints?

The adjoint arises naturally in any workflow that needs **gradients with respect to data values** $\partial L / \partial \mathbf{f}$.

| Application | How the Adjoint Appears |
|-------------|------------------------|
| **Inverse problems** | Fit grid data $f$ by minimizing $\lVert Wf + c - y_\text{obs} \rVert^2$. Gradient: $\nabla_f L = 2 W^\top (Wf + c - y_\text{obs})$. |
| **PDE-constrained optimization** | Propagate sensitivities through interpolation steps without forming the full Jacobian. |
| **Neural network layers** | Backpropagation through a spline interpolation layer requires the adjoint. |
| **Data assimilation** | Map observation-space increments back to state-space corrections (4D-Var). |

---

## Computing the Adjoint

There are two approaches:

### Automatic Differentiation

Pass `f` as a live variable through the one-shot API and let an AD backend differentiate.
See [Adjoint via AD](../guides/adjoint_ad.md) for details and backend compatibility.

```julia
# Zygote — uses CubicAdjoint internally via rrule
using Zygote
∇f = Zygote.gradient(f -> sum(cubic_interp(x, f, xq)), f)[1]
```

### Native Adjoint Operator

FastInterpolations provides a **matrix-free adjoint operator** that computes $W^\top \bar{\mathbf{y}}$
directly in $O(n + m)$ time — one transpose Thomas solve plus one scatter pass:

```julia
adj = cubic_adjoint(x, xq; bc=CubicFit())
f̄ = adj(ȳ)                  # allocating
adj(f̄, ȳ)                   # in-place, zero-allocation
```

The native operator exploits the cubic spline structure and is the recommended approach
for performance-critical code. **Zygote** and **Enzyme** use it internally via registered AD rules,
so you get native performance through the standard AD interface.

See [Cubic Adjoint (1D)](cubic_1d_adjoint.md) and [Cubic Adjoint (ND)](cubic_nd_adjoint.md) for the full API reference.

!!! tip "Mathematical Derivation"
    For the detailed mathematical formulation of how the cubic adjoint decomposes into
    scatter, transpose-solve, and RHS-adjoint steps,
    see [Cubic Adjoint Derivation](cubic_adjoint_derivation.md) in the Internals section.

---

## Currently Supported

| Method | Native Adjoint | AD-based $\partial f / \partial y$ |
|--------|:--------------:|:------------------------------------:|
| **Cubic (1D)** | [`CubicAdjoint`](@ref) | ForwardDiff, Zygote, Enzyme |
| **Cubic (ND)** | [`CubicAdjointND`](@ref) | ForwardDiff, Zygote, Enzyme |
| **Linear** | — (planned) | ForwardDiff, Zygote |
| **Quadratic** | — (planned) | ForwardDiff |
| **Constant** | — (planned) | ForwardDiff, Zygote |

## See Also

- **[Cubic Adjoint (1D)](cubic_1d_adjoint.md)**: 1D API reference and examples
- **[Cubic Adjoint (ND)](cubic_nd_adjoint.md)**: ND API reference and examples
- **[Adjoint via AD](../guides/adjoint_ad.md)**: Using AD backends for `∂f/∂y`
- **[Cubic Adjoint Derivation](cubic_adjoint_derivation.md)**: Mathematical formulation (internals)
