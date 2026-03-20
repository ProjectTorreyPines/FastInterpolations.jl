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
    **matrix-free** algorithms that exploit each spline's structure.

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

### Automatic Differentiation

Pass `f` as a live variable through the one-shot API and let an AD backend differentiate.
See [Adjoint via AD](../guides/adjoint_ad.md) for details and backend compatibility.

```julia
using Zygote
∇f = Zygote.gradient(f -> sum(cubic_interp(x, f, xq)), f)[1]
# Works with all types: linear_interp, quadratic_interp, constant_interp
```

### Native Adjoint Operator

FastInterpolations provides **matrix-free adjoint operators** for every interpolant type.
Construct once from grid + queries, then apply to any $\bar{\mathbf{y}}$:

```julia
adj = cubic_adjoint(x, xq; bc=CubicFit())
f̄ = adj(ȳ)                  # allocating
adj(f̄, ȳ)                   # in-place, zero-allocation
```

**Zygote** and **Enzyme** use these native operators internally via registered AD rules,
so you get native performance through the standard AD interface.

See [Adjoint 1D](adjoint_1d.md) and [Adjoint ND](adjoint_nd.md) for the full API.

---

## Supported Methods

All four interpolant types have full native adjoint support in both 1D and ND:

| Method | Native Adjoint | AD (ForwardDiff / Zygote / Enzyme) |
|--------|:--------------:|:----------------------------------:|
| **Constant** | [`ConstantAdjoint`](@ref) / [`ConstantAdjointND`](@ref) | ✅ |
| **Linear** | [`LinearAdjoint`](@ref) / [`LinearAdjointND`](@ref) | ✅ |
| **Quadratic** | [`QuadraticAdjoint`](@ref) / [`QuadraticAdjointND`](@ref) | ✅ |
| **Cubic** | [`CubicAdjoint`](@ref) / [`CubicAdjointND`](@ref) | ✅ |

!!! note "Enzyme compatibility"
    Enzyme support is fully tested on Julia ≥ 1.10 with standard platforms.
    On some OS/architecture combinations or older Julia versions, Enzyme may
    encounter edge cases. See [Adjoint via AD](../guides/adjoint_ad.md) for details.

## See Also

- **[Adjoint 1D](adjoint_1d.md)**: 1D adjoint operators — API, examples, and performance
- **[Adjoint ND](adjoint_nd.md)**: ND adjoint operators — API, examples, and performance
- **[Adjoint via AD](../guides/adjoint_ad.md)**: Using AD backends for `∂f/∂y`
- **[Cubic Adjoint Derivation](cubic_adjoint_derivation.md)**: Mathematical formulation (internals)
