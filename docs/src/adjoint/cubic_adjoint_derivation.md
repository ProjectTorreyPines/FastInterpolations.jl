# Cubic Adjoint: Mathematical Derivation

!!! note "Internal Documentation"
    This page documents the mathematical derivation of the cubic adjoint operator.
    For the user-facing API, see [Adjoint 1D](adjoint_1d.md).

## Forward Operator Decomposition

For cubic splines, the forward interpolation at query point $x_q$ in interval $[x_i, x_{i+1}]$ is:

```math
y(x_q) = (1-t) f_i + t \, f_{i+1} + t(1-t)\bigl[(1-t) \, h_i^2 z_i / 6 + t \, h_i^2 z_{i+1} / 6\bigr]
```

where $t = (x_q - x_i) / h_i$ and the moments $\mathbf{z}$ satisfy the tridiagonal system
$A \mathbf{z} = R \mathbf{f} + \mathbf{b}_\text{bc}$ ($\mathbf{b}_\text{bc}$ contains BC-value terms).

In matrix form, the full forward operator decomposes as:

```math
\mathbf{y} = \underbrace{(E_y + E_z \, A^{-1} R)}_{W} \, \mathbf{f} + \underbrace{E_z \, A^{-1} \mathbf{b}_\text{bc}}_{\mathbf{c}}
```

- $E_y$: evaluation weights for y-values (sparse, 2 entries per row)
- $E_z$: evaluation weights for z-moments (sparse, 2 entries per row)
- $A$: tridiagonal moment matrix (from the spline equations)
- $R$: finite-difference RHS operator (maps $\mathbf{f}$ to the RHS of the moment system)
- $\mathbf{b}_\text{bc}$: boundary condition value contributions (zero when BC values are zero)

## Adjoint Algorithm

The **adjoint** reverses the linear part of this pipeline:

```math
W^\top = E_y^\top + R^\top \, A^{-\top} \, E_z^\top
```

This translates to a three-step algorithm:

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | $E_y^\top \bar{\mathbf{y}} \to \bar{\mathbf{f}}, \quad E_z^\top \bar{\mathbf{y}} \to \bar{\mathbf{z}}$ | **Scatter**: distribute query-space sensitivities to grid f-values and z-moments |
| 2 | $A^{-\top} \bar{\mathbf{z}} \to \bar{\mathbf{r}}$ | **Transpose solve**: back-substitute through the Thomas factorization |
| 3 | $\bar{\mathbf{f}} \leftarrow \bar{\mathbf{f}} + R^\top \bar{\mathbf{r}}$ | **RHS adjoint**: accumulate finite-difference contributions |

Each step is $O(n)$ — the transpose Thomas solve reuses the same LU factorization from the
forward interpolation, and the scatter/RHS steps are sparse operations.

!!! note "Derivative Adjoint"
    The same decomposition applies to the *derivative* operators $W_d$ ($d = 1, 2, 3$).
    Only the evaluation weights $E_y, E_z$ change — the $A^{-\top}$ and $R^\top$ steps
    are identical. The `deriv` keyword selects which set of weights to use.
