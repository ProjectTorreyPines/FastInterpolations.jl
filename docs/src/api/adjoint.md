# Adjoint API

## Overview

| Function | Description |
|----------|-------------|
| `cubic_adjoint(x, xq)` | 1D adjoint operator $W^\top$ |
| `cubic_adjoint(grids, queries)` | ND adjoint operator $W^\top$ |
| `linear_adjoint(x, xq)` | 1D linear adjoint operator $W^\top$ |
| `linear_adjoint(grids, queries)` | ND linear adjoint operator $W^\top$ |
| `adj(ȳ)` | Allocating apply: $\bar{f} = W^\top \bar{y}$ |
| `adj(f̄, ȳ)` | In-place apply (zero allocation) |
| `adj(ȳ; deriv=DerivOp(1))` | Derivative adjoint |
| `Matrix(adj)` | Materialize $W^\top$ as dense matrix |
| `Matrix(itp, xq)` | Materialize forward $W$ from interpolant (1D) |

---

## Constructor

```@docs
cubic_adjoint
```

## Adjoint Types

```@docs
AbstractAdjoint
CubicAdjoint
CubicAdjointND
LinearAdjoint
LinearAdjointND
```

## Linear Adjoint Constructor

```@docs
linear_adjoint
```

## Matrix Materialization

Materialize the full weight matrix for verification and debugging.
This is $O(n \times m)$ and **not** intended for production use.

**1D Adjoint:**
```julia
Wᵀ = Matrix(adj)                         # (n_grid, n_query)
Wᵀ = Matrix(adj; deriv=DerivOp(1))       # derivative adjoint matrix
```

**ND Adjoint:**
```julia
Wᵀ = Matrix(adj)                         # (prod(grid_sizes), n_query)
Wᵀ = Matrix(adj; deriv=(DerivOp(1), EvalValue()))  # mixed partial
```

**Forward matrix from interpolant (1D convenience):**
```julia
W = Matrix(itp, xq)                      # (n_query, n_grid)
W = Matrix(itp, xq; deriv=DerivOp(1))    # derivative forward matrix
```
