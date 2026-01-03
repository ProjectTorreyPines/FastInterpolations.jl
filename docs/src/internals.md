# Internals & Design Documents

This section provides links to internal design documents for developers.

## Design Documents

The following documents are available directly in the GitHub repository:

- [Derivative Design](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/docs/design/derivative_design.md): Mathematical foundations and kernel implementation for derivative computation
- [Interpolant Derivative API](https://github.com/ProjectTorreyPines/FastInterpolations.jl/blob/master/docs/design/interpolant_derivative_api.md): Hybrid B+C API design decisions

## Architecture Overview

FastInterpolations.jl internal architecture:

- **Operation Types** (`src/ops.jl`): `EvalValue`, `EvalDeriv1`, `EvalDeriv2` traits for dispatch
- **Kernel Functions** (`src/*_kernels.jl`): Pure math functions for interpolation and derivatives
- **Dispatch Macros** (`src/utils.jl`): Runtime-to-compile-time conversion via `@_dispatch_deriv`
- **Boundary Conditions** (`src/bc_types.jl`): `NaturalBC`, `ClampedBC`, `PeriodicBC` types
