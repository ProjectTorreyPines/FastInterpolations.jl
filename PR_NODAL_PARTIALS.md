# PR: `nodal_partials` — Public API for Precomputed Partial Derivatives

## Summary

Add `nodal_partials(itp, order)` that provides zero-copy, type-stable access to precomputed partial derivatives at grid nodes. Replaces brittle internal field access (`itp.nodal_derivs.partials[2, i, j]`) with a semantic tuple-based API (`nodal_partials(itp, (1, 0))`).

Also renames internal storage types `NodalDerivativesND` -> `_NodalDerivativesND` and `HeteroPartials` -> `_HeteroPartials` to clarify they are not public API.

## API

```julia
# CubicInterpolantND / QuadraticInterpolantND (bit-encoded, 2^N partials)
nodal_partials(itp, (0, 0))   # f(xᵢ, yⱼ)
nodal_partials(itp, (1, 0))   # ∂f/∂x at nodes
nodal_partials(itp, (0, 1))   # ∂f/∂y at nodes
nodal_partials(itp, (1, 1))   # ∂²f/∂x∂y at nodes

# HeteroInterpolantND with PreCompute() (compact mixed-radix)
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()), coeffs=PreCompute())
nodal_partials(itp, (1, 0))   # OK — cubic axis has derivatives
nodal_partials(itp, (0, 1))   # ERROR — "axis 2 uses LinearInterp() which does not store..."
```

Returns `@view` into internal storage — zero allocation, zero copy.

## Validation & Error Messages

| Case | Error type | Message |
|------|-----------|---------|
| Wrong tuple length | `DimensionMismatch` | "order tuple has M elements but interpolant is N-dimensional" |
| Invalid order value | `ArgumentError` | "derivative order at axis d must be 0 or 1, got ..." |
| Non-derivative axis (Hetero) | `ArgumentError` | "axis d uses LinearInterp() which does not store nodal derivatives" |
| `LinearInterpolantND` | `ArgumentError` | "does not store nodal partial derivatives" |
| `ConstantInterpolantND` | `ArgumentError` | "does not store nodal partial derivatives" |
| `OnTheFly()` strategy | `ArgumentError` | "requires PreCompute() strategy" |

## Changes

- **NEW** `src/nodal_partials.jl` — dispatch methods, validation, index translation
- **NEW** `test/test_nodal_partials.jl` — 80 tests (structural, value-correctness, errors, `@inferred`)
- **Rename** `NodalDerivativesND` -> `_NodalDerivativesND` (15 src files)
- **Rename** `HeteroPartials` -> `_HeteroPartials` (6 src files)
- **Modified** `src/FastInterpolations.jl` — include + `export nodal_partials`
- **Modified** `test/runtests.jl` — wire test file

## Test Coverage

- Cubic 2D/3D: all `2^N` derivative combos, zero-copy verification, `@inferred`
- Quadratic 2D: all 4 combos
- Hetero PreCompute 2D: Cubic x Linear — valid/invalid axis checks
- Hetero PreCompute 2D: Cubic x Quadratic — analytic derivative comparison
- Hetero PreCompute 3D: Cubic x Quadratic x Linear — 3D value correctness
- Error cases: DimensionMismatch, invalid order, unsupported types, OnTheFly
- Value correctness: polynomial test functions (exact reproduction by spline)
