# feat: Add Constant (Step) Interpolation

## Summary

Add constant (piecewise step) interpolation to FastInterpolations.jl with zero-allocation performance, full extrapolation support, and type preservation.

### Key Features

- **Three side conventions**: `:nearest` (default), `:left`, `:right` for step function behavior
- **Zero-allocation hot paths**: Type-stable implementation with no allocations for Float64/Float32 inputs
- **Full extrapolation support**: `:none`, `:constant`, `:extension`, `:wrap` modes
- **Callable interpolant**: `ConstantInterpolant` struct for repeated evaluation
- **Derivative support**: 1st and 2nd derivatives (always zero for constant functions)
- **Type preservation**: Non-Float inputs maintain their return type

## API

```julia
# Scalar evaluation
constant_interp(x, y, xi; extrap=:none, side=:nearest, deriv=0)

# Vector evaluation (in-place and allocating)
constant_interp!(output, x, y, x_targets; ...)
constant_interp(x, y, x_targets; ...)

# Callable form
itp = constant_interp(x, y; extrap=:none, side=:nearest)
itp(xi)  # evaluate
itp.(x_targets)  # broadcast
```

## Side Convention Behavior

| Side | Interval Behavior | At Grid Point |
|------|-------------------|---------------|
| `:left` | Always left value | Value at that point |
| `:right` | Always right value | Value at that point |
| `:nearest` | Closer value (left tie-break) | Value at that point |

## Changes

### New Files
- `src/constant_kernels.jl` - Pure mathematical kernel functions
- `src/constant_interp.jl` - API and `ConstantInterpolant` struct
- `test/test_constant.jl` - Comprehensive test suite (187 tests)

### Modified Files
- `src/ops.jl` - Added `SideVal` union type
- `src/utils.jl` - Added `@_dispatch_side` macro and `_to_float` identity specialization
- `src/derivative_view.jl` - Added `deriv1`/`deriv2` for `ConstantInterpolant`
- `src/FastInterpolations.jl` - Added includes and exports

## Test Coverage

- Core functionality (uniform/non-uniform grids, all side options)
- All extrapolation modes
- Derivatives (value and type preservation)
- Zero-allocation verification (ALLOC_THRESHOLD for Julia version compatibility)
- Type stability (`@inferred`)
- Edge cases (grid points, boundaries, wrap domain)
- Type conversion warnings
