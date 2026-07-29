# Polynomial Basis Vectorization Optimization

## Overview

Added a new fused polynomial evaluation function that computes polynomial value and all three diagonal second derivatives (∂²/∂x², ∂²/∂y², ∂²/∂z²) in a **single loop pass** instead of three separate loops.

## Function Signature

```julia
_phs_eval_coeffs_value_and_all_diag_deriv2(
    coeffs::AbstractVector{Tv},
    phys_offsets::Vector{<:NTuple{N, Tg}},
    query::NTuple{N, <:Real},
    base_coords::NTuple{N, Tg},
    ::Val{K}
) -> (value, deriv2_xx, deriv2_yy, deriv2_zz)
```

## Performance Impact

**Theoretical Speedup: 3×** 
- Old approach: 3 separate loops through ns stencil points (3×ns iterations)
- New approach: 1 single loop through ns points (1×ns iterations)

**Practical Application:**
- Located at [src/phs/phs_eval.jl](src/phs/phs_eval.jl#L478)
- Can reduce polynomial evaluation hotspot from 215 samples to ~70 samples (67% reduction)
- Combined with blend loop optimization, potential 30-40% additional speedup
- This would bring total cascade speedup to **2.5-3.2× from baseline**

## Implementation Details

### For K=3 (PHS Degree 3):
- Inner loop processes all stencil points once
- Computes radial basis value: ϕ(r) = r³
- Computes three diagonal Hessian components simultaneously:
  - ∂²ϕ/∂x² = 3r + 9x²/r
  - ∂²ϕ/∂y² = 3r + 9y²/r
  - ∂²ϕ/∂z² = 3r + 9z²/r
- Accumulates polynomial basis contributions separately

### For General K:
- Generic loop structure supporting arbitrary PHS degrees
- Computes radial derivatives φ'(r) and φ''(r)
- Uses Leibniz rule: ∂²ϕ/∂x² = ϕ''(r)·(x/r)² + ϕ'(r)/r·(1-(x/r)²)

### Optimizations Applied:
- `@fastmath` macro for aggressive floating-point optimizations
- `@inbounds` for bounds-checking elimination
- `@simd` for inner distance/accumulation loops
- Single accumulation pass with multiple output registers

## Integration Path

To use this optimization, the blend loop needs to be restructured to call this fused function when computing diagonal Hessians. Current structure makes three separate blend loop calls (one per axis), each with independent polynomial evaluations.

Proposed integration:
1. Detect when three diagonal Hessian calls are made on the same query points
2. Route to optimized unified blend function
3. Call `_phs_eval_coeffs_value_and_all_diag_deriv2` once per node
4. Accumulate all three components simultaneously

## Files Modified

- [src/phs/phs_eval.jl](src/phs/phs_eval.jl): Added `_phs_eval_coeffs_value_and_all_diag_deriv2` function (~90 lines)

## Testing

- ✓ Function compiles without errors
- ✓ Type signature verified
- Ready for integration into blend loop

## Next Steps

1. Create optimized blend function using fused polynomial evaluation
2. Add routing logic to dispatch three-call sequences to optimized path
3. Benchmark to measure actual speedup
4. Full test suite validation

## Performance Projections

| Optimization | Speedup | Cumulative | Status |
|--------------|---------|-----------|--------|
| Weight-ordered early termination | 1.19× | 1.19× | ✓ Implemented |
| Polynomial basis vectorization | 3×* | 2.5-3.2× | 🔄 Ready for integration |
| Cache optimization | 1.2× | 3-4× | 📋 Proposed |

*Local speedup within polynomial loop; global speedup depends on blend loop integration
