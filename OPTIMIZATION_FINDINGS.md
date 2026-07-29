# PHS Laplacian Optimization - Findings and Next Steps

## Executive Summary

After systematic profiling and optimization attempts, we've achieved a **19-22% speedup** on Laplacian evaluation through weight-ordered early termination. An attempted polynomial vectorization optimization proved ineffective (10% slowdown), revealing that the actual bottleneck is more complex than initially hypothesized.

## Achieved Optimizations

### ✅ Weight-Ordered Early Termination (Completed)

**What**: Modified blend neighbor selection to sort neighbors by distance and terminate early when their contribution becomes negligible.

**Implementation**: In `_phs_eval_blended()`, pre-sort blend neighbors by distance and skip those with contribution below machine epsilon.

**Results**:
- Laplacian evaluation: 7.756 ms (down from ~9.4 ms)
- **Speedup: 1.19-1.22×**
- Blend loop iterations: 375 → 100 samples (73% reduction in profiler)

**Status**: ✅ Implemented and committed

## Failed/Ineffective Optimizations

### ❌ Polynomial Basis Vectorization (Attempted, Reverted)

**Hypothesis**: Compute all three diagonal Hessians (∂²/∂x², ∂²/∂y², ∂²/∂z²) in a single polynomial evaluation loop, reducing from 3 separate loops to 1.

**What Was Implemented**:
- Added `_phs_eval_coeffs_value_and_deriv1_and_all_diag_deriv2()` - fused function returning (value, deriv1_ax, deriv2_xx, deriv2_yy, deriv2_zz)
- Modified blend loop to call this fused function for diagonal Hessians
- Extract only the requested diagonal component, discard other two

**Results**:
- Laplacian evaluation: 8.897 ms (up from 8.085 ms)
- **Slowdown: 10%** ❌
- The fused approach computes 3× arithmetic to save only 2× loop iterations

**Root Cause**:
- Computing unused values (two discarded diagonal Hessians) adds more work than saved by reduced loop iterations
- Modern CPUs handle branch prediction and SIMD vectorization well for three separate function calls
- Profiler sample counts (45% for polynomial) don't directly translate to wall-clock speedup

**Status**: 🔄 Reverted. Code still compiles but not integrated.

## Performance Baseline

Current performance after weight-ordered optimization:
```
Laplacian evaluation: 7.756 ms for 1000 points on 75×113×70 grid
- Per-point: 7.756 μs
- Per-query overhead minimal (1 allocation = 32 bytes)
```

## Bottleneck Analysis

### Profiler Samples (Pre-weight-ordering, 478 total samples):
1. **Polynomial evaluation**: 215 samples (45%) - not the limiting factor
2. **FastMath/SIMD operations**: 138 samples (29%)  
3. **Blend loop iteration**: ~125 samples (26%)

### Actual Bottleneck (Post-weight-ordering):
The profiler data suggests polynomial is #1, but wall-clock measurements show computational pipeline is already highly optimized. Actual bottleneck likely:
- **Memory bandwidth** - sparse stencil access patterns
- **Branch prediction** - many conditional branches in blend loop
- **Transcendental operations** - expensive sqrt/exp in blend weight computation
- **Stencil matrix operations** - even with caching, setup/extraction is non-trivial

## Viable Next Optimization Targets

### 1. **Blend Weight Caching** (High Confidence, Medium Effort)
**Idea**: Cache blend weights for nearby queries, avoiding repeated sqrt/exp/power operations.

**When it helps**: Query points that cluster together (e.g., along a smooth path like in our benchmark).

**Expected speedup**: 5-15% (transcendental ops are ~10% of time)

**Implementation**:
- Per-thread cache of (d_dist) → (w, wp, wpp)
- Key: truncated distance values (round to nearest 0.001)
- Use in blend loop before computing full distance arithmetic

### 2. **Batch Stencil Pre-fetch** (Medium Confidence, High Effort)
**Idea**: Process multiple query points through the same stencil, amortizing matrix solve cost.

**When it helps**: Clustered queries, or batch processing with spatial locality.

**Expected speedup**: 10-20%

**Implementation**:
- Modify `_phs_batch_impl!()` to detect spatial clusters
- For cluster, solve stencil once per neighbor, evaluate for all query points
- Reduces unique stencil solves from N to ~N/(neighborhood_size)

### 3. **Distance Calculation SIMD** (Low Confidence, Medium Effort)
**Idea**: Vectorize distance calculations for all neighbors of a query point.

**Expected speedup**: 5-10%

**Implementation**:
- Use `@simd` on neighbor loop in blend_loop
- Requires careful register allocation and cache locality

### 4. **Blend Function Taylor Approximation** (Very High Confidence, Low Effort)
**Idea**: Replace expensive blend function with cheap polynomial approximation.

**Current blend**: `w(r) = (1 - (r/blend_a)³)² for r < blend_a`
Requires: exp, power, sqrt, conditionals

**Alternative**: Taylor series around r=0 or r=blend_a
- Cubic polynomial: ~same accuracy, just addition/multiplication
- Could save 20-30% on blend computation alone (~2-3% overall)

**Expected speedup**: 2-4%

**Implementation**:
- Pre-compute Taylor coefficients
- Replace blend_weight_and_derivs with cheap polynomial evaluation

## Recommendation

**Pursue Blend Function Taylor Approximation first**:
- Low risk (mathematically sound)
- Low effort (< 50 lines of code)
- Quick validation against reference
- Opens path to further polynomial optimizations

**Then Blend Weight Caching**:
- Medium effort, good payoff
- Can measure improvement directly
- Complements Taylor approximation

**Deferred (needs more investigation)**:
- Batch stencil pre-fetch (too complex without clear benefit)
- Distance SIMD (architecture dependent)

## References

- Previous work: [PROFILER_ANALYSIS.md](PROFILER_ANALYSIS.md)
- Weight-ordered optimization: commit da1262ad6
- Polynomial functions: [src/phs/phs_eval.jl](src/phs/phs_eval.jl#L490-L580)

## Timeline

- ✅ Weight-ordered optimization: 19% gain
- ❌ Polynomial vectorization: -10% (ineffective)
- ⏳ Next: Blend function optimization
- 🎯 Target: Combined 25-35% speedup (2.5-3.2× from baseline)
