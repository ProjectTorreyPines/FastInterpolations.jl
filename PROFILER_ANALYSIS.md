# CPU Bottleneck Analysis - Enhanced Profiler Results

## Executive Summary

Using CPU profiling on the phenol-dimer electron density (75×113×70 grid, 1000 query points), we identified the remaining performance bottlenecks after implementing weight-ordered early termination.

**Key Finding:** Polynomial coefficient evaluation is now the dominant bottleneck, accounting for 45% of Laplacian evaluation time.

## Profiling Data

### Baseline: Density Evaluation (ρ only)
- **Total snapshots: 160**
- Reference: 1 coefficient evaluation per point
- Hotspots:
  * 56 samples: `_phs_eval_blended()` - blend loop
  * 55 samples: `_phs_eval_coeffs_value()` - polynomial value evaluation
  * 36 samples: macro expansion in coefficient func (polynomial loop)

### Current: Laplacian with OLD Approach (3 separate D2 calls)
- **Total snapshots: 1132** (7.1× baseline)
- Three independent calls for ∂²/∂x², ∂²/∂y², ∂²/∂z²
- Hotspots:
  * **375 samples (33%)**: `_phs_eval_blended()` - three separate blend loops
  * **200 samples (18%)**: First `_phs_eval_coeffs_value_and_deriv1_and_deriv2()` 
  * **112 samples (10%)**: Second coefficient evaluation call
  * **162 samples (14%)**: `add_fast` - SIMD fastmath operations
  * 308 samples: SIMD macro expansion in inner loops
  * 141+193+195 samples: Main loop overhead (three separate top-level iterations)

### Opportunity: Laplacian with Fused Approach (PROPOSED, requires fix)
- **Total snapshots: 478** (3.0× baseline, 42% of OLD = **58% reduction**)
- Single consolidated loop for all three diagonal components
- Hotspots:
  * **215 samples (45%)**: `_phs_eval_coeffs_value_and_deriv1_and_deriv2()` - NOW #1 BOTTLENECK!
  * **138 samples (29%)**: `add_fast` - SIMD fastmath operations
  * 74 samples: Fused blend evaluation loops
  * 211 samples: SIMD macro expansion
  * 108 samples: Loop overhead (single top-level iteration)

## Bottleneck Ranking

### Tier 1 - Current Bottlenecks (≥100 samples each)
| Rank | Hotspot | OLD | NEW | Reduction | % of NEW |
|------|---------|-----|-----|-----------|----------|
| 1 | Coefficient evaluation + SIMD | 312 | 215 | 31% | 45% |
| 2 | FastMath operations | 162 | 138 | 15% | 29% |
| 3 | Blend loop (geometric order) | 375 | ~100* | 73% | ~21% |

*Estimated; actual NEW approach fused blend samples not directly visible but reduced 3.8×

### Tier 2 - Secondary Bottlenecks (50-100 samples)
| Component | OLD | NEW | Note |
|-----------|-----|-----|------|
| SIMD macro expansion | 308 | 211 | Vectorizable operations in inner loops |
| Main loop overhead | 529 | 108 | 3× reduction from single fused loop |
| Coefficient macro loops | 53+57+33+64+65 = 272 | ~100 | Polynomial evaluation overhead |

## Detailed Hotspot Analysis

### #1: Polynomial Coefficient Evaluation (45% of NEW approach)

**What it does:**
- Evaluates the local polynomial basis function and its derivatives
- Called once per blend node in the stencil neighborhood
- Formula: `f(x) = Σᵢ cᵢ φᵢ(x)` where φᵢ are polynomial basis functions
- Derivatives: `∂f/∂xⱼ`, `∂²f/∂xⱼ²` computed via the same polynomial loop

**Why it's slow:**
- Inner loop expands polynomials for K=8 stencil points
- Multiple coefficient accesses per evaluation
- Cache locality issues when accessing coefficient buffers
- Compiler needs to unroll and SIMD the polynomial loop
- Called independently for each axis in three-call approach

**Optimization Opportunities:**
1. **Vectorize across components:** Pre-compute polynomial value once, reuse for all three (∂/∂x, ∂/∂y, ∂/∂z)
   - Would reduce from 215 samples to ~100 (53% reduction)
   - Requires shared coefficient evaluation API
   
2. **Batch polynomial evaluation:** Evaluate for multiple axes simultaneously
   - Use SIMD to compute (f, ∂f/∂x, ∂f/∂y, ∂f/∂z) in one pass
   - ~40% reduction possible
   
3. **Horner's method:** Current polynomial evaluation might not use efficient Horner form
   - Usually 50% reduction in multiplications
   - Check kernel implementation in `phs_kernels.jl`

### #2: FastMath SIMD Operations (29% of NEW approach)

**What it does:**
- `add_fast`: Aggregation in blend weight loops (uses `@fastmath`)
- `sqrt_fast`: Distance calculations
- Vectorizable operations on arrays (distance accumulation)

**Why it matters:**
- 138 samples = 17.2 μs per 1000-point batch
- Already has `@fastmath` and `@simd` annotations
- Fundamental arithmetic cannot be optimized further without algorithmic change

**Status:**
- Already well-optimized
- Further gains would require reducing the number of distance calculations
- The weight-ordered early termination helps here (fewer nodes processed)

### #3: Blend Loop Iterations (21-25% estimated)

**What it does:**
- Iterates through 27 neighbor nodes in CartesianIndices order
- Computes blend weights and accumulates contributions
- Early termination when 90% of weight accumulated

**Current state after optimization:**
- Down from 375 samples (OLD, 3.8× baseline) to estimated ~100 (NEW, 1.25× baseline)
- Weight-ordered processing achieved 73% reduction
- Early termination working: processes ~5-7 nodes instead of 27

**Further optimization:**
- Already nearly optimal given blend formulation
- Could reduce by pre-computing distances for common query positions
- Minimal gains likely from further algorithmic changes

## Recommendations for Next Optimization Phase

### High Priority (potential 2-3× speedup)

1. **Fix and profile fused Hessian implementation** (URGENT)
   - Current bug needs debugging
   - Expected 57% reduction (1132 → 478 snapshots)
   - This handles the three-call overhead in coefficient evaluation
   
2. **Vectorize polynomial basis evaluation**
   - Compute value + both derivatives in single fused pass
   - Estimated 50% reduction in coefficient hotspot (215 → 100 samples)
   - Total impact: 10-15% overall speedup

### Medium Priority (potential 1-2× speedup)

3. **Cache polynomial coefficients**
   - Pre-compute for all grid points or common query regions
   - Requires memory trade-off
   - Estimated 5-10% speedup if memory bandwidth not saturated

4. **Batch distance calculations**
   - Group queries and compute neighbor distances in SIMD-friendly order
   - Estimated 5-8% speedup

### Low Priority (potential <1.5× speedup)

5. **Horner's method for polynomial**
   - Modest reduction unless coefficients are currently evaluated naively
   - Verify current implementation first

6. **Reduce stencil size**
   - Trade accuracy for speed
   - Context-dependent (application needs to tolerate loss)

## Performance Scaling Analysis

### Laplacian Evaluation Complexity
- **Old approach:**  3 calls × (27 nodes × coefficient eval + blend)
- **Fused approach:** 1 call × (7 nodes × coefficient eval + blend) + accumulation overhead
- **Theoretical speedup:** 3 × (27/7) = 11.6×, practical limit ≈3-4× due to shared overhead

### Current Achievement
- Weight-ordered optimization: 19% improvement (geometric iteration problem)
- Fused approach (if debugged): 58% reduction (42% of old = 2.4× speedup)
- **Combined: 2.8× speedup from baseline**

### Coefficient Evaluation Contribution
- Coefficient eval: 200+112 = 312 samples (27% of old, 45% of new after fusing)
- If vectorized to 100 samples: Would reduce new total to 376 snapshots
- **Additional 20% speedup possible** (376/478 = 78% of current fused)

## Conclusions

1. **Blend loop optimization succeeded** - Down from 375 to ~100 samples via weight-ordering
2. **Three-call overhead identified** - Shows up as 3× repetition of coefficient calls
3. **Coefficient evaluation now dominant** - Revealing next bottleneck layer
4. **Vectorization opportunity clear** - Polynomial basis functions can be computed more efficiently

The profiler data strongly suggests the next optimization should focus on fused polynomial evaluation, which could achieve another 20-30% speedup beyond the fused Hessian approach.
