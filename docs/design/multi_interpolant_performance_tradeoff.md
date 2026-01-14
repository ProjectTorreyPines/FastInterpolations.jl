# Multi-Interpolant Performance Trade-off Analysis

## Context

PR#14 introduced a **unified matrix storage** approach for `CubicMultiInterpolant`, replacing the previous composition-based `Vector{CubicInterpolant}` design. This enables SIMD vectorization across series but introduces overhead for small series counts.

## Benchmark Results (PR#14 CI)

| Benchmark | New (ns) | Old (ns) | Ratio | Analysis |
|-----------|----------|----------|-------|----------|
| `construct_s001_q100` | 2183 | 1960 | 1.11 | **11% regression** |
| `construct_s010_q100` | 15,878 | 20,498 | 0.77 | 23% improvement |
| `construct_s100_q100` | 143,779 | 203,014 | 0.71 | **29% improvement** |
| `eval_s001_q100` | 2041 | 1787 | 1.14 | **14% regression** |
| `eval_s010_q100` | 3842 | 3741 | 1.03 | ~same |
| `eval_s100_q100` | 20,531 | 21,954 | 0.94 | 6% improvement |

### Key Observations

1. **Small series (s001)**: 10-14% slower due to SIMD setup overhead
2. **Medium series (s010)**: Construct faster, eval roughly same
3. **Large series (s100)**: Significant wins across the board

### Comparison with Single Interpolant

| API | Eval q100 (ns) | Per-query (ns) |
|-----|----------------|----------------|
| `CubicInterpolant` (single) | 492 | ~5 |
| `CubicMultiInterpolant` s001 | 2041 | ~20 |

**Single `CubicInterpolant` is 4x faster than Multi with 1 series.**

## Design Options Considered

### Option 1: Dual Implementation with Threshold Dispatch

```julia
struct CubicMultiInterpolant{T, ...}
    # Runtime dispatch based on series count
    _use_legacy::Bool
    _legacy_itps::Union{Nothing, Vector{CubicInterpolant}}
    _unified_data::Union{Nothing, Matrix}
end

function (mitp::CubicMultiInterpolant)(x)
    if mitp._use_legacy
        # Vector{Interpolant} path for small series
    else
        # Unified matrix path for large series
    end
end
```

**Pros:**
- Optimal performance for all series counts
- Automatic optimization without user intervention

**Cons:**
- Code duplication and maintenance burden
- Runtime branching overhead
- Threshold tuning complexity (what's the right cutoff?)
- Two code paths to test and maintain
- Type instability risks

### Option 2: Documentation-Only Approach (Chosen)

Keep the unified implementation and provide clear usage guidelines.

**Pros:**
- Simple, maintainable codebase
- Single code path to test
- Consistent behavior
- Clear API semantics

**Cons:**
- Users with 1-2 series see ~10-14% regression vs theoretical optimal
- Requires users to read documentation

## Decision: Option 2 (Documentation)

### Rationale

1. **Misuse Case**: Using `CubicMultiInterpolant` for 1 series is already suboptimal
   - Single `CubicInterpolant` is 4x faster
   - Multi API exists for "evaluate multiple series together" use case

2. **Absolute Impact is Small**
   - Regression is ~250ns per 100-point evaluation
   - Not meaningful for most applications

3. **Maintenance Cost**
   - Dual implementation doubles test surface
   - Threshold tuning is architecture-dependent
   - Code complexity increases significantly

4. **Consistency with Other Types**
   - `LinearMultiInterpolant` uses `Vector{LinearInterpolant}` composition
   - `QuadraticMultiInterpolant` uses `Vector{QuadraticInterpolant}` composition
   - Future unification should follow same pattern (document, don't dual-implement)

## Current Implementation Status

| Type | Implementation | Notes |
|------|----------------|-------|
| `LinearMultiInterpolant` | Vector{Interpolant} | Composition-based |
| `QuadraticMultiInterpolant` | Vector{Interpolant} | Composition-based |
| `ConstantMultiInterpolant` | Vector{Interpolant} | Composition-based |
| `CubicMultiInterpolant` | Unified Matrix + SIMD | PR#14 |

### Future Direction

When unifying Linear/Quadratic to matrix storage:
- Apply same documentation strategy
- Do NOT add dual-implementation complexity
- Accept small-series overhead as acceptable trade-off

## Usage Guidelines (for Documentation)

```markdown
## Choosing the Right API

| Series Count | Recommended API |
|-------------|-----------------|
| 1 | `cubic_interp(x, y)` - Single interpolant (fastest) |
| 2-4 | Individual interpolants or Multi (similar performance) |
| 5+ | `cubic_interp(x, [y1, y2, ...])` - SIMD benefits begin |
| 50+ | Multi interpolant strongly recommended (20-30% faster) |

### When to Use MultiInterpolant

Use `CubicMultiInterpolant` when:
- You have **5+ series** sharing the same x-grid
- You need to evaluate **all series at the same query points**
- You want **zero-allocation** batch evaluation

For 1-2 series, prefer individual `CubicInterpolant` instances.
```

## Performance Characteristics

### SIMD Overhead Breakdown

The unified approach has fixed overhead:
1. **Transpose snapshot** (lazy, amortized): First scalar query triggers transpose
2. **SIMD loop setup**: Register allocation, bounds checking
3. **Result gathering**: Extracting values from SIMD lanes

For 1 series, this overhead exceeds the computation cost.
For 10+ series, overhead is amortized across many lanes.

### Crossover Analysis

```
Series:    1     2     3     4     5    10    50   100
Overhead:  ████████████████████
Benefit:                    ░░░░░░░░████████████████████

Crossover point: ~4-5 series (architecture-dependent)
```

## Conclusion

The 10-14% regression for single-series `CubicMultiInterpolant` is accepted because:

1. It's a **misuse case** - single `CubicInterpolant` is the correct API
2. Absolute impact is **~250ns** - negligible for real applications
3. Dual implementation would **double maintenance cost**
4. Clear documentation provides **better user experience** than hidden complexity

**Action Items:**
- [ ] Add performance guidelines to main documentation
- [ ] Add docstring note about series count recommendations
- [ ] Consider benchmark CI to track regression thresholds
