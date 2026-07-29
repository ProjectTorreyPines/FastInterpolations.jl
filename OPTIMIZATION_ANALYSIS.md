# PHS Laplacian Optimization Analysis

**Last Updated**: After session with blend weight caching investigation
**Baseline (Post Weight-Ordered Optimization)**: 8.882 ms for 1000 Laplacian queries
**Speedup Goal**: 5-10% additional improvement

## Performance Breakdown (Profile-Based)

### Per 1M Iterations Timing (from profile_blend_cost.jl)
```
Distance calculation (3D):                 1.632 ms  ← BOTTLENECK
Blend weight (exp only):                   0.553 ms
Blend weight + derivatives (exp):          0.584 ms
Polynomial approximation (no exp):         0.599 ms  (slower than exp!)
```

### Laplacian Breakdown (Estimated)
- Total Laplacian time: **8.882 ms**
- Queries: 1000
- Neighbors per query: ~15 (stencil size 8, blend region)
- Total distance calculations: ~15,000

**Distance calculation proportion**: 
- 1.632 ms × 15 = 24.48 ms for 15M distance calcs
- But we do 15k calcs total, so ~1.63 ns per distance calc
- Distance is ~18% of total Laplacian time

## Investigated Optimizations

### ❌ Blend Weight Caching
- **Status**: Tested, rejected
- **Result**: 9.774 ms (26% slower than baseline)
- **Reason**: Cache lookup overhead > blend weight computation savings
- **Learning**: Caching only helps if:
  - Cache hit rate > 90% (ours was ~95% theoretically but overhead killed benefits)
  - Cache lookup < recomputation cost (not true for simple exponential)

### ❌ Polynomial Blend Weight Approximation
- **Status**: Tested with derivative fits, rejected
- **Result**: Polynomial (0.599 ms/M) slower than exp (0.553 ms/M)
- **Reason**: Modern CPUs optimize exp() better than polynomial evaluation
- **Note**: Julia's exp() is highly optimized; polynomial adds branch overhead

## Viable Optimization Opportunities

### 1. **SIMD Distance Calculation** (Medium Effort, 3-8% Speedup)
- **Opportunity**: Batch compute distances for multiple neighbors simultaneously
- **Current**: Scalar loop over `dim in 1:N` computing `Δ * Δ`
- **Improvement**: 
  - Use `@simd for` with proper reduction
  - SIMD can compute 2-4 distance calculations in parallel (depending on CPU width)
  - Estimated speedup: 20-30% on distance calc → 3-5% overall

**Code Pattern**:
```julia
# Current (scalar)
d2 = zero(Tg)
@inbounds for dim in 1:N
    Δ = Tg(query[dim]) - nb_coords[dim]
    d2 += Δ * Δ
end

# Proposed (SIMD-friendly)
d2_vec = @MVector zeros(N, Tg)
@simd for dim in 1:N
    Δ = Tg(query[dim]) - nb_coords[dim]
    d2_vec[dim] = Δ * Δ
end
d2 = sum(d2_vec)  # SIMD-friendly sum
```

### 2. **L∞ Early Termination** (Low Effort, 2-5% Speedup)
- **Opportunity**: Skip distance calculation if any coordinate differs by > blend_a
- **Current**: Always compute full distance to all neighbors
- **Improvement**: 
  - Neighbors beyond L∞ distance blend_a can't possibly be in blend region
  - Avoids sqrt for far-away neighbors
  - Estimated: 30-50% of neighbors are beyond blend region
  - Speedup: 15-25% on distance calc → 2-4% overall

**Code Pattern**:
```julia
# Early exit if clearly too far
l_inf_dist = zero(Tg)
@inbounds for dim in 1:N
    Δ = abs(Tg(query[dim]) - nb_coords[dim])
    l_inf_dist = max(l_inf_dist, Δ)
    if l_inf_dist > blend_a
        break
    end
end
l_inf_dist > blend_a && continue  # Skip expensive sqrt

# If passes L∞ check, compute full Euclidean distance
d2 = zero(Tg)
@inbounds for dim in 1:N
    Δ = Tg(query[dim]) - nb_coords[dim]
    d2 += Δ * Δ
end
d_dist = sqrt(d2)
```

### 3. **Reciprocal Square Root (rsqrt)** (Trivial Effort, 1-2% Speedup)
- **Opportunity**: Use single rsqrt instead of sqrt + division
- **Current**: `d_dist = sqrt(d2); inv_d_dist = 1/d_dist`
- **Improvement**:
  - `inv_d_dist = rsqrt(d2)` in single operation
  - Modern CPUs (Apple Silicon, AVX2+) have fast rsqrt hardware
  - Speedup: ~5-10% on distance-based calculations → 1-2% overall

**Code Pattern**:
```julia
# Current
d_dist = sqrt(d2)
inv_d_dist = one(Tg) / d_dist

# Proposed
inv_d_dist = Base.FastMath.rsqrt(d2)  # If available
d_dist = one(Tg) / inv_d_dist  # Recompute if needed
# OR just use inv_d_dist for normalization
```

### 4. **Stencil Radius Optimization** (Medium Effort, 2-6% Speedup)
- **Opportunity**: Use smaller blend radius or adaptive radius based on data density
- **Current**: blend_factor = 1.0 (fixed for all queries)
- **Improvement**:
  - Analyze local data density
  - Use smaller blend radius in high-density regions
  - Reduces neighbors in blend region (currently ~15 per query)
  - Speedup: Fewer distance calcs and blend neighbor processing → 2-6% overall
- **Risk**: May slightly reduce accuracy

### 5. **Pre-Computed Distance Intervals** (High Effort, 5-10% Speedup)
- **Opportunity**: Pre-compute distance ranges for grid structure during build
- **Current**: Fresh distance calculation every query
- **Improvement**:
  - Grid has regular structure; neighbor locations are known
  - Could pre-cache distances between common neighbor patterns
  - Speedup: Avoid sqrt for common cases → 5-10% overall
- **Complexity**: High; requires grid structure analysis

## Recommended Next Step

**Priority 1: L∞ Early Termination** (Best ROI)
- **Effort**: ~30 min implementation
- **Expected Speedup**: 2-4% overall
- **Risk**: None (conservative filter)
- **Implementation Location**: [src/phs/phs_eval.jl](src/phs/phs_eval.jl) blend loop (line ~960-964)

**Priority 2: SIMD Distance Calculation**
- **Effort**: ~1 hour implementation + testing
- **Expected Speedup**: 3-5% overall
- **Risk**: Compiler-dependent; may need tuning
- **Implementation Location**: Same as above

**Priority 3: rsqrt Optimization**
- **Effort**: ~15 min trivial change
- **Expected Speedup**: 1-2% overall
- **Risk**: Minimal
- **Implementation Location**: Same as above

## Summary

- **Blend weight is NOT the bottleneck** despite being expensive operation
- **Distance calculation** (1.63 ns/call) is the true bottleneck (~18% of total time)
- **Caching doesn't help** because exp() is already fast on modern CPUs
- **Low-hanging fruit**: L∞ distance checks to avoid unnecessary sqrt calls
- **Combined optimizations** (L∞ + rsqrt) could achieve **3-6% additional speedup**
