# [Search & Hints Overview](@id search_hints)

Search policies control how interpolants find the correct grid interval for a query point. Choosing the right policy can significantly improve performance for specific access patterns.

!!! note "This page applies to Vector grids"
    For **uniform grids**, use `Range` instead—lookup is always **O(1)** regardless of search policy. See [Grid Selection](../advanced_overview.md#Grid-Selection:-Range-vs-Vector) for details. This page focuses on **Vector grids** where choosing the right search policy matters.

## How It Works

Every interpolation query on a `Vector` grid requires finding which interval contains the query point. By default, this uses binary search (`O(log n)`), but for sequential or streaming queries, **hinted search** can achieve `O(1)` amortized lookup.

## Quick Examples

```@example
using FastInterpolations

x = collect(range(0, 1.0, length=1000))
y = x.^3

# For random queries
xq = rand(1000)
linear_interp(x, y, xq; search=Binary())         # Default: optimal for random access
linear_interp(x, y, xq; search=LinearBinary())  # Could be slower

# For sorted/monotonic queries
xq_sorted = sort(xq)
linear_interp(x, y, xq_sorted; search=Binary())         # Same as random
linear_interp(x, y, xq_sorted; search=LinearBinary())  # Much faster!
nothing # hide
```

## Decision Matrix

| Policy | Best For | Complexity | Thread Safety |
|:-------|:---------|:-----------|:--------------|
| [`Binary()`](@ref search_policies) | Random access (default) | O(log n) | ✓ Stateless |
| [`HintedBinary()`](@ref search_policies) | Repeated queries in same region | O(1) hit, O(log n) miss | ✓ With hint |
| [`LinearBinary()`](@ref search_policies) | **Monotonic queries (recommended)** | O(1) local, O(log n) fallback | ✓ With hint |
| [`Linear()`](@ref search_policies) | Close + monotonic queries (expert) | O(1) amortized | ✓ With hint |

## Quick Selection Guide

**Which policy should I use?**

- **Random access / general use** → `Binary()` (default, most consistent performance)
- **Queries cluster in same region** → `HintedBinary()`
- **Monotonic queries (sorted, streaming, ODE)** → `LinearBinary()` ✅ **recommended**
- **Strictly monotonic, performance-critical** → `Linear()` (benchmark first! see below)

!!! note "Benchmark Before Choosing Linear()"
    `Linear()` can be faster for some monotonic patterns but slower for others. Always benchmark with your actual query patterns using `benchmark/search_policy_benchmark.jl`.

## Performance Comparison

Benchmark results on a 500-point grid with 1000 queries:

```@example search
using FastInterpolations  # hide
using BenchmarkTools  # hide
using Random  # hide

n_grid, n_queries = 500, 1000
x = collect(range(0.0, 1.0, length=n_grid))
y = sin.(2π .* x)

# Create interpolants with different search policies
itp_binary = linear_interp(x, y; search=Binary())
itp_lb = linear_interp(x, y; search=LinearBinary(linear_window=4))
itp_linear = linear_interp(x, y; search=Linear())

Random.seed!(42)  # hide
nothing  # hide
```

### Random Queries (Unsorted)

```@example search
queries_random = rand(n_queries)

# Binary is optimal - no locality to exploit
@btime for q in $queries_random; $itp_binary(q); end
# LinearBinary/Linear waste time on failed linear probes
@btime (hint = Ref(1); for q in $queries_random; $itp_lb(q; hint=hint); end)
nothing  # hide
```

| Policy | Time | Relative |
|:-------|-----:|:---------|
| `Binary()` | ~10 μs | 1.0x |
| `LinearBinary(4)` | ~27 μs | 2.5x slower |
| `Linear()` | ~77 μs | **7x slower** |

**→ Winner: `Binary()`** — Random access has no locality to exploit.

### Sorted Queries (Monotonic)

```@example search
queries_sorted = sort(rand(n_queries))

@btime for q in $queries_sorted; $itp_binary(q); end
@btime (hint = Ref(1); for q in $queries_sorted; $itp_lb(q; hint=hint); end)
nothing  # hide
```

| Policy | Time | Relative |
|:-------|-----:|:---------|
| `Binary()` | ~10 μs | 1.0x |
| `LinearBinary(4)` | ~2 μs | **5x faster** |
| `Linear()` | ~2 μs | **5x faster** |

**→ Winner: `LinearBinary()`** — Exploits locality with fallback safety.

### Dense Monotonic (ODE-style)

```@example search
# 1000 queries in narrow range [0.3, 0.35]
queries_dense = collect(range(0.3, 0.35, length=n_queries))

@btime for q in $queries_dense; $itp_binary(q); end
@btime (hint = Ref(1); for q in $queries_dense; $itp_lb(q; hint=hint); end)
nothing  # hide
```

| Policy | Time | Relative |
|:-------|-----:|:---------|
| `Binary()` | ~10 μs | 1.0x |
| `LinearBinary(4)` | ~2 μs | **6x faster** |
| `Linear()` | ~2 μs | **6x faster** |

**→ Winner: `LinearBinary()` or `Linear()`** — Both achieve O(1) for dense sequential access.

!!! details "📋 Full Benchmark Code (복사용)"
    ```julia
    using FastInterpolations
    using BenchmarkTools
    using Random

    # Setup: 500-point grid, 1000 queries
    n_grid, n_queries = 500, 1000
    x = collect(range(0.0, 1.0, length=n_grid))
    y = sin.(2π .* x)

    # Create interpolants with different search policies
    itp_binary = linear_interp(x, y; search=Binary())
    itp_lb = linear_interp(x, y; search=LinearBinary(linear_window=4))
    itp_linear = linear_interp(x, y; search=Linear())

    Random.seed!(42)

    # Test 1: Random queries
    queries_random = rand(n_queries)
    @btime for q in $queries_random; $itp_binary(q); end
    @btime (hint = Ref(1); for q in $queries_random; $itp_lb(q; hint=hint); end)
    @btime (hint = Ref(1); for q in $queries_random; $itp_linear(q; hint=hint); end)

    # Test 2: Sorted queries
    queries_sorted = sort(rand(n_queries))
    @btime for q in $queries_sorted; $itp_binary(q); end
    @btime (hint = Ref(1); for q in $queries_sorted; $itp_lb(q; hint=hint); end)
    @btime (hint = Ref(1); for q in $queries_sorted; $itp_linear(q; hint=hint); end)

    # Test 3: Dense monotonic (ODE-style)
    queries_dense = collect(range(0.3, 0.35, length=n_queries))
    @btime for q in $queries_dense; $itp_binary(q); end
    @btime (hint = Ref(1); for q in $queries_dense; $itp_lb(q; hint=hint); end)
    @btime (hint = Ref(1); for q in $queries_dense; $itp_linear(q; hint=hint); end)
    ```

## Next Steps

- [Search Policies](@ref search_policies) — Detailed explanation of each policy
- [Using Hints](@ref using_hints) — Hint patterns, thread safety, and examples
