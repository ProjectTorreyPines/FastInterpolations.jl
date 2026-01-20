# [Search & Hints Overview](@id search_hints)

Search policies control how interpolants find the correct grid interval for a query point. Choosing the right policy can significantly improve performance for specific access patterns.

!!! note "This page applies to Vector grids"
    For **uniform grids**, use `Range` instead—lookup is always **O(1)** regardless of search policy. See [Grid Selection](../advanced_overview.md#Grid-Selection:-Range-vs-Vector) for details. This page focuses on **Vector grids** where choosing the right search policy matters.

## How It Works

Every interpolation query on a `Vector` grid requires finding which interval contains the query point. By default, this uses binary search (`O(log n)`), but for sequential or streaming queries, **hinted search** can achieve `O(1)` amortized lookup.

## Quick Examples

```julia
using FastInterpolations

x = collect(range(0, 1.0, length=1000))
y = x.^3

# For random queries
xq = rand(1000)
linear_interp(x, y, xq; search=Binary())         # Default: optimal for random access
linear_interp(x, y, xq; search=LinearBounded())  # Could be slower

# For sorted/monotonic queries
xq_sorted = sort(xq)
linear_interp(x, y, xq_sorted; search=Binary())         # Same as random
linear_interp(x, y, xq_sorted; search=LinearBounded())  # Much faster!
```

## Decision Matrix

| Policy | Best For | Complexity | Thread Safety |
|:-------|:---------|:-----------|:--------------|
| [`Binary()`](@ref search_policies) | Random access (default) | O(log n) | ✓ Stateless |
| [`HintedBinary()`](@ref search_policies) | Repeated queries in same region | O(1) hit, O(log n) miss | ✓ With hint |
| [`LinearBounded()`](@ref search_policies) | Sequential/streaming queries | O(1) local, O(log n) fallback | ✓ With hint |

## Quick Selection Guide

**Which policy should I use?**

- **Random access / general use** → `Binary()` (default, no setup needed)
- **Queries cluster in same region** → `HintedBinary()`
- **Sorted/monotonic queries (ODE, streaming)** → `LinearBounded()`

## Next Steps

- [Search Policies](@ref search_policies) — Detailed explanation of each policy
- [Using Hints](@ref using_hints) — Hint patterns, thread safety, and examples
