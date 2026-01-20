# [Search & Hints Overview](@id search_hints)

Search policies control how interpolants find the correct grid interval for a query point. Choosing the right policy can significantly improve performance for specific access patterns.

## How It Works

Every interpolation query requires finding which grid interval contains the query point. By default, this uses binary search (`O(log n)`), but for sequential or streaming queries, **hinted search** can achieve `O(1)` amortized lookup.

## Decision Matrix

| Policy | Best For | Complexity | Thread Safety |
|:-------|:---------|:-----------|:--------------|
| [`Binary()`](@ref search_policies) | Random access (default) | O(log n) | ✓ Stateless |
| [`HintedBinary()`](@ref search_policies) | Repeated queries in same region | O(1) hit, O(log n) miss | ✓ With hint |
| [`LinearBounded()`](@ref search_policies) | Sequential/streaming queries | O(1) local, O(log n) fallback | ✓ With hint |

!!! tip "Range grids are always O(1)"
    For uniform grids defined as `Range` (e.g., `0.0:0.1:10.0`), interval lookup is always O(1) regardless of search policy. Search policies primarily affect `Vector` grids.

## Quick Selection Guide

**Which policy should I use?**

- **Random access / general use** → `Binary()` (default, no setup needed)
- **Queries cluster in same region** → `HintedBinary()`
- **Sorted/monotonic queries (ODE, streaming)** → `LinearBounded()`

## Next Steps

- [Search Policies](@ref search_policies) — Detailed explanation of each policy
- [Using Hints](@ref using_hints) — Hint patterns, thread safety, and examples
