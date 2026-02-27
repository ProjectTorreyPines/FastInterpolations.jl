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

# AutoSearch is the default — adapts automatically per call
xq = rand(1000)
linear_interp(x, y, xq)                         # AutoSearch → Binary() (random detected)
linear_interp(x, y, xq; search=Binary())         # Explicit: same result

# For sorted/monotonic queries
xq_sorted = sort(xq)
linear_interp(x, y, xq_sorted)                   # AutoSearch → LinearBinary() (sorted detected)
linear_interp(x, y, xq_sorted; search=LinearBinary())  # Same result, explicit
nothing # hide
```

## Decision Matrix

| Policy | Best For | Complexity | Thread Safety |
|:-------|:---------|:-----------|:--------------|
| [`AutoSearch()`](@ref search_policies) | **General use (default)** — adapts per query type | Delegates to Binary/LinearBinary | ✓ Stateless |
| [`Binary()`](@ref search_policies) | Random access (explicit) | O(log n) | ✓ Stateless |
| [`LinearBinary()`](@ref search_policies) | **Monotonic queries (explicit)** | O(1) local, O(log n) fallback | ✓ With hint |
| [`Linear()`](@ref search_policies) | Close + monotonic queries (expert) | O(1) amortized | ✓ With hint |

!!! note "Why No Hunt Algorithm?"
    The Hunt (correlated) algorithm offers *theoretical* O(log k) for nearby queries and O(log n) worst-case.
    However, our benchmarks showed no practical advantage over existing policies—each access pattern
    already has a better-suited option.

## Quick Selection Guide

**Which policy should I use?**

- **General use / unknown pattern** → `AutoSearch()` ✅ **default** — adapts per query type and access pattern (sorted→`LinearBinary`, random→`Binary`)
- **Known random access** → `Binary()` (explicit; skips AutoSearch dispatch)
- **Queries cluster in same region** → `LinearBinary(linear_window=0)` (hint check only)
- **Known monotonic queries (sorted, streaming, ODE)** → `LinearBinary()` (explicit)
- **Strictly monotonic, performance-critical** → `Linear()` (benchmark first! see below)

!!! note "Benchmark Before Choosing Linear()"
    `Linear()` can be faster for some monotonic patterns but slower for others. Always benchmark with your actual query patterns.

## Next Steps

- [Search Policies](@ref search_policies) — Detailed explanation of each policy
- [Using Hints](@ref using_hints) — Hint patterns, thread safety, and examples
