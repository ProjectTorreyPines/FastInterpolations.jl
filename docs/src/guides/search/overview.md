# [Search & Hints Overview](@id search_hints)

Search policies control how interpolants find the correct grid interval for a query point.

!!! note "This page applies to Vector grids"
    For **uniform grids**, use `Range` instead—lookup is always **O(1)** regardless of search policy. See [Grid Selection](../advanced_overview.md#Grid-Selection:-Range-vs-Vector) for details. This page focuses on **Vector grids** where choosing the right search policy matters.

## Quick Summary

For most use cases, **the default works well out of the box**. `AutoSearch()` detects your query pattern at call time and picks the right strategy automatically:

- **Random queries** → pure binary search
- **Sorted/monotonic queries** → linear walk with binary fallback
- **Scalar + hint** → auto-upgrades to walk-based search

For sequential or streaming patterns, passing a **hint** is the simplest way to get O(1) lookup:

```julia
hint = Ref(1)
for xi in streaming_data
    val = itp(xi; hint=hint)   # AutoSearch auto-upgrades to LinearBinarySearch
end
```

In most cases, no manual policy selection is needed.

## How It Works

Every interpolation query on a `Vector` grid requires finding which interval contains the query point. By default, `AutoSearch()` resolves this at call time — it uses binary search (`O(log n)`) for random patterns and walk-based search (`O(1)` amortized) for sorted/streaming patterns.

## Quick Examples

```@example
using FastInterpolations

x = collect(range(0, 1.0, length=1000))
y = x.^3

# AutoSearch is the default — adapts automatically per call
xq = rand(1000)
linear_interp(x, y, xq)  # AutoSearch → BinarySearch()

# For sorted/monotonic queries — AutoSearch detects this automatically
xq_sorted = sort(xq)
linear_interp(x, y, xq_sorted)  # AutoSearch → LinearBinarySearch()

# Streaming/scalar with hint — auto-upgrades to walk-based search
hint = Ref(1)
itp = linear_interp(x, y)
for xi in xq_sorted
    itp(xi; hint=hint)  # AutoSearch + hint → LinearBinarySearch()
end
nothing # hide
```

!!! tip "Factory Functions"
    The [`Search()`](@ref factory_functions) factory provides a single discoverable entry point for all search policies. For ND per-axis configuration, use the multi-arg form: `Search(:binary, :linear_binary, :auto)`. See [Factory Functions](@ref factory_functions) for details.

## Decision Matrix

| Policy | Best For | Complexity | Thread Safety | Recommendation |
|:-------|:---------|:-----------|:--------------|:---------------|
| [`AutoSearch()`](@ref search_policies) | **All use cases (default)** — adapts per query type | Delegates to BinarySearch/LinearBinarySearch | ✓ Stateless | ✅ **Use this** |
| [`BinarySearch()`](@ref search_policies) | Random access (explicit) | O(log n) | ✓ Stateless | Optional explicit override |
| [`LinearBinarySearch()`](@ref search_policies) | Monotonic queries (explicit) | O(1) local, O(log n) fallback | ✓ With hint | Optional explicit override |
| [`LinearSearch()`](@ref search_policies) | Expert only — see warning | O(1) best, **O(n) worst** | ✓ With hint | ⚠️ Not recommended |

!!! warning "Avoid LinearSearch() Unless You Know What You're Doing"
    `LinearSearch()` has **no binary fallback**. If queries are far apart or not strictly monotonic, it walks the entire grid — degrading to **O(n) per query**. In almost all cases, `LinearBinarySearch()` (or just the default `AutoSearch()`) is safer and equally fast for truly monotonic data.

!!! note "Why No Hunt Algorithm?"
    The Hunt (correlated) algorithm offers *theoretical* O(log k) for nearby queries and O(log n) worst-case.
    However, our benchmarks showed no practical advantage over existing policies—each access pattern
    already has a better-suited option.

## Quick Selection Guide

**Which policy should I use?**

1. **Almost everyone** → Don't set a policy. `AutoSearch()` is the default and handles random, sorted, and mixed patterns automatically.
2. **Streaming / ODE / sequential scalar queries** → Just pass `hint=Ref(1)`. AutoSearch will auto-upgrade to `LinearBinarySearch()`.
3. **Known random access, skip adaptive check** → `BinarySearch()` (marginal benefit; skips prefix monotonicity check)
4. **Known sorted, skip adaptive check** → `LinearBinarySearch()` (marginal benefit; skips prefix monotonicity check)
5. **Expert: strictly monotonic, closely-spaced, benchmarked** → `LinearSearch()` ⚠️ (see [LinearSearch warning](@ref search_policies))

!!! tip "Hints Are the Main Lever"
    Instead of choosing between policies manually, **just inject a hint** and let `AutoSearch` do the right thing. A `Ref{Int}` hint tells the system that your queries have locality — it will automatically use walk-based search with binary fallback.

## Next Steps

- [Search Policies](@ref search_policies) — Detailed explanation of each policy
- [Using Hints](@ref using_hints) — Hint patterns, thread safety, and examples
