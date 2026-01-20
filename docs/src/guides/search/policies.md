# [Search Policies](@id search_policies)

This page explains each search policy in detail, including when to use it and how it works internally.

## Binary (Default)

Pure binary search. Stateless, thread-safe, no setup required.

**Complexity**: O(log n) per query

**When to use**:
- Random access patterns
- One-off queries
- When you don't know the query pattern in advance

```julia
itp = linear_interp(x, y)
val = itp(0.5)                    # uses Binary() by default
val = itp(0.5; search=Binary())   # explicit
```

**How it works**: Standard binary search on the grid. Each query starts fresh with no memory of previous queries.

---

## HintedBinary

Caches the last-found interval. If the next query falls in the same interval, lookup is O(1).

**Complexity**: O(1) cache hit, O(log n) cache miss

**When to use**:
- Queries that cluster in the same region
- Monte Carlo sampling within a subregion
- Iterative refinement around a point

```julia
itp = linear_interp(x, y; search=HintedBinary())
for xi in query_points
    val = itp(xi)  # O(1) when consecutive queries hit same interval
end
```

**How it works**: Before binary search, checks if the query falls in the cached interval. If yes, returns immediately. If no, performs full binary search and updates the cache.

---

## Linear

Maximum-speed linear search for **strictly monotonic, closely-spaced queries**. Scans the grid sequentially one interval at a time from the hint until the target is found—no binary fallback, no window limit.

**Complexity**: O(1) amortized for monotonic, closely-spaced sequences

!!! warning "Performance Warning"
    `Linear()` walks the grid one interval at a time without any fallback. This delivers **best performance** when consecutive queries are close together (typical in ODE integration), but can become **extremely slow** if:

    1. Queries are far apart (sparse sampling across a large grid)
    2. Queries jump around randomly
    3. Query direction reverses frequently

    In these cases, `Linear()` degrades to **O(n)** per query, making it much slower than `Binary()` or `LinearBinary()`.

**When to use**:
- ODE integration with strictly monotonic, fine-grained time stepping
- Streaming evaluation where consecutive queries are close together
- Performance-critical loops with **guaranteed** closely-spaced, monotonic queries

**When NOT to use**:
- Random access patterns → use `Binary()`
- Queries that may be far apart → use `LinearBinary()`
- Large grids with sparse query spacing → use `LinearBinary()`
- General use cases → use `LinearBinary()` (safer default)

```julia
# ODE-style monotonic evaluation (fastest for closely-spaced queries)
itp = linear_interp(x, y)
hint = Ref(1)
for t in t_values  # strictly increasing, closely-spaced
    y = itp(t; search=Linear(), hint=hint)
end
```

**How it works**: Walks linearly from the hint position one interval at a time. Without a step limit, it will traverse the entire grid if necessary—which is optimal for close queries but catastrophic for distant ones.

### Performance Summary

| Query Pattern | `Binary()` | `LinearBinary()` | `Linear()` |
|:--------------|:-----------|:-----------------|:-----------|
| **Random** | ✅ Best | ~2-3x slower | ❌ Up to 7x slower |
| **Monotonic** | Baseline | ✅ ~5x faster | ✅ ~6x faster |

!!! note "Results Vary"
    These are approximate results from a 500-point grid with 1000 queries. Actual performance depends on your **grid size** and **query spacing**. Run benchmark with your own data to find the best policy.

---

## LinearBinary

Performs a bounded linear search from the hint position. Ideal for **sorted or monotonic queries**.

**Complexity**: O(1) within bounds, O(log n) fallback

**When to use**:
- Sorted query sequences
- Monotonically increasing time (ODE solvers)
- Streaming data with local continuity

```julia
sorted_queries = sort(rand(1000))
itp = linear_interp(x, y; search=LinearBinary())
vals = itp(sorted_queries)  # O(1) amortized for sorted input
```

**How it works**: Starting from the hint position, walks linearly left or right (up to `linear_window`). If the target interval isn't found within bounds, falls back to binary search.

### Tuning linear_window

You can tune the linear search window size before falling back to binary search:

```julia
LinearBinary()             # default: linear_window=8
LinearBinary(linear_window=4)  # smaller bound for tightly spaced queries
LinearBinary(linear_window=16) # larger bound for sparser queries
```

**Guidelines**:
- **Small `linear_window` (4)**: Better when queries are very close together
- **Large `linear_window` (16-32)**: Better when queries might skip a few intervals
- **Default (8)**: Good balance for most use cases

!!! note "Type Parameter Restriction"
    `linear_window` is restricted to powers of 2 (1, 2, 4, 8, 16, 32, 64) to prevent type parameter explosion. Each unique value creates a specialized method, so limiting choices keeps compile times reasonable.

---

## Baked-in vs Override

### Baked-in Policy (Constructor)

Set the default policy when creating the interpolant:

```julia
# All queries use LinearBinary by default
itp = linear_interp(x, y; search=LinearBinary())
val = itp(0.5)  # uses LinearBinary
```

### Override at Call Time

Override the baked-in policy for specific queries:

```julia
itp = linear_interp(x, y; search=LinearBinary())

# Override for a single call
val = itp(0.5; search=Binary())  # uses Binary for this call only
```

This is useful when most queries benefit from one policy, but occasional queries need different behavior.

---

## See Also

- [Overview](@ref search_hints) — Quick decision matrix
- [Using Hints](@ref using_hints) — External hints for maximum performance
