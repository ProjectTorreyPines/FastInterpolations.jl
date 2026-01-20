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

## LinearBounded

Performs a bounded linear search from the hint position. Ideal for **sorted or monotonic queries**.

**Complexity**: O(1) within bounds, O(log n) fallback

**When to use**:
- Sorted query sequences
- Monotonically increasing time (ODE solvers)
- Streaming data with local continuity

```julia
sorted_queries = sort(rand(1000))
itp = linear_interp(x, y; search=LinearBounded())
vals = itp(sorted_queries)  # O(1) amortized for sorted input
```

**How it works**: Starting from the hint position, walks linearly left or right (up to `max_steps`). If the target interval isn't found within bounds, falls back to binary search.

### Tuning max_steps

You can tune the maximum linear steps before falling back to binary search:

```julia
LinearBounded()             # default: max 8 steps
LinearBounded(max_steps=4)  # smaller bound for tightly spaced queries
LinearBounded(max_steps=16) # larger bound for sparser queries
```

**Guidelines**:
- **Small `max_steps` (4)**: Better when queries are very close together
- **Large `max_steps` (16-32)**: Better when queries might skip a few intervals
- **Default (8)**: Good balance for most use cases

!!! note "Type Parameter Restriction"
    `max_steps` is restricted to powers of 2 (1, 2, 4, 8, 16, 32, 64) to prevent type parameter explosion. Each unique value creates a specialized method, so limiting choices keeps compile times reasonable.

---

## Baked-in vs Override

### Baked-in Policy (Constructor)

Set the default policy when creating the interpolant:

```julia
# All queries use LinearBounded by default
itp = linear_interp(x, y; search=LinearBounded())
val = itp(0.5)  # uses LinearBounded
```

### Override at Call Time

Override the baked-in policy for specific queries:

```julia
itp = linear_interp(x, y; search=LinearBounded())

# Override for a single call
val = itp(0.5; search=Binary())  # uses Binary for this call only
```

This is useful when most queries benefit from one policy, but occasional queries need different behavior.

---

## See Also

- [Overview](@ref search_hints) — Quick decision matrix
- [Using Hints](@ref using_hints) — External hints for maximum performance
