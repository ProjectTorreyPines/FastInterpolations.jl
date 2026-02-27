# [Search Policies](@id search_policies)

This page explains each search policy in detail, including when to use it and how it works internally.

## AutoSearch (Default)

Automatically selects `BinarySearch()` or `LinearBinarySearch()` based on query type, hint presence, and data access pattern at call time. This is the **default** for all interpolants — no configuration needed.

**Resolution rules**:

| Query type | Hint | Resolved policy | Rationale |
|:-----------|:-----|:----------------|:----------|
| Scalar (`Real`) | none | `BinarySearch()` | Stateless; no locality to exploit |
| Scalar (`Real`) | `Ref{Int}` | → `LinearBinarySearch()` | Auto-upgraded: hint implies sequential pattern (e.g. `SeriesInterpolant`) |
| Vector (`AbstractVector`) | none | `BinarySearch()` or `LinearBinarySearch()` | Adaptive: prefix monotonicity check on first K=8 elements |
| Vector (`AbstractVector`) | `Ref{Int}` | `LinearBinarySearch()` | Caller has walk state, expects locality |
| ND scalar (`NTuple{N,Real}`) | none | `BinarySearch()` per axis | Same as 1D scalar |
| ND SoA batch (`NTuple{N,AbstractVector}`) | none | `BinarySearch()` or `LinearBinarySearch()` per axis | Adaptive per axis |
| Broadcast (`itp.(xs)`) | — | `BinarySearch()` per element | Each broadcast call is independent |

```julia
itp = linear_interp(x, y)      # stores AutoSearch (the default)
itp(0.5)                       # → BinarySearch() internally (scalar query)
itp(sorted_xs)                 # → LinearBinarySearch() (sorted detected)
itp(rand_xs)                   # → BinarySearch() (random detected)
itp.(xs)                       # → BinarySearch() per element (broadcast)

# Hint auto-upgrade (scalar + hint → LinearBinarySearch)
hint = Ref(1)
itp(0.5; hint=hint)            # → LinearBinarySearch() (hint implies locality)

# Override when you know the pattern in advance:
itp(0.5; search=BinarySearch())              # force BinarySearch for all calls
itp(sorted_xs; search=LinearBinarySearch())  # force LinearBinarySearch for all calls
```

**How it works**: `AutoSearch` is resolved in two phases at the call site, not at construction time:

1. **Policy resolution** (`_resolve_search`): Picks `BinarySearch` or `LinearBinarySearch` based on query type and, for vector queries without a hint, a prefix monotonicity check on the first 8 elements.
2. **Searcher creation** (`_to_searcher`): When a `Ref{Int}` hint is provided alongside `BinarySearch`, it auto-upgrades to `LinearBinarySearch` — because the hint implies the caller expects walk-based locality (e.g., `SeriesInterpolant` scalar evaluation).

**When to override with an explicit policy**:
- You know queries are **always random** → explicit `BinarySearch()` skips the adaptive check
- You know queries are **always sorted** → explicit `LinearBinarySearch()` skips the adaptive check
- For most use cases, keeping `AutoSearch()` is the right choice

---

## BinarySearch

Pure binary search. Stateless, thread-safe, no setup required.

**Complexity**: O(log n) per query

**When to use**:
- Random access patterns
- One-off queries
- When you want consistent O(log n) for any query order

```julia
itp = linear_interp(x, y)
val = itp(0.5)                    # AutoSearch → BinarySearch() for scalar queries
val = itp(0.5; search=BinarySearch())   # explicit BinarySearch
```

**How it works**: Standard binary search on the grid. Each query starts fresh with no memory of previous queries.

---

## Linear

Maximum-speed linear search for **strictly monotonic, closely-spaced queries**. Scans the grid sequentially one interval at a time from the hint until the target is found—no binary fallback, no window limit.

**Complexity**: O(1) amortized for monotonic, closely-spaced sequences

!!! warning "Performance Warning"
    `LinearSearch()` walks the grid one interval at a time without any fallback. This delivers **best performance** when consecutive queries are close together (typical in ODE integration), but can become **extremely slow** if:

    1. Queries are far apart (sparse sampling across a large grid)
    2. Queries jump around randomly
    3. Query direction reverses frequently

    In these cases, `LinearSearch()` degrades to **O(n)** per query, making it much slower than `BinarySearch()` or `LinearBinarySearch()`.

**When to use**:
- ODE integration with strictly monotonic, fine-grained time stepping
- Streaming evaluation where consecutive queries are close together
- Performance-critical loops with **guaranteed** closely-spaced, monotonic queries

**When NOT to use**:
- Random access patterns → use `BinarySearch()`
- Queries that may be far apart → use `LinearBinarySearch()`
- Large grids with sparse query spacing → use `LinearBinarySearch()`
- General use cases → use `LinearBinarySearch()` (safer default)

```julia
# ODE-style monotonic evaluation (fastest for closely-spaced queries)
itp = linear_interp(x, y)
hint = Ref(1)
for t in t_values  # strictly increasing, closely-spaced
    y = itp(t; search=LinearSearch(), hint=hint)
end
```

**How it works**: Walks linearly from the hint position one interval at a time. Without a step limit, it will traverse the entire grid if necessary—which is optimal for close queries but catastrophic for distant ones.

---

## LinearBinarySearch

Performs a bounded linear search from the hint position. Ideal for **sorted or monotonic queries**.

**Complexity**: O(1) within bounds, O(log n) fallback

**When to use**:
- Sorted query sequences
- Monotonically increasing time (ODE solvers)
- Streaming data with local continuity

```julia
sorted_queries = sort(rand(1000))
itp = linear_interp(x, y; search=LinearBinarySearch())
vals = itp(sorted_queries)  # O(1) amortized for sorted input
```

**How it works**: Starting from the hint position, walks linearly left or right (up to `linear_window`). If the target interval isn't found within bounds, falls back to binary search.

### Tuning linear_window

You can tune the linear search window size before falling back to binary search:

```julia
LinearBinarySearch()                   # default: linear_window=8
LinearBinarySearch(linear_window=0)    # hint check only, no walk (minimal random overhead)
LinearBinarySearch(linear_window=4)    # narrow window for tight jitter patterns
LinearBinarySearch(linear_window=16)   # wider window for sparser-spaced sorted queries
```

**Guidelines**:
- **Zero (0)**: Hint check only, no walk — minimal random-query overhead. Good when queries cluster in the same interval.
- **Small `linear_window` (1–2)**: Minimal overhead for mixed query patterns
- **Medium values (4)**: Good for narrow jitter patterns (step size < 2 intervals)
- **Default (8)**: Best balance — +2.5ns random overhead, covers jitter up to ~6 intervals
- **Large `linear_window` (16–128)**: For wide jitter, highly localized queries, or very large datasets

!!! note "Type Parameter Restriction"
    `linear_window` is restricted to `0` plus powers of 2 (1, 2, 4, 8, 16, 32, 64, 128) to prevent type parameter explosion. Each unique value creates a specialized method, so limiting choices keeps compile times reasonable.

---

## Performance Summary

| Query Pattern | `AutoSearch()` | `BinarySearch()` | `LinearBinarySearch{8}()` | `LinearSearch()` |
|:--------------|:---------------|:-----------|:--------------------|:-----------|
| **Random** | ✅ Same as `BinarySearch()` | ✅ Best | ~2-3x slower | ❌ Up to 7x slower |
| **Sorted** | ✅ Same as `LinearBinarySearch()` | Baseline | ✅ ~5x faster (~2 ns) | ✅ ~5x faster |
| **Jittered** | ✅ Same as `LinearBinarySearch()` | Baseline | ✅ ~2-5x faster | ✅ ~2-5x faster |

`AutoSearch()` **adapts to your data**: it detects sorted queries and uses `LinearBinarySearch`, detects random queries and falls back to `BinarySearch`. You get near-optimal performance for both patterns without choosing manually.

!!! note "Results Vary"
    Approximate results from vector batch calls for a specific test case. Run benchmarks with your own data for accurate numbers.

---

## Baked-in vs Override

### Baked-in Policy (Constructor)

Override the default `AutoSearch` when creating the interpolant. The stored policy applies to all subsequent calls:

```julia
itp = linear_interp(x, y)                    # stores AutoSearch() — adapts per call
itp = linear_interp(x, y; search=BinarySearch())   # always BinarySearch(), regardless of query type
itp = linear_interp(x, y; search=LinearBinarySearch())  # always LinearBinarySearch()
```

### Override at Call Time

Override the stored policy for a single call without changing the interpolant:

```julia
itp = linear_interp(x, y)  # stores AutoSearch (default)

# Call-site override — does not change itp.search_policy
val  = itp(0.5; search=BinarySearch())              # force BinarySearch for this call only
vals = itp(sorted_xs; search=LinearBinarySearch())  # force LinearBinarySearch for this call only
```

This is useful when most queries benefit from one policy, but occasional queries need different behavior.

---

## See Also

- [Overview](@ref search_hints) — Quick decision matrix
- [Using Hints](@ref using_hints) — External hints for maximum performance
