# [Search Policies](@id search_policies)

This page explains each search policy in detail, including when to use it and how it works internally.

## AutoSearch (Default)

Automatically selects `Binary()` or `LinearBinary()` based on query type at call time. This is the **default** for all interpolants — no configuration needed.

**Resolution rules**:

| Query type | Resolved policy | Rationale |
|:-----------|:----------------|:----------|
| Scalar (`Real`) | `Binary()` | Stateless; no hint locality to exploit |
| Vector (`AbstractVector`) | `LinearBinary()` | Exploits hint locality for sorted sequences |
| ND scalar (`Tuple{Vararg{Real}}`) | `Binary()` per axis | Same as 1D scalar |
| ND SoA batch (`NTuple{N, AbstractVector}`) | `LinearBinary()` per axis | Same as 1D vector |
| Broadcast (`itp.(xs)`) | `Binary()` per element | Each broadcast call is independent |

```julia
itp = linear_interp(x, y)      # stores AutoSearch (the default)
itp(0.5)                       # → Binary() internally (scalar query)
itp([0.1, 0.5, 0.9])           # → LinearBinary() internally (vector query)
itp.(xs)                       # → Binary() per element (broadcast)

# Override when you know the pattern in advance:
itp(0.5; search=Binary())              # force Binary for all calls
itp(sorted_xs; search=LinearBinary())  # force LinearBinary for all calls
```

**How it works**: `AutoSearch` is resolved at the call site, not at construction time. The interpolant stores `AutoSearch()` and dispatches on the query argument's concrete type each time the interpolant is called.

**When to override with an explicit policy**:
- You know queries are **always random** → explicit `Binary()` skips the dispatch check
- You know queries are **always sorted** → explicit `LinearBinary()` skips the dispatch check
- For most use cases, keeping `AutoSearch()` is the right choice

---

## Binary

Pure binary search. Stateless, thread-safe, no setup required.

**Complexity**: O(log n) per query

**When to use**:
- Random access patterns
- One-off queries
- When you want consistent O(log n) for any query order

```julia
itp = linear_interp(x, y)
val = itp(0.5)                    # AutoSearch → Binary() for scalar queries
val = itp(0.5; search=Binary())   # explicit Binary
```

**How it works**: Standard binary search on the grid. Each query starts fresh with no memory of previous queries.

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
LinearBinary()                   # default: linear_window=8
LinearBinary(linear_window=0)    # hint check only, no walk (minimal random overhead)
LinearBinary(linear_window=4)    # narrow window for tight jitter patterns
LinearBinary(linear_window=16)   # wider window for sparser-spaced sorted queries
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

| Query Pattern | `AutoSearch()` | `Binary()` | `LinearBinary()` | `Linear()` |
|:--------------|:---------------|:-----------|:-----------------|:-----------|
| **Random** | ✅ Same as `Binary()` | ✅ Best | ~2.5-3x slower | ❌ Up to 7x slower |
| **Monotonic** | ✅ Same as `LinearBinary()` | Baseline | ✅ ~4-6x faster | ✅ ~6x faster |

`AutoSearch()` has negligible overhead — it dispatches to the same underlying code as the resolved policy, so you only pay for the type dispatch, not for any extra search work.

!!! note "Results Vary"
    These are approximate results from **vector batch calls** (`itp(out, queries)`) on 500–2000 point grids.
    The random penalty (~2.5-3x) comes from the hint walk landing at the wrong position before falling back to
    binary. The monotonic speedup grows with grid size (larger grids benefit more from hint locality).
    For **scalar calls** without a persistent hint, the difference is much smaller (~1.2x).
    Run benchmark with your own data to find the best policy.

---

## Baked-in vs Override

### Baked-in Policy (Constructor)

Override the default `AutoSearch` when creating the interpolant. The stored policy applies to all subsequent calls:

```julia
itp = linear_interp(x, y)                    # stores AutoSearch() — adapts per call
itp = linear_interp(x, y; search=Binary())   # always Binary(), regardless of query type
itp = linear_interp(x, y; search=LinearBinary())  # always LinearBinary()
```

### Override at Call Time

Override the stored policy for a single call without changing the interpolant:

```julia
itp = linear_interp(x, y)  # stores AutoSearch (default)

# Call-site override — does not change itp.search_policy
val  = itp(0.5; search=Binary())              # force Binary for this call only
vals = itp(sorted_xs; search=LinearBinary())  # force LinearBinary for this call only
```

This is useful when most queries benefit from one policy, but occasional queries need different behavior.

---

## See Also

- [Overview](@ref search_hints) — Quick decision matrix
- [Using Hints](@ref using_hints) — External hints for maximum performance
