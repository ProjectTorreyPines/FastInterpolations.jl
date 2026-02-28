# [Using Hints](@id using_hints)

Hints allow you to persist search state across calls, enabling O(1) lookup for sequential or streaming queries. **Hints are the primary performance lever** — instead of manually choosing search policies, just pass a hint and let `AutoSearch` do the right thing.

## Basic Usage

Provide an external `Ref{Int}` to persist the search position across calls:

```julia
itp = linear_interp(x, y)
hint = Ref(1)  # external hint storage

# Hint persists across calls — AutoSearch auto-upgrades to LinearBinarySearch
for xi in streaming_data
    val = itp(xi; hint=hint)
end
```

The hint stores the last-found interval index, allowing the next query to start searching from that position. When `AutoSearch` (the default) sees a hint, it automatically upgrades scalar queries to `LinearBinarySearch()` — no manual policy selection needed.

## When to Use External Hints

| Use Case | Why It Helps | Example |
|:---------|:-------------|:--------|
| **ODE solver callbacks** | Time increases monotonically; hint tracks position | `itp(t; hint=hint)` |
| **Streaming data** | Continuous data flow with local continuity | `itp(xi; hint=hint)` |
| **Multiple interpolants** | Share hint when querying same x-position across interpolants | Same `hint` for all |
| **Sequential scalar loops** | Without a hint, each scalar call does full binary search | `hint=Ref(1)` in the loop |

## Auto-Upgrade Behavior

When you provide a `hint` argument, `AutoSearch` automatically upgrades to `LinearBinarySearch()` — because the presence of a hint implies your queries have locality:

```julia
hint = Ref(1)
val = itp(0.5; hint=hint)  # AutoSearch → auto-upgrades to LinearBinarySearch{8}
```

This works regardless of whether you explicitly set a search policy:

```julia
hint = Ref(1)
val = itp(0.5; hint=hint)                           # AutoSearch + hint → LinearBinarySearch{8}
val = itp(0.5; search=BinarySearch(), hint=hint)     # BinarySearch + hint → also auto-upgrades!
```

Without a hint, scalar queries use pure binary search (no hint tracking). This is the safe default for one-off queries.

## Thread Safety

Each thread must have its own hint to avoid data races:

```julia
# Thread-safe pattern — per-thread hint
Threads.@threads for i in 1:n
    local_hint = Ref(1)  # each thread gets its own hint
    for xi in chunks[i]
        val = itp(xi; hint=local_hint)  # AutoSearch + hint → LinearBinarySearch
    end
end
```

!!! warning "Shared hints cause data races"
    Never share a single `Ref{Int}` hint across threads. Each thread modifies the hint during search, leading to race conditions.

**Safe patterns**:
- Create hint inside the threaded loop (per-thread)
- Use thread-local storage
- Without hints, `AutoSearch` / `BinarySearch` are stateless and inherently thread-safe

---

## Examples

### ODE Solver Callback

```julia
using FastInterpolations

x = 0.0:0.01:10.0
y = sin.(x)
itp = linear_interp(x, y)

# External hint persists across solver steps
hint = Ref(1)

function ode_callback!(du, u, p, t)
    # t increases monotonically → hint enables O(1) lookup
    forcing = itp(t; hint=hint)
    du[1] = -u[1] + forcing
end
```

!!! tip "No need to set search policy for ODE"
    Just pass `hint=Ref(1)`. `AutoSearch` detects the hint and uses `LinearBinarySearch()` — which gives O(1) for monotonic time stepping with a safe O(log n) fallback if the solver ever takes an unusual step.

### Batch Processing with Sorted Queries

For vector queries, `AutoSearch` detects sorted data automatically — no hint needed:

```julia
x = collect(range(0.0, 10.0, 10001))
y = sin.(2π .* x)
itp = linear_interp(x, y)  # stores AutoSearch (default)

# Sort queries for optimal performance — AutoSearch detects this
queries = sort(rand(100_000) .* 10)
results = itp(queries)  # AutoSearch → LinearBinarySearch → O(n) total
```

### Random Access Pattern

For random access, hints provide no benefit. `AutoSearch` detects random patterns via a prefix monotonicity check and resolves to `BinarySearch()` automatically:

```julia
x = collect(range(0.0, 10.0, 10001))
y = sin.(2π .* x)
itp = linear_interp(x, y)  # stores AutoSearch (default)

# Random queries → AutoSearch detects non-monotonic prefix → BinarySearch()
random_queries = rand(100_000) .* 10
results = itp(random_queries)  # AutoSearch → BinarySearch (adaptive)
```

### Shared Hint Across Multiple Interpolants

When querying multiple interpolants at the same x-position, a single shared hint works because all interpolants use the same grid:

```julia
x = 0.0:0.01:10.0
itp_temp = linear_interp(x, temperature)
itp_pressure = linear_interp(x, pressure)
itp_density = linear_interp(x, density)

hint = Ref(1)  # shared hint for same x-grid

for t in time_points
    # All three use same hint since they share the x-grid
    # AutoSearch + hint → LinearBinarySearch automatically
    T = itp_temp(t; hint=hint)
    P = itp_pressure(t; hint=hint)
    ρ = itp_density(t; hint=hint)
end
```

---

## See Also

- [Overview](@ref search_hints) — Quick decision matrix
- [Search Policies](@ref search_policies) — Detailed policy explanations
