# [Using Hints](@id using_hints)

Hints allow you to persist search state across calls, enabling O(1) lookup for sequential or streaming queries.

## Basic Usage

For stateful policies (`HintedBinary`, `LinearBounded`), provide an external `Ref{Int}` to persist the hint:

```julia
itp = linear_interp(x, y)
hint = Ref(1)  # external hint storage

# Hint persists across calls
for xi in streaming_data
    val = itp(xi; search=LinearBounded(), hint=hint)
end
```

The hint stores the last-found interval index, allowing the next query to start searching from that position.

## When to Use External Hints

External hints are particularly useful for:

| Use Case | Why It Helps |
|:---------|:-------------|
| **ODE solver callbacks** | Time increases monotonically; hint tracks position |
| **Streaming data** | Continuous data flow with local continuity |
| **Multiple interpolants** | Share hint when querying same x-position across interpolants |

## Auto-Upgrade Behavior

When you provide a `hint` argument with `Binary()`, the search automatically upgrades to `HintedBinary` behavior:

```julia
hint = Ref(1)
val = itp(0.5; search=Binary(), hint=hint)  # auto-upgrades to HintedBinary
```

This means you can:
- Keep `Binary()` as your default policy
- Pass a hint only when beneficial
- Get hinted behavior automatically when a hint is provided

Without a hint, pure binary search is used.

## Thread Safety

Each thread must have its own hint to avoid data races:

```julia
# Thread-safe pattern
Threads.@threads for i in 1:n
    local_hint = Ref(1)  # per-thread hint
    for xi in chunks[i]
        val = itp(xi; search=LinearBounded(), hint=local_hint)
    end
end
```

!!! warning "Shared hints cause data races"
    Never share a single `Ref{Int}` hint across threads. Each thread modifies the hint during search, leading to race conditions.

**Safe patterns**:
- Create hint inside the threaded loop (per-thread)
- Use thread-local storage
- Use `Binary()` without hints (stateless, inherently thread-safe)

---

## Examples

### ODE Solver Callback

```julia
using FastInterpolations

# Create interpolant with LinearBounded (optimal for monotonic time)
x = 0.0:0.01:10.0
y = sin.(x)
itp = linear_interp(x, y; search=LinearBounded())

# External hint persists across solver steps
hint = Ref(1)

function ode_callback!(du, u, p, t)
    # t increases monotonically → O(1) lookup with hint
    forcing = itp(t; hint=hint)
    du[1] = -u[1] + forcing
end
```

### Batch Processing with Sorted Queries

```julia
x = collect(range(0.0, 10.0, 10001))
y = sin.(2π .* x)
itp = linear_interp(x, y; search=LinearBounded())

# Sort queries for optimal performance
queries = sort(rand(100_000) .* 10)
results = itp(queries)  # O(n) total instead of O(n log n)
```

### Random Access Pattern

For random access, hints provide no benefit. Use the default `Binary()`:

```julia
x = collect(range(0.0, 10.0, 10001))
y = sin.(2π .* x)
itp = linear_interp(x, y)  # default Binary()

# Random queries → Binary search is optimal
random_queries = rand(100_000) .* 10
results = itp(random_queries)
```

### Shared Hint Across Multiple Interpolants

When querying multiple interpolants at the same x-position:

```julia
x = 0.0:0.01:10.0
itp_temp = linear_interp(x, temperature)
itp_pressure = linear_interp(x, pressure)
itp_density = linear_interp(x, density)

hint = Ref(1)  # shared hint for same x-grid

for t in time_points
    # All three use same hint since they share the x-grid
    T = itp_temp(t; search=LinearBounded(), hint=hint)
    P = itp_pressure(t; search=LinearBounded(), hint=hint)
    ρ = itp_density(t; search=LinearBounded(), hint=hint)
end
```

---

## See Also

- [Overview](@ref search_hints) — Quick decision matrix
- [Search Policies](@ref search_policies) — Detailed policy explanations
