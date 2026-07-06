# [Performance Tips](@id performance_tips)

FastInterpolations.jl is fast by default — for most code you do not need to tune
anything. This page is an advanced-user checklist for the cases where you *do*
want the last few nanoseconds, in rough order of impact. Each lever links to the
guide with the full details, and to the [Benchmarks](@ref) page for measured
numbers.

## At a Glance

| Lever | One-liner |
|:------|:----------|
| [Grid type](@ref perf-grid) | Represent a uniform grid with the most specific `Range` type that is still correct. |
| [Batch queries](@ref perf-batch) | Pass a vector; never loop over scalar calls. Use the `!` in-place form in hot loops. |
| [Search hints](@ref perf-hints) | For sequential access, pass `hint = Ref(1)`. |
| [`InBounds`](@ref perf-inbounds) | When queries are known in-domain, skip the domain check. |
| [`value_gradient`](@ref perf-valgrad) | Need value **and** gradient? Locate the interval once. |

---

## [Choose the Leanest Grid Type](@id perf-grid)

The grid's *type* — not just its values — decides how much work each query spends
locating its interval. Represent a uniform grid with the **most specific `Range`
type that is still correct**. This is a correctness-preserving choice, not a free
speed dial: swapping `0.0:0.1:10.0` for `1:100` changes the *coordinates*, so it
only applies when your grid genuinely has that shape.

For a grid whose points are `1, 2, …, length(y)`, worst to best:

```julia
x = collect(1.0:length(y))   # ❌ Vector      — O(log n) binary search
x = 1.0:length(y)            # ⚠️ Float range — O(1), but the index needs a ×(1/step)
x = 1:length(y)              # ✅ UnitRange    — Int, step ≡ 1 pinned at the type level
x = Base.OneTo(length(y))    # ✅ OneTo        — also 1-based at the type level
x = axes(y, 1)               # ✅ idiomatic    — a vector's own axis *is* a `Base.OneTo`
```

Each step down the list pins more of the grid's structure into the *type*, so the
compiler drops work from the per-query index math: the `Vector` loses O(1) lookup
entirely, a float range carries a runtime reciprocal, a `UnitRange` fixes the step
to `1`, and `Base.OneTo` fixes the origin to `1` so the interval index *is* the
query position (index-space search).

Pick a row, then use it the same way — build once or one-shot:

```julia
linear_interp(x, y)          # persistent interpolant
linear_interp(x, y, xq)      # one-shot

linear_interp(axes(y), y)    # `axes(y)` is the natural whole-array form (routes to 1-D)
```

The gains are small and matter only in tight scalar/batch loops — but they cost
nothing when the grid genuinely has that shape (samples on `1, 2, …, n`). Use
`1:n` / `Base.OneTo(n)` rather than `Float64.(1:n)` or `collect(1:n)` when the
x-coordinate *is* the index.

!!! note "`1:n` vs `Base.OneTo(n)`"
    `1:n` keeps the unit-step fast path but not the type-level 1-based one — its
    `first == 1` is a runtime value, and promoting on it would make the
    interpolant type value-dependent. The search still detects the `lo == 1` case
    at run time, so `1:n` gets most of the benefit; `Base.OneTo(n)` makes it a
    type-level guarantee. See [Grid Selection](@ref grid_selection) for the basic
    Range-vs-Vector trade-off.

For **non-uniform** grids a `Vector` is required — see the next levers and
[Search & Hints](@ref search_hints) to optimize its lookup.

---

## [Batch Queries and In-place Output](@id perf-batch)

When querying multiple points, pass the whole vector — the batch path amortizes
per-call overhead that a scalar loop pays on every iteration:

```julia
# ❌ per-call overhead on every element
out = [linear_interp(x, y, q) for q in xq]

# ✅ batch path
out = linear_interp(x, y, xq)
```

In a hot loop, pre-allocate the output once and use the `!` in-place form for
true zero-allocation:

```julia
out = similar(xq)
linear_interp!(out, x, y, xq)   # reuses `out`, no per-call allocation
```

See [Memory & Allocation](@ref memory_allocation) for the full treatment.

---

## [Search Hints for Sequential Access](@id perf-hints)

On **`Vector`** grids the interval search matters (uniform `Range` grids are
already O(1)). The default `AutoSearch()` adapts per call and suits almost
everyone. For sequential or streaming access (ODE time-stepping, sorted scans), a
`hint = Ref(1)` is the single most effective lever — it upgrades scalar queries to
O(1) walk-based search with a safe O(log n) fallback:

```julia
itp  = linear_interp(x, y)
hint = Ref(1)
for t in monotonic_times
    itp(t; hint = hint)   # O(1) amortized
end
```

See [Search & Hints](@ref search_hints) for the decision matrix and thread-safety
patterns.

---

## [The `InBounds` Fast Path](@id perf-inbounds)

When you already know a query lies inside the grid domain, skip the extrapolation
domain check with `InBounds()` — the interpolation analogue of Julia's
`@inbounds`. For a batch this elides the whole O(n) in-domain scan; for a scalar
it removes a compare-and-branch.

```julia
# One-shot form (always supported):
linear_interp(x, y, xq; extrap = InBounds())

# Persistent interpolant — override per call, without rebuilding:
itp = cubic_interp(x, y; extrap = NoExtrap())   # stored contract
itp(xq)                                          # runs the domain check
itp(xq; extrap = InBounds())                     # skips it (this call only)
```

The stored extrapolation contract is unchanged — `InBounds` is a per-call
*assertion*, not a new extrapolation mode, so it composes onto any stored extrap
(including periodic `WrapExtrap`).

For strictly-interior queries on unit-step range grids, the narrower
`InBounds(last = :exclusive)` promises `first(x) ≤ xq < last(x)` and unlocks an
even leaner direct-index search:

```julia
itp((xq, yq, zq); extrap = InBounds(last = :exclusive))
```

!!! danger "`InBounds` is unsafe"
    Like `@inbounds`, violating the promise is undefined behavior — an actually
    out-of-domain query reads outside the grid. Only opt in when the caller
    guarantees the query is in-domain. See [Extrapolation](../extrapolation.md) for the full
    contract table.

---

## [Value and Gradient Together](@id perf-valgrad)

In optimization inner loops you often need both the value and the gradient at a
point. [`value_gradient`](@ref) performs the interval search **once** and returns
both, instead of paying for two separate locates — the ideal match for Optim.jl's
`only_fg!` interface:

```julia
val, grad = value_gradient(itp, (x, y))   # one locate, both outputs
```

See [Optimization](@ref optimization_guide) for the full `fg!` pattern.

---

## See Also

- [Memory & Allocation](@ref memory_allocation) — grid choice, batch API, zero-alloc
- [Search & Hints](@ref search_hints) — search policies and hint patterns
- [Extrapolation](../extrapolation.md) — `InBounds` and the full extrapolation contract
- [Optimization (Optim.jl)](@ref optimization_guide) — `value_gradient`, analytical derivatives
- [Benchmarks](@ref) — measured comparisons against Interpolations.jl
