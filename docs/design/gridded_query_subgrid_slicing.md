# Design: Lazy sub-`GriddedQuery` slicing

> Status: Reviewed (Codex + empirical prototype) → Implemented
> Created: 2026-07-13
> Depends on: `GriddedQuery{T,E,N} <: AbstractArray{E,N}` (branch `feat/gridded-query-collection`)
>
> **Outcome:** implemented as designed. The Aqua-ambiguity blocker below was
> resolved by adding 9 latent-only disambiguator methods (Groups A/B/C); the
> package reports **0 ambiguities** and `Aqua.test_all` + the full gridded suite
> pass. Cartesian all-vector/colon slices return a lazy, view-backed
> sub-`GriddedQuery` (0 coordinate-data copy, range axes stay ranges, fast-path
> re-entry); scalar-drop and linear slices materialize as before.

## Motivation

`GriddedQuery` is now an `AbstractArray{E,N}` of coordinate points (the lazy,
value-valued analogue of `CartesianIndices`). Under the default `AbstractArray`
machinery, a Cartesian slice **materializes**:

```julia
gq = GriddedQuery((1.0:100.0, collect(1.0:100.0)))
gq[1:2, 3:4]            # → Matrix{Tuple{Float64,Float64}}  (dense, eager)
```

Two things are lost by materializing:

1. **The separable fast path.** The result is a `Matrix{Tuple}` (array-of-structs),
   not a `GriddedQuery`. Fed back to `interp`, it takes the generic point-wise
   batch path — O(∏Mₐ) independent evaluations instead of O(ΣMₐ) per-axis anchor
   reuse.
2. **The shape**, when routed through the batch protocol (the generic
   `_query_size` is flat), so `interp(grids, data, gq[1:2,3:4])` returns a flat
   `Vector`, not a 2×2 block.

Both vanish if a Cartesian slice returns a **lazy sub-`GriddedQuery`** over the
sliced axes: it re-enters the fast path and preserves shape. This is exactly the
Base precedent — a range sliced by a range stays a range, and `CartesianIndices`
sliced by ranges stays a `CartesianIndices`.

## Goal / non-goals

- **Goal**: `gq[I₁, …, I_N]` with every `Iₐ` a vector/range/colon returns a lazy
  sub-`GriddedQuery` over the per-axis sub-selections, with **no coordinate-data
  copy** (only view wrappers), preserving element type, shape, and fast-path
  eligibility.
- **Non-goal**: changing scalar indexing (`gq[i,j]` → point), linear indexing
  (`gq[k]`, `gq[2:4]` on an N≥2 grid), interpolation kernels, or the
  interpolation hot path (`_query_*`). No mutation (`setindex!`) is added.

## Precedent (Base)

| Expression | Returns | Lazy |
|---|---|---|
| `(1:10)[2:4]` | `UnitRange` | ✅ |
| `view(1:10, 2:4)` | `StepRangeLen`/range | ✅ (range preserved) |
| `view(vec, 2:4)` | `SubArray` | ✅ (no data copy) |
| `CartesianIndices((5,5))[1:2,3:4]` | `CartesianIndices` (sub-grid) | ✅ |
| `CartesianIndices[2, 1:3]` | `Vector{CartesianIndex}` | ❌ scalar drops a dim |
| `(1:10)[[1,3,5]]` | `Vector` | ❌ gather materializes |

We mirror this table exactly, substituting "sub-`GriddedQuery`" for "sub-grid".

## Semantics

Let `gq::GriddedQuery{T,E,N}`.

| Index form | Result | Rationale |
|---|---|---|
| `gq[i₁,…,i_N]` all `Integer` | point `E` (scalar) | existing scalar accessor |
| `gq[I₁,…,I_N]` **all** `AbstractVector`/`Colon` | **lazy sub-`GriddedQuery`** | this design |
| `gq[…, scalar, …]` mixed scalar+vector | `Array{E,M}` (dim-dropped) | can't be an N-axis grid; Base default |
| `gq[k]` / `gq[2:4]` **linear** on N≥2 | point / `Vector{E}` | linear span crosses axis boundaries; Base default |
| `gq[k]` linear on **N=1** | identical to the per-axis form (1 axis) | consistent |

Only the **all-nonscalar N-index Cartesian** form is intercepted; everything else
keeps its current Base behavior.

## Core primitive: `view` per axis

The whole slice is one `ntuple` of per-axis `view`s:

```julia
Base.@propagate_inbounds function Base.getindex(
        gq::GriddedQuery{T,E,N}, I::Vararg{Union{AbstractVector,Colon}, N}
    ) where {T,E,N}
    @boundscheck checkbounds(gq, I...)
    return GriddedQuery(ntuple(d -> @inbounds(_sub_axis(gq.axes[d], I[d])), Val(N)))
end

@inline _sub_axis(ax, I) = view(ax, I)   # range→range, vector→SubArray, both 0-copy
@inline _sub_axis(ax, ::Colon) = ax       # whole axis: share the parent object, no wrapper
```

Why `view` (not `getindex`) for the axis selection — measured (`@allocated`):

| axis type | `ax[3:8]` | `view(ax, 3:8)` | `view` result type |
|---|---:|---:|---|
| `StepRangeLen` (range) | 0 B | **0 B** | range (preserved) |
| `Vector` | **80 B (copy)** | **0 B** | `SubArray` (no copy) |

`view` is uniformly zero-copy **and** preserves range-ness (so a sliced range axis
keeps the O(1)-locate fast path). `getindex` on a `Vector` axis would copy — the
hidden allocation to avoid. `Colon` returns the parent axis object directly (no
`SubArray` wrapper at all).

### Allocation budget (hidden-alloc audit)

For `gq[1:2, 3:8]` on a `(range, vector)` grid:

- per-axis `view`: **0 B** each (range slice is a range; vector slice is a
  `SubArray` header, no data copy) — Colon axes: 0 B (parent reused).
- the returned `GriddedQuery` struct: **one small immutable wrapper** holding the
  N-tuple of axis references (a `SubArray` is itself a small struct; no coordinate
  arrays are duplicated). If the result doesn't escape, escape analysis may stack-
  allocate it; if returned, it is a single small heap object.
- **No coordinate data is ever copied.** Total cost is O(N) view headers + one
  wrapper, independent of the number of sliced points.

Contrast: the current materializing path allocates the full `Matrix{E}` — O(∏
selected points).

## Fast-path re-entry

A sliced `SubArray`/range **target** axis is consumed by the gridded machinery as a
query-target vector (`_gridded_build_searcher(grid, targets)`, `first`/`last`/
`length`/iterate over `targets`) — never as the interpolant grid — so SubArray-ness
never touches the grid searcher. Verified: `itp(GriddedQuery((tx[3:20],
view(ty,3:20))))` runs the separable path and allocates only its output. Thus a
sliced sub-grid fed back to `interp`/`itp` is fast-path-eligible, unlike the
materialized `Matrix{Tuple}`.

## AbstractArray contract compliance

A sub-`GriddedQuery` is itself an `AbstractArray{E,N}` with:

- **same `eltype` `E`** — `view` preserves axis element types, so the point tuple
  type is unchanged (a slice preserving eltype is required by the array contract).
- **shape** = `map(length, sliced axes)` = the selected block's shape.
- **elements** equal to the selected points (`gq[1:2,3:4][i,j] == gq[i+0, j+2]`).

So returning it from `getindex(::AbstractVector…)` satisfies "indexing returns an
array of the selected elements" — it is simply lazy, exactly like `(1:10)[2:4]`.
Consequences to document: the result is **read-only** (no `setindex!`) and its
concrete type is `GriddedQuery`, not `Array` (same non-surprise as range slicing);
`collect`, broadcasting, `==`, and further indexing all still work.

## Inferrability

`view(axis_d, I_d)` has a type determined by the (compile-time) axis and index
types, `ntuple(…, Val(N))` is type-stable, and `GriddedQuery(sub_axes)` computes
`E`/`N` at compile time. The whole slice is inferrable and allocation-flat.

## `view()` hook (optional)

Because `getindex` already returns a lazy sub-grid, `Base.view(gq, I...)` is
somewhat redundant. For consistency we may add:

```julia
Base.view(gq::GriddedQuery{T,E,N}, I::Vararg{Union{AbstractVector,Colon},N}) where {T,E,N} =
    getindex(gq, I...)   # same lazy sub-grid
```

so `@view gq[1:2,3:4]` and `gq[1:2,3:4]` agree. Open question: keep both, or let
`view` fall through to Base's `SubArray` (which would wrap the *point array*, not
the grid)? Recommendation: add the hook so `view` never silently produces a
non-grid `SubArray{Tuple}`.

## Edge cases

- **Empty slice** `gq[1:0, :]` → sub-grid with a length-0 axis (size `(0, M)`),
  consistent with the empty-axis grid model.
- **N=1**: `gq[2:4]` is the per-axis form (single axis) → sub-grid over
  `view(axis, 2:4)`. A range-vs-linear ambiguity does not arise because N=1's
  linear and per-axis indices coincide.
- **0-D** (`GriddedQuery()`): no non-scalar index form applies; unaffected.
- **Integer-vector (gather)** `gq[[1,3], :]` → `view(axis, [1,3])` (SubArray
  holding the index vector; no coordinate-data copy). Still a valid sub-grid.
- **Colon everywhere** `gq[:, :]` → sub-grid sharing both parent axes (all axes
  returned as-is); equals `gq` elementwise, distinct object.
- **Bounds**: `@boundscheck checkbounds(gq, I...)` gives the standard N-D
  `BoundsError`; `view` would also throw, but the explicit check localizes it.
- **Non-1-based / OffsetArray axes**: `view` respects the axis's own indices; the
  same pre-existing caveat as scalar indexing applies (documented, not solved
  here).

## Test plan

- Return type: `gq[1:2,3:4] isa GriddedQuery`, `gq[:,2:3] isa GriddedQuery`,
  `gq[[1,3],:] isa GriddedQuery`.
- Laziness / no-copy: `@allocated gq[1:2,3:8]` stays at the small wrapper bound
  (no per-point growth); range axes stay ranges (`gq[1:2,:].axes[1] isa
  AbstractRange`); vector axes become views (`… isa SubArray`); Colon axis is the
  parent object (`gq[:,1:2].axes[1] === gq.axes[1]`).
- Correctness: `gq[1:2,3:4][i,j] == gq[i, j+2]`; `collect(gq[1:2,3:4]) ==
  collect(gq)[1:2,3:4]`; `size`/`eltype` preserved.
- Fast-path re-entry: `itp(gq[1:2,3:4])` equals `itp` over the materialized block
  (≈, FP contract) and is 0-alloc warm in-place; `@which`/type shows the separable
  method.
- Dim-drop / linear still materialize: `gq[2,1:3] isa Vector`, `gq[2:4] isa
  Vector` (N≥2).
- Inference: `@inferred gq[1:2,3:4]`.
- Read-only: mutation of the sub-grid throws.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|---|---:|---:|---|
| Slice return-type change surprises array-expecting callers | M | L | Documented; identical to `(1:10)[2:4]`; `collect` recovers an `Array` |
| `SubArray` axis slows some fast-path arm | M | L | Verified separable path unaffected; add an alloc/perf test |
| Ambiguity vs scalar/linear `getindex` methods | M | L | `Vararg{Union{AbstractVector,Colon},N}` is disjoint from `Vararg{Int,N}` and the 1-arg linear form |
| Escape-analysis fails → wrapper heap-allocs in a hot loop | L | L | One small struct; document; recommend the axis-tuple form for hot reuse |
| `view` on exotic axis types (rare) behaves unexpectedly | L | L | Falls back to Base `view` semantics; covered by axis-type tests |

## ⚠️ Review finding (Codex + empirical prototype): Aqua ambiguity blocker

Making `GriddedQuery{T,E,N} <: AbstractArray{E,N}` means `GriddedQuery{…,1}` **is an
`AbstractVector`**, which structurally overlaps the package's extensive
`AbstractVector` method surface. `Test.detect_ambiguities(FastInterpolations)`
reports **16 ambiguities on this branch (base: 0)** — a hard `Aqua.test_all`
failure (test/test_aqua.jl):

- **12** from the 1-D×`GriddedQuery` dispatch (`(itp::AbstractInterpolant1D)(gq::GriddedQuery{,,1})`)
  vs the per-type anchored-query batch callables `(itp)(::AbstractVector{<:_…AnchoredQuery})`.
- **4** from the AbstractArray migration itself: one-shot
  `linear_interp/constant_interp(grids, data, gq::GriddedQuery{<:NTuple{N}})` vs
  `(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real})`.
- The **slice `getindex`** proposed here adds **1 more**: `Vararg{Int,N}` (scalar) vs
  `Vararg{Union{AbstractVector,Colon},N}` (slice) both accept the **zero-arg** call at
  `N=0` (`gq[]` on a 0-D grid).

These are **latent** — the intersecting types are unconstructable (a `GriddedQuery`
element is always a coordinate tuple, never `<:Real`/`<:_AnchoredQuery`), so no real
call is ambiguous and runtime is correct — but `Aqua.test_ambiguities` flags them.
The package already carries "N=0 Aqua disambiguator" methods for exactly this class
of latent overlap, so there is precedent for the fix.

**Prototype validation (independent of the ambiguity issue):** the slice design's
substance holds — `gq[1:2,3:4]`/`gq[:,3:4]` intercept correctly and return a
sub-`GriddedQuery`; `gq[2:4]` (2-D linear) and `gq[2,1:3]` (mixed) correctly fall
through to Base materialization; `@allocated` is **0 B** warm (view-based, no data
copy; Colon returns the parent axis); `@inferred` passes; and `itp(subgrid)` re-enters
the separable fast path with matching values.

### Resolution options

1. **Keep AbstractArray + add N=1 (and N=0-slice) disambiguator methods** — ~8
   one-line forwarding methods (Linear/Constant/Quadratic × {1-arg, 2-arg} anchored,
   + `linear_interp`/`constant_interp` N=1 one-shot, + a 0-D `getindex`), mirroring the
   existing N=0 disambiguator pattern. Keeps free slicing/broadcast/generics.
   Maintenance: the disambiguator set grows if new `AbstractVector` interpolant
   overloads are added. **(Recommended.)**
2. **Drop the AbstractArray subtype** — revert to the plain shaped-collection
   `GriddedQuery` and add explicit slice/`getindex` methods by hand. Zero ambiguity
   debt, but loses the free array generics that motivated the migration.

## Open questions for review

1. Should `view(gq, …)` be hooked (recommended) or left to Base?
2. Should `Colon` return the parent axis object (`===`, proposed) or `view(ax,:)`
   for type uniformity across axes?
3. Any fast-path arm (constant/hermite/cubic gridded) that assumes `Vector`/range
   **target** axes and would mis-handle a `SubArray` target? (Linear verified.)
4. Is intercepting only the all-nonscalar Cartesian form the right boundary, or
   should mixed scalar+vector also return a sub-grid via a length-1 axis (rejected
   here as violating dim-drop expectations)?
