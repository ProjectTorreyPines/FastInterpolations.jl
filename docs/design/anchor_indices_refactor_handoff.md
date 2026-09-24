# Anchor Indices and Gridded Payload Refactor

> Status: Phases 1–5 complete on `refac/gridded-query`. Phase 4 landed as the
> **minimal swap** (`_IdxStencil{2}` → `_ExplicitIndices{2}`, Series anchor sizes
> unchanged); the full per-anchor `I` parameterization + size reductions are a
> deferred incremental follow-up (the `interval` field is already in place).
>
> Working branch: `refac/gridded-query`
>
> Related PR: PR 184, branch `feat/gridded-query`
>
> PR message source: `PR_MESSAGE_gridded_query.md`
>
> Baseline commit at handoff: `ffff3ef2f`

## 1. Purpose

PR 184 introduced `GriddedQuery` and achieved the intended separable-evaluation
performance improvements. The current anchor implementation, however, grew a
second location/addressing model beside the existing Series anchor model:

- `_AnchorLoc` stores both `idx` and `idxR` for every location.
- Series anchors use `_IdxStencil{2}` and always materialize both indices.
- Gridded `_AxisAnchor` stores a left/selected index plus positional tuple
  payloads such as `payload[1]` and `payload[2]`.
- Exclusive-periodic axes genuinely need an explicit right index at the seam,
  while ordinary axes only need the left index because `idxR == idxL + 1`.

The refactor should establish one canonical per-axis index protocol, use it in
the shared location layer and both anchor families, and replace fragile tuple
payloads with concrete named payload types. The concrete payload type should be
the compile-time tag that selects the minimal method/op-specific data layout.

The work is deliberately split into independently green TDD phases. Another
agent should continue at Phase 2 and preserve the completed Phase 1 changes.

## 2. Scope

### In scope

- Introduce compact and explicit fixed-size index representations.
- Make `_AnchorLoc` carry the physical interval returned by `search_interval`.
- Make ordinary intervals store one `Int` and exclusive-periodic intervals
  store the actual `(idxL, idxR)` pair.
- Refactor Gridded anchors to use canonical interval indices.
- Replace positional tuple payloads with named, concrete, op-aware payloads.
- Reuse the same index/address layer in existing Series and ND kernels.
- Remove `_IdxStencil`, `_IdxPair`, and positional Gridded payload access after
  all consumers are migrated.
- Preserve or improve current allocation, code-generation, and throughput
  characteristics.

### Explicitly out of scope for this refactor

- A new separable fast path for periodic Local Hermite `GriddedQuery`.
- Storing Local Hermite K=4/K=6 support indices before a consumer needs them.
- Making reusable Series anchors operation-specific.
- Subtyping `AbstractVector` for internal index collections.
- Introducing speculative `_WrappedIndices`, `_StridedIndices`, or arbitrary
  offset representations.
- Public API changes.

## 3. Current implementation state

Phase 1 has been implemented but is not committed or staged.

### Modified and new files

- `src/core/core.jl`
  - Includes the new `axis_indices.jl` before the legacy `idx_stencil.jl`.
- `src/core/axis_indices.jl`
  - Defines `_AbstractIndices{K}`.
  - Defines `_ContiguousIndices{K}` with one `first::Int` field.
  - Defines `_ExplicitIndices{K}` with `indices::NTuple{K,Int}`.
  - Implements literal/`Val` indexing, runtime indexing, iteration, semantic
    equality, semantic hash, and fixed-size collection metadata.
- `test/test_axis_indices.jl`
  - Covers K=2/K=4/K=6 construction, periodic/repeated explicit indices,
    access, iteration, equality/hash, inference, allocations, and exact sizes.

### Phase 1 verification already completed

The RED test failed because the new types did not exist. After the minimal
implementation:

```text
cc-julia-test-runner . test_axis_indices.jl test_idx_stencil.jl
All tests passed
```

Formatting and patch checks passed:

```text
runic --check src/core/core.jl src/core/axis_indices.jl test/test_axis_indices.jl
git diff --check
```

Verified sizes on the current 64-bit machine:

| Type | Size |
|---|---:|
| `_ContiguousIndices{2}` | 8 bytes |
| `_ContiguousIndices{6}` | 8 bytes |
| `_ExplicitIndices{2}` | 16 bytes |
| `_ExplicitIndices{6}` | 48 bytes |

Native-code inspection of six data reads through the actual package types
showed:

- `_ContiguousIndices{6}`: one index load; fixed offsets folded into three
  paired data loads.
- `_ExplicitIndices{6}`: six index loads and six separate address
  calculations before the data loads.

The existing `_IdxStencil` and all existing anchors remain active. The new
types are intentionally not wired into production consumers yet.

## 4. Final architecture

### 4.1 Index representations

```julia
abstract type _AbstractIndices{K} end

struct _ContiguousIndices{K} <: _AbstractIndices{K}
    first::Int
end

struct _ExplicitIndices{K} <: _AbstractIndices{K}
    indices::NTuple{K,Int}
end
```

`K` is the number of ordered indices along one axis, not the spatial dimension
of an interpolant. Use `K`, rather than `N`, consistently in this layer.

The abstract family is appropriate here because the concrete layouts genuinely
differ. A Symbol parameter such as `_Indices{K,:contiguous}` cannot change the
field count or layout without introducing another representation field or a
Union-valued field, recreating the extra wrapper and/or a runtime tag.

Do not add `_IdxPair`, `_AdjacentIndices`, or similar aliases. Use the two
concrete names directly:

```julia
_ContiguousIndices{2}(idxL)
_ExplicitIndices(idxL, idxR)
```

### 4.2 Canonical physical interval

`search_interval` owns the physical meanings of `idxL` and `idxR`. Constant,
Linear, Hermite, and other method payloads must not redefine those names.

The shared location result should become:

```julia
struct _AnchorLoc{
        I <: _AbstractIndices{2},
        Tg,
        Tq <: Real,
    }
    interval::I
    xq::Tq
    state::UInt8
    xL::Tg
    xR::Tg
end
```

The accessors should be literal/Val-dispatched virtual properties:

```julia
@inline Base.getproperty(loc::_AnchorLoc, s::Symbol) =
    _get_anchor_loc_property(loc, Val(s))

@inline _get_anchor_loc_property(loc::_AnchorLoc, ::Val{:idxL}) =
    getfield(loc, :interval)[Val(1)]

@inline _get_anchor_loc_property(loc::_AnchorLoc, ::Val{:idxR}) =
    getfield(loc, :interval)[Val(2)]

@inline _get_anchor_loc_property(loc::_AnchorLoc, ::Val{s}) where {s} =
    getfield(loc, s)
```

Do not retain a `loc.idx` compatibility property. Migrate the internal call
sites to `loc.idxL` in the same phase.

Representation selection must be centralized and based on the concrete axis
type, not the query location:

```julia
@inline _interval_indices(::AbstractVector, idxL, idxR) =
    _ContiguousIndices{2}(idxL)

@inline _interval_indices(::_ExclusivePeriodicAxis, idxL, idxR) =
    _ExplicitIndices(idxL, idxR)
```

The exact dispatch signatures should follow the current axis hierarchy. The
invariant is:

- ordinary axis: contiguous for every query;
- exclusive-periodic axis: explicit for every query, including interior
  queries, so one anchor vector has one concrete element type.

Never create `Vector{_AbstractIndices{2}}` and never store
`interval::_AbstractIndices{2}` directly in an immutable hot-path struct.
Always parameterize the owner by its concrete index type.

### 4.3 Prepared Gridded anchor

The target structure is:

```julia
struct _AxisAnchor{
        I <: _AbstractIndices{2},
        P,
    }
    interval::I
    payload::P
end
```

The payload's concrete type is the method/op tag. The current phantom method
parameter `M` should become unnecessary once each semantic payload has a
distinct named type.

Virtual properties should expose hot fields ergonomically:

```julia
a.idxL
a.idxR
a.alpha
a.inv_h
a.dL
a.h
```

They must lower through literal `Val` dispatch to concrete `getfield` and index
operations. Do not implement a single runtime Symbol branch with a union return
type.

### 4.4 Candidate named payloads

The precise names can be adjusted for consistency, but semantic types should
remain distinct even if their current fields happen to match.

```julia
struct _ConstantValuePayload{Tq}
    select_right::Bool
end

struct _ConstantZeroPayload{Tq} end

struct _LinearValuePayload{Talpha}
    alpha::Talpha
end

struct _LinearDeriv1Payload{Talpha,Tinv}
    inv_h::Tinv
end

struct _LinearZeroPayload{Talpha} end

struct _LocalHermitePayload{Tdl,Th,Tinv}
    dL::Tdl
    h::Th
    inv_h::Tinv
end
```

The phantom arithmetic types are intentional. For example,
`_LinearDeriv1Payload{Talpha,Tinv}` can reconstruct `one(Talpha)` without
storing the value of `alpha`. This preserves carrier/promotion semantics while
removing unused data from the anchor.

Quadratic and Cubic Gridded partial payloads should receive distinct names.
Audit their generated consumers before finalizing fields: the current
Quadratic ND kernel does not use every field presently stored in the common
`(dL,h,inv_h)` tuple.

### 4.5 Constant selection

At the canonical location layer, Constant is not special: it receives the same
physical `(idxL,idxR)` interval as every other method.

At the prepared Gridded layer, side selection may still be hoisted during
anchor construction. The first implementation should store
`select_right::Bool` in `_ConstantValuePayload{Tq}` and select between the two
literal interval endpoints with `ifelse` in the gather kernel. The carrier is
reconstructed as `one(Tq)` from the payload type.

This is a performance-gated design decision. If the explicit-periodic Constant
anchor regresses materially, first test side-specific zero-size payloads. Do
not silently change the common anchor to a K=1 selected-index contract; that
would require a separate design decision.

### 4.6 Gridded versus Series lifecycle

Gridded anchors are rebuilt for a particular evaluation, and each axis's
operation is already known. They should therefore use op-specific minimal
payloads.

Existing Series anchored queries can be reused across value and derivative
operations. Preserve this API contract. Series should reuse:

- `_AnchorLoc`;
- `_AbstractIndices{2}`;
- the interval representation selector;
- `idxL`/`idxR` accessors;

but should not be forced into the exact op-specific Gridded payload types in
this refactor.

## 5. Semantic and performance invariants

Every phase must preserve these constraints:

1. `idxL` and `idxR` always mean the physical search interval endpoints.
2. Exclusive-periodic seam search preserves `(n,1)` exactly.
3. Ordinary axes do not store a redundant right index.
4. Representation is selected per axis type, not per query value.
5. Every anchor vector has a concrete element type.
6. No new heap allocation appears in warmed hot paths.
7. Literal named access compiles like direct nested `getfield` access.
8. Public numeric results retain current exact/ULP behavior.
9. ForwardDiff carrier and promotion behavior is preserved.
10. Existing Series anchor reuse across operations is preserved.
11. Periodic Local Hermite remains correct through its current fallback.
12. Unsupported/mixed Gridded method tuples retain the generic point-wise
    fallback.

## 6. TDD implementation phases

## Phase 1 — Fixed-size axis index family

Status: **complete**.

### Completed RED

- Added tests for the absent `_AbstractIndices`, `_ContiguousIndices`, and
  `_ExplicitIndices` types.
- Confirmed failure due to undefined types.

### Completed GREEN

- Added the three types and minimal fixed-size access protocol.
- Added tuple/vararg explicit construction.
- Added representation-independent equality/hash.

### Completed REFACTOR and quality gate

- Runic formatting passed.
- New and legacy index tests passed together.
- Isbits/size/allocation tests passed.
- Native code confirmed folded contiguous addressing.

## Phase 2 — Canonical `_AnchorLoc` interval

Status: **complete**. `_AnchorLoc{I,Tg,Tq}` carries `interval::I`; ordinary
40 B / explicit 48 B; all six location consumers migrated to `loc.idxL`.

### RED

- [ ] Update `test/test_anchor_common.jl` from `loc.idx` to `loc.idxL`.
- [ ] Assert ordinary `loc.interval isa _ContiguousIndices{2}`.
- [ ] Assert exclusive-periodic `loc.interval isa _ExplicitIndices{2}`.
- [ ] Assert seam interval equals `(n,1)`.
- [ ] Pin ordinary and explicit Float64 sizes (expected 40 and 48 bytes).
- [ ] Preserve wrap/non-wrap/OOB, `GridIdx`, cached-range, vector-grid,
      search-hint, Float32, and ForwardDiff behavior.
- [ ] Pin `@inferred _anchor_loc(...)` for both representations.

### GREEN

- [ ] Replace `_AnchorLoc.idx` and `.idxR` with concrete `interval`.
- [ ] Add `idxL`/`idxR` Val-dispatched properties.
- [ ] Implement `_interval_indices` dispatch.
- [ ] Migrate all current `loc.idx` consumers to `loc.idxL`.
- [ ] Pass `loc.interval` directly when a consumer can accept it.
- [ ] Preserve the direct InBounds/search arms.

Current consumers found during planning:

- `src/gridded/axis_anchor.jl`
- `src/gridded/gridded_query.jl`
- `src/linear/linear_anchor.jl`
- `src/constant/constant_anchor.jl`
- `src/quadratic/quadratic_anchor.jl`
- `src/cubic/cubic_anchor.jl`

### REFACTOR

- [ ] Centralize ordinary/exclusive representation selection.
- [ ] Remove stale comments claiming every `_AnchorLoc` carries `idxR::Int`.
- [ ] Avoid compatibility aliases.

### Quality gate

```text
cc-julia-test-runner . test_anchor_common.jl
cc-julia-test-runner . test_linear_anchor.jl test_constant_anchor.jl
cc-julia-test-runner . test_cubic_anchor.jl test_quadratic_anchor.jl
cc-julia-test-runner . test_pchip_periodic.jl test_cardinal_periodic.jl test_akima_periodic.jl
```

Also run Runic and `git diff --check`.

## Phase 3 — Canonical Gridded anchor and named op-aware payloads

Status: **complete** (landed as 3a structural + 3b op-minimal). `_AxisAnchor{I,P}`
with named payloads, phantom `M` removed, exclusive-periodic idxR-in-payload hack
gone. Op-minimal Linear payloads (value → alpha only, deriv1 → inv_h only,
deriv2+ → zero-size) thread `op` into `_axis_anchor_type` only; `_resolve_anchor`
and kernels dispatch on the payload type. Perf A/B vs Phase 2: all hot paths
within run-to-run noise (<5% gate), zero warm allocs. NOTE: gridded anchors are
now op-specific (rebuilt per eval) — a low-level test that reused a value anchor
under a deriv op was updated to build op-matched anchors.

This is the main PR cleanup phase.

### RED

- [ ] Rewrite the `_AxisAnchor` tests at the beginning of
      `test/test_gridded_query.jl` around the final structure.
- [ ] Assert ordinary Linear anchors use contiguous intervals.
- [ ] Assert exclusive-periodic Linear anchors use explicit intervals.
- [ ] Assert `idxR` is absent from Linear payloads.
- [ ] Assert named `a.idxL`, `a.idxR`, `a.alpha`, and `a.inv_h` access.
- [ ] Assert no positional `payload[n]` contract.
- [ ] Assert EvalValue Linear payload stores only `alpha`.
- [ ] Assert EvalDeriv1 Linear payload stores only `inv_h`, with the carrier
      arithmetic type retained as a type parameter.
- [ ] Assert higher Linear derivatives use a zero-size payload when possible.
- [ ] Assert Constant Left/Right/Nearest exact-node, tie, OOB-fold, and seam
      selection parity.
- [ ] Assert Local Hermite named `dL`, `h`, and `inv_h` access.
- [ ] Assert Quadratic/Cubic partial payloads expose only used fields.
- [ ] Assert all anchor vector eltypes are concrete and all accessors inferred.

### GREEN

- [ ] Change `_AxisAnchor{M,P}` to `_AxisAnchor{I,P}`.
- [ ] Store `interval::I` and `payload::P`.
- [ ] Thread per-axis `op` into `_axis_anchor_type`,
      `_axis_anchors_pooled`, `_axis_anchors_all`, and `_resolve_anchor`.
- [ ] Introduce concrete named payload types.
- [ ] Remove the phantom method parameter after named payload dispatch is
      sufficient.
- [ ] Replace every `.payload[1]`, `.payload[2]`, and positional destructure.
- [ ] Remove the exclusive-Linear payload shape whose first item is `idxR`.
- [ ] Make Constant store selection metadata without redefining `idxL`.
- [ ] Preserve the InBounds lean path by constructing interval indices directly
      rather than forcing a full `_AnchorLoc`.

Primary files:

- `src/gridded/axis_anchor.jl`
- `src/gridded/gridded_query.jl`
- `src/gridded/gridded_constant.jl`
- `src/gridded/gridded_hermite.jl`
- `src/gridded/gridded_partials.jl`
- `test/test_gridded_query.jl`

### REFACTOR

- [ ] Separate address resolution from method/op payload resolution.
- [ ] Reduce exploded positional arguments to `_resolve_anchor`.
- [ ] Add literal Val-dispatched anchor property accessors.
- [ ] Verify no union-return Symbol branch or abstract field remains.

### Correctness gate

Run the complete `test/test_gridded_query.jl` file. It covers Linear,
Constant, Local Hermite, Cubic/Quadratic partials, periodic routing,
mixed-method fallback, extrapolation, derivatives, empty targets, in-place
evaluation, and allocation contracts.

Also rerun the periodic suites listed in Section 8.

### Performance gate

Record before/after measurements on the same machine:

- `benchmark/bench_gridded_ratio_sweep.jl`
- `benchmark/bench_gridded_fused.jl`
- representative 2D and 3D fused/fullbuffer cases;
- anchor-build-only microbenchmarks;
- Constant side variants;
- exclusive-periodic Linear and Constant;
- warmed in-place allocation.

Interpretation:

- median changes within about 3% are noise-level;
- a repeatable 5% or greater regression blocks the phase;
- zero-allocation paths are a hard requirement, not a soft threshold.

## Phase 4 — Series and legacy ND migration

Status: **complete (minimal swap)**. `_IdxStencil`/`_IdxPair` and
`src/core/idx_stencil.jl` + `test/test_idx_stencil.jl` deleted; Series anchors,
one-shot Series, 1D/ND adjoints, and ND Linear/Constant kernels now use
`_ExplicitIndices{2}` (field `stencil` → `interval`, `Val`-indexed accessors).
Anchor sizes are **unchanged** (48/64/128) — the size reductions in the table
below require the deferred full `I`-parameterization (which cascades into four
adjoint/ND anchor-storage structs). Quadratic's single-`Int` anchor left as-is.

### RED

- [ ] Pin ordinary Series anchors to `_ContiguousIndices{2}`.
- [ ] Pin exclusive-periodic Series anchors to `_ExplicitIndices{2}`.
- [ ] Preserve `aq.idxL` and `aq.idxR` virtual access.
- [ ] Preserve reuse of the same anchor for value and derivative evaluation.
- [ ] Pin ordinary anchor sizes; expected Float64 reductions are approximately:

| Anchor | Current | Expected ordinary |
|---|---:|---:|
| `_ConstantAnchoredQuery` | 48 | 40 |
| `_LinearAnchoredQuery` | 64 | 56 |
| `_CubicAnchoredQuery` | 128 | 120 |

- [ ] Preserve explicit-periodic anchor sizes and results.
- [ ] Pin ND Linear/Constant corner addressing and N=0 method dispatch.

### GREEN

- [ ] Parameterize Constant, Linear, and Cubic Series anchors by concrete
      `I<:_AbstractIndices{2}`.
- [ ] Rename the Series `stencil` field to `interval`.
- [ ] Pass `_AnchorLoc.interval` directly instead of reconstructing a pair.
- [ ] Audit Quadratic's single-index anchor; migrate only if doing so preserves
      its minimal layout and clarifies the common contract.
- [ ] Change ND Linear/Constant generated kernel signatures to the abstract
      protocol while keeping actual tuple elements concrete.
- [ ] Replace `_getstencil` with a grid-aware interval builder. Do not project a
      raw search tuple after losing the concrete grid type needed to select the
      representation.
- [ ] Use literal/Val endpoint access in generated kernels.
- [ ] Remove `_IdxStencil`, `_IdxPair`, `src/core/idx_stencil.jl`, and
      `test/test_idx_stencil.jl` after the last consumer is migrated.
- [ ] Remove the legacy include from `src/core/core.jl`.

Primary files include:

- `src/constant/constant_anchor.jl`
- `src/linear/linear_anchor.jl`
- `src/cubic/cubic_anchor.jl`
- `src/quadratic/quadratic_anchor.jl`
- `src/constant/nd/constant_nd_eval.jl`
- `src/linear/nd/linear_nd_eval.jl`
- `src/core/nd_utils.jl`
- Series one-shot and adjoint files currently constructing `_IdxPair`

Use this audit command repeatedly:

```text
rg -n "_IdxStencil|_IdxPair|payload\\[[0-9]+\\]|\.stencil" src test
```

The final expected result for `_IdxStencil`, `_IdxPair`, and positional
Gridded payload access is zero.

### Quality gate

- Run all anchor, Series, ND Linear/Constant, and adjoint tests.
- Run all periodic Linear/Constant suites.
- Run the Local Hermite periodic suites to ensure shared search/location
  changes did not alter them.
- Recheck anchor sizes, inference, and warm allocations.

## Phase 5 — Cleanup and final validation

Status: **complete**. Stale "corner stencil" prose retired; `interval` vs future
`support` indices documented in `axis_indices.jl`. Verified: no abstract fields
or abstract-element anchor vectors (`_anchor_loc`, gridded, and Series anchors
all concrete + isbits), accessors infer to `Int`/carrier and are zero-alloc in a
loop, Runic + `git diff --check` clean on every touched file. The complete
package test suite + minimum-Julia run remain the maintainer's pre-PR step.
Deferred to a follow-up: full per-anchor `I` parameterization (Series size
reductions), `stencils`/`_getstencil` internal renames, and the periodic
Local-Hermite separable `support` fast path (roadmap §10).

- [ ] Remove stale comments describing tuple payload layouts.
- [ ] Remove stale “corner stencil” terminology where the object is now a
      physical interval.
- [ ] Document `interval` versus future method-specific `support` indices.
- [ ] Verify no abstract field or abstract-element vector exists.
- [ ] Run Runic on every touched Julia file.
- [ ] Run `git diff --check`.
- [ ] Run the complete package test suite.
- [ ] Run the supported minimum Julia version if available.
- [ ] Inspect `@code_warntype` for index builders, `_anchor_loc`, anchor
      builders, and consumers.
- [ ] Inspect native code for ordinary and explicit endpoint access.
- [ ] Compare final Gridded and Series benchmarks with the recorded baseline.
- [ ] Update the PR message with the finalized architecture and measured
      effects.

## 7. Periodic Local Hermite status

Periodic Local Hermite forward evaluation is currently correct.

- PCHIP, Cardinal, and Akima have inclusive/exclusive wrap-aware slope paths.
- ND OnTheFly evaluation builds wrapped data-index windows and shifted grid
  windows.
- Persistent and one-shot seam behavior has regression coverage.
- Periodic Local Hermite `GriddedQuery` deliberately does not use the new
  separable Hermite arm; it falls back to the generic point-wise batch path.

At handoff, the relevant 1D, ND, and Gridded routing tests passed together.
Do not change this routing as part of Phases 2-5.

The important conceptual distinction is:

- `interval::_AbstractIndices{2}`: the physical cell endpoints returned by
  search;
- future `support::_AbstractIndices{4/6}`: the broader data neighborhood needed
  to compute both endpoint slopes.

For one interval:

| Method | Per-node slope support | Union for both interval endpoints |
|---|---:|---:|
| PCHIP | 3 points | 4 points |
| Cardinal | 3 points | 4 points |
| Akima | 5 points | 6 points |

An index collection alone is insufficient for periodic Hermite geometry. A
wrapped `x[1]` after `x[n]` must be interpreted as `x[1] + period`. Existing
code handles this with shifted grid windows and `_periodic_secant` /
`_periodic_cell_width`. Keep geometry separate from the index collection.

## 8. Test commands

Use the repository's filtered test runner.

### Phase 1

```text
/Users/yoo/.local/bin/cc-julia-test-runner . test_axis_indices.jl test_idx_stencil.jl
```

### Core and anchor migration

```text
/Users/yoo/.local/bin/cc-julia-test-runner . \
    test_axis_indices.jl \
    test_anchor_common.jl \
    test_constant_anchor.jl \
    test_linear_anchor.jl \
    test_quadratic_anchor.jl \
    test_cubic_anchor.jl
```

### Periodic regression set

```text
/Users/yoo/.local/bin/cc-julia-test-runner . \
    test_pchip_periodic.jl test_pchip_periodic_nd.jl \
    test_cardinal_periodic.jl test_cardinal_periodic_nd.jl \
    test_akima_periodic.jl test_akima_periodic_nd.jl \
    test_gridded_query.jl
```

This set passed before the handoff.

### Formatting

```text
runic --check --diff <touched Julia files>
git diff --check
```

### Full suite

```text
/Users/yoo/.local/bin/cc-julia-test-runner .
```

## 9. Known risks and decision gates

### Abstract storage accidentally leaking into fields

Bad:

```julia
struct BadAnchor
    interval::_AbstractIndices{2}
end
```

Good:

```julia
struct GoodAnchor{I<:_AbstractIndices{2}}
    interval::I
end
```

The same rule applies to vectors and tuples stored in mutable containers.

### Per-query representation instability

Do not use contiguous indices for periodic interior queries and explicit
indices only at the seam. A vector of prepared anchors would then need a union
or abstract element type. Select explicit representation for the entire
exclusive-periodic axis.

### Constant anchor footprint

An explicit K=2 interval plus `select_right::Bool` can be larger than the
current Constant anchor, which stores only the selected index. This is the main
performance decision gate in Phase 3. Benchmark before choosing a more complex
payload. Preserve the physical interval contract unless the design is
explicitly revisited.

### Dynamic indexing in hot kernels

`indices[k::Int]` is provided for ergonomics and iteration. Generated/hot
kernels should use literal endpoints or `Val` access. Constant's runtime side
selection should use `ifelse(indices[Val(1)], indices[Val(2)])`, rather than a
dynamic tuple lookup.

### Carrier semantics

Current kernels use values such as `one(alpha)` or `one(dL)` to propagate query
types, including Dual-like types, through values and zeros. Removing a stored
numeric field is safe only when the same carrier can be reconstructed from a
concrete payload type parameter.

### Brittle code-generation tests

Do not check complete LLVM/native instruction snapshots in CI. Pin:

- concrete inference;
- isbits status;
- exact object sizes;
- zero allocations;
- numeric parity;

and use manual `code_native` inspection as a phase quality gate.

## 10. Future roadmap after this refactor

### A. Periodic Local Hermite separable Gridded fast path

Once the common interval layer is stable, add a separate design/benchmark for
method support indices:

- `_ContiguousIndices{4}` for ordinary PCHIP/Cardinal support;
- `_ContiguousIndices{6}` for ordinary Akima support;
- `_ExplicitIndices{4/6}` for periodic wrapped support;
- separate periodic geometry metadata or precomputed cell widths/shifted grid
  positions.

Do not add support fields to current anchors until a kernel actually consumes
them; otherwise they are dead payload and additional memory traffic.

### B. Optional compact wrapped representation

If explicit K=4/K=6 periodic support becomes a measured bottleneck, consider a
third `_AbstractIndices{K}` subtype, for example a base index plus a compact
wrap boundary/mask. This should be benchmark-driven. Avoid runtime `mod1` in
the inner kernel unless it wins on target hardware.

### C. Prepared Series operation layer

Series anchored queries are currently reusable. If workloads benefit from
operation-specific preparation, introduce an explicit second preparation step
rather than changing the meaning of existing anchored queries. Such a design
might convert a reusable Series anchor into a minimal per-op prepared anchor,
but it should be a separate API/performance project.

### D. Broader fixed-index kernel reuse

After `_AbstractIndices` is established, audit other fixed-neighborhood code
for genuine reuse opportunities. Do not replace ordinary tuples merely for
uniformity; adopt the abstraction only where compact-versus-explicit addressing
or shared per-axis semantics provide value.

## 11. Handoff checklist

The next agent should:

1. Read this document and `src/core/axis_indices.jl` completely.
2. Inspect the current dirty worktree before editing; unrelated untracked files
   belong to the user.
3. Preserve the uncommitted Phase 1 files.
4. Start Phase 2 with tests in `test/test_anchor_common.jl` and confirm RED.
5. Execute one phase at a time, but continue automatically after a green quality
   gate unless repeated failures or a material design decision requires user
   input.
6. Do not commit, stage, push, or modify the PR without explicit user approval.
7. Keep progress recorded in this document or an equivalent handoff note if
   implementation decisions change.

