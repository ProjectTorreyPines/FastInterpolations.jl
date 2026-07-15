# PLAN: Query-Shape-Preserving Forward Evaluation

> Status: Draft — implementation not started
> Owner: Codex + yoo
> Created: 2026-07-14
> Architecture: `docs/design/query_shape_preserving_evaluation.md`

## Overview

- **Goal**: Make all four forward-evaluation quadrants preserve query `size`:
  `(1-D / N-D) × (persistent / one-shot)`, including allocating and in-place
  forms, dedicated family APIs, and unified `interp` / `interp!`.
- **Supported shaped queries**:
  - 1-D `AbstractArray{<:Real}`.
  - N-D AoS `AbstractArray` whose elements are N-coordinate point objects.
  - N-D shaped SoA `NTuple{N,AbstractArray{<:Real}}` with identical per-axis
    sizes.
  - Existing Vector AoS/SoA as the one-dimensional special case.
  - Existing `GriddedQuery` Cartesian-product semantics, unchanged.
- **Non-goals**:
  - Series, adjoints, pre-anchored query buffers, and vector-calculus layouts.
  - Preserving query container type or nonstandard axes; allocating calls return
    a dense `Array` of the same `size`.
  - GPU execution or a numeric points-in-columns matrix encoding.
  - New AD rules for shaped array queries; existing scalar/vector AD must remain
    unchanged.
- **User-facing API / constraints**:
  - `result = f(query)` implies `size(result) == _query_size(query)` for every
    batch form.
  - `f(output, query)` requires `size(output) == _query_size(query)` for shaped
    methods; equal length with a different shape is rejected.
  - Scalar inputs remain scalar; vectors remain vectors.
  - N-D `AbstractVector{<:Real}` remains one coordinate point.
  - `(qx, qy, ...)` is pairwise SoA; `GriddedQuery(qx, qy, ...)` is the Cartesian
    product.
- **Performance targets**:
  - Existing vector in-place allocations remain identical (normally 0 B warm).
  - New dense/view shaped in-place calls are 0 B warm where the corresponding
    vector path is 0 B.
  - Allocating shaped calls add only the returned dense array over the existing
    vector method's inherent work.
  - No `vec`/`reshape` wrappers in hot paths.
  - No confirmed >10% regression under the repository's same-machine benchmark
    and re-verification policy.
- **Compatibility targets**:
  - Julia 1.10+.
  - Existing BC/extrap/search/derivative/coefficient semantics.
  - Zero Aqua method ambiguities.

## Architecture decisions

- **Decision 1: `_query_size` is the single output-shape protocol.**
  - Rationale: it already exists for `GriddedQuery` and is already consumed by
    the unified N-D one-shot allocator.
  - Implementation: ordinary `AbstractArray` reports `size(q)`; shaped SoA
    reports `size(first(q))`; non-array custom containers keep the flat default.
  - Risk / mitigation: this is an intentional return-shape change for matrix/
    higher-dimensional batches; document `vec(f(q))` as the flattening migration.

- **Decision 2: shaped SoA is included.**
  - Rationale: `(qx::Matrix, qy::Matrix)` already contains an unambiguous pairwise
    shape and is the natural extension of tuple-of-vectors SoA.
  - Constraint: all coordinate arrays must have identical `size`, not merely
    equal length.
  - Risk / mitigation: tuple dispatch currently assumes vectors; add explicit
    tuple-of-arrays protocol methods and dimensionality tests.

- **Decision 3: write shaped outputs directly by logical query number.**
  - Rationale: every batch kernel already uses query number `k`; `output[k]`
    preserves column-major alignment without a data copy or view object.
  - Alternative rejected: `vec`/`reshape` is zero-copy but feasibility probes
    retained 32–64 B wrappers in relevant calls.
  - Risk / mitigation: benchmark dense arrays and noncontiguous views; leave
    exotic axes/container preservation out of scope.

- **Decision 4: preserve existing public vector dispatch.**
  - Rationale: it isolates the established hot path and makes regression evidence
    attributable.
  - Implementation: add less-specific shaped public overloads; internal sinks may
    accept `AbstractArray` while still specializing on concrete `Vector`.
    Existing N-D `output::AbstractVector, queries` methods must nevertheless
    validate `_query_size(queries)`, not length, and need a narrow intersection
    method if the new array overload creates crossed-signature ambiguity.
  - Risk / mitigation: Aqua and `@which` routing tests after every phase.

- **Decision 5: one-shot shaped methods reuse kernels, not persistent builders.**
  - Rationale: building a persistent interpolant as an adapter can copy data,
    change pool/cache lifetime, and introduce allocations.
  - Implementation: add shaped output/query overloads to the existing one-shot
    preparation and family loops.

- **Decision 6: N=1 tuple-grid calls collapse to native 1-D shaped methods.**
  - Rationale: keeps tuple-grid and bare-grid APIs bit/allocation-identical and
    avoids treating scalar elements of a matrix as N-D point containers.

## Phase plan (TDD)

**CRITICAL**: Execute phases **one at a time**.

For each phase:
- 🔴 RED: tests first, confirm they fail for the intended reason
- 🟢 GREEN: minimal implementation for this phase only
- 🔵 REFACTOR: cleanup while staying green
- ✅ QUALITY GATE: required correctness, allocation, routing, and ambiguity gates

---

### Phase 1 — Query-shape protocol + persistent N-D quadrant

**Estimated size**: 3–4 hours

**Goal**: Every persistent N-D interpolant returns/writes the shape of AoS arrays
and shaped SoA arrays, while existing vector/scalar behavior remains unchanged.

**Primary files**:

- `src/core/query_protocol.jl`
- `src/core/interpolant_protocol.jl`
- `src/core/nd_utils.jl`
- `src/core/search.jl`
- `src/gridded/gridded_dispatch.jl`
- `test/test_query_shape_preservation.jl`

**Test strategy**:

- Unit tests:
  - `_query_size`, `_query_length`, `_query_extract`, `_query_eltype`, and
    `_query_validate` for Matrix/3-D AoS and Matrix SoA.
  - SoA mismatched `size` throws before evaluation.
  - custom non-array query default remains flat; `GriddedQuery` remains shaped.
- Integration tests:
  - persistent Constant, Linear, Quadratic, Cubic, user-Hermite, and Hetero
    interpolants on AoS Matrix and shaped SoA Matrix.
  - allocating result equals scalar `map` reference and has exact size.
  - in-place result writes to exact-size Matrix and returns it.
- Edge cases:
  - empty dimensions, 3-D query container, SVector AoS, mixed coordinate types,
    derivative ops, Clamp/Fill/Wrap, and same-length/wrong-shape output.
  - exact-size rejection for `Vector` output paired with a same-length Matrix
    query, which must not fall through the current length-only N-D method.
  - N-D `Vector{<:Real}` remains a scalar point.
  - `Vector{NTuple}` remains a vector batch.

**Tasks**:

- [ ] 🔴 RED: create `test/test_query_shape_preservation.jl` with query-protocol
  tests for AoS arrays and shaped SoA arrays; run and capture current flat/failing
  behavior.
- [ ] 🔴 RED: add persistent N-D allocating/in-place tests for Linear and Hetero
  as two protocol-wide representatives; confirm failures before implementation.
- [ ] 🔴 RED: add dispatch guards for scalar real vectors, vector AoS, and
  `GriddedQuery`.
- [ ] 🟢 GREEN: implement `_query_size(::AbstractArray) = size(q)` and non-empty
  tuple-of-arrays SoA protocol methods.
- [ ] 🟢 GREEN: validate shaped SoA using exact `size`; add size-specific error
  helpers without changing existing scalar error behavior.
- [ ] 🟢 GREEN: add shared dense output allocation and exact-size validation
  helpers.
- [ ] 🟢 GREEN: make the existing N-D `output::AbstractVector, queries` path use
  exact `_query_size`; route it and the new `output::AbstractArray, queries`
  method through a common helper. Add an intersection tie-breaker only if the
  final signatures constrain both arguments and Aqua requires it.
- [ ] 🟢 GREEN: make persistent N-D allocation use `_query_size`; add direct
  `AbstractArray` output sinks in `_nd_batch_pointwise!` / `_interp_nd_batch!`.
- [ ] 🟢 GREEN: remove `vec(out)` from the non-separable persistent
  `GriddedQuery` fallback only; keep separable dispatch unchanged.
- [ ] 🟢 GREEN: add shaped SoA monotonicity/domain fast checks where vector-only
  specializations would otherwise be lost.
- [ ] 🔵 REFACTOR: retain explicit vector public methods and centralize only
  shape-neutral setup/loop bodies.
- [ ] 🔵 REFACTOR: update query-protocol comments to define AoS, shaped SoA, and
  `GriddedQuery` semantics.

**Quality gate**:

- [ ] `cc-julia-test-runner . "query shape"` passes for Phase 1 items.
- [ ] Existing N-D batch, hint-persistence, GriddedQuery, domain, FillExtrap,
  mixed-precision, and duck-typing tests pass.
- [ ] Persistent Matrix AoS and Matrix SoA in-place calls allocate 0 B warm.
- [ ] Existing Vector AoS/SoA allocation tests remain unchanged.
- [ ] `Aqua.test_ambiguities(FastInterpolations)` reports zero ambiguities.
- [ ] No source changes for one-shot or 1-D behavior are included early.

**Rollback**:

- Revert Phase 1 edits in the five core/gridded files and the Phase 1 test items.
- No family-specific implementation files should need rollback in this phase.

---

### Phase 2 — One-shot N-D quadrant: unified + dedicated APIs

**Estimated size**: 3–4 hours

**Goal**: All N-D one-shot APIs preserve AoS/shaped-SoA query size with direct
shaped in-place output.

**Primary files**:

- `src/hetero/hetero_oneshot.jl`
- `src/linear/nd/linear_nd_oneshot.jl`
- `src/constant/nd/constant_nd_oneshot.jl`
- `src/quadratic/nd/quadratic_nd_oneshot.jl`
- `src/cubic/nd/cubic_nd_oneshot.jl`
- `src/hermite/nd/hermite_nd_oneshot.jl`
- `src/hetero/local_hermite_nd_forward.jl`
- `test/test_query_shape_preservation.jl`

**Test strategy**:

- Unit/integration tests:
  - unified `interp`/`interp!` and dedicated named APIs on shaped AoS/SoA.
  - Constant, Linear, Quadratic, Cubic, user `HermitePartials`, PCHIP,
    Cardinal, Akima, and a mixed Hetero method tuple.
  - unified/dedicated value, shape, eltype, and allocation parity.
- Edge cases:
  - `PreCompute`/`OnTheFly`/`AutoCoeffs`, periodic axes, FillExtrap,
    per-axis derivatives, empty query shape, and wrong output shape.
  - supported `GriddedQuery` still reaches the separable path; unsupported method
    tuples still use correct direct pointwise shaped fallback.

**Tasks**:

- [ ] 🔴 RED: add failing unified N-D one-shot AoS/SoA shape tests.
- [ ] 🔴 RED: add failing dedicated family tests, including user-Hermite and local
  Hermite forwarders.
- [ ] 🔴 RED: add same-length/wrong-shape output tests and assert output remains
  untouched when validation/domain checks fail.
- [ ] 🟢 GREEN: add exact output-size validation in generic unified `interp!`.
- [ ] 🟢 GREEN: remove the unified one-shot `vec(output)` adapter and pass shaped
  output directly into batch dispatch.
- [ ] 🟢 GREEN: widen/add family internal output sinks to `AbstractArray` while
  preserving vector-specialized public methods.
- [ ] 🟢 GREEN: make every dedicated N-D public/vector sink validate exact
  `_query_size`, including `Vector output × Matrix query` rejection.
- [ ] 🟢 GREEN: change each dedicated allocator to the shared shape allocator,
  preserving its existing output-eltype formula.
- [ ] 🟢 GREEN: widen PCHIP/Cardinal/Akima N-D forwarding methods to shaped
  output/query containers.
- [ ] 🔵 REFACTOR: deduplicate shape validation/allocation only; do not merge
  family coefficient/BC preparation.

**Quality gate**:

- [ ] Phase 1 and Phase 2 query-shape tests pass.
- [ ] Existing dedicated N-D one-shot and unified API suites pass.
- [ ] Warm shaped in-place calls have the same allocation count as corresponding
  vector calls.
- [ ] Unified/dedicated allocations are equal after warmup.
- [ ] GriddedQuery routing and zero-allocation tests pass.
- [ ] Aqua ambiguity checks pass.

**Rollback**:

- Revert Phase 2 family/unified one-shot edits and Phase 2 tests.
- Persistent N-D shaped behavior from Phase 1 remains independently useful.

---

### Phase 3 — Persistent 1-D quadrant

**Estimated size**: 3–4 hours

**Goal**: All persistent 1-D interpolants and their DerivativeViews accept shaped
real query arrays and preserve size.

**Primary files**:

- `src/core/interpolant_protocol.jl`
- `src/core/search.jl`
- `src/core/utils.jl`
- `src/linear/linear_interpolant.jl`
- `src/constant/constant_interpolant.jl`
- `src/quadratic/quadratic_interpolant.jl`
- `src/cubic/cubic_eval.jl`
- `src/hermite/hermite_eval.jl`
- `src/derivative_view.jl`
- `test/test_query_shape_preservation.jl`

**Test strategy**:

- Integration tests:
  - persistent Constant, Linear, Quadratic, Cubic, Hermite, PCHIP, Cardinal, and
    Akima on Matrix and 3-D real query arrays.
  - allocating/in-place parity with `map(itp, q)`.
  - `deriv1/2/3` and `deriv_view` allocating/in-place forwarding.
  - native shaped queries and the persistent one-axis SoA spelling `(q,)` have
    identical values, shapes, and allocations.
- Edge cases:
  - dense Matrix, noncontiguous view, empty dimensions, alias-exact query/output
    where the vector API already supports it, mixed precision, No/Clamp/Fill/
    Wrap, explicit/adaptive search, and external hint.

**Tasks**:

- [ ] 🔴 RED: add failing persistent 1-D shaped tests for all eight families.
- [ ] 🔴 RED: add warm allocation pins for dense Matrix and noncontiguous view;
  pin existing vector allocations in the same function barriers.
- [ ] 🔴 RED: add DerivativeView shaped in-place test and dispatch guard.
- [ ] 🟢 GREEN: add less-specific shaped allocating/in-place methods on
  `AbstractInterpolant1D`; leave vector methods unchanged.
- [ ] 🟢 GREEN: add `AbstractArray` batch domain checks, including empty arrays and
  `_CachedRange` exclusive-last promotion.
- [ ] 🟢 GREEN: add array AutoSearch prefix classification in logical order.
- [ ] 🟢 GREEN: add array-capable peers for Constant/Linear/Quadratic/Cubic/
  Hermite-family internal loops and function barriers.
- [ ] 🟢 GREEN: add shaped in-place DerivativeView forwarding.
- [ ] 🟢 GREEN: widen persistent one-axis tuple forwarders to `(q::AbstractArray,)`
  and exact-shape in-place output, reusing the native 1-D path.
- [ ] 🔵 REFACTOR: audit inference so extrap/searcher unions are resolved outside
  inner loops; avoid higher-order closures unless codegen proves zero-cost.

**Quality gate**:

- [ ] Persistent 1-D shaped tests pass across all families.
- [ ] Existing 1-D scalar/vector, periodic, mixed-precision, search, derivative,
  and AD extension tests pass.
- [ ] New dense/view in-place paths allocate 0 B warm.
- [ ] Existing vector allocation counts remain identical.
- [ ] Representative vector microbenchmarks are within noise before continuing.
- [ ] Aqua ambiguity checks pass, including GriddedQuery N=1/anchor methods.

**Rollback**:

- Revert Phase 3 protocol/domain/search/family-loop/DerivativeView edits and tests.
- N-D shaped behavior from Phases 1–2 remains intact.

---

### Phase 4 — One-shot 1-D core families

**Estimated size**: 2–4 hours

**Goal**: Dedicated Constant, Linear, Quadratic, Cubic, and user-Hermite one-shot
APIs preserve shaped query size without building persistent adapters.

**Primary files**:

- `src/constant/constant_oneshot.jl`
- `src/linear/linear_oneshot.jl`
- `src/quadratic/quadratic_oneshot.jl`
- `src/cubic/cubic_oneshot.jl`
- `src/hermite/hermite_oneshot.jl`
- `test/test_query_shape_preservation.jl`

**Test strategy**:

- Allocating and in-place Matrix/3-D query tests for all five families.
- Cubic raw-grid and `CubicSplineCache` entry points.
- output eltype parity with vector calls under Int/Float32/Float64/Dual-compatible
  type combinations already covered by the repo.
- BC/extrap/derivative/search parity and wrong-shape validation.

**Tasks**:

- [ ] 🔴 RED: add failing shaped one-shot tests for the five core families and
  cache/raw-grid Cubic entry points.
- [ ] 🔴 RED: add allocating-output byte parity and warm in-place allocation
  parity against vectors.
- [ ] 🟢 GREEN: add shaped in-place overloads that reuse existing axis/BC/cache/
  coefficient preparation and array-capable Phase 3 loops.
- [ ] 🟢 GREEN: add shaped allocating overloads using each family's unchanged
  output-eltype formula and shared shape allocator.
- [ ] 🔵 REFACTOR: keep vector public methods as the more-specific dispatch;
  centralize only allocation/shape checks.

**Quality gate**:

- [ ] Phase 4 shaped tests pass.
- [ ] Existing Constant/Linear/Quadratic/Cubic/Hermite one-shot suites pass.
- [ ] In-place allocation parity with vector methods passes after warmup.
- [ ] Cubic cache/pool/thread-safety tests pass.
- [ ] Aqua ambiguity checks pass.

**Rollback**:

- Revert only the five Phase 4 one-shot files and Phase 4 test items.
- Persistent 1-D shaped behavior remains available.

---

### Phase 5 — One-shot 1-D local-Hermite + unified API + N=1 collapse

**Estimated size**: 3–4 hours

**Goal**: PCHIP/Cardinal/Akima, unified `interp`/`interp!`, and every tuple-grid
N=1 forward expose the same shaped 1-D contract.

**Primary files**:

- `src/pchip/pchip_oneshot.jl`
- `src/cardinal/cardinal_oneshot.jl`
- `src/akima/akima_oneshot.jl`
- `src/hetero/interp_1d.jl`
- `src/linear/nd/linear_nd_interpolant.jl`
- `src/constant/nd/constant_nd_interpolant.jl`
- `src/quadratic/nd/quadratic_nd_interpolant.jl`
- `src/cubic/nd/cubic_nd_interpolant.jl`
- `src/hetero/local_hermite_nd_forward.jl`
- `test/test_query_shape_preservation.jl`

**Test strategy**:

- Dedicated PCHIP/Cardinal/Akima allocating/in-place shape tests under
  `AutoCoeffs`, `PreCompute`, and `OnTheFly` as supported.
- Unified `interp`/`interp!` parity with every routable 1-D method object.
- Bare-grid and one-axis tuple-grid parity for direct `q::AbstractArray` and
  single-axis SoA `(q,)`.
- Periodic, tension, side, BC, deriv, and extrap keyword forwarding.

**Tasks**:

- [ ] 🔴 RED: add failing shaped one-shot tests for PCHIP/Cardinal/Akima.
- [ ] 🔴 RED: add unified/dedicated shape, value, output-eltype, and allocation
  parity tests for all routable method objects.
- [ ] 🔴 RED: add N=1 tuple-grid parity and dispatch-collision tests for all named
  families, including `(q,)` forwarding.
- [ ] 🟢 GREEN: add local-Hermite shaped allocating/in-place overloads using the
  Phase 3 Hermite loop sinks.
- [ ] 🟢 GREEN: add shaped unified 1-D methods through `_interp1d_route`; preserve
  wrapper zero-cost allocation parity.
- [ ] 🟢 GREEN: add explicit N=1 shaped collapse forwards and unwrap per-axis
  keyword tuples exactly as current vector forwards do.
- [ ] 🔵 REFACTOR: consolidate repeated N=1 forwarding patterns only if Aqua and
  inference remain clean; prefer explicit narrow methods over broad ambiguous ones.

**Quality gate**:

- [ ] All four API quadrants now pass the complete shape test matrix.
- [ ] Existing unified 1-D allocation-parity tests remain exact.
- [ ] Existing PCHIP/Cardinal/Akima coefficient/periodic tests pass.
- [ ] Tuple-grid N=1 and native 1-D results are bit-identical where the current
  vector contract is bit-identical.
- [ ] Warm in-place unified and dedicated shaped calls have equal allocations.
- [ ] Aqua ambiguity checks pass.

**Rollback**:

- Revert Phase 5 local-Hermite/unified/N=1 forwarding edits and tests.
- Core-family one-shot support from Phase 4 remains independently valid.

---

### Phase 6 — Full compatibility, performance, and documentation gate

**Estimated size**: 3–4 hours

**Goal**: Close the feature with full-suite, supported-version, allocation,
benchmark, method-table, and documentation evidence.

**Primary files**:

- `benchmark/ci_benchmark.jl`
- `docs/src/nd/overview.md`
- `docs/src/guides/performance_tips.md`
- relevant 1-D/ND API docstrings
- release notes for the target release
- this plan's Notes / learnings section

**Test strategy**:

- Full test suite on primary Julia and Julia 1.10.
- Aqua all checks.
- Benchmark matrix:
  - existing Vector AoS/SoA sentinels;
  - 1-D Matrix and noncontiguous view;
  - N-D AoS Matrix and shaped SoA Matrix;
  - persistent and one-shot;
  - Range/Vector grids;
  - sorted/random logical order;
  - Linear plus Cubic or PCHIP.
- Documentation examples execute and distinguish pairwise shaped SoA from
  Cartesian-product `GriddedQuery`.

**Tasks**:

- [ ] 🔴 RED: perform a final public method-table and docs audit; add tests first
  for any uncovered callable or ambiguity before changing it.
- [ ] 🟢 GREEN: add CI benchmark cases and baseline names for vector and shaped
  paths.
- [ ] 🟢 GREEN: update docstrings that say “Returns a Vector” and all 1-D/ND batch
  examples.
- [ ] 🟢 GREEN: add release-note migration guidance for callers relying on flattened
  matrix-of-points output.
- [ ] 🔵 REFACTOR: remove stale flattening comments and ensure no hot path calls
  `vec`/`reshape` merely to satisfy vector signatures.
- [ ] ✅ QUALITY GATE: run full suites, allocation audit, and same-machine
  benchmark with re-verification; record commands/results below.

**Quality gate**:

- [ ] Full package test suite passes.
- [ ] Julia 1.10 targeted/full suite passes.
- [ ] Aqua passes with zero ambiguities.
- [ ] Existing vector allocation counts are identical.
- [ ] New shaped in-place dense/view paths meet 0 B warm targets.
- [ ] No confirmed benchmark regression exceeds 10% after re-verification.
- [ ] User docs and release notes fully state the new shape contract.

**Rollback**:

- Docs/benchmarks are independently revertible.
- A performance failure blocks completion and rolls back the responsible earlier
  phase, not merely the benchmark assertion.

## Required test matrix

The implementing agent must treat this as a completion checklist, not optional
coverage.

| Dimension | Lifetime | API | Query layout | Allocating | In-place |
|---|---|---|---|---:|---:|
| 1-D | persistent | callable | real Matrix / 3-D array / view | [ ] | [ ] |
| 1-D | persistent | N=1 SoA forward | `(Matrix,)` | [ ] | [ ] |
| 1-D | one-shot | dedicated core families | real Matrix / 3-D array / view | [ ] | [ ] |
| 1-D | one-shot | PCHIP/Cardinal/Akima | real Matrix / view | [ ] | [ ] |
| 1-D | one-shot | unified `interp` | real Matrix / view | [ ] | [ ] |
| 1-D | one-shot | N=1 tuple-grid forwards | real Matrix / `(Matrix,)` | [ ] | [ ] |
| N-D | persistent | callable | AoS Matrix / SVector Matrix | [ ] | [ ] |
| N-D | persistent | callable | shaped SoA matrices | [ ] | [ ] |
| N-D | one-shot | dedicated named | AoS / shaped SoA | [ ] | [ ] |
| N-D | one-shot | unified `interp` | AoS / shaped SoA | [ ] | [ ] |
| N-D | either | `GriddedQuery` | independent target axes | [ ] | [ ] |
| 1-D/N-D | persistent | `DerivativeView` | shaped query | [ ] | [ ] |

## Risks

| Risk | Impact | Likelihood | Mitigation | Detection |
|---|---:|---:|---|---|
| Broad array overload intersects GriddedQuery/anchors | H | M | Real-element constraints and narrow tie-breakers | Aqua + `@which` tests |
| N-D real vector changes from scalar point to batch | H | L | keep existing method more specific | Optim/ForwardDiff dispatch guards |
| shaped SoA accidentally treated as scalar Tuple | H | M | explicit tuple-of-arrays protocol | extraction/ndims tests |
| `vec` wrapper adds heap allocation | M | H | direct shaped output sinks | exact warm `@allocated` |
| existing Vector codegen slows | H | L | retain public vector methods, benchmark sentinels | CI benchmark + allocation parity |
| NoExtrap writes before throwing | H | L | preserve whole-batch pre-scan | sentinel-output tests |
| family allocator changes output eltype | H | M | keep every existing `Tout` formula | eltype parity matrix |
| N=1 tuple-grid falls into generic N-D batch | H | M | explicit shaped collapse forwards | bit/allocation parity tests |
| same-length wrong-shape output silently accepted | M | M | exact-size helper | mismatch tests |
| custom query users relied on flat array default | M | M | intentional compatibility note; `vec` migration | release notes |
| exotic IndexCartesian container is slow | M | M | dense/view benchmark scope; custom axes non-goal | benchmark matrix |
| one-shot adapter builds persistent state | H | L | prohibit constructor adapter in design | allocation/code review |

## Progress tracking

- Phase 1: ☑ planned ☑ in progress ☑ done
- Phase 2: ☑ planned ☑ in progress ☑ done
- Phase 3: ☑ planned ☑ in progress ☑ done
- Phase 4: ☑ planned ☑ in progress ☑ done
- Phase 5: ☑ planned ☑ in progress ☑ done
- Phase 6: ☑ planned ☑ in progress ☑ done (impl/docs/review; full-suite + Julia 1.10 + benchmark = user)

## Notes / learnings

- Current `src/core/query_protocol.jl` already defines `_query_size`, but its
  generic result is flat; only `GriddedQuery` overrides it.
- Unified N-D one-shot allocation already consumes `_query_size`, while
  persistent N-D and dedicated N-D one-shot allocators still construct vectors.
- Existing N-D pointwise kernels already evaluate by logical `k`, so shaped
  output needs no numerical-kernel change.
- 1-D persistent family traits are centralized, but their core loops and the
  one-shot public signatures are vector-restricted.
- `DerivativeView` already forwards shaped out-of-place queries; only shaped
  in-place forwarding is missing.
- Tuple-grid N=1 forwarders are explicitly vector-restricted in every family and
  must be part of the 1-D one-shot completion gate.
- Local Julia 1.12.6 feasibility probes showed direct Matrix loops at 0 B/call;
  `vec` adapters retained 32–64 B. Release claims require Phase 6 benchmarks.
- `docs/plans` is a symlink outside the workspace. This repository-local plan is
  intentionally stored under `docs/design`.

## Implementation results (fill during execution)

### Phase 1 — DONE (protocol + persistent N-D quadrant)

- **Branch/worktree**: `feat/query-shape-preserving` @ `~/.julia/dev/FastInterpolations-query-shape` off master `a1bed78e7`.
- **Source edits (3 files, runic-clean)**:
  - `src/core/query_protocol.jl`: `_query_size(::AbstractArray)=size`, SoA-array peers for
    `_query_length/_query_size/_query_extract/_query_eltype/_query_validate/_query_check_ndims`
    (the last two MANDATORY — a tuple-of-matrices otherwise throws a spurious ndims error), plus
    `_alloc_query_output`/`_check_query_output_size` + `_throw_query_axis_size_mismatch`/`_throw_query_output_size_mismatch`.
    Vector-SoA methods KEPT more-specific (bit-identical hot path).
  - `src/core/interpolant_protocol.jl`: widened `_interp_nd_batch!`/`_nd_batch_pointwise!` sinks to
    `AbstractArray`; in-place = `AbstractArray`+`AbstractVector` peers through one `_nd_batch_inplace!`
    with exact `_check_query_output_size` (replaces the length-only check → rejects vector-sink-for-matrix-query);
    allocation via `_alloc_query_output` (matrix query → matrix; vector query → `Vector` unchanged).
  - `src/gridded/gridded_dispatch.jl`: removed the `vec(out)` non-separable fallback (sink now `AbstractArray`).
- **Tests**: `test/test_query_shape_preservation.jl` — 6 testitems, 111 assertions, all pass. Protocol 15,
  SoA validation+ndims 6, persistent N-D 5-family 60, dispatch guards 8, edge cases (deriv/NoExtrap-sentinel/
  Clamp/view/mixed-precision × Linear+Cubic) 18, allocation parity 4.
- **Allocation**: all four in-place lanes (AoS matrix, SoA matrix, AoS vector, SoA vector) = 0 B warm,
  measured inside a function barrier. A/B vs master (`~/.julia/dev/FastInterpolations-master-ab`) confirms
  SoA-vector in-place is 0 B on both — the 32 B a first (barrier-less) draft saw was a boxed-local
  measurement artifact, not code allocation.
- **Aqua**: `Method ambiguity` check passes (0 ambiguities) — the broad `AbstractArray` overloads add no
  ambiguity vs GriddedQuery/anchors/SoA, as the pre-implementation verification predicted.
- **Regression**: ND In-Place Batch 42/42, GriddedQuery matrix 86/86 + all GriddedQuery-collection/zero-alloc
  testitems, hint-persistence, domain/promotion/clamp, mixed-precision, duck-typing (624) — all green.
- **Deviation from design**: §7.2 SoA-array fast-check peers (`_validate_nd_domain`/`_check_mono_nd` for
  `Tuple{Vararg{AbstractArray{<:Real},N}}`) DEFERRED to Phase 6 perf gate. Rationale: they are a SPEED
  optimization (min/max scan vs per-query), NOT allocation — SoA-matrix already hits 0 B via the generic
  path (same one AoS uses). "Measure first": add only if Phase 6 shows the SoA-matrix path >10% slower than
  SoA-vector. Avoids unmeasured method surface + a re-Aqua cycle.

### Phase 2 — DONE (one-shot N-D quadrant)

- **Source edits (7 files, runic-clean)**: `hetero/hetero_oneshot.jl` (unified: hetero core sink →
  `AbstractArray`; exact-size gate at the public entry, SKIPPED for GridIdx queries; `vec(output)` adapter
  removed from the main dispatch, kept locally only for the vector-typed GridIdx branch);
  `{linear,constant,quadratic,cubic,hermite}/nd/*_nd_oneshot.jl` (batch-core sinks → `AbstractArray`,
  `length`-check → `_check_query_output_size`, allocators → `_alloc_query_output`); `hetero/local_hermite_nd_forward.jl`
  (3 ND in-place forwarders → `AbstractArray`; allocating forwarders already preserved shape via the unified path).
- **Tests**: one-shot unified 7, dedicated core families 24 (incl. cubic in-place), local-Hermite + user-Hermite
  23, allocation parity 6 — all green.
- **Allocation**: every shaped one-shot in-place lane (unified + dedicated, AoS matrix + shaped SoA) = 0 B warm,
  matching the vector lanes (A/B standalone confirms). The `vec` removal was necessary for the unified 0-B target.
- **Aqua**: zero new ambiguities. **Regression**: hetero_oneshot, cubic_nd_oneshot, hermite_nd_partials,
  local_hermite_nd_forward, calltime_extrap_override, nd1_collapse (incl. GridIdx) — all green.
- **Deviation/finding**: GridIdx queries `(q, GridIdx(i))` mix free arrays with pinned scalar indices, so
  `_query_size` reports the tuple arity (wrong); the top-level exact-size gate is skipped for them (their
  pre-slice sub-problem self-validates), preserving master's GridIdx behavior. Full GridIdx shape support is
  out of scope (adjacent to the §12 pre-anchored-query non-goal). Caught by the existing nd1_collapse regression pin.

### Phase 3 — DONE (persistent 1-D quadrant + DerivativeView)

- **Source edits (9 files, runic-clean)**: `core/interpolant_protocol.jl` (less-specific
  `(itp)(q::AbstractArray{<:Real})` allocating + `(out::AbstractArray, q::AbstractArray{<:Real})` in-place with
  exact-size gate; shaped `(q_matrix,)` single-axis SoA forwarders); `core/utils.jl` + `core/search.jl` (batch
  domain `_check_domain`/`_is_all_inbounds`/`_throw_batch_oob` and search `_is_likely_monotone`/
  `_resolve_search_policy` widened `AbstractVector{<:Real}`→`AbstractArray{<:Real}` — annotation-only);
  5 family eval loops (`linear/constant/quadratic/cubic/hermite`) widened `output`/`xq` to `AbstractArray`;
  `derivative_view.jl` (shaped in-place forward for 1-D parents + N-D `(AbstractArray, queries)` forward + the
  three ND-specialized tie-breakers that keep it unambiguous).
- **Tests**: persistent 1-D shape 88 (11 × 8 families: Matrix/3-D/view/exact-size/N=1-SoA), DerivativeView
  deriv1/2/3 shaped 15, allocation parity 3 — all green.
- **Allocation**: shaped 1-D in-place (dense Matrix + noncontiguous view) = 0 B warm; vector lane unchanged.
  test_allocation.jl 155/155 confirms the hot-path widening is compile-time-identical for the Vector path.
- **Aqua**: zero ambiguities (the DerivativeView tie-breaker set is the design's under-specified §4.1 gap, resolved).
  **Regression**: allocation, constant (incl. DerivativeView), hermite_1d, akima_1d, cardinal_1d — all green.
- **Design refinement (adj #1 & #6)**: the design's §4.1 named only two DerivativeView in-place forwards, which
  produce 3 Aqua ambiguities; added the full ND tie-breaker set (verified 0). Kept `eachindex(xq,output)` in the
  widened loops rather than switching to linear indexing (§5.4/adj #6) — it is 0-alloc and axes-correct for every
  in-scope case (dense/3-D/contiguous+noncontiguous view); offset axes stay a §12 non-goal either way.

### Phase 4 — DONE (one-shot 1-D core families)

- **Source edits (5 files, runic-clean)**: `{linear,constant,quadratic,cubic,hermite}/*_oneshot.jl` — public
  allocating/in-place sinks widened `AbstractVector`→`AbstractArray`; allocators → `_alloc_query_output`;
  `@assert length`/`@boundscheck` output-length check → `_check_query_output_size`. Linear's own one-shot loop
  widened; constant/quadratic/cubic/hermite reuse the Phase-3-widened persistent family loops; cubic covers both
  raw-grid and cache in-place entries + the periodic/bcpair batch helpers.
- **Tests**: one-shot 1-D core families 25 (5 families × Matrix shape / in-place / exact-size rejection) +
  allocation parity 4 — all green.
- **Allocation**: shaped 1-D one-shot in-place (dense Matrix) = 0 B warm, matching the vector lane (A/B standalone
  confirms both linear and cubic at 0 B — pool-amortized spline build).
- **Aqua**: zero new ambiguities. **Regression**: cubic, linear, quadratic, allocation suites — green.

### Phase 5 — DONE (local-Hermite one-shot + unified interp + N=1 collapse)

- **Source edits (10 files, runic-clean)**: `{pchip,cardinal,akima}/*_oneshot.jl` (public + OnTheFly/PreCompute
  helper sinks → `AbstractArray`, `@boundscheck` output check → `_check_query_output_size`, allocators →
  `_alloc_query_output`) + `core/coeff_policy.jl` (the AutoCoeffs query-length crossover
  `_resolve_coeffs(::AutoCoeffs, ::AbstractVector, ::AbstractArray)` widened — it drives OnTheFly-vs-PreCompute); `hetero/interp_1d.jl` (unified `interp`/`interp!` widened; delegate through
  `_interp1d_route` so the dedicated methods size-check); N=1 collapse forwarders in
  `{linear,constant,quadratic,cubic}/nd/*_nd_interpolant.jl` + `hetero/local_hermite_nd_forward.jl` — the
  QUERY widens (`AbstractVector{<:Real}`→`AbstractArray{<:Real}`, `Tuple{AbstractVector}`→`Tuple{AbstractArray}`)
  and the in-place output widens, GRID stays vector-only (`only(grids)` → genuine 1-D). For Cubic this also
  fixes an AutoCoeffs matrix query that previously mis-routed to the generic `_resolve_coeffs`.
- **Tests**: local-Hermite 1-D one-shot, unified interp/interp! 1-D, N=1 tuple-grid collapse (bare-grid ==
  tuple-grid == single-axis-SoA, in-place) across 5 families — all green. Aqua zero new ambiguities.
- **Coverage note**: OnTheFly + N=1-collapse + matrix query is a narrow untested edge (my tests use the default
  AutoCoeffs path → native 1-D); flag for Phase 6 if full OnTheFly-collapse coverage is required.

### Phase 6 — DONE (review + docs + edge cases; perf/version = user)

- **Adversarial review**: a 6-agent workflow reviewed the full `master..HEAD` diff across 5 dimensions
  (missed-sinks, dispatch/ambiguity, allocation/specialization, correctness/contracts, edge/non-goals);
  895K subagent tokens, 202 tool-uses, ~17 min. **Zero real defects survived adversarial verification.** The
  only candidate (LOW) was a test-coverage note (no empty/zero-dim pin) — actioned below.
- **Edge-case tests**: added `empty + degenerate shaped queries` testitem (7 assertions) — 1-D empty Matrix →
  `(0,3)`, N-D empty AoS/SoA, empty in-place, wrong-shape sink rejection. Green.
- **Docs**: shape-aware docstrings on the 5 "Returns Vector" one-shot APIs + a "Shape preservation" section in
  `docs/src/nd/overview.md` with the `vec(itp(q))` migration note.
- **Test suite total**: `test/test_query_shape_preservation.jl` = 19 testitems covering all four quadrants,
  both APIs, both call forms, 8 families, DerivativeView, N=1 collapse, allocation parity, edge cases.

### Remaining (user-run, per the package's test-tier policy)
- Full package test suite (all files) + extension tests.
- Julia 1.10 (LTS) run.
- CI benchmark (the 10% same-machine regression gate). §7.2 SoA-array fast-check peers were deferred here as a
  potential SPEED (not alloc) optimization — add only if the SoA-matrix path measures >10% slower than SoA-vector.
