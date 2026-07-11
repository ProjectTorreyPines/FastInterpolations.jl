# Cubic Series op/extrap-aware `_AxisAnchor` payloads

**Status**: design approved after external (Codex) review, revised 2026-07-10; implementation pending
**Branch plan**: off `perf/series-contiguous-indices` (PR #186) — `_ContiguousIndices` is a prerequisite
**PR boundaries**: PR-A = Phase 1 (this perf change). PR-B = Phase 2 (scalar migration). PR-C = Phase 3 (pre-built API removal, full anchor demoted to adjoint-internal). Follow-up PR = adjoint payload redesign.

## 1. Motivation

`_CubicAnchoredQuery` bakes weights for all four eval ops up front: `w0`(4) + `w1`(4) + `w2`(2) + `w3`(2) = 96 B of weights, 120 B total per anchor. Every internal batch path knows `deriv` at anchor-build time, so for any single call three of the four weight sets are dead — wasted build flops, wasted size, and wasted cache bandwidth exactly where it hurts: PR #186 showed the Series K×Q batch is streamed-anchor-bound (an 8 B shrink bought −13.6…−24.6%).

Full baking is only required where a consumer needs several ops from one anchor. Auditing consumers: eval surfaces (batch, scalar) always know the op → migratable to lean anchors; the pre-built reuse API (`_anchor_query` → `itp(aq; deriv=…)`) is internal, unexported, and unused externally → removable; **the adjoints are the one genuine multi-op consumer** (the adjoint struct stores `anchors::Vector{_CubicAnchoredQuery}` and selects the weight field per `DerivOp`, `cubic_adjoint.jl:105/165`; a value-rrule pullback needs w0 for the ∂/∂y scatter and w1 for ∂/∂xq; `_bake_cubic_clampfill_anchors` adds adjoint-specific weight fixups). So the end-state is: lean anchors everywhere in eval, full anchor demoted to an adjoint-owned type, eventually replaced by an adjoint payload.

## 2. Design summary

Wire the gridded `_AxisAnchor{I, P}` backbone into the cubic Series paths with **op-aware, extrap-aware payloads**:

- **Op-aware**: one payload type per eval op, carrying only that op's weights (`NTuple{4}` for value/deriv1, `NTuple{2}` for deriv2/3, empty for `DerivOp{N≥4}`). Weight formulas are the existing `_compute_anchor_weights` verbatim → bit-identical weights by construction.
- **Extrap-aware**: extrap class is known at build time and selects the payload representation at compile time. Extend/Wrap/InBounds/NoExtrap get the bare payload and a branch-free kernel; Clamp/Fill get the bare payload wrapped in the generic `_StatefulPayload{P}` (adds one `state::UInt8`) and keep the current eval-time state-branch semantics exactly — via **two surface-specific extrap adapters** (persistent-matrix vs one-shot raw-vector), because the two surfaces use different OOB formulas today (§7).

**Extrap → payload selection matrix** (the crux: extrap is known at build time, so this is compile-time dispatch — the wrapper is the ONLY thing that varies, every payload detail follows the bare type):

| extrap | payload | hot loop |
|---|---|---|
| Extend / Wrap / InBounds | bare op-payload | branch-free |
| NoExtrap | bare op-payload (throws at build) | branch-free |
| Clamp / Fill | `_StatefulPayload{op-payload}` | current-style state branch, then delegate to the bare kernel |

Anchor sizes (ordinary grids, `_ContiguousIndices`, Float64; all three columns verified by a Julia sizeof probe during review):

| op | bare (Extend/Wrap/InBounds/NoExtrap) | stateful (Clamp/Fill) | current |
|---|---|---|---|
| value / deriv1 | 8 + 32 = **40 B** | 8 + 32 + 1 → **48 B** | 120 B |
| deriv2 / deriv3 | 8 + 16 = **24 B** | 8 + 16 + 1 → **32 B** | 120 B |
| deriv N≥4 | **8 B** | 16 B | 120 B |

## 3. Naming

Rule: **a dimension suffix appears only where one family has two data representations.** Codebase convention puts the qualifier last (`_QuadraticAdjointAnchor1D`, `CubicInterpolantND`).

- New 1D (y/z-spline weights, op-tagged): `_CubicValuePayload1D{Tq}`, `_CubicDeriv1Payload1D{Tq}`, `_CubicDeriv2Payload1D{Tq}`, `_CubicDeriv3Payload1D{Tq}`, `_CubicZeroPayload1D{Tq}`.
- Renames (ND nodal-partials geometry, internal-only, mechanical): `_CubicPartialsPayload` → `_CubicPartialsPayloadND`, `_QuadraticPartialsPayload` → `_QuadraticPartialsPayloadND`. Isolated leading commit in PR-A (zero semantic content, trivially reviewable).
- Unchanged (representation-universal per-axis payloads; a future Linear Series port can reuse them as-is): `_LinearValuePayload`, `_LinearDeriv1Payload`, `_LinearZeroPayload`, `_ConstantValuePayload`, `_LocalHermitePayload`.
- New generic wrapper (family- and dimension-agnostic): `_StatefulPayload{P}` (§5). Placed in core alongside the relocated `_AxisAnchor` because Phase 2/3 and the planned Linear/Quadratic Series ports are concrete second consumers.

## 4. Include-order prerequisite (structural blocker, fix first)

`_AxisAnchor` is currently defined AFTER the cubic module (`FastInterpolations.jl`: cubic at line 15, `gridded/axis_anchor.jl` at line 28), so cubic files cannot reference it. First commit of PR-A (after the rename commit) splits `axis_anchor.jl`:

- **→ core** (new `core/axis_anchor_types.jl`, included right after `anchor_common.jl`): the `_AxisAnchor{I, P}` struct, its virtual-property accessors, `_interval_type`, and `_StatefulPayload{P}`.
- **stays in gridded** (`axis_anchor.jl`): the gridded-specific resolution loop `_axis_anchors_loop!` / `_axis_anchors_pooled` / `_axis_anchors_all` (they depend on gridded searcher helpers and `AbstractInterpMethod` dispatch).

This is a pure move — no signature or behavior changes; gridded tests are the guard.

## 5. Components (Phase 1)

All new cubic code lives in cubic-owned files (e.g. `cubic/cubic_series_payloads.jl`).

**Payloads** — five structs as named in §3, each holding `w::NTuple{K, Tq}` (K = 4/4/2/2/0), plus the generic stateful wrapper, pinned verbatim:

```julia
# generic wrapper — not cubic-specific: wraps ANY payload; the inner payload
# keeps every op/type detail, the wrapper only adds the OOB classification.
struct _StatefulPayload{P}
    inner::P
    state::UInt8      # IN_DOMAIN / OOB_LEFT / OOB_RIGHT
end
```

**Payload selection** — a cubic-Series-owned selector `_cubic_series_anchor_type(op, extrap, x, Tq)` mapping (op → base payload) × (extrap isa `_ClampOrFill` → `_StatefulPayload` wrap). It must NOT be a new `_axis_anchor_type(::CubicInterp, ...)` method — that signature is already taken by the ND partials path (`gridded_partials.jl:26`) and would collide.

**Resolution & build loop** — `_resolve_anchor` overloads dispatching on the payload type inside `A` (no collision with the partials overloads: the payload type differs), signatures pinned to preempt ambiguity (the Aqua ambiguity gate is the backstop):

```julia
# mirrors the partials overloads' shape; only the payload type in `A` differs
_resolve_anchor(m::CubicInterp, ::Type{_AxisAnchor{I, _CubicValuePayload1D{Tq}}},
    grid, idxL::Int, idxR::Int, xq, xL, xR, extrap) where {I, Tq}
# stateful variant is resolved by the Series-owned build loop, which passes loc.state:
_resolve_series_anchor(m::CubicInterp, ::Type{_AxisAnchor{I, _StatefulPayload{P}}},
    grid, loc::_AnchorLoc, extrap) where {I, P}
```

Internals: existing `_get_h`/`_get_inv_h` + `_compute_anchor_weights(op, h, inv_h, dL, dR)`. The stateful variant needs `loc.state`, which the gridded backbone loop does not pass, so cubic Series owns its small build loop (search → wrap flag → resolve). Under NoExtrap it throws on the first OOB coordinate via `_throw_domain_error(xq, x, 0)` (`utils.jl:563/576` — untyped args, so no mixed-precision `MethodError`; `dim = 0` keeps the axis-agnostic "query point" phrasing matching today's 1D message) — deliberately NOT the same-typed `_throw_extrap_domain_error` (§7).

**Kernel / adapter layering** (per review: the earlier "one generic stateful kernel" cannot preserve both surfaces' OOB formulas, and the stateful path needs `extrap` for `fill_value`):

```julia
# bare kernels — op/payload only, no extrap/surface knowledge; weights from a.w,
# taps from a.idxL/a.idxR (virtual properties). One method per payload type.
# Named after the payload family (not "series"): the same kernels serve any
# future consumer of the lean anchors (e.g. unified `interp` batch routing, §11).
_cubic_payload_kernel(y::Matrix,        z::Matrix,        k::Int, a)   # matrix layout, column k
_cubic_payload_kernel(y::AbstractVector, z::AbstractVector,        a)   # raw-vector layout

# extrap adapters — own the state branch; extrap passed for fill_value.
# These ARE surface-specific (they preserve each surface's OOB formula, §7),
# hence the "series" name stays here. In-domain: rebuild the bare anchor
# `_AxisAnchor(a.interval, a.inner)` (isbits, zero-cost) and delegate.
_cubic_series_eval(y::Matrix,        z::Matrix,        k, a_stateful, extrap)  # persistent formula: val * one(Tq)
_cubic_series_eval(y::AbstractVector, z::AbstractVector,   a_stateful, extrap)  # one-shot formula: _eval_extrapolation(op, val, extrap, zero(Tq))
```

The `op` instance for the OOB arm is recovered from the inner payload type via a tiny trait (`_payload_op(::Type{_CubicValuePayload1D{Tq}}) = EvalValue()` etc.), keeping kernels op-argument-free. Both adapters use `xq` only through `one(Tq)` / `zero(Tq)`, which are value-independent — this is why the payload never stores `xq`.

**Consumers (all four batch surfaces)** — every batch path that knows `deriv` at build time switches to lean anchors:

1. persistent Series batch `sitp(outputs, xq; deriv)` (`cubic_series_interp.jl:838`);
2. one-shot Series non-periodic vector batch (`cubic_oneshot_series.jl:280`);
3. one-shot Series **periodic** vector batch (`cubic_oneshot_series.jl:147`, reached via the PeriodicBC branch at `:268`) — required, otherwise the exclusive-periodic seam claim in §8 is untestable on the new path;
4. pre-built path `sitp(outputs, aq_vec; deriv)` and all scalar paths keep full anchors in PR-A (they are the equivalence reference).

Pool acquisition keys on the concrete `_AxisAnchor{I, P}` type (per op × extrap-class × Tq — more pool entries than before; accepted).

## 6. Later phases

- **✅ DONE — scalar migration + cleanup (one PR, `perf/series-scalar-unify`)**: both persistent scalar entries and the one-shot scalar (bcpair + periodic) route through the shared lean anchor build + a payload-generic **point-contiguous** SIMD kernel. Key correction vs the original expectation: scalar is **not** perf-flat via the batch kernel — its only parallelism axis is across the K series, so the point-contiguous transpose (`y_point[k,idx]`) is SIMD-mandatory (measured 1.6–4.5× vs a series-contiguous loop at K=16–1024; the lean point kernel is at parity with the old op-specialized one, `@code_llvm` vectorized). Net a **12→~3 function** reduction (op→payload, extrap→state). Also removed the pre-built eval API (`itp(aq)`, `itp(aq_vec)`, `sitp(outputs, aq_vec)` — none exported/documented) and the now-dead kernels; `_CubicAnchoredQuery` renamed **`_CubicAdjointAnchor`** (only the 1D/ND adjoints consume it now, via the retained `_anchor_query`). The batch oracle was rebased onto an inline port of the series formula (full anchors still built by the retained `_fill_anchors!`), keeping the exact OOB signed-zero reference.
- **Follow-up PR — adjoint payload redesign**: replace the full anchor inside the adjoints with an op-pair payload (a value rrule needs exactly (w0, w1); a deriv1 rrule (w1, w2)), then delete `_CubicAnchoredQuery` entirely. Touches AD extensions (ChainRules/Enzyme) — kept out of the perf series on purpose. `_bake_cubic_clampfill_anchors`'s weight-fixup logic moves with it.
- The earlier "rim+core restructure" idea is **dropped**: once the full anchor is adjoint-internal and slated for replacement, restructuring its interior buys nothing.

## 7. Behavior notes (all verified against source during review)

- **Two coexisting OOB formulas are preserved as-is in PR-A.** Persistent Series OOB uses `val * one(aq.xq)` (`series_utils.jl:145-167`); one-shot Series OOB uses `val + zero(xq)*zero(val)` (`_eval_extrapolation`, `utils.jl:819-822`). These differ observably on signed zero (`-0.0 * 1.0 = -0.0` vs `-0.0 + 0.0 = +0.0`) and the divergence exists on master for value and derivative ops alike. The two adapters in §5 preserve each surface's formula bit-exactly. Unifying them is a separate consistency PR if ever desired.
- **NoExtrap is a correctness fix, not just error timing.** Today the persistent batch throws `_throw_extrap_domain_error(xq::T, x_min::T, x_max::T)` with a same-type constraint (`series_utils.jl:133`), so a Float32-grid + Float64-OOB-query batch raises `MethodError`, not `DomainError`. The new build loop throws the generic axis-named `DomainError` before any output is written. Publicly observable improvements, RED-pinned and called out in the PR message: (a) exception type fixed to `DomainError` in mixed precision; (b) throw happens before any output mutation; (c) `DomainError.val` carries the offending coordinate — treat the message text itself as non-contractual.
- **1D vs ND Clamp-deriv contract divergence (pre-existing, discovered during this design; out of scope)**: 1D surfaces return flat-extension zero for Clamp OOB derivatives; ND pointwise and GriddedQuery return the boundary-cell derivative (clamp-then-eval), tested in `test_gridded_query.jl:298`; NaN-neighbor import splits the same way. Record as a separate consistency issue (`fix/nd_consistency` family) — this work keeps the 1D contract for Series.

## 8. Verification plan

- **RED pins first** (TDD): NaN-neighbor Clamp (one NaN in y → z all-NaN via the global solve; OOB query still returns the finite boundary sample); signed-zero OOB per surface (persistent `-0.0` vs one-shot `+0.0`, pinned separately); Float32-grid + Float64-query NoExtrap OOB → `DomainError` (RED: currently `MethodError`); outputs untouched on NoExtrap throw; `DerivOp(N≥4)` keeps the `0 * y[idxL]` carrier/NaN semantics; empty query vector.
- **Equivalence oracle**: the retained internal full-anchor evaluator chain called directly (NOT the public pre-built method — its `outputs::AbstractVector{<:AbstractVector{Tv}}` constraint (`cubic_series_interp.jl:911`) rejects mixed-precision output buffers and cannot cover the matrix below). Bit-identical (`===` elementwise) across: op {value, d1, d2, d3, d≥4} × extrap {NoExtrap, Extend, Clamp, Fill, Wrap, InBounds} × precision {F64/F64, **F32 query on F64 grid, F64 query on F32 grid**} × grid {Vector, Range, exclusive-periodic seam (uses consumer 3)} × values {finite, signed zero, NaN, Inf, Complex Tv, NaN/Inf fill_value}.
- **Type stability**: `@inferred` on selector, build loop, and both adapters; concrete anchor `eltype` for every (op × extrap × Tq) combination; alloc pins (batch entry zero-alloc after pool warmup per payload type). Explicitly pin `@allocated == 0` on the adapter's in-domain delegation (the `_AxisAnchor(a.interval, a.inner)` rebuild) — if the compiler ever boxes that intermediate, the streamed-anchor win is negated, so this is a dedicated test, not a byproduct of the batch alloc pin.
- Affected-test tier only locally (`cc-julia-test-runner`); full suite is the user/CI gate. `runic -i .` before commits.

## 9. Benchmark plan

Worktree A/B vs the base branch (`~/.julia/dev/FastInterpolations-<name>`, not /tmp): K×Q series-contiguous batch for value/deriv1/deriv2, K ∈ {2 (extreme-ratio check), 8, 64}, Q large.

- Arms: {current 120 B full anchor} vs {bare payloads (Extend)} vs {stateful payloads (Clamp)} vs {geometry payload probe (§10)}.
- **Clamp/Fill OOB sweep**: OOB fraction ∈ {0, 1, 10, 50, 100}% × {clustered, random} placement — the state branch vs post-pass trade-off is dominated by OOB ratio and branch predictability, so this sweep is what would trigger building the post-pass fallback arm (§10).
- The "80 B shrink ≥ the #186 8 B win" extrapolation is a hypothesis, NOT an acceptance criterion. Acceptance: no regression anywhere measured; statistically clear win on K×Q streamed shapes; scalar/Q×K flat (loop-invariant anchors).

### Results (2026-07-10, Apple M1 Pro, same-process A/B via `benchmark/cubic_series_payload_ab.jl`, N=256 Q=4096)

Methodology switched from worktree A/B to same-process runtime-swap (the retained full-anchor building blocks `_fill_anchors!` + `_eval_series_vector!` ARE the old batch entry, with a prealloc'd anchor buffer — a slightly flattered baseline), per the carrier-aware-blend lesson: no cross-env drift. Ratio = new/old, < 1.0 = lean anchors faster.

| shape | value | deriv1 | deriv2 |
|---|---|---|---|
| K=2 Extend | 0.714 | 0.633 | 0.574 |
| K=2 Clamp | 0.821 | 0.759 | 0.669 |
| K=8 Extend | 0.713 | 0.671 | 0.527 |
| K=8 Clamp | 0.796 | 0.750 | 0.591 |
| K=64 Extend | 0.676 | 0.669 | **0.423** |
| K=64 Clamp | 0.733 | 0.730 | 0.484 |

Clamp OOB-fraction sweep (K=8, value): stateful wins at every point — 0% 0.794, 1% 0.80/0.88 (random/clustered), 10% 0.77/0.88, 50% 0.74/0.84, 100% 0.635 — so the **post-pass fallback arm is not needed** (§10). Scalar path 65 ns (unchanged, zero diff). One-shot batch measured absolute only (84.1/60.5 µs value/deriv2, K=8; its old arm was replaced in-place — bit-equivalence to the retained scalar path is the correctness gate). The geometry-payload probe was skipped: with wins of this size in the baked arms and deriv2/3 (where geometry would be strictly worse) leading, the probe cannot change the decision.

## 10. Rejected alternatives (with evidence)

- **Weight-folding Clamp/Fill into baked weights** (no state): (a) Fill's `fill_value` is not in span{y, z} — unrepresentable by weights. (b) Clamp+deriv must return the flat-extension zero under the 1D contract, but weights computed at the clamped coordinate give the boundary slope S′(x₁). (c) Even Clamp+value on finite data drifts by ULP (`h * inv(h) ≠ 1` exactly), and the z channel imports NaN globally. Would also split batch semantics from the retained adjoint/full-anchor path.
- **Post-pass overwrite** (build-time OOB classification, branch-free hot loop, second pass rewrites OOB positions): correct and 8 B leaner for Clamp/Fill, but adds `any_oob` tracking + a second pass, and the stateful design is structure-preserving vs today (strictly easier to verify; today's code already pays the same predictable branch at 120 B). Fallback arm, built only if the §9 OOB sweep shows the stateful branch losing.
- **One generic stateful kernel for both surfaces**: refuted during review — persistent and one-shot use different OOB formulas with observably different signed-zero results (§7). Two thin adapters instead.
- **Build-time zero-payload substitution for Clamp/Fill OOB derivatives** (swap in `_CubicZeroPayload1D` per OOB element instead of the state branch): rejected twice over. (a) The OOB deriv result is NOT a plain zero — it is `0 * src` where `src` is the per-series boundary sample `y[boundary, k]` (Clamp) or `fill_value` (Fill), so NaN/Inf taint must propagate per series k; a build-time payload cannot reference per-k data. (b) Mixing payload types per element makes the anchors vector heterogeneous → `Union` eltype → boxing, destroying exactly the streamed-anchor win. Corollary pinned for future readers: the OOB deriv arm MUST keep its boundary-data load — do not "optimize away" the `y` read.
- **Geometry payload for Series** (store `dL, h, inv_h` like `_CubicPartialsPayloadND`; kernel derives weights): 8 B smaller for value/deriv1 but ~10 extra flops (two cubes) per (k, j) in the innermost K×Q loop; deriv2/3 would be both larger AND slower than baked. The gridded cubic uses geometry because its anchors are loop-invariant operands (tensor-product reuse); Series K×Q anchors are streamed operands, the opposite regime. One probe arm in §9 closes this empirically.
- **NTuple{4} uniform weights for all ops**: resurrects the 16 B + dead y-loads the current deriv2/3 z-only optimization removed.
- **New abstract supertype + adapter over a standalone struct**: superseded by `_AxisAnchor` reuse (no new type hierarchy, payload identity is the tag).
- **Keeping the full anchor as a "backward-compat" eval surface**: the pre-built reuse API is internal, unexported, and externally unused; its only irreplaceable consumer is the adjoints (multi-op requirement, §1) — hence demotion (Phase 3) instead of preservation.

## 11. Out of scope / follow-ups

- Linear/Constant/Quadratic Series ports onto the same backbone (linear can likely reuse the existing gridded payloads directly; `_StatefulPayload` already covers their Clamp/Fill).
- 1D↔ND Clamp OOB deriv contract unification (§7, separate issue).
- Persistent↔one-shot signed-zero OOB formula unification (§7, separate consistency PR if desired).
- Adjoint payload redesign and final `_CubicAnchoredQuery` deletion (§6 follow-up PR).
- Unified-API `interp` batch routing through the lean path.
