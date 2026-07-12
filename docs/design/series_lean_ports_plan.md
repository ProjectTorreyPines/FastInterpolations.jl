# Linear/Constant/Quadratic Series → lean `_AxisAnchor` backbone (ports)

**Status**: in progress. Branch `perf/series-lean-ports` off `perf/series-scalar-unify` HEAD (assumes #188 merges).
**Goal**: finish the forward-eval Series unification started for cubic (#187/#188) across the remaining families, then delete the duplicate OOB-fill family in `core/series_utils.jl` (pre-merge-review finding C1). **Behavior-preserving** (bit-identical / ≤1 ULP), same zero-alloc contracts. Adjoint payload redesign and Hermite Series are **out of scope** (different risk classes).

## Design (mirror cubic `cubic_series_payloads.jl`)

Shared, family-agnostic machinery — generalize the cubic build loop from a hardcoded `CubicInterp()` to a passed interp method `m` (logic unchanged; this is added dispatch, not a generic-logic edit):

- `_maybe_stateful_payload(extrap, P)` — already generic, keep.
- `_resolve_series_anchor(m, A, grid, loc, extrap)` — already method-agnostic body; relax the `m::CubicInterp` annotation so any family dispatches.
- `_build_series_anchor(m, A, x, xq, extrap, wrap, searcher)` / `_fill_series_anchors!(m, buffer, …)` — add `m` param; cubic callers pass `CubicInterp()`.

Per family (dedicated dispatch only):

- Reuse the gridded payloads where they exist: Linear → `_LinearValuePayload` / `_LinearDeriv1Payload` / `_LinearZeroPayload`; Constant → `_ConstantValuePayload` (+ zero). Quadratic → new `_Quadratic*Payload1D` if none reusable.
- `_payload_op(::Type{<:_LinearValuePayload}) = EvalValue()`, etc.
- `_<fam>_series_payload_type(op, Tq)` selector + `_<fam>_series_anchor_type(op, extrap, x, Tq)`.
- `_resolve_anchor(::LinearInterp, ::Type{_AxisAnchor{I,P}}, grid, idxL, idxR, xq, xL, xR, extrap)` — the family weight formula (linear α = dL·inv_h).
- Lean kernels in 3 layouts (matrix series-contiguous, point-contiguous `!`, raw-vector) + the two `_<fam>_series_eval` adapters (stateful state-branch → `_fill_constant_extrap_simd!(…,::Type{Tq})`; bare → kernel).
- Wire the 4 surfaces (persistent scalar/batch, one-shot scalar/batch, periodic) onto the lean build + kernels; delete the old `_make_anchor` / `_eval_series_*` op×extrap chains and the family `_XxxAnchoredQuery` if unused elsewhere (keep it if the family adjoint consumes it, like cubic keeps `_CubicAdjointAnchor`).

## Phases (each a quality gate; TDD, affected-tests-only locally, user/CI runs full)

- **P1 — shared machinery**: parameterize build loop on `m`. Gate: cubic Series tests green + cubic scalar/batch A/B unchanged vs main-dir baseline (`~/.julia/dev/FastInterpolations`).
- **P2 — Linear**: port + delete old chains. RED-pin any OOB contract change first (check if linear scalar/batch OOB currently diverge like cubic's did). Gate: linear scalar≡batch bit-identical (ops×extraps×F64/F32/Complex/Dual, signed-zero, OOB); zero-alloc persistent; A/B no regression vs baseline.
- **P3 — Constant**: same, simpler (value + zero only).
- **P4 — Quadratic**: same (more ops; 7 op×extrap fns today).
- **P5 — cleanup**: delete the `aq`-based `_constant_extrap_boundary_value` / `_fill_constant_extrap_simd!` twins from `series_utils.jl` once no family consumes them. Gate: all Series families green.

## Verification per family
- Equivalence oracle: lean path vs the retained old path, bit-identical (`===` elementwise, ≤ULP where FMA-scheduling differs), across op × extrap × precision × grid{Vector,Range,periodic seam} × values{finite,signed-zero,NaN,Inf,Complex}.
- Alloc pins: persistent scalar/batch `@allocated == 0` after warmup.
- A/B: worktree main-dir (pre-port HEAD) vs this branch, scalar+batch, K∈{2,8,64}, per family. No regression.
