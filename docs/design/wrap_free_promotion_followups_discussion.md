# Design discussion — wrap-free promotion follow-ups

**Status:** Resolved — all three topics implemented on `fix/convex_linear_kernel` (2026-06-28). Discussion retained below as the decision record.

## Resolution (implemented)

Decisions and what shipped (TDD: a failing test pinned each fix first):

- **Naming:** kept `_fielddiff` / `_fieldsum` (the first arg *is* the field type; `_floatdiff` rejected as inaccurate — the field is not always float; `_safediff` deemed vaguer + churn). No new helper introduced.
- **Topic 1 (Tc consistency):** canonicalized the `Base.promote_op(*, …)` spellings to the existing **`_promote_eltype(_coeff_op, Tg, Tv)`** (no `_coeff_field_type` alias — the explicit op-shape is clearer and *always widens*, unlike `promote_op(*, T, T)`). Migrated `polyfit_kernels.jl` (both overload families), `cubic/nd/cubic_nd_math.jl`, `quadratic/nd/quadratic_nd_eval.jl`, `integral/integrate_kernels.jl`. Buffer-eltype / `Tz` / `typeof(zL)` left as in-scope witnesses. Verified type-identical on all reachable carriers.
- **Topic 2 (PolyFit footgun):** the homogeneous `_compute_deriv1` overload now derives `Tc` through `_promote_eltype(_coeff_op, …)` so a narrow `T` widens (was `promote_op(*, UInt8, UInt8) === UInt8` → wrapped). Pinned by a direct-kernel degree-1 regression test.
- **Topic 3 (N0f8 adjoint — correctness):** `pchip_adjoint` / `akima_adjoint` recompute their data-secants through `_fielddiff` (NoBC + periodic kernels), so wrap-safety no longer depends on the `_PromotableValue` whitelist. `N0f8` adjoints now match the float-data reference exactly (was off by ~0.216 / ~0.017). `_PromotableValue` is **kept** (it is the eager-promotion whitelist, not a wrap-safety predicate) with a guard comment at its definition. cubic/cardinal adjoints have no data-dependent `sign`-branch secants → already wrap-safe (no change). PCHIP+`Gray{N0f8}` still errors on `sign(::Gray)` (clear failure, left as-is).

Performance: `_fielddiff`/`_fieldsum` fast paths compile to bare `fsub`/`fadd`; inside `muladd` they preserve FMA fusion (verified identical native asm). `_promote_eltype` is compile-time. Zero runtime overhead on the float hot paths.

---

**Status (original):** Discussion / RFC, with second-review notes added 2026-06-28.
**Scope:** Three design questions raised during the pre-merge review of the
`fix/convex_linear_kernel` branch. Topics 1 and 2 remain cleanup / latent-risk
items. Topic 3 is now partly a correctness follow-up: `N0f8` adjoint paths are
reachable through the public API and do not match the corresponding float-data
adjoints.
**Related:** `docs/src/architecture/type_promotion_rules.md`. Earlier drafts also
referenced `docs/design/coordinate_promotion_unification.md`, but that file is not
present in this checkout.

---

## 0. Background you need to engage cold

FastInterpolations.jl is a Julia interpolation package whose hard contracts are:

- **Zero heap allocation** on the evaluation path after warmup.
- **Bit-identical / ≤1-ULP** results across the API matrix
  `(one-shot · persistent) × (scalar · batch) × (1D · ND)`.
- Hot kernels stay **type-stable**, inferrable, and FMA-friendly.

The package supports unusual element types: standard floats, `Rational`, `ForwardDiff.Dual`
(autodiff through both data and grid), and **finite/fixed-point carriers** — `UInt8`,
`Int8`, `N0f8` (`FixedPointNumbers`), and colorants `Gray{N0f8}` / `RGB{N0f8}`
(image data).

### The wrap-free work (context for all three topics)

A class of bugs existed where interpolation kernels computed `y[i+1] - y[i]` (or
`y[i] + y[i+1]`) **directly on a narrow carrier type before the coefficient machinery
promoted it to a float**. For finite carriers this wraps modularly:
`UInt8(50) - UInt8(200) == UInt8(106)` (should be `-150`), so e.g. a descending-cell
linear midpoint returned `253.0` instead of `125.0`.

The fix introduced two helpers (`src/core/utils.jl`):

```julia
# Tc is the method's coefficient/output field type — never a forced Float.
@inline _fielddiff(::Type{Tc}, a::Tc, b::Tc) where {Tc} = a - b                       # fast path
@inline _fielddiff(::Type{Tc}, a, b) where {Tc} = convert(Tc, a) - convert(Tc, b)     # widen-then-subtract
@inline _fieldsum(::Type{Tc}, a::Tc, b::Tc) where {Tc} = a + b
@inline _fieldsum(::Type{Tc}, a, b) where {Tc} = convert(Tc, a) + convert(Tc, b)
```

The contract: pass the method's **existing coefficient field type** `Tc`. When the operand
is already `Tc` (floats, duck/AD types) it is a plain `a - b` (zero overhead, bit-identical
to the pre-fix code). When the operand is narrower than `Tc` (a finite carrier the
coefficient machinery widens), it converts *before* the `±`, which is what kills the wrap.

Two type helpers define "the coefficient field type":

```julia
@inline _coeff_op(h::Tg, yv::Tv) where {Tg, Tv} = yv + yv * inv(h)   # utils.jl:189
_promote_eltype(_coeff_op, Tg, Tv)   # = typeof(yv + yv*inv(h)) = the coefficient field type
```

`_fielddiff` / `_fieldsum` are now called at ~40 sites across linear / cubic / quadratic /
pchip / akima / cardinal / hermite slope builders, solvers, and integral kernels.

---

## 0.1. Second-review recommendation

The recommended refactor should maximize consistency where the semantics are the
same, while keeping the behavioral surface unchanged:

1. Add a greppable helper such as
   `_coeff_field_type(Tg, Tv) = _promote_eltype(_coeff_op, Tg, Tv)`.
2. Prefer that helper at field-arithmetic sites whenever the desired type is the
   coefficient field. The `Base.promote_op(*, ...)` spelling should be migrated
   first because it is the least expressive form.
3. Make the PolyFit homogeneous overload robust (or collapse it into the mixed
   overload family) and add a direct narrow-type regression test.
4. Treat `N0f8` adjoints as an explicit policy decision:
   - support them by using wrap-safe field arithmetic when recomputing secants in
     `pchip_adjoint` / `akima_adjoint`, or
   - reject them clearly at construction.

The guardrail is strict: any consistency refactor must preserve inference, allocation,
and behavior on the supported carrier matrix. If a site's local witness (`eltype(dy)`,
`Tz`, `typeof(zL)`, etc.) is already the same type, replacing it is acceptable only
when tests and inference checks show no regression.

---

## Topic 1 — Canonical "coefficient field type" (`Tc`) derivation

### Observation

The `Tc` argument passed to `_fielddiff` / `_fieldsum` is spelled **seven different ways**
across the call sites:

| Spelling | Example site | Meaning |
|---|---|---|
| `eltype(dy)` / `eltype(d)` / `eltype(dydx)` | `akima_slopes.jl:63`, `cubic_solver.jl:256`, `cubic_nd_math.jl:250` | eltype of the slope/coeff output buffer |
| `_promote_eltype(_coeff_op, Tg, Tv)` | `linear_kernels.jl:55`, `linear_anchor.jl:131`, `hermite_local_slopes.jl:52`, `cubic_solver.jl:327` | the canonical coefficient field type |
| `_promote_eltype(_coeff_op, eltype(x), eltype(y))` | `hermite_local_slopes.jl:84` | same, from arrays not type params |
| `typeof(zL)` | `coeffs.jl:121` | the cubic moment type |
| `Tz` | `cubic_kernels.jl:94` | the cubic moment type (where-param) |
| `Base.promote_op(*, Tv, Tg)` / `Base.promote_op(*, T, T)` | `polyfit_kernels.jl:121,173` | `typeof(yv * inv_h)` |
| `Base.promote_op(*, typeof(inv_h), typeof(fL))` / `Base.promote_op(*, Tg, typeof(yL))` | `quadratic_nd_eval.jl:34`, `cubic_nd_math.jl:186`, `integrate_kernels.jl` | `typeof(yv * inv_h)`, computed from values |

The review verified that **within each method family these resolve to the same concrete
type** for every currently-supported eltype combination (the coeff buffers `dy`/`d`/`dydx`
are themselves allocated as `_promote_eltype(_coeff_op, …)`, and `Tz`/`typeof(zL)` are that
same moment type). So this is a **DRY / consistency smell, not a bug** on shipped paths.

### Why it might matter

1. **Copy-paste hazard.** A new kernel author copying a sibling site has no single canonical
   "give me the coefficient field type" call to reach for, and could pick a spelling that is
   subtly wrong for their context (see Topic 2 — `Base.promote_op(*, T, T)` is exactly such a
   wrong pick).
2. **`_coeff_op` (`+`) vs `promote_op(*)` can diverge for exotic types.** `_promote_eltype(_coeff_op, …)`
   is `typeof(yv + yv*inv(h))`; `Base.promote_op(*, Tg, Tv)` is `typeof(yv*inv(h))`. For all
   `Real`/`Dual`/colorant types these coincide, but a carrier that overloads `*` and `+` to
   different result types (e.g. some `Unitful`/measurement combos) would see the two spellings
   disagree. The `promote_op(*)` sites silently assume `* === +` in result type.

### Options

- **A. Leave as-is + one doc line.** Add a sentence at the `_fielddiff` definition: "the
  canonical `Tc` is `_promote_eltype(_coeff_op, Tg, Tv)`; `eltype(dy)` / `Tz` / `typeof(zL)`
  are in-scope spellings of it." Zero churn, zero risk. Relies on reviewers to keep new sites
  consistent.
- **B. Introduce a named alias** `_coeff_field_type(Tg, Tv) = _promote_eltype(_coeff_op, Tg, Tv)`
  and migrate all semantically equivalent coefficient-field sites to it, starting with
  the `Base.promote_op(*, …)` sites. Makes the intent greppable and removes the
  `*`-vs-`+` assumption. Medium churn, low risk if gated by inference/allocation/behavior
  tests, and it standardizes future sites.
- **C. Full invariant enforcement.** Make every coefficient-buffer site derive `Tc` from one
  helper and assert (in tests) that `eltype(buf) === _coeff_field_type(...)` everywhere.
  Highest churn; turns an implicit invariant into an enforced one.

### Open questions for the reviewer

- Is the `*`-vs-`+` divergence a real concern for any eltype this package intends to support,
  or purely theoretical? (If purely theoretical, A or B; if real, B/C and the `promote_op(*)`
  sites are arguably latent bugs.)
- Is "one canonical helper" worth the churn given the values are provably equal today, or does
  it just add an indirection over `eltype(dy)` (which is already clear at slope-buffer sites)?

### Second-review recommendation

Choose **B**, with consistency as the goal and quality gates as the limiter:

- Introduce `_coeff_field_type(Tg, Tv)` as the canonical name.
- Migrate the explicit `Base.promote_op(*, ...)` field-arithmetic sites first.
- Prefer the helper more broadly where it means the same thing as `eltype(dy)`,
  `eltype(d)`, `eltype(dydx)`, `Tz`, or `typeof(zL)`.
- Require targeted `@inferred` checks, existing allocation tests, and finite/colorant/AD
  behavior checks before accepting broader rewrites.

For value-computed sites that only know `inv_h` rather than `h`, be careful not to
pretend that `typeof(inv_h)` is always equivalent to the grid type `Tg` for exotic
dimensioned types. If those sites need a shared name, use a separate helper/witness
for "scaled value field" rather than overloading the meaning of `_coeff_field_type`.

---

## Topic 2 — PolyFit homogeneous overload: `Base.promote_op(*, T, T)` does not widen

### Observation

`src/core/polyfit_kernels.jl` has two overload families for the BC-stencil first-derivative
kernel `_compute_deriv1`:

```julia
# Homogeneous (grid eltype == value eltype == T):
@inline function _compute_deriv1(::PolyFit{1}, ::LeftSide, f::NTuple{2, T}, inv_h::T) where {T}
    Tc = Base.promote_op(*, T, T)               # <-- for T = UInt8, this is UInt8 (no widening!)
    return _fielddiff(Tc, f[2], f[1]) * inv_h
end

# Mixed (value Tv, grid Tg):
@inline function _compute_deriv1(::PolyFit{1}, ::LeftSide, f::NTuple{2, Tv}, inv_h::Tg) where {Tv, Tg}
    Tc = Base.promote_op(*, Tv, Tg)             # for Tv=UInt8, Tg=Float64 → Float64 (correct)
    return _fielddiff(Tc, f[2], f[1]) * inv_h
end
```

`Base.promote_op(*, UInt8, UInt8) === UInt8`. So in the **homogeneous** overload, for a narrow
`T`, `Tc` is *not* wider than the operand, `_fielddiff` takes its fast path, and the subtraction
**still wraps**. (`Base.promote_op(*, Gray{N0f8}, Gray{N0f8})` is even `Union{}`.)

More generally, `Base.promote_op(*, T, T)` is the wrong question for this kernel.
It asks "what is the result type of `T * T`?", not "what field type should a
divided difference use?". For most ordinary homogeneous scalar cases it is either
just `T` or otherwise unrelated to the `inv(h)`-weighted coefficient field. The
field helper is therefore not merely nicer spelling; it encodes the intended
arithmetic shape.

### Why it is not a live bug today

This overload is only selected when `inv_h::T` is itself the narrow type. The only real caller
(`_estimate_endpoint_derivative`) always passes `inv_h = inv(T(step))`, which is a float, so it
routes to the **mixed** overload (correct widening). Verified: a cubic build on a genuine `UInt8`
range grid + `UInt8` values matches the float build exactly. So the homogeneous narrow path is
**unreachable via the public API** — a latent inconsistency, not a shipped bug.

### Options

- **A. Leave + comment** that the homogeneous overload assumes a float `inv_h`.
- **B. Make it robust:** derive `Tc` from the coefficient-field helper (or, less preferably,
  `float(Base.promote_op(*, T, T))`) so a hypothetical direct narrow-`inv_h` caller can't wrap.
  Tiny change, removes the footgun, and aligns the two overload families.
- **C. Collapse** the homogeneous and mixed families into one (they differ only in the `Tc`
  derivation). Removes the divergence entirely; slightly more churn.

### Open question

Is preserving a separate homogeneous overload worth it (any perf/inference reason), or should it
fold into the mixed one (Option C)? If kept, B is the cheap safety belt.

### Second-review recommendation

Fix this rather than document it. The homogeneous overload is a private-kernel
footgun: it is not reached by the current public constructor path, but direct tests
or future callers can still observe wrapped subtraction.

Preferred implementation:

- Collapse the homogeneous and mixed `_compute_deriv1` methods if inference and
  allocation tests remain green.
- Otherwise keep both families but derive `Tc` through the same robust helper.
- Add a direct regression test with `f::NTuple{2,UInt8}` and `inv_h::UInt8` (or an
  equivalent narrow homogeneous case) so the latent path is pinned.
- Treat `Base.promote_op(*, T, T)` as not useful for coefficient-field derivation;
  it should not survive in this kernel unless a benchmark proves the helper regresses.

---

## Topic 3 — Two-layer wrap strategy: forward (per-site) vs adjoint (boundary promotion)

### Observation

The codebase currently fixes the wrap problem at **two different layers**, and they coexist
without an explicit rationale:

- **Forward kernels** (slope builders, solvers, value/deriv kernels) use **per-site `_fielddiff`**
  (~40 sites). These run on whatever eltype reaches them.
- **Adjoint kernels** (`akima_adjoint.jl`, `pchip_adjoint.jl`, `cardinal_adjoint.jl`,
  `cubic_nd_adjoint.jl`) still use **raw `y[j+1] - y[j]`** (e.g. `akima_adjoint.jl:244-246,309`).
  These are safe for `_PromotableValue` carriers only because the adjoint entry points call
  `_promote_itp_inputs(x, y)`, which promotes standard numeric `y` values to a float at the
  boundary before any arithmetic.

### Confirmed exception: fixed-point `Real` adjoints

`N0f8` is a `Real`, but it is **not** in `_PromotableValue`
(`Union{Integer, AbstractFloat, Rational, Complex}`). Therefore `_promote_itp_inputs`
does not float-promote `N0f8` data at the adjoint boundary.

Observed behavior from the public API:

```julia
x = [1.0, 2.0, 3.0, 4.0]
y = N0f8.([0.9, 0.1, 0.8, 0.2])
yf = Float64.(y)
ybar = [1.0, 1.0]

maximum(abs.(pchip_adjoint(x, y, [1.5, 2.5])(ybar) .-
             pchip_adjoint(x, yf, [1.5, 2.5])(ybar)))  # about 0.216

maximum(abs.(akima_adjoint(x, y, [1.5, 2.5])(ybar) .-
             akima_adjoint(x, yf, [1.5, 2.5])(ybar)))  # about 0.017
```

For `UInt8` data, the same check matches exactly because `UInt8 <: Integer` and is
promoted to `Float64` by `_promote_itp_inputs`. The bug class is fixed-point `Real`
carriers outside `_PromotableValue`, not all narrow carriers.

### The subtlety that motivates documenting (or unifying) this

For `_PromotableValue` narrow types (`UInt8`/`Int8`), the *forward constructors also*
promote `y` to `Float64` at build time. So when you call `cubic_interp(x, UInt8[...])`,
the slope builder actually sees `Float64` data and the per-site `_fielddiff` calls hit
their **fast path** — i.e. for those numeric carriers, **the ~15 build-site `_fielddiff`
edits are redundant** (the boundary promotion already happened).

The per-site forward edits become **load-bearing only for un-promoted duck/colorant carriers**
(`Gray{N0f8}`), which the constructors deliberately do *not* promote (to return `Gray{Float64}`
rather than a flattened scalar). The direct-kernel call path (external/duck consumers calling
`_linear_kernel`, `_cubic_kernel`, … with raw narrow values) is the other consumer.

So the two layers are not arbitrary:

- boundary promotion handles `_PromotableValue` carriers + AD-sensitive grid policy
  (adjoints, and forward standard-numeric builds),
- per-site `_fielddiff` handles un-promoted duck/colorant carriers + direct-kernel consumers.

But this is **nowhere stated**, and the redundancy-for-numerics reads as "every build
site guards UInt8" when it does not. It also hides the fact that fixed-point `Real`
carriers such as `N0f8` can reach data-dependent adjoint secant recomputation unpromoted.

### A related asymmetry (noted, not necessarily in scope)

`pchip` + `Gray{N0f8}` currently throws (`sign(::Gray{Float64})` is undefined — PCHIP monotonicity
needs `sign`). So PCHIP's per-site `_fielddiff` edits are **defensive-only** — there is no reachable
colorant path for them. cubic/quadratic/akima/cardinal *do* accept colorants.

### Options

- **A. Document the layering** (one paragraph at the `_fielddiff` definition + a note in the
  adjoint files): "forward kernels are wrap-safe per-site because colorant/duck carriers reach
  them un-promoted; `_PromotableValue` carriers are promoted at the boundary
  (`_promote_itp_inputs`) and hit only the fast path." This is necessary, but no longer
  sufficient by itself because of the `N0f8` adjoint mismatch above.
- **B. Unify on boundary promotion.** Promote *all* inputs (including colorant) at construction,
  then drop the per-site forward edits. Pro: one mental model. Con: forces `Gray{N0f8}` →
  `Gray{Float64}` eagerly even when a caller wanted to keep the carrier; changes the duck-type
  story and possibly output types; larger blast radius.
- **C. Add per-site field arithmetic to data-dependent adjoint secant recomputation.** Keep
  boundary promotion for its existing AD/grid-policy reasons, but also use `_fielddiff` where
  PCHIP/Akima adjoints recompute secants from `y`. Pro: fixes `N0f8` without changing the
  constructor promotion contract. Con: a few redundant fast-path calls for already-promoted
  numeric data.
- **D. Explicitly reject fixed-point / non-promoted `Real` data in data-dependent adjoints.**
  Pro: tiny implementation and clear contract. Con: gives up support for a forward-supported
  scalar carrier even though the local arithmetic fix is small.

### Open questions

- Is fixed-point `Real` adjoint support part of the intended API? If yes, choose C. If no,
  choose D and make the rejection explicit.
- Is the two-layer split intentional architecture worth keeping? The review's read is yes,
  but only after documenting the narrower invariant: boundary promotion protects
  `_PromotableValue` carriers, while per-site field arithmetic protects unpromoted carriers.
- Should PCHIP+colorant be (i) explicitly rejected with a clear error, (ii) supported by routing
  monotonicity through a carrier-aware `sign`, or (iii) left as-is? This affects whether PCHIP's
  per-site edits are dead code.

### Second-review recommendation

Choose **C** for `pchip_adjoint` and `akima_adjoint` unless the project wants to explicitly
exclude fixed-point `Real` adjoints. The public API currently constructs these adjoints with
`N0f8` data, and the result differs from the float-data reference.

Keep boundary promotion. Do not try to unify everything on eager boundary promotion, because
that would change the colorant/duck story that the forward kernels intentionally preserve.

For `pchip + Gray{N0f8}`, prefer an explicit clear rejection over a carrier-aware `sign`
until there is a concrete design for monotonicity over color spaces.

---

## Summary table

| # | Topic | Severity | Cheapest safe option | Biggest open question |
|---|---|---|---|---|
| 1 | `Tc` spelled 7 ways | smell (no bug) | B: helper-first consistency, gated by inference/allocation/behavior tests | Need separate helper for `inv_h`-only scaled-value sites? |
| 2 | PolyFit `promote_op(*,T,T)` no-widen | latent private-kernel footgun | B/C: helper-derived field type or collapse overloads | Any inference/perf reason to keep homogeneous methods? |
| 3 | Forward per-site vs adjoint boundary | confirmed `N0f8` adjoint mismatch + undocumented architecture | C: add field arithmetic to PCHIP/Akima adjoint secants, or D: reject | Is fixed-point `Real` adjoint support intended? + PCHIP-colorant policy |

Topics 1 and 2 are decoupled cleanup items. Topic 3 should be decided before claiming the
wrap-free promotion work covers fixed-point scalar carriers across forward and adjoint APIs.
