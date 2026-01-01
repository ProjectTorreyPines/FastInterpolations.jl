# Cubic Autocache Refactor (Future) — Design Notes

## Purpose
Prepare for a **post-quadratic refactor** that extracts shared autocache logic between
cubic and quadratic interpolation **without changing cubic behavior**.

This document captures decisions/constraints so a future refactor can proceed safely
and consistently with the quadratic design.

---

## Goals
- Keep **public cubic API and behavior unchanged**.
- Preserve **zero-allocation** and **thread-safety** guarantees.
- Extract **shared RCU cache core** usable by both cubic and quadratic.
- Optionally improve diagnostics (store BC provenance) and micro-optimizations (inv_h) if justified.

## Non-Goals
- Changing existing cubic interpolation semantics or default BC behavior.
- Rewriting cubic math kernels or solver logic.

---

## Current State (Cubic)
- `cubic_autocache.jl` implements RCU cache with:
  - `CacheBank`, `BankSnapshot`, `GlobalRegistry`
  - lock-free hits, copy-on-write misses, ring eviction
- `CubicSplineCache` stores `x`, `h`, `lu_factor`, `bc_config`
- `CubicInterpolant` stores `cache`, `y`, `z`, `extrap` (no BC field)
- No `inv_h` in cache; divisions appear in RHS construction but are not precomputed

---

## Proposed Refactor (No Behavior Change)
### 1) Extract Shared RCU Core
Create `src/autocache_core.jl` containing:
- `AbstractCacheEntry`, `BankSnapshot`, `CacheBank`, `GlobalRegistry`
- `_registry_lookup`, `_get_bank`, `_rcu_lookup`, `_lookup_or_insert!`

**Cubic:** `cubic_autocache.jl` becomes a thin wrapper that defines entry types
and uses the core functions. Behavior stays identical.

**Quadratic:** `quadratic_autocache.jl` uses the same core.

### 2) Keep per‑algorithm entry types
- Cubic retains `CacheEntry` / `PeriodicCacheEntry` with existing cache build logic
- Quadratic has its own `QuadraticCacheEntry` and builder

This avoids any semantic coupling while still sharing RCU machinery.

---

## Planned Enhancements (Committed)
These changes are **planned** as part of the cubic refactor (not optional), while still
preserving public behavior and correctness.

### A) Store BC in `CubicInterpolant` (Planned)
**Motivation:** provenance/debugging and parity with quadratic design.

**Planned change:**
```
struct CubicInterpolant{T,C,B}
    cache::C
    y::Vector{T}
    z::Vector{T}
    extrap::ExtrapVal
    bc::B
end
```
- `bc` should be stored as a **concrete type** (normalized `BCPair` or `PeriodicBC`).
- No runtime overhead in eval path; type-stable field.
- **Risk:** more type parameters (compile time). Acceptable for planned refactor.

### B) Add `inv_h` to `CubicSplineCache` (Planned)
**Motivation:** reduce divisions in RHS construction and align cache structure with quadratic.

**Plan:**
- Extend `CubicSplineCache` to include `inv_h::Vector{T}`.
- Use `inv_h` in RHS assembly where it replaces repeated divisions.
- Keep evaluation kernels unchanged (they use `h`).

---

## Compatibility Requirements
- No changes to `cubic_interp` signatures or defaults.
- `cubic_autocache` hit/miss behavior must remain identical.
- Existing tests and benchmarks must pass unchanged.

---

## Validation Checklist
- Unit tests for cubic interpolation and autocache pass unchanged.
- Allocation tests (`@allocated`) show no regressions.
- Benchmark: cache hit latency within noise of current implementation.
- No changes in output for existing cubic test vectors.

---

## Suggested Refactor Order (Future)
1. Introduce `autocache_core.jl` with zero behavioral changes.
2. Refactor `cubic_autocache.jl` to use core (pure mechanical change).
3. Add `quadratic_autocache.jl` using core.
4. (Optional) Add BC field to `CubicInterpolant` with gated decision.
5. (Optional) Add `inv_h` to `CubicSplineCache` if benchmarking supports.

---

## Notes for Future Review
- Keep cubic behavior **frozen** during refactor.
- Any optional change (BC field, inv_h) should be justified with measured benefits.
- Prefer small, staged PRs to isolate risk.
