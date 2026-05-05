# Drop `spacing(s)::S` from Linear/Constant/Quadratic/Hetero forward + split persistent axis pipeline

Migrates the entire 1D/Series/ND forward path off the legacy `spacing(s)::S`
field by treating the wrapped axis (`_CachedRange` / `_CachedVector` /
`_ExclusivePeriodicAxis`) as the single source of truth for `h`/`inv_h`.

## What's new vs master

### Struct changes

`spacings::S` field removed from:
- `LinearInterpolantND`
- `ConstantInterpolantND`
- `HeteroInterpolantND`
- `QuadraticInterpolantND`

`spacing::S` field removed from:
- `LinearSeriesInterpolant`
- `ConstantSeriesInterpolant`
- `QuadraticInterpolant` (1D)
- `QuadraticSeriesInterpolant`

`itp.x` for migrated 1D types now stores wrapped axis directly. Quadratic
gains `_CachedVector`/`_CachedRange` storage for free (was plain `Vector{Tg}`
with a separate `VectorSpacing` field).

### Public API impact

User-facing API surfaces are unchanged. All `*_interp(...)` calls produce the
same types and values. Internal storage layout differs, so any external code
that read `itp.spacings` / `itp.spacing` / `_spacing(itp)` on the migrated
families needs to use `itp.grids[d]` / `itp.x` instead.

### New internal helpers

| | role |
|---|---|
| `_cache_axis(x, bc, Tg)` | persistent surface API: zero-copy buffer wrap + allocate cached `h`/`inv_h`. Verb-form parallel to existing `_resolve_axis` (one-shot) |
| `_cache_axis_for_method(g, bc, m)` | HeteroND-only: passes `NoInterp` singleton axes through raw |
| `_convert_copy(x, Tg)` | wrapper-aware ownership copy + element-type promotion (replaces `_resolve_axis_copied(x, NoBC(), Tg)` inside inner ctors) |

`_resolve_axis_copied` retained for Cubic only.

## Bugs fixed

- **`integrate(itp::ConstantInterpolantND)` FieldError**: ND integrate helpers
  now read `itp.grids[d]` instead of the removed `itp.spacings`.
- **`HeteroInterpolantND` mutation propagation**: wrapper-preserving `Base.copy`
  + per-axis `_convert_copy` now isolates user-supplied wrapped grids correctly.
- **`itp(GridIdx(k))` returns `NaN` on 1D scalar queries**: bare `GridIdx(k)`
  was not resolved at the scalar eval kernel head, so its `NaN` sentinel
  propagated. Affects Linear/Cubic/PCHIP/Cardinal/Akima/Hermite-OnTheFly across
  all extrapolation modes.
- **Float32 promotion lost on Integer Range + Float32 data**: e.g.
  `linear_interp(1:4, Float32[...])` silently produced `Float64` storage.
  Now correctly preserved as `_CachedRange{Float32}` + `Vector{Float32}`.

## Trade-off

`Base.copy(::_CachedVector)` performs a true deep copy (every field independent).
Construction allocations on raw `Vector` grids increase ~+50% (1D) / ~+5× (ND)
relative to a pure-aliasing approach; chosen for clean ownership semantics.
Query-time eval cost is unchanged.

## Scope

**In**: Linear/Constant 1D + Series + ND, Hetero ND, Cardinal/Akima/PCHIP 1D
+ their adjoints (caching wrap; periodic adjoint support unchanged).

**Out** (deferred, marked with `# TODO(spacing-cleanup)`):
- 5 ND adjoint structs (Linear/Constant/Hetero/Cubic/Quadratic adjoint) —
  migrate off `spacings::S` next; final cleanup will remove
  `AbstractGridSpacing` / `ScalarSpacing` / `VectorSpacing` / `_create_spacing*`
  / `_resolve_axis_copied` entirely.
- Cubic forward path (still on `_resolve_axis_copied`).
- Hermite ND custom-`dy` form (separate design).
- Non-cubic adjoint PeriodicBC (feature work, references Cubic adjoint Phase 3).

## Verification

- 22+ Tier 1 test files green
- Full suite green on Julia 1.12 (4124+ tests, 1 broken — known Aqua
  `unbound_args` false positive on the `NTuple{N, AbstractVector{Tg}}` pattern)
- New regression tests cover the 5 bugs listed above

```
65 files, +1511 / -401
```
