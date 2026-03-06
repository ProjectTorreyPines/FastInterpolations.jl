# Type Promotion Rules

`FastInterpolations.jl` uses a unified, two-tier promotion strategy throughout its construction
and evaluation paths:

1. **Promotable scalars** (`_PromotableValue`): standard Julia numeric types where Julia's
   `convert` and `promote_type` are always well-defined. These are promoted automatically.
2. **Duck types**: custom value types (e.g., `SVector`, `ForwardDiff.Dual`, `Measurement`)
   where no general conversion to a grid float type exists. These are preserved as-is.

The distinction is encoded in a single union:

```julia
const _PromotableValue = Union{Integer, AbstractFloat, Rational, Complex}
```

Every promotion decision in the codebase checks `T <: _PromotableValue` at compile time
(inside `@inline` functions), so the check is zero-cost — dead branches are eliminated by
the compiler.

---

## Grid Type Promotion

**Function**: `_promote_grid_float(::Type{Tg}, ::Type{Tv}) -> Type{<:AbstractFloat}`

The grid type `Tg` is always an `AbstractFloat`. Its precision is determined by the wider of
the raw grid element type and the value type, but **only when the value type is promotable**.

```
_PromotableValue  →  float(promote_type(Tg_raw, real_eltype(Tv)))   # widen to wider precision
duck type         →  float(Tg_raw)                                   # grid precision only
```

Examples:

| Grid (`Tg_raw`) | Data (`Tv_raw`) | Resulting `Tg` |
|-----------------|-----------------|----------------|
| `Float32`          | `Float64`          | `Float64`      |
| `Float64`          | `Float32`          | `Float64`      |
| `Int`              | `Float64`          | `Float64`      |
| `Float32`          | `Complex{Float64}` | `Float64`      |
| `Float32`          | `Complex{Float32}` | `Float32`      |
| `Float64`          | `Complex{Float32}` | `Float64`      |
| `Float32`          | `SVector{...}`     | `Float32`      |
| `Float64`          | `Dual{...}`        | `Float64`      |

Rationale: when both grid and values are standard numerics, computing in the narrower type
would silently lose precision. Widening to `float(promote_type(...))` makes the precision
loss visible and avoids per-element conversion overhead in hot evaluation paths.

!!! note "`real_eltype` for Complex"
    For `Complex{T}`, `real_eltype` returns `T` so that the grid widens to accommodate the
    real part's precision — not to `Complex` itself (grids are always real-valued).
    For example, `Float32` grid + `Complex{Float64}` data → `Tg = Float64`.

---

## Value Type (`Tv`) Determination

**Function**: `_value_type(::Type{Tv_raw}, ::Type{Tg}) -> Type`

After `Tg` is determined, `Tv` is resolved via three dispatch methods (listed most-to-least specific):

```julia
# Method 1 — Complex (most specific, dispatches before the _PromotableValue fallback)
_value_type(::Type{Complex{T}}, ::Type{Tg}) where {T<:Real, Tg<:AbstractFloat} = Complex{Tg}

# Method 2 — All other _PromotableValue (Integer, AbstractFloat, Rational)
_value_type(::Type{T}, ::Type{Tg}) where {T<:_PromotableValue, Tg<:AbstractFloat} = Tg

# Method 3 — Duck types (SVector, Dual, Measurement, ...)
_value_type(::Type{T}, ::Type{Tg}) where {T, Tg<:AbstractFloat} = T
```

### Dispatch priority: why Complex is handled separately

`Complex <: _PromotableValue`, so without the dedicated method, `Complex{Float32}` would
dispatch to Method 2 and return bare `Tg` (e.g., `Float64`) — stripping the `Complex`
wrapper. Method 1 is more specific and dispatches first, returning `Complex{Tg}` to
preserve the complex structure while adapting only the float precision.

### Summary of rules

| Category           | `Tv_raw` example        | Rule                       | Resulting `Tv`         |
|--------------------|-------------------------|----------------------------|------------------------|
| Real scalar        | `Float32`, `Float64`    | conform to grid float      | `Tg`                   |
| Real scalar        | `Integer`, `Rational`   | conform to grid float      | `Tg`                   |
| Complex (float)    | `Complex{Float32}`      | preserve wrapper, adapt `T`| `Complex{Tg}`          |
| Complex (non-float)| `Complex{Rational{Int}}`| preserve wrapper, adapt `T`| `Complex{Tg}`          |
| Duck type          | `SVector{3,Float64}`    | preserve exactly           | `SVector{3,Float64}`   |
| Duck type          | `Dual{...}`             | preserve exactly           | `Dual{...}`            |

### Examples

| `Tv_raw`                | `Tg`      | `Tv`                   | Method |
|-------------------------|-----------|------------------------|--------|
| `Float32`               | `Float64` | `Float64`              | 2      |
| `Float64`               | `Float32` | `Float64`              | 2 (Tg was widened in grid phase) |
| `Complex{Float32}`      | `Float64` | `Complex{Float64}`     | 1      |
| `Complex{Float64}`      | `Float32` | `Complex{Float64}`     | 1 (Tg was widened in grid phase) |
| `Complex{Float32}`      | `Float32` | `Complex{Float32}`     | 1      |
| `Complex{Rational{Int}}`| `Float64` | `Complex{Float64}`     | 1      |
| `Rational{Int}`         | `Float64` | `Float64`              | 2      |
| `SVector{3,Float64}`    | `Float32` | `SVector{3,Float64}`   | 3      |

!!! tip "Key invariant"
    `Tv` is always the same "kind" of type as `Tv_raw` — real stays real, complex stays
    complex, duck type stays duck type. Only the float precision is adapted.

---

## Fill Value Promotion (`FillExtrap`)

**Type**: `FillExtrap{T} <: AbstractExtrap`
**Functions**: `_promote_extrap`, `_promote_extraps_nd`

### Why `FillExtrap` is parametric

`FillExtrap{T}` stores `value::T`. The `{T}` parameter is required for **eval-path type
stability**: `_constant_extrap_result` returns `e.value` directly, and the compiler must
know its type at compile time to avoid heap allocation. Without `{T}`, `value::Any` would
cause boxing on every out-of-domain query.

!!! note "Historical context"
    `{T}` was originally introduced to distinguish `ConstExtrap{Nothing}` (clamped) from
    `ConstExtrap{Tv}` (fill). Since the refactor split these into separate types
    (`ClampExtrap` and `FillExtrap`), the `{T}` parameter now serves only the
    type-stability purpose.

### Outer constructors

```julia
FillExtrap(v::Real)                      # → FillExtrap{typeof(float(v))}(float(v))
FillExtrap(v::Complex{T<:AbstractFloat}) # → FillExtrap{Complex{T}}(v)
FillExtrap(v)                            # → FillExtrap{typeof(v)}(v)  [auto-gen, duck types]
```

The `Real` constructor converts via `float()` immediately, so `FillExtrap(0)` and
`FillExtrap(1//3)` both produce `FillExtrap{Float64}` — `FillExtrap{Integer}` or
`FillExtrap{Rational}` are not reachable through the public API.

### Promotion at interpolant construction

When an interpolant is built with known `Tv`, fill values are promoted to match using the
same `_PromotableValue` pattern:

```julia
# _PromotableValue: auto-convert (always safe, convert is well-defined)
_promote_extrap(e::FillExtrap{<:_PromotableValue}, ::Type{Tv}) =
    FillExtrap{Tv}(convert(Tv, e.value))

# Duck types: pass through unchanged (user is responsible for correct type)
_promote_extrap(e::FillExtrap, ::Type) = e

# All other AbstractExtrap (ClampExtrap, NoExtrap, etc.): pass through
_promote_extrap(e::AbstractExtrap, ::Type) = e
```

!!! warning "Duck-type fill values"
    For duck-type values such as `SVector`, `ForwardDiff.Dual`, or user-defined structs,
    there is no general `convert(Tv, value)` defined. The user must pass a fill value
    whose type is already compatible with `Tv`. Attempting unconditional conversion would
    produce a `MethodError` at construction time.

### ND fill value constraint

In ND interpolation, all axes that use `FillExtrap` must share the **same fill value** — the
OOB short-circuit returns a single scalar result regardless of which axis triggered it.
This is validated at `_resolve_extrap_nd` time (before promotion) via `_validate_fill_values_nd`.

---

## Two-Phase Extrap Setup (ND)

ND interpolant construction separates extrap handling into two ordered phases:

```
Phase 1: _resolve_extrap_nd(extrap, bcs, Val(N))
          - Shape normalization: scalar → ntuple, tuple length check
          - Periodic BC compatibility check
          - Fill value consistency validation (===, on original values)
          → NTuple{N, AbstractExtrap}

Phase 2: _promote_extraps_nd(extrap_vals, Tv)
          - Per-axis _promote_extrap call (unrolled via @generated)
          → NTuple{N, AbstractExtrap} with fill values matching Tv
```

!!! danger "Ordering constraint"
    These phases **must not be merged**. `_validate_fill_values_nd` uses `===` to compare
    fill values. After promotion, `FillExtrap{Float32}(0.0f0)` and `FillExtrap{Float64}(0.0)`
    are different objects — even if semantically equal — so validation would silently pass
    conflicting values. Validation must run on the original, un-promoted values.

---

## Summary Table

| Stage | Promotable scalars | Duck types |
|-------|--------------------|------------|
| Grid type `Tg` | `float(promote_type(Tg_raw, real_eltype(Tv)))` | `float(Tg_raw)` |
| Value type `Tv` | `Tg` (or `Complex{Tg}`) | preserved as-is |
| Fill value `FillExtrap` | `convert(Tv, value)` at construction | pass through |
| Query type at eval | `promote_type(Tv, Tq)` (standard widening) | same |

---

## Related Source Files

| File | Contents |
|------|----------|
| `src/core/utils.jl` | `_PromotableValue`, `_promote_grid_float`, `_value_type`, `_promote_value_type`, `_promote_xy_types` |
| `src/core/eval_ops.jl` | `FillExtrap{T}`, `_promote_extrap` |
| `src/core/nd_utils.jl` | `_resolve_extrap_nd`, `_promote_extraps_nd`, `_validate_fill_values_nd` |
