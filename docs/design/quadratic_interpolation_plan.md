# Quadratic Interpolation Implementation Plan

## Overview
Add **quadratic (C1) spline interpolation** to FastInterpolations.jl.
Focus: API consistency with `linear_interp`/`cubic_interp`, **zero-allocation** hot paths, and an **RCU-style autocache** for repeated x-grid reuse.

Key requirements:
- C1 continuous quadratic spline.
- Boundary condition specified at **one endpoint only**.
- Zero-allocation `quadratic_interp!(...)` after warm-up.
- Autocache **separate from cubic** (no changes to cubic behavior), but structured to allow later refactor into shared core.

---

## Goals
- C1 quadratic spline interpolation for scalar, vector, and in-place paths.
- Consistent `extrap`/`deriv` behavior with existing APIs.
- Zero-allocation hot paths via cached x-grid + pool workspaces.
- Autocache with lock-free hits (RCU pattern), separate from cubic.

## Non-Goals
- Supporting endpoint constraints at **both** ends simultaneously (over-constrained for quadratic C1 spline).
- Replacing cubic autocache internals now.

---

## Mathematical Model (C1 Quadratic Spline)
For each interval `[x_i, x_{i+1}]`:

```
S_i(x) = a_i (x - x_i)^2 + b_i (x - x_i) + c_i
c_i = y_i
```

Let:
- `h_i = x_{i+1} - x_i`
- `s_i = (y_{i+1} - y_i) / h_i` (secant)
- `d_i = S_i'(x_i)` (left slope at knot i)

Constraints:
- Interpolation: `S_i(x_{i+1}) = y_{i+1}`
- C1 continuity: `S_i'(x_{i+1}) = S_{i+1}'(x_{i+1})`

Then:

```
a_i = (s_i - d_i) / h_i
b_i = d_i
c_i = y_i

# Recurrence for slopes
 d_{i+1} = 2 s_i - d_i
```

---

## Degree of Freedom & Boundary Specification
A C1 quadratic spline has **one degree of freedom**. Therefore **only one endpoint condition can be specified**.

We support an explicit **Left/Right endpoint tag** with either Deriv1 or Deriv2:

```
Left(Deriv1(val))   # specify slope at x[1]
Left(Deriv2(val))   # specify curvature at x[1]
Right(Deriv1(val))  # specify slope at x[end]
Right(Deriv2(val))  # specify curvature at x[end]
```

No `BCPair` is supported for quadratic.

### Mapping BC to d[1]
All evaluation paths ultimately need the left slope `d_1` to drive the recurrence.

**Left(Deriv1(d1))**
- `d1` is given directly.

**Left(Deriv2(kappa))**
- `S''(x_1) = kappa` and `S'' = 2 a_1`
- `a_1 = kappa / 2`
- `d_1 = s_1 - a_1 * h_1`

**Right(Deriv1(dn))**
- compute `d_n` given
- recover `d_1` by reverse recurrence:
  - `d_i = 2 s_i - d_{i+1}` for `i = n-1, n-2, ..., 1`

**Right(Deriv2(kappa))**
- `a_{n-1} = kappa / 2`
- `d_{n-1} = s_{n-1} - a_{n-1} * h_{n-1}`
- compute `d_n = 2*a_{n-1}*h_{n-1} + d_{n-1}`
- then reverse recurrence to obtain `d_1`

### Default BC (Natural for Quadratic)
**Default:** `bc = Left(Deriv2(0.0))`
- This makes the first interval linear (`a_1 = 0`), which is the most natural quadratic default.
- This is **not** the same as cubic `NaturalBC()` (which enforces curvature zero at both ends).

### Numerical Behavior Note (Ringing / Error Propagation)
Quadratic C1 splines propagate endpoint slope information through the grid. If data are noisy
or have sharp local variations, errors can accumulate and manifest as oscillation/ringing
near the far end. For global smoothness or noisy data, cubic splines may be preferable.

---

## Boundary Condition Types (Zero-Cost Tags)
Add to `bc_types.jl`:

```
struct Left{T<:AbstractFloat,B<:PointBC{T}} <: AbstractBC{T}
    bc::B
end

struct Right{T<:AbstractFloat,B<:PointBC{T}} <: AbstractBC{T}
    bc::B
end

Left(bc::PointBC{T}) where {T<:AbstractFloat}  = Left{T,typeof(bc)}(bc)
Right(bc::PointBC{T}) where {T<:AbstractFloat} = Right{T,typeof(bc)}(bc)
```

These tags allow dispatch without runtime symbols and preserve zero-allocation paths.
If the base type is **not** parameterized (i.e., `abstract type AbstractBC end`), then
drop the `{T}` parameter and use `Left{B} <: AbstractBC`. Cubic paths should **ignore**
these new BC types (no changes to cubic behavior).

---

## Extrapolation Semantics (Consistent with Existing API)
- `:none` → DomainError for out-of-range queries
- `:constant` → clamp to boundary value; **deriv=1/2 return 0**
- `:extension` → extend using boundary quadratic polynomial
- `:wrap` → wrap into `[x_min, x_max)` using `_wrap_to_domain`, then evaluate
  - Note: quadratic splines are not periodic; `:wrap` may introduce a value discontinuity at the seam.

---

## Core Evaluation Kernel
Quadratic kernel (value/deriv) uses precomputed `a[i]`, `d[i]`, and `y[i]`:

```
# dt = xi - x[i]
value  = a[i]*dt^2 + d[i]*dt + y[i]
first  = 2*a[i]*dt + d[i]
second = 2*a[i]
```

---

## Autocache Strategy (Quadratic Only, Future-Ready)
### Requirement
`quadratic_interp!(out, x, y, xq)` must be zero-allocation on hot path. That requires caching x-grid preprocessing (`h`, `inv_h`, normalized ranges) and reusing it via RCU autocache.

### Decision
- **Do NOT modify cubic cache** now.
- Implement a **quadratic-specific autocache** that mirrors cubic’s behavior (RCU, lock-free hits, ring eviction), but in **separate files**.
- Structure the quadratic cache so that future refactor can extract shared core without changing behavior.

### Quadratic Cache Contents
```
struct QuadraticSplineCache{T,X}
    x::X
    h::Vector{T}
    inv_h::Vector{T}
end
```
`inv_h` is kept to speed secant computation (`s_i = (y[i+1]-y[i]) * inv_h[i]`) and avoid
divisions in tight loops. It is used during **coefficient build** (both in the hot-path
`quadratic_interp!` and when constructing `QuadraticInterpolant`). The evaluation kernel
itself uses precomputed `(a,d,y)` and does not require `inv_h`.

Mirror `CubicSplineCache` naming/structure where possible (x/h fields, constructor style)
to reduce future refactor cost when extracting shared cache interfaces.

### Cache Key
- Keyed only by normalized x-grid (type-stable, range-normalized like cubic).
- BC does not affect the cache (BC only affects coefficient generation).

### Quadratic Autocache API
```
_get_quadratic_cache(x; autocache::Bool) -> QuadraticSplineCache
set_quadratic_cache_size!(n)
get_quadratic_cache_size()
clear_quadratic_cache!()
```

RCU behavior should follow cubic’s pattern:
- Lock-free cache hit
- Lock-protected miss, copy-on-write snapshot, ring eviction

---

## Zero-Allocation Strategy
Use `AdaptiveArrayPools` for per-call workspace arrays:
- `s::Vector{T}` secants (n-1)
- `d::Vector{T}` slopes (n)
- `a::Vector{T}` quadratic coeffs (n-1)

Pseudo-flow (hot path):

```
@with_pool pool begin
  cache = _get_quadratic_cache(x, autocache)
  s = similar!(pool, y, n-1)
  d = similar!(pool, y, n)
  a = similar!(pool, y, n-1)

  compute s
  compute d[1] from bc (Left/Right + Deriv1/Deriv2)
  forward or backward recurrence to fill d
  compute a

  eval loop: interval search -> quadratic kernel
end
```

**Scope note**: Keep coefficient computation *and* the evaluation loop inside the
`@with_pool` block so pool-allocated arrays remain valid throughout evaluation,
preventing hidden allocations.

---

## API Design (Consistent with Existing Interpolants)
### Functions
```
# Scalar
quadratic_interp(x, y, xi; bc=Left(Deriv2(0.0)), extrap=:none, autocache=true, deriv=0)

# Vector (allocating)
quadratic_interp(x, y, xq; bc=Left(Deriv2(0.0)), extrap=:none, autocache=true, deriv=0)

# Vector (in-place)
quadratic_interp!(out, x, y, xq; bc=Left(Deriv2(0.0)), extrap=:none, autocache=true, deriv=0)

# 2-arg callable form
quadratic_interp(x, y; bc=Left(Deriv2(0.0)), extrap=:none, autocache=true) -> QuadraticInterpolant
```

### Interpolant type
```
struct QuadraticInterpolant{T,X<:AbstractVector{T},B<:AbstractBC{T}}
    cache::QuadraticSplineCache{T,X}
    y::Vector{T}
    a::Vector{T}
    d::Vector{T}
    extrap::ExtrapVal
    bc::B
end
```
- Coefficients precomputed once
- Supports `deriv` keyword on call

### BC Handling
- Accept only: `Left(Deriv1)`, `Left(Deriv2)`, `Right(Deriv1)`, `Right(Deriv2)`
- `BCPair` is **not supported** for quadratic
- Default: `Left(Deriv2(0.0))`
  - Entry-point should reject `bc isa BCPair` early with `ArgumentError` for clarity.
  - `QuadraticInterpolant` stores the normalized BC for provenance/debugging; no runtime cost
    in evaluation (type-stable field, no dynamic dispatch).

---

## Integration Plan
### New files
- `src/quadratic_types.jl` (QuadraticSplineCache, QuadraticInterpolant)
- `src/quadratic_kernels.jl` (value/deriv kernels)
- `src/quadratic_autocache.jl` (RCU cache for quadratic)
- `src/quadratic_interp.jl` (public API + core evaluation)

### Modified files
- `src/bc_types.jl` (add `Left`/`Right` wrappers; no cubic behavior changes)
- `src/FastInterpolations.jl` (includes + exports)
- `src/derivative_view.jl` (support QuadraticInterpolant)

### Exports
```
export quadratic_interp, quadratic_interp!, QuadraticInterpolant
export QuadraticSplineCache
export Left, Right
export set_quadratic_cache_size!, get_quadratic_cache_size, clear_quadratic_cache!
```

---

## Tests
### Correctness
- Exact fit for quadratic data
- C1 continuity at knots (left/right derivative match)
- Deriv1/Deriv2 correctness vs analytic quadratic

### BC Handling
- `bc=Left(Deriv1(v))` works
- `bc=Left(Deriv2(v))` works
- `bc=Right(Deriv1(v))` works
- `bc=Right(Deriv2(v))` works
- `BCPair` rejected with ArgumentError

### Extrapolation
- `:none` out-of-domain → DomainError
- `:constant` clamps values; deriv=1/2 returns zero
- `:extension` uses boundary polynomial
- `:wrap` maps `x_max` to `x_min` and evaluates

### Performance / Allocation
- `@allocated quadratic_interp!(...) == 0` after warm-up
- autocache hit path allocation-free

### Type Stability
- `@inferred` for scalar paths and interpolant call

### DerivativeView
- `deriv1(itp::QuadraticInterpolant)` / `deriv2(...)` should call `itp(x; deriv=1/2)`
- Mirror the existing pattern used for Linear/Cubic in `derivative_view.jl`

---

## Risks & Mitigations
- **BC confusion**: Document clearly that only one endpoint BC is accepted.
- **Autocache duplication**: Quadratic cache mirrors cubic but is separate; later refactor can extract shared core.
- **Allocation regressions**: enforce `@allocated` tests and pool usage.

---

## Implementation Order
1. Add `Left` / `Right` BC wrappers in `bc_types.jl` (no cubic changes)
2. Implement quadratic cache + autocache (`quadratic_autocache.jl`)
3. Add types/kernels (`quadratic_types.jl`, `quadratic_kernels.jl`)
4. Implement API (`quadratic_interp.jl`)
5. Integrate includes/exports + derivative view
6. Add tests (`test_quadratic.jl`)

---

## Future Refactor Note
Quadratic autocache should be structured to allow future extraction of shared RCU core with cubic, but **cubic behavior must remain unchanged** in this phase.
