# [Migrating to v0.3](@id migration_v0_3)

v0.3.0 completes the transition from Symbol-based APIs to fully typed dispatch. Most migrations are mechanical find-and-replace operations — **all breaking changes produce compile-time errors**, so your test suite will catch everything.

!!! tip "Factory Functions"
    v0.3 introduces [`Search()`](@ref), [`Extrap()`](@ref), and [`Side()`](@ref) factory functions as the recommended high-level API. These are shorter to type and easier to discover than the concrete types. See [Factory Functions](@ref factory_functions) for details.

## Quick Search-Replace Reference

Run these replacements across your codebase. Order matters for partial-match safety.

### Extrapolation (Symbols → Types)

| v0.2 (removed) | v0.3 (factory) | v0.3 (direct type) |
|:----------------|:----------------|:--------------------|
| `extrap=:none` | `extrap=Extrap(:none)` | `extrap=NoExtrap()` |
| `extrap=:constant` | `extrap=Extrap(:constant)` | `extrap=ConstExtrap()` |
| `extrap=:extension` | `extrap=Extrap(:extend)` | `extrap=ExtendExtrap()` |
| `extrap=:wrap` | `extrap=Extrap(:wrap)` | `extrap=WrapExtrap()` |

!!! note
    The old symbol was `:extension` (with `-ion`); the new factory symbol is `:extend` (shorter, matches the type name better).

Also removed: `ParabolaFit()` → use `QuadraticFit()`.

### Side Selection (Symbols → Types)

| v0.2 (removed) | v0.3 (factory) | v0.3 (direct type) |
|:----------------|:----------------|:--------------------|
| `side=:nearest` | `side=Side(:nearest)` | `side=NearestSide()` |
| `side=:left` | `side=Side(:left)` | `side=LeftSide()` |
| `side=:right` | `side=Side(:right)` | `side=RightSide()` |

### Derivatives (Int/Val → DerivOp)

| v0.2 (removed) | v0.3 |
|:----------------|:------|
| `deriv=0` | `deriv=DerivOp(0)` or `EvalValue()` |
| `deriv=1` | `deriv=DerivOp(1)` or `EvalDeriv1()` |
| `deriv=2` | `deriv=DerivOp(2)` or `EvalDeriv2()` |
| `deriv=3` | `deriv=DerivOp(3)` or `EvalDeriv3()` |
| `deriv=Val(n)` | `deriv=DerivOp(n)` |
| `deriv=(1, 0)` | `deriv=DerivOp(1, 0)` |
| `deriv=Val((1, 0))` | `deriv=DerivOp(1, 0)` |

The old alias names (`EvalValue`, `EvalDeriv1`, `EvalDeriv2`, `EvalDeriv3`) remain exported and work unchanged.

### Boundary Conditions (Renames)

| v0.2 (removed) | v0.3 |
|:----------------|:------|
| `NaturalBC()` | `ZeroCurvBC()` — S''=0 at boundaries |
| `ClampedBC()` | `ZeroSlopeBC()` — S'=0 at boundaries |

**Default BC changed**: `cubic_interp(x, y)` now uses `CubicFit()` (was `ZeroCurvBC()`). To preserve the old behavior:

```julia
itp = cubic_interp(x, y; bc=ZeroCurvBC())  # explicitly specify old default
```

!!! warning "CubicFit minimum grid size"
    `CubicFit()` requires ≥ 4 grid points. For 3-point grids, use `bc=ZeroCurvBC()`.

### Search Policies (Renames)

| v0.2 (removed) | v0.3 (factory) | v0.3 (direct type) |
|:----------------|:----------------|:--------------------|
| `Binary()` | `Search(:binary)` | `BinarySearch()` |
| `Linear()` | `Search(:linear)` | `LinearSearch()` ⚠️ |
| `LinearBinary()` | `Search(:linear_binary)` | `LinearBinarySearch()` |
| `LinearBinary{4}()` | `Search(:linear_binary; linear_window=4)` | `LinearBinarySearch{4}()` |
| `HintedBinary()` | — | `LinearBinarySearch(linear_window=0)` |

!!! tip "Recommended: just use the default"
    In v0.3, `AutoSearch()` is the default for all interpolants and adapts automatically. If you were using `Linear()` / `LinearBinary()` in v0.2, consider **removing the explicit search policy entirely** and relying on `AutoSearch()` + `hint=Ref(1)` for sequential patterns. See [Search & Hints](@ref search_hints).

!!! warning "LinearSearch() is no longer recommended"
    The old `Linear()` (now `LinearSearch()`) has no binary fallback and degrades to O(n) for distant queries. Use `LinearBinarySearch()` or `AutoSearch()` (default) instead — they provide the same O(1) performance for monotonic data with a safe O(log n) fallback.

## Behavioral Changes (Non-Breaking)

### Default Search Policy: AutoSearch

The default search policy changed from `BinarySearch()` to `AutoSearch()`. Existing code compiles without changes, but **runtime behavior differs for vector queries**:

| Query Type | v0.2 Default | v0.3 Default | Effect |
|:-----------|:-------------|:-------------|:-------|
| Scalar | `BinarySearch()` | `BinarySearch()` (via AutoSearch) | No change |
| Sorted vector | `BinarySearch()` | `LinearBinarySearch()` (via AutoSearch) | **Faster** (up to ~5×) |
| Random vector | `BinarySearch()` | `LinearBinarySearch()` (via AutoSearch) | **Slower** (~2–3×) |

If you have **random** vector queries and observe slowdowns, restore v0.2 behavior explicitly:

```julia
itp = cubic_interp(x, y; search=Search(:binary))
# or
itp = cubic_interp(x, y; search=BinarySearch())
```

## Detailed Examples

### Extrapolation

```julia
# ─── v0.2 (removed) ──────────────────────────────
cubic_interp(x, y, xq; extrap=:constant)
linear_interp(x, y, xq; extrap=:extension)

# ─── v0.3 (recommended: factory) ─────────────────
cubic_interp(x, y, xq; extrap=Extrap(:constant))
linear_interp(x, y, xq; extrap=Extrap(:extend))

# ─── v0.3 (alternative: direct type) ─────────────
cubic_interp(x, y, xq; extrap=ConstExtrap())
linear_interp(x, y, xq; extrap=ExtendExtrap())

# ─── ND per-axis (new variadic form) ─────────────
cubic_interp((x, y, z), data; extrap=Extrap(:extend, :constant, :none))
```

### Side Selection

```julia
# ─── v0.2 (removed) ──────────────────────────────
constant_interp(x, y, xq; side=:left)

# ─── v0.3 ────────────────────────────────────────
constant_interp(x, y, xq; side=Side(:left))
constant_interp(x, y, xq; side=LeftSide())

# ─── ND per-axis ─────────────────────────────────
constant_interp((x, y), data; side=Side(:left, :nearest))
```

### Derivatives

```julia
# ─── v0.2 (removed) ──────────────────────────────
cubic_interp(x, y, xq; deriv=1)
itp(q; deriv=Val((1, 0)))

# ─── v0.3 ────────────────────────────────────────
cubic_interp(x, y, xq; deriv=DerivOp(1))
itp(q; deriv=DerivOp(1, 0))
```

### Boundary Conditions

```julia
# ─── v0.2 (removed) ──────────────────────────────
cubic_interp(x, y; bc=NaturalBC())
cubic_interp(x, y; bc=ClampedBC())

# ─── v0.3 ────────────────────────────────────────
cubic_interp(x, y; bc=ZeroCurvBC())    # S''=0
cubic_interp(x, y; bc=ZeroSlopeBC())   # S'=0
```

### Search Policies

```julia
# ─── v0.2 (removed) ──────────────────────────────
linear_interp(x, y, xq; search=Binary())
linear_interp(x, y, xq; search=LinearBinary())

# ─── v0.3 (recommended: just use the default) ────
linear_interp(x, y, xq)                    # AutoSearch() adapts automatically

# ─── v0.3 (explicit override, if needed) ─────────
linear_interp(x, y, xq; search=Search(:binary))
linear_interp(x, y, xq; search=Search(:linear_binary))

# ─── v0.3 (alternative: direct type) ─────────────
linear_interp(x, y, xq; search=BinarySearch())
linear_interp(x, y, xq; search=LinearBinarySearch())

# ─── ND per-axis ─────────────────────────────────
cubic_interp((x, y, z), data; search=Search(:binary, :linear_binary, :auto))
```

## Error Messages

v0.3 produces clear errors for old API usage:

```julia
julia> cubic_interp(x, y; extrap=:constant)
ERROR: MethodError: no method matching ...
# → Use Extrap(:constant) or ConstExtrap()

julia> cubic_interp(x, y, xq; deriv=1)
ERROR: TypeError: ...
# → Use DerivOp(1) or EvalDeriv1()

julia> linear_interp(x, y, xq; search=Binary())
ERROR: UndefVarError: `Binary` not defined
# → Use Search(:binary) or BinarySearch()

julia> Search(:bianry)
ERROR: ArgumentError: unknown search policy :bianry;
       valid options are :auto, :binary, :linear, :linear_binary
# → Factory functions show valid options on typos
```

## Migration Checklist

1. **Update `extrap` kwargs** — replace `:symbol` with `Extrap(:symbol)` or typed constructors
2. **Update `side` kwargs** — replace `:symbol` with `Side(:symbol)` or typed constructors
3. **Update `deriv` kwargs** — replace `Int`/`Val` with `DerivOp(n)`
4. **Update BC types** — `NaturalBC` → `ZeroCurvBC`, `ClampedBC` → `ZeroSlopeBC`
5. **Update search types** — `Binary` → `BinarySearch`, etc.
6. **Check default BC** — if you relied on the implicit `ZeroCurvBC` default, specify it explicitly
7. **Check search policy usage** — `AutoSearch()` is now the recommended default; if you were using `Linear()`, switch to `AutoSearch()` + `hint=Ref(1)` (see [Search & Hints](@ref search_hints))
8. **Run tests** — all changes produce compile-time errors, so tests catch everything
