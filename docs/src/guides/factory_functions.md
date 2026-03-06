# [Factory Functions](@id factory_functions)

Factory functions provide a **single entry point** for configuring search policies, extrapolation modes, and side selection. They map concise symbols to concrete types, making the API more discoverable and ND configurations more ergonomic.

!!! tip "Zero Performance Cost"
    All factories are resolved at construction time. They return the same concrete singleton types used internally — **zero allocation, zero overhead** on hot paths.

## Overview

| Factory | Keyword | Options | Purpose |
|:--------|:--------|:--------|:--------|
| [`Search`](@ref) | `search=` | `:auto`, `:binary`, `:linear`, `:linear_binary` | Interval lookup strategy |
| [`Extrap`](@ref) | `extrap=` | `:none`, `:constant`, `:extend`, `:wrap` | Out-of-domain behavior |
| [`Side`](@ref) | `side=` | `:nearest`, `:left`, `:right` | Constant interpolation side |

Each factory also supports a **passthrough** method — if you pass an already-constructed type, it's returned unchanged:

```julia
Search(:binary)        # → BinarySearch()
Search(BinarySearch())  # → BinarySearch()  (passthrough)
```

This is useful in generic code where inputs may be either symbols or pre-constructed types.

## [Search](@id factory_search)

Controls how interpolants find the correct grid interval for a query point.

```julia
Search(:auto)            # AutoSearch() — adapts per query type (default, recommended)
Search(:binary)          # BinarySearch() — O(log n), stateless
Search(:linear_binary)   # LinearBinarySearch{8}() — linear walk with binary fallback
Search(:linear)          # LinearSearch() — expert only, no fallback (see warning below)

# LinearBinarySearch supports a keyword argument
Search(:linear_binary; linear_window=4)   # LinearBinarySearch{4}()
```

!!! tip "Most users don't need to call `Search()` at all"
    The default `AutoSearch()` handles random, sorted, and streaming patterns automatically. For sequential/ODE patterns, just pass `hint=Ref(1)` — no need to set a search policy.

!!! warning "`:linear` is for experts only"
    `LinearSearch()` has **no binary fallback** — it degrades to O(n) for non-monotonic or distant queries. Use `:linear_binary` (or just the default `:auto`) instead. See [Search Policies](@ref search_policies) for details.

!!! note "Keywords are only for `:linear_binary`"
    Passing keywords to other policies raises an `ArgumentError`:
    ```julia
    Search(:binary; linear_window=8)  # ERROR: ArgumentError
    ```

See [Search & Hints](@ref search_hints) for detailed policy descriptions and benchmarks.

## Extrap

Controls behavior when query points fall outside the data domain.

```julia
Extrap(:none)              # NoExtrap() — throw DomainError (default)
Extrap(:clamped)           # ClampExtrap() — clamp to boundary values
Extrap(:fill; value=NaN)   # FillExtrap(NaN) — fill with constant value
Extrap(:extend)            # ExtendExtrap() — extend boundary polynomial
Extrap(:wrap)              # WrapExtrap() — periodic coordinate wrapping
```

See [Extrapolation](../extrapolation.md) for visual comparisons and detailed behavior.

## Side

Controls which neighbor value to use at non-grid-point locations in constant interpolation.

```julia
Side(:nearest)  # NearestSide() — nearest neighbor, left tie-break (default)
Side(:left)     # LeftSide() — always use left (floor) value
Side(:right)    # RightSide() — use right (ceiling) value
```

## ND Multi-Arg Form

For N-dimensional interpolation, pass multiple symbols to create a **per-axis tuple** in a single call. This follows the same pattern as `DerivOp(1, 0)`:

```julia
# Instead of writing:
extrap = (ExtendExtrap(), ClampExtrap(), NoExtrap())

# Write:
extrap = Extrap(:extend, :constant, :none)

# Both produce the same Tuple{ExtendExtrap, ClampExtrap, NoExtrap}
```

All three factories support this:

```julia
# 3D example — different policy per axis
itp = cubic_interp((x, y, z), data;
    search = Search(:binary, :linear_binary, :auto),
    extrap = Extrap(:extend, :constant, :none))

# 2D constant interpolation
itp = constant_interp((x, y), data;
    side = Side(:left, :nearest))
```

!!! tip "Mix & Match"
    If one axis needs a keyword argument (e.g., custom `linear_window`), use the tuple form with single-arg factories:
    ```julia
    search = (Search(:binary), Search(:linear_binary; linear_window=4))
    ```

## Two-Tier API

Factories are a convenience layer — **direct type constructors remain fully supported**:

| Level | Style | When to Use |
|:------|:------|:------------|
| **High-level** | `Search(:binary)`, `Extrap(:extend)` | Daily usage, discoverability |
| **Low-level** | `BinarySearch()`, `ExtendExtrap()` | Library internals, extensions, maximum control |

Both levels produce identical results and have identical performance.

## Error Messages

Invalid symbols produce helpful `ArgumentError` messages listing all valid options:

```julia-repl
julia> Search(:bianry)
ERROR: ArgumentError: unknown search policy :bianry; valid options are :auto, :binary, :linear, :linear_binary

julia> Extrap(:const)
ERROR: ArgumentError: unknown extrapolation mode :const; valid options are :none, :constant, :extend, :wrap
```

## API Reference

```@docs
Search
Extrap
Side
```
