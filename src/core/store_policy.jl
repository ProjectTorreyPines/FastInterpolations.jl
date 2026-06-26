# ========================================
# StorePolicy — copy vs reference storage for persistent interpolant inputs
# ========================================
#
# Default construction copies (and promotes) grid + values so the interpolant
# owns immutable private data. `StorePolicy` lets advanced callers opt into
# aliasing the caller's arrays instead — zero-copy, at the cost of the caller
# guaranteeing those arrays are not mutated/resized for the interpolant's
# lifetime.
#
# The choice is encoded in TYPE PARAMETERS (not a runtime Bool), so the inner
# constructor specializes on it and the stored-array choice never destabilizes
# inference.
#
# Type-transparent contract: reference aliases only when the stored element type
# is unchanged (Float / Range / dense — the dominant case). When a conversion
# would be required (e.g. an Int grid promoted to Float), the array is still
# copied; the interpolant TYPE is identical to copy mode.

"""
    StorePolicy(; copy=true, copy_grid=copy, copy_values=copy)

Storage policy for persistent interpolant constructors (`linear_interp`,
`constant_interp`). Controls whether the grid and value arrays are **copied**
(owned, immutable — the default) or **aliased** (referenced, zero-copy).

`copy` is the master switch; `copy_grid` / `copy_values` override per component.

# Examples
```julia
linear_interp(x, y)                                          # StorePolicy() — copy all (default)
linear_interp(x, y; store = StorePolicy(copy = false))       # alias grid + values (zero-copy)
linear_interp(x, y; store = StorePolicy(copy_values = false))# alias values, copy grid
```

!!! warning "Lifetime contract"
    Under `copy*=false` the caller must not mutate, resize, or free the aliased
    arrays for the interpolant's lifetime. The failure mode is method-dependent:

    - **grid** mutation always goes silently stale — spacing caches (`h`/`inv_h`),
      and any spline/slope coefficients, are snapshotted at construction.
    - **value** mutation is read live by `linear`/`constant`, but goes silently
      stale for coefficient-building methods (`cubic`/`akima`/`pchip`/`cardinal`/
      `hermite`), whose slopes / second-derivatives are derived from the values at
      construction.

    The rule: a stored array that feeds a build-time derived cache cannot be
    mutated without silently invalidating results.
"""
struct StorePolicy{CopyGrid, CopyValues} end

@inline StorePolicy(; copy::Bool = true, copy_grid::Bool = copy, copy_values::Bool = copy) =
    StorePolicy{copy_grid, copy_values}()

@inline copies_grid(::StorePolicy{CG, CV}) where {CG, CV} = CG
@inline copies_values(::StorePolicy{CG, CV}) where {CG, CV} = CV

# ---------- storage helpers (tag-dispatched, compile-time branch) ----------

# Grid axis: `xcache` is the already-wrapped axis (`_CachedVector`/`_CachedRange`/…).
# Copy → wrapper-preserving ownership copy. Reference → alias the wrapper as-is
# (its inner buffer already aliases the caller's grid when no float conversion
# happened upstream).
@inline _own_or_ref_axis(xcache, ::Type{Tg}, ::StorePolicy{true, CV}) where {Tg, CV} =
    _convert_copy(xcache, Tg)
@inline _own_or_ref_axis(xcache, ::Type{Tg}, ::StorePolicy{false, CV}) where {Tg, CV} =
    xcache

# 1D values: copy → own (+ promote to `Tv`). Reference + matching eltype → alias.
# Reference + eltype mismatch → copy fallback (keeps `Tv`, stays type-transparent).
@inline _own_or_ref_values(y::AbstractVector, ::Type{Tv}, ::StorePolicy{CG, true}) where {Tv, CG} =
    _convert_copy(y, Tv)
@inline _own_or_ref_values(y::AbstractVector{Tv}, ::Type{Tv}, ::StorePolicy{CG, false}) where {Tv, CG} =
    y
@inline _own_or_ref_values(y::AbstractVector, ::Type{Tv}, ::StorePolicy{CG, false}) where {Tv, CG} =
    _convert_copy(y, Tv)

# ND data: copy → materialize a fresh dense `Array` (owned). Reference → alias
# the caller's array as-is — any `AbstractArray` (dense `Array`, `SubArray`,
# reshaped, …). The interpolant's parametric `data::D` field absorbs the type.
@inline _own_or_ref_data(data::AbstractArray, ::StorePolicy{CG, true}) where {CG} =
    Array(data)
@inline _own_or_ref_data(data::AbstractArray, ::StorePolicy{CG, false}) where {CG} =
    data

# ---------- unsupported-path guard ----------
# Some constructors cannot honor reference storage (e.g. PreCompute ND that
# transforms data into a derived partials array). When the caller explicitly
# asks for reference there, warn once and fall back to copy — never silently
# ignore the request, never error.
# Per-`what` `_id` gives each distinct path its own `maxlog` budget, so every
# unsupported method warns once with its own message (vs one warning total).
@noinline _warn_store_unsupported(what) = @warn(
    "store=StorePolicy(copy=false) is not supported for $what — falling back to copy. " *
        "See the StorePolicy docstring for the methods/strategies that support reference storage.",
    maxlog = 1,
    _id = Symbol("store_ref_unsupported_", what),
)
@inline _check_store(::StorePolicy{true, true}, _) = nothing        # default (copy) — no-op
@inline _check_store(::StorePolicy, what) = _warn_store_unsupported(what)
