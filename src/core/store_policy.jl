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
    StorePolicy(; copy=true, copy_grid=copy, copy_values=copy, cache_axis=true)

Storage policy passed via the `store=` keyword to the persistent interpolant
constructors — `linear_interp`, `constant_interp`, `quadratic_interp`, `cubic_interp`,
`akima_interp`, `pchip_interp`, `cardinal_interp`, `hermite_interp`, and `interp`.
Controls whether the grid and value arrays are **copied** (owned, immutable — the
default) or **aliased** (referenced, zero-copy).

`copy` is the master switch; `copy_grid` / `copy_values` override per component.

`cache_axis=false` stores the grid axis **raw**, skipping the per-cell spacing
cache (`_CachedVector`'s `h`/`inv_h` arrays: n-1 divisions + two allocations);
reads fall back to on-the-fly `x[i+1]-x[i]` / `inv(h)`. Construction gets much
cheaper, queries slightly slower — intended for one-shot / integrate-heavy use
(`integrate(x, y; method)` uses it internally). Vector axes only: a Range keeps
its free, stack-only `_CachedRange`. Cubic 1D (axis lives in its spline cache)
and hetero ND ignore the flag.

# Support

`copy=false` is best-effort — it aliases what each method can and copies the rest:
- **Full** grid + value/data reference: `linear` / `constant` (1D + ND, including
  `view`s), `quadratic` / `akima` / `pchip` / `cardinal` / `hermite` (1D), and
  `interp` / cubic ND **OnTheFly**.
- **Value-only** (grid stays owned, in the spline cache): `cubic` 1D.
- **PreCompute ND** (`cubic` ND, `interp(...; coeffs=PreCompute())`) builds a derived
  partials array and keeps no raw data, so reference cannot be honored — it **warns
  once and copies**.

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
      stale for coefficient-building methods (`quadratic`/`cubic`/`akima`/`pchip`/
      `cardinal`/`hermite`), whose slopes / second-derivatives are derived from the
      values at construction.

    The rule: a stored array that feeds a build-time derived cache cannot be
    mutated without silently invalidating results.
"""
struct StorePolicy{CopyGrid, CopyValues, CacheAxis} end

@inline StorePolicy(;
    copy::Bool = true, copy_grid::Bool = copy, copy_values::Bool = copy,
    cache_axis::Bool = true,
) = StorePolicy{copy_grid, copy_values, cache_axis}()

@inline copies_grid(::StorePolicy{CG, CV, CA}) where {CG, CV, CA} = CG
@inline copies_values(::StorePolicy{CG, CV, CA}) where {CG, CV, CA} = CV
@inline caches_axis(::StorePolicy{CG, CV, CA}) where {CG, CV, CA} = CA

# ---------- storage helpers (tag-dispatched, compile-time branch) ----------

# Grid axis: `xcache` is the already-wrapped axis (`_CachedVector`/`_CachedRange`/…).
# Copy → wrapper-preserving ownership copy. Reference → alias the wrapper as-is
# (its inner buffer already aliases the caller's grid when no float conversion
# happened upstream).
@inline _own_or_ref_axis(xcache, ::Type{Tg}, ::StorePolicy{true, CV, CA}) where {Tg, CV, CA} =
    _convert_copy(xcache, Tg)
@inline _own_or_ref_axis(xcache, ::Type{Tg}, ::StorePolicy{false, CV, CA}) where {Tg, CV, CA} =
    xcache

# Axis resolution honoring `cache_axis`, split like the ctor layering:
# `_policy_axis` = type-normalization WITHOUT ownership (outer kwarg wrappers call
# it so extrap/BC resolution sees the effective axis; idempotent on re-entry);
# `_store_axis` = normalization + copy/reference — the single inner-ctor entry.
# cache_axis=false: a Vector axis stays RAW (no h/inv_h precompute; downstream
# falls back to on-the-fly diffs). A Range still wraps to the stack-only
# `_CachedRange` (nothing to skip), pre-wrapped axes pass through, and
# `:exclusive` periodic keeps its thin wrapper around the raw inner.
@inline _policy_axis(x, bc::AbstractBC, ::Type{Tg}, ::StorePolicy{CG, CV, true}) where {Tg, CG, CV} =
    _cache_axis(x, bc, Tg)
@inline _policy_axis(x::AbstractRange, bc::AbstractBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} =
    _cache_axis(x, bc, Tg)
@inline _policy_axis(x::_CachedVector, ::AbstractBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} = x
@inline _policy_axis(x::_ExclusivePeriodicAxis, ::AbstractBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} = x
@inline _policy_axis(x::AbstractVector, bc::AbstractBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} =
    _wrap_exclusive_raw(_to_eltype_vec(x, Tg), bc)
@inline _to_eltype_vec(x::AbstractVector{Tg}, ::Type{Tg}) where {Tg} = x
@inline _to_eltype_vec(x::AbstractVector, ::Type{Tg}) where {Tg} = Vector{Tg}(x)   # eltype mismatch → convert (owned)
@inline _wrap_exclusive_raw(raw, ::AbstractBC) = raw
@inline _wrap_exclusive_raw(raw, bc::PeriodicBC{:exclusive}) = _wrap_exclusive(raw, bc)

@inline _store_axis(x, bc::AbstractBC, ::Type{Tg}, store::StorePolicy) where {Tg} =
    _own_or_ref_axis(_policy_axis(x, bc, Tg, store), Tg, store)

# Tg-less `_policy_axis` (ND factories: grids arrive already value-promoted, so
# the wrap mirrors the 2-arg `_cache_axis(g, bc)` contract — no eltype work).
@inline _policy_axis(x, bc::AbstractBC, ::StorePolicy{CG, CV, true}) where {CG, CV} =
    _cache_axis(x, bc)
@inline _policy_axis(x::AbstractRange, bc::AbstractBC, ::StorePolicy{CG, CV, false}) where {CG, CV} =
    _cache_axis(x, bc)
@inline _policy_axis(x::_CachedVector, ::AbstractBC, ::StorePolicy{CG, CV, false}) where {CG, CV} = x
@inline _policy_axis(x::_ExclusivePeriodicAxis, ::AbstractBC, ::StorePolicy{CG, CV, false}) where {CG, CV} = x
@inline _policy_axis(x::AbstractVector, bc::AbstractBC, ::StorePolicy{CG, CV, false}) where {CG, CV} =
    _wrap_exclusive_raw(x, bc)

# 1D values: copy → own (+ promote to `Tv`). Reference + matching eltype → alias.
# Reference + eltype mismatch → copy fallback (keeps `Tv`, stays type-transparent).
@inline _own_or_ref_values(y::AbstractVector, ::Type{Tv}, ::StorePolicy{CG, true, CA}) where {Tv, CG, CA} =
    _convert_copy(y, Tv)
@inline _own_or_ref_values(y::AbstractVector{Tv}, ::Type{Tv}, ::StorePolicy{CG, false, CA}) where {Tv, CG, CA} =
    y
@inline _own_or_ref_values(y::AbstractVector, ::Type{Tv}, ::StorePolicy{CG, false, CA}) where {Tv, CG, CA} =
    _convert_copy(y, Tv)

# ND data: copy → materialize a fresh dense `Array` (owned). Reference → alias
# the caller's array as-is — any `AbstractArray` (dense `Array`, `SubArray`,
# reshaped, …). The interpolant's parametric `data::D` field absorbs the type.
@inline _own_or_ref_data(data::AbstractArray, ::StorePolicy{CG, true, CA}) where {CG, CA} =
    Array(data)
@inline _own_or_ref_data(data::AbstractArray, ::StorePolicy{CG, false, CA}) where {CG, CA} =
    data

# Eltype-aware form (mirrors `_own_or_ref_values`' type-transparent contract):
# alias/copy per policy only when the stored eltype already matches; a mismatch
# always promote-copies to `Array{Tv}` regardless of the policy.
@inline _own_or_ref_data(data::AbstractArray{Tv}, ::Type{Tv}, store::StorePolicy) where {Tv} =
    _own_or_ref_data(data, store)
@inline _own_or_ref_data(data::AbstractArray, ::Type{Tv}, ::StorePolicy) where {Tv} =
    Array{Tv}(data)

# ---------- unsupported-path guard ----------
# Some constructors cannot honor reference storage (e.g. PreCompute ND that
# transforms data into a derived partials array). When the caller explicitly
# asks for reference there, warn once and fall back to copy — never silently
# ignore the request, never error.
# Per-`what` `_id` gives each distinct path its own `maxlog` budget, so every
# unsupported method warns once with its own message (vs one warning total).
@noinline _warn_store_unsupported(store, what) = @warn(
    "reference storage (copy_grid=$(copies_grid(store)), copy_values=$(copies_values(store))) " *
        "is not supported for $what — falling back to copy. " *
        "See the StorePolicy docstring for the methods/strategies that support reference storage.",
    maxlog = 1,
    _id = Symbol("store_ref_unsupported_", what),
)
@inline _check_store(::StorePolicy{true, true, CA}, _) where {CA} = nothing   # copy mode — no-op (cache_axis is orthogonal)
@inline _check_store(store::StorePolicy, what) = _warn_store_unsupported(store, what)
