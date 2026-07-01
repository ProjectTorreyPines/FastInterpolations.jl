# ========================================
# _ExclusivePeriodicAxis: axis-side representation transform for `:exclusive` PeriodicBC
# ========================================
#
# A wrapper that takes the user's raw n-length non-uniform Vector grid plus
# period and **presents** it as if it were already in inclusive (n+1)
# canonical form:
#
#     User input form (Vector + :exclusive)  →  Internal representation
#     ────────────────────────────────────────────────────────────────
#     PeriodicBC{:exclusive, L} on a Vector  →  _ExclusivePeriodicAxis(inner, L)
#                                                length n+1 (virtual)
#
# Range inputs do NOT need this wrapper. A `:exclusive` `AbstractRange` is
# converted directly into a length-(n+1) inclusive `_CachedRange` via
# `_to_float_adding_endpoint` (cached_range.jl) — uniform spacing makes the
# extension exact. Only non-uniform `Vector` inputs benefit from a zero-copy
# wrapper, since materializing the (n+1)-th point would require a fresh
# allocation.
#
# Companion to `_ExclusivePeriodicData` (periodic_data.jl) which wraps the
# y/data side. Together they let eval kernels use plain `length(x) ==
# length(y)`, `y[idx_R]`, `last(y)` without explicit `_resolve_idx` calls —
# both wrappers report n+1 and the data wrapper auto-cycles `[n+1] → [1]`.
#
# Why "Axis" vs "Data": this wrapper carries *coordinate* info (the period
# span, the `inner[1] + period` virtual endpoint at the seam). The data
# companion is purely a cyclic-indexing wrapper with no coord semantics.
#
# Why a wrapper instead of `_CachedVector` extension: `_CachedVector.inv_h` has length n-1
# (only normal cells). Without this wrapper, the seam cell (idx=n, idx_R=1
# wrap) would force every eval kernel to special-case OOB lookups. With this
# wrapper, search returns `idx_R = n+1` (virtual), `_getindex(g, n+1)` yields
# `g.inner[1] + period`, normal-cell `inv_h[idx]` lookups stay safe, and the
# seam cell's `_get_h` / `_get_inv_h` are computed on-the-fly from `period`
# at the wrapper level (single branch, predicted not-seam).
#
# Include order: search.jl → periodic.jl → periodic_axis.jl → periodic_data.jl → ...

# `_ExclusivePeriodicAxis` struct definition + inner ctor + validation helpers
# + outer convenience ctor live in `axis_types.jl` (loaded earlier). This file
# owns the wrapper's behavior: `Base.copy` / `_convert_copy`, AbstractArray
# interface, `_get_h` / `_get_inv_h` (seam-aware), `_resolve_axis` /
# `_cache_axis(g::_ExclusivePeriodicAxis, ...)` passthroughs, view, search.

# ---------- Wrapper-preserving copy ----------
# Default `Base.copy(::AbstractVector)` uses `similar` + `copyto!` which
# materializes this wrapper as a plain `Vector{Tg}` of length `n+1`,
# destroying the periodic seam-fold contract (search returns `idx_R = 1`
# at seam → eval reads `inner[1]` directly). Persistent ND constructors
# do `map(copy, grids)` for mutation safety; without this overload, that
# call would silently break OnTheFly Hetero ND eval on `:exclusive`
# periodic axes.
#
# Recurses into `inner` to produce a fresh wrapper that owns its inner
# buffer. The `period` and cached `_x_max` are scalars (immutable copy).
@inline Base.copy(g::_ExclusivePeriodicAxis) =
    _ExclusivePeriodicAxis(copy(g.inner), g.period)

# ---------- Wrapper-aware `_convert_copy` ----------
# Same-type → delegate to `Base.copy` (recursive wrapper-preserving copy).
# Different-type → rebuild wrapper from a type-converted inner — single
# allocation pass via `_convert_copy(g.inner, T)` (which itself dispatches
# wrapper-aware on `_CachedVector`/`_CachedRange`/raw Vector).
@inline _convert_copy(g::_ExclusivePeriodicAxis{T}, ::Type{T}) where {T} = copy(g)
@inline _convert_copy(g::_ExclusivePeriodicAxis, ::Type{T}) where {T} =
    _ExclusivePeriodicAxis(_convert_copy(g.inner, T), g.period)

# ---------- AbstractVector interface ----------
# `length` reports virtual extended (n+1) so search algorithms find the seam
# cell at boundary. `Base.getindex` is **cyclic**: for `i ∈ 1:n` returns
# `inner[i]`; for `i == n+1` returns the virtual seam coord `inner[1] + period`.
# This satisfies the AbstractArray contract — every valid index 1..length(g)
# must be reachable. Generic Base algorithms (`view`, `copyto!`, `iterate`,
# broadcast, etc.) work correctly through this cyclic layer.
#
# Hot kernels that want zero-branch raw access pierce straight to `g.inner[i]`
# (caller responsible for `i ≤ length(g.inner)`). The cyclic branch here is
# essentially free in real eval patterns (~0.01 ns / access via cmov; bench
# confirmed it's bulk-SIMD-loss that's expensive, not branch prediction).
Base.length(g::_ExclusivePeriodicAxis) = length(g.inner) + 1
Base.size(g::_ExclusivePeriodicAxis) = (length(g),)
@inline Base.@propagate_inbounds function Base.getindex(g::_ExclusivePeriodicAxis, i::Int)
    n = length(g.inner)
    # Bounds: valid indices are `1:n+1`. The seam slot (`i == n+1`) returns
    # the cached `_x_max = inner[1] + period`; any other out-of-range index
    # is rejected to prevent silent off-by-one bugs (and to avoid arbitrary
    # memory reads at negative `i` from the `@inbounds` inner indexing).
    @boundscheck (1 <= i <= n + 1) || throw(BoundsError(g, i))
    @inbounds return i <= n ? g.inner[i] : g._x_max
end
@inline Base.firstindex(::_ExclusivePeriodicAxis) = 1
@inline Base.lastindex(g::_ExclusivePeriodicAxis) = length(g)
Base.eltype(::Type{<:_ExclusivePeriodicAxis{Tg}}) where {Tg} = Tg
Base.IndexStyle(::Type{<:_ExclusivePeriodicAxis}) = IndexLinear()

# `first`/`last` capture the *virtual* extended span — eval kernels read these
# directly to derive the wrap domain `[first(g), last(g))`. Without these
# overrides, `last(g)` would resolve to `g[length(g)] = g[n+1]` which raises
# BoundsError (since `Base.getindex` forwards to `inner`). `_x_max` is
# precomputed at construction so `last(g)` is a single field read.
@inline Base.first(g::_ExclusivePeriodicAxis) = @inbounds g.inner[1]
@inline Base.last(g::_ExclusivePeriodicAxis) = g._x_max

# Classification bounds: inner's widened left endpoint (so the true left endpoint
# maps to the first cell, not the seam) + the virtual seam `_x_max`. Wrap-fold
# geometry still uses the actual `inner[1]`/`_x_max`.
@inline _domain_bounds(g::_ExclusivePeriodicAxis) = (_domain_bounds(g.inner)[1], g._x_max)

# Forward `step` to inner — meaningful only when inner is a Range/`_CachedRange`
# (uniform-spacing). Vector inners will hit the inner's `MethodError(::step)`,
# which is the desired behavior: callers asking for `step` already assume a
# uniform axis.
@inline Base.step(g::_ExclusivePeriodicAxis) = step(g.inner)

# Raw data length — the count of physical knots, used by callers that need to
# distinguish "inner indices" (where `Base.getindex(g, i)` is a zero-branch
# raw passthrough) from "edge indices" (where `i+1` would land at the virtual
# seam and require `_getindex` for cyclic-aware access). Slope helpers and
# similar inner-loop kernels iterate over `1:_data_length(x)` for the
# branchless path, dispatching to the edge variant only when `i == 1` or
# `i == _data_length(x)`.
#
# For non-wrapper axes (Vector, Range, _CachedVector, _CachedRange,
# inclusive raw vector), `_data_length(x) == length(x)` — no overhead.
# For `_ExclusivePeriodicAxis`, returns `length(g.inner)` (raw n), distinct
# from `length(g) = n+1` (virtual length used by search/wrap).
@inline _data_length(x::AbstractVector) = length(x)
@inline _data_length(g::_ExclusivePeriodicAxis) = length(g.inner)

# ========================================
# `_raw`: strip wrapping for branch-free hot loops
# ========================================
#
# Hot inner loops that index strictly within `1:length(_raw(x))` should
# unwrap once via `xi = _raw(x)` and use `xi[i]` / `_get_inv_h(xi, i)`,
# avoiding the wrapper's per-call cyclic-vs-seam branch. Indices that
# touch the seam (`> length(_raw(x))`) MUST go through the wrapper.
#
# Plain vectors / ranges / `_CachedRange` / `_CachedVector` are already raw
# — `_raw(x) === x`, zero overhead. Wrapper overloads expose `.inner`.
@inline _raw(x::AbstractVector) = x
@inline _raw(g::_ExclusivePeriodicAxis) = g.inner

# ========================================
# Helpers: virtual access + ND fold-back
# ========================================
#
# `_getindex` is branch-aware for `_ExclusivePeriodicAxis`; identity for plain
# vectors. `_periodic_fold_axis!` is the adjoint-side cousin used by N-D
# scatter code to merge the virtual seam-right gradient back into the
# physical first cell.

"""
    _getindex(arr, i::Int)

Like `Base.getindex` but branch-aware for `_ExclusivePeriodicAxis`:
- Plain `AbstractVector` / Range / `_CachedRange` / `_CachedVector` → identical to
  `@inbounds arr[i]` (zero overhead).
- `_ExclusivePeriodicAxis` with `i ≤ length(g.inner)` → forward to inner.
- `_ExclusivePeriodicAxis` with `i == length(g.inner) + 1` → return the virtual
  point `g.inner[1] + g.period` (seam right endpoint).

Used by eval kernels at edge points (right endpoint of the seam cell).
"""
@inline Base.@propagate_inbounds _getindex(arr::AbstractVector, i::Int) = @inbounds arr[i]
@inline Base.@propagate_inbounds function _getindex(g::_ExclusivePeriodicAxis, i::Int)
    n = length(g.inner)
    @inbounds return i <= n ? g.inner[i] : g._x_max
end

"""
    _periodic_fold_axis!(arr, dim::Int, n_period::Int)

Fold the seam adjoint contribution into the cyclic-1 cell along axis `dim`:

    arr[..., 1, ...] += arr[..., n_period + 1, ...]

Used by adjoint code to merge the virtual seam-right gradient back into the
physical first cell. Generalizes to N-D via `selectdim`.

For non-periodic axes, callers branch on `grids[dim] isa _ExclusivePeriodicAxis`
before calling this helper.
"""
@inline function _periodic_fold_axis!(arr, dim::Int, n_period::Int)
    selectdim(arr, dim, 1) .+= selectdim(arr, dim, n_period + 1)
    return arr
end

"""
    _check_compatible_length(x, y)

Throw `ArgumentError` via `_throw_length_mismatch` if the grid `x` and value
container `y` have incompatible lengths. Single generic method based on
`length` since both `_ExclusivePeriodicAxis` (axis) and `_ExclusivePeriodicData`
(value, defined in periodic_data.jl) report their *virtual* `n+1` length —
so `length(x) == length(y)` is correct uniformly across plain vectors and
wrapped pairs.

Per-pair dispatch isn't needed at this layer; if a future asymmetric pairing
needs special handling, it can add an additional method without disturbing
existing callsites.
"""
@inline function _check_compatible_length(x::AbstractVector, y::AbstractArray)
    length(x) == length(y) || _throw_length_mismatch(length(x), length(y))
    return nothing
end

# ========================================
# Spacing accessors: idx-aware with seam fast-path
# ========================================
#
# `_get_h(g, idx)` / `_get_inv_h(g, idx)` 2-arg form work uniformly across
# all grid types. For `_ExclusivePeriodicAxis`, the wrapper branches on idx:
#   - idx < n  → delegate to `inner` (cached lookup for `_CachedVector`,
#                on-the-fly for raw `Vector`)
#   - idx == n → seam cell: compute width as `_x_max - inner[idx]` (single
#                sub from the pre-cached `_x_max`). For Vector inners, this
#                is the natural seam width. For Range inners, see the
#                `_CachedRange`-specific overload below.
@inline Base.@propagate_inbounds _get_h(g::_ExclusivePeriodicAxis, idx::Int) =
    idx < length(g.inner) ? _get_h(g.inner, idx) : @inbounds(g._x_max - g.inner[idx])
@inline Base.@propagate_inbounds _get_inv_h(g::_ExclusivePeriodicAxis, idx::Int) =
    idx < length(g.inner) ? _get_inv_h(g.inner, idx) : @inbounds(inv(g._x_max - g.inner[idx]))

# No `_CachedRange`-specific specialization for `_get_h` / `_get_inv_h`: the
# generic wrapper overload above already does the right thing for both
# `_CachedRange` and `_CachedVector` / `Vector` inners — interior cells
# delegate to the inner type (cached field for `_CachedRange`, cached array
# for `_CachedVector`, on-the-fly for raw `Vector`), and the seam uses
# `_x_max - inner[idx]` uniformly. The user-supplied period is honored
# whether it auto-infers to `step*n` (within `sqrt(eps)` tolerance) or
# carries an off-bit offset; the seam width matches the cubic solver's
# baked `h_n` for both Range and Vector inners.

# `_wrap_to_domain` wrapper-specific overload — bypass the `first(x)` /
# `last(x)` method dispatch chain that the generic `(::AbstractVector)` form
# in periodic.jl goes through. Saves one method-call layer in WrapExtrap eval
# kernels, which matters on the constant rng+perEx persistent path where the
# baseline is only 3-4 ns.
#   - `g.inner[1]` instead of `first(g)` → `inner.lo` directly (no Base.first
#     dispatch through wrapper → inner getindex chain).
#   - `g._x_max` → cached field, single load.
@inline function _wrap_to_domain(xq, g::_ExclusivePeriodicAxis)
    # In-domain by the inner's widened bounds → no fold (the true left endpoint
    # must stay in the first cell, not fold to the seam). Only genuinely-OOB
    # queries fold against the actual seam span `[inner[1], _x_max]`.
    _is_inbounds(g, xq) && return xq
    return _wrap_to_domain(xq, @inbounds(g.inner[1]), g._x_max)
end

# 4-arg `(g, idx, xL, xR)` form. Search already produced all four; dispatch
# picks the per-type cheapest path.
#   - `idx < length(g.inner)` (interior cell): delegate to inner — uses
#      `_CachedRange.h`/`_CachedVector.h[idx]` cache.
#   - `idx == length(g.inner)` (seam cell): inner's idx-lookup would index
#      out of `inv_h` (length n-1). Use `xR - xL` directly — caller already
#      has `xR == g._x_max` from search, so this is the natural seam width.
@inline Base.@propagate_inbounds _get_h(g::_ExclusivePeriodicAxis, idx::Int, xL::Real, xR::Real) =
    idx < length(g.inner) ? _get_h(g.inner, idx, xL, xR) : xR - xL
@inline Base.@propagate_inbounds _get_inv_h(g::_ExclusivePeriodicAxis, idx::Int, xL::Real, xR::Real) =
    idx < length(g.inner) ? _get_inv_h(g.inner, idx, xL, xR) : inv(xR - xL)

# `_alpha_of` for the wrapper: seam-aware so the value computation shares a
# denominator with the 4-arg `_get_inv_h` at the seam cell.
#   - Interior cell (`R != g._x_max`): defer to inner so `_CachedRange` uses
#     its cached `inv_h` field (avoids `(R - L)` cancellation in the denom).
#   - Seam cell (`R == g._x_max`): use the actual seam width `R - L`,
#     matching `_get_inv_h(g, idx, xL, xR) = inv(xR - xL)` at `idx == n`.
#
# Without the seam branch, a Range inner with an explicit off-tolerance period
# (within `sqrt(eps)` rtol of `step*length`, accepted by `_validate_exclusive_period`)
# computed `α` from inner step but derivative from `inv(xR - xL)`, leaving
# value/derivative on disagreeing denominators. Relative error scales as
# `n * sqrt(eps)`, so Float32 grids at n ~ 100 or Float64 grids at n ~ 10⁶
# could show percent-level seam-cell discrepancy.
#
# Float equality on `_x_max` is exact: the wrapper ctor enforces
# `inner[end] < _x_max`, so `inner[idx+1] == _x_max` is unreachable for
# interior cells. At the seam, the wrapper's `search_interval` returns
# `xR = g._x_max` by direct field read — bit-equal to the comparand here.
@inline _alpha_of(q::Real, L::Real, R::Real, g::_ExclusivePeriodicAxis) =
    R == g._x_max ? (q - L) / float(R - L) : _alpha_of(q, L, R, g.inner)

# ========================================
# View specialization: preserve wrapper for full-virtual range
# ========================================
#
# `view(g, 1:n+1)` on the wrapper would otherwise produce a `SubArray{T, 1,
# _ExclusivePeriodicAxis, ...}` that loses the wrapper-as-axis identity:
# downstream `_resolve_exclusive_period` no longer sees a `_ExclusivePeriodicAxis`
# and tries to infer period from a `SubArray` (no `step`, no Range trait) →
# ArgumentError on Vector inners or non-uniform-Range inners.
#
# When the requested range covers the full virtual length, return the wrapper
# itself (zero-cost identity). For partial ranges that stay within the raw
# inner indices, fall through to a raw view of `g.inner` (loses wrapper
# identity but is correct since the slice doesn't include the seam). For
# partial ranges that include the virtual seam slot (`last(r) == length(g)`
# but the range is not full), build a `SubArray` over the wrapper so the
# seam access resolves through our cyclic `Base.getindex` instead of
# touching `g.inner` past its raw length.
@inline function Base.view(g::_ExclusivePeriodicAxis, r::AbstractUnitRange{Int})
    @boundscheck (first(r) >= 1 && last(r) <= length(g)) || throw(BoundsError(g, r))
    if first(r) == 1 && last(r) == length(g)
        return g
    elseif last(r) <= length(g.inner)
        return @inbounds view(g.inner, r)
    else
        return SubArray(g, (r,))
    end
end
@inline Base.view(g::_ExclusivePeriodicAxis, ::Colon) = g


# ========================================
# Surface-API resolvers — used by ALL method-family factories (1D)
# ========================================
#
# `_resolve_axis(x, bc)` — zero-alloc reference-wrapping for the x/axis side.
# Reshapes the user input into the canonical axis form for this BC, all by
# composing existing primitives:
#
#   x type        + bc                          →  result
#   ─────────────────────────────────────────────────────────────────
#   AbstractVector + NoBC / PeriodicBC{:inclusive} → x (passthrough, zero-alloc)
#   AbstractVector + PeriodicBC{:exclusive}        → _ExclusivePeriodicAxis(x, period)
#                                                       (zero-alloc reference wrap)
#   AbstractRange  + NoBC / PeriodicBC{:inclusive} → _CachedRange{T} (stack alloc)
#   AbstractRange  + PeriodicBC{:exclusive}        → _CachedRange{T} of length n+1
#                                                       (stack alloc, exact extension)
#
# `_resolve_data(y, bc)` — zero-alloc reference-wrapping for the y/data side:
#
#   y type        + bc                          →  result
#   ─────────────────────────────────────────────────────────────────
#   AbstractArray  + NoBC                          → y (passthrough)
#   AbstractArray  + PeriodicBC{:inclusive}        → y (passthrough; endpoint-checked)
#   AbstractArray  + PeriodicBC{:exclusive}        → _ExclusivePeriodicData(y) (zero-alloc)
#
# For persistent-storage callers that need owned + eltype-promoted axes, the
# canonical pattern is `_cache_axis(_convert_copy(x, Tg), bc)` — copy first
# (one buffer copy + eltype conversion), then wrap (aliases the fresh copy,
# builds h/inv_h once). Both resolvers replace the older `_periodic_extend_1d` /
# `_*_resolve_periodic_grid` per-method helpers with a single uniform surface
# for every method-family.

# ----- Surface (oneshot, zero-alloc) ---------------------------------------
#
# Non-`:exclusive` `_resolve_axis` overloads live next to their type definitions
# (`cached_range.jl` for Range / `_CachedRange`, `cached_vector.jl` for Vector /
# `_CachedVector`). Below: `:exclusive` overloads (produce `_ExclusivePeriodicAxis`)
# + the `_ExclusivePeriodicAxis` passthrough.

# `_resolve_axis(::AbstractRange/AbstractVector, ::PeriodicBC{:exclusive})`
# live in `cached_range.jl` / `cached_vector.jl` (owner files).

@inline _resolve_data(y::AbstractArray, ::AbstractBC) = y
@inline function _resolve_data(y::AbstractArray, bc::PeriodicBC{:inclusive})
    # Inclusive: user already supplied length-(n+1) data with closed cycle.
    # Validate once at the surface (cheap; `check=false` skips for hot paths).
    # `periodic_check(bc)` reads the `C` type-parameter — zero-cost.
    periodic_check(bc) && _check_periodic_endpoints(bc, y)
    return y
end
# `:extended` is constructed by `_bc_after_extend` with `C=false` pinned, so
# `periodic_check(bc)` is always `false` — passthrough is the entire behavior.
@inline _resolve_data(y::AbstractArray, ::PeriodicBC{:extended}) = y
@inline _resolve_data(y::AbstractArray, ::PeriodicBC{:exclusive}) =
    _ExclusivePeriodicData(y)

# ----- Caching wrap (persistent surface API, zero-copy of buffer) ----------
#
# `_cache_axis(x, bc)` is the persistent-path counterpart to one-shot's
# `_resolve_axis(x, bc)`. Same dispatch shape; the difference is that for
# raw `Vector` inputs it allocates the cached `h`/`inv_h` arrays (so the
# stored axis supports O(1) `_get_h(x, i)` lookup downstream) while still
# *aliasing* the user's data buffer in `_CachedVector.inner`. The inner
# constructor's `_convert_copy(x, Tg)` is responsible for the ownership
# copy of `inner` plus any element-type promotion. Persistent builders that
# need owned + eltype-promoted axes in a single pass use the canonical
# `_cache_axis(_convert_copy(x, Tg), bc)` pattern: `_convert_copy` does one
# buffer copy + eltype conversion, then `_cache_axis` aliases that fresh
# buffer and builds h/inv_h once.
#
#   x type        + bc                          →  result
#   ─────────────────────────────────────────────────────────────────
#   AbstractVector + NoBC / inclusive              → _CachedVector(x)
#                                                      (aliases x in `inner`,
#                                                       allocates fresh h/inv_h)
#   AbstractVector + :exclusive                    → _ExclusivePeriodicAxis(
#                                                      _CachedVector(x), period)
#   AbstractRange  + NoBC / inclusive              → _to_float(x, float(eltype(x)))
#                                                      (immutable `_CachedRange`,
#                                                       no buffer alloc)
#   AbstractRange  + :exclusive                    → _ExclusivePeriodicAxis(
#                                                      _to_float(x, ...), period)
#
# Pre-wrapped inputs (`_CachedVector`, `_CachedRange`, `_ExclusivePeriodicAxis`)
# are idempotent passthroughs — wrapping is already done; the inner ctor's
# `_convert_copy` handles ownership transfer + optional type conversion.

# `_ExclusivePeriodicAxis`-specific resolvers — wrapper passthroughs.
# Cross-type `:exclusive` dispatches (raw Range/Vector → wrapper, pre-wrapped
# `_CachedRange`/`_CachedVector` → wrapper) live in their owner files
# (`cached_range.jl` / `cached_vector.jl`).
@inline _resolve_axis(g::_ExclusivePeriodicAxis) = g
@inline _cache_axis(g::_ExclusivePeriodicAxis) = g
@inline _cache_axis(g::_ExclusivePeriodicAxis, ::AbstractBC) = g
@inline _cache_axis(g::_ExclusivePeriodicAxis, ::PeriodicBC{:exclusive}) = g

# ─────────────────────────────────────────────────────────────────────────────
# Tg-aware 3-arg overloads (`_cache_axis(x, bc, Tg)`)
# ─────────────────────────────────────────────────────────────────────────────
# Outer surface APIs (`linear_interp`, `constant_interp`, etc.) compute the
# promoted `Tg = _promote_grid_float(eltype(x), eltype(y))` *before* calling
# `_cache_axis`, then thread `Tg` here so Range inputs convert directly to the
# correct float type. Without this, `Int` ranges + `Float32` data silently
# produced `_CachedRange{Float64}` because the 2-arg path defaulted to
# `float(eltype(x)) = Float64` for integer eltypes — and the inner ctor's
# `_promote_grid_float(Float64, Float32) = Float64` then widened `y` to
# `Float64` too, breaking the documented Float32 promotion.
#
# DISPATCH TABLE (and exactly *which* paths respect `Tg`):
#
#   x type                 +  bc                       Tg respected?  result
#   ───────────────────────────────────────────────────────────────────────────
#   AbstractVector         +  AbstractBC                ✅            `_CachedVector{Tg}`
#   AbstractVector         +  PeriodicBC{:exclusive}    ✅            `_ExclusivePeriodicAxis(_CachedVector{Tg}, period)`
#   AbstractRange          +  AbstractBC                ✅            `_CachedRange{Tg}`
#   AbstractRange          +  PeriodicBC{:exclusive}    ✅            `_ExclusivePeriodicAxis(_CachedRange{Tg}, period)`
#   _CachedRange{S}        +  any                       ⚠️ IGNORED    passthrough (eltype S preserved)
#   _CachedVector{S}       +  any                       ⚠️ IGNORED    passthrough or `:exclusive` wrap (eltype S preserved)
#   _ExclusivePeriodicAxis +  any                       ⚠️ IGNORED    passthrough (eltype unchanged)
#
# Why pre-wrapped paths ignore `Tg` (INTENTIONAL CONTRACT):
#   Pre-wrapped axes have *already committed* their element type via their
#   cached `h`/`inv_h` buffers. Re-converting eltype would require dropping
#   and rebuilding those caches — work that belongs to `_convert_copy(_, Tg)`,
#   not to `_cache_axis`. The canonical persistent-build pattern is:
#
#       xc = _convert_copy(_cache_axis(x, bc, Tg), Tg)
#
#   For raw `x`: `_cache_axis` returns a Tg-typed wrapper, then `_convert_copy`
#   degrades to a same-eltype `Base.copy` (one buffer copy, no eltype work).
#   For pre-wrapped `x` whose eltype already matches `Tg`: same — passthrough +
#   identity copy. For pre-wrapped `x` with mismatching eltype: `_cache_axis`
#   passes it through unchanged, then `_convert_copy(_, Tg)` rebuilds with the
#   new eltype (one allocation pass — fastest possible for this rare case).
#
#   In short: `_cache_axis(x, bc, Tg)` guarantees only that ROOT raw inputs
#   become `Tg`-typed wrappers. Pre-wrapped inputs round-trip unchanged so the
#   downstream `_convert_copy` can do the eltype work in a single pass.
# `_ExclusivePeriodicAxis` passthroughs for 3-arg form.
# Raw `AbstractRange`/`AbstractVector` + `:exclusive` + Tg, and pre-wrapped
# `_CachedRange`/`_CachedVector` + `:exclusive` + Tg live in their owner files.
@inline _cache_axis(g::_ExclusivePeriodicAxis, bc::AbstractBC, ::Type{Tg}) where {Tg} = _cache_axis(g, bc)
@inline _cache_axis(g::_ExclusivePeriodicAxis, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} = _cache_axis(g, bc)

# ========================================
# Search policy: preserve inner's trait
# ========================================
#
# `_resolve_search_policy(::AbstractRange, ...)` returns `DirectSearch()` for
# the O(1) Range fast-path; the `AbstractVector` fallback returns
# Binary/LinearBinary. `_ExclusivePeriodicAxis <: AbstractVector`, so without
# this override a wrapped `_CachedRange` inner would lose its Range trait and
# fall through to BinarySearch (5–13× regression at N=10²–10⁴ on the
# Per-excl Range path). Delegating to `g.inner` lets `_CachedRange` /
# `AbstractRange` inners surface their `DirectSearch` while `_CachedVector` /
# `Vector` inners still resolve to Binary/LinearBinary.
@inline _resolve_search_policy(g::_ExclusivePeriodicAxis, xq, search::AbstractSearchPolicy, hint) =
    _resolve_search_policy(g.inner, xq, search, hint)

# ========================================
# Specialized search: zero-overhead seam fast-path
# ========================================
#
# The reason "wrapper-based unification" works WITHOUT perf cost: inside the
# search loop, accesses go through `Base.getindex` on `g.inner` (raw Vector
# or _CachedVector). NO branch inserted by the wrapper. The seam check is
# hoisted ONCE before search — same cost as a single `<:PeriodicBC{:exclusive>`
# dispatcher invocation in current master.

"""
    _search_binary(g::_ExclusivePeriodicAxis, xq) -> (idx, xL, xR)

Binary search on `_ExclusivePeriodicAxis`. ONE upfront seam check; if the
query is past `g.inner[n]`, return the seam tuple `(n, x[n], x[1]+period)`.
Otherwise delegate to standard binary search on the raw inner Vector — zero
wrapper overhead for the hot loop.
"""
@inline function _search_binary(g::_ExclusivePeriodicAxis{T}, xq::Real) where {T}
    n = length(g.inner)
    @inbounds if xq >= g.inner[n]
        return n, g.inner[n], g.inner[1] + g.period
    end
    return _search_binary(g.inner, xq)
end

# ========================================
# search_interval entry point: 4-tuple return
# ========================================
#
# Matches the existing `search_interval(s, x, xq)` shape so eval kernels can
# unpack `(idx_L, idx_R, xL, xR)` uniformly. For `_ExclusivePeriodicAxis`,
# `idx_R` is **post-fold**: at the seam cell (`idx_L == n`), `idx_R = 1` instead
# of the virtual `n+1`. This lets ND eval kernels read raw `data[..., idx_R, ...]`
# directly without needing an N-D periodic data wrapper. The 1D `:exclusive`
# path keeps `_ExclusivePeriodicData` for `last(y)` semantics, but `y[idx_R=1]`
# yields the same value as `y[idx_R=n+1]` through the cyclic wrapper.

# GridIdx queries — explicit index semantics, no query wrapping. Mirrors
# `search_interval(s, x::AbstractVector, ::GridIdx)` in search.jl, but with
# the seam-fold contract of `_ExclusivePeriodicAxis`: when the resolved cell
# is the seam (`idx == n`), `idx_R` folds to `1` so eval kernels can read
# `_raw(y)[idx_R]` safely (the raw inner has length n; `idx_R = n+1` would be
# OOB). The seam-fold here is the SAME contract as the Real-query path
# below — only the search itself bypasses the `xq >= g.inner[n]` fast-path
# (no wrap of the explicit user index).
@inline function search_interval(s::Searcher, g::_ExclusivePeriodicAxis, xq::GridIdx)
    n = length(g.inner)
    idx, xL, xR = _search_grididx_dispatch(s.hint, g, xq)
    idx_R = idx < n ? idx + 1 : 1
    return idx, idx_R, xL, xR
end

# Real-query wrapper dispatch: seam fast-path → policy-aware delegate to inner.
#
# Crucially, we MUST route the interior search through `_search_interval_real`
# on the inner — NOT through `_search_binary(g, xq)`. The unconditional binary
# fallback would discard the `Searcher`'s policy parameter `P`: when inner is
# `_CachedRange` (or any `AbstractRange`) and the resolved policy is
# `DirectSearch`, the inner's specialized `_search_interval_real(::Searcher{
# DirectSearch}, ::AbstractRange, ::Real)` overload computes the cell index
# in O(1) (single mul/floor/clamp). Forcing `_search_binary` instead would
# downgrade Range axes to O(log n), which scales as the bench shows
# (Range Per-excl: 3.5 ns @ N=10 → 44 ns @ N=10000).
@inline function search_interval(s::Searcher, g::_ExclusivePeriodicAxis, xq::Real)
    n = length(g.inner)
    @inbounds if xq >= g.inner[n]
        # Seam-cell write-back: `_search_interval_real` is the path that
        # normally updates `RefHint.idx[]` (LinearSearch/LinearBinarySearch
        # consume the hint on the next call). Short-circuiting here without
        # writing leaves the hint stuck at a stale interior index, breaking
        # O(1) locality for monotone hinted streaming across the seam.
        _write_hint!(s.hint, n)
        return n, 1, g.inner[n], g.inner[1] + g.period
    end
    idx, xL, xR = _search_interval_real(s, g.inner, xq)
    return idx, idx + 1, xL, xR
end

# InBounds fast path must NOT bypass the seam. The generic
# `search_interval(s, x::AbstractVector, xq, ::InBounds)` (search.jl) routes any `AbstractVector`
# — including `_ExclusivePeriodicAxis` — into `_search_interval_real_inbounds`, which delegates to
# `_search_interval_real(s, g, xq)`: undefined for the periodic axis (the *inner* range is the
# searchable object, and a seam query needs the wrap cell). A periodic axis is never genuinely
# `InBounds`-lean (WrapExtrap semantics), so route it to the seam-aware 3-arg search verbatim —
# restoring the pre-lean-work dispatch. Reached e.g. from `_cubic_interp_periodic_scalar`'s
# in-bounds branch, which passes `InBounds()` on a `_ExclusivePeriodicAxis` grid.
@inline search_interval(s::Searcher, g::_ExclusivePeriodicAxis, xq::Real, ::InBounds) =
    search_interval(s, g, xq)
