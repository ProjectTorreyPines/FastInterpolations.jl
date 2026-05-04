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
# Why this matters for spacing-cleanup: `_CachedVector.inv_h` has length n-1
# (only normal cells). Without this wrapper, the seam cell (idx=n, idx_R=1
# wrap) would force every eval kernel to special-case OOB lookups. With this
# wrapper, search returns `idx_R = n+1` (virtual), `_getindex(g, n+1)` yields
# `g.inner[1] + period`, normal-cell `inv_h[idx]` lookups stay safe, and the
# seam cell's `_get_h` / `_get_inv_h` are computed on-the-fly from `period`
# at the wrapper level (single branch, predicted not-seam).
#
# Include order: search.jl → periodic.jl → periodic_axis.jl → periodic_data.jl → ...
#
# See `claudedocs/TODO/periodic_grid_wrapper_design.md` for the full design
# rationale, PoC perf data (Vector hot loop = 0.99×), and migration plan.

"""
    _ExclusivePeriodicAxis{Tg, X<:AbstractVector{Tg}, Tp} <: AbstractVector{Tg}

Representation wrapper for `PeriodicBC{:exclusive, L}` non-uniform Vector
*axis* grids (companion to `_ExclusivePeriodicData` for the y/data side).

`inner` is the user's raw grid (length n, NO copy). `period` is the period
span (one period of the user's data). The wrapper reports `length(g) = n+1`
so search algorithms naturally find the seam cell, but `Base.getindex(g, i)`
forwards to `inner[i]` WITHOUT any branch — keeping hot search loops at zero
overhead vs raw `inner`.

Virtual-index access (i.e. `g[n+1]`) is provided through the `_getindex`
helper (defined below), which branches on the virtual range.

The wrapper also caches `_x_max = inner[1] + period` so `last(g)` is a single
field read on every query (matches the cost of the legacy `WrapExtrap{T}._x_max`
field read). This makes the axis a self-sufficient single source of truth for
the wrap domain `[first(g), last(g))` — `WrapExtrap` itself stays a pure tag
struct.

# Fields
- `inner::X` — user's raw non-uniform grid (length n). Can be `Vector{Tg}`
  or `_CachedVector{Tg, Tinv}`. Range inputs are NOT wrapped — they go
  directly through `_to_float_adding_endpoint` to a length-(n+1) `_CachedRange`.
- `period::Tp` — period span. Type independent of `Tg` to allow e.g. `Float64`
  grids with `Float32` period.
- `_x_max::Tg` — virtual-endpoint cache (`inner[1] + Tg(period)`); precomputed
  at construction so `Base.last(g)` is a single field read.

# Example
```julia
x = [0.0, 0.25, 0.5, 0.75]                       # raw Vector grid, length 4
g = _ExclusivePeriodicAxis(x, 1.0)               # presented as length 5
g[1], g[2], g[3], g[4]                           # 0.0, 0.25, 0.5, 0.75 (raw)
length(g) == 5                                   # virtual extension
_getindex(g, 5)                                  # 1.0 (= x[1] + period)
last(g)                                          # 1.0 (single field read)
```
"""
struct _ExclusivePeriodicAxis{Tg, X <: AbstractVector{Tg}, Tp} <: AbstractVector{Tg}
    inner::X
    period::Tp
    _x_max::Tg

    # Inner constructor: precompute `_x_max = inner[1] + period` (used as the
    # virtual seam coord and as the upper end of the wrap domain), and validate
    # `inner[end] < _x_max` (period must exceed the grid span — otherwise the
    # seam cell would collapse or overlap interior cells). Accepts any
    # `AbstractVector` inner (`Vector`, `_CachedVector`, `AbstractRange`,
    # `_CachedRange`) so the same wrapper unifies the `:exclusive` periodic
    # representation for 1D and ND zero-copy paths.
    function _ExclusivePeriodicAxis{Tg, X, Tp}(inner::X, period::Tp) where {Tg, X <: AbstractVector{Tg}, Tp}
        x_max = @inbounds inner[1] + Tg(period)
        @inbounds inner[end] < x_max || _throw_excl_axis_period_too_small(period, x_max, inner[end])
        # Cross-validate user period against Range-inferred period (Range inners
        # only — Vector inners cannot infer). Done once at wrapper construction
        # so hot-path resolvers (`_resolve_exclusive_period`, `_resolve_bc_period`)
        # can be trust-mode and pay no per-call validation tax.
        _validate_exclusive_period(inner, period)
        return new{Tg, X, Tp}(inner, period, x_max)
    end
end

@noinline _throw_excl_axis_period_too_small(period, x_max, last_x) = throw(
    ArgumentError(
        "PeriodicBC(:exclusive) period=$period places virtual endpoint at $x_max, " *
            "not after last grid point x[end]=$last_x"
    )
)

# No-op for Vector inners (period unverifiable) — caller's responsibility.
@inline _validate_exclusive_period(::AbstractVector, _) = nothing
# Range inners: cross-validate against `step × length` (= total period for the
# n-cell exclusive form). Tolerance pattern mirrors `_check_periodic_endpoints`
# in `core/periodic.jl` so the period check and the inclusive-y endpoint check
# share one numerical contract. Per-eltype dispatch:
#   - AbstractFloat / Complex{<:AbstractFloat} → `atol = 8*eps, rtol = sqrt(eps)`
#   - Integer / Rational                       → default `isapprox` (effectively `==`)
#   - Duck types (Dual, Measurement, ...)      → strict equality on the primal
@inline _validate_exclusive_period(inner::AbstractRange, period) =
    _validate_exclusive_period_impl(inner, period, eltype(inner))

@inline function _validate_exclusive_period_impl(
        inner::AbstractRange, period, ::Type{T}
    ) where {T <: AbstractFloat}
    inferred = step(inner) * length(inner)
    isapprox(T(period), T(inferred); atol = 8 * eps(T), rtol = sqrt(eps(T))) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@inline function _validate_exclusive_period_impl(
        inner::AbstractRange, period, ::Type{Complex{T}}
    ) where {T <: AbstractFloat}
    inferred = step(inner) * length(inner)
    isapprox(Complex{T}(period), Complex{T}(inferred); atol = 8 * eps(T), rtol = sqrt(eps(T))) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@inline function _validate_exclusive_period_impl(
        inner::AbstractRange, period, ::Type{T}
    ) where {T <: _PromotableValue}
    # Integer / Rational grids: exact equality (default isapprox tolerance).
    inferred = step(inner) * length(inner)
    isapprox(period, inferred) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@inline function _validate_exclusive_period_impl(inner::AbstractRange, period, ::Type)
    # Duck-type fallback (Dual, Measurement, ...): strict equality on primal.
    inferred = step(inner) * length(inner)
    _extract_primal(period) == _extract_primal(inferred) ||
        _throw_excl_axis_period_mismatch(period, first(inner), inferred)
    return nothing
end

@noinline _throw_excl_axis_period_mismatch(period, x0, inferred) = throw(
    ArgumentError(
        "PeriodicBC's period=$period conflicts with Range-inferred period = " *
            "$(x0 + inferred) - $x0 = $inferred. " *
            "Either adjust `period` or omit it for auto-inference."
    )
)

# Convenience outer constructor — type params inferred from inputs.
@inline _ExclusivePeriodicAxis(inner::AbstractVector{Tg}, period) where {Tg} =
    _ExclusivePeriodicAxis{Tg, typeof(inner), typeof(period)}(inner, period)

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
# precomputed at construction so `last(g)` is a single field read — same cost
# as the legacy `WrapExtrap{T}._x_max` lookup.
@inline Base.first(g::_ExclusivePeriodicAxis) = @inbounds g.inner[1]
@inline Base.last(g::_ExclusivePeriodicAxis) = g._x_max

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
@inline _wrap_to_domain(xq, g::_ExclusivePeriodicAxis) =
    _wrap_to_domain(xq, @inbounds(g.inner[1]), g._x_max)

# 3-arg `(g, xL, xR)` form for oneshot kernels that already have `xL`/`xR`
# in registers from search. Delegate to the inner type so `_CachedRange`
# inners use the cached `h` field (single field load — matches master's
# perf for non-wrapped Range grids). For `_CachedVector` / `Vector` inners
# the inner overload computes `xR - xL` directly, same as the wrapper-level
# fallback would.
#
# Seam correctness: at the seam, `xR == g._x_max == inner[1] + period` and
# `xL == g.inner[n]`. Delegating to `_get_h(_CachedRange, xL, xR) = x.h`
# returns the cached `step`. For correctly-supplied periods this equals
# `xR - xL` exactly (the wrapper constructor's cross-validation enforces
# `period ≈ step × length`); off-bit periods within `sqrt(eps)` rtol differ
# by ≤1 ULP — numeric noise, well below kernel precision.
@inline _get_h(g::_ExclusivePeriodicAxis, xL::Real, xR::Real) = _get_h(g.inner, xL, xR)
@inline _get_inv_h(g::_ExclusivePeriodicAxis, xL::Real, xR::Real) = _get_inv_h(g.inner, xL, xR)

# `_alpha_of` for the wrapper: defer to inner so `_CachedRange` uses cached
# `inv_h` (avoids `(R - L)` cancellation in the denominator).
@inline _alpha_of(q::Real, L::Real, R::Real, g::_ExclusivePeriodicAxis) =
    _alpha_of(q, L, R, g.inner)

# `_create_spacing` for the wrapper: defer to inner so the spacing buffer has
# `length(inner) - 1 = n - 1` entries (interior cells only — the seam cell is
# handled per-query via the wrapper-level `_get_h` seam fast-path). The default
# `_create_spacing(::AbstractVector)` would walk `1:length(g)-1 = 1:n` and read
# `g[n+1]` (the virtual seam slot), forwarding to `inner[n+1]` and tripping a
# `BoundsError`.
@inline _create_spacing(g::_ExclusivePeriodicAxis) = _create_spacing(g.inner)

# `_ExclusivePeriodicAxis` passes through `_store_grid_cached` (defined in
# cached_vector.jl). When a method-family factory wraps a Vector + `:exclusive`
# PeriodicBC grid before delegating to the persistent constructor, the
# wrapper must survive the constructor's `_store_grid_cached` call rather
# than being re-wrapped or mistakenly converted.
@inline _store_grid_cached(x::_ExclusivePeriodicAxis, ::Type{Tg}) where {Tg} = x

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
# `_resolve_axis_copied(x, bc, Tg)` — owned-copy variant for **persistent**
# interpolant constructors. Same dispatch shape as `_resolve_axis` but:
#   - bakes h/inv_h into a `_CachedVector` for Vector inputs (eval-time
#     `_get_h(x, i)` becomes cached lookup),
#   - copies user buffers (mutation-safe ownership transfer),
#   - promotes element type to `Tg`.
# Range inputs are already cached via `_to_float`/`_to_float_adding_endpoint`
# (immutable structs of scalar fields → no buffer copy needed regardless).
#
#   x type        + bc                          →  result
#   ─────────────────────────────────────────────────────────────────
#   AbstractVector + NoBC / inclusive              → _CachedVector(_convert_copy(x, Tg))
#   AbstractVector + :exclusive                    → _ExclusivePeriodicAxis(
#                                                      _CachedVector(_convert_copy(x, Tg)),
#                                                      period)
#   AbstractRange  + NoBC / inclusive              → _to_float(x, Tg)
#   AbstractRange  + :exclusive                    → _to_float_adding_endpoint(x, Tg)
#
# Pre-wrapped inputs (`_CachedVector`, `_CachedRange`, `_ExclusivePeriodicAxis`)
# delegate to wrapper-aware `_convert_copy(x, Tg)` so both type promotion
# (e.g., `_CachedVector{Float32}` → `_CachedVector{Float64}`) AND mutation-safe
# ownership are guaranteed regardless of how the wrapper was originally built.
#
# All three resolvers replace the older `_periodic_extend_1d` /
# `_*_resolve_periodic_grid` per-method helpers with a single uniform
# surface for every method-family. The same flow works for 1D oneshot
# (zero-alloc) AND persistent (cached) — just pick the right axis variant.

# ----- Surface (oneshot, zero-alloc) ---------------------------------------

@inline _resolve_axis(x::AbstractVector, ::AbstractBC) = x
@inline _resolve_axis(x::AbstractRange, ::AbstractBC) = _to_float(x, float(eltype(x)))
@inline function _resolve_axis(x::AbstractRange, bc::PeriodicBC{:exclusive})
    # Wrap the (cached, length-n) Range so the seam-fold idx_R=1 is shared
    # with the Vector path — uniform behavior in 1D and ND zero-copy.
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_to_float(x, float(eltype(x))), bc_resolved.period)
end
@inline function _resolve_axis(x::AbstractVector, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(x, bc_resolved.period)
end

@inline _resolve_data(y::AbstractArray, ::AbstractBC) = y
@inline function _resolve_data(y::AbstractArray, bc::PeriodicBC{:inclusive})
    # Inclusive: user already supplied length-(n+1) data with closed cycle.
    # Validate once at the surface (cheap; `check=false` skips for hot paths).
    # `periodic_check(bc)` reads the `C` type-parameter — zero-cost.
    periodic_check(bc) && _check_periodic_endpoints(bc, y)
    return y
end
@inline _resolve_data(y::AbstractArray, ::PeriodicBC{:exclusive}) =
    _ExclusivePeriodicData(y)

# ----- Resolve + copy (persistent interpolant) -----------------------------

@inline _resolve_axis_copied(x::AbstractVector, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _CachedVector(_convert_copy(x, Tg))
@inline _resolve_axis_copied(x::AbstractRange, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _to_float(x, Tg)
@inline function _resolve_axis_copied(x::AbstractRange, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_to_float(x, Tg), bc_resolved.period)
end
@inline function _resolve_axis_copied(x::AbstractVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    cv = _CachedVector(_convert_copy(x, Tg))
    return _ExclusivePeriodicAxis(cv, bc_resolved.period)
end
# Pre-wrapped re-entry (e.g. user passes already-wrapped axis, OR a wrapper
# threads through a builder that calls `_resolve_axis_copied` a second time).
# Each pre-wrapped type has BOTH a `<:AbstractBC` overload AND an explicit
# `<:PeriodicBC{:exclusive}` overload to resolve ambiguity against the
# `(AbstractRange|AbstractVector, PeriodicBC{:exclusive}, Tg)` entries above
# (`_CachedRange <: AbstractRange`, `_CachedVector <: AbstractVector`,
# `_ExclusivePeriodicAxis <: AbstractVector`).
#
# All re-entry paths route through wrapper-aware `_convert_copy(x, Tg)`:
#   - same Tg → delegates to `Base.copy(x)` (recursive copy of inner buffer
#     for `_CachedVector`/`_ExclusivePeriodicAxis`; same-ref for the
#     immutable `_CachedRange`),
#   - different Tg → rebuilds wrapper from a type-converted inner in a single
#     allocation pass (no double copy via intermediate plain Vector).
# This guarantees mutation safety AND honors `Tg` regardless of whether the
# user constructed the wrapper themselves with a shared buffer.
@inline _resolve_axis_copied(x::_CachedRange, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _convert_copy(x, Tg)
@inline function _resolve_axis_copied(x::_CachedRange, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_convert_copy(x, Tg), bc_resolved.period)
end
@inline _resolve_axis_copied(x::_CachedVector, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _convert_copy(x, Tg)
@inline function _resolve_axis_copied(x::_CachedVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_convert_copy(x, Tg), bc_resolved.period)
end
@inline _resolve_axis_copied(x::_ExclusivePeriodicAxis, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _convert_copy(x, Tg)
@inline _resolve_axis_copied(x::_ExclusivePeriodicAxis, ::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} =
    _convert_copy(x, Tg)

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
#
# Two BC-constrained methods exist to avoid ambiguity with the existing
# generic dispatch (`Searcher{...,<:AbstractBC}, AbstractVector, Real`) and
# the seam-specialized dispatch (`Searcher{...,<:PeriodicBC{:exclusive}},
# AbstractVector, Real`). When the grid is `_ExclusivePeriodicAxis`, the
# wrapper handles the seam — the searcher's BC seam logic is bypassed.

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

# Generic BC + wrapper: seam fast-path → policy-aware delegate to inner.
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
@inline function search_interval(
        s::Searcher{P, H, <:AbstractBC}, g::_ExclusivePeriodicAxis, xq::Real
    ) where {P, H}
    n = length(g.inner)
    @inbounds if xq >= g.inner[n]
        return n, 1, g.inner[n], g.inner[1] + g.period
    end
    idx, xL, xR = _search_interval_real(s, g.inner, xq)
    return idx, idx + 1, xL, xR
end

# `:exclusive` BC + wrapper: explicit override so this is unambiguous against
# the existing `<:PeriodicBC{:exclusive}` dispatch on plain AbstractVector.
# Wrapper takes precedence — searcher's `bc.period` is ignored (grid carries
# its own period). Same policy-aware shape as the generic-BC overload above.
@inline function search_interval(
        s::Searcher{P, H, <:PeriodicBC{:exclusive}}, g::_ExclusivePeriodicAxis, xq::Real
    ) where {P, H}
    n = length(g.inner)
    @inbounds if xq >= g.inner[n]
        return n, 1, g.inner[n], g.inner[1] + g.period
    end
    idx, xL, xR = _search_interval_real(s, g.inner, xq)
    return idx, idx + 1, xL, xR
end
