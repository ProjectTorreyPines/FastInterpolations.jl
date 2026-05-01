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

    # Inner constructor: precompute `_x_max = inner[1] + period` and validate
    # `inner[end] < _x_max` (period must exceed the grid span — otherwise the
    # seam cell would collapse or overlap interior cells). Accepts any
    # `AbstractVector` inner (`Vector`, `_CachedVector`, `AbstractRange`,
    # `_CachedRange`) so the same wrapper unifies the `:exclusive` periodic
    # representation for 1D and ND zero-copy paths.
    function _ExclusivePeriodicAxis{Tg, X, Tp}(inner::X, period::Tp) where {Tg, X <: AbstractVector{Tg}, Tp}
        x_max = @inbounds inner[1] + Tg(period)
        @inbounds inner[end] < x_max || _throw_excl_axis_period_too_small(period, x_max, inner[end])
        return new{Tg, X, Tp}(inner, period, x_max)
    end
end

@noinline _throw_excl_axis_period_too_small(period, x_max, last_x) = throw(
    ArgumentError(
        "PeriodicBC(:exclusive) period=$period places virtual endpoint at $x_max, " *
            "not after last grid point x[end]=$last_x"
    )
)

# Convenience outer constructor — type params inferred from inputs.
@inline _ExclusivePeriodicAxis(inner::AbstractVector{Tg}, period) where {Tg} =
    _ExclusivePeriodicAxis{Tg, typeof(inner), typeof(period)}(inner, period)

# ---------- AbstractVector interface ----------
# `length` reports virtual extended (n+1) so search algorithms find the seam
# cell at boundary. `getindex` forwards to inner WITHOUT branch — accessing
# `g[n+1]` raises BoundsError from inner. Use `_getindex` helper for virtual.
Base.length(g::_ExclusivePeriodicAxis) = length(g.inner) + 1
Base.size(g::_ExclusivePeriodicAxis) = (length(g),)
@inline Base.@propagate_inbounds Base.getindex(g::_ExclusivePeriodicAxis, i::Int) =
    g.inner[i]
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
    @inbounds return i <= n ? g.inner[i] : g.inner[1] + g.period
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
# all grid types after Step 1.5. For `_ExclusivePeriodicAxis`, the wrapper
# branches on idx:
#   - idx < n  → delegate to `inner` (cached lookup for `_CachedVector`,
#                on-the-fly for raw `Vector`)
#   - idx == n → seam cell: compute on-the-fly from `period`
#                (`h = inner[1] + period - inner[n]`, `inv_h = inv(h)`)
#
# Branch is highly predictable (seam fires at most once per query batch),
# and the wrapper-level branch keeps inner-level dispatches simple.
@inline Base.@propagate_inbounds function _get_h(g::_ExclusivePeriodicAxis, idx::Int)
    n = length(g.inner)
    @inbounds return idx < n ? _get_h(g.inner, idx) : g.inner[1] + g.period - g.inner[n]
end
@inline Base.@propagate_inbounds function _get_inv_h(g::_ExclusivePeriodicAxis, idx::Int)
    n = length(g.inner)
    @inbounds return idx < n ? _get_inv_h(g.inner, idx) : inv(g.inner[1] + g.period - g.inner[n])
end

# Range-inner specialization: every cell (interior + seam) has the cached step
# width, so dispatch to the inner's cached `h`/`inv_h` regardless of idx. This
# avoids the seam-cell `inner[1] + period - inner[n]` cancellation on
# large-offset Ranges (e.g. `range(1e8, step=0.1, ...)`), and matches the
# precision of the persistent extended-Range path bit-for-bit.
@inline _get_h(g::_ExclusivePeriodicAxis{Tg, <:_CachedRange}, ::Int) where {Tg} = g.inner.h
@inline _get_inv_h(g::_ExclusivePeriodicAxis{Tg, <:_CachedRange}, ::Int) where {Tg} = g.inner.inv_h

# 3-arg `(xL, xR)` overloads for the wrapper: delegate to cached inner so
# eval kernels using the legacy `_get_h(x, xL, xR)` shape (linear/cubic
# `_eval_at_point` fallbacks) avoid `xR - xL` cancellation. For non-Range
# inner (`_CachedVector`/`Vector`), the inner's 3-arg form falls through to
# `xR - xL` for non-seam cells, but per-cell `h` from `_CachedVector` is the
# typical path; the seam cell at large offset still suffers cancellation
# unless callers switch to the 2-arg index-based dispatch.
@inline _get_h(g::_ExclusivePeriodicAxis{Tg, <:_CachedRange}, ::Real, ::Real) where {Tg} = g.inner.h
@inline _get_inv_h(g::_ExclusivePeriodicAxis{Tg, <:_CachedRange}, ::Real, ::Real) where {Tg} = g.inner.inv_h

# `_alpha_of` for the wrapper: defer to inner so `_CachedRange` uses cached
# `inv_h` (avoids `(R - L)` cancellation in the denominator).
@inline _alpha_of(q::Real, L::Real, R::Real, g::_ExclusivePeriodicAxis) =
    _alpha_of(q, L, R, g.inner)

# `_ExclusivePeriodicAxis` passes through `_store_grid_cached` (defined in
# cached_vector.jl). When a method-family factory wraps a Vector + `:exclusive`
# PeriodicBC grid before delegating to the persistent constructor, the
# wrapper must survive the constructor's `_store_grid_cached` call rather
# than being re-wrapped or mistakenly converted.
@inline _store_grid_cached(x::_ExclusivePeriodicAxis, ::Type{Tg}) where {Tg} = x


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
# `_caching_axis(x, bc, Tg)` — caching variant for **persistent** interpolant
# constructors. Same shape as `_resolve_axis` but bakes h/inv_h into a
# `_CachedVector` for Vector inputs (so eval-time `_get_h(x, i)` is cached
# lookup), and converts/copies for type promotion. Range inputs are
# already cached via `_to_float`/`_to_float_adding_endpoint` → identical
# treatment to oneshot.
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

# ----- Caching (interpolant) -----------------------------------------------

@inline _caching_axis(x::AbstractVector, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _CachedVector(_convert_copy(x, Tg))
@inline _caching_axis(x::AbstractRange, ::AbstractBC, ::Type{Tg}) where {Tg} =
    _to_float(x, Tg)
@inline function _caching_axis(x::AbstractRange, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_to_float(x, Tg), bc_resolved.period)
end
@inline function _caching_axis(x::AbstractVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    cv = _CachedVector(_convert_copy(x, Tg))
    return _ExclusivePeriodicAxis(cv, bc_resolved.period)
end
# Idempotent passthroughs (handle re-entry, e.g. wrapper passed back through).
# Each cached/wrapped type has BOTH a `<:AbstractBC` passthrough AND an explicit
# `<:PeriodicBC{:exclusive}` overload to resolve ambiguity against the
# `(AbstractRange|AbstractVector, PeriodicBC{:exclusive}, Tg)` entries above
# (`_CachedRange <: AbstractRange`, `_CachedVector <: AbstractVector`,
# `_ExclusivePeriodicAxis <: AbstractVector`).
@inline _caching_axis(x::_CachedRange, ::AbstractBC, ::Type{Tg}) where {Tg} = x
@inline function _caching_axis(x::_CachedRange, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(x, bc_resolved.period)
end
@inline _caching_axis(x::_CachedVector, ::AbstractBC, ::Type{Tg}) where {Tg} = x
@inline function _caching_axis(x::_CachedVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(x, bc_resolved.period)
end
@inline _caching_axis(x::_ExclusivePeriodicAxis, ::AbstractBC, ::Type{Tg}) where {Tg} = x
@inline _caching_axis(x::_ExclusivePeriodicAxis, ::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} = x

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

# GridIdx queries — bypass the seam fast-path entirely (user-supplied explicit
# index semantics, no wrap). Mirrors `search_interval(s, x::AbstractVector, ::GridIdx)`
# in search.jl. Required to avoid Aqua method-ambiguity warnings.
@inline function search_interval(s::Searcher, g::_ExclusivePeriodicAxis, xq::GridIdx)
    idx, xL, xR = _search_grididx_dispatch(s.hint, g, xq)
    return idx, idx + 1, xL, xR
end

# Generic BC + wrapper: always use grid-type seam handling.
@inline function search_interval(
        s::Searcher{P, H, <:AbstractBC}, g::_ExclusivePeriodicAxis, xq::Real
    ) where {P, H}
    idx, xL, xR = _search_binary(g, xq)
    n = length(g.inner)
    idx_R = ifelse(idx == n, 1, idx + 1)
    return idx, idx_R, xL, xR
end

# `:exclusive` BC + wrapper: explicit override so this is unambiguous against
# the existing `<:PeriodicBC{:exclusive}` dispatch on plain AbstractVector.
# Wrapper takes precedence — searcher's `bc.period` is ignored (grid carries
# its own period). Migrated method-family commits will use NoBC searcher
# when wrapping; this override is purely for transitional safety.
@inline function search_interval(
        s::Searcher{P, H, <:PeriodicBC{:exclusive}}, g::_ExclusivePeriodicAxis, xq::Real
    ) where {P, H}
    idx, xL, xR = _search_binary(g, xq)
    n = length(g.inner)
    idx_R = ifelse(idx == n, 1, idx + 1)
    return idx, idx_R, xL, xR
end
