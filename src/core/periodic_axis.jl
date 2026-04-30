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

# Fields
- `inner::X` — user's raw non-uniform grid (length n). Can be `Vector{Tg}`
  or `_CachedVector{Tg, Tinv}`. Range inputs are NOT wrapped — they go
  directly through `_to_float_adding_endpoint` to a length-(n+1) `_CachedRange`.
- `period::Tp` — period span. Type independent of `Tg` to allow e.g. `Float64`
  grids with `Float32` period.

# Example
```julia
x = [0.0, 0.25, 0.5, 0.75]                       # raw Vector grid, length 4
g = _ExclusivePeriodicAxis(x, 1.0)               # presented as length 5
g[1], g[2], g[3], g[4]                           # 0.0, 0.25, 0.5, 0.75 (raw)
length(g) == 5                                   # virtual extension
_getindex(g, 5)                                  # 1.0 (= x[1] + period)
```
"""
struct _ExclusivePeriodicAxis{Tg, X <: AbstractVector{Tg}, Tp} <: AbstractVector{Tg}
    inner::X
    period::Tp

    # Inner constructor: reject AbstractRange inner (those go through
    # `_to_float_adding_endpoint` → `_CachedRange` of length n+1 instead).
    function _ExclusivePeriodicAxis{Tg, X, Tp}(inner::X, period::Tp) where {Tg, X <: AbstractVector{Tg}, Tp}
        X <: AbstractRange && throw(ArgumentError(
            "_ExclusivePeriodicAxis does not wrap AbstractRange grids. " *
                "Convert exclusive Range to inclusive `_CachedRange` of length n+1 via " *
                "`_to_float_adding_endpoint` instead."
        ))
        return new{Tg, X, Tp}(inner, period)
    end
end

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

# `first`/`last` capture the *virtual* extended span — `WrapExtrap(g)` relies
# on these to compute the wrap domain. Without these overrides, `last(g)`
# would resolve to `g[length(g)] = g[n+1]` which raises BoundsError (since
# `Base.getindex` forwards to `inner`).
@inline Base.first(g::_ExclusivePeriodicAxis) = @inbounds g.inner[1]
@inline Base.last(g::_ExclusivePeriodicAxis) = @inbounds g.inner[1] + g.period

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

# `_ExclusivePeriodicAxis` passes through `_store_grid_cached` (defined in
# cached_vector.jl). When a method-family factory wraps a Vector + `:exclusive`
# PeriodicBC grid before delegating to the persistent constructor, the
# wrapper must survive the constructor's `_store_grid_cached` call rather
# than being re-wrapped or mistakenly converted.
@inline _store_grid_cached(x::_ExclusivePeriodicAxis, ::Type{Tg}) where {Tg} = x

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
# `idx_R` is the *virtual* index (= n+1 at seam, = idx+1 elsewhere). With the
# data side also wrapped in `_ExclusivePeriodicData`, eval kernels can do
# plain `y[idx_R]` (no `_resolve_idx`) — the data wrapper auto-cycles
# `[n+1] → [1]` at the wrapped slot.
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
    return idx, idx + 1, xL, xR
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
    return idx, idx + 1, xL, xR
end
