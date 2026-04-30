# ========================================
# _ExclusivePeriodicVector: representation transform for `:exclusive` PeriodicBC
# ========================================
#
# A wrapper that takes the user's raw n-length non-uniform Vector grid plus
# period and **presents** it as if it were already in inclusive (n+1)
# canonical form:
#
#     User input form (Vector + :exclusive)  →  Internal representation
#     ────────────────────────────────────────────────────────────────
#     PeriodicBC{:exclusive, L} on a Vector  →  _ExclusivePeriodicVector(inner, L)
#                                                length n+1 (virtual)
#
# Range inputs do NOT need this wrapper. A `:exclusive` `AbstractRange` is
# converted directly into a length-(n+1) inclusive `_CachedRange` via
# `_to_float_adding_endpoint` (cached_range.jl) — uniform spacing makes the
# extension exact. Only non-uniform `Vector` inputs benefit from a zero-copy
# wrapper, since materializing the (n+1)-th point would require a fresh
# allocation.
#
# Inclusive PeriodicBC and NoBC paths are unchanged — they keep raw grid as
# before. The wrapper is invoked ONLY when user passes `:exclusive` AND the
# grid is a non-Range AbstractVector.
#
# Why this matters for spacing-cleanup: `_CachedVector.inv_h` has length n-1
# (only normal cells). Without this wrapper, the seam cell (idx=n, idx_R=1
# wrap) would force every eval kernel to special-case OOB lookups. With this
# wrapper, search returns `idx_R = n+1` (virtual), `_getindex(g, n+1)` yields
# `g.inner[1] + period`, normal-cell `inv_h[idx]` lookups stay safe, and the
# seam cell's `_get_h` / `_get_inv_h` are computed on-the-fly from `period`
# at the wrapper level (single branch, predicted not-seam).
#
# Include order: search.jl → periodic.jl → periodic_vector.jl → ...
#
# See `claudedocs/TODO/periodic_grid_wrapper_design.md` for the full design
# rationale, PoC perf data (Vector hot loop = 0.99×), and migration plan.

"""
    _ExclusivePeriodicVector{Tg, X<:AbstractVector{Tg}, Tp} <: AbstractVector{Tg}

Representation wrapper for `PeriodicBC{:exclusive, L}` non-uniform Vector grids.

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
g = _ExclusivePeriodicVector(x, 1.0)             # presented as length 5
g[1], g[2], g[3], g[4]                           # 0.0, 0.25, 0.5, 0.75 (raw)
length(g) == 5                                   # virtual extension
_getindex(g, 5)                                  # 1.0 (= x[1] + period)
```
"""
struct _ExclusivePeriodicVector{Tg, X <: AbstractVector{Tg}, Tp} <: AbstractVector{Tg}
    inner::X
    period::Tp

    # Inner constructor: reject AbstractRange inner (those go through
    # `_to_float_adding_endpoint` → `_CachedRange` of length n+1 instead).
    function _ExclusivePeriodicVector{Tg, X, Tp}(inner::X, period::Tp) where {Tg, X <: AbstractVector{Tg}, Tp}
        X <: AbstractRange && throw(ArgumentError(
            "_ExclusivePeriodicVector does not wrap AbstractRange grids. " *
                "Convert exclusive Range to inclusive `_CachedRange` of length n+1 via " *
                "`_to_float_adding_endpoint` instead."
        ))
        return new{Tg, X, Tp}(inner, period)
    end
end

# Convenience outer constructor — type params inferred from inputs.
@inline _ExclusivePeriodicVector(inner::AbstractVector{Tg}, period) where {Tg} =
    _ExclusivePeriodicVector{Tg, typeof(inner), typeof(period)}(inner, period)

# ---------- AbstractVector interface ----------
# `length` reports virtual extended (n+1) so search algorithms find the seam
# cell at boundary. `getindex` forwards to inner WITHOUT branch — accessing
# `g[n+1]` raises BoundsError from inner. Use `_getindex` helper for virtual.
Base.length(g::_ExclusivePeriodicVector) = length(g.inner) + 1
Base.size(g::_ExclusivePeriodicVector) = (length(g),)
@inline Base.@propagate_inbounds Base.getindex(g::_ExclusivePeriodicVector, i::Int) =
    g.inner[i]
@inline Base.firstindex(::_ExclusivePeriodicVector) = 1
@inline Base.lastindex(g::_ExclusivePeriodicVector) = length(g)
Base.eltype(::Type{<:_ExclusivePeriodicVector{Tg}}) where {Tg} = Tg
Base.IndexStyle(::Type{<:_ExclusivePeriodicVector}) = IndexLinear()

# `first`/`last` capture the *virtual* extended span — `WrapExtrap(g)` relies
# on these to compute the wrap domain. Without these overrides, `last(g)`
# would resolve to `g[length(g)] = g[n+1]` which raises BoundsError (since
# `Base.getindex` forwards to `inner`).
@inline Base.first(g::_ExclusivePeriodicVector) = @inbounds g.inner[1]
@inline Base.last(g::_ExclusivePeriodicVector) = @inbounds g.inner[1] + g.period

# ========================================
# Helpers: virtual access + index resolution + ND fold-back
# ========================================
#
# These helpers no-op on plain AbstractVector arguments (zero overhead for
# default path). They branch only when the argument is `_ExclusivePeriodicVector`.

"""
    _getindex(arr, i::Int)

Like `Base.getindex` but branch-aware for `_ExclusivePeriodicVector`:
- Plain `AbstractVector` / Range / `_CachedRange` / `_CachedVector` → identical to
  `@inbounds arr[i]` (zero overhead).
- `_ExclusivePeriodicVector` with `i ≤ length(g.inner)` → forward to inner.
- `_ExclusivePeriodicVector` with `i == length(g.inner) + 1` → return the virtual
  point `g.inner[1] + g.period` (seam right endpoint).

Used by eval kernels at edge points (right endpoint of the seam cell).
"""
@inline Base.@propagate_inbounds _getindex(arr::AbstractVector, i::Int) = @inbounds arr[i]
@inline Base.@propagate_inbounds function _getindex(g::_ExclusivePeriodicVector, i::Int)
    n = length(g.inner)
    @inbounds return i <= n ? g.inner[i] : g.inner[1] + g.period
end

"""
    _resolve_idx(i::Int, arr) -> Int

Resolve a (possibly virtual) index `i` into a physical index suitable for
data access:
- Plain `AbstractVector` / Range / `_CachedRange` / `_CachedVector` → `i` unchanged.
- `_ExclusivePeriodicVector` with `i ≤ length(g.inner)` → `i` unchanged.
- `_ExclusivePeriodicVector` with `i == length(g.inner) + 1` → `1` (cyclic wrap).

Used by eval kernels for `data[idx_R]` access where `idx_R` may be the
virtual `n+1` returned by search at the seam cell.
"""
@inline _resolve_idx(i::Int, ::AbstractVector) = i
@inline function _resolve_idx(i::Int, g::_ExclusivePeriodicVector)
    n = length(g.inner)
    return i <= n ? i : 1
end

"""
    _periodic_fold_axis!(arr, dim::Int, n_period::Int)

Fold the seam adjoint contribution into the cyclic-1 cell along axis `dim`:

    arr[..., 1, ...] += arr[..., n_period + 1, ...]

Used by adjoint code to merge the virtual seam-right gradient back into the
physical first cell. Generalizes to N-D via `selectdim`.

For non-periodic axes, callers branch on `grids[dim] isa _ExclusivePeriodicVector`
before calling this helper.
"""
@inline function _periodic_fold_axis!(arr, dim::Int, n_period::Int)
    selectdim(arr, dim, 1) .+= selectdim(arr, dim, n_period + 1)
    return arr
end

"""
    _grid_length(x) -> Int

Number of *physical* knot points on the axis grid `x`. Identity to
`length(x)` for plain vectors and ranges (the user-supplied grid IS the
data grid). For `_ExclusivePeriodicVector` returns `length(g.inner)` (n,
the user's original grid size) rather than the virtual n+1 that `length(g)`
reports for the extended search domain.

Used by interpolant constructors to validate that the y values supplied
correspond one-to-one with the user's original grid points (not with the
virtually extended search domain).
"""
@inline _grid_length(x::AbstractVector) = length(x)
@inline _grid_length(g::_ExclusivePeriodicVector) = length(g.inner)

# `_ExclusivePeriodicVector` passes through `_store_grid_cached` (defined
# in cached_vector.jl). When a method-family factory wraps a Vector +
# `:exclusive` PeriodicBC grid before delegating to the persistent
# constructor, the wrapper must survive the constructor's `_store_grid_cached`
# call rather than being re-wrapped or mistakenly converted.
@inline _store_grid_cached(x::_ExclusivePeriodicVector, ::Type{Tg}) where {Tg} = x

# ========================================
# Spacing accessors: idx-aware with seam fast-path
# ========================================
#
# `_get_h(g, idx)` / `_get_inv_h(g, idx)` 2-arg form work uniformly across
# all grid types after Step 1.5. For `_ExclusivePeriodicVector`, the wrapper
# branches on idx:
#   - idx < n  → delegate to `inner` (cached lookup for `_CachedVector`,
#                on-the-fly for raw `Vector`)
#   - idx == n → seam cell: compute on-the-fly from `period`
#                (`h = inner[1] + period - inner[n]`, `inv_h = inv(h)`)
#
# Branch is highly predictable (seam fires at most once per query batch),
# and the wrapper-level branch keeps inner-level dispatches simple.
@inline Base.@propagate_inbounds function _get_h(g::_ExclusivePeriodicVector, idx::Int)
    n = length(g.inner)
    @inbounds return idx < n ? _get_h(g.inner, idx) : g.inner[1] + g.period - g.inner[n]
end
@inline Base.@propagate_inbounds function _get_inv_h(g::_ExclusivePeriodicVector, idx::Int)
    n = length(g.inner)
    @inbounds return idx < n ? _get_inv_h(g.inner, idx) : inv(g.inner[1] + g.period - g.inner[n])
end

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
    _search_binary(g::_ExclusivePeriodicVector, xq) -> (idx, xL, xR)

Binary search on `_ExclusivePeriodicVector`. ONE upfront seam check; if the
query is past `g.inner[n]`, return the seam tuple `(n, x[n], x[1]+period)`.
Otherwise delegate to standard binary search on the raw inner Vector — zero
wrapper overhead for the hot loop.
"""
@inline function _search_binary(g::_ExclusivePeriodicVector{T}, xq::Real) where {T}
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
# unpack `(idx_L, idx_R, xL, xR)` uniformly. For `_ExclusivePeriodicVector`,
# `idx_R` is the *virtual* index (= n+1 at seam, = idx+1 elsewhere). Eval
# kernels use `_resolve_idx(idx_R, grid)` to turn it into a physical index
# for data access (`y[_resolve_idx(idx_R, grid)]`).
#
# Two BC-constrained methods exist to avoid ambiguity with the existing
# generic dispatch (`Searcher{...,<:AbstractBC}, AbstractVector, Real`) and
# the seam-specialized dispatch (`Searcher{...,<:PeriodicBC{:exclusive}},
# AbstractVector, Real`). When the grid is `_ExclusivePeriodicVector`, the
# wrapper handles the seam — the searcher's BC seam logic is bypassed.

# GridIdx queries — bypass the seam fast-path entirely (user-supplied explicit
# index semantics, no wrap). Mirrors `search_interval(s, x::AbstractVector, ::GridIdx)`
# in search.jl. Required to avoid Aqua method-ambiguity warnings.
@inline function search_interval(s::Searcher, g::_ExclusivePeriodicVector, xq::GridIdx)
    idx, xL, xR = _search_grididx_dispatch(s.hint, g, xq)
    return idx, idx + 1, xL, xR
end

# Generic BC + wrapper: always use grid-type seam handling.
@inline function search_interval(
        s::Searcher{P, H, <:AbstractBC}, g::_ExclusivePeriodicVector, xq::Real
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
        s::Searcher{P, H, <:PeriodicBC{:exclusive}}, g::_ExclusivePeriodicVector, xq::Real
    ) where {P, H}
    idx, xL, xR = _search_binary(g, xq)
    return idx, idx + 1, xL, xR
end
