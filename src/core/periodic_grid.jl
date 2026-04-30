# ========================================
# _ExclusivePeriodicGrid: representation transform for `:exclusive` PeriodicBC
# ========================================
#
# A wrapper that takes the user's raw n-length grid + period and **presents**
# it as if it were already in inclusive (n+1) canonical form:
#
#     User input form              →  Internal representation
#     ────────────────────────────────────────────────────────
#     PeriodicBC{:exclusive, L}    →  _ExclusivePeriodicGrid(raw_n, L)
#                                       length n+1 (virtual)
#
# Inclusive PeriodicBC and NoBC paths are unchanged — they keep raw grid as
# before. The wrapper is invoked ONLY when user passes `:exclusive`.
#
# After wrapping (or non-wrapping), eval kernels see a uniform "AbstractVector
# of length n+1" abstraction. Inclusive: physically n+1. Exclusive: virtually
# n+1 via wrapper.
#
# Why this matters for spacing-cleanup: `_CachedVector.inv_h` has length n-1
# (only normal cells). Without this wrapper, the seam cell (idx=n, idx_R=1
# wrap) would force every eval kernel to special-case OOB lookups. With this
# wrapper, search returns idx_R=n+1 (a virtual index), `_getindex(g, n+1)`
# yields `g.inner[1] + period`, and normal-cell `inv_h[idx]` lookups stay safe.
#
# Include order: search.jl → periodic.jl → periodic_grid.jl → ...
#
# See `claudedocs/TODO/periodic_grid_wrapper_design.md` for the full design
# rationale, PoC perf data (Vector hot loop = 0.99×, Range = noise-level),
# and migration plan.

"""
    _ExclusivePeriodicGrid{Tg, X<:AbstractVector{Tg}, Tp} <: AbstractVector{Tg}

Representation wrapper for `PeriodicBC{:exclusive, L}` grids.

`inner` is the user's raw grid (length n, NO copy). `period` is the period
span (one period of the user's data). The wrapper reports `length(g) = n+1`
so search algorithms naturally find the seam cell, but `Base.getindex(g, i)`
forwards to `inner[i]` WITHOUT any branch — keeping hot search loops at zero
overhead vs raw `inner`.

Virtual-index access (i.e. `g[n+1]`) is provided through the `_getindex`
helper (defined below), which branches on the virtual range.

# Fields
- `inner::X` — user's raw grid (length n). Can be `Vector{Tg}`, `_CachedRange{Tg}`,
  or `_CachedVector{Tg, Tinv}`. Search dispatch specializes on this inner type.
- `period::Tp` — period span. Type independent of `Tg` to allow e.g. `Float64`
  grids with `Float32` period.

# Example
```julia
x = [0.0, 0.25, 0.5, 0.75]                       # raw grid, length 4
g = _ExclusivePeriodicGrid(x, 1.0)               # presented as length 5
g[1], g[2], g[3], g[4]                           # 0.0, 0.25, 0.5, 0.75 (raw)
length(g) == 5                                   # virtual extension
_getindex(g, 5)                                  # 1.0 (= x[1] + period)
```
"""
struct _ExclusivePeriodicGrid{Tg, X <: AbstractVector{Tg}, Tp} <: AbstractVector{Tg}
    inner::X
    period::Tp
end

# ---------- AbstractVector interface ----------
# `length` reports virtual extended (n+1) so search algorithms find the seam
# cell at boundary. `getindex` forwards to inner WITHOUT branch — accessing
# `g[n+1]` raises BoundsError from inner. Use `_getindex` helper for virtual.
Base.length(g::_ExclusivePeriodicGrid) = length(g.inner) + 1
Base.size(g::_ExclusivePeriodicGrid) = (length(g),)
@inline Base.@propagate_inbounds Base.getindex(g::_ExclusivePeriodicGrid, i::Int) =
    g.inner[i]
@inline Base.firstindex(::_ExclusivePeriodicGrid) = 1
@inline Base.lastindex(g::_ExclusivePeriodicGrid) = length(g)
Base.eltype(::Type{<:_ExclusivePeriodicGrid{Tg}}) where {Tg} = Tg
Base.IndexStyle(::Type{<:_ExclusivePeriodicGrid}) = IndexLinear()

# Convenience constructor — type params inferred from inputs.
@inline _ExclusivePeriodicGrid(inner::AbstractVector{Tg}, period::Real) where {Tg} =
    _ExclusivePeriodicGrid{Tg, typeof(inner), typeof(period)}(inner, period)

# ========================================
# Helpers: virtual access + index resolution + ND fold-back
# ========================================
#
# These helpers no-op on plain AbstractVector arguments (zero overhead for
# default path). They branch only when the argument is `_ExclusivePeriodicGrid`.

"""
    _getindex(arr, i::Int)

Like `Base.getindex` but branch-aware for `_ExclusivePeriodicGrid`:
- Plain `AbstractVector` / Range / `_CachedRange` / `_CachedVector` → identical to
  `@inbounds arr[i]` (zero overhead).
- `_ExclusivePeriodicGrid` with `i ≤ length(g.inner)` → forward to inner.
- `_ExclusivePeriodicGrid` with `i == length(g.inner) + 1` → return the virtual
  point `g.inner[1] + g.period` (seam right endpoint).

Used by eval kernels at edge points (right endpoint of the seam cell).
"""
@inline Base.@propagate_inbounds _getindex(arr::AbstractVector, i::Int) = @inbounds arr[i]
@inline Base.@propagate_inbounds function _getindex(g::_ExclusivePeriodicGrid, i::Int)
    n = length(g.inner)
    @inbounds return i <= n ? g.inner[i] : g.inner[1] + g.period
end

"""
    _resolve_idx(i::Int, arr) -> Int

Resolve a (possibly virtual) index `i` into a physical index suitable for
data access:
- Plain `AbstractVector` / Range / `_CachedRange` / `_CachedVector` → `i` unchanged.
- `_ExclusivePeriodicGrid` with `i ≤ length(g.inner)` → `i` unchanged.
- `_ExclusivePeriodicGrid` with `i == length(g.inner) + 1` → `1` (cyclic wrap).

Used by eval kernels for `data[idx_R]` access where `idx_R` may be the
virtual `n+1` returned by search at the seam cell.
"""
@inline _resolve_idx(i::Int, ::AbstractVector) = i
@inline function _resolve_idx(i::Int, g::_ExclusivePeriodicGrid)
    n = length(g.inner)
    return i <= n ? i : 1
end

"""
    _periodic_fold_axis!(arr, dim::Int, n_period::Int)

Fold the seam adjoint contribution into the cyclic-1 cell along axis `dim`:

    arr[..., 1, ...] += arr[..., n_period + 1, ...]

Used by adjoint code to merge the virtual seam-right gradient back into the
physical first cell. Generalizes to N-D via `selectdim`.

For non-periodic axes, callers branch on `grids[dim] isa _ExclusivePeriodicGrid`
before calling this helper.
"""
@inline function _periodic_fold_axis!(arr, dim::Int, n_period::Int)
    selectdim(arr, dim, 1) .+= selectdim(arr, dim, n_period + 1)
    return arr
end

# ========================================
# Specialized search: zero-overhead seam fast-path
# ========================================
#
# The reason "wrapper-based unification" works WITHOUT perf cost: inside the
# search loop, accesses go through `Base.getindex` on `g.inner` (raw Vector
# or Range). NO branch inserted by the wrapper. The seam check is hoisted
# ONCE before search — same cost as a single `<:PeriodicBC{:exclusive>`
# dispatcher invocation in current master.

"""
    _search_binary(g::_ExclusivePeriodicGrid, xq) -> idx

Binary search on `_ExclusivePeriodicGrid`. ONE upfront seam check; if the
query is past `g.inner[n]`, return `n` (seam cell — caller resolves
`idx_R = n+1` virtual right endpoint via `_getindex`/`_resolve_idx`).
Otherwise delegate to standard binary search on the raw inner Vector — zero
wrapper overhead for the hot loop.
"""
@inline function _search_binary(g::_ExclusivePeriodicGrid{T}, xq::Real) where {T}
    n = length(g.inner)
    @inbounds if xq >= g.inner[n]
        return n, g.inner[n], g.inner[1] + g.period
    end
    return _search_binary(g.inner, xq)
end

"""
    _search_direct(g::_ExclusivePeriodicGrid{<:Any, <:AbstractRange}, xq) -> (idx, xL, xR)

O(1) direct-search specialization for Range-backed `_ExclusivePeriodicGrid`.
Same seam fast-path as the binary version, then delegates to the inner
Range's standard `_search_direct` for normal cells.

Note: `_CachedRange <: AbstractRange`, so this method handles both raw Range
and `_CachedRange` inner types via the `<:AbstractRange` constraint.
"""
@inline function _search_direct(
        g::_ExclusivePeriodicGrid{T, <:AbstractRange}, xq::Real
    ) where {T}
    n = length(g.inner)
    @inbounds if xq >= g.inner[n]
        return n, g.inner[n], g.inner[1] + g.period
    end
    return _search_direct(g.inner, xq)
end

# ========================================
# search_interval entry point: 4-tuple return
# ========================================
#
# Matches the existing `search_interval(s, x, xq)` shape so eval kernels can
# unpack `(idx_L, idx_R, xL, xR)` uniformly. For `_ExclusivePeriodicGrid`,
# `idx_R` is the *virtual* index (= n+1 at seam, = idx+1 elsewhere). Eval
# kernels use `_resolve_idx(idx_R, grid)` to turn it into a physical index
# for data access (`y[_resolve_idx(idx_R, grid)]`).
#
# Two BC-constrained methods exist to avoid ambiguity with the existing
# generic dispatch (`Searcher{...,<:AbstractBC}, AbstractVector, Real`) and
# the seam-specialized dispatch (`Searcher{...,<:PeriodicBC{:exclusive}},
# AbstractVector, Real`). When the grid is `_ExclusivePeriodicGrid`, the
# wrapper handles the seam — the searcher's BC seam logic is bypassed.

# Generic BC + wrapper: always use grid-type seam handling.
@inline function search_interval(
        s::Searcher{P, H, <:AbstractBC}, g::_ExclusivePeriodicGrid, xq::Real
    ) where {P, H}
    idx, xL, xR = _periodic_grid_search(s, g, xq)
    return idx, idx + 1, xL, xR
end

# `:exclusive` BC + wrapper: explicit override so this is unambiguous against
# the existing `<:PeriodicBC{:exclusive}` dispatch on plain AbstractVector.
# Wrapper takes precedence — searcher's `bc.period` is ignored (grid carries
# its own period). Migrated method-family commits will use NoBC searcher
# when wrapping; this override is purely for transitional safety.
@inline function search_interval(
        s::Searcher{P, H, <:PeriodicBC{:exclusive}}, g::_ExclusivePeriodicGrid, xq::Real
    ) where {P, H}
    idx, xL, xR = _periodic_grid_search(s, g, xq)
    return idx, idx + 1, xL, xR
end

# Range-backed inner: O(1) direct search after seam check.
@inline _periodic_grid_search(
    ::Searcher, g::_ExclusivePeriodicGrid{T, <:AbstractRange}, xq::Real
) where {T} = _search_direct(g, xq)

# Vector-backed inner: O(log n) binary search after seam check.
@inline _periodic_grid_search(
    ::Searcher, g::_ExclusivePeriodicGrid{T, <:AbstractVector}, xq::Real
) where {T} = _search_binary(g, xq)
