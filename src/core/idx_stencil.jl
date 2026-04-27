# ========================================
# _IdxStencil{K} — wrap-aware per-axis index stencil
# ========================================
# Unified corner-addressing type used by all multi-point interpolation methods.
# Wraps `NTuple{K, Int}` with a distinct type so:
#
#   * Non-periodic interior cells store `(idx, idx+1, …, idx+K-1)` (no wrap).
#   * Periodic-exclusive seam cells store wrapped indices (e.g. `(n, 1)` for
#     K=2, or `(n-1, n, 1, 2)` for K=4) without any grid/value copy.
#
# Design rationale:
#   * K is a type parameter so one kernel function serves Linear/Constant K=2,
#     (future) ND Hermite K=4, etc. via parametric dispatch.
#   * Distinct struct identity avoids the N=0 dispatch ambiguity that
#     `NTuple{N, NTuple{K, Int}}` would create against `NTuple{N, Int}`
#     (both collapse to `Tuple{}` at N=0). The remedy is not the type wrap
#     alone — it is having **one** unified function shape, so overload
#     ambiguity never arises in the first place.
#   * Memory layout is identical to a raw `NTuple{K, Int}` (inline, no
#     boxing). All getters are `@inline` — compiler produces the same ASM as
#     raw tuple indexing after inlining.
#
# Not exported — internal API.

"""
    _IdxStencil{K}

Per-axis index stencil carrying `K` grid indices. Corners the query reads on
one axis; the tuple may contain wrapped indices for periodic-exclusive seam
cells.

Fields:
- `indices::NTuple{K, Int}`: the K grid indices

Construction:
```julia
s = _IdxStencil((5, 6))            # K inferred (=2)
s = _IdxStencil{2}((5, 6))         # K specified
s = _IdxStencil((5, 6, 7, 8))      # K=4 stencil
s = _IdxPair(5, 6)                 # K=2 convenience (2-arg constructor)
const _IdxPair = _IdxStencil{2}    # alias
```

Access:
```julia
s[1]         # first corner (= idxL for K=2)
s[2]         # second corner (= idxR for K=2)
length(s)    # K
```
"""
struct _IdxStencil{K}
    indices::NTuple{K, Int}
end

# ────────────────────────────────────────
# Constructors
# ────────────────────────────────────────
#
# The struct's auto-generated inner constructor
#   `_IdxStencil{K}(t::NTuple{K, Int}) where K`
# handles tuple-style construction for both `_IdxStencil((5, 6))` and
# `_IdxStencil{2}((5, 6))`. We do NOT add an outer for the tuple form —
# Julia normalizes any such outer to the same method signature as the
# inner, triggering a precompile-time method-overwriting error.
#
# Below we add ONE positional-args outer (Vararg{Int, K}) so callers can
# write `_IdxStencil{K}(i1, i2, …, iK)` without the explicit tuple
# parentheses. This is purely cosmetic — body just forwards to the inner
# with the captured Vararg tuple.

@inline _IdxStencil{K}(idxs::Vararg{Int, K}) where {K} = _IdxStencil{K}(idxs)

# K=2 convenience alias (most common case: Linear/Constant 1D+ND, Cubic 1D).
# Combined with the Vararg outer above, `_IdxPair(idxL, idxR)` works directly:
# Julia resolves `_IdxPair === _IdxStencil{2}` then matches the Vararg method
# with K=2. No separate `_IdxPair(...)` method needed.
const _IdxPair = _IdxStencil{2}

# ────────────────────────────────────────
# Accessors — explicit, no AbstractVector inheritance
# ────────────────────────────────────────
# We deliberately do NOT subtype AbstractVector:
# - Inheriting iterate/show/broadcast fallbacks can force type instability or
#   heap-allocating defaults in edge cases.
# - The only operations needed inside hot kernels are `getindex(s, k)` and
#   `length(s)` — both provided inline below.

@inline Base.getindex(s::_IdxStencil, k::Int) = @inbounds s.indices[k]
@inline Base.length(::_IdxStencil{K}) where {K} = K
@inline Base.firstindex(::_IdxStencil) = 1
@inline Base.lastindex(::_IdxStencil{K}) where {K} = K
@inline Base.first(s::_IdxStencil) = @inbounds s.indices[1]
@inline Base.last(s::_IdxStencil{K}) where {K} = @inbounds s.indices[K]

# `iterate` provided for `for i in stencil` ergonomics; still allocation-free
# because it's a compile-time-known fixed-K loop.
@inline function Base.iterate(s::_IdxStencil{K}, state::Int = 1) where {K}
    state > K && return nothing
    return (@inbounds s.indices[state], state + 1)
end

# ────────────────────────────────────────
# Equality (for test assertions and debugging)
# ────────────────────────────────────────

@inline Base.:(==)(a::_IdxStencil{K}, b::_IdxStencil{K}) where {K} = a.indices == b.indices
@inline Base.hash(s::_IdxStencil, h::UInt) = hash(s.indices, hash(:_IdxStencil, h))
