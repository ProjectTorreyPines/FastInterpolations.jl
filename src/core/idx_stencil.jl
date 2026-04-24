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
s = _pair(5, 6)                    # K=2 convenience
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
# serves BOTH call shapes once K is reachable from the tuple length:
#   _IdxStencil{2}((5, 6))   -- K specified, inner matches directly
#   _IdxStencil((5, 6))      -- Julia's `where K` inference on the inner
#                               resolves K=2 from the NTuple{2, Int} argument
# No outer constructor needed — adding one here would overwrite the inner
# (same normalized method signature at the Julia type system level) and
# trigger a precompile-time method-overwriting error.

# K=2 convenience alias (most common case: Linear/Constant 1D+ND, Cubic 1D)
const _IdxPair = _IdxStencil{2}

# Construct a pair stencil from left + right indices (Phase 6 style call sites).
@inline _pair(idxL::Int, idxR::Int) = _IdxStencil((idxL, idxR))

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
