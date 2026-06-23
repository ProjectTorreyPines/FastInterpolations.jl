# ========================================
# _CachedVector methods — Vector grid with cached spacing
# ========================================
#
# Struct definition + `_CachedVector(::_CachedVector)` identity ctor live in
# `axis_types.jl` (loaded earlier). This file owns:
#   - AbstractArray interface (`size`, `length`, `getindex`, `firstindex`,
#     `lastindex`, `eltype`, `IndexStyle`)
#   - `_CachedVector(x::AbstractVector{T})` build ctor (computes h/inv_h)
#   - `Base.copy` / `_convert_copy`
#   - `_get_h` / `_get_inv_h`
#   - `_resolve_axis` / `_cache_axis` / `_cache_axis_pooled` (all BC variants
#     for raw `AbstractVector` and pre-wrapped `_CachedVector` input)
#
# Include order: ... → axis_types.jl → cached_range.jl → cached_vector.jl → ...

# ---------- AbstractVector interface ----------
Base.size(c::_CachedVector) = (length(c.inner),)
Base.length(c::_CachedVector) = length(c.inner)
@inline Base.@propagate_inbounds Base.getindex(c::_CachedVector, i::Int) = c.inner[i]
@inline Base.firstindex(::_CachedVector) = 1
@inline Base.lastindex(c::_CachedVector) = length(c.inner)
Base.eltype(::Type{<:_CachedVector{T}}) where {T} = T
Base.IndexStyle(::Type{<:_CachedVector}) = IndexLinear()

# ---------- Wrapper-preserving copy ----------
# Default `copy(::AbstractVector)` materializes to plain `Vector{T}`, destroying
# the wrapper. Persistent ND ctors do `map(copy, grids)` for mutation safety —
# this overload preserves wrapper + h/inv_h caches via independent buffers
# (every field is a fresh allocation owned by the result; aliasing would
# violate `copy`'s ownership contract).
@inline Base.copy(c::_CachedVector) =
    _CachedVector(copy(c.inner), copy(c.h), copy(c.inv_h))

# ---------- Wrapper-aware `_convert_copy` ----------
# Same-type → `Base.copy`. Different-type → rebuild from type-converted inner
# (one pass; ctor recomputes h/inv_h for the new eltype).
@inline _convert_copy(c::_CachedVector{T}, ::Type{T}) where {T} = copy(c)
@inline _convert_copy(c::_CachedVector, ::Type{T}) where {T} =
    _CachedVector(_convert_copy(c.inner, T))

# ---------- Construction from AbstractVector ----------
"""
    _CachedVector(x::AbstractVector{T})

Construct a `_CachedVector` from a raw vector. Computes `h[i] = x[i+1] - x[i]`
and `inv_h[i] = inv(h[i])` once, in a single pass.

If `x` is not already a concrete `Vector{T}`, it is converted via `Vector(x)`
to ensure stable storage layout.

Throws `ArgumentError` if `length(x) < 2`.
"""
function _CachedVector(x::AbstractVector{T}) where {T}
    n = length(x)
    n >= 2 || throw(ArgumentError("_CachedVector requires at least 2 grid points, got $n"))
    Tinv = typeof(inv(oneunit(T)))
    h = Vector{T}(undef, n - 1)
    inv_h = Vector{Tinv}(undef, n - 1)
    @inbounds for i in 1:(n - 1)
        h[i] = x[i + 1] - x[i]
        inv_h[i] = inv(h[i])
    end
    inner = x isa Vector{T} ? x : Vector{T}(x)
    return _CachedVector(inner, h, inv_h)
end

# ---------- Pool-backed construction (transient caches) ----------
# Pool-acquired `h` / `inv_h` (reused across calls). For non-`Vector{T}`
# input (e.g. `view(big, k:k+N)`), the inner buffer is *also* pool-acquired
# + `copyto!` rather than `Vector{T}(x)` — keeps zero-alloc on view input
# while preserving the `inner::Vector{T}` invariant.
@inline function _cache_axis_pooled(pool, x::AbstractVector{T}) where {T}
    n = length(x)
    n >= 2 || throw(ArgumentError("_cache_axis_pooled requires at least 2 grid points, got $n"))
    Tinv = typeof(inv(oneunit(T)))
    inner = if x isa Vector{T}
        x
    else
        buf = acquire!(pool, T, n)
        copyto!(buf, x)
        buf
    end
    h = acquire!(pool, T, n - 1)
    inv_h = acquire!(pool, Tinv, n - 1)
    @inbounds for i in 1:(n - 1)
        h[i] = inner[i + 1] - inner[i]
        inv_h[i] = inv(h[i])
    end
    return _CachedVector(inner, h, inv_h)
end

# Pre-wrapped `_CachedVector` passthrough — cache already built.
@inline _cache_axis_pooled(_, x::_CachedVector) = x
# (Range / `_CachedRange` variants live in `cached_range.jl`.)

# ─────────────────────────────────────────────────────────────────────────────
# 3-arg form: `_cache_axis_pooled(pool, x, Tg)` — target eltype explicit
# ─────────────────────────────────────────────────────────────────────────────
# Catch-all delegation: `_to_float(x, Tg)` first (eltype-aware normalization),
# then dispatch to the matching 2-arg overload by output type. This is the
# *one-shot* twin of the persistent `_cache_axis(x, bc, Tg)` 3-arg form — but
# UNLIKE the persistent variant (which *intentionally* ignores Tg on pre-wrapped
# axes — see the DISPATCH TABLE in `periodic_axis.jl`), the pooled 3-arg form
# honors `Tg` uniformly via `_to_float` for every input type.
#
#   x type         + Tg            → result   (notes)
#   ─────────────────────────────────────────────────────────────────────────
#   Vector{Tg}      + Tg            → pool-backed `_CachedVector{Tg}`
#                                       (`_to_float` identity → 2-arg pool wrap)
#   Vector{S}       + Tg, S ≠ Tg    → pool-backed `_CachedVector{Tg}` ⚠️ warns once
#                                       (`_to_float` heap-allocates `Tg.(x)` then
#                                        2-arg builds the wrapper)
#   AbstractRange   + Tg            → `_CachedRange{Tg}`  (zero pool, immutable
#                                       struct; eltype always Tg)
#   _CachedRange{Tg}  + Tg            → identity passthrough
#   _CachedRange{S}   + Tg, S ≠ Tg    → fresh `_CachedRange{Tg}` (heap-allocs
#                                       inv_h scalar; NOT pool-backed)
#   _CachedVector{Tg} + Tg            → identity passthrough
#   _CachedVector{S}  + Tg, S ≠ Tg    → ⚠️ pre-built h/inv_h cache is DROPPED:
#                                       `_to_float(::AbstractVector, Tg)`
#                                       fallback broadcasts `Tg.(c)` into a
#                                       fresh Vector (losing wrapper fields),
#                                       then 2-arg rebuilds `_CachedVector{Tg}`
#                                       with pool-backed h/inv_h
#
# Bottom line: `_cache_axis_pooled(pool, x, Tg)` *always* returns a wrapper
# with eltype `Tg`. The fast paths (same-eltype Vector/Range/wrapped) are
# zero-alloc; eltype mismatches incur a one-time conversion alloc (and warn
# on raw `Vector{S}` per `_to_float`'s `_warn_type_conversion`).
@inline _cache_axis_pooled(pool, x, ::Type{Tg}) where {Tg} =
    _cache_axis_pooled(pool, _to_float(x, Tg))

# ========================================
# _get_h / _get_inv_h accessors (Vector hierarchy)
# ========================================
# Vector forms only; the Range forms (`_CachedRange`, `AbstractRange`) live in cached_range.jl.

# _CachedVector — cached vector lookup (most specific for AbstractVector hierarchy)
@inline Base.@propagate_inbounds _get_h(x::_CachedVector, i::Int) = @inbounds x.h[i]
@inline Base.@propagate_inbounds _get_inv_h(x::_CachedVector, i::Int) = @inbounds x.inv_h[i]

# AbstractVector — on-the-fly diff (one-shot, no cache). Raw eltype preserved
# (`Int→Int`, `Rational→Rational`); downstream kernels auto-promote, so no eager-Float.
@inline Base.@propagate_inbounds _get_h(x::AbstractVector, i::Int) =
    @inbounds x[i + 1] - x[i]
@inline Base.@propagate_inbounds _get_inv_h(x::AbstractVector, i::Int) =
    inv(_get_h(x, i))

# 4-arg form: `(x, idx, xL, xR)` — every search result delivers all four;
# dispatch picks the cheapest path per axis type.
#
# `_CachedVector`: idx-indexed cache wins (1 load, no fp ops). The xL/xR
# fields are ignored — they're already encoded in `c.inv_h[idx]`.
@inline Base.@propagate_inbounds _get_h(x::_CachedVector, idx::Int, ::Real, ::Real) =
    @inbounds x.h[idx]
@inline Base.@propagate_inbounds _get_inv_h(x::_CachedVector, idx::Int, ::Real, ::Real) =
    @inbounds x.inv_h[idx]

# Raw `AbstractVector` fallback: no cache → `xR - xL` straight from search.
# `idx` is ignored here (kept in signature for uniform call shape).
@inline _get_h(::AbstractVector, ::Int, xL::Real, xR::Real) = xR - xL
@inline _get_inv_h(::AbstractVector, ::Int, xL::Real, xR::Real) = inv(xR - xL)

# Persistent axis wrapping is split into two stages (see `periodic_axis.jl`):
#   - outer surface API: `_cache_axis(x, bc)` — bc-aware wrap, zero-copy
#     of buffer (Vector → `_CachedVector`, Range → `_CachedRange`,
#     `:exclusive` → `_ExclusivePeriodicAxis`),
#   - inner constructor:  `_convert_copy(x, Tg)` — ownership copy +
#     element-type promotion (wrapper-preserving `Base.copy`).
# Cubic 1D builds owned cache axes via `_cache_axis(_convert_copy(x, T), bc)`
# (copy-then-wrap): one buffer copy + eltype conversion, then alias the
# fresh buffer and build h/inv_h once.

# ========================================
# `_resolve_axis` — one-shot Vector wrapping
# ========================================
# Raw `Vector` is the canonical one-shot form (passthrough); `_CachedVector`
# round-trips unchanged.
@inline _resolve_axis(x::AbstractVector) = x
@inline _resolve_axis(x::AbstractVector, ::AbstractBC) = x
@inline _resolve_axis(c::_CachedVector) = c

# `:exclusive` raw-input one-shot path — wrap into `_ExclusivePeriodicAxis`.
@inline function _resolve_axis(x::AbstractVector, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(x, bc_resolved.period)
end

# ========================================
# `_cache_axis` — persistent-path Vector wrapping
# ========================================
# Full DISPATCH TABLE for `_cache_axis(x, bc, Tg)` (cross-input-type) lives in
# `periodic_axis.jl` as the central reference. Owner-file entries below.
@inline _cache_axis(x::AbstractVector) = _CachedVector(x)
@inline _cache_axis(x::AbstractVector, ::AbstractBC) = _CachedVector(x)
@inline _cache_axis(c::_CachedVector) = c
@inline _cache_axis(c::_CachedVector, ::AbstractBC) = c

# `:exclusive` 2-arg variants — produce `_ExclusivePeriodicAxis(_CachedVector, ·)`.
@inline function _cache_axis(x::AbstractVector, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(x, bc)
    return _ExclusivePeriodicAxis(_CachedVector(x), bc_resolved.period)
end
@inline function _cache_axis(c::_CachedVector, bc::PeriodicBC{:exclusive})
    bc_resolved = _resolve_bc_period(c, bc)
    return _ExclusivePeriodicAxis(c, bc_resolved.period)
end

# 3-arg Tg-aware. Raw Vector respects Tg via `_to_float` then wraps.
# Pre-wrapped `_CachedVector` passes through — downstream `_convert_copy(_, Tg)`
# enforces Tg (intentional contract; see DISPATCH TABLE in `periodic_axis.jl`).
@inline _cache_axis(x::AbstractVector, bc::AbstractBC, ::Type{Tg}) where {Tg} =
    _cache_axis(_to_float(x, Tg), bc)
@inline _cache_axis(c::_CachedVector, bc::AbstractBC, ::Type{Tg}) where {Tg} =
    _cache_axis(c, bc)
@inline _cache_axis(x::AbstractVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} =
    _cache_axis(_to_float(x, Tg), bc)
@inline _cache_axis(c::_CachedVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg} =
    _cache_axis(c, bc)
