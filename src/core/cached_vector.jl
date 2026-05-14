# ========================================
# _CachedVector: Vector grid with cached spacing
# ========================================
#
# Non-uniform Vector grid wrapper that caches per-cell spacing (`h`) and
# reciprocal (`inv_h`) for fast lookup. Parallel to `_CachedRange` (which
# wraps uniform AbstractRange grids).
#
# All non-Range AbstractVector inputs to persistent interpolant constructors
# are normalized to `_CachedVector` via `_store_grid`. After normalization,
# the grid type space for persistent paths is `{_CachedRange, _CachedVector}`,
# enabling a single 2-arg `_get_h(grid, i)` dispatch surface.
#
# Include order: grid_spacing.jl → cached_range.jl → cached_vector.jl → ...

"""
    _CachedVector{T, Tinv, I} <: AbstractVector{T}

Vector grid wrapper that caches per-cell spacing (`h`) and reciprocal
(`inv_h`) for fast O(1) lookup during persistent interpolant evaluation.

# Fields
- `inner::I` — Original grid values (length n). Storage type varies by
  construction path: persistent paths materialize to `Vector{T}` for
  ownership (mutation safety + stable layout); one-shot pool paths alias
  the caller's input (e.g. `SubArray{T}` view) to avoid an otherwise
  pointless copy.
- `h::Vector{T}` — Cell widths, `h[i] = inner[i+1] - inner[i]` (length n-1).
  Persistent path: fresh `Vector{T}` from heap. One-shot path:
  `acquire!(pool, T, n-1)` returns a `Vector{T}` from the pool's task-local
  storage (reused across calls — zero-alloc after warmup).
- `inv_h::Vector{Tinv}` — Precomputed reciprocals (length n-1). Same storage
  rule as `h`.

# Type parameters
- `T` — Grid element type. Can be `Int`, `Float64`, `ForwardDiff.Dual`,
  `Measurements.Measurement`, etc. Any type with `-` and `inv` defined.
- `Tinv` — Reciprocal type, equal to `typeof(inv(oneunit(T)))`. For Float
  grids `Tinv == T`; for Int grids `Tinv == Float64` (since `inv(::Int)`
  returns `Float64`).
- `I <: AbstractVector{T}` — Concrete inner-storage type. Tracked as a
  type parameter so `getindex(c, i) = c.inner[i]` specializes on the
  concrete storage (no runtime dispatch on `SubArray`/`Vector` mix).
  `h` / `inv_h` are NOT parametrized — they are always `Vector{T}` /
  `Vector{Tinv}` because `acquire!` returns a `Vector`, not a view.

# Behavior
- `<: AbstractVector{T}` so existing search and eval kernels accept it
  transparently — same trick `_CachedRange` uses with `<: AbstractRange`.
- `Base.getindex(c, i)` forwards to `inner[i]` — zero overhead vs raw `Vector`.
- `length(c) == length(c.inner)` — same as raw vector.
- `Base.copy(c)` materializes `inner` to `Vector{T}` (ownership contract)
  regardless of source storage type — so the copy path normalizes one-shot
  view-backed wrappers into owned `Vector`-backed wrappers suitable for
  long-lived persistent storage. `h` / `inv_h` are already `Vector`, so
  `copy` of those is a straight memcpy.

# Example
```julia
x = [0.0, 0.3, 0.7, 1.0]      # non-uniform Float grid
cv = _CachedVector(x)          # _CachedVector{Float64, Float64, Vector{Float64}}

x_int = [0, 1, 3, 6]           # Int grid
cv_int = _CachedVector(x_int)  # _CachedVector{Int, Float64, Vector{Int}}
```
"""
struct _CachedVector{T, Tinv, I <: AbstractVector{T}} <: AbstractVector{T}
    inner::I
    h::Vector{T}        # length n-1; heap (persistent) or pool-acquired (one-shot)
    inv_h::Vector{Tinv} # length n-1; same storage rule as `h`
end

# ---------- AbstractVector interface ----------
Base.size(c::_CachedVector) = (length(c.inner),)
Base.length(c::_CachedVector) = length(c.inner)
@inline Base.@propagate_inbounds Base.getindex(c::_CachedVector, i::Int) = c.inner[i]
@inline Base.firstindex(::_CachedVector) = 1
@inline Base.lastindex(c::_CachedVector) = length(c.inner)
Base.eltype(::Type{<:_CachedVector{T}}) where {T} = T
Base.IndexStyle(::Type{<:_CachedVector}) = IndexLinear()

# ---------- Idempotent passthrough ----------
# `_CachedVector(::_CachedVector)` returns the input unchanged. Mirrors
# `_to_float(x::_CachedRange{T}, ::Type{T}) = x` (cached_range.jl) — re-wrapping
# would discard the existing cached `h`/`inv_h` for no reason.
_CachedVector(c::_CachedVector) = c

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
# Same shape as the outer ctor, but `h`/`inv_h` come from `pool` (zero-alloc
# after warmup). UNLIKE the persistent outer ctor, this *aliases* `x` directly
# — the wrapper lifetime is bounded by the enclosing `@with_pool` scope, so
# aliasing is safe and avoids materializing a `Vector{T}` from view inputs
# (e.g. `view(big_grid, k:k+N)`). The `I` type parameter tracks the concrete
# storage type so downstream `getindex` specializes per concrete inner.
@inline function _cache_axis_pooled(pool, x::AbstractVector{T}) where {T}
    n = length(x)
    n >= 2 || throw(ArgumentError("_cache_axis_pooled requires at least 2 grid points, got $n"))
    Tinv = typeof(inv(oneunit(T)))
    h = acquire!(pool, T, n - 1)
    inv_h = acquire!(pool, Tinv, n - 1)
    @inbounds for i in 1:(n - 1)
        h[i] = x[i + 1] - x[i]
        inv_h[i] = inv(h[i])
    end
    return _CachedVector(x, h, inv_h)
end

# Range / pre-wrapped passthroughs — no buffer to acquire (Range has cached
# scalar h, pre-wrapped axes already carry their caches).
@inline _cache_axis_pooled(_, x::AbstractRange) = _to_float(x, float(eltype(x)))
@inline _cache_axis_pooled(_, x::_CachedRange) = x
@inline _cache_axis_pooled(_, x::_CachedVector) = x

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
# Unified 2-arg accessors: _get_h, _get_inv_h
# ========================================
#
# After this file loads, the dispatch surface for `_get_h(grid, i)` is:
#   _CachedVector → cached vector lookup (this file)
#   _CachedRange  → cached scalar       (this file, also covers AbstractRange via inheritance)
#   AbstractRange → step()              (this file, fallback for non-_CachedRange ranges)
#   AbstractVector → on-the-fly diff    (this file, fallback for raw Vectors / one-shot path)

# _CachedVector — cached vector lookup (most specific for AbstractVector hierarchy)
@inline Base.@propagate_inbounds _get_h(x::_CachedVector, i::Int) = @inbounds x.h[i]
@inline Base.@propagate_inbounds _get_inv_h(x::_CachedVector, i::Int) = @inbounds x.inv_h[i]

# _CachedRange — cached scalar (most specific for AbstractRange hierarchy)
@inline _get_h(x::_CachedRange, ::Int) = x.h
@inline _get_inv_h(x::_CachedRange, ::Int) = x.inv_h

# AbstractRange (non-_CachedRange) — uniform spacing via step()
@inline _get_h(x::AbstractRange, ::Int) = step(x)
@inline _get_inv_h(x::AbstractRange, ::Int) = inv(step(x))

# AbstractVector — compute on-the-fly (one-shot path, no cache available).
# Raw eltype preserved (`Int → Int`, `Rational → Rational`). Arithmetic kernels
# that follow this with `inv_h * y * dL` auto-promote naturally — no need to
# eager-Float here. `_get_inv_h` returns `typeof(inv(h))` (Float64 for Int h,
# Rational for Rational h, T for Float T).
@inline Base.@propagate_inbounds _get_h(x::AbstractVector, i::Int) =
    @inbounds x[i + 1] - x[i]
@inline Base.@propagate_inbounds _get_inv_h(x::AbstractVector, i::Int) =
    inv(_get_h(x, i))

# 3-arg: caller already has `xL` / `xR` in registers from search → bypass the
# `x[i]` / `x[i+1]` loads. Used by oneshot kernels (and the wrapper-level
# `_ExclusivePeriodicAxis` 3-arg overload at `periodic_axis.jl:384`, which
# delegates here for any `_CachedVector` / raw `Vector` inner).
@inline _get_h(::AbstractVector, xL::Real, xR::Real) = xR - xL
@inline _get_inv_h(::AbstractVector, xL::Real, xR::Real) = inv(xR - xL)

# Persistent axis wrapping is split into two stages (see `periodic_axis.jl`):
#   - outer surface API: `_cache_axis(x, bc)` — bc-aware wrap, zero-copy
#     of buffer (Vector → `_CachedVector`, Range → `_CachedRange`,
#     `:exclusive` → `_ExclusivePeriodicAxis`),
#   - inner constructor:  `_convert_copy(x, Tg)` — ownership copy +
#     element-type promotion (wrapper-preserving `Base.copy`).
# Cubic 1D builds owned cache axes via `_cache_axis(_convert_copy(x, T), bc)`
# (copy-then-wrap): one buffer copy + eltype conversion, then alias the
# fresh buffer and build h/inv_h once.
