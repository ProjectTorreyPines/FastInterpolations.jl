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
# the grid type space for persistent paths is exactly:
#   {_CachedRange{T}, _CachedVector{T,Tinv}}
# enabling a single 2-arg `_get_h(grid, i)` dispatch surface.
#
# Include order: grid_spacing.jl → cached_range.jl → cached_vector.jl → ...

"""
    _CachedVector{T, Tinv} <: AbstractVector{T}

Vector grid wrapper that caches per-cell spacing (`h`) and reciprocal
(`inv_h`) for fast O(1) lookup during persistent interpolant evaluation.

# Fields
- `inner::Vector{T}` — Original grid values (length n). Always concrete `Vector{T}`.
- `h::Vector{T}` — Cell widths, `h[i] = inner[i+1] - inner[i]` (length n-1).
- `inv_h::Vector{Tinv}` — Precomputed reciprocals (length n-1).

# Type parameters
- `T` — Grid element type. Can be `Int`, `Float64`, `ForwardDiff.Dual`,
  `Measurements.Measurement`, etc. Any type with `-` and `inv` defined.
- `Tinv` — Reciprocal type, equal to `typeof(inv(oneunit(T)))`. For Float
  grids `Tinv == T`; for Int grids `Tinv == Float64` (since `inv(::Int)`
  returns `Float64`). Mirrors the historical `VectorSpacing{T,Tinv}` design.

# Behavior
- `<: AbstractVector{T}` so existing search and eval kernels accept it
  transparently — same trick `_CachedRange` uses with `<: AbstractRange`.
- `Base.getindex(c, i)` forwards to `inner[i]` — zero overhead vs raw `Vector`.
- `length(c) == length(c.inner)` — same as raw vector.

# Example
```julia
x = [0.0, 0.3, 0.7, 1.0]      # non-uniform Float grid
cv = _CachedVector(x)          # _CachedVector{Float64, Float64}

x_int = [0, 1, 3, 6]           # Int grid
cv_int = _CachedVector(x_int)  # _CachedVector{Int, Float64}
```
"""
struct _CachedVector{T, Tinv} <: AbstractVector{T}
    inner::Vector{T}
    h::Vector{T}        # length n-1
    inv_h::Vector{Tinv} # length n-1
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
# Default Base `copy(::AbstractVector)` uses `similar` + `copyto!` which
# materializes wrappers as plain `Vector{T}`, destroying the wrapper type.
# Persistent ND constructors do `map(copy, grids)` for mutation safety;
# without this overload, that call would silently drop the cached `h`/`inv_h`
# fields on every grid axis. Recursive copy of inner buffers preserves the
# wrapper structure AND ensures the returned wrapper owns fresh storage.
@inline Base.copy(c::_CachedVector) =
    _CachedVector{eltype(c.inner), eltype(c.inv_h)}(copy(c.inner), copy(c.h), copy(c.inv_h))

# ---------- Wrapper-aware `_convert_copy` ----------
# Single-pass type-conversion + ownership. Same-type → delegate to
# `Base.copy` (preserves wrapper, recursive copy of inner). Different-type
# → rebuild wrapper from a type-converted inner (one allocation pass —
# `_convert_copy(c.inner, T)` produces the new buffer; `_CachedVector`
# constructor recomputes `h`/`inv_h` for the converted eltype).
@inline _convert_copy(c::_CachedVector{T}, ::Type{T}) where {T} = copy(c)
@inline _convert_copy(c::_CachedVector, ::Type{T}) where {T} =
    _CachedVector(_convert_copy(c.inner, T))

# ---------- Construction from AbstractVector ----------
"""
    _CachedVector(x::AbstractVector{T})

Construct a `_CachedVector` from a raw vector. Computes `h[i] = x[i+1] - x[i]`
and `inv_h[i] = inv(h[i])` once, in a single pass.

If `x` is not already a concrete `Vector{T}`, it is converted via `Vector(x)`
to ensure stable storage layout (mirrors `VectorSpacing` precedent).

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
    return _CachedVector{T, Tinv}(inner, h, inv_h)
end

# ========================================
# Unified 2-arg accessors: _get_h, _get_inv_h
# ========================================
#
# After this file loads, the dispatch surface for `_get_h(grid, i)` is:
#   _CachedVector → cached vector lookup (this file)
#   _CachedRange  → cached scalar       (this file, also covers AbstractRange via inheritance)
#   AbstractRange → step()              (this file, fallback for non-_CachedRange ranges)
#   AbstractVector → on-the-fly diff    (this file, fallback for raw Vectors / one-shot path)
#
# The pre-existing 2-arg `_get_h(::ScalarSpacing,i)` / `_get_h(::VectorSpacing,i)`
# in grid_spacing.jl remain alongside (dual-path) until Step 3 cleanup.
# Dispatch is unambiguous because spacing types are NOT <: AbstractVector.

# _CachedVector — cached vector lookup (most specific for AbstractVector hierarchy)
@inline Base.@propagate_inbounds _get_h(x::_CachedVector, i::Int) = @inbounds x.h[i]
@inline Base.@propagate_inbounds _get_inv_h(x::_CachedVector, i::Int) = @inbounds x.inv_h[i]

# _CachedRange — cached scalar (most specific for AbstractRange hierarchy)
@inline _get_h(x::_CachedRange, ::Int) = x.h
@inline _get_inv_h(x::_CachedRange, ::Int) = x.inv_h

# AbstractRange (non-_CachedRange) — uniform spacing via step()
# float() ensures Int ranges produce Float h/inv_h for kernel compatibility.
@inline _get_h(x::AbstractRange, ::Int) = float(step(x))
@inline _get_inv_h(x::AbstractRange, ::Int) = inv(float(step(x)))

# AbstractVector — compute on-the-fly (one-shot path, no cache available)
# Used for raw user `x::Vector` in oneshot APIs. Subtraction is single-cycle
# and `x[i]`/`x[i+1]` are typically cache-resident from the search step.
@inline Base.@propagate_inbounds _get_h(x::AbstractVector, i::Int) =
    @inbounds float(x[i + 1] - x[i])
@inline Base.@propagate_inbounds _get_inv_h(x::AbstractVector, i::Int) =
    inv(_get_h(x, i))

# ========================================
# Backward-compat: _create_spacing(::_CachedVector) → VectorSpacing
# ========================================
#
# Existing 13 interpolant constructors call `_create_spacing(xc)` where `xc`
# is the result of `_store_grid(x, Tg)`. Once Step 1.5 extends `_store_grid`
# to wrap Vector inputs in `_CachedVector`, those constructors hand a
# `_CachedVector` (not a raw `Vector`) to `_create_spacing`.
#
# This shim returns a `VectorSpacing` extracted from the cached fields —
# zero recomputation, zero allocation beyond the new struct. After Step 2
# migrates all methods to grid-based access, this shim and `VectorSpacing`
# itself are deleted in Step 3.
@inline _create_spacing(c::_CachedVector{T, Tinv}) where {T, Tinv} =
    VectorSpacing{T, Tinv}(c.h, c.inv_h)

# ========================================
# _store_grid_cached: persistent grid wrapping
# ========================================
#
# Companion to `_store_grid` (utils.jl). Where `_store_grid` is the
# lightweight normalizer used by cache-lookup paths (Vector inputs stay
# raw to avoid per-call wrap allocations on autocache hits),
# `_store_grid_cached` is for persistent interpolant constructors that
# build the struct ONCE — they pay the wrap cost upfront so every
# subsequent eval gets O(1) `_get_h`/`_get_inv_h` cached lookups.
#
# After Step 2 migration, persistent constructors call this instead of
# `_store_grid`. Resulting `xc` is always one of:
#   {`_CachedRange{Tg}`, `_CachedVector{Tg, Tinv}`}
# enabling unified `_get_h(xc, i)` dispatch with cached lookup.
@inline _store_grid_cached(x::AbstractVector, ::Type{Tg}) where {Tg} =
    _CachedVector(_convert_copy(x, Tg))
@inline _store_grid_cached(x::AbstractRange, ::Type{Tg}) where {Tg} = _to_float(x, Tg)
@inline _store_grid_cached(x::_CachedVector, ::Type{Tg}) where {Tg} = x
@inline _store_grid_cached(x::_CachedRange, ::Type{Tg}) where {Tg} = x
