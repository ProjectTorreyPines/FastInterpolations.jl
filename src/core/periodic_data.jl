# ========================================
# _ExclusivePeriodicData: data-side cyclic wrapper for `:exclusive` PeriodicBC
# ========================================
#
# Companion to `_ExclusivePeriodicAxis` (periodic_axis.jl): when the axis
# grid is wrapped to length n+1 to expose the virtual seam endpoint, the
# corresponding y/data array also gains a virtual `(n+1)`-th slot. Unlike
# the axis wrapper, the data wrapper has NO coordinate semantics — its
# only role is to cyclically expose `inner[1]` at the virtual slot so that
# `y[idx_R]` and `last(y)` work without explicit `_resolve_idx` calls.
#
# Generic over array dimension N so that:
#   - 1D `_ExclusivePeriodicData{Tv, 1, Vector{Tv}}` wraps the y vector for
#     scalar `LinearInterpolant` / `ConstantInterpolant`.
#   - Future ND `_ExclusivePeriodicData{Tv, N, Array{Tv,N}}` will wrap the
#     data array for ND interpolants. ND case needs to know *which* dims are
#     cyclic (since not every axis is necessarily periodic) — the design
#     stub for that lives below; for now only the 1D path is enabled.
#
# Include order: ... → periodic_axis.jl → periodic_data.jl → ...

"""
    _ExclusivePeriodicData{Tv, N, A<:AbstractArray{Tv, N}} <: AbstractArray{Tv, N}

Cyclic-indexing wrapper for the y/data side of a `:exclusive` PeriodicBC
interpolant. Companion to `_ExclusivePeriodicAxis` on the x/axis side.

`inner` is the user's raw value array (size matches the user's original
grid size, NO copy). Currently only N=1 is wired through interpolant
constructors; the type signature is generic over N so the same wrapper
naturally extends to ND data arrays in a future commit.

# 1D contract (N=1)
- `length(c) = length(c.inner) + 1` (virtual extension)
- `c[i] = c.inner[i]` for `i ≤ length(inner)`, `c[length(inner)+1] = c.inner[1]` (cyclic)
- `first(c) = inner[1]`, `last(c) = inner[1]` (cyclic boundary)

# Why "Data" vs "Axis"
This wrapper has no coordinate semantics — `last(c) = inner[1]`, NOT
`inner[1] + period` (that's the axis wrapper). It is a pure cyclic-indexing
adapter so eval kernels can write `y[idx_R]` uniformly without branching.

# Example
```julia
y = [10.0, 20.0, 30.0, 40.0]                       # length 4
yw = _ExclusivePeriodicData(y)                     # presented as length 5
yw[1], yw[2], yw[3], yw[4]                         # 10, 20, 30, 40 (raw)
yw[5]                                              # 10.0 (cyclic, = yw[1])
length(yw) == 5
last(yw) == 10.0
```
"""
struct _ExclusivePeriodicData{Tv, N, A <: AbstractArray{Tv, N}} <: AbstractArray{Tv, N}
    inner::A
end

# Convenience outer constructor — type params inferred. Currently expects 1D
# input; ND constructor is reserved for future commits that thread per-axis
# cyclicity through the wrapper.
@inline _ExclusivePeriodicData(inner::AbstractVector{Tv}) where {Tv} =
    _ExclusivePeriodicData{Tv, 1, typeof(inner)}(inner)

# Idempotent: re-wrapping returns input unchanged. Mirrors
# `_CachedVector(::_CachedVector) === input` and
# `_ExclusivePeriodicAxis` constructor patterns.
_ExclusivePeriodicData(c::_ExclusivePeriodicData) = c

# ---------- AbstractArray interface (1D specialized) ----------
Base.size(c::_ExclusivePeriodicData{Tv, 1}) where {Tv} = (length(c.inner) + 1,)
Base.length(c::_ExclusivePeriodicData{Tv, 1}) where {Tv} = length(c.inner) + 1
Base.eltype(::Type{<:_ExclusivePeriodicData{Tv}}) where {Tv} = Tv
Base.IndexStyle(::Type{<:_ExclusivePeriodicData}) = IndexLinear()
@inline Base.firstindex(::_ExclusivePeriodicData{Tv, 1}) where {Tv} = 1
@inline Base.lastindex(c::_ExclusivePeriodicData{Tv, 1}) where {Tv} = length(c)

# `getindex(c, n+1)` returns `inner[1]` (cyclic value at the virtual seam slot).
# In-bounds for normal indices it forwards to `inner` with @inbounds for full
# zero-overhead. The branch is single-compare and predicted not-seam.
@inline Base.@propagate_inbounds function Base.getindex(c::_ExclusivePeriodicData{Tv, 1}, i::Int) where {Tv}
    n = length(c.inner)
    @inbounds return i <= n ? c.inner[i] : c.inner[1]
end

# `first` / `last` follow the wrapper's virtual span: first stays inner[1];
# last is *also* inner[1] because the wrapped tail equals the wrapped head.
@inline Base.first(c::_ExclusivePeriodicData{Tv, 1}) where {Tv} = @inbounds c.inner[1]
@inline Base.last(c::_ExclusivePeriodicData{Tv, 1}) where {Tv} = @inbounds c.inner[1]

# ---------- `_convert_copy` overload ----------
# When the constructor calls `_convert_copy(y, Tv)` and `y` is already a
# `_ExclusivePeriodicData`, copy the inner vector so mutations don't leak
# between user and stored representation. Wrapper rebuilt on copied inner.
@inline _convert_copy(c::_ExclusivePeriodicData{T}, ::Type{T}) where {T} =
    _ExclusivePeriodicData(_convert_copy(c.inner, T))
@inline _convert_copy(c::_ExclusivePeriodicData, ::Type{T}) where {T} =
    _ExclusivePeriodicData(_convert_copy(c.inner, T))
