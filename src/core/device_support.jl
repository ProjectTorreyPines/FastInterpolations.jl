# ========================================
# Device storage support (GPU-agnostic)
# ========================================
# FI-owned types the GPU extensions build interpolants from. Nothing here depends on a GPU
# package: the extensions decide WHEN storage is device storage; these types define how the
# shared evaluation code reads it.


"""
    _DeviceArray{T, N, A} <: AbstractArray{T, N}
    _DeviceVector{T, A} = _DeviceArray{T, 1, A}

Device storage leaf: the GPU counterpart of the CPU storage an interpolant holds (plain
`Vector`/`Array` data, `_CachedVector` grids). Wraps a GPU array — or, inside a kernel, its
device view — by reference; it never copies. Grids and data use the same type: the enclosing
field decides the role.

- Endpoint reads are unchecked: interpolant storage holds ≥ 2 elements by construction, and
  inside a GPU kernel a bounds check leaves an exception path that costs every launch.
- No spacing caches: a device grid computes `x[i+1] - x[i]` per query (on GPUs this measured
  no slower than loading cached `h`/`inv_h`, and faster for random queries on large grids).
- Not read element-wise on the host (each read would be a device transfer): `show` prints
  the storage type and size only.
"""
struct _DeviceArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    inner::A
end
const _DeviceVector{T, A} = _DeviceArray{T, 1, A}

Base.parent(a::_DeviceArray) = a.inner
Base.size(a::_DeviceArray) = size(a.inner)
Base.IndexStyle(::Type{<:_DeviceArray}) = IndexLinear()
@inline Base.@propagate_inbounds Base.getindex(a::_DeviceArray, i::Int) = a.inner[i]
@inline Base.first(a::_DeviceArray) = @inbounds a.inner[begin]
@inline Base.last(a::_DeviceArray) = @inbounds a.inner[end]

# Allocation, copies and broadcasts follow the wrapped array, as for Base's array wrappers:
# code working on stored fields gets device arrays back and never indexes them on the host.
Base.similar(a::_DeviceArray, ::Type{T}, dims::Dims) where {T} = similar(a.inner, T, dims)
Base.copy(a::_DeviceArray) = _DeviceArray(copy(a.inner))
Base.BroadcastStyle(::Type{<:_DeviceArray{T, N, A}}) where {T, N, A} = Base.BroadcastStyle(A)
Base.dataids(a::_DeviceArray) = Base.dataids(a.inner)

Base.show(io::IO, a::_DeviceArray) =
    print(io, "_DeviceArray(", join(size(a), "×"), " ", nameof(typeof(a.inner)), "{", eltype(a), "})")
Base.show(io::IO, ::MIME"text/plain", a::_DeviceArray) = show(io, a)

# Interpolant display: a device grid prints its size only (its endpoints would need a transfer).
function _show_grid_row(io::IO, is_last::Bool, x::_DeviceVector)
    _show_print(io, is_last ? "└─ " : "├─ ", :light_black)
    _show_print(io, "Grid:  ", :light_black)
    _show_print(io, "Device vector", :magenta)
    return print(io, ", $(length(x)) points")
end

function _show_axis_detail(io::IO, g::_DeviceVector)
    _show_print(io, "Device vector", :magenta)
    return print(io, ", $(length(g)) points")
end

# Search on a device grid: threads share no hint, so AutoSearch is a per-query binary search
# (the policy counterpart of `AbstractRange → DirectSearch`); query order is never probed.
@inline _resolve_search_policy(::_DeviceVector, xq, ::AutoSearch, hint) = BinarySearch()

# Whether (adapted) storage lives on a device. The GPUArrays extension answers `true` for
# GPU arrays; everything else but the leaf is host storage.
_is_device_storage(::Any) = false
_is_device_storage(::_DeviceArray) = true

# ── Storage protocol ──
# A family declares its storage by role, and how to rebuild itself around new storage.
# Moving an interpolant (Adapt) and admitting it to GPU batch evaluation are written once
# against these two methods, not per family:
#   _storage(itp) -> (grids = (…,), values = (…,), extraps = (…,))
#     grids    the axis vectors, in axis order
#     values   the arrays the evaluation reads; the first one carries the value type
#     extraps  the extrapolations (a fill value follows the value type)
#   _rebuild(itp, s) -> the same family around storage `s`, through its raw constructor
# A family without a declaration is left unchanged by moves and is not admitted to GPU
# batch evaluation.
_storage(::Any) = nothing

# ── Construction seams ──
# The GPUArrays extension turns device inputs into `_DeviceVector` leaves at the shared
# normalization seams. A leaf is already normalized: re-entry (outer → inner constructor)
# passes it through whether or not axis caching is requested (a device grid keeps no
# spacing caches), converting its eltype on the device when the promoted grid type differs;
# ownership copies stay on the device.
#
# Boundary conditions whose stored grid is a plain grid (exclusive periodic data is extended
# to `:extended` before the grid is normalized); the GPU seams accept exactly these.
const _PlainAxisBC = Union{NoBC, PeriodicBC{:inclusive}, PeriodicBC{:extended}}

@inline _policy_axis(x::_DeviceVector, ::_PlainAxisBC, ::Type{Tg}, ::StorePolicy{CG, CV, true}) where {Tg, CG, CV} =
    _device_eltype(x, Tg)
@inline _policy_axis(x::_DeviceVector, ::_PlainAxisBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} =
    _device_eltype(x, Tg)
# Tg-less form (ND factories: grids arrive already value-promoted).
@inline _policy_axis(x::_DeviceVector, ::_PlainAxisBC, ::StorePolicy{CG, CV, true}) where {CG, CV} = x
@inline _policy_axis(x::_DeviceVector, ::_PlainAxisBC, ::StorePolicy{CG, CV, false}) where {CG, CV} = x
@inline _device_eltype(x::_DeviceVector{T}, ::Type{T}) where {T} = x
@inline _device_eltype(x::_DeviceVector, ::Type{T}) where {T} = _DeviceArray(T.(x))

@inline _convert_copy(x::_DeviceVector{T}, ::Type{T}) where {T} = copy(x)
@inline _convert_copy(x::_DeviceVector, ::Type{T}) where {T} = _DeviceArray(T.(x))
