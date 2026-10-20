# ========================================
# Device storage support (GPU-agnostic)
# ========================================
# FI-owned types the GPU extensions build interpolants from. Nothing here depends on a GPU
# package: the extensions decide WHEN storage is device storage; these types define how the
# shared evaluation code reads it.

"""
    _RawFields

Sentinel selecting an interpolant's raw inner constructor: fields are stored as given,
already normalized by an earlier construction. Rebuilds an interpolant around moved arrays
(Adapt), where the normalizing constructor would copy or normalize on the host.
"""
struct _RawFields end

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
@inline Base.first(v::_DeviceVector) = @inbounds v.inner[begin]
@inline Base.last(v::_DeviceVector) = @inbounds v.inner[end]

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

# Whether (adapted) storage lives on a device. The GPUArrays extension answers `true` for
# GPU arrays; everything else is host storage.
_is_device_storage(::Any) = false
