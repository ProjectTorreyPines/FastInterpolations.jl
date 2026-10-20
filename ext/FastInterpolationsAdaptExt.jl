# ========================================
# Adapt Extension for FastInterpolations.jl
# ========================================
# Moves an interpolant between storage kinds and converts it into a kernel argument:
#   ditp = adapt(MtlArray, linear_interp(x, y; extrap = ClampExtrap()))   # or CuArray, …
#   ditp.(xq_gpu)                                                         # broadcast kernel
#   adapt(Array, ditp)                                                    # back to the host
#
# Interpolants are always rebuilt from their adapted fields through the raw constructor, so
# a host adaptation (`adapt(Array, itp)`) yields a value `===` the original. The parent rule
# knows each field's role, so the one device leaf (`_DeviceArray`) returns to the right host
# storage: a grid to `_CachedVector` (caches rebuilt), data to a plain array. Under kernel
# conversion a leaf only swaps its wrapped array for the device view.
#
# NoExtrap builds its error message with String interpolation, which GPU compilers reject:
# under broadcast use a non-throwing extrap.

module FastInterpolationsAdaptExt

using FastInterpolations
using FastInterpolations: LinearInterpolant, DerivativeView, _CachedRange, _CachedVector,
    _DeviceArray, _DeviceVector, _Generic, _RawFields, _is_device_storage, _to_float,
    _promote_extrap
import Adapt

function Adapt.adapt_structure(to, itp::LinearInterpolant)
    y = _adapt_data(to, itp.y)
    x = _adapt_axis(to, itp.x, eltype(itp.y), eltype(y))
    extrap = eltype(y) === eltype(itp.y) ? itp.extrap : _promote_extrap(itp.extrap, eltype(y))
    return LinearInterpolant(_RawFields(), x, y, extrap, itp.search_policy)
end

Adapt.adapt_structure(to, d::DerivativeView{O}) where {O} = _derivative_view(Val(O), Adapt.adapt(to, d.parent))
@inline _derivative_view(::Val{O}, p) where {O} = DerivativeView{O, typeof(p)}(p)

# ── Data ──
_adapt_data(to, y::_DeviceArray) = _rewrap_data(Adapt.adapt(to, parent(y)))
_adapt_data(to, y) = _wrap_device(Adapt.adapt(to, y))

_rewrap_data(a::Array) = a                      # back on the host
_rewrap_data(a) = _DeviceArray(a)               # kernel view (or another device array)

_wrap_device(a) = _is_device_storage(a) ? _DeviceArray(a) : a

# ── Grids ──
# A uniform grid is plain scalars: it travels by value. When the move changes the data's
# float precision (e.g. `Metal.mtl`: Float64 → Float32), a grid of that same precision is
# rebuilt in the new one, as a fresh range would be.
_adapt_axis(_, x::_CachedRange, _, _) = x
_adapt_axis(_, x::_CachedRange{T, Tinv, _Generic}, ::Type{T}, ::Type{Tv2}) where {T <: AbstractFloat, Tinv, Tv2 <: AbstractFloat} =
    T === Tv2 ? x : _to_float(x, Tv2)

function _adapt_axis(to, x::_CachedVector, _, _)
    inner = Adapt.adapt(to, x.inner)
    _is_device_storage(inner) && return _DeviceArray(inner)     # spacing caches dropped
    return inner === x.inner ? x : _CachedVector(inner)         # host copy / conversion
end

function _adapt_axis(to, x::_DeviceVector, _, _)
    inner = Adapt.adapt(to, parent(x))
    return inner isa Array ? _CachedVector(inner) : _DeviceArray(inner)
end

_adapt_axis(to, x::AbstractVector, _, _) = _wrap_device(Adapt.adapt(to, x))   # raw grids

# Standalone leaves keep their kind (interpolants route their fields through the rules above).
Adapt.adapt_structure(_, x::_CachedRange) = x
Adapt.adapt_structure(to, a::_DeviceArray) = _DeviceArray(Adapt.adapt(to, parent(a)))

end # module
