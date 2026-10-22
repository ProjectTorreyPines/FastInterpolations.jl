# ========================================
# Adapt Extension for FastInterpolations.jl
# ========================================
# Moves an interpolant between storage kinds and converts it into a kernel argument:
#   ditp = adapt(MtlArray, linear_interp(x, y; extrap = ClampExtrap()))   # or CuArray, …
#   ditp.(xq_gpu)                                                         # broadcast kernel
#   adapt(Array, ditp)                                                    # back to the host
#
# One rule serves every family that declares its storage (`_storage` / `_rebuild`): each
# stored array is adapted by its role and the family is rebuilt through its raw
# constructor, so a host adaptation (`adapt(Array, itp)`) yields a value `===` the original.
# The role returns the one device leaf (`_DeviceArray`) to the right host storage: a grid to
# `_CachedVector` (caches rebuilt), values to a plain array. Under kernel conversion a leaf
# only swaps its wrapped array for the device view.
#
# Precision follows the adaptor (e.g. `cu`, `mtl`: Float64 → Float32): it converts the
# arrays and the fill values, and a uniform grid converts as the vector grid it stands for.
#
# NoExtrap builds its error message with String interpolation, which GPU compilers reject:
# under broadcast use a non-throwing extrap.

module FastInterpolationsAdaptExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant1D, AbstractInterpolantND, DerivativeView, _CachedRange, _CachedVector,
    _DeviceArray, _DeviceVector, _RawFields, _is_device_storage, _storage, _rebuild, _to_float,
    _promote_extrap, _ExclusivePeriodicAxis, _ExclusivePeriodicData
import Adapt

Adapt.adapt_structure(to, itp::Union{AbstractInterpolant1D, AbstractInterpolantND}) =
    _adapt_itp(to, itp, _storage(itp))

_adapt_itp(_, itp, ::Nothing) = itp                 # no storage declaration: left as it is
function _adapt_itp(to, itp, s::NamedTuple)
    values = map(v -> _adapt_data(to, v), s.values)
    Tv, Tv2 = eltype(first(s.values)), eltype(first(values))
    extraps = map(e -> _adapt_extrap(to, e, Tv, Tv2), s.extraps)
    return _rebuild(itp, (grids = map(g -> _adapt_axis(to, g), s.grids), values = values, extraps = extraps))
end

Adapt.adapt_structure(to, d::DerivativeView{O}) where {O} = _derivative_view(Val(O), Adapt.adapt(to, d.parent))
@inline _derivative_view(::Val{O}, p) where {O} = DerivativeView{O, typeof(p)}(p)

# ── Values ──
_adapt_data(to, y::_DeviceArray) = _rewrap_data(Adapt.adapt(to, parent(y)))
_adapt_data(to, y) = _wrap_device(Adapt.adapt(to, y))

_rewrap_data(a::Array) = a                      # back on the host
_rewrap_data(a) = _DeviceArray(a)               # kernel view (or another device array)

_wrap_device(a) = _is_device_storage(a) ? _DeviceArray(a) : a

# A fill value moves as the adaptor moves scalars (`mtl`: Float64 → Float32, inside static
# vectors too), then takes the new value type, as at construction.
_adapt_extrap(to, e, ::Type{Tv}, ::Type{Tv}) where {Tv} = Adapt.adapt(to, e)
_adapt_extrap(to, e, _, ::Type{Tv2}) where {Tv2} = _promote_extrap(Adapt.adapt(to, e), Tv2)
Adapt.adapt_structure(to, e::FillExtrap) = (v = Adapt.adapt(to, e.fill_value); FillExtrap{typeof(v)}(v))

# ── Grids ──
# A uniform grid is plain scalars: it travels by value, in the eltype the adaptor gives a
# vector grid of its eltype (`_to_float` keeps the range tag and re-derives a
# `_WidenedDomain` bracket in the new precision).
function _adapt_axis(to, x::_CachedRange{T}) where {T}
    T2 = _adapted_eltype(to, T)
    return T2 === T ? x : _to_float(x, T2)
end
_adapted_eltype(to, ::Type{T}) where {T} = eltype(Adapt.adapt(to, Vector{T}(undef, 0)))   # empty probe

function _adapt_axis(to, x::_CachedVector)
    inner = Adapt.adapt(to, x.inner)
    _is_device_storage(inner) && return _DeviceArray(inner)     # spacing caches dropped
    return inner === x.inner ? x : _CachedVector(inner)         # host copy / conversion
end

function _adapt_axis(to, x::_DeviceVector)
    inner = Adapt.adapt(to, parent(x))
    return inner isa Array ? _CachedVector(inner) : _DeviceArray(inner)
end

_adapt_axis(to, x::AbstractVector) = _wrap_device(Adapt.adapt(to, x))   # raw grids

# Periodic wrappers (one-shot `:exclusive` storage) keep their kind around the adapted inner.
Adapt.adapt_structure(to, g::_ExclusivePeriodicAxis{Te, X, Tp}) where {Te, X, Tp} =
    (inner = Adapt.adapt(to, g.inner); _ExclusivePeriodicAxis{Te, typeof(inner), Tp}(_RawFields(), inner, g.period, g._x_max))
Adapt.adapt_structure(to, c::_ExclusivePeriodicData) = _ExclusivePeriodicData(Adapt.adapt(to, c.inner))

# Standalone leaves (interpolants route their fields through the rules above).
Adapt.adapt_structure(to, x::_CachedRange) = _adapt_axis(to, x)
Adapt.adapt_structure(to, a::_DeviceArray) = _DeviceArray(Adapt.adapt(to, parent(a)))

end # module
