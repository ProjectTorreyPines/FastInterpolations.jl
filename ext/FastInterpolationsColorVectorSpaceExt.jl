# Componentwise lincomb opt-in for parametric arithmetic colorants.
#
# Gate: a family opts into `_LincombComponentwise` only when the per-channel
# blend `weight × channel` lands in a native-FMA type (`_channel_blend_fuses`)
# — `Float64 × Gray{N0f8}` qualifies (channel math is Float64); BigFloat/Dual
# weights or channels stay generic. Packed colorants (Gray24/RGB24/ARGB32) get
# NO style method: they keep the generic path and its widened arithmetic output
# (a naive mapc would re-quantize into the packed type — wrong value AND type).
#
# ColorVectorSpace is a trigger (not just ColorTypes) so this extension is a
# pure perf overlay on arithmetic that already exists — it must never make
# color interpolation newly work without ColorVectorSpace loaded.
module FastInterpolationsColorVectorSpaceExt

using ColorTypes: AGray, ARGB, Colorant, Gray, GrayA, RGB, RGBA, mapc
using ColorVectorSpace: ColorVectorSpace
using FastInterpolations
import FastInterpolations:
    _LincombComponentwise,
    _channel_blend_fuses,
    _lincomb2,
    _lincomb_style,
    _style_from_fuses

@inline _lincomb_style(::Type{A}, ::Type{Gray{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{RGB{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{AGray{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{GrayA{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
# 4-channel families opt in for native-float channels ONLY. RGBA{N0f8}/
# ARGB{N0f8} measurably regress (~1.5×) on the real 2D kernel path despite
# passing the channel-FMA gate: three 4-channel FMA chains plus the N0f8
# conversions exceed the kernel's register/inlining budget (the isolated
# value-stream blend shows parity — the effect is kernel-context only).
@inline _lincomb_style(::Type{A}, ::Type{RGBA{T}}) where {A, T <: Base.IEEEFloat} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{ARGB{T}}) where {A, T <: Base.IEEEFloat} =
    _style_from_fuses(_channel_blend_fuses(A, T))

# Componentwise blend for SAME-type colorant pairs: per-channel scalar
# `_lincomb2` (native FMA), reassembled by `mapc` into the widened arithmetic
# color type (Float64 channel results ⇒ e.g. RGB{N0f8} → RGB{Float64}, matching
# the generic path's output). A mixed concrete pair skips this method and falls
# to the core safety net — reachable only through a direct `_lincomb2` call:
# FI's linear kernel constrains both endpoints to one `Tv`, so kernel traffic
# is always same-type.
@inline _lincomb2(::_LincombComponentwise, a, x::C, b, y::C) where {C <: Colorant} =
    mapc((xc, yc) -> _lincomb2(a, xc, b, yc), x, y)

end
