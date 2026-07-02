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
    _linear_value_blend,
    _style_from_fuses

@inline _lincomb_style(::Type{A}, ::Type{Gray{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{RGB{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{AGray{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{GrayA{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
# 4-channel families included: with the CONVEX blend hook below (α preserved
# into each channel), RGBA{N0f8}/ARGB{N0f8} win on the real 2D kernel
# (interleaved min 8.39 vs 9.27 / 8.39 vs 9.83 ns/eval, M1). Under the earlier
# α-flattened `_lincomb2(1-α, xc, α, yc)` channel form they LOST 2-4% — the
# convex structure is load-bearing, as is the blend method's @inline (without
# it the 4-channel N0f8 body exceeds the inline cost threshold and a
# per-node call costs ~55%).
@inline _lincomb_style(::Type{A}, ::Type{RGBA{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))
@inline _lincomb_style(::Type{A}, ::Type{ARGB{T}}) where {A, T} =
    _style_from_fuses(_channel_blend_fuses(A, T))

# Componentwise CONVEX blend for SAME-type colorant pairs: α is preserved into
# each channel so the per-channel call re-enters `_linear_value_blend` and
# takes the verbatim 2-FMA float form (muladd(α, yc, muladd(-α, xc, xc))).
# Lowering to `_lincomb2(1-α, xc, α, yc)` instead loses the convex structure —
# the channel gets mul+muladd, the fusion breaks, and roughly half the win
# evaporates (and 4-channel N0f8 turns into a small loss). `mapc` reassembles
# the widened arithmetic color type (RGB{N0f8} → RGB{Float64}, matching the
# generic path's output). A mixed concrete pair skips this method and falls to
# the core safety net — FI's linear kernel constrains both endpoints to one
# `Tv`, so kernel traffic is always same-type.
@inline _linear_value_blend(::_LincombComponentwise, α, x::C, y::C) where {C <: Colorant} =
    mapc((xc, yc) -> _linear_value_blend(α, xc, yc), x, y)

# General weighted-pair primitive for direct `_lincomb2` users (future kernels
# with non-convex weights). The linear blend above does NOT route through this.
@inline _lincomb2(::_LincombComponentwise, a, x::C, b, y::C) where {C <: Colorant} =
    mapc((xc, yc) -> _lincomb2(a, xc, b, yc), x, y)

end
