# Componentwise convex blend for parametric arithmetic colorants.
#
# The core blend (`_linear_value_blend`, src/linear/linear_kernels.jl) knows
# only FMA/Generic. This extension owns ALL colorant dispatch: it overrides
# the blend ENTRY for same-type parametric colorant pairs and picks per pair:
#
#   componentwise — when the CHANNEL result `weight × channel` is native-FMA
#                   (the core `_linear_blend_style` rule applied to the
#                   channel type): `mapc` re-enters the styled blend per
#                   channel with α PRESERVED, so each channel takes the
#                   verbatim 2-FMA float form. Flattening α to a generic
#                   weight pair instead costs the channel fusion (~half the
#                   win, and 4-channel N0f8 turns into a small loss).
#   generic       — ineligible channels (BigFloat) or weights (Dual/BigFloat):
#                   explicit escape to the core generic styled method — the
#                   plain ColorVectorSpace arithmetic path, no recursion.
#
# Coverage is per parametric family (Gray/RGB/RGBA/AGray/GrayA/ARGB), every
# gate-eligible channel — benchmark-verified wins on the real 2D kernel
# (M1, min ns/eval vs generic: Gray{N0f8} 2.93/3.43, RGB{Float64} 3.08/4.10,
# RGBA{N0f8} 8.39/9.27, AGray{N0f8} 4.69/19.11). Packed colorants
# (Gray24/RGB24/ARGB32) do NOT match these entries (`Gray24 <: AbstractGray`
# but not `<: Gray`) and mixed concrete pairs fail the same-`C` constraint —
# both fall through to the core generic path and keep the widened arithmetic
# output (a naive mapc would re-quantize packed storage: wrong value AND type).
#
# ColorVectorSpace is a trigger (not just ColorTypes) so this extension is a
# pure perf overlay on arithmetic that already exists — it must never make
# color interpolation newly work without ColorVectorSpace loaded.
#
# `@inline` is load-bearing on the blend entries: without it the 4-channel
# N0f8 body exceeds the inline cost threshold and a per-node call costs ~55%.
module FastInterpolationsColorVectorSpaceExt

using ColorTypes: AGray, ARGB, Colorant, Gray, GrayA, RGB, RGBA, mapc
using ColorVectorSpace: ColorVectorSpace
using FastInterpolations
import FastInterpolations:
    _LinearBlendFMA,
    _LinearBlendGeneric,
    _LinearBlendStyle,
    _linear_blend_style,
    _linear_value_blend

# Entry overrides — one per parametric family (keeps per-family opt-out
# possible and packed types structurally excluded).
@inline _linear_value_blend(α, yL::C, yR::C) where {C <: Gray} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)
@inline _linear_value_blend(α, yL::C, yR::C) where {C <: RGB} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)
@inline _linear_value_blend(α, yL::C, yR::C) where {C <: RGBA} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)
@inline _linear_value_blend(α, yL::C, yR::C) where {C <: AGray} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)
@inline _linear_value_blend(α, yL::C, yR::C) where {C <: GrayA} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)
@inline _linear_value_blend(α, yL::C, yR::C) where {C <: ARGB} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)

# Gate = the core style rule applied to the CHANNEL type (weight included:
# a Dual/BigFloat weight disqualifies even native-float channels).
@inline _color_linear_blend_style(::Type{A}, ::Type{C}) where {A, C <: Colorant} =
    _color_style_tag(_linear_blend_style(A, eltype(C)))
@inline _color_style_tag(::_LinearBlendFMA) = Val(:componentwise)
@inline _color_style_tag(::_LinearBlendStyle) = Val(:generic)

# Componentwise: α preserved into each channel; `mapc` reassembles the widened
# arithmetic color type (RGB{N0f8} → RGB{Float64}, matching the generic path).
@inline _color_linear_value_blend(::Val{:componentwise}, α, yL, yR) =
    mapc((l, r) -> _linear_value_blend(α, l, r), yL, yR)
# Generic: explicit escape to the core styled method (no recursion through
# the colorant entries above).
@inline _color_linear_value_blend(::Val{:generic}, α, yL, yR) =
    _linear_value_blend(_LinearBlendGeneric(), α, yL, yR)

end
