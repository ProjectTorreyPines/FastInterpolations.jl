# Componentwise convex blend for arithmetic colorants.
#
# The core blend (`_linear_value_blend`, src/linear/linear_kernels.jl) knows
# only FMA/Generic. This extension owns ALL colorant dispatch: it overrides
# the blend ENTRY for same-type arithmetic-colorant pairs and picks per pair:
#
#   componentwise — `mapc` re-enters the styled blend per channel with α
#                   PRESERVED, so each channel takes the verbatim 2-FMA float
#                   form. (Flattening α to a generic weight pair instead costs
#                   the channel fusion — ~half the win, and 4-channel N0f8
#                   turns into a small loss.)
#   generic       — explicit escape to the core generic styled method: the
#                   plain ColorVectorSpace arithmetic path, no recursion.
#
# GATE (fully compile-time, two conditions):
#   1. channel blend is native-FMA — the core `_linear_blend_style` rule on
#      `(weight, eltype(C))`; a Dual/BigFloat weight or channel stays generic.
#   2. the componentwise reconstruction type ≡ the natural whole-color
#      arithmetic type: `promote_op(mapc-blend, A, C, C) === promote_op(*, A, C)`.
#      This is what excludes PACKED colorants (Gray24/RGB24/ARGB32): CVS
#      arithmetic widens them (`0.3*Gray24 → Gray{Float64}`) but `mapc` can
#      only rebuild the packed type itself (no `Gray24{Float64}` exists) —
#      re-quantizing values, changing the result type, and throwing on
#      out-of-[0,1] extrapolants. Type mismatch ⇒ generic escape, which
#      preserves today's widened output exactly.
#
# Entry scope `_ArithColorant` = the four ColorTypes-public abstractions that
# ColorVectorSpace defines vector-space arithmetic for (≡ CVS's internal
# `MathTypes`, spelled without depending on it). Non-RGB color spaces (HSV,
# Lab, …) are outside — CVS deliberately defines no arithmetic there (hue is
# angular), so they fall to the core generic entry and fail with the same
# MethodError as today instead of silently inventing channelwise semantics.
# Mixed concrete pairs cannot bind the single `C` and also fall to core.
#
# ColorVectorSpace is a trigger (not just ColorTypes) so this extension is a
# pure perf overlay on arithmetic that already exists — it must never make
# color interpolation newly work without ColorVectorSpace loaded.
#
# `@inline` is load-bearing on the blend entry: without it the 4-channel
# N0f8 body exceeds the inline cost threshold and a per-node call costs ~55%.
module FastInterpolationsColorVectorSpaceExt

using ColorTypes:
    AbstractGray, AbstractRGB, Colorant, TransparentGray, TransparentRGB, mapc
using ColorVectorSpace: ColorVectorSpace
using FastInterpolations
import FastInterpolations:
    _LinearBlendFMA,
    _LinearBlendGeneric,
    _LinearBlendStyle,
    _linear_blend_style,
    _linear_value_blend

const _ArithColorant =
    Union{AbstractGray, AbstractRGB, TransparentGray, TransparentRGB}

@inline _linear_value_blend(α, yL::C, yR::C) where {C <: _ArithColorant} =
    _color_linear_value_blend(_color_linear_blend_style(typeof(α), C), α, yL, yR)

# Inference probe for gate condition 2 — never called at runtime, only fed to
# `promote_op` to ask "what type would the componentwise path produce?".
_cw_blend_result(α, yL, yR) = mapc((l, r) -> _linear_value_blend(α, l, r), yL, yR)

@inline _color_linear_blend_style(::Type{A}, ::Type{C}) where {A, C <: Colorant} =
    _color_style_tag(
    _linear_blend_style(A, eltype(C)),
    _same_type_tag(
        Base.promote_op(*, A, C),
        Base.promote_op(_cw_blend_result, A, C, C),
    ),
)
@inline _same_type_tag(::Type{W}, ::Type{W}) where {W} = Val(true)
@inline _same_type_tag(::Type, ::Type) = Val(false)
@inline _color_style_tag(::_LinearBlendFMA, ::Val{true}) = Val(:componentwise)
@inline _color_style_tag(::_LinearBlendStyle, ::Val) = Val(:generic)

# Componentwise: α preserved into each channel; `mapc` reassembles the widened
# arithmetic color type (RGB{N0f8} → RGB{Float64}, matching the generic path —
# gate condition 2 is exactly this guarantee).
@inline _color_linear_value_blend(::Val{:componentwise}, α, yL, yR) =
    mapc((l, r) -> _linear_value_blend(α, l, r), yL, yR)
# Generic: explicit escape to the core styled method (no recursion through
# the colorant entry above).
@inline _color_linear_value_blend(::Val{:generic}, α, yL, yR) =
    _linear_value_blend(_LinearBlendGeneric(), α, yL, yR)

end
