# Componentwise convex blend for arithmetic colorants.
#
# The core blend protocol (src/linear/linear_kernels.jl) is
#     entry → _linear_blend_style(weight, value) → styled body
# with two core styles (FMA / Generic). This extension plugs into that
# protocol — no entry override: it adds ONE style, its rule method, and its
# styled body.
#
#   componentwise — `mapc` re-enters the styled blend per channel with α
#                   PRESERVED, so each channel takes the verbatim 2-FMA float
#                   form. (Flattening α to a generic weight pair instead costs
#                   the channel fusion — ~half the win, and 4-channel N0f8
#                   turns into a small loss.)
#
# The rule returns componentwise only when BOTH hold (a plain `if`: every
# operand is type-computable, so it constant-folds and the dead arm vanishes —
# pinned by the `@inferred` tests and the A/B benchmark):
#   1. the CHANNEL blend is native-FMA — the core rule on
#      `(weight, eltype(C))`; a Dual/BigFloat weight or channel stays generic.
#   2. the componentwise reconstruction type ≡ the natural whole-color
#      arithmetic type: `promote_op(mapc-blend, A, C, C) === promote_op(*, A, C)`.
#      This is what excludes PACKED colorants (Gray24/RGB24/ARGB32): CVS
#      arithmetic widens them (`0.3*Gray24 → Gray{Float64}`) but `mapc` can
#      only rebuild the packed type itself (no `Gray24{Float64}` exists) —
#      re-quantizing values, changing the result type, and throwing on
#      out-of-[0,1] extrapolants. Type mismatch ⇒ `_LinearBlendGeneric`,
#      which preserves today's widened output exactly.
#
# The styled body requires a same-type pair. A MIXED pair can still classify
# componentwise (the core entry styles on `promote_type` of the two values),
# so an escape method forwards it to the core generic body — plain
# ColorVectorSpace arithmetic with its own internal promotion, unchanged.
#
# Rule scope `_ArithColorant` = the four ColorTypes-public abstractions that
# ColorVectorSpace defines vector-space arithmetic for (≡ CVS's internal
# `MathTypes`, spelled without depending on it). Non-RGB color spaces (HSV,
# Lab, …) are outside — CVS deliberately defines no arithmetic there (hue is
# angular), so they classify through the core rule and fail with the same
# MethodError as today instead of silently inventing channelwise semantics.
#
# ColorVectorSpace is a trigger (not just ColorTypes) so this extension is a
# pure perf overlay on arithmetic that already exists — it must never make
# color interpolation newly work without ColorVectorSpace loaded.
#
# `@inline` is load-bearing on the styled body: without it the 4-channel
# N0f8 body exceeds the inline cost threshold and a per-node call costs ~55%.
module FastInterpolationsColorVectorSpaceExt

using ColorTypes:
    AbstractGray, AbstractRGB, TransparentGray, TransparentRGB, mapc
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

struct _LinearBlendComponentwise <: _LinearBlendStyle end

# Inference probe for rule condition 2 — never called at runtime, only fed to
# `promote_op` to ask "what type would the componentwise path produce?".
_cw_blend_result(α, yL, yR) = mapc((l, r) -> _linear_value_blend(α, l, r), yL, yR)

@inline function _linear_blend_style(::Type{A}, ::Type{C}) where {A, C <: _ArithColorant}
    if _linear_blend_style(A, eltype(C)) isa _LinearBlendFMA &&
            Base.promote_op(_cw_blend_result, A, C, C) === Base.promote_op(*, A, C)
        return _LinearBlendComponentwise()
    else
        return _LinearBlendGeneric()
    end
end

# α preserved into each channel; `mapc` reassembles the widened arithmetic
# color type (RGB{N0f8} → RGB{Float64}) — rule condition 2 is exactly the
# guarantee that this matches the generic path's output type.
@inline _linear_value_blend(::_LinearBlendComponentwise, α, yL::C, yR::C) where {C} =
    mapc((l, r) -> _linear_value_blend(α, l, r), yL, yR)
# Mixed-pair escape → core generic body (no recursion).
@inline _linear_value_blend(::_LinearBlendComponentwise, α, yL, yR) =
    _linear_value_blend(_LinearBlendGeneric(), α, yL, yR)

end
