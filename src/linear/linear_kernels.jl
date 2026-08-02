# ========================================
# Linear Interpolation Kernels
# ========================================
# Pure mathematical kernel functions for linear interpolation.
# No dependencies - can be tested independently.
#
# Unified API with two signatures:
# 1. Standard: _linear_kernel(op, yL, yR, inv_h, α)
#    - Caller computes α via grid-type-dispatched `_alpha_of(q, L, R, grid)`
#      (cached `*inv_h` for `_CachedRange`, direct `(q-L)/(R-L)` for plain
#      `AbstractVector`) and pulls `inv_h` separately. EvalValue uses α only;
#      EvalDeriv1 uses inv_h only — LLVM DCE removes the unused side per op.
#
# 2. Anchored: _linear_kernel(op, yL, yR, aq::_LinearAnchoredQuery)
#    - Defined in linear_anchor.jl (after anchor type)
#    - Uses precomputed alpha/inv_h from anchor
#
# TYPE PARAMETERS:
# - Tg: Grid type (AbstractFloat) - for inv_h (inverse interval width)
# - Tv: Value type - for yL, yR (can be Complex{Tg}, Tg, etc.)
# - α: Unconstrained — can be Tg or Dual{Tg} for AD support

# ========================================
# Convex value blend (EvalValue's final 2-value combine)
# ========================================
# α·yR + (1−α)·yL, style-dispatched on the weighted RESULT type
# `promote_op(*, α, value)` — fixed-point values (N0f8) under a float weight
# classify as native-FMA. The negation lands on the float weight `α`, never on
# data, so finite/colorant values appear only as `weight × value` (wrap-free).
# Endpoint-exact at α=0,1. For ordered real values the α∈[0,1] result stays
# within [min(yL,yR), max(yL,yR)]; composite carriers (complex, colorants)
# inherit that per real component. Extrapolation intentionally passes α
# outside [0,1].
#
#   _LinearBlendFMA     — result is IEEEFloat/Complex{IEEEFloat}: verbatim
#                         2-FMA form (free-addend fusion).
#   _LinearBlendGeneric — duck-safe factoring for everything else (colorants,
#                         Dual, BigFloat, undefined `*`): complement on the
#                         SCALAR weight, value touched once fewer per node.
#
# Componentwise colorant support lives ENTIRELY in the ColorVectorSpace
# extension, through this same style protocol: it adds a third style plus a
# `_linear_blend_style` method returning it for eligible colorant pairs, and
# the matching styled body maps each channel back into this blend (α
# preserved ⇒ channels take the 2-FMA form; the convex structure must not be
# flattened to a generic weight pair — that costs the channel fusion).
# Ineligible colorants classify as `_LinearBlendGeneric` — no safety net
# needed.
abstract type _LinearBlendStyle end
struct _LinearBlendFMA <: _LinearBlendStyle end
struct _LinearBlendGeneric <: _LinearBlendStyle end

@inline _linear_blend_style(::Type{A}, ::Type{Y}) where {A, Y} =
    _linear_blend_style(Base.promote_op(*, A, Y))
@inline _linear_blend_style(::Type{T}) where {T <: Base.IEEEFloat} = _LinearBlendFMA()
@inline _linear_blend_style(::Type{Complex{T}}) where {T <: Base.IEEEFloat} = _LinearBlendFMA()
@inline _linear_blend_style(::Type) = _LinearBlendGeneric()
@inline _linear_blend_style(::Type{Union{}}) = _LinearBlendGeneric()   # undefined `*` ⇒ safe generic (also disambiguates ⊥)

@inline function _linear_value_blend(α, yL, yR)
    style = _linear_blend_style(typeof(α), promote_type(typeof(yL), typeof(yR)))
    return _linear_value_blend(style, α, yL, yR)
end
@inline _linear_value_blend(::_LinearBlendFMA, α, yL, yR) =
    muladd(α, yR, muladd(-α, yL, yL))
@inline _linear_value_blend(::_LinearBlendGeneric, α, yL, yR) =
    muladd(α, yR, (one(α) - α) * yL)

# ========================================
# Standard Kernel (inv_h, α signature)
# ========================================

"""
    _linear_kernel(::EvalValue, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Evaluate linear interpolation value at normalized cell coordinate `α`.
Returns: α*yR + (1-α)*yL (convex form). `α` is typically in [0, 1] for in-domain
queries but may fall outside that range under linear extrapolation
(`ExtendExtrap`/`WrapExtrap`) where callers intentionally evaluate
outside the cell. `inv_h` is unused for value eval — DCE'd by LLVM
when the caller's `inv_h` extraction has no other live use.

The convex form `_linear_value_blend(α, yL, yR)` is wrap-free for non-field
eltypes (e.g. UInt8, N0f8): it never subtracts endpoints, so finite-range
values remain in-range. It is also endpoint-exact: α=0 → yL, α=1 → yR.
"""
@inline function _linear_kernel(::EvalValue, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    return _linear_value_blend(α, yL, yR)  # α*yR + (1-α)*yL — convex, wrap-free, endpoint-exact
end

"""
    _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Evaluate first derivative (slope) of linear interpolation: `_fielddiff(Tc, yR, yL) * inv_h` (wrap-safe).
The trailing `* one(α)` carries the query's carrier (Dual partials,
Measurement uncertainty, …) — for plain `Real` `α`, LLVM const-folds the
`1.0` factor away.
"""
@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, inv_h::Ti, α) where {Ti, Tv}
    # Value-space widen: the diff stays in Tv units; the 1/X dimension enters
    # via `* inv_h` (coeff-space Tc would convert y into slope units).
    Tw = _promote_eltype(_interp_op, Ti, Tv, Ti)
    return _fielddiff(Tw, yR, yL) * inv_h * one(α)
end

"""
    _linear_kernel(::EvalDeriv2, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Evaluate second derivative of linear interpolation. Zero for finite data
(linear function has no curvature); a NaN/Inf in either cell endpoint still
propagates (cell-local, matching the flat weight form).

Note: Mathematically, the second derivative is a Dirac delta at knots,
but we return zero everywhere as a practical approximation.
"""
@inline function _linear_kernel(::EvalDeriv2, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    # ×0 (no curvature), but touch BOTH endpoints so a NaN/Inf in either cell corner
    # survives the multiply (cell-local propagation). `0*yL + 0*yR` avoids the overflow
    # of `yL+yR`/`yL*yR` that would manufacture a spurious NaN from large finite data.
    # `oneunit(inv_h)²` carries the value/grid² units (unit grids); `1.0` on Real grids.
    return (0 * yL + 0 * yR) * oneunit(inv_h)^2 * one(α)
end

"""
    _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Third derivative of linear interpolation is always zero.
Linear functions have constant first derivative (slope), zero second and third derivatives.
"""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    # value/grid³ units via `oneunit(inv_h)³`; NaN-propagating `0*yL + 0*yR` — see EvalDeriv2.
    return (0 * yL + 0 * yR) * oneunit(inv_h)^3 * one(α)
end

"""Generic fallback: N-th derivative of degree-1 polynomial is zero for N ≥ 2."""
@inline function _linear_kernel(::DerivOp{N}, yL::Tv, yR::Tv, inv_h::Tg, α) where {N, Tg, Tv}
    # value/gridᴺ units; `literal_pow` keeps `oneunit(inv_h)ᴺ` type-stable for a type-param N.
    return (0 * yL + 0 * yR) * Base.literal_pow(^, oneunit(inv_h), Val(N)) * one(α)
end

# ========================================
# Normalized cell coordinate: α = (q - L) / h
# ========================================
# Shared by 1D (`linear_oneshot.jl`) and ND (`linear_nd_eval.jl`); grid-type dispatched:
#   - `inv_h` form: caller supplies the reciprocal (ND `_locate_cell`).
#   - `_CachedRange`: pull cached `inv_h` via the accessor — a `_UnitStep` grid returns
#     `one`, so LLVM folds the `×inv_h` away (α = q - L). No `_UnitStep` method needed.
#   - plain `AbstractVector`: divide by the on-the-fly `R - L` (EvalValue can then DCE
#     the separately-extracted `inv_h`, which only EvalDeriv1 kernels use).
# `_coord_value` unwraps a resolved `GridIdx` (identity otherwise): the search
# is already done, and on a unit axis the wrapper cannot ride promotion.
@inline _alpha_of(q, L, inv_h) = (_coord_value(q) - L) * inv_h
@inline _alpha_of(q, L, R, x::_CachedRange) = (_coord_value(q) - L) * _get_inv_h(x)
@inline _alpha_of(q, L, R, ::AbstractVector) = (_coord_value(q) - L) / float(R - L)
