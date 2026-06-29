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
@inline function _linear_kernel(::EvalDeriv1, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    Tc = _promote_eltype(_coeff_op, Tg, Tv)
    return _fielddiff(Tc, yR, yL) * inv_h * one(α)
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
    return (0 * yL + 0 * yR) * one(α)
end

"""
    _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}

Third derivative of linear interpolation is always zero.
Linear functions have constant first derivative (slope), zero second and third derivatives.
"""
@inline function _linear_kernel(::EvalDeriv3, yL::Tv, yR::Tv, inv_h::Tg, α) where {Tg, Tv}
    return (0 * yL + 0 * yR) * one(α)  # touch both endpoints for cell-local NaN — see EvalDeriv2
end

"""Generic fallback: N-th derivative of degree-1 polynomial is zero for N ≥ 2."""
@inline function _linear_kernel(::DerivOp{N}, yL::Tv, yR::Tv, ::Tg, α) where {N, Tg, Tv}
    return (0 * yL + 0 * yR) * one(α)  # touch both endpoints for cell-local NaN — see EvalDeriv2
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
@inline _alpha_of(q::Real, L::Real, inv_h::Real) = (q - L) * inv_h
@inline _alpha_of(q::Real, L::Real, R::Real, x::_CachedRange) = (q - L) * _get_inv_h(x)
@inline _alpha_of(q::Real, L::Real, R::Real, ::AbstractVector) = (q - L) / float(R - L)
