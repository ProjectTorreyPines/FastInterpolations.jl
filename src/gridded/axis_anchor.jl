# ============================================================================
# Op-aware axis anchors — the gridded-query per-axis primitive
# ============================================================================
#
# An axis anchor is a RETAINED (grid × query-coordinate) resolution artifact:
# the interval index plus the minimal data-free live set of ONE (method, op)
# kernel, with extrapolation already folded in. It is the materialized twin of
# the fused scalar path's register-level DCE: the fused `_linear_kernel(op,
# yL, yR, inv_h, α)` passes everything and lets LLVM drop the unused argument;
# across a memory boundary that cannot happen, so the `Op` type parameter
# selects a NamedTuple payload holding exactly what that op's kernel reads
# (EvalValue → (alpha,); a future EvalDeriv1 tier → (inv_h,)).
#
# Matched-pair contract: each (method, op) defines a builder (`_axis_anchor`)
# + kernel (`_eval_anchor`) pair; the payload names are their private
# agreement. Upper levels read only `a.idx` (tap address) and call the kernel.

"""
    _LinearAnchor{T, Op, P}

Linear-method axis anchor: interval index + the `Op`-selected payload.
`T` is the payload scalar type, `Op <: AbstractEvalOp` selects the payload
contract, `P <: NamedTuple` is its storage. Build ONLY through
[`_axis_anchor`](@ref) — the builders are the (Op ↔ payload-shape)
invariant's enforcement point.
"""
struct _LinearAnchor{T, Op <: AbstractEvalOp, P <: NamedTuple} <: _AbstractAxisAnchor
    idx::Int      # left node index (right node = idx + 1); public field
    payload::P    # op-selected named fields; private to the builder+kernel pair
end
@inline _LinearAnchor{T, Op}(idx::Int, payload::P) where {T, Op, P <: NamedTuple} =
    _LinearAnchor{T, Op, P}(idx, payload)

# Val-dispatch virtual properties (house convention — see linear_anchor.jl:
# a `===` branch chain's union return is not reliably inlined in hot loops;
# Val(s) on a literal symbol is a compile-time constant → static dispatch).
@inline Base.getproperty(a::_LinearAnchor, s::Symbol) = _get_axis_anchor_prop(a, Val(s))
@inline _get_axis_anchor_prop(a::_LinearAnchor, ::Val{:idx}) = getfield(a, :idx)
@inline _get_axis_anchor_prop(a::_LinearAnchor, ::Val{s}) where {s} =
    getfield(getfield(a, :payload), s)
@inline Base.propertynames(a::_LinearAnchor) =
    (:idx, propertynames(getfield(a, :payload))...)

# ---- extrap folding (weight-level; ClampExtrap freezes OOB at the boundary
# node — `_anchor_loc`'s boundary interval + clamped alpha reproduces it) ----
@inline _resolve_alpha(α, ::ClampExtrap) = clamp(α, zero(α), one(α))
@inline _resolve_alpha(α, ::AbstractExtrap) = α

# ---- matched kernel: this pair knows payload ≡ (alpha,) --------------------
@inline _eval_anchor(a::_LinearAnchor{T, EvalValue}, yL, yR) where {T} =
    _linear_value_blend(a.alpha, yL, yR)

"""
    _axis_anchor(::LinearInterp, ::EvalValue, loc, g, ex, ::Type{Tα})

Scalar builder for the Linear × EvalValue tier: geometry from an
[`_AnchorLoc`](@ref), converted to `Tα`, extrap folded. The builder is the
(method, op) → concrete-anchor-type mapping; callers never spell the type.
"""
@inline function _axis_anchor(
        ::LinearInterp,
        ::EvalValue,
        loc::_AnchorLoc,
        g::AbstractVector,
        ex::AbstractExtrap,
        ::Type{Tα}
    ) where {Tα}
    α = Tα(_alpha_of(loc.xq, loc.xL, loc.xR, g))
    return _LinearAnchor{Tα, EvalValue}(loc.idx, (alpha = _resolve_alpha(α, ex),))
end
