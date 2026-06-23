# ───────────────────────────────────────────────────────────────────────────────
# Per-axis tensor-product OUTPUT-eltype fold for heterogeneous ND interpolation.
#
# Each hetero axis applies its own 1D kernel; `_collapse_dims` threads the
# intermediate value type through the axis chain. `_hetero_output_eltype` predicts
# that final (widest) type at compile time via `Base.promote_op` of each axis's
# kernel-shape witness — exactly the per-method op-awareness the homogeneous `To`
# rule already uses, lifted to a mixed-method tuple.
#
#   - dividing methods (Linear/Cubic/Quadratic/Hermite): `_interp_op` → float Int.
#   - Constant: `_select_op` (`yv * one(xq - xL)`) → keep Int.
#   - NoInterp: a lookup, contributes no widening → `_passthrough_op`.
#
# The fold is pure type-level (`Base.promote_op` over concrete method types); it
# constant-folds during inference, so there is zero runtime cost — the same
# mechanism as `_promote_query_eltype`.
# ───────────────────────────────────────────────────────────────────────────────

@inline _passthrough_op(h::Tg, yv::Tv, q::Tq) where {Tg, Tv, Tq} = yv

@inline _axis_output_op(::ConstantInterp) = _select_op
@inline _axis_output_op(::NoInterp) = _passthrough_op
@inline _axis_output_op(::AbstractInterpMethod) = _interp_op

# Base case: no more axes — the threaded type is the answer.
@inline _hetero_output_eltype(::Tuple{}, ::Type{Tg}, ::Type{T}, ::Tuple{}) where {Tg, T} = T

# Recursive case: apply axis-1's witness to the threaded value type, recurse on the tail.
@inline function _hetero_output_eltype(
        methods::Tuple{AbstractInterpMethod, Vararg{AbstractInterpMethod}},
        ::Type{Tg}, ::Type{T},
        q::Tuple{Real, Vararg{Real}},
    ) where {Tg, T}
    op = _axis_output_op(first(methods))
    Tn = Base.promote_op(op, Tg, T, typeof(first(q)))
    Tnext = (Tn === Union{} || Tn === Any) ? T : Tn      # duck / inference fallback
    return _hetero_output_eltype(Base.tail(methods), Tg, Tnext, Base.tail(q))
end
