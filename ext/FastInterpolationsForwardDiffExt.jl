# ========================================
# ForwardDiff Extension for FastInterpolations.jl
# ========================================
# This extension enables automatic differentiation support
# when ForwardDiff.jl is loaded.
#
# Usage:
#   using FastInterpolations, ForwardDiff
#   itp = linear_interp(x, y; extrap=ExtendExtrap())
#   ForwardDiff.derivative(itp, 2.5)  # AD through interpolation

module FastInterpolationsForwardDiffExt

    using FastInterpolations
    using ForwardDiff: Dual, value

    import FastInterpolations: _extract_primal

    # ForwardDiff support: extract primal value from Dual for index search
    # - Use _extract_primal(xq) for comparisons and index lookup
    # - Use original xq for arithmetic (preserves AD derivatives)
    @inline _extract_primal(xq::Dual{T, V, N}) where {T, V, N} = _extract_primal(value(xq))

    # NOTE: the bespoke `_promote_coord(::Dual, Tg)` method was removed — the
    # canonical `_coord_eltype` rule (src/core/utils.jl) subsumes it: a Dual query on
    # a Float grid resolves to the Dual (concrete `promote_type`), and `convert` lifts
    # an Int-backed Dual to a Float-backed one, matching the old `xq + zero(Tg)`.

end # module
