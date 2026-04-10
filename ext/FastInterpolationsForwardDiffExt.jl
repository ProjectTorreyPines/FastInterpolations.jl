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

    import FastInterpolations: _extract_primal, _promote_for_anchor, _to_grid_type

    # ForwardDiff support: extract primal value from Dual for index search
    # - Use _extract_primal(xq) for comparisons and index lookup
    # - Use original xq for arithmetic (preserves AD derivatives)
    @inline _extract_primal(xq::Dual{T, V, N}) where {T, V, N} = _extract_primal(value(xq))

    # Anchor promotion: promote to float-backed Dual for weight computation
    # - Float-backed Dual (V<:AbstractFloat): xq * one(Tg) ≈ identity (compiler optimizes)
    # - Int-backed Dual: xq * one(Tg) promotes to float-backed (required for weight arithmetic)
    # This enables AD through anchor-based series evaluation with proper type consistency
    @inline _promote_for_anchor(xq::Dual{T, V, N}, ::Type{Tg}) where {T, V, N, Tg <: AbstractFloat} = xq + zero(Tg)

    # ── Grid-side Dual support ─────────────────────────────────────────────
    #
    # When the grid is Vector{Dual}, binary search calls `_to_grid_type(xq, Tg)`
    # with `Tg <: Dual`. The package-level fallback at `src/core/utils.jl:309`
    # does `Tg(_extract_primal(xq))` which becomes `Dual(Float64)` — a MethodError
    # because abstract `Dual` has no constructor from a single float.
    #
    # Fix: for a Dual-typed grid, return the primal of xq (a plain float). The
    # subsequent `x[mid] <= xq` comparison inside `_search_binary` then compares
    # `Dual <= Float`, which ForwardDiff forwards to the primal of `x[mid]` —
    # correct and cheap. Arithmetic later (xq - xL, dL * inv_h) uses the original
    # grid scalars and therefore carries the grid-side Dual partials through.
    @inline _to_grid_type(xq::Real, ::Type{<:Dual}) = _extract_primal(xq)

end # module
0
