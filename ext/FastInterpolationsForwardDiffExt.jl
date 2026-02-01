# ========================================
# ForwardDiff Extension for FastInterpolations.jl
# ========================================
# This extension enables automatic differentiation support
# when ForwardDiff.jl is loaded.
#
# Usage:
#   using FastInterpolations, ForwardDiff
#   itp = linear_interp(x, y; extrap=:extension)
#   ForwardDiff.derivative(itp, 2.5)  # AD through interpolation

module FastInterpolationsForwardDiffExt

using FastInterpolations
using ForwardDiff: Dual, value

import FastInterpolations: _extract_primal, _promote_for_anchor

# ForwardDiff support: extract primal value from Dual for index search
# - Use _extract_primal(xq) for comparisons and index lookup
# - Use original xq for arithmetic (preserves AD derivatives)
@inline _extract_primal(xq::Dual{T,V,N}) where {T,V,N} = value(xq)

# Anchor promotion: preserve Dual type (don't convert to grid type)
# This enables AD through anchor-based series evaluation
@inline _promote_for_anchor(xq::Dual{T,V,N}, ::Type{Tg}) where {T,V,N,Tg<:AbstractFloat} = xq

end # module
