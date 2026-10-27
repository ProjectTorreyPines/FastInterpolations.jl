# ========================================
# Unitful Extension for FastInterpolations.jl
# ========================================
# Unit-carrying grids and queries are duck-typed and need no extension. This one
# only lets `_extract_primal` see through a `Quantity` wrapping an AD carrier, e.g.
# the `Quantity{<:Dual}` of `ForwardDiff.derivative(t -> itp(t * u"s"), t0)`, so
# domain checks compare the primal: ForwardDiff orders Duals by their partials on a
# value tie, which would read an endpoint query seeded outward as out of domain.

module FastInterpolationsUnitfulExt

using FastInterpolations
using Unitful: AbstractQuantity, ustrip, unit

import FastInterpolations: _extract_primal

# Strip the number inside (recursing through nested Duals) and re-attach the unit.
@inline _extract_primal(q::AbstractQuantity) = _rewrap_primal(q, _extract_primal(ustrip(q)))

# Number already primal (plain `Quantity{Float64}`, …): return `q` itself, not a rebuilt copy,
# so plain unit paths compile to the same native code as without this extension.
@inline _rewrap_primal(q::AbstractQuantity{T}, ::T) where {T} = q
@inline _rewrap_primal(q::AbstractQuantity, p) = p * unit(q)

end # module
