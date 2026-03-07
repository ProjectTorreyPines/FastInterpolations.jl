# ========================================
# Enzyme Extension for FastInterpolations.jl
# ========================================
# Imports ChainRulesCore rrules into Enzyme's rule system via @import_rrule.
# Loaded automatically when both Enzyme and ChainRulesCore are available.

module FastInterpolationsEnzymeExt

using FastInterpolations
using Enzyme
using ChainRulesCore

# Import the cubic_interp rrule (∂/∂f) for Enzyme reverse mode.
# Vector query: cubic_interp(x::AbstractVector, f::AbstractVector, xq::AbstractVector; ...)
Enzyme.@import_rrule typeof(cubic_interp) AbstractVector{<:AbstractFloat} AbstractVector AbstractVector{<:AbstractFloat}

# Scalar query: cubic_interp(x::AbstractVector, f::AbstractVector, xq::Real; ...)
Enzyme.@import_rrule typeof(cubic_interp) AbstractVector{<:AbstractFloat} AbstractVector Real

end # module
