# ========================================
# StaticArrays Extension for FastInterpolations.jl
# ========================================
#
# Enables Vector{SVector{N,T}} as query input for all ND APIs:
#   itp(output, queries)          — interpolant callable
#   cubic_interp(grids, data, queries)  — one-shot
#   cubic_adjoint(grids, queries)       — adjoint
#
# Only _query_extract needs overriding: converts SVector → NTuple on extraction.
# _query_length and _query_eltype work via built-in AbstractVector fallbacks.

module FastInterpolationsStaticArraysExt

    using FastInterpolations
    using StaticArrays: StaticVector

    import FastInterpolations: _query_extract

    @inline function _query_extract(
            q::AbstractVector{<:StaticVector{N, T}}, k, ::Val{N}
        ) where {N, T <: Real}
        return Tuple(@inbounds q[k])
    end

end # module
