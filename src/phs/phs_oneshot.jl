# ========================================
# PHSInterpolantND — One-Shot API
# ========================================
#
# One-shot forms: build a temporary PHSInterpolantND then evaluate.
# phi_inv must always be precomputed (cannot be bypassed for a single query),
# so these share the same construction cost as the persistent interpolant API.
# Use phs_interp(grids, data) when evaluating at many points.
# Keywords other than `deriv` go to the persistent constructor. Only the blend default
# differs: 2.0 here, 1.0 in the persistent constructor.

"""
    phs_interp(grids, data, query::NTuple{N,Real}; kwargs...) -> scalar

One-shot N-dimensional PHS interpolation at a single query point.

See `phs_interp(grids, data)` for keyword argument documentation.
"""
function phs_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}};
        blend_factor::Real = 2.0,
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kwargs...,
    ) where {Tv, N}
    return phs_interp(grids, data; blend_factor, kwargs...)(query; deriv)
end

"""
    phs_interp(grids, data, queries; kwargs...) -> Vector

One-shot N-dimensional PHS interpolation at a batch of query points.
`queries` is any query-protocol-compatible container (SoA tuple, AoS vector, etc.).

Builds a temporary interpolant (same construction cost as `phs_interp(grids, data)`),
then allocates and fills the output vector.
"""
function phs_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        blend_factor::Real = 2.0,
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kwargs...,
    ) where {Tv, N}
    return phs_interp(grids, data; blend_factor, kwargs...)(queries; deriv)
end

"""
    phs_interp!(out, grids, data, queries; kwargs...)

In-place one-shot N-dimensional PHS interpolation.
Writes results into pre-allocated `out`.
"""
function phs_interp!(
        out::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        blend_factor::Real = 2.0,
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kwargs...,
    ) where {Tv, N}
    return phs_interp(grids, data; blend_factor, kwargs...)(out, queries; deriv)
end
