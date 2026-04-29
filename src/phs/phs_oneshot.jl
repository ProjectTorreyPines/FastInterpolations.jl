# ========================================
# PHSInterpolantND — One-Shot API
# ========================================
#
# One-shot forms: build a temporary PHSInterpolantND then evaluate.
# phi_inv must always be precomputed (cannot be bypassed for a single query),
# so these share the same construction cost as the persistent interpolant API.
# Use phs_interp(grids, data) when evaluating at many points.

"""
    phs_interp(grids, data, query::NTuple{N,Real}; kwargs...) -> scalar

One-shot N-dimensional PHS interpolation at a single query point.

See `phs_interp(grids, data)` for keyword argument documentation.
"""
function phs_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}};
        stencil_size::Int = 8,
        degree::Int = 3,
        blend_factor::Real = 2.0,
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        reference_interp = nothing,
    ) where {Tv, N}
    itp = phs_interp(grids, data; stencil_size, degree, blend_factor, extrap, search, reference_interp)
    return itp(query; deriv)
end

"""
    phs_interp(grids, data, queries; kwargs...) -> Vector

One-shot N-dimensional PHS interpolation at a batch of query points.
`queries` is any query-protocol-compatible container (SoA tuple, AoS vector, etc.).

Only allocates the output vector; all workspace is pool-allocated.
"""
function phs_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        stencil_size::Int = 8,
        degree::Int = 3,
        blend_factor::Real = 2.0,
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        reference_interp = nothing,
    ) where {Tv, N}
    itp = phs_interp(grids, data; stencil_size, degree, blend_factor, extrap, search, reference_interp)
    return itp(queries; deriv)
end

"""
    phs_interp!(out, grids, data, queries; kwargs...)

In-place one-shot N-dimensional PHS interpolation.
Writes results into pre-allocated `out`. Parallelized via `Threads.@threads`.
"""
function phs_interp!(
        out::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        stencil_size::Int = 8,
        degree::Int = 3,
        blend_factor::Real = 2.0,
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        reference_interp = nothing,
    ) where {Tv, N}
    itp = phs_interp(grids, data; stencil_size, degree, blend_factor, extrap, search, reference_interp)
    return itp(out, queries; deriv)
end
