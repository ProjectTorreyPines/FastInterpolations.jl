# ========================================
# CubicHermiteInterpolantND — One-shot Evaluation (Phase 1a)
# ========================================
#
# Phase 1a strategy: build the interpolant, evaluate, return. Allocates a
# `_NodalDerivativesND` buffer per call (1 × Array{Tv, N+1}), so it's not
# pool-grade zero-alloc. Cleaner code and correctness come first; an
# allocation-tight pool-based path is a Phase 2 follow-up (mirroring
# `_cubic_interp_nd_oneshot`).

# ========================================
# Scalar one-shot
# ========================================

"""
    hermite_interp(grids, data, partials, query::Tuple) -> value

ND cubic Hermite one-shot at a single query point. Equivalent to
`hermite_interp(grids, data, partials)(query)` but accepted as a single call.

Keywords: same as the persistent-build form (`bc`, `extrap`, `search`,
`deriv`).
"""
function hermite_interp(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N},
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    itp = CubicHermiteInterpolantND(grids, data, partials; bc, extrap, search)
    return itp(query; deriv, hint)
end

# ========================================
# Batch one-shot (allocating)
# ========================================

"""
    hermite_interp(grids, data, partials, queries) -> Vector

ND cubic Hermite one-shot at multiple query points. `queries` follows the
project's query protocol (`_query_length` / `_query_extract` / etc.). The
return type follows the standard ND output-eltype computation.
"""
function hermite_interp(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {Tv, N}
    _, Tg, _, _ = _nd_promote_grids(grids, data)
    Tq = _query_eltype(queries)
    Tr = _output_eltype(_arithmetic_kernel_shape, Tg, Tv, Tq)
    output = Vector{Tr}(undef, _query_length(queries))
    hermite_interp!(output, grids, data, partials, queries;
                    deriv, bc, extrap, search, hint)
    return output
end

# ========================================
# Batch one-shot (in-place)
# ========================================

"""
    hermite_interp!(output, grids, data, partials, queries) -> output

In-place batch ND cubic Hermite one-shot. `output` must have length
`_query_length(queries)`. Builds the interpolant once and dispatches to the
shared batch-evaluation pipeline.
"""
function hermite_interp!(
        output::AbstractVector,
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N},
        queries;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {N}
    itp = CubicHermiteInterpolantND(grids, data, partials; bc, extrap, search)
    # Delegate to the shared callable's in-place batch path
    # (`AbstractInterpolantND` callable handles all batch shapes).
    return itp(output, queries; deriv, hint)
end
