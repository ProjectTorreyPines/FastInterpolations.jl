# ========================================
# ND Quadratic Interpolation Public API
# ========================================
#
# Extends quadratic_interp() to support tuple grid inputs for ND.
# Follows the same pattern as cubic_nd_api.jl.

# ========================================
# BC Resolution for Quadratic ND
# ========================================

"""
    _resolve_bcs_nd_quadratic(bc, Val(N)) -> NTuple{N, QuadraticBC}

Resolve boundary condition input to canonical N-tuple of QuadraticBC.

Converts common BC types to their quadratic equivalents:
- `QuadraticBC` (Left, Right, MinCurvFit): pass through
- `NaturalBC()`: → Right(Deriv2(0)) (zero curvature at right endpoint)
- `CubicFit()`: → Right(CubicFit())
- `QuadraticFit()`: → Right(QuadraticFit())
- `LinearFit()`: → Right(LinearFit())
"""
@inline function _resolve_bcs_nd_quadratic(bc::QuadraticBC, ::Val{N}) where {N}
    ntuple(_ -> bc, Val(N))
end

@inline function _resolve_bcs_nd_quadratic(bc::NTuple{N, QuadraticBC}, ::Val{N}) where {N}
    bc
end

# Convert common non-quadratic BCs to quadratic equivalents
@inline function _resolve_bcs_nd_quadratic(bc::NaturalBC, ::Val{N}) where {N}
    ntuple(_ -> Right(Deriv2(0.0)), Val(N))
end

@inline function _resolve_bcs_nd_quadratic(bc::PolyFit{D}, ::Val{N}) where {D, N}
    ntuple(_ -> Right(bc), Val(N))
end

# Handle heterogeneous BC tuple: convert each element
@inline function _resolve_bcs_nd_quadratic(bc::NTuple{N, AbstractBC}, ::Val{N}) where {N}
    ntuple(Val(N)) do d
        _to_quadratic_bc(bc[d])
    end
end

# Single non-quadratic BC: broadcast after conversion
@inline function _resolve_bcs_nd_quadratic(bc::AbstractBC, ::Val{N}) where {N}
    qbc = _to_quadratic_bc(bc)
    ntuple(_ -> qbc, Val(N))
end

"""
    _to_quadratic_bc(bc) -> QuadraticBC

Convert an AbstractBC to its QuadraticBC equivalent.
"""
@inline _to_quadratic_bc(bc::QuadraticBC) = bc
@inline _to_quadratic_bc(::NaturalBC) = Right(Deriv2(0.0))
@inline _to_quadratic_bc(bc::PolyFit) = Right(bc)
@inline _to_quadratic_bc(bc::AbstractBC) = throw(ArgumentError(
    "Unsupported BC for quadratic ND: $(typeof(bc)). " *
    "Use Left(...), Right(...), MinCurvFit, NaturalBC(), or PolyFit variants."
))

# ========================================
# GENERIC ND: N-ARGUMENT FORM
# ========================================

"""
    quadratic_interp(grids::NTuple{N,AbstractVector}, data::AbstractArray{<:Any,N}; kwargs...)

Create an N-dimensional quadratic interpolant from grid vectors and data array.

# Arguments
- `grids::NTuple{N,AbstractVector}`: Tuple of grid vectors for each dimension
- `data::AbstractArray{<:Any,N}`: Function values at grid points

# Keywords
- `bc=Left(QuadraticFit())`: Boundary condition(s). Can be:
  - Single BC: Applied to all axes
  - `NTuple{N}`: Per-axis BCs
- `extrap=:none`: Extrapolation mode(s)
- `search=Binary()`: Search policy(s)

# Returns
- `QuadraticInterpolantND{Tg, Tv, N, ...}`: Callable interpolant object

# Examples
```julia
x = range(0.0, 2.0, 20)
y = range(0.0, 1.0, 15)
data = [xi^2 + yi^2 for xi in x, yi in y]
itp = quadratic_interp((x, y), data)
itp((1.0, 0.5))  # Evaluate
itp((1.0, 0.5); deriv=(1, 0))  # ∂f/∂x
```
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N};
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {N, Tv_raw}
    # Zero-allocation type promotion
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64

    # Zero-allocation grid conversion
    grids_typed = _convert_grids_typed(grids, Tg)

    # Get value type
    Tv = eltype(data)

    # Validate dimensions
    _validate_nd_grids(grids_typed, data)

    # Resolve per-axis options
    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Build interpolant
    return _build_nd_quadratic_interpolant(grids_typed, data, bcs, extraps, searches)
end

# ========================================
# ZERO-ALLOC ONE-SHOT IMPLEMENTATION
# ========================================
#
# Pool-based evaluation functions that bypass Interpolant construction.
# Nodal derivatives are computed in pool buffers (reused across calls).
# Zero heap allocation for scalar queries after warmup.

"""
    _quadratic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops)

Pool-based scalar one-shot ND quadratic evaluation.
Computes 2^N partial derivatives in a pool buffer and evaluates at a single point.
Zero-allocation after warmup (pool reuse).
"""
@with_pool pool function _quadratic_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    # 1. Pool-allocate partials array (THE KEY: pool instead of heap)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data)...))

    # 2. Compute all partial derivatives in-place
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)

    # 3. Create spacings (ScalarSpacing for Range grids = zero alloc)
    spacings = _create_spacings_typed(grids)

    # 4. Eval pipeline (all standalone functions, no Interpolant needed)
    q_eval = _handle_all_extraps(query, grids, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
    hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)

    # 5. Tensor-product kernel evaluation
    return _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _quadratic_interp_nd_oneshot_soa(grids, data, queries, bcs, extraps_val, searches, ops)

Pool-based SoA batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points.
Output vector is heap-allocated (return value).
"""
@with_pool pool function _quadratic_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end

    # Build phase (done once)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)
    spacings = _create_spacings_typed(grids)

    # Allocate output (heap — this IS the return value)
    output = Vector{Tv}(undef, n_queries)

    # Eval loop
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

"""
    _quadratic_interp_nd_oneshot_aos(grids, data, queries, bcs, extraps_val, searches, ops)

Pool-based AoS batch one-shot ND quadratic evaluation.
Computes partials ONCE, then evaluates at all query points.
Output vector is heap-allocated (return value).
"""
@with_pool pool function _quadratic_interp_nd_oneshot_aos(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    n_queries = length(queries)

    # Build phase (done once)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data)...))
    _compute_nd_partials_quadratic!(partials, grids, data, bcs)
    spacings = _create_spacings_typed(grids)

    # Allocate output (heap — this IS the return value)
    output = Vector{Tv}(undef, n_queries)

    # Eval loop
    @inbounds for k in 1:n_queries
        query_k = queries[k]
        q_eval = _handle_all_extraps(query_k, grids, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_eval, grids, spacings, searches)
        hs, inv_hs, dLs = _compute_all_local_params(q_eval, spacings, indices, Ls)
        output[k] = _eval_nd_quad_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

# ========================================
# One-Shot API
# ========================================

"""
    quadratic_interp(grids, data, query; deriv=0, kwargs...)

One-shot ND quadratic interpolation at a single point.
Zero-allocation after warmup.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot(
                    grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot(
                grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot(
                grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        end
    end
end

"""
    quadratic_interp(grids, data, queries::NTuple{N,AbstractVector}; deriv=0, kwargs...)

One-shot ND quadratic interpolation at multiple points (batch SoA).
Only allocates the output vector.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot_soa(
                    grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot_soa(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot_soa(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

"""
    quadratic_interp(grids, data, queries::AbstractVector{<:NTuple}; deriv=0, kwargs...)

One-shot ND quadratic interpolation at multiple points (batch AoS).
Only allocates the output vector.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=Left(QuadraticFit()),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd_quadratic(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _quadratic_interp_nd_oneshot_aos(
                    grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _quadratic_interp_nd_oneshot_aos(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _quadratic_interp_nd_oneshot_aos(
                grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

# ========================================
# INTERNAL BUILDER
# ========================================

function _build_nd_quadratic_interpolant(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, QuadraticBC},
    extraps::NTuple{N, Symbol},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    # Build nodal derivatives using quadratic recurrence
    nodal_derivs = _build_nd_coeffs_quadratic(grids, data, bcs)

    # Create spacings (uses @generated to avoid closure boxing)
    spacings = _create_spacings_typed(grids)

    # Store BCs as-is (already QuadraticBC)
    bcs_store = bcs

    # Convert extrap symbols to Val types
    extraps_val = ntuple(Val(N)) do d
        _symbol_to_extrap_val(extraps[d])
    end

    # Construct the interpolant
    NP1 = N + 1
    return QuadraticInterpolantND{
        Tg, Tv, N, NP1,
        typeof(grids), typeof(spacings), typeof(bcs_store),
        typeof(extraps_val), typeof(searches)
    }(grids, spacings, nodal_derivs, bcs_store, extraps_val, searches)
end
