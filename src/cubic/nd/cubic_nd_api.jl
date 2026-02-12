# ========================================
# ND Cubic Interpolation Public API
# ========================================
#
# Unified public API for N-dimensional cubic interpolation.
# Extends cubic_interp() to support tuple grid inputs.

# ========================================
# HELPER FUNCTIONS
# ========================================

"""
    _convert_grid(x, Tg) -> AbstractVector{Tg}

Convert grid to target float type, preserving Range type where possible.
"""
function _convert_grid(x::AbstractRange, ::Type{Tg}) where {Tg}
    if eltype(x) === Tg
        return x
    else
        return range(Tg(first(x)), Tg(last(x)), length(x))
    end
end

function _convert_grid(x::AbstractVector, ::Type{Tg}) where {Tg}
    if eltype(x) === Tg
        return x
    else
        return Tg.(x)
    end
end

# ========================================
# GENERIC ND: N-ARGUMENT FORM
# ========================================

"""
    cubic_interp(grids::NTuple{N,AbstractVector}, data::AbstractArray{<:Any,N}; kwargs...)

Create an N-dimensional cubic Hermite interpolant from grid vectors and data array.

# Arguments
- `grids::NTuple{N,AbstractVector}`: Tuple of grid vectors for each dimension
- `data::AbstractArray{<:Any,N}`: Function values at grid points

# Keywords
- `bc=NaturalBC()`: Boundary condition(s). Can be:
  - Single `AbstractBC`: Applied to all axes
  - `NTuple{N,AbstractBC}`: Per-axis BCs
- `extrap=:none`: Extrapolation mode(s). Can be:
  - Single `Symbol`: Applied to all axes (`:none`, `:constant`, `:wrap`)
  - `NTuple{N,Symbol}`: Per-axis modes
- `search=Binary()`: Search policy(s). Can be:
  - Single `AbstractSearchPolicy`: Applied to all axes
  - `NTuple{N,AbstractSearchPolicy}`: Per-axis policies
- `coeffs=PreCompute()`: Coefficient computation strategy

# Returns
- `CubicInterpolantND{Tg, Tv, N, ...}`: Callable interpolant object

# Type Inference
- Grid type `Tg`: Promoted from all grid element types (always AbstractFloat)
- Value type `Tv`: Element type of data (can be real, complex, or AD types)

# Examples
```julia
# 3D interpolation
x = range(0.0, 2π, 20)
y = range(0.0, π, 15)
z = range(0.0, 1.0, 10)
data = [sin(xi) * cos(yj) * zk for xi in x, yj in y, zk in z]
itp = cubic_interp((x, y, z), data)
itp((1.0, 0.5, 0.3))  # Evaluate at (1.0, 0.5, 0.3)

# With per-axis options
itp = cubic_interp((x, y, z), data;
    bc=(NaturalBC(), PeriodicBC(), NaturalBC()),
    extrap=(:none, :wrap, :constant))

# Complex-valued data
data_c = [sin(xi) * cos(yj) * zk + im * cos(xi) for xi in x, yj in y, zk in z]
itp_c = cubic_interp((x, y, z), data_c)
```
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N};
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=NaturalBC(),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary(),
    coeffs::AbstractCoeffStrategy=PreCompute()
) where {N, Tv_raw}
    # Zero-allocation type promotion (uses @generated function)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64  # Ensure AbstractFloat
    
    # Zero-allocation grid conversion (uses @generated function)
    grids_typed = _convert_grids_typed(grids, Tg)

    # Get value type
    Tv = eltype(data)

    # Validate dimensions
    _validate_nd_grids(grids_typed, data)

    # Resolve per-axis options
    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Dispatch on strategy
    return _build_nd_interpolant(grids_typed, data, bcs, extraps, searches, coeffs)
end

"""
    cubic_interp(grids, data, query; deriv=0, kwargs...)

One-shot ND cubic interpolation at a single point.
Zero-allocation after warmup: uses pool-based partials instead of constructing an Interpolant.

# Keywords
- `deriv`: `Int` (0-3) or `Val((d1,d2,...))` for mixed partials
- `bc`, `extrap`, `search`, `coeffs`: Same as the Interpolant constructor form
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=NaturalBC(),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary(),
    coeffs::AbstractCoeffStrategy=PreCompute()
) where {Tv, N}
    # Type promotion + validation (same as constructor path)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Validate periodic+extrap compatibility (once, before dispatch)
    _check_periodic_extrap(bcs, extraps, Val(N))

    # Dispatch extrap → concrete Val tuple, then deriv → concrete ops
    # Both dispatches happen BEFORE entering @with_pool for type stability
    # Type assertion (::Tv) prevents boxing from multi-branch return type inference failure
    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _cubic_interp_nd_oneshot(grids_typed, data, query, bcs, extraps_val, searches, ops)::Tv
        end
    end
end

"""
    cubic_interp(grids, data, queries::NTuple{N,AbstractVector}; deriv=0, kwargs...)

One-shot ND cubic interpolation at multiple points (SoA batch).
Zero-allocation for workspace after warmup; output vector is heap-allocated.
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    bc::Union{AbstractBC, NTuple{N,AbstractBC}}=NaturalBC(),
    extrap::Union{Symbol, NTuple{N,Symbol}}=:none,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=Binary(),
    coeffs::AbstractCoeffStrategy=PreCompute()
) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    _validate_nd_grids(grids_typed, data)

    bcs = _resolve_bcs_nd(bc, Val(N))
    extraps = _resolve_extrap_nd(extrap, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    _check_periodic_extrap(bcs, extraps, Val(N))

    @_dispatch_extrap_nd extraps bcs => extraps_val begin
        if deriv isa Int
            @_dispatch_deriv deriv => op begin
                ops = ntuple(_ -> op, Val(N))
                return _cubic_interp_nd_oneshot_soa(grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
            end
        elseif deriv isa Val
            ops = _resolve_deriv_nd(deriv, Val(N))
            return _cubic_interp_nd_oneshot_soa(grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        else
            ops = _resolve_deriv_nd(Val(deriv), Val(N))
            return _cubic_interp_nd_oneshot_soa(grids_typed, data, queries, bcs, extraps_val, searches, ops)::Vector{Tv}
        end
    end
end

# ========================================
# GENERIC ND INTERNAL BUILDERS
# ========================================

"""
    _build_nd_interpolant(grids, data, bcs, extraps, searches, ::PreCompute)

Build CubicInterpolantND with precomputed coefficients.
"""
function _build_nd_interpolant(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, AbstractBC},
    extraps::NTuple{N, Symbol},
    searches::NTuple{N, AbstractSearchPolicy},
    ::PreCompute
) where {Tg<:AbstractFloat, Tv, N}
    # Validate periodic BC + extrap compatibility (Val recursion to avoid hetero tuple boxing)
    _check_periodic_extrap(bcs, extraps, Val(N))

    # Extend grids/data for exclusive periodic axes (build-time only)
    # After this, all periodic axes have inclusive-form data.
    grids, data, bcs = _prepare_periodic_nd(grids, data, bcs)

    # Build nodal derivatives using generic ND builder
    nodal_derivs = _build_nd_coeffs(grids, data, bcs)

    # Create spacings (uses @generated to avoid closure boxing for heterogeneous grids)
    spacings = _create_spacings_typed(grids)

    # Normalize BCs for storage — preserve endpoint and resolved period for periodic axes
    bcs_store = ntuple(Val(N)) do d
        if _is_periodic_bc(bcs[d])
            period = last(grids[d]) - first(grids[d])
            _with_resolved_period(bcs[d], period)
        else
            _normalize_bc(bcs[d], Tv)
        end
    end

    # Convert extrap symbols to Val types
    extraps_val = ntuple(Val(N)) do d
        is_periodic = _is_periodic_bc(bcs[d])
        is_periodic ? Val(:wrap) : _symbol_to_extrap_val(extraps[d])
    end

    # Construct the interpolant
    NP1 = N + 1
    return CubicInterpolantND{
        Tg, Tv, N, NP1,
        typeof(grids), typeof(spacings), typeof(bcs_store),
        typeof(extraps_val), typeof(searches)
    }(grids, spacings, nodal_derivs, bcs_store, extraps_val, searches)
end

# ========================================
# ND Internal Helpers (Val-recursive)
# ========================================

@inline _check_periodic_extrap(
    bcs::NTuple{N, AbstractBC},
    extraps::NTuple{N, Symbol},
    ::Val{N}
) where {N} = _check_periodic_extrap(bcs, extraps, Val(1), Val(N))

@inline function _check_periodic_extrap(
    bcs::NTuple{N, AbstractBC},
    extraps::NTuple{N, Symbol},
    ::Val{D},
    ::Val{N}
) where {D, N}
    is_periodic = _is_periodic_bc(bcs[D])
    if is_periodic && extraps[D] != :none && extraps[D] != :wrap
        throw(ArgumentError("Periodic BC on dim $D only supports extrap=:none or :wrap, got :$(extraps[D])"))
    end
    if D < N
        _check_periodic_extrap(bcs, extraps, Val(D + 1), Val(N))
    end
    return nothing
end

"""
    _build_nd_interpolant(..., ::OnTheFly)

Build CubicInterpolantND with on-the-fly coefficient computation.
Not yet implemented.
"""
function _build_nd_interpolant(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, AbstractBC},
    extraps::NTuple{N, Symbol},
    searches::NTuple{N, AbstractSearchPolicy},
    ::OnTheFly
) where {Tg<:AbstractFloat, Tv, N}
    throw(ArgumentError("OnTheFly strategy is not yet implemented for ND. Use PreCompute() (default)."))
end

# ========================================
# POOL-BASED ND ONE-SHOT IMPLEMENTATION
# ========================================
#
# Zero-allocation ND one-shot evaluation using pool-based partials.
# Bypasses Interpolant construction entirely — computes partials in a pool buffer,
# evaluates at the query point(s), then releases all buffers on scope exit.
#
# All eval pipeline functions are standalone:
#   _handle_all_extraps, _search_all_intervals, _compute_all_local_params, _eval_nd_cell

"""
    _cubic_interp_nd_oneshot(grids, data, query, bcs, extraps_val, searches, ops)

Pool-based scalar one-shot ND cubic Hermite evaluation.
Computes 2^N partial derivatives in a pool buffer and evaluates at a single point.
Zero-allocation after warmup (pool reuse).

`extraps_val` must be a pre-resolved tuple of `Val` types (e.g., `(Val(:none), Val(:none))`),
computed via `@_dispatch_extrap_nd` in the API layer for type stability.
"""
@with_pool pool function _cubic_interp_nd_oneshot(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    query::Tuple{Vararg{Real, N}},
    bcs::NTuple{N, AbstractBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    # 1. Extend exclusive periodic axes (pool-based, zero heap alloc)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)

    # 2. Pool-allocate partials array (THE KEY: pool instead of heap)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data_p)...))

    # 3. Compute all partial derivatives in-place
    #    (internally uses autocached 1D caches + nested @with_pool for temp buffers)
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)

    # 4. Create spacings (ScalarSpacing for Range grids = zero alloc)
    spacings = _create_spacings_typed(grids_p)

    # 5. Eval pipeline (all standalone functions, no Interpolant needed)
    q_evals = _handle_all_extraps(query, grids_p, extraps_val)
    indices, Ls, _ = _search_all_intervals(q_evals, grids_p, spacings, searches)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, spacings, indices, Ls)

    # 6. Tensor-product kernel evaluation
    return _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _cubic_interp_nd_oneshot_soa(grids, data, queries_soa, bcs, extraps_val, searches, ops)

Pool-based SoA batch one-shot ND cubic Hermite evaluation.
Computes partials ONCE, then evaluates at all query points.
Output vector is heap-allocated (return value).

`extraps_val` must be a pre-resolved tuple of `Val` types.
"""
@with_pool pool function _cubic_interp_nd_oneshot_soa(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    bcs::NTuple{N, AbstractBC},
    extraps_val::NTuple{N, Val},
    searches::NTuple{N, AbstractSearchPolicy},
    ops::NTuple{N, AbstractEvalOp}
) where {Tg<:AbstractFloat, Tv, N}
    # Validate query lengths
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end

    # Build phase (same as scalar, done once)
    grids_p, data_p, bcs_p = _prepare_periodic_nd_pooled(pool, grids, data, bcs)
    n_partials = 1 << N
    partials = unsafe_acquire!(pool, Tv, (n_partials, size(data_p)...))
    _compute_nd_partials!(partials, grids_p, data_p, bcs_p)
    spacings = _create_spacings_typed(grids_p)

    # Allocate output (heap — this IS the return value)
    output = Vector{Tv}(undef, n_queries)

    # Eval loop: search + kernel per query point
    @inbounds for k in 1:n_queries
        query_k = ntuple(d -> queries[d][k], Val(N))
        q_evals = _handle_all_extraps(query_k, grids_p, extraps_val)
        indices, Ls, _ = _search_all_intervals(q_evals, grids_p, spacings, searches)
        hs, inv_hs, dLs = _compute_all_local_params(q_evals, spacings, indices, Ls)
        output[k] = _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
    end
    return output
end

# ========================================
# EXPORTS
# ========================================
# Note: Exports are handled in the main module file (FastInterpolations.jl)
# Types: CubicInterpolantND, PreCompute, OnTheFly, AbstractCoeffStrategy
