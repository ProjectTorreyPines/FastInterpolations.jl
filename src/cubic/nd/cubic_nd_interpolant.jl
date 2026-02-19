# ========================================
# ND Cubic Interpolation — Interpolant Construction
# ========================================
#
# Constructor API and internal builders for CubicInterpolantND.
# One-shot evaluation is in cubic_nd_oneshot.jl.

# ========================================
# GENERIC ND: N-ARGUMENT FORM (Constructor)
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
    extrap::Union{Symbol, NTuple{N,Symbol}, AbstractExtrapMode, NTuple{N,AbstractExtrapMode}}=NoExtrap(),
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
    searches = _resolve_search_nd(search, Val(N))

    if extrap isa AbstractExtrapMode || extrap isa Tuple{Vararg{AbstractExtrapMode}}
        extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N))
        return _build_nd_interpolant(grids_typed, data, bcs, extraps_val, searches, coeffs)
    else
        Base.depwarn(_EXTRAP_SYMBOL_DEPWARN, :cubic_interp)
        extraps = _resolve_extrap_nd(extrap, Val(N))
        _check_periodic_extrap(bcs, extraps, Val(N))
        @_dispatch_extrap_nd extraps bcs => extraps_val begin
            return _build_nd_interpolant(grids_typed, data, bcs, extraps_val, searches, coeffs)
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
    extraps_val::Tuple{Vararg{AbstractExtrapMode, N}},
    searches::NTuple{N, AbstractSearchPolicy},
    ::PreCompute
) where {Tg<:AbstractFloat, Tv, N}
    # Extend grids/data for exclusive periodic axes (build-time only)
    # After this, all periodic axes have inclusive-form data.
    grids, data, bcs = _prepare_periodic_nd(grids, data, bcs)

    # Build nodal derivatives using generic ND builder
    nodal_derivs = _build_nd_coeffs(grids, data, bcs)

    # Create spacings (uses @generated to avoid closure boxing for heterogeneous grids)
    spacings = _create_spacings_typed(grids)

    # Normalize BCs for storage — preserve endpoint and resolved period for periodic axes
    # Uses map instead of ntuple so each bc element gets its concrete type
    # (ntuple indexes with Int → Union return; map dispatches per-element → compile-time branch)
    bcs_store = map(bcs, grids) do bc, grid
        if _is_periodic_bc(bc)
            period = last(grid) - first(grid)
            _with_resolved_period(bc, period)
        else
            _normalize_bc(bc, Tv)
        end
    end

    # extraps_val already dispatched to concrete Val types at API boundary
    # (via @_dispatch_extrap_nd in cubic_interp)

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
    extraps_val::Tuple{Vararg{AbstractExtrapMode, N}},
    searches::NTuple{N, AbstractSearchPolicy},
    ::OnTheFly
) where {Tg<:AbstractFloat, Tv, N}
    throw(ArgumentError("OnTheFly strategy is not yet implemented for ND. Use PreCompute() (default)."))
end
