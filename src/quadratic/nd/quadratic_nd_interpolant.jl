# ========================================
# ND Quadratic Interpolation — Interpolant Construction
# ========================================
#
# BC resolution, constructor API, and internal builder for QuadraticInterpolantND.
# One-shot evaluation is in quadratic_nd_oneshot.jl.

# ========================================
# BC Resolution for Quadratic ND
# ========================================

"""
    _resolve_bcs_nd_quadratic(bc, Val(N)) -> NTuple{N, QuadraticBC}

Resolve boundary condition input to canonical N-tuple of QuadraticBC.

Converts common BC types to their quadratic equivalents:
- `QuadraticBC` (Left, Right, MinCurvFit): pass through
- `ZeroCurvBC()`: → Right(Deriv2(0)) (zero curvature at right endpoint)
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
@inline function _resolve_bcs_nd_quadratic(bc::ZeroCurvBC, ::Val{N}) where {N}
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
@inline _to_quadratic_bc(::ZeroCurvBC) = Right(Deriv2(0.0))
@inline _to_quadratic_bc(bc::PolyFit) = Right(bc)
@inline _to_quadratic_bc(bc::AbstractBC) = throw(ArgumentError(
    "Unsupported BC for quadratic ND: $(typeof(bc)). " *
    "Use Left(...), Right(...), MinCurvFit, ZeroCurvBC(), or PolyFit variants."
))

# ========================================
# GENERIC ND: N-ARGUMENT FORM (Constructor)
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
- `extrap=NoExtrap()`: Extrapolation mode(s)
- `search=AutoSearch()`: Search policy(s)

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
    extrap::Union{AbstractExtrap, NTuple{N,AbstractExtrap}}=NoExtrap(),
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=AutoSearch()
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
    searches = _resolve_search_nd(search, Val(N))

    extraps_val = _resolve_extrap_nd(extrap, bcs, Val(N))
    return _build_nd_quadratic_interpolant(grids_typed, data, bcs, extraps_val, searches)
end

# ========================================
# INTERNAL BUILDER
# ========================================

function _build_nd_quadratic_interpolant(
    grids::NTuple{N, AbstractVector{Tg}},
    data::AbstractArray{Tv, N},
    bcs::NTuple{N, QuadraticBC},
    extraps_val::Tuple{Vararg{AbstractExtrap, N}},
    searches::NTuple{N, AbstractSearchPolicy}
) where {Tg<:AbstractFloat, Tv, N}
    # Build nodal derivatives using quadratic recurrence
    nodal_derivs = _build_nd_coeffs_quadratic(grids, data, bcs)

    # Create spacings (uses @generated to avoid closure boxing)
    spacings = _create_spacings_typed(grids)

    # Store BCs as-is (already QuadraticBC)
    bcs_store = bcs

    # extraps_val already resolved to concrete types at API boundary

    # Construct the interpolant
    NP1 = N + 1
    return QuadraticInterpolantND{
        Tg, Tv, N, NP1,
        typeof(grids), typeof(spacings), typeof(bcs_store),
        typeof(extraps_val), typeof(searches)
    }(grids, spacings, nodal_derivs, bcs_store, extraps_val, searches)
end
