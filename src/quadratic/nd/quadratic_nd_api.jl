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

"""
    quadratic_interp(grids, data, query; deriv=0, kwargs...)

One-shot ND quadratic interpolation at a single point.
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    queries::Tuple{Vararg{Real, N}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    kwargs...
) where {N}
    itp = quadratic_interp(grids, data; kwargs...)
    return itp(queries; deriv=deriv)
end

"""
    quadratic_interp(grids, data, queries; deriv=0, kwargs...)

One-shot ND quadratic interpolation at multiple points (batch).
"""
function quadratic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    queries::NTuple{N, <:AbstractVector{<:Real}};
    deriv::Union{Int, Val, NTuple{N,Int}}=0,
    kwargs...
) where {N}
    itp = quadratic_interp(grids, data; kwargs...)
    return itp(queries; deriv=deriv)
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
