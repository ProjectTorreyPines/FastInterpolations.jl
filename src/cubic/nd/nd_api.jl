# ========================================
# ND Cubic Interpolation Public API
# ========================================
#
# Unified public API for N-dimensional cubic interpolation.
# Extends cubic_interp() to support tuple grid inputs.
#
# Implements:
# - Generic ND with PreCompute strategy (uses CubicInterpolantND for all N≥2)
# - bicubic_interp() for specialized 2D (deprecated, for testing only)

# ========================================
# BICUBIC_INTERP: Specialized 2D (Deprecated)
# ========================================

"""
    bicubic_interp(grids::Tuple{AbstractVector, AbstractVector}, data::AbstractMatrix; kwargs...)

Create a specialized 2D bicubic interpolant (BicubicInterpolant).

!!! warning "Deprecated"
    This function is deprecated and will be removed in a future version.
    Use `cubic_interp((x, y), data)` instead, which returns a `CubicInterpolantND{...,2,...}`.

# Arguments
- `grids::Tuple{AbstractVector, AbstractVector}`: Tuple of (x, y) grid vectors
- `data::AbstractMatrix`: Function values at grid points, size (length(x), length(y))

# Keywords
- `bc=NaturalBC()`: Boundary condition(s)
- `extrap=:none`: Extrapolation mode(s)
- `search=Binary()`: Search policy(s)
- `coeffs=PreCompute()`: Coefficient computation strategy

# Returns
- `BicubicInterpolant{Tg, Tv, ...}`: Callable interpolant object

# Examples
```julia
x = range(0.0, 2π, 50)
y = range(0.0, π, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
itp = bicubic_interp((x, y), data)
itp(1.0, 0.5)  # Evaluate at (1.0, 0.5)
```
"""
function bicubic_interp(
    grids::Tuple{AbstractVector, AbstractVector},
    data::AbstractMatrix;
    bc::Union{AbstractBC, Tuple{AbstractBC, AbstractBC}}=NaturalBC(),
    extrap::Union{Symbol, Tuple{Symbol, Symbol}}=:none,
    search::Union{AbstractSearchPolicy, Tuple{AbstractSearchPolicy, AbstractSearchPolicy}}=Binary(),
    coeffs::AbstractCoeffStrategy=PreCompute()
)
    x_raw, y_raw = grids

    # Promote grid types
    Tg = promote_type(eltype(x_raw), eltype(y_raw))
    Tg = Tg <: AbstractFloat ? Tg : Float64  # Ensure AbstractFloat

    # Convert grids if needed
    x = _convert_grid(x_raw, Tg)
    y = _convert_grid(y_raw, Tg)

    # Validate dimensions
    _validate_2d_grids(x, y, data)

    # Resolve per-axis options
    bc_x, bc_y = _resolve_bcs_nd(bc, Val(2))
    extrap_x, extrap_y = _resolve_extrap_nd(extrap, Val(2))
    search_x, search_y = _resolve_search_nd(search, Val(2))

    # Build using specialized 2D path
    return _build_bicubic_interpolant(x, y, data, bc_x, bc_y, extrap_x, extrap_y, search_x, search_y, coeffs)
end

"""
    bicubic_interp(grids, data, queries; kwargs...)

One-shot specialized 2D bicubic interpolation (deprecated).
"""
function bicubic_interp(
    grids::Tuple{AbstractVector, AbstractVector},
    data::AbstractMatrix,
    queries::Tuple{Real, Real};
    deriv::Tuple{Int,Int}=(0, 0),
    kwargs...
)
    itp = bicubic_interp(grids, data; kwargs...)
    return itp(queries[1], queries[2]; deriv=deriv)
end

function bicubic_interp(
    grids::Tuple{AbstractVector, AbstractVector},
    data::AbstractMatrix,
    queries::Tuple{AbstractVector{<:Real}, AbstractVector{<:Real}};
    deriv::Tuple{Int,Int}=(0, 0),
    kwargs...
)
    itp = bicubic_interp(grids, data; kwargs...)
    return itp(queries[1], queries[2]; deriv=deriv)
end

# ========================================
# INTERNAL BUILDERS (for bicubic_interp)
# ========================================

"""
    _build_bicubic_interpolant(x, y, data, bc_x, bc_y, extrap_x, extrap_y, search_x, search_y, ::PreCompute)

Build BicubicInterpolant with precomputed coefficients.
"""
function _build_bicubic_interpolant(
    x::AbstractVector{Tg},
    y::AbstractVector{Tg},
    data::AbstractMatrix{Tv},
    bc_x::AbstractBC,
    bc_y::AbstractBC,
    extrap_x::Symbol,
    extrap_y::Symbol,
    search_x::AbstractSearchPolicy,
    search_y::AbstractSearchPolicy,
    ::PreCompute
) where {Tg<:AbstractFloat, Tv}
    # Validate periodic BC + extrap compatibility
    is_periodic_x = _is_periodic_bc(bc_x)
    is_periodic_y = _is_periodic_bc(bc_y)

    if is_periodic_x && extrap_x != :none && extrap_x != :wrap
        throw(ArgumentError("Periodic BC on x only supports extrap=:none or :wrap, got :$extrap_x"))
    end
    if is_periodic_y && extrap_y != :none && extrap_y != :wrap
        throw(ArgumentError("Periodic BC on y only supports extrap=:none or :wrap, got :$extrap_y"))
    end

    # Build nodal derivatives
    nodal_derivs = _build_bicubic_coeffs(x, y, data, bc_x, bc_y)

    # Create spacings
    spacing_x = _create_spacing(x)
    spacing_y = _create_spacing(y)

    # Normalize BCs for storage
    bc_x_store = _is_periodic_bc(bc_x) ? PeriodicBC() : _normalize_bc(bc_x, Tv)
    bc_y_store = _is_periodic_bc(bc_y) ? PeriodicBC() : _normalize_bc(bc_y, Tv)

    # Convert extrap symbols to Val types
    extrap_x_val = is_periodic_x ? Val(:wrap) : _symbol_to_extrap_val(extrap_x)
    extrap_y_val = is_periodic_y ? Val(:wrap) : _symbol_to_extrap_val(extrap_y)

    return BicubicInterpolant{
        Tg, Tv,
        typeof(x), typeof(y),
        typeof(spacing_x), typeof(spacing_y),
        typeof(bc_x_store), typeof(bc_y_store),
        typeof(extrap_x_val), typeof(extrap_y_val),
        typeof(search_x), typeof(search_y)
    }(
        x, y,
        spacing_x, spacing_y,
        nodal_derivs,
        bc_x_store, bc_y_store,
        extrap_x_val, extrap_y_val,
        search_x, search_y
    )
end

"""
    _build_bicubic_interpolant(..., ::OnTheFly)

Build BicubicInterpolant with on-the-fly coefficient computation.
Not yet implemented.
"""
function _build_bicubic_interpolant(
    x::AbstractVector{Tg},
    y::AbstractVector{Tg},
    data::AbstractMatrix{Tv},
    bc_x::AbstractBC,
    bc_y::AbstractBC,
    extrap_x::Symbol,
    extrap_y::Symbol,
    search_x::AbstractSearchPolicy,
    search_y::AbstractSearchPolicy,
    ::OnTheFly
) where {Tg<:AbstractFloat, Tv}
    throw(ArgumentError("OnTheFly strategy is not yet implemented. Use PreCompute() (default)."))
end

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
    # Promote grid types
    Tg = promote_type(map(eltype, grids)...)
    Tg = Tg <: AbstractFloat ? Tg : Float64  # Ensure AbstractFloat

    # Convert grids if needed
    grids_typed = ntuple(d -> _convert_grid(grids[d], Tg), Val(N))

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

# Keywords
- `deriv`: `Int` (0-3) or `Val((d1,d2,...))` for mixed partials
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    queries::Tuple{Vararg{Real, N}};  # Allow heterogeneous Real types (AD support)
    deriv::Union{Int, Val}=0,
    kwargs...
) where {N}
    itp = cubic_interp(grids, data; kwargs...)
    return itp(queries; deriv=deriv)
end

"""
    cubic_interp(grids, data, queries; deriv=0, kwargs...)

One-shot ND cubic interpolation at multiple points (batch).

# Keywords
- `deriv`: `Int` (0-3) or `Val((d1,d2,...))` for mixed partials
"""
function cubic_interp(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{<:Any, N},
    queries::NTuple{N, <:AbstractVector{<:Real}};
    deriv::Union{Int, Val}=0,
    kwargs...
) where {N}
    itp = cubic_interp(grids, data; kwargs...)
    return itp(queries; deriv=deriv)
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
    # Validate periodic BC + extrap compatibility
    for d in 1:N
        is_periodic = _is_periodic_bc(bcs[d])
        if is_periodic && extraps[d] != :none && extraps[d] != :wrap
            throw(ArgumentError("Periodic BC on dim $d only supports extrap=:none or :wrap, got :$(extraps[d])"))
        end
    end

    # Build nodal derivatives using generic ND builder
    nodal_derivs = _build_nd_coeffs(grids, data, bcs)

    # Create spacings
    spacings = ntuple(d -> _create_spacing(grids[d]), Val(N))

    # Normalize BCs for storage (use Tv for value-typed BCs)
    bcs_store = ntuple(Val(N)) do d
        _is_periodic_bc(bcs[d]) ? PeriodicBC() : _normalize_bc(bcs[d], Tv)
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
# EXPORTS
# ========================================
# Note: Exports are handled in the main module file (FastInterpolations.jl)
# Types: BicubicInterpolant, CubicInterpolantND, PreCompute, OnTheFly, AbstractCoeffStrategy
