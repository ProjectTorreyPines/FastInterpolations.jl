# ========================================
# TensorProductInterpolantND — Constructor
# ========================================
# Public API: interp_nd(grids, data; methods, ...)
# Internal:   _build_tensor_product_nd(grids, data, methods, extrap, search)

# ========================================
# Per-Axis Validation
# ========================================

function _validate_axis_methods(grids, methods, extraps)
    for d in eachindex(grids)
        _validate_axis_method(grids[d], methods[d], extraps[d], d)
    end
    return nothing
end

function _validate_axis_method(grid, ::CubicInterp{BC}, extrap, d) where {BC}
    n = length(grid)
    if BC <: PeriodicBC
        n < 3 && throw(ArgumentError("Axis $d: cubic periodic needs ≥3 points, got $n"))
        extrap isa WrapExtrap || throw(ArgumentError("Axis $d: PeriodicBC requires WrapExtrap"))
    else
        n < 4 && throw(ArgumentError("Axis $d: cubic needs ≥4 points, got $n"))
    end
    return nothing
end

function _validate_axis_method(grid, ::LinearInterp, _, d)
    n = length(grid)
    n < 2 && throw(ArgumentError("Axis $d: linear needs ≥2 points, got $n"))
    return nothing
end

function _validate_axis_method(grid, ::QuadraticInterp, _, d)
    n = length(grid)
    n < 3 && throw(ArgumentError("Axis $d: quadratic needs ≥3 points, got $n"))
    return nothing
end

function _validate_axis_method(grid, ::ConstantInterp, _, d)
    n = length(grid)
    n < 2 && throw(ArgumentError("Axis $d: constant needs ≥2 points, got $n"))
    return nothing
end

# ========================================
# Internal Builder
# ========================================

function _build_tensor_product_nd(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv_raw, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap,
        search,
    ) where {N, Tv_raw}
    # 1. Validate grid dimensions
    _validate_nd_grids(grids, data)

    # 2. Promote grid type (Int → Float64)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64

    # 3. Convert grids to target type (preserving Range structure)
    grids_typed = _convert_grids_typed(grids, Tg)

    # 4. Create spacings
    spacings = _create_spacings_typed(grids_typed)

    # 5. Promote data type
    Tv = _value_type(Tv_raw, Tg)
    data_typed = Tv === Tv_raw ? Array(data) : Array{Tv}(data)

    # 6. Resolve per-axis configuration
    extraps = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    searches = _resolve_search_nd(search, Val(N))

    # 7. Per-axis method validation
    _validate_axis_methods(grids_typed, methods, extraps)

    return TensorProductInterpolantND{
        Tg, Tv, N,
        typeof(grids_typed), typeof(spacings), typeof(methods),
        typeof(extraps), typeof(searches),
    }(
        grids_typed, spacings, data_typed, methods, extraps, searches
    )
end

# ========================================
# Public API
# ========================================

"""
    interp_nd(grids, data; methods, extrap=NoExtrap(), search=AutoSearch())

Create an N-dimensional interpolant with per-axis method specification.

Constructs a [`TensorProductInterpolantND`](@ref) that evaluates via sequential
1D interpolation along each axis. Supports heterogeneous methods across
dimensions (e.g., cubic on axis 1, linear on axis 2).

# Arguments
- `grids::NTuple{N, AbstractVector}`: Grid vectors per dimension
- `data::AbstractArray{<:Any, N}`: N-dimensional data array

# Keyword Arguments
- `methods::Tuple{Vararg{AbstractInterpMethod, N}}`: Method per axis (**required**)
- `extrap=NoExtrap()`: Extrapolation mode(s) — single or per-axis tuple
- `search=AutoSearch()`: Search policy(ies) — single or per-axis tuple

# Examples
```julia
x, y = range(0, 1, 50), range(0, 1, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# Heterogeneous: cubic along x, linear along y
itp = interp_nd((x, y), data; methods=(CubicInterp(), LinearInterp()))
itp((0.5, 0.3))

# With per-axis extrapolation
itp = interp_nd((x, y), data;
    methods=(CubicInterp(), LinearInterp()),
    extrap=(ClampExtrap(), NoExtrap()))

# Derivatives
itp((0.5, 0.3); deriv=(DerivOp(1), DerivOp(0)))  # ∂f/∂x
gradient(itp, (0.5, 0.3))                          # (∂f/∂x, ∂f/∂y)
```
"""
function interp_nd(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    ) where {N}
    # v0.1: always use TensorProductInterpolantND
    # Future: detect homogeneous methods and dispatch to existing ND types
    return _build_tensor_product_nd(grids, data, methods, extrap, search)
end
