# ========================================
# TensorProductInterpolantND — Constructor
# ========================================
# Public API: interp_nd(grids, data; methods, ...)
# Internal:   _build_tensor_product_nd(grids, data, methods, extrap, search)

# ========================================
# Homogeneous Auto-Dispatch
# ========================================
# When all axes use the same method type, delegate to the existing optimized
# ND constructors (cubic_interp, linear_interp, etc.) which have richer features
# (adjoint, oneshot, AD). Per-axis options (bc, side) are forwarded as tuples —
# the existing constructors already accept Union{Single, NTuple} for these kwargs.

function _interp_nd_dispatch(
        grids, data, methods::Tuple{<:CubicInterp, Vararg{<:CubicInterp}}, coeffs, extrap, search
    )
    bcs = map(m -> m.bc, methods)
    return cubic_interp(grids, data; bc = bcs, extrap = extrap, search = search, coeffs = coeffs)
end

function _interp_nd_dispatch(
        grids, data, ::Tuple{LinearInterp, Vararg{LinearInterp}}, ::Any, extrap, search
    )
    return linear_interp(grids, data; extrap = extrap, search = search)
end

function _interp_nd_dispatch(
        grids, data, methods::Tuple{<:QuadraticInterp, Vararg{<:QuadraticInterp}}, ::Any, extrap, search
    )
    bcs = map(m -> m.bc, methods)
    return quadratic_interp(grids, data; bc = bcs, extrap = extrap, search = search)
end

function _interp_nd_dispatch(
        grids, data, methods::Tuple{<:ConstantInterp, Vararg{<:ConstantInterp}}, ::Any, extrap, search
    )
    sides = map(m -> m.side, methods)
    return constant_interp(grids, data; side = sides, extrap = extrap, search = search)
end

# Heterogeneous (fallback) → TensorProductInterpolantND
function _interp_nd_dispatch(
        grids, data, methods::Tuple{Vararg{AbstractInterpMethod, N}}, coeffs, extrap, search
    ) where {N}
    if coeffs isa PreCompute
        return _build_tensor_product_precomputed(grids, data, methods, extrap, search)
    else
        return _build_tensor_product_nd(grids, data, methods, extrap, search)
    end
end

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
        typeof(extraps), typeof(searches), typeof(data_typed),
    }(
        grids_typed, spacings, data_typed, methods, extraps, searches
    )
end

# ========================================
# Precomputed Builder
# ========================================

function _build_tensor_product_precomputed(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv_raw, N},
        methods::Tuple{Vararg{AbstractInterpMethod, N}},
        extrap,
        search,
    ) where {N, Tv_raw}
    _validate_nd_grids(grids, data)
    Tg = _promote_grid_eltype(grids)
    Tg = Tg <: AbstractFloat ? Tg : Float64
    grids_typed = _convert_grids_typed(grids, Tg)
    Tv = _value_type(Tv_raw, Tg)
    extraps = _resolve_extrap_nd(extrap, nothing, Val(N), Tv)
    searches = _resolve_search_nd(search, Val(N))
    _validate_axis_methods(grids_typed, methods, extraps)

    # Extend exclusive periodic axes to inclusive form (same as CubicInterpolantND).
    # Must happen before spacings + partials so the stored grid matches the data.
    bcs_periodic = map(_bc_for_periodic_check, methods)
    grids_typed, data_ext, _ = _prepare_periodic_nd(grids_typed, data, bcs_periodic)
    spacings = _create_spacings_typed(grids_typed)

    # Build partials on the (possibly extended) data
    hetero_partials = _build_nd_coeffs_hetero(grids_typed, Tv, data_ext, methods)

    return TensorProductInterpolantND{
        Tg, Tv, N,
        typeof(grids_typed), typeof(spacings), typeof(methods),
        typeof(extraps), typeof(searches), typeof(hetero_partials),
    }(
        grids_typed, spacings, hetero_partials, methods, extraps, searches
    )
end

# ========================================
# Public API
# ========================================

"""
    interp(grids, data; method, coeffs=PreCompute(), extrap=NoExtrap(), search=AutoSearch())

Unified N-dimensional interpolation constructor with per-axis method specification.

Automatically dispatches to the optimal implementation:
- **Homogeneous** (all axes same method): delegates to existing optimized types
  (`CubicInterpolantND`, `LinearInterpolantND`, etc.) with full feature support
  (adjoint, oneshot, AD)
- **Heterogeneous** (mixed methods): creates `TensorProductInterpolantND`

# Arguments
- `grids::NTuple{N, AbstractVector}`: Grid vectors per dimension
- `data::AbstractArray{<:Any, N}`: N-dimensional data array

# Keyword Arguments
- `method`: Interpolation method(s) (**required**)
  - Single `AbstractInterpMethod`: broadcast to all axes (e.g., `method=CubicInterp()`)
  - `Tuple{Vararg{AbstractInterpMethod, N}}`: per-axis (e.g., `method=(CubicInterp(), LinearInterp())`)
- `coeffs=PreCompute()`: Coefficient strategy (heterogeneous only)
  - `PreCompute()`: Precompute partial derivatives (O(1) eval)
  - `OnTheFly()`: Build 1D per query (zero build cost, O(n) eval)
- `extrap=NoExtrap()`: Extrapolation mode(s) — single or per-axis tuple
- `search=AutoSearch()`: Search policy(ies) — single or per-axis tuple

# Examples
```julia
x, y = range(0, 1, 50), range(0, 1, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# Single method → all axes (dispatches to CubicInterpolantND)
itp = interp((x, y), data; method=CubicInterp())

# Heterogeneous → TensorProductInterpolantND
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))

# Per-axis BCs
itp = interp((x, y), data; method=(CubicInterp(CubicFit()), CubicInterp(ZeroCurvBC())))

# Derivatives
itp((0.5, 0.3); deriv=(DerivOp(1), DerivOp(0)))  # ∂f/∂x
gradient(itp, (0.5, 0.3))                          # (∂f/∂x, ∂f/∂y)
```
"""
function interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        coeffs::AbstractCoeffStrategy = PreCompute(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method
    return _interp_nd_dispatch(grids, data, method_tuple, coeffs, extrap, search)
end
