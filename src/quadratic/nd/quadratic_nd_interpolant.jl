# ========================================
# ND Quadratic Interpolation — Interpolant Construction
# ========================================
#
# BC resolution, constructor API, and internal builder for QuadraticInterpolantND.
# One-shot evaluation is in quadratic_nd_oneshot.jl.

# ========================================
# GENERIC ND: N-ARGUMENT FORM (Constructor)
# ========================================
# BC resolution uses _resolve_bcs_nd from core/nd_utils.jl (shared with cubic).
# AbstractBC flows through lazily — normalization happens in _slope_1d_quadratic!
# when Tv is known (lazy normalization pattern, consistent with cubic ND).

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
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = Left(QuadraticFit()),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = PreCompute()
    ) where {N, Tv_raw}

    # OnTheFly → delegate to HeteroInterpolantND (sequential 1D collapse)
    if coeffs isa OnTheFly
        bcs_tuple = bc isa AbstractBC ? ntuple(_ -> bc, Val(N)) : bc
        methods = map(QuadraticInterp, bcs_tuple)
        return _build_hetero_nd(grids, data, methods, extrap, search)
    end

    # Zero-allocation type promotion and grid conversion
    grids_typed, _, Tv, _ = _nd_promote_grids(grids, data)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    # Validate dimensions
    _validate_nd_grids(grids_typed, data_typed)

    # Resolve per-axis options
    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    # Quadratic only supports inclusive periodic → grid-span equals period →
    # 5-arg `_resolve_extrap` with BCs is safe (no extension needed).
    extraps_val = _resolve_extrap(extrap, bcs, grids_typed, Val(N), Tv)
    return _build_nd_quadratic_interpolant(grids_typed, data_typed, bcs, extraps_val, searches)
end

# ========================================
# INTERNAL BUILDER
# ========================================

function _build_nd_quadratic_interpolant(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC},
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy}
    ) where {Tg, Tv, N}
    # Cache axes for the build phase — inner ctor of `QuadraticInterpolantND`
    # handles the owned `_convert_copy` separately, so we only wrap (no copy)
    # here. Already-cached axes pass through idempotently in the ctor.
    grids_cached = map((g, bc) -> _cache_axis(g, bc, Tg), grids, bcs)
    nodal_derivs = _build_nd_coeffs_quadratic(grids_cached, data, bcs)

    return QuadraticInterpolantND(grids_cached, nodal_derivs, bcs, extraps_val, searches)
end
