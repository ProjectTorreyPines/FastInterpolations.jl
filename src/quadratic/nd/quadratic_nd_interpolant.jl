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

    # Quadratic does not support `PeriodicBC` of any flavor — reject before
    # `_resolve_extrap` (which would project exclusive periodic into a wrap
    # extrap on a grid the build path cannot consume) and before `_cache_axis`
    # (which would wrap the axis to virtual `n+1` length, producing a
    # `DimensionMismatch` instead of the intended `ArgumentError`).
    _validate_quadratic_bcs_nd(bcs)
    extraps_val = _resolve_extrap(extrap, bcs, grids_typed, Val(N), Tv)
    return _build_nd_quadratic_interpolant(grids_typed, data_typed, bcs, extraps_val, searches)
end

# ========================================
# BC validation — reject unsupported BCs upfront
# ========================================
#
# Quadratic ND does not support `PeriodicBC`. Without this guard, the
# `:inclusive` variant fires `ArgumentError` deep inside `_slope_1d_quadratic!`
# (via the `AbstractBC` fallback), and the `:exclusive` variant fires a
# misleading `DimensionMismatch` because `_cache_axis` wraps the grid to
# virtual `n+1` length before any BC validation runs.
@noinline _throw_quadratic_periodic_unsupported(axis_idx::Int, bc) = throw(
    ArgumentError(
        "Quadratic interpolation does not support PeriodicBC (axis $axis_idx: $(typeof(bc))). " *
            "Supported: Left(...), Right(...), MinCurvFit, ZeroCurvBC, ZeroSlopeBC, or PolyFit variants."
    )
)

@inline function _validate_quadratic_bcs_nd(bcs::NTuple{N, AbstractBC}) where {N}
    for d in 1:N
        bc = bcs[d]
        bc isa PeriodicBC && _throw_quadratic_periodic_unsupported(d, bc)
    end
    return nothing
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
    grids_cached = map(_cache_axis, grids, bcs, ntuple(_ -> Tg, Val(N)))
    nodal_derivs = _build_nd_coeffs_quadratic(grids_cached, data, bcs)

    return QuadraticInterpolantND(grids_cached, nodal_derivs, bcs, extraps_val, searches)
end

# N=1 collapse: a 1-axis grid tuple forwards to the genuine 1D quadratic path (lean
# 1D batch loop; per-axis 1-tuple kwargs unwrap to scalar). More specific than the
# `NTuple{N}` method above, so it only claims N=1. See linear_nd_interpolant.jl.
@inline quadratic_interp(grids::Tuple{AbstractVector}, data::AbstractVector; kwargs...) =
    quadratic_interp(only(grids), data; _unwrap_nd_kwargs(values(kwargs))...)

# N=1 scalar one-shot: bare scalar → scalar query `(q,)` → ND scalar one-shot
# (scalar output, not `[val]`). See linear_nd_interpolant.jl.
@inline quadratic_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Real; kwargs...) =
    quadratic_interp(grids, data, (q,); kwargs...)

# N=1 batch one-shot → lean 1D batch one-shot (bit-identical). See linear_nd_interpolant.jl.
@inline quadratic_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; coeffs::AbstractCoeffStrategy = AutoCoeffs(), kwargs...) =
    quadratic_interp(only(grids), data, q; _unwrap_nd_batch_kwargs(values(kwargs))...)
@inline quadratic_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; coeffs::AbstractCoeffStrategy = AutoCoeffs(), kwargs...) =
    quadratic_interp!(output, only(grids), data, q; _unwrap_nd_batch_kwargs(values(kwargs))...)
# Single-axis SoA `(xv,)` → 1D batch. See linear_nd_interpolant.jl.
@inline quadratic_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; coeffs::AbstractCoeffStrategy = AutoCoeffs(), kwargs...) =
    quadratic_interp(only(grids), data, only(q); _unwrap_nd_batch_kwargs(values(kwargs))...)
@inline quadratic_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; coeffs::AbstractCoeffStrategy = AutoCoeffs(), kwargs...) =
    quadratic_interp!(output, only(grids), data, only(q); _unwrap_nd_batch_kwargs(values(kwargs))...)
