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
- `bc=CubicFit()`: Boundary condition(s). Can be:
  - Single `AbstractBC`: Applied to all axes
  - `NTuple{N,AbstractBC}`: Per-axis BCs
- `extrap=NoExtrap()`: Extrapolation mode(s). Can be:
  - Single `AbstractExtrap`: Applied to all axes (`NoExtrap()`, `ClampExtrap()`, `WrapExtrap()`)
  - `NTuple{N,AbstractExtrap}`: Per-axis modes
- `search=AutoSearch()`: Search policy(s). Can be:
  - Single `AbstractSearchPolicy`: Applied to all axes
  - `NTuple{N,AbstractSearchPolicy}`: Per-axis policies
- `coeffs=PreCompute()`: Coefficient computation strategy

# Returns
- `CubicInterpolantND{Tg, Tv, N, ...}`: Callable interpolant object

# Type Inference
- Grid type `Tg`: Promoted from all grid element types (supports duck types like ForwardDiff.Dual)
- Value type `Tv`: Element type of data (unconstrained)

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
    bc=(CubicFit(), PeriodicBC(), CubicFit()),
    extrap=(NoExtrap(), WrapExtrap(), ClampExtrap()))

# Complex-valued data
data_c = [sin(xi) * cos(yj) * zk + im * cos(xi) for xi in x, yj in y, zk in z]
itp_c = cubic_interp((x, y, z), data_c)
```
"""
# Public ND constructor (N≥2; N=1 is intercepted by the collapse method below and
# only reaches the ND builder for the no-1D-equivalent `coeffs=OnTheFly()` case).
@inline cubic_interp(grids::NTuple{N, AbstractVector}, data::AbstractArray{<:Any, N}; kwargs...) where {N} =
    _cubic_interp_nd(grids, data; kwargs...)

function _cubic_interp_nd(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv_raw, N};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = CubicFit(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = PreCompute(),
        store::StorePolicy = StorePolicy()
    ) where {N, Tv_raw}
    # Zero-allocation type promotion + grid conversion
    grids_typed, _, Tv, _ = _nd_promote_grids(grids, data)

    # Promote data type (Int→Float64, Complex{T}→Complex{Tg}, custom types preserved)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    # Validate dimensions
    _validate_nd_grids(grids_typed, data_typed)

    # Resolve per-axis options
    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    extraps_val = _resolve_extrap(extrap, bcs, Val(N), Tv)

    # OnTheFly → delegate to HeteroInterpolantND (sequential 1D collapse)
    if coeffs isa OnTheFly
        methods = map(CubicInterp, bcs)
        return _build_hetero_nd(grids, data, methods, extrap, search; store = store)
    end

    # PreCompute keeps no raw data (it builds a 2^N nodal-derivatives array) →
    # data-ref is structurally N/A; warn + copy if reference was requested.
    _check_store(store, "cubic ND PreCompute (CubicInterpolantND)")
    return _build_nd_interpolant(grids_typed, data_typed, bcs, extraps_val, searches, coeffs)
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
        extraps_val::Tuple{Vararg{AbstractExtrap, N}},
        searches::NTuple{N, AbstractSearchPolicy},
        ::PreCompute
    ) where {Tg, Tv, N}
    # Extend grids/data for exclusive periodic axes; periodic bcs are
    # promoted to `:extended` per axis (via `_bc_after_extend` inside the
    # helper) so downstream dispatch reflects the closed-cycle layout.
    grids, data, bcs = _prepare_periodic_nd(grids, data, bcs)

    # Wrap extended raw grids into cached axes so eval reads `_get_h(grids[d], i)`
    # in O(1) (cached scalar/vector lookup) instead of falling back to on-the-fly
    # `float(x[i+1] - x[i])`. Mirrors the forward path on Linear/Constant
    # /Quadratic/Hetero ND.
    grids = map(_cache_axis, grids, bcs)

    # Per-axis materialization via 2-arg primitive — post-extension, `last(grid) -
    # first(grid) == period`, so grid-span is the correct wrap domain and we avoid
    # the bc-aware constructor's pre-extension `last(x) < x_max` check.
    extraps_val = map(_resolve_extrap, extraps_val, grids)

    # Build nodal derivatives using generic ND builder
    nodal_derivs = _build_nd_coeffs(grids, data, bcs)

    # Periodic bcs are already `:inclusive`; materialize period from the
    # extended grid span for introspection. Non-periodic axes go through
    # `_normalize_bc` for BCPair/PointBC handling.
    bcs_store = map(bcs, grids) do bc, grid
        if _is_periodic_bc(bc)
            _with_resolved_period(bc, last(grid) - first(grid))
        else
            _normalize_bc(bc, first(data))
        end
    end

    # extraps_val already resolved to concrete AbstractExtrap instances at API boundary
    # (via _resolve_extrap_nd in cubic_interp)
    return CubicInterpolantND(grids, nodal_derivs, bcs_store, extraps_val, searches)
end

# ── N=1 collapse (shared rationale in linear_nd_interpolant.jl) ──
# Cubic is the one method where 1D and ND differ in a kwarg: 1D is inherently
# PreCompute (`autocache::Bool`, no `coeffs`), ND accepts `coeffs`. `coeffs` is a
# compile-time-known kwarg, so the `coeffs isa OnTheFly` guard folds away — the common
# path forwards to the lean 1D method; OnTheFly (local, no 1D equivalent) stays on the
# factored ND internals. The 1D branch passes a bare-vector grid (`only(grids)`), so it
# hits the genuine 1D method and is never re-intercepted (no infinite recursion).
@inline function cubic_interp(grids::Tuple{AbstractVector}, data::AbstractVector; coeffs::AbstractCoeffStrategy = PreCompute(), kwargs...)
    coeffs isa OnTheFly && return _cubic_interp_nd(grids, data; coeffs, kwargs...)
    return cubic_interp(only(grids), data; _unwrap_nd_kwargs(values(kwargs))...)
end

# Scalar one-shot: bare scalar → `(q,)` → ND scalar one-shot (handles coeffs natively).
@inline cubic_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Real; kwargs...) =
    cubic_interp(grids, data, (q,); kwargs...)

# Batch one-shot (bare vector; the SoA `(xv,)` form below unwraps into it).
@inline function cubic_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; coeffs::AbstractCoeffStrategy = AutoCoeffs(), kwargs...)
    coeffs isa OnTheFly && return _cubic_interp_nd_oneshot_alloc(grids, data, q; coeffs, kwargs...)
    return cubic_interp(only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
end
@inline function cubic_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; coeffs::AbstractCoeffStrategy = AutoCoeffs(), kwargs...)
    coeffs isa OnTheFly && return _cubic_interp_nd_oneshot_batch!(output, grids, data, q; coeffs, kwargs...)
    return cubic_interp!(output, only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
end
@inline cubic_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    cubic_interp(grids, data, only(q); kwargs...)
@inline cubic_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    cubic_interp!(output, grids, data, only(q); kwargs...)

# OnTheFly is handled in cubic_interp() above (delegates to _build_hetero_nd).
# No _build_nd_interpolant(::OnTheFly) needed.
