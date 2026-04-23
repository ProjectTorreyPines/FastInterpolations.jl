# ========================================
# Periodic Boundary Condition Helpers
# ========================================
#
# Functions for handling PeriodicBC: wrapping, validation,
# exclusive endpoint extension (1D + ND).
#
# Depends on: bc_types.jl (PeriodicBC{E,P}), utils.jl (_real_eltype, _extract_primal)

# ========================================
# Query Wrapping
# ========================================

"""
    _wrap_to_domain(xi::FT, x_min::FT, x_max::FT) where {FT<:AbstractFloat}

Wrap a query point `xi` to the domain [x_min, x_max).
Used for periodic boundary conditions and extrap=WrapExtrap().

Optimized: skips expensive `mod()` when xi is already in domain.
"""
@inline function _wrap_to_domain(xi::Tg, x_min::Tg, x_max::Tg) where {Tg}
    # Single-branch check: outside domain → slow path
    if (xi < x_min) || (xi >= x_max)
        period = x_max - x_min
        return x_min + mod(xi - x_min, period)
    end
    # Fast path: already in domain (most common case)
    return xi
end

# Generic wrapper: handles Dual, Int, Float32 on Float64 grid, etc.
# IMPORTANT: Preserves AD Dual type through the entire operation.
# mod() is compatible with ForwardDiff.Dual, so we use it directly on xi.
@inline function _wrap_to_domain(xi::Real, x_min::Tg, x_max::Tg) where {Tg}
    xi_primal = _extract_primal(xi)
    # Fast path: already in domain, return original xi (preserves Dual type for AD)
    if (xi_primal >= x_min) && (xi_primal < x_max)
        return xi
    end
    # Slow path: outside domain, wrap using mod (preserves Dual type for AD)
    # mod() works correctly with ForwardDiff.Dual: d/dx[mod(x,p)] = 1
    period = x_max - x_min
    return x_min + mod(xi - x_min, period)
end

# ────────────────────────────────────────────────────────
# Extrap-aware 2-arg wrapper
# ────────────────────────────────────────────────────────
#
# Kernels hand off the materialized `WrapExtrap{T}` directly — no phantom
# `first(x)`, `last(x)` arguments needed. `_resolve_extrap` is responsible for
# materializing any `WrapExtrap{Nothing}` build-time placeholder into
# `WrapExtrap{T}` before evaluation. An unmaterialized one reaching this
# wrapper is a contract violation — the explicit `::WrapExtrap{Nothing}`
# overload below raises a clear error instead of letting `nothing - nothing`
# fail cryptically one layer down.

@inline _wrap_to_domain(xi, e::WrapExtrap) =
    _wrap_to_domain(xi, e._x_min, e._x_max)

@inline _wrap_to_domain(::Any, ::WrapExtrap{Nothing}) =
    _throw_unmaterialized_wrap_extrap()

@noinline _throw_unmaterialized_wrap_extrap() =
    throw(
    ArgumentError(
        "WrapExtrap reached evaluator unmaterialized (WrapExtrap{Nothing}). " *
            "This is a contract violation — `_resolve_extrap` should have " *
            "upgraded it to WrapExtrap{T} at the API boundary."
    )
)

# ========================================
# Endpoint Validation
# ========================================

"""
    _check_periodic_endpoints(y::AbstractVector)

Validate that `y[1] ≈ y[end]` for periodic boundary conditions (inclusive endpoint).
Called once at construction time (zero runtime overhead).

Four-tier dispatch based on element type:

- **`AbstractFloat`**: `isapprox` with `atol = 8eps(T)` and `rtol = √eps(T)` — the atol
  covers near-zero noise floor (e.g., `sin(0)` vs `sin(2π)`), while rtol handles relative
  differences at larger magnitudes (e.g., `cos(0) ≈ cos(2π)`).
  Both constants are compile-time folded (zero overhead vs plain `isapprox`).
- **`Complex{<:AbstractFloat}`**: same, using `eps(real(T))`.
- **Other `_PromotableValue`** (Integer, Rational): `isapprox` with default tolerances.
- **Duck types** (Dual, SVector, ...): strict `==` (isapprox semantics not guaranteed).

!!! note "Scaled near-zero endpoints"
    `atol = 8eps` covers direct evaluations (e.g., `sin.(x)`), but not scaled
    variants (e.g., `1e6 .* sin.(x)` where noise ≈ 1e6·eps). For those cases,
    set `y[end] = y[1]` explicitly, or use `PeriodicBC(check=false)` to skip
    this validation.

Throws `ArgumentError` if endpoints differ.
"""
@inline function _check_periodic_endpoints(bc::PeriodicBC, y::AbstractVector)
    periodic_check(bc) || return nothing
    _check_periodic_endpoints(y)
    return nothing
end

@inline function _check_periodic_endpoints(y::AbstractVector{T}) where {T <: AbstractFloat}
    isapprox(first(y), last(y); atol = 8 * eps(T), rtol = sqrt(eps(T))) ||
        _throw_periodic_endpoint_error(first(y), last(y))
    return nothing
end

@inline function _check_periodic_endpoints(y::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    isapprox(first(y), last(y); atol = 8 * eps(T), rtol = sqrt(eps(T))) ||
        _throw_periodic_endpoint_error(first(y), last(y))
    return nothing
end

# Integer, Rational: isapprox with default tolerances (effectively ==)
@inline function _check_periodic_endpoints(y::AbstractVector{<:_PromotableValue})
    isapprox(first(y), last(y)) || _throw_periodic_endpoint_error(first(y), last(y))
    return nothing
end

# Duck-type fallback: strict == (isapprox not guaranteed for arbitrary types)
@inline function _check_periodic_endpoints(y::AbstractVector)
    _extract_primal(first(y)) == _extract_primal(last(y)) || _throw_periodic_endpoint_error(first(y), last(y))
    return nothing
end

@noinline function _throw_periodic_nd_slice_mismatch(d::Int)
    throw(
        ArgumentError(
            "PeriodicBC(endpoint=:inclusive) on dim $d: first and last slices of " *
                "`data` along that axis differ beyond tolerance. Either make the " *
                "endpoint slices match, switch to PeriodicBC(endpoint=:exclusive) " *
                "if your data does not repeat the first slice, or pass " *
                "PeriodicBC(check=false) on that axis to skip this validation."
        )
    )
end

# ND per-axis :inclusive slice equality check.
#
# Mirrors 1D `_check_periodic_endpoints(y)` but on the first vs last slice along
# axis `d`. `:exclusive` axes do not need this — extension constructs
# `data[...,n+1,...] = data[...,1,...]` by construction, so the check is
# trivially satisfied post-extension.
#
# Dispatch mirrors the 1D element-type split (Float / Complex / Integer-Rational /
# duck). Uses array-level `isapprox` (norm-based) for Float/Complex, element-wise
# `==` for the fallback. `atol = 8eps(T)` + `rtol = sqrt(eps(T))` matches 1D.
# Element-wise (NOT norm-based) comparison matching 1D scalar semantics.
# Array-`isapprox` accumulates near-zero noise across slice elements (e.g.
# `sin(0)` vs `sin(2π)` on a fine y/z mesh: per-element noise ~eps, slice norm
# ~√N·eps) and would reject legitimate periodic data.
#
# Using `all(splat, zip(...))` is stack-only and short-circuits on the first
# mismatch. A broadcasted `isapprox.(a, b; atol, rtol)` would allocate a
# BitArray and evaluate every element — strictly worse.

@inline function _check_periodic_slice_inclusive(data::AbstractArray{T}, d::Int) where {T <: AbstractFloat}
    n = size(data, d)
    first_s = selectdim(data, d, 1)
    last_s = selectdim(data, d, n)
    atol = 8 * eps(T)
    rtol = sqrt(eps(T))
    all(((a, b),) -> isapprox(a, b; atol = atol, rtol = rtol), zip(first_s, last_s)) ||
        _throw_periodic_nd_slice_mismatch(d)
    return nothing
end

@inline function _check_periodic_slice_inclusive(data::AbstractArray{Complex{T}}, d::Int) where {T <: AbstractFloat}
    n = size(data, d)
    first_s = selectdim(data, d, 1)
    last_s = selectdim(data, d, n)
    atol = 8 * eps(T)
    rtol = sqrt(eps(T))
    all(((a, b),) -> isapprox(a, b; atol = atol, rtol = rtol), zip(first_s, last_s)) ||
        _throw_periodic_nd_slice_mismatch(d)
    return nothing
end

@inline function _check_periodic_slice_inclusive(data::AbstractArray{<:_PromotableValue}, d::Int)
    n = size(data, d)
    first_s = selectdim(data, d, 1)
    last_s = selectdim(data, d, n)
    all(((a, b),) -> isapprox(a, b), zip(first_s, last_s)) ||
        _throw_periodic_nd_slice_mismatch(d)
    return nothing
end

# Fallback: strict element-wise ==
@inline function _check_periodic_slice_inclusive(data::AbstractArray, d::Int)
    n = size(data, d)
    first_s = selectdim(data, d, 1)
    last_s = selectdim(data, d, n)
    first_s == last_s || _throw_periodic_nd_slice_mismatch(d)
    return nothing
end

# Per-axis inclusive check with compile-time `d` and runtime `check` flag.
# No-op unless axis `d` is PeriodicBC{:inclusive} AND `check=true`.
@inline function _check_periodic_axis_nd(data, bcs::NTuple{N, AbstractBC}, ::Val{d}) where {N, d}
    @inbounds bc = bcs[d]
    bc isa PeriodicBC{:inclusive} || return nothing
    periodic_check(bc) || return nothing
    _check_periodic_slice_inclusive(data, d)
    return nothing
end

# Compile-time unroll over axes. Emits N straight-line calls so each `bcs[d]`
# indexes into the heterogeneous tuple at a compile-time-known position,
# avoiding runtime Union boxing (same pattern as `_extend_all_slices!`).
@generated function _validate_periodic_slices_nd(
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC},
        ::Val{N}
    ) where {Tv, N}
    calls = [:(_check_periodic_axis_nd(data, bcs, Val($d))) for d in 1:N]
    return Expr(:block, calls..., :(return nothing))
end

@noinline function _throw_periodic_endpoint_error(y1, yn)
    throw(
        ArgumentError(
            "PeriodicBC (inclusive endpoint) requires y[1] ≈ y[end], " *
                "got y[1]=$y1, y[end]=$yn. " *
                "Tip: set y[end] = y[1] explicitly, use " *
                "PeriodicBC(endpoint=:exclusive) if your data does not repeat the first point, " *
                "or PeriodicBC(check=false) to skip this validation."
        )
    )
end

@noinline function _throw_periodic_series_error(k, y_first, y_last)
    throw(
        ArgumentError(
            "PeriodicBC (inclusive endpoint) requires y[1] == y[end] for all series, " *
                "but series $k has y[1]=$y_first, y[end]=$y_last. " *
                "Tip: set y[end] = y[1] for each series, or use " *
                "PeriodicBC(endpoint=:exclusive) if your data does not repeat the first point."
        )
    )
end

"""
    _validate_series_endpoints(bc::PeriodicBC, y_mat::AbstractMatrix) -> Nothing

Series analog of `_check_periodic_endpoints`. For `:inclusive` BC with
`check=true`, validate that each column's first and last elements match
within the usual tolerance (dispatch-driven, reuses 1D helper on column
views). `:exclusive` is a no-op here because the extension phase writes
`y_mat[end, :] = y_mat[1, :]` by construction.

Used by Linear / Constant Series persistent + oneshot paths.
"""
@inline function _validate_series_endpoints(bc::PeriodicBC, y_mat::AbstractMatrix)
    periodic_check(bc) || return nothing
    bc isa PeriodicBC{:inclusive} || return nothing
    @inbounds for k in axes(y_mat, 2)
        _check_periodic_endpoints(bc, view(y_mat, :, k))
    end
    return nothing
end

@noinline function _throw_periodic_nd_error(d, v_first, v_last)
    throw(
        ArgumentError(
            "Periodic BC on dim $d requires data[1,...] ≈ data[end,...], " *
                "but found data[1,...]=$v_first, data[end,...]=$v_last. " *
                "Tip: set the last slice equal to the first along dim $d, " *
                "or use PeriodicBC(check=false) to skip this validation."
        )
    )
end

# ========================================
# Exclusive Endpoint Extension (1D)
# ========================================

"""
    _prepare_periodic(x, y, bc::PeriodicBC)

Prepare grid and values for periodic interpolation.
- Inclusive endpoint (default): no-op, returns `(x, y)` unchanged.
- Exclusive endpoint: extends `x` and `y` with a virtual endpoint at `x[1] + period`.

Called once at construction time before the periodic solver pipeline.
Uses dispatch on PeriodicBC{E} type parameter for type stability.
"""
@inline _prepare_periodic(x, y, ::PeriodicBC{:inclusive}) = (x, y)
@inline _prepare_periodic(x, y, bc::PeriodicBC{:exclusive}) = _extend_exclusive(x, y, bc)

"""
    _can_infer_period(x) -> Bool

Check if the period can be inferred from the grid (true for AbstractRange).
"""
@inline _can_infer_period(::AbstractRange) = true
@inline _can_infer_period(::AbstractVector) = false

"""
    _resolve_exclusive_period(x, bc::PeriodicBC)

Resolve the period for exclusive endpoint PeriodicBC:
- If `bc.period` is provided, cross-validate against Range inference when applicable.
- If `bc.period` is nothing, infer from Range or error for non-uniform grids.
"""
function _resolve_exclusive_period(x, bc::PeriodicBC)
    inferred = _can_infer_period(x) ? step(x) * length(x) : nothing

    if bc.period !== nothing
        # User provided period — cross-validate against Range inference.
        # Compare in grid precision (Tg) to avoid mixed-type ≈ using Float32's
        # generous rtol (~3e-4) when a Float32 period is given on a Float64 grid.
        # Duck-safe promotion: Integer/Rational grids lack a meaningful `eps`,
        # so lift to `float(Tg)` for the tolerance math. Duck grids (Dual,
        # Measurement, ...) keep their own eltype and use their own `eps`.
        Tg_raw = eltype(x)
        Tg = Tg_raw <: _PromotableValue ? float(Tg_raw) : Tg_raw
        if inferred !== nothing && !isapprox(Tg(bc.period), Tg(inferred); rtol = sqrt(eps(Tg)))
            x0 = first(x); x1 = x0 + inferred
            throw(
                ArgumentError(
                    "PeriodicBC's period=$(bc.period) conflicts with Range-inferred period = $x1 - $x0 = $inferred. " *
                        "Either adjust `period` or omit it for auto-inference."
                )
            )
        end
        return bc.period
    end

    # No user period — must infer from Range
    inferred !== nothing || throw(
        ArgumentError(
            "PeriodicBC(endpoint=:exclusive) requires `period` for non-uniform grids. " *
                "Use PeriodicBC(endpoint=:exclusive, period=T)."
        )
    )
    return inferred
end

"""
    _with_resolved_period(bc::PeriodicBC, period) -> PeriodicBC

Return a copy of `bc` with the resolved period baked in.
Used so that `itp.bc` always carries the actual period for display/introspection.
Uses the inner constructor directly to bypass keyword-constructor validation
(which rejects `period` for inclusive BCs).
"""
@inline _with_resolved_period(::PeriodicBC{E, <:Any, C}, period::T) where {E, T, C} =
    PeriodicBC{E, T, C}(period)

"""
    _extend_exclusive(x, y, bc::PeriodicBC) -> (x_ext, y_ext)

Extend grid and values for exclusive endpoint periodic data.
Appends a virtual endpoint at `x[1] + period` with value `y[1]`.
Preserves Range type for Range inputs (step consistency guaranteed by `_resolve_exclusive_period`).
"""
function _extend_exclusive(x::AbstractVector, y::AbstractVector, bc::PeriodicBC)
    period = _resolve_exclusive_period(x, bc)
    # Duck-safe grid promotion. `_PromotableValue` (Integer / AbstractFloat /
    # Rational / Complex) is promoted via `float(...)` so Int-typed Ranges can
    # form a valid `_CachedRange{Float}` (otherwise `inv(step::Int)::Float64`
    # cannot be stored in the `Int`-typed `inv_h` field → InexactError).
    # Duck grids (Dual, Measurement, SVector, ...) fall through with their
    # original type; they handle their own `inv`/arithmetic within their type.
    # Mirrors `_promote_grid_float` and `_check_periodic_endpoints` dispatch.
    Tg_raw = eltype(x)
    Tg = Tg_raw <: _PromotableValue ? float(Tg_raw) : Tg_raw
    x_end = first(x) + Tg(period)

    # Validate: virtual endpoint must be strictly after last grid point
    last(x) < x_end || throw(
        ArgumentError(
            "period=$period places virtual endpoint at $x_end, " *
                "not after last grid point x[end]=$(last(x))"
        )
    )

    # Type-stable grid extension: isa branch (compile-time narrowing) instead of
    # runtime ≈ check. _resolve_exclusive_period already validates period ≈ step(x)*length(x)
    # for Range grids, so Range → Range extension is always safe.
    x_ext = if x isa AbstractRange
        _to_float_adding_endpoint(x, Tg)
    else
        vcat(x, x_end)
    end
    y_ext = _extend_values(y)
    return x_ext, y_ext
end

# Matrix overload for CubicSeriesInterpolant
function _extend_exclusive(x::AbstractVector, y_mat::AbstractMatrix, bc::PeriodicBC)
    period = _resolve_exclusive_period(x, bc)
    # Duck-safe grid promotion — see the Vector overload above for rationale.
    Tg_raw = eltype(x)
    Tg = Tg_raw <: _PromotableValue ? float(Tg_raw) : Tg_raw
    x_end = first(x) + Tg(period)

    last(x) < x_end || throw(
        ArgumentError(
            "period=$period places virtual endpoint at $x_end, " *
                "not after last grid point x[end]=$(last(x))"
        )
    )

    x_ext = if x isa AbstractRange
        _to_float_adding_endpoint(x, Tg)
    else
        vcat(x, x_end)
    end
    y_ext = vcat(y_mat, y_mat[1:1, :])
    return x_ext, y_ext
end

# Value extension: append first element
_extend_values(y::AbstractVector) = vcat(y, first(y))

# ========================================
# BC-Aware WrapExtrap Constructors
# ========================================
#
# Constructor layering that replaces the old `_resolve_periodic_extrap` BC-factory
# family. Depends on bc_types.jl (AbstractBC / PeriodicBC), so these live in
# periodic.jl rather than eval_ops.jl (where the struct is declared).
#
# - `WrapExtrap(x, ::AbstractBC)` → delegate to `WrapExtrap(x)`. Covers NoBC,
#   PeriodicBC{:inclusive}, CubicFit, QuadraticFit, and any future BC — the grid
#   span already IS the wrap domain (inclusive) or BC is irrelevant to wrapping
#   (non-periodic).
# - `WrapExtrap(x, ::PeriodicBC{:exclusive, <:Real})` → `[first(x), first(x)+period)`,
#   since `x` has NOT been extended at zero-copy oneshot call sites.
# - `WrapExtrap(x::AbstractRange, ::PeriodicBC{:exclusive, Nothing})` → infer
#   period from `step(x) * length(x)`.
# - Non-Range + Nothing period → error.

@inline WrapExtrap(x::AbstractVector, ::AbstractBC) = WrapExtrap(x)

@inline function WrapExtrap(x::AbstractVector, bc::PeriodicBC{:exclusive, <:Real})
    x_min = first(x)
    # Route through `_resolve_exclusive_period` so Range grids get the same
    # `period ≈ step(x) * length(x)` cross-check that persistent paths rely on.
    # Without this, `linear_interp(range(0,step=0.1,length=10), y, q;
    # bc=PeriodicBC(:exclusive, period=2.0))` would silently accept a period
    # that disagrees with the grid's implied period — persistent throws on the
    # same input, so oneshot must too.
    period = _resolve_exclusive_period(x, bc)
    x_max = x_min + period
    # Virtual endpoint must lie strictly beyond the last grid point so the seam
    # cell [x[end], x_min+period] is non-empty and the grid covers at most one
    # period. Matches the contract formerly in `_periodic_extend_1d_pooled!`.
    last(x) < x_max || _throw_wrap_virtual_endpoint_error(period, x_max, last(x))
    T = typeof(x_max)
    return WrapExtrap{T}(T(x_min), T(x_max))
end

@inline function WrapExtrap(x::AbstractRange, ::PeriodicBC{:exclusive, Nothing})
    Tg_raw = eltype(x)
    Tg = Tg_raw <: _PromotableValue ? float(Tg_raw) : Tg_raw
    period = Tg(step(x)) * length(x)
    x_min = Tg(first(x))
    return WrapExtrap{Tg}(x_min, x_min + period)
end

WrapExtrap(::AbstractVector, ::PeriodicBC{:exclusive, Nothing}) =
    _throw_wrap_nonrange_period_error()

# Error helpers — `@noinline` keeps the `ArgumentError` formatting out of the
# happy-path inlined body (cold-path I-cache friendly). Mirrors the existing
# `_throw_periodic_*` pattern earlier in this file.
@noinline function _throw_wrap_virtual_endpoint_error(period, x_max, last_x)
    throw(
        ArgumentError(
            "period=$period places virtual endpoint at $x_max, " *
                "not after last grid point x[end]=$last_x"
        )
    )
end

@noinline _throw_wrap_nonrange_period_error() =
    throw(
    ArgumentError(
        "PeriodicBC(:exclusive) requires explicit `period` for non-Range grid"
    )
)

# ========================================
# Extrap Resolution (_resolve_extrap)
# ========================================
#
# Single name for every extrap materialization / validation step. `extrap` is
# always the 1st arg — consistent with `_resolve_search`, `_resolve_coeffs`,
# `_resolve_grididx` naming family. Layers (dispatched by arg shape):
#
# 1D primitives (per-axis):
#   (extrap, x)                          — grid-only; upgrade {Nothing} or passthrough
#   (extrap, bc, x)                      — BC-aware; PeriodicBC forces WrapExtrap
#
# 1D bundled: validate + materialize
#   (extrap, bc, x, y)                   — `:inclusive` endpoint check + primitive
#
# ND: expand + promote + per-axis materialize (1-line for pre-extension / adjoints)
#   (extrap, bcs, grids, Val(N), Tv)     — with BCs → per-axis 3-arg primitive
#   (extrap, ::Nothing, grids, Val(N), Tv) — no BCs → per-axis 2-arg primitive
#
# ND expand-only (no materialize — post-extension 2-step callers):
#   (extrap, bcs, Val(N), Tv)            — same shape as old `_resolve_extrap_nd`
#
# ND bundled (oneshot): slice validation + materialize
#   (extraps, bcs, grids, data, Val(N))  — zero-copy ND oneshot entry
#
# Kernels only ever see fully-materialized `WrapExtrap{T}`; the `{Nothing}`
# variant is a build-time-only placeholder that never reaches query paths.

# ── Primitive: 2-arg (no BC info; for Hermite family + non-periodic persistent) ──
@inline _resolve_extrap(::WrapExtrap{Nothing}, x::AbstractVector) = WrapExtrap(x)
@inline _resolve_extrap(extrap::AbstractExtrap, ::AbstractVector) = extrap

# ── Primitive: 3-arg (grid + value type — composes WrapExtrap upgrade with FillExtrap promote) ──
# Collapses the common `_resolve_extrap(extrap, x)` + `_promote_extrap(·, Tv)` pair at
# 1D persistent-interpolant entries. Orthogonal sub-operations:
#   - `_resolve_extrap(·, x)`      : WrapExtrap{Nothing} → WrapExtrap{T} (grid-span)
#   - `_promote_extrap(·, Tv)`     : FillExtrap{Int}(v) → FillExtrap{Tv}(convert(Tv, v))
# Both passthrough for extraps that don't match their respective trigger.
@inline _resolve_extrap(extrap, x::AbstractVector, ::Type{Tv}) where {Tv} =
    _promote_extrap(_resolve_extrap(extrap, x), Tv)

# ── Primitive: 3-arg (BC-aware) ──
# PeriodicBC forces WrapExtrap regardless of user's extrap. The explicit
# `WrapExtrap{Nothing}` tiebreaker resolves the ambiguity between the
# "PeriodicBC forces" rule and the "Nothing upgrade" rule when both match.
@inline _resolve_extrap(::WrapExtrap{Nothing}, bc::PeriodicBC, x::AbstractVector) = WrapExtrap(x, bc)
@inline _resolve_extrap(::AbstractExtrap, bc::PeriodicBC, x::AbstractVector) = WrapExtrap(x, bc)
@inline _resolve_extrap(::WrapExtrap{Nothing}, ::AbstractBC, x::AbstractVector) = WrapExtrap(x)
@inline _resolve_extrap(extrap::AbstractExtrap, ::AbstractBC, ::AbstractVector) = extrap

# ── 1D Bundled: validate + materialize ──
"""
    _resolve_extrap(extrap, bc, x, y) -> AbstractExtrap

Zero-copy 1D oneshot entry: validate `:inclusive` endpoint (`y[1] ≈ y[end]`) when
required, then hand off to the BC-aware 3-arg primitive.
"""
@inline function _resolve_extrap(
        extrap::AbstractExtrap, bc::AbstractBC,
        x::AbstractVector, y::AbstractVector
    )
    bc isa PeriodicBC{:inclusive} && _check_periodic_endpoints(bc, y)
    return _resolve_extrap(extrap, bc, x)
end

# ── ND Bundled (oneshot): per-axis slice validation + materialize ──
"""
    _resolve_extrap(extraps, bcs, grids, data, Val(N)) -> NTuple{N, AbstractExtrap}

Zero-copy ND oneshot entry: validate `:inclusive` axes' first/last slice
equality on `data`, then materialize per-axis via the 3-arg primitive.
"""
@inline function _resolve_extrap(
        extraps::NTuple{N, AbstractExtrap},
        bcs::NTuple{N, AbstractBC},
        grids::NTuple{N, AbstractVector},
        data::AbstractArray,
        ::Val{N},
    ) where {N}
    _validate_periodic_slices_nd(data, bcs, Val(N))
    return map(_resolve_extrap, extraps, bcs, grids)
end

"""
    _periodic_extend_1d(x, y, bc, extrap) -> (x_eff, y_eff, extrap_eff)

Non-pool 1D periodic dispatch for the **persistent-interpolant path**.
Returns a single `(grid, values, extrap)` triple whether or not `bc` is
periodic — callers invoke this once and feed the result into their normal
build flow without branching on `_is_periodic_bc`:

- Non-periodic `bc` → `(x, y, extrap)` passthrough.
- `PeriodicBC{:inclusive}` → `(x, y, typed WrapExtrap)` (validates `y[1] ≈ y[end]`).
- `PeriodicBC{:exclusive}` → `(x_ext, y_ext, typed WrapExtrap)` where the grid/values
  are extended by one virtual endpoint via `_extend_exclusive` (heap copy
  consistent with existing non-periodic persistent-path copy semantics).

Intended consumers: Linear, Constant, and (future Phase 2/3) PCHIP/Cardinal/
Akima/Quadratic persistent interpolant constructors. Cubic stays on its
dedicated `_build_interpolant_periodic` path because its solver branches on
periodicity, not just on the grid representation.
"""
@inline function _periodic_extend_1d(
        x::AbstractVector,
        y::AbstractVector,
        bc::AbstractBC,
        extrap::AbstractExtrap
    )
    if _is_periodic_bc(bc)
        x_ext, y_ext = _prepare_periodic(x, y, bc)
        # Endpoint validation is meaningful only for `:inclusive` — `:exclusive`
        # sets `y_ext[end] = y_ext[1]` by construction so the check is trivially true.
        bc isa PeriodicBC{:inclusive} && _check_periodic_endpoints(bc, y_ext)
        # After extension, `last(x_ext) - first(x_ext) == period`, so the grid-span
        # `WrapExtrap(x_ext)` already carries the correct wrap domain — no need
        # for the BC-aware constructor here.
        return x_ext, y_ext, WrapExtrap(x_ext)
    end
    # Non-periodic: still materialize in case the user passed `WrapExtrap()` (legacy
    # singleton); `_resolve_extrap` upgrades it to `WrapExtrap(x)` and leaves
    # other extraps untouched.
    return x, y, _resolve_extrap(extrap, bc, x)
end

"""
    _periodic_extend_1d_pooled!(pool, x, y, bc, extrap) -> (x_eff, y_eff, extrap_eff)

Pool-based sibling of `_periodic_extend_1d` for the **one-shot hot path**.
Same contract as the non-pool variant, but the `:exclusive` extension uses
`acquire!(pool, …)` for the extended buffers (zero-alloc after warmup, caller's
`@with_pool` scope owns the lifetime).

Range grids are extended to `_CachedRange` via `_to_float_adding_endpoint`
(heap-free); Vector grids go through the pool. Value buffers are always
pool-acquired when extending.

Intended consumers: Linear, Constant, and (future Phase 2/3) PCHIP/Cardinal/
Akima/Quadratic oneshot paths. Cubic oneshot uses this via
`_cubic_periodic_solve!`, which wraps the extension + tridiagonal solve.
"""
@inline function _periodic_extend_1d_pooled!(
        pool::AbstractArrayPool,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::AbstractBC,
        extrap::AbstractExtrap
    ) where {Tg, Tv}
    if !_is_periodic_bc(bc)
        # Non-periodic: still materialize in case user passed `WrapExtrap()` singleton.
        return x, y, _resolve_extrap(extrap, bc, x)
    end
    # Duck-safe extension buffer type: promote Integer / Complex{Integer} grids
    # to float so the extended pool buffer and `_CachedRange` `inv_h` field can
    # hold `inv(step)`. Duck grids (Dual, Measurement, ...) keep their type so
    # AD / uncertainty chains survive. Tv (value type) is never promoted here —
    # user y-vector semantics are preserved.
    Tg_ext = Tg <: _PromotableValue ? float(Tg) : Tg
    if bc isa PeriodicBC{:exclusive}
        period = _resolve_exclusive_period(x, bc)
        x_end = first(x) + Tg_ext(period)
        last(x) < x_end || _throw_wrap_virtual_endpoint_error(period, x_end, last(x))
        n = length(x)
        if x isa AbstractRange
            x_p = _to_float_adding_endpoint(x, Tg_ext)
        else
            x_p = acquire!(pool, Tg_ext, n + 1)
            @inbounds copyto!(x_p, 1, x, 1, n)
            @inbounds x_p[n + 1] = x_end
        end
        y_p = acquire!(pool, Tv, n + 1)
        @inbounds copyto!(y_p, 1, y, 1, n)
        @inbounds y_p[n + 1] = y[1]
    else
        x_p, y_p = x, y
    end
    # `:exclusive` path constructs `y_p[end] = y_p[1]` by extension, so the check
    # is trivially satisfied. Run validation only for `:inclusive`.
    bc isa PeriodicBC{:inclusive} && _check_periodic_endpoints(bc, y_p)
    # After extension (or no-op for inclusive), the extended grid span IS the wrap
    # domain — use `WrapExtrap(x_p)` directly, no BC-aware construction needed.
    return x_p, y_p, WrapExtrap(x_p)
end

"""
    _prepare_1d_oneshot!(pool, x, y, bc, extrap) -> (x_eff, y_eff, extrap_eff)

Thin oneshot-API convenience that fuses `_prepare_grid(x)` with
`_periodic_extend_1d_pooled!(pool, …, bc, extrap)`. Call once per 1D oneshot
entry point; the return triple feeds directly into searcher + eval kernel.

Separating this from `_periodic_extend_1d_pooled!` keeps the latter's name
semantically accurate — it really only does work on periodic `bc` — while
letting user-facing oneshot entry points use a single, non-misleading call.
"""
@inline _prepare_1d_oneshot!(pool, x, y, bc, extrap) =
    _periodic_extend_1d_pooled!(pool, _prepare_grid(x), y, bc, extrap)

# ========================================
# ND Exclusive Endpoint Extension
# ========================================

# Compile-time predicate "does any axis have a BC of type T?" on the `bcs` tuple.
# `@generated` inspects the concrete tuple type at specialization time and
# collapses to `true`/`false`, so non-periodic callers (CubicFit, QuadraticFit,
# NoBC, ...) pay ZERO runtime cost — no loop, no per-axis check, no validation.
# Crucial for the ND oneshot hot path where `_prepare_periodic_nd_pooled` is
# called per query.
#
# Callers pass `::Type{T}` as a `Val{T}`-like singleton so the type parameter
# is visible to the generator. Example:
#   _has_any_bc(bcs, Val(N), PeriodicBC)              # any periodic axis?
#   _has_any_bc(bcs, Val(N), PeriodicBC{:exclusive})  # any exclusive axis?
@generated function _has_any_bc(
        bcs::NTuple{N, AbstractBC}, ::Val{N}, ::Type{T}
    ) where {N, T}
    for d in 1:N
        fieldtype(bcs, d) <: T && return :(true)
    end
    return :(false)
end

"""
    _prepare_periodic_nd(grids, data, bcs) -> (grids_ext, data_ext, bcs_resolved)

Prepare N-dimensional grids and data for periodic interpolation.

For each axis with `PeriodicBC(endpoint=:exclusive)`, extends the grid and data
along that dimension by appending a virtual endpoint (same pattern as 1D `_prepare_periodic`).
Axes with inclusive or non-periodic BCs are left unchanged.

Called once at build time before `_build_nd_coeffs`.

# Returns
- `grids_ext`: Grids with exclusive axes extended (Range type preserved when possible)
- `data_ext`: Data with first slice appended along each exclusive axis
- `bcs_resolved`: BCs with resolved period for exclusive axes (for display/introspection)
"""
function _prepare_periodic_nd(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC}
    ) where {Tg, Tv, N}
    return _prepare_periodic_nd_impl(grids, data, bcs, _extend_grid_vcat, _allocate_array)
end

# Heap-based Vector grid extender (build-time; non-pool). Range axes never reach
# this callback — they're handled inline via `_to_float_adding_endpoint`.
@inline _extend_grid_vcat(grid_d::AbstractVector, x_end, ::Type{Tg}) where {Tg} =
    vcat(grid_d, x_end)

# Heap-based data allocator. `final_sizes` is NTuple{N, Int}; `Tv` is value type.
@inline _allocate_array(final_sizes::NTuple{N, Int}, ::Type{Tv}) where {N, Tv} =
    Array{Tv, N}(undef, final_sizes)

# ────────────────────────────────────────────────────────
# Shared core for `_prepare_periodic_nd` and `_prepare_periodic_nd_pooled`.
# Both variants share the ultra-fast path, validation, per-axis grid+bc
# resolution, and data copy/fill pattern. They differ ONLY in:
#   1. Vector-axis grid extension: vcat vs pool-acquire
#   2. Data allocation: `Array(undef, ...)` vs pool-acquire
# Injected via two callbacks (callable structs for the pool variant so the
# pool reference is carried via a type parameter — concrete dispatch, no Box).
# ────────────────────────────────────────────────────────
@inline function _prepare_periodic_nd_impl(
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC},
        extend_vector_grid::F_ext,
        allocate_data::F_alloc,
    ) where {Tg, Tv, N, F_ext, F_alloc}
    # Ultra-fast path: non-periodic call-sites collapse this to a no-op via
    # the `@generated` predicate (zero runtime cost on the hot non-periodic path).
    _has_any_bc(bcs, Val(N), PeriodicBC) || return (grids, data, bcs)
    # Validate :inclusive endpoint matching (:exclusive extension is constructive,
    # so no pre-validation needed for exclusive axes).
    _validate_periodic_slices_nd(data, bcs, Val(N))
    # Fast path: purely inclusive → no extension needed.
    _has_any_bc(bcs, Val(N), PeriodicBC{:exclusive}) || return (grids, data, bcs)

    # Per-axis grid extension + bc resolution. `map` with `ntuple(identity, Val(N))`
    # dispatches per-element with concrete (grid, bc) types → each closure call
    # compiles to a specialization with concrete return type. A `do d …` with
    # runtime indexing `bcs[d]` / `grids[d]` over a heterogeneous tuple would
    # leave the return type as `Union{...}` → heap-boxed Refs (~2 KB/query).
    # See MEMORY.md "ND Constructor Inferrability Pattern".
    processed = map(ntuple(identity, Val(N)), grids, bcs) do d, grid_d, bc_d
        bc_d isa PeriodicBC{:exclusive} || return (grid_d, bc_d)
        period = _resolve_exclusive_period(grid_d, bc_d)
        x_end = first(grid_d) + Tg(period)
        last(grid_d) < x_end ||
            _throw_prepare_periodic_nd_endpoint(d, period, x_end, last(grid_d))
        # Range axes use type-preserving `_to_float_adding_endpoint`; Vector axes
        # delegate to the caller-supplied `extend_vector_grid` (vcat vs pool).
        grid_ext = grid_d isa AbstractRange ?
            _to_float_adding_endpoint(grid_d, Tg) :
            extend_vector_grid(grid_d, x_end, Tg)
        return (grid_ext, _with_resolved_period(bc_d, period))
    end
    grids_out = map(first, processed)
    bcs_out = map(last, processed)

    # Extend data: caller-supplied allocator + shared fill. `_extend_all_slices!`
    # is @generated so the per-axis `d` is literal at each call — avoids the
    # 48 B/query Union box that a runtime `for d in 1:N` loop produced on
    # heterogeneous axis configurations.
    final_sizes = ntuple(Val(N)) do d
        bcs[d] isa PeriodicBC{:exclusive} ? size(data, d) + 1 : size(data, d)
    end
    data_out = allocate_data(final_sizes, Tv)
    orig_inds = ntuple(d -> 1:size(data, d), Val(N))
    copyto!(view(data_out, orig_inds...), data)
    _extend_all_slices!(data_out, data, bcs, Val(N))

    return (grids_out, data_out, bcs_out)
end

# Cold-path error body (kept out of the happy-path inlined code).
@noinline function _throw_prepare_periodic_nd_endpoint(d, period, x_end, last_x)
    throw(
        ArgumentError(
            "PeriodicBC(endpoint=:exclusive) on dim $d: period=$period places " *
                "virtual endpoint at $x_end, not after last grid point x[end]=$last_x"
        )
    )
end

# ========================================
# Pool-Based ND Exclusive Endpoint Extension
# ========================================

"""
    _prepare_periodic_nd_pooled(pool, grids, data, bcs) -> (grids_ext, data_ext, bcs_resolved)

Pool-based variant of `_prepare_periodic_nd` for zero-allocation one-shot evaluation.

All temporary arrays (extended grids, extended data) are acquired from the pool
via `acquire!`, so they must NOT escape the enclosing `@with_pool` scope.

# Safety
- Extended data is consumed by `_compute_nd_partials!` within the same pool scope
- Extended grids are used for spacing/search within the same pool scope
- Pool rewind in the outer `@with_pool` automatically releases all buffers
"""
@inline function _prepare_periodic_nd_pooled(
        pool::AbstractArrayPool,
        grids::NTuple{N, AbstractVector{Tg}},
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC}
    ) where {Tg, Tv, N}
    return _prepare_periodic_nd_impl(
        grids, data, bcs,
        _PoolGridExtender(pool),
        _PoolDataAllocator(pool),
    )
end

# Callable structs carry the pool as a type-parameterized field, so Julia
# dispatches with concrete `_PoolGridExtender{P}` / `_PoolDataAllocator{P}` —
# no runtime boxing of the `pool::AbstractArrayPool` abstract reference.
struct _PoolGridExtender{P <: AbstractArrayPool}
    pool::P
end
@inline function (e::_PoolGridExtender)(grid_d::AbstractVector, x_end, ::Type{Tg}) where {Tg}
    n = length(grid_d)
    g_ext = acquire!(e.pool, Tg, n + 1)
    @inbounds copyto!(g_ext, 1, grid_d, 1, n)
    @inbounds g_ext[n + 1] = x_end
    return g_ext
end

struct _PoolDataAllocator{P <: AbstractArrayPool}
    pool::P
end
@inline (a::_PoolDataAllocator)(final_sizes::NTuple{N, Int}, ::Type{Tv}) where {N, Tv} =
    acquire!(a.pool, Tv, final_sizes)

# Per-axis slice extension with compile-time dimension index `d`.
@inline function _extend_slice_along!(
        data_out::AbstractArray{Tv, N},
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC},
        ::Val{d},
        ::Val{N}
    ) where {Tv, N, d}
    bcs[d] isa PeriodicBC{:exclusive} || return nothing
    nd = size(data, d)
    cur_ranges = ntuple(Val(N)) do i
        if i == d
            1:1
        elseif i < d && bcs[i] isa PeriodicBC{:exclusive}
            1:(size(data, i) + 1)
        else
            1:size(data, i)
        end
    end
    src_inds = ntuple(i -> i == d ? (1:1) : cur_ranges[i], Val(N))
    dst_inds = ntuple(i -> i == d ? ((nd + 1):(nd + 1)) : cur_ranges[i], Val(N))
    copyto!(view(data_out, dst_inds...), view(data_out, src_inds...))
    return nothing
end

# Compile-time unroll of the per-axis extension loop. Emits N straight-line calls
# to `_extend_slice_along!` with literal `Val(d)` — no runtime loop, no tuple of
# Nothing returns, no closure capturing a runtime index.
@generated function _extend_all_slices!(
        data_out::AbstractArray{Tv, N},
        data::AbstractArray{Tv, N},
        bcs::NTuple{N, AbstractBC},
        ::Val{N}
    ) where {Tv, N}
    calls = [:(_extend_slice_along!(data_out, data, bcs, Val($d), Val(N))) for d in 1:N]
    return Expr(:block, calls..., :(return nothing))
end
