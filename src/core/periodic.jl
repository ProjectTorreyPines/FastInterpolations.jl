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

Wrap a query point `xi` to the domain [x_min, x_max].
Used for periodic boundary conditions and extrap=WrapExtrap().

Closed-domain convention: `xi == x_max` is an in-domain boundary query
(returns `xi` unchanged); only strictly-OOB queries (`xi < x_min` or
`xi > x_max`) take the cold `mod()` path. `PeriodicBC{:inclusive}` is
forward-**value**-invariant because `y[1] ≈ y[end]` by construction; the
adjoint sensitivity at the seam now scatters to slot `n` instead of slot `1`
(delta-equivalent under the cycle constraint). `:exclusive` is fully invariant
(forward + adjoint) because `_ExclusivePeriodicAxis.search_interval` already
returns `idx_R = 1` for `xq >= inner[n]` at the seam.

Optimized: skips expensive `mod()` when xi is already in domain.
"""
@inline function _wrap_to_domain(xi::Tg, x_min::Tg, x_max::Tg) where {Tg}
    # Hot path: already in domain — return as-is (no arith). Cold `mod()`
    # work goes through `@noinline` `_wrap_to_domain_slow` so it doesn't
    # bloat the caller (every WrapExtrap eval kernel) with mod-related
    # asm. On constant rng+perEx persistent (3-4 ns baseline, 138 lines
    # before split), this collapses the eval kernel to ~75 lines.
    if (xi >= x_min) && (xi <= x_max)
        return xi
    end
    return _wrap_to_domain_slow(xi, x_min, x_max)
end

# Generic wrapper: handles Dual, Int, Float32 on Float64 grid, etc.
# IMPORTANT: Preserves AD Dual type through the entire operation.
# Same hot/cold split as the AbstractFloat overload above.
@inline function _wrap_to_domain(xi::Real, x_min::Tg, x_max::Tg) where {Tg}
    xi_primal = _extract_primal(xi)
    # Fast path: already in domain, return original xi (preserves Dual type for AD)
    if (xi_primal >= x_min) && (xi_primal <= x_max)
        return xi
    end
    return _wrap_to_domain_slow(xi, x_min, x_max)
end

# Cold path — `mod()` work hoisted out of the inlined hot path.
# `mod()` works correctly with `ForwardDiff.Dual`: d/dx[mod(x,p)] = 1.
@noinline function _wrap_to_domain_slow(xi, x_min, x_max)
    period = x_max - x_min
    return x_min + mod(xi - x_min, period)
end

# ────────────────────────────────────────────────────────
# Axis-aware 2-arg wrapper (axis as single source of truth)
# ────────────────────────────────────────────────────────
#
# After the surface-API axis resolution (`_resolve_axis` / `_cache_axis`),
# every supported axis exposes `first/last` matching the canonical wrap
# domain — including `_ExclusivePeriodicAxis`, whose `last` reports the
# virtual endpoint `inner[1] + period`. With `WrapExtrap` reduced to a tag
# struct (no `_x_min/_x_max` fields), the axis IS the wrap-domain context —
# no extrap parameter needed for this helper. Callers know they want to
# wrap because they're inside a `WrapExtrap`-dispatched eval branch.
#
# Arg order `(xq, x)`: matches the 3-arg primitive `(xq, x_min, x_max)` —
# the operand always comes first; axis bounds (or extracted bounds) follow.
@inline _wrap_to_domain(xq, x::AbstractVector) =
    _wrap_to_domain(xq, first(x), last(x))
# Wrapper-specific overload lives in `periodic_axis.jl` where
# `_ExclusivePeriodicAxis` is defined.

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
@inline function _check_periodic_endpoints(bc::PeriodicBC{:inclusive}, y::AbstractVector)
    periodic_check(bc) || return nothing
    _check_periodic_endpoints(y)
    return nothing
end

# `:extended` is constructed by `_bc_after_extend` with `C=false` pinned, so
# `periodic_check(bc)` is always `false` and validation would be a no-op.
@inline _check_periodic_endpoints(::PeriodicBC{:extended}, ::AbstractVector) = nothing

# `:exclusive` form has no endpoint-matching constraint: the user provides n
# distinct samples and the seam is virtual (constructed from `bc.period`).
# Calling `_check_periodic_endpoints(y)` on raw exclusive y would falsely
# reject valid data (e.g., `y = sin.(2π .* x_excl)` where `y[1] ≠ y[n]`).
@inline _check_periodic_endpoints(::PeriodicBC{:exclusive}, ::AbstractVector) = nothing

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
    _prepare_periodic_grid(x, bc) -> x_ext

x-only sibling of `_prepare_periodic` for data-free callers (adjoint
operators that don't carry per-grid data). Mirrors the extension policy
on the x side exactly:
- `PeriodicBC{:inclusive}`  → passthrough (already closed-cycle).
- `PeriodicBC{:exclusive}`  → length-(n+1) extension via `_extend_exclusive`'s
                              x-side validation + `vcat` / Range endpoint append.
"""
@inline _prepare_periodic_grid(x, ::AbstractBC) = x          # NoBC / non-periodic — passthrough
@inline _prepare_periodic_grid(x, ::PeriodicBC{:inclusive}) = x

@inline function _prepare_periodic_grid(x::AbstractVector, bc::PeriodicBC{:exclusive})
    period = _resolve_exclusive_period(x, bc)
    _validate_exclusive_period(x, period)
    # Shape-only extension — eltype preserved. Grid Tg is set downstream by
    # `_cache_axis(x_ext, bc_eff, Tg)` (single source of truth).
    T = eltype(x)
    x_end = first(x) + T(period)
    last(x) < x_end || _throw_excl_endpoint_too_small(period, x_end, last(x))
    return x isa AbstractRange ?
        range(first(x); step = step(x), length = length(x) + 1) :
        vcat(x, x_end)
end

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
    # Wrapped axes already carry the resolved period — no re-resolution needed.
    # Without this, downstream callers that wrap once (oneshot surface) and then
    # invoke periodic slope helpers would re-enter `_resolve_exclusive_period`
    # on the wrapper, where `_can_infer_period(::_ExclusivePeriodicAxis)` is
    # false, and erroneously raise on Vector-inner wrappers.
    x isa _ExclusivePeriodicAxis && return x.period

    # User-provided period: trust it. Cross-validation against Range-inferred
    # `step(x) * length(x)` is performed once at `_ExclusivePeriodicAxis`
    # construction (see `_validate_exclusive_period` in periodic_axis.jl) — no
    # per-call validation tax in hot resolve paths (oneshot / search).
    bc.period !== nothing && return bc.period

    # No user period — must infer from Range
    _can_infer_period(x) || _throw_exclusive_period_required()
    return step(x) * length(x)
end

@noinline _throw_exclusive_period_required() = throw(
    ArgumentError(
        "PeriodicBC(endpoint=:exclusive) requires `period` for non-uniform grids. " *
            "Use PeriodicBC(endpoint=:exclusive, period=T)."
    )
)

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
    # Cross-validate user-supplied period against Range-inferred (Range only;
    # Vector trust). Mirrors the validation done in `_ExclusivePeriodicAxis`
    # constructor for the wrapper-using paths (linear/constant). Cubic's
    # extension-based path doesn't construct a wrapper, so we explicitly
    # invoke the same check here so `bc.period` mismatches are caught
    # uniformly across all method families.
    _validate_exclusive_period(x, period)
    # Shape-only extension — eltype preserved. Grid Tg is set downstream by
    # `_cache_axis(x_ext, bc_eff, Tg)` (single source of truth). `T(period)`
    # throws `InexactError` on incompatible user period (e.g. Int range +
    # irrational period) instead of silently widening to Float64.
    T = eltype(x)
    x_end = first(x) + T(period)

    # Sanity: virtual endpoint must be strictly after last grid point. Catches
    # a too-small period even when Vector grids skipped the cross-validation.
    last(x) < x_end || _throw_excl_endpoint_too_small(period, x_end, last(x))

    # Type-stable grid extension: isa branch (compile-time narrowing) instead of
    # runtime ≈ check. _resolve_exclusive_period already validates period ≈ step(x)*length(x)
    # for Range grids, so Range → Range extension is always safe.
    x_ext = if x isa AbstractRange
        range(first(x); step = step(x), length = length(x) + 1)
    else
        vcat(x, x_end)
    end
    y_ext = _extend_values(y)
    return x_ext, y_ext
end

@noinline _throw_excl_endpoint_too_small(period, x_end, last_x) = throw(
    ArgumentError(
        "period=$period places virtual endpoint at $x_end, " *
            "not after last grid point x[end]=$last_x"
    )
)

# Matrix overload for CubicSeriesInterpolant
function _extend_exclusive(x::AbstractVector, y_mat::AbstractMatrix, bc::PeriodicBC)
    period = _resolve_exclusive_period(x, bc)
    _validate_exclusive_period(x, period)
    # Eltype-preserving — see Vector overload for rationale.
    T = eltype(x)
    x_end = first(x) + T(period)

    last(x) < x_end || _throw_excl_endpoint_too_small(period, x_end, last(x))

    x_ext = if x isa AbstractRange
        range(first(x); step = step(x), length = length(x) + 1)
    else
        vcat(x, x_end)
    end
    y_ext = vcat(y_mat, y_mat[1:1, :])
    return x_ext, y_ext
end

# Value extension: append first element
_extend_values(y::AbstractVector) = vcat(y, first(y))

# ========================================
# WrapExtrap is a tag struct (eval_ops.jl)
# ========================================
#
# After the surface-API axis resolution (`_resolve_axis` / `_cache_axis`),
# every supported axis exposes `first/last` matching the canonical wrap
# domain — including `_ExclusivePeriodicAxis`, whose `last` reports the
# precomputed virtual endpoint. Eval kernels read those bounds directly
# from the axis via `_wrap_to_domain(xq, x, ::WrapExtrap)`. No BC-aware
# `WrapExtrap` constructors, no `WrapExtrap{Nothing}` materialization, no
# duplicated `_x_min/_x_max` fields.

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
# `WrapExtrap` is a tag struct — no `{Nothing}` placeholder, no materialization.
# The axis carries the canonical wrap domain via `first/last` after the
# surface-API axis resolution.
#
# 1D entries (per-axis):
#   (extrap, x)                          — passthrough + FillExtrap promote (no Tv → identity)
#   (extrap, bc, x)                      — BC-aware: PeriodicBC forces WrapExtrap, otherwise passthrough
#   (extrap, x, Tv)                      — FillExtrap promote (eltype → Tv)
#
# 1D bundled: validate + dispatch
#   (extrap, bc, x, y)                   — `:inclusive` endpoint check + primitive
#
# ND bundled (oneshot): slice validation + per-axis materialize
#   (extraps, bcs, grids, data, Val(N))  — zero-copy ND oneshot entry

# ── Primitive: 2-arg (no BC info; non-periodic persistent / Hermite family) ──
@inline _resolve_extrap(extrap::AbstractExtrap, ::AbstractVector) = extrap

# ── Primitive: 3-arg (grid + value type — FillExtrap promote on persistent path) ──
# `_promote_extrap(·, Tv)` only acts on `FillExtrap{Int}(v)` → `FillExtrap{Tv}(convert(Tv, v))`;
# passthrough for everything else.
@inline _resolve_extrap(extrap, x::AbstractVector, ::Type{Tv}) where {Tv} =
    _promote_extrap(extrap, Tv)

# ── Primitive: 3-arg (BC-aware) ──
# PeriodicBC forces WrapExtrap regardless of the user's extrap; otherwise passthrough.
@inline _resolve_extrap(::AbstractExtrap, ::PeriodicBC, ::AbstractVector) = WrapExtrap()
@inline _resolve_extrap(extrap::AbstractExtrap, ::AbstractBC, ::AbstractVector) = extrap

# ── 1D Bundled: validate + dispatch ──
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
    _periodic_extend_1d(x, y, bc, extrap) -> (x_eff, y_eff, bc_eff, extrap_eff)

Non-pool 1D periodic dispatch for the **persistent-interpolant path**.
Fuses the closed-cycle grid extension with the BC normalization that always
follows it — callers invoke this once and destructure the 4-tuple, feeding
each piece into their normal build flow without branching on `_is_periodic_bc`:

- Non-periodic `bc` → `(x, y, bc, extrap)` passthrough.
- `PeriodicBC{:inclusive}` → `(x, y, bc, typed WrapExtrap)`
  (validates `y[1] ≈ y[end]`; layout already closed-cycle).
- `PeriodicBC{:exclusive}` → `(x_ext, y_ext, PeriodicBC{:extended,...}, typed WrapExtrap)`
  where the grid/values are extended by one virtual endpoint via
  `_extend_exclusive` (heap copy consistent with existing non-periodic
  persistent-path copy semantics), and `bc_eff` flips to `:extended` to
  record the internal layout promotion (see `_bc_after_extend`).
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
        # Bake resolved period into bc_eff (only for `:exclusive`; other branches
        # compile-time-eliminated) so callers storing `bc_eff` retain the inferred
        # period for introspection. `WrapExtrap` is a tag struct.
        bc_eff = _bc_after_extend(bc)
        bc isa PeriodicBC{:exclusive} &&
            (bc_eff = _with_resolved_period(bc_eff, _resolve_exclusive_period(x, bc)))
        return x_ext, y_ext, bc_eff, WrapExtrap()
    end
    return x, y, bc, extrap
end

# ────────────────────────────────────────────
# BC normalization after grid extension
# ────────────────────────────────────────────
# `_periodic_extend_1d` and `_prepare_periodic_nd_impl` produce a length-(n+1)
# closed-cycle grid for `:exclusive` input. After extension, dispatch should
# reflect the new layout:
#   - `:inclusive` passes through (user's data was already closed-cycle).
#   - `:exclusive` swaps to `:extended` — records that the library promoted
#     the layout, while preserving the period for adjoint output sizing and
#     introspection. `C=false` because the extension constructs a bit-exact
#     seam (`y[end] = y[1]`), so endpoint validation would be a noop.
#   - Non-periodic BCs pass through unchanged.
# See claudedocs/design/bc_extended_symbol.md §4 + §5.2 for the full rationale.
@inline _bc_after_extend(bc::AbstractBC) = bc
@inline _bc_after_extend(bc::PeriodicBC{:inclusive}) = bc
@inline _bc_after_extend(bc::PeriodicBC{:exclusive}) =
    PeriodicBC{:extended, typeof(bc.period), false}(bc.period)

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
    _prepare_periodic_nd(grids, data, bcs) -> (grids_ext, data_ext, bcs_post_extend)

Prepare N-dimensional grids and data for periodic interpolation.

For each axis with `PeriodicBC(endpoint=:exclusive)`, extends the grid and data
along that dimension by appending a virtual endpoint (same pattern as 1D `_prepare_periodic`).
Axes with inclusive or non-periodic BCs are left unchanged.

Called once at build time before `_build_nd_coeffs`.

# Returns
- `grids_ext`: Grids with exclusive axes extended (Range type preserved when possible)
- `data_ext`: Data with first slice appended along each exclusive axis
- `bcs_post_extend`: Per-axis BCs after extension. Exclusive periodic axes are
  promoted to `PeriodicBC{:extended}` via `_bc_after_extend` (period preserved,
  `C=false` because the extension constructs a bit-exact seam). Other axes
  (`:inclusive`, non-periodic) pass through unchanged. Downstream dispatch
  should use the `_is_periodic_seam_folded` trait rather than `isa
  PeriodicBC{:exclusive}` so both `:exclusive` (one-shot) and `:extended`
  (post-extension) are handled uniformly.
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
        _validate_exclusive_period(grid_d, period)
        x_end = first(grid_d) + Tg(period)
        last(grid_d) < x_end ||
            _throw_prepare_periodic_nd_endpoint(d, period, x_end, last(grid_d))
        # Range axes use type-preserving `_to_float_adding_endpoint`; Vector axes
        # delegate to the caller-supplied `extend_vector_grid` (vcat vs pool).
        grid_ext = grid_d isa AbstractRange ?
            _to_float_adding_endpoint(grid_d, Tg) :
            extend_vector_grid(grid_d, x_end, Tg)
        # Promote bc to `:extended` post-extension: the extended grid IS a
        # closed-cycle layout (length n+1, last point at x[1]+period). The
        # `:extended` symbol records this promotion so downstream solvers and
        # cache builders can distinguish "user gave :exclusive, we extended"
        # from "user gave :inclusive" while still routing through the
        # `_is_periodic_seam_folded` trait. Without this, BC-aware solvers
        # (e.g. cubic) would interpret `:exclusive` as "raw n-grid" and
        # miscount the cycle.
        return (grid_ext, _bc_after_extend(bc_d))
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
