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
        Tg = eltype(x)
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
    Tg = eltype(x)
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
    Tg = eltype(x)
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

"""
    _extend_exclusive_pooled!(pool, x, y, bc::PeriodicBC) -> (x_p, y_p)

Pool-based sibling of `_extend_exclusive`. Returns extended grid/value pair for
the 1D one-shot hot path:
- `:inclusive`: passthrough `(x, y)` (caller still holds the originals).
- `:exclusive`: extend by one endpoint at `x[1] + period`, with value `y[1]`.
  - Grid: `Range → _CachedRange` via `_to_float_adding_endpoint` (no heap);
          `Vector → acquire!(pool, ...)` (zero-alloc after warmup).
  - Values: always pool-acquired (even for Range grids, since `y` may be Vector).

Calls `_check_periodic_endpoints(bc, y_p)` before returning — for `:inclusive`
this validates `y[1] ≈ y[end]`; for `:exclusive` it is trivially satisfied by
construction.

Shared by `linear_interp` / `constant_interp` / `cubic_interp` one-shot paths;
all three previously inlined a near-identical copy of this extension block.
Lifetime of returned buffers is tied to the caller's `@with_pool` scope.
"""
@inline function _extend_exclusive_pooled!(
        pool::AbstractArrayPool,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::PeriodicBC
    ) where {Tg, Tv}
    if bc isa PeriodicBC{:exclusive}
        period = _resolve_exclusive_period(x, bc)
        x_end = first(x) + Tg(period)
        last(x) < x_end || throw(
            ArgumentError(
                "period=$period places virtual endpoint at $x_end, " *
                    "not after last grid point x[end]=$(last(x))"
            )
        )
        n = length(x)
        # _resolve_exclusive_period guarantees period = step(x)*length(x) for Range,
        # so Range → _CachedRange extension is always valid and heap-free.
        if x isa AbstractRange
            x_p = _to_float_adding_endpoint(x, Tg)
        else
            x_p = acquire!(pool, Tg, n + 1)
            @inbounds copyto!(x_p, 1, x, 1, n)
            @inbounds x_p[n + 1] = x_end
        end
        y_p = acquire!(pool, Tv, n + 1)
        @inbounds copyto!(y_p, 1, y, 1, n)
        @inbounds y_p[n + 1] = y[1]
    else
        x_p, y_p = x, y
    end
    _check_periodic_endpoints(bc, y_p)
    return x_p, y_p
end

# ========================================
# ND Exclusive Endpoint Extension
# ========================================

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
    # Fast path: no exclusive axes
    has_exclusive = false
    for d in 1:N
        if bcs[d] isa PeriodicBC{:exclusive}
            has_exclusive = true
            break
        end
    end
    has_exclusive || return (grids, data, bcs)

    # Per-axis grid extension + BC resolution via map (preserves concrete types per-element,
    # unlike Vector{AbstractVector} intermediary which erases concrete grid types)
    processed = map(ntuple(identity, Val(N)), grids, bcs) do d, grid_d, bc_d
        bc_d isa PeriodicBC{:exclusive} || return (grid_d, bc_d)

        period = _resolve_exclusive_period(grid_d, bc_d)
        x_end = first(grid_d) + Tg(period)

        # Validate: virtual endpoint must be strictly after last grid point
        last(grid_d) < x_end || throw(
            ArgumentError(
                "PeriodicBC(endpoint=:exclusive) on dim $d: period=$period places " *
                    "virtual endpoint at $x_end, not after last grid point x[end]=$(last(grid_d))"
            )
        )

        # Type-stable grid extension: Range → _CachedRange, Vector → Vector
        grid_ext = if grid_d isa AbstractRange
            _to_float_adding_endpoint(grid_d, Tg)
        else
            vcat(grid_d, x_end)
        end
        bc_ext = _with_resolved_period(bc_d, period)
        return (grid_ext, bc_ext)
    end
    grids_out = map(first, processed)
    bcs_out = map(last, processed)

    # Extend data: allocate final shape once, then fill slices (avoids O(k) cat copies)
    final_sizes = ntuple(Val(N)) do d
        bcs[d] isa PeriodicBC{:exclusive} ? size(data, d) + 1 : size(data, d)
    end
    data_out = Array{Tv, N}(undef, final_sizes)

    # Copy original data into the sub-array
    orig_inds = ntuple(d -> 1:size(data, d), Val(N))
    copyto!(view(data_out, orig_inds...), data)

    # Fill extended slices dim-by-dim (earlier extensions are visible to later dims)
    for d in 1:N
        bcs[d] isa PeriodicBC{:exclusive} || continue
        nd = size(data, d)  # original size in this dim
        # Build valid ranges: already-extended dims use full size, others use original
        cur_ranges = ntuple(Val(N)) do i
            if i == d
                1:1  # placeholder, overridden below
            elseif i < d && bcs[i] isa PeriodicBC{:exclusive}
                1:(size(data, i) + 1)  # already extended
            else
                1:size(data, i)
            end
        end
        src_inds = ntuple(i -> i == d ? (1:1) : cur_ranges[i], Val(N))
        dst_inds = ntuple(i -> i == d ? ((nd + 1):(nd + 1)) : cur_ranges[i], Val(N))
        copyto!(view(data_out, dst_inds...), view(data_out, src_inds...))
    end

    return (grids_out, data_out, bcs_out)
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
    # Fast path: no exclusive axes
    has_exclusive = false
    for d in 1:N
        if bcs[d] isa PeriodicBC{:exclusive}
            has_exclusive = true
            break
        end
    end
    has_exclusive || return (grids, data, bcs)

    # Build (extended_grid, resolved_bc) pair per axis.
    #
    # CRITICAL: use `map(f, grids, bcs)` rather than `ntuple(Val(N)) do d ... end`.
    # `map` dispatches per-element with concrete (grid, bc) types → each f call
    # compiles to a specialization with concrete return type. `ntuple do d` with
    # runtime indexing `bcs[d]` / `grids[d]` over a heterogeneous tuple leaves the
    # return type as Union{Vector{Tg}, _CachedRange{Tg}, ...} → every tuple slot
    # becomes a heap-boxed Ref, causing ~2 KB/query allocation on mixed grids.
    # See MEMORY.md "ND Constructor Inferrability Pattern".
    # 3-arg `map` with `ntuple(identity, Val(N))` threads the axis index `d` into
    # the closure so error messages can name the failing axis, while still keeping
    # per-element concrete-type dispatch (no runtime Union boxing). Matches the
    # pattern used by the non-pooled sibling `_prepare_periodic_nd`.
    processed = map(ntuple(identity, Val(N)), grids, bcs) do d, grid_d, bc_d
        bc_d isa PeriodicBC{:exclusive} || return (grid_d, bc_d)

        period = _resolve_exclusive_period(grid_d, bc_d)
        x_end = first(grid_d) + Tg(period)
        last(grid_d) < x_end || throw(
            ArgumentError(
                "PeriodicBC(endpoint=:exclusive) on dim $d: period=$period places " *
                    "virtual endpoint at $x_end, not after last grid point x[end]=$(last(grid_d))"
            )
        )

        grid_ext = if grid_d isa AbstractRange
            _to_float_adding_endpoint(grid_d, Tg)
        else
            n = length(grid_d)
            g_ext = acquire!(pool, Tg, n + 1)
            @inbounds copyto!(g_ext, 1, grid_d, 1, n)
            @inbounds g_ext[n + 1] = x_end
            g_ext
        end
        return (grid_ext, _with_resolved_period(bc_d, period))
    end
    grids_out = map(first, processed)
    bcs_out = map(last, processed)

    # Extend data: pool-allocate final shape, then fill slices
    final_sizes = ntuple(Val(N)) do d
        bcs[d] isa PeriodicBC{:exclusive} ? size(data, d) + 1 : size(data, d)
    end
    data_out = acquire!(pool, Tv, final_sizes)

    # Copy original data into the sub-array
    orig_inds = ntuple(d -> 1:size(data, d), Val(N))
    copyto!(view(data_out, orig_inds...), data)

    # Fill extended slices dim-by-dim (earlier extensions are visible to later dims).
    # `@generated` unrolls the N calls at compile time, so each `bcs[d]` index is
    # literal and resolves to the concrete per-axis BC type — no runtime Union box.
    # A naive `for d in 1:N` loop with runtime `d` leaked 48 bytes/query on mixed
    # grids; `ntuple(Val(N)) do d ... end` returning NTuple{N,Nothing} allocated 64.
    _extend_all_slices!(data_out, data, bcs, Val(N))

    return (grids_out, data_out, bcs_out)
end

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
