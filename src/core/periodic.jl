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
@inline function _wrap_to_domain(xi::Tg, x_min::Tg, x_max::Tg) where {Tg <: AbstractFloat}
    # Single-branch check: outside domain → slow path
    if xi < x_min || xi >= x_max
        period = x_max - x_min
        return x_min + mod(xi - x_min, period)
    end
    # Fast path: already in domain (most common case)
    return xi
end

# Generic wrapper: handles Dual, Int, Float32 on Float64 grid, etc.
# IMPORTANT: Preserves AD Dual type through the entire operation.
# mod() is compatible with ForwardDiff.Dual, so we use it directly on xi.
@inline function _wrap_to_domain(xi::Real, x_min::Tg, x_max::Tg) where {Tg <: AbstractFloat}
    xi_primal = _extract_primal(xi)
    # Fast path: already in domain, return original xi (preserves Dual type for AD)
    if xi_primal >= x_min && xi_primal < x_max
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

Validate that `y[1] == y[end]` for periodic boundary conditions (inclusive endpoint).
Called once at construction time (zero runtime overhead).

Uses strict `==` equality — no approximate comparison. This is universal for all
value types (scalars, vectors, duck-typed custom types) without requiring `norm`,
`isapprox`, or any tolerance parameters.

If your data is computed (e.g., `sin.(range(0, 2π, n))`), set `y[end] = y[1]`
explicitly to ensure exact periodicity.

Throws `ArgumentError` if endpoints differ.
"""
@inline function _check_periodic_endpoints(y::AbstractVector)
    _extract_primal(first(y)) == _extract_primal(last(y)) || _throw_periodic_endpoint_error(first(y), last(y))
    return nothing
end

@noinline function _throw_periodic_endpoint_error(y1, yn)
    throw(
        ArgumentError(
            "PeriodicBC (inclusive endpoint) requires y[1] == y[end], " *
                "got y[1]=$y1, y[end]=$yn. " *
                "Tip: set y[end] = y[1] to ensure exact periodicity, or use " *
                "PeriodicBC(endpoint=:exclusive) if your data does not repeat the first point."
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
            "Periodic BC on dim $d requires data[1,...] == data[end,...], " *
                "but found data[1,...]=$v_first, data[end,...]=$v_last. " *
                "Tip: set the last slice equal to the first along dim $d."
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
@inline _with_resolved_period(::PeriodicBC{E}, period::T) where {E, T} =
    PeriodicBC{E, T}(period)

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
        range(first(x), step = step(x), length = length(x) + 1)
    else
        vcat(x, [x_end])
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
        range(first(x), step = step(x), length = length(x) + 1)
    else
        vcat(x, [x_end])
    end
    y_ext = vcat(y_mat, y_mat[1:1, :])
    return x_ext, y_ext
end

# Value extension: append first element
_extend_values(y::AbstractVector) = vcat(y, [first(y)])

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
    ) where {Tg <: AbstractFloat, Tv, N}
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

        # Type-stable grid extension: isa branch for compile-time type narrowing
        # (Range → Range, Vector → Vector — concrete type preserved per element)
        grid_ext = if grid_d isa AbstractRange
            range(first(grid_d), step = step(grid_d), length = length(grid_d) + 1)
        else
            vcat(grid_d, [x_end])
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
via `unsafe_acquire!`, so they must NOT escape the enclosing `@with_pool` scope.

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
    ) where {Tg <: AbstractFloat, Tv, N}
    # Fast path: no exclusive axes
    has_exclusive = false
    for d in 1:N
        if bcs[d] isa PeriodicBC{:exclusive}
            has_exclusive = true
            break
        end
    end
    has_exclusive || return (grids, data, bcs)

    # Build extended grids tuple directly (no heap Vector intermediary)
    # Grid extensions are independent per dimension: grids[d] is unmodified input
    grids_out = ntuple(Val(N)) do d
        bc_d = bcs[d]
        bc_d isa PeriodicBC{:exclusive} || return grids[d]

        grid_d = grids[d]
        period = _resolve_exclusive_period(grid_d, bc_d)
        x_end = first(grid_d) + Tg(period)

        # Validate: virtual endpoint must be strictly after last grid point
        last(grid_d) < x_end || throw(
            ArgumentError(
                "PeriodicBC(endpoint=:exclusive) on dim $d: period=$period places " *
                    "virtual endpoint at $x_end, not after last grid point x[end]=$(last(grid_d))"
            )
        )

        # Extend grid: Range → direct construction (O(1)), Vector → pool
        # IMPORTANT: Range branch returns Range unconditionally to prevent Union return type.
        # _resolve_exclusive_period already validates period ≈ step(x)*length(x) for Range grids,
        # so the extended Range always has the correct step and endpoint.
        if grid_d isa AbstractRange
            return range(first(grid_d), step = step(grid_d), length = length(grid_d) + 1)
        else
            n = length(grid_d)
            g_ext = unsafe_acquire!(pool, Tg, n + 1)
            @inbounds copyto!(g_ext, 1, grid_d, 1, n)
            @inbounds g_ext[n + 1] = x_end
            return g_ext
        end
    end

    # Build extended BCs tuple
    bcs_out = ntuple(Val(N)) do d
        bc_d = bcs[d]
        if bc_d isa PeriodicBC{:exclusive}
            period = _resolve_exclusive_period(grids[d], bc_d)
            return _with_resolved_period(bc_d, period)
        else
            return bc_d
        end
    end

    # Extend data: pool-allocate final shape, then fill slices
    final_sizes = ntuple(Val(N)) do d
        bcs[d] isa PeriodicBC{:exclusive} ? size(data, d) + 1 : size(data, d)
    end
    data_out = unsafe_acquire!(pool, Tv, final_sizes)

    # Copy original data into the sub-array
    orig_inds = ntuple(d -> 1:size(data, d), Val(N))
    copyto!(view(data_out, orig_inds...), data)

    # Fill extended slices dim-by-dim (earlier extensions are visible to later dims)
    for d in 1:N
        bcs[d] isa PeriodicBC{:exclusive} || continue
        nd = size(data, d)  # original size in this dim
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
    end

    return (grids_out, data_out, bcs_out)
end
