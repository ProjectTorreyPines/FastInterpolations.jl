# ========================================
# CubicHermiteInterpolantND Build Path
# ========================================
#
# Pack data + user-supplied partials into `_NodalDerivativesND`, extending
# grid + every stored array along each `PeriodicBC{:exclusive}` axis
# (`:exclusive → :extended`).
#
# Only `NoBC` / `PeriodicBC{:inclusive,:exclusive}` are accepted: other BC
# families shape partial *derivation*, which is moot when the user supplies
# every partial directly.

# ========================================
# BC validation
# ========================================

@noinline function _throw_unsupported_hermite_nd_bc(d::Int, bc)
    throw(
        ArgumentError(
            "CubicHermiteInterpolantND: bc[$d] = $(bc) is unsupported. " *
                "Only `NoBC()`, `PeriodicBC(...; endpoint=:inclusive)`, and " *
                "`PeriodicBC(...; endpoint=:exclusive)` are accepted because user " *
                "partials supersede BC-derived ones. Use the auto-slope methods " *
                "(`pchip_interp`, `cardinal_interp`, `akima_interp`) for richer BC families.",
        )
    )
end

@inline _is_hermite_nd_bc_allowed(::NoBC) = true
@inline _is_hermite_nd_bc_allowed(::PeriodicBC) = true
@inline _is_hermite_nd_bc_allowed(::AbstractBC) = false

function _validate_hermite_nd_bcs(bcs::Tuple{Vararg{AbstractBC, N}}) where {N}
    @inbounds for d in 1:N
        _is_hermite_nd_bc_allowed(bcs[d]) || _throw_unsupported_hermite_nd_bc(d, bcs[d])
    end
    return nothing
end

# ========================================
# Size consistency between data and every stored partial
# ========================================

@noinline function _throw_data_partial_size_mismatch(sz_data, mask::Int, sz_part)
    throw(
        DimensionMismatch(
            "CubicHermiteInterpolantND: data size $(sz_data) ≠ partial size $(sz_part) " *
                "for mask=$mask. data and every partial array must share `size`.",
        )
    )
end

function _validate_partial_sizes(
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
    ) where {Tv, N, K}
    sz = size(data)
    @inbounds for m in 1:K
        size(partials.partials[m]) == sz || _throw_data_partial_size_mismatch(sz, m, size(partials.partials[m]))
    end
    return nothing
end

# ========================================
# Inclusive periodic seam validation
# ========================================
#
# For axes with `PeriodicBC{:inclusive}`, user-supplied data + every partial
# array must already be closed-cycle along that axis. Validate once at
# build time so downstream search/eval can assume the invariant.

@noinline function _throw_inclusive_seam_mismatch(slot_name::String, d::Int)
    throw(
        ArgumentError(
            "CubicHermiteInterpolantND: $slot_name has unequal endpoints along axis $d " *
                "but `bcs[$d]` is PeriodicBC{:inclusive}. Either fix the data or use " *
                "`endpoint=:exclusive`.",
        )
    )
end

# Per-eltype seam tolerance: 8·eps atol, sqrt(eps) rtol for floats; exact
# `==` (atol/rtol = false) for integer / duck types.
@inline _seam_atol(::Type{T}) where {T <: AbstractFloat} = 8 * eps(T)
@inline _seam_atol(::Type{Complex{T}}) where {T <: AbstractFloat} = 8 * eps(T)
@inline _seam_atol(::Type) = false   # integer / duck-type fallback → exact ==
@inline _seam_rtol(::Type{T}) where {T <: AbstractFloat} = sqrt(eps(T))
@inline _seam_rtol(::Type{Complex{T}}) where {T <: AbstractFloat} = sqrt(eps(T))
@inline _seam_rtol(::Type) = false

# Element-wise inclusive-seam check along compile-time axis `d`: compare the
# first vs. last slice (scalar-by-scalar, alloc-free) over the other N-1 axes.
@inline function _check_inclusive_seam_along!(
        arr::AbstractArray{Tv, N}, ::Val{d},
    ) where {Tv, N, d}
    n_d = size(arr, d)
    n_d >= 2 || return true
    atol = _seam_atol(Tv)
    rtol = _seam_rtol(Tv)
    other_sizes = ntuple(Val(N)) do i
        i == d ? 1 : size(arr, i)
    end
    @inbounds for I in CartesianIndices(other_sizes)
        idx_first = ntuple(Val(N)) do i
            i == d ? 1 : I[i]
        end
        idx_last = ntuple(Val(N)) do i
            i == d ? n_d : I[i]
        end
        a = arr[idx_first...]
        b = arr[idx_last...]
        if Tv <: AbstractFloat || Tv <: Complex{<:AbstractFloat}
            isapprox(a, b; atol = atol, rtol = rtol) || return false
        else
            a == b || return false
        end
    end
    return true
end

@generated function _validate_inclusive_seams(
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        bcs::Tuple{Vararg{AbstractBC, N}},
    ) where {Tv, N, K}
    # Unroll over axes; for each `:inclusive` axis check the data array and
    # every stored partial.
    blocks = Expr[]
    for d in 1:N
        push!(
            blocks, quote
                if bcs[$d] isa PeriodicBC{:inclusive}
                    _check_inclusive_seam_along!(data, Val($d)) ||
                        _throw_inclusive_seam_mismatch("data", $d)
                    @inbounds for m in 1:K
                        _check_inclusive_seam_along!(partials.partials[m], Val($d)) ||
                            _throw_inclusive_seam_mismatch("partials[mask=$m]", $d)
                    end
                end
            end
        )
    end
    return Expr(:block, blocks..., :(return nothing))
end

# ========================================
# Pack and extend
# ========================================

"""
    _pack_and_extend_nodal_derivs(grids, data, partials, bcs)
        -> (grids_ext, nodal_derivs, bcs_post)

Pack `data` + the `2^N-1` partials into a `(2^N, n_ext...)` buffer, append a
wrap row along each `:exclusive` axis, extend those grids, and promote their
BCs `:exclusive → :extended`. Heap-allocating; see the pooled twin for the
zero-alloc one-shot path.
"""
function _pack_and_extend_nodal_derivs(
        grids::Tuple{Vararg{AbstractVector{Tg}, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        bcs::Tuple{Vararg{AbstractBC, N}},
    ) where {Tg, Tv, N, K}
    # 1. Per-axis extension flag + final size
    extended = ntuple(d -> bcs[d] isa PeriodicBC{:exclusive}, Val(N))
    n_orig = size(data)
    n_ext = ntuple(d -> extended[d] ? n_orig[d] + 1 : n_orig[d], Val(N))

    # 2. Allocate packed buffer
    K_total = 1 << N  # 2^N (data + all mixed partials)
    NP1 = N + 1
    buf = Array{Tv, NP1}(undef, K_total, n_ext...)

    # 3. Node-major fill — all 2^N slots at each node written in one burst.
    #    Wrap rows for `:exclusive` axes are filled separately afterwards.
    _fill_packed_src_region!(buf, (data, partials.partials...))
    _wrap_all_axes_all_slots!(buf, extended, Val(N))

    nodal_derivs = _NodalDerivativesND{Tv, N, NP1}(buf)

    # 4. Extend grids (Range type preserved). `map` is used over
    #    `ntuple(d -> ...bcs[d]...)` so each axis dispatches with a concrete
    #    `bc_d` type — runtime-indexing the heterogeneous `bcs` tuple would
    #    box the Union into the result.
    grids_ext = map(grids, bcs) do grid_d, bc_d
        bc_d isa PeriodicBC{:exclusive} ? _extend_grid_one_axis(grid_d, bc_d, Tg) : grid_d
    end

    # 5. Promote `:exclusive` BCs to `:extended` and materialize the period
    #    from the extended grid span (inclusive period materialized too).
    bcs_post = map(bcs, grids_ext) do bc_d, grid_ext_d
        bc_eff = _bc_after_extend(bc_d)
        bc_eff isa PeriodicBC ?
            _with_resolved_period(bc_eff, last(grid_ext_d) - first(grid_ext_d)) :
            bc_eff
    end

    return grids_ext, nodal_derivs, bcs_post
end

# ========================================
# Packed-buffer fill (node-major @generated unroll)
# ========================================
#
# Slot-major (`buf[slot, :, …] .= src`, 2^N passes) re-touches every cache
# line once per slot; for L1-spilling grids each line is re-fetched 2^N
# times. Node-major writes all 2^N partials at one node contiguously (mask
# is dim 1, the fast axis), filling each cache line in one burst. The slot
# loop is unrolled at compile time so each source array indexes with a
# concrete element type. Wrap rows for `:exclusive` axes are added after.
@inline function _fill_packed_src_region!(
        buf::AbstractArray{Tv, NP1},
        sources::Tuple{Vararg{AbstractArray{Tv, N}}},
    ) where {Tv, NP1, N}
    return _fill_packed_src_region_unrolled!(buf, sources, Val(N))
end

@generated function _fill_packed_src_region_unrolled!(
        buf::AbstractArray{Tv, NP1},
        sources::Tuple{Vararg{AbstractArray{Tv, N}, K}},
        ::Val{N},
    ) where {Tv, NP1, N, K}
    # Per-slot scalar writes at one node; literal `sources[$k]` access keeps
    # each write concretely typed.
    loop_vars = [Symbol(:i_, d) for d in 1:N]
    inner = Expr(:block)
    for k in 1:K
        push!(inner.args, :(@inbounds buf[$k, $(loop_vars...)] = sources[$k][$(loop_vars...)]))
    end
    # Nest loops with i_1 innermost (column-major fast axis of buf).
    body = inner
    for d in 1:N
        body = quote
            @inbounds for $(loop_vars[d]) in 1:size(sources[1], $d)
                $body
            end
        end
    end
    return quote
        Base.@_inline_meta
        $body
        return buf
    end
end

# ========================================
# Wrap rows for `:exclusive` axes — all-slots at once
# ========================================
#
# For each `extended[d]` axis, copy slice 1 → the wrap row across all 2^N
# mask slots in one `copyto!` (every slot wraps identically).
@inline function _wrap_one_axis_all_slots!(
        buf::AbstractArray{Tv, NP1},
        extended::NTuple{N, Bool},
        ::Val{d}, ::Val{N},
    ) where {Tv, NP1, N, d}
    extended[d] || return nothing
    n_buf_d = size(buf, d + 1)
    src_inds = ntuple(Val(N)) do i
        i == d ? (1:1) : (1:size(buf, i + 1))
    end
    dst_inds = ntuple(Val(N)) do i
        i == d ? (n_buf_d:n_buf_d) : (1:size(buf, i + 1))
    end
    copyto!(view(buf, :, dst_inds...), view(buf, :, src_inds...))
    return nothing
end

# Compile-time unroll over data axes.
@generated function _wrap_all_axes_all_slots!(
        buf::AbstractArray{Tv, NP1},
        extended::NTuple{N, Bool}, ::Val{N},
    ) where {Tv, NP1, N}
    calls = [
        :(_wrap_one_axis_all_slots!(buf, extended, Val($d), Val(N)))
            for d in 1:N
    ]
    return Expr(:block, calls..., :(return nothing))
end

# Per-axis grid extension for an `:exclusive` axis. Range-preserving for
# AbstractRange inputs; vcat-based for Vector inputs.
@inline function _extend_grid_one_axis(
        grid::AbstractVector{Tg}, bc::PeriodicBC{:exclusive}, ::Type{Tg},
    ) where {Tg}
    period = _resolve_exclusive_period(grid, bc)
    _validate_exclusive_period(grid, period)
    x_end = first(grid) + Tg(period)
    last(grid) < x_end ||
        throw(
        ArgumentError(
            "CubicHermiteInterpolantND: exclusive-axis grid last point $(last(grid)) " *
                "≥ extended endpoint $(x_end); check the period.",
        )
    )
    if grid isa AbstractRange
        return range(first(grid); step = step(grid), length = length(grid) + 1)
    else
        return vcat(grid, x_end)
    end
end

# ========================================
# Pool-based pack + extend (one-shot zero-alloc backend)
# ========================================
#
# `_pack_and_extend_nodal_derivs` with `acquire!(pool, ...)` for the packed
# buffer and any Vector-grid extension.
#
# SAFETY: the returned `packed_buf` is pool-owned and reclaimed on `@with_pool`
# rewind — it must NOT escape that scope. The caller consumes it in-line
# through `_eval_nd_cell` and returns a scalar.
@inline function _pack_and_extend_nodal_derivs_pooled(
        pool::AbstractArrayPool,
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        bcs::Tuple{Vararg{AbstractBC, N}},
    ) where {Tv, N, K}
    # `Tg` only feeds the `:exclusive` virtual-endpoint extension below (the common
    # non-periodic path leaves each axis untouched); raw grids may be Int/heterogeneous.
    Tg = _promote_grid_eltype(grids)
    extended = ntuple(d -> bcs[d] isa PeriodicBC{:exclusive}, Val(N))
    n_orig = size(data)
    n_ext = ntuple(d -> extended[d] ? n_orig[d] + 1 : n_orig[d], Val(N))

    K_total = 1 << N
    buf = acquire!(pool, Tv, (K_total, n_ext...))

    _fill_packed_src_region!(buf, (data, partials.partials...))
    _wrap_all_axes_all_slots!(buf, extended, Val(N))

    # `map` (not `ntuple` over `bcs[d]`) so each axis dispatches concretely
    # without boxing the Union — see the heap twin.
    grids_ext = map(grids, bcs) do grid_d, bc_d
        bc_d isa PeriodicBC{:exclusive} ?
            _extend_grid_one_axis_pooled(pool, grid_d, bc_d, Tg) :
            grid_d
    end

    bcs_post = map(bcs, grids_ext) do bc_d, grid_ext_d
        bc_eff = _bc_after_extend(bc_d)
        bc_eff isa PeriodicBC ?
            _with_resolved_period(bc_eff, last(grid_ext_d) - first(grid_ext_d)) :
            bc_eff
    end

    return grids_ext, buf, bcs_post
end

# Pool-based per-axis grid extension. Range case preserves Range type
# (alloc-free); Vector case acquires a pool buffer + writes the wrap endpoint.
@inline function _extend_grid_one_axis_pooled(
        pool::AbstractArrayPool,
        grid::AbstractVector{Tg}, bc::PeriodicBC{:exclusive}, ::Type{Tg},
    ) where {Tg}
    period = _resolve_exclusive_period(grid, bc)
    _validate_exclusive_period(grid, period)
    x_end = first(grid) + Tg(period)
    last(grid) < x_end ||
        throw(
        ArgumentError(
            "CubicHermiteInterpolantND: exclusive-axis grid last point $(last(grid)) " *
                "≥ extended endpoint $(x_end); check the period.",
        )
    )
    if grid isa AbstractRange
        return range(first(grid); step = step(grid), length = length(grid) + 1)
    else
        n = length(grid)
        g_ext = acquire!(pool, Tg, n + 1)
        @inbounds copyto!(g_ext, 1, grid, 1, n)
        @inbounds g_ext[n + 1] = x_end
        return g_ext
    end
end
