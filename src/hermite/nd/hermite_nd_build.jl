# ========================================
# CubicHermiteInterpolantND Build Path
# ========================================
#
# Pack data + user-supplied partials into `_NodalDerivativesND`, extending
# grid + every stored array along each `PeriodicBC{:exclusive}` axis. Mirrors
# the BC-promotion pattern used by `_prepare_periodic_nd` (`:exclusive →
# :extended`).
#
# Phase 1a BC policy: only `NoBC`, `PeriodicBC{:inclusive}`,
# `PeriodicBC{:exclusive}` are accepted. Other BCs (BCPair, NaturalBC, etc.)
# affect partial *derivation* — irrelevant here because the user supplies
# every partial directly.

# ========================================
# BC validation (Phase 1a)
# ========================================

@noinline function _throw_unsupported_hermite_nd_bc(d::Int, bc)
    throw(ArgumentError(
        "CubicHermiteInterpolantND (Phase 1a): bc[$d] = $(bc) is unsupported. " *
        "Only `NoBC()`, `PeriodicBC(...; endpoint=:inclusive)`, and " *
        "`PeriodicBC(...; endpoint=:exclusive)` are accepted because user " *
        "partials supersede BC-derived ones. Use the auto-slope methods " *
        "(`pchip_interp`, `cardinal_interp`, `akima_interp`) for richer BC families.",
    ))
end

@inline _is_hermite_nd_bc_allowed(::NoBC) = true
@inline _is_hermite_nd_bc_allowed(::PeriodicBC) = true
@inline _is_hermite_nd_bc_allowed(::AbstractBC) = false

function _validate_hermite_nd_bcs_phase1a(bcs::Tuple{Vararg{AbstractBC, N}}) where {N}
    @inbounds for d in 1:N
        _is_hermite_nd_bc_allowed(bcs[d]) || _throw_unsupported_hermite_nd_bc(d, bcs[d])
    end
    return nothing
end

# ========================================
# Size consistency between data and every stored partial
# ========================================

@noinline function _throw_data_partial_size_mismatch(sz_data, mask::Int, sz_part)
    throw(DimensionMismatch(
        "CubicHermiteInterpolantND: data size $(sz_data) ≠ partial size $(sz_part) " *
        "for mask=$mask. data and every partial array must share `size`.",
    ))
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
    throw(ArgumentError(
        "CubicHermiteInterpolantND: $slot_name has unequal endpoints along axis $d " *
        "but `bcs[$d]` is PeriodicBC{:inclusive}. Either fix the data or use " *
        "`endpoint=:exclusive`.",
    ))
end

# Per-eltype tolerance matching `_check_periodic_endpoints` in
# `src/core/periodic.jl` (`atol = 8 * eps(T), rtol = sqrt(eps(T))` for floats).
@inline _seam_atol(::Type{T}) where {T <: AbstractFloat} = 8 * eps(T)
@inline _seam_atol(::Type{Complex{T}}) where {T <: AbstractFloat} = 8 * eps(T)
@inline _seam_atol(::Type) = false   # integer / duck-type fallback → exact ==
@inline _seam_rtol(::Type{T}) where {T <: AbstractFloat} = sqrt(eps(T))
@inline _seam_rtol(::Type{Complex{T}}) where {T <: AbstractFloat} = sqrt(eps(T))
@inline _seam_rtol(::Type) = false

# Element-wise inclusive-seam check along compile-time axis `d`. Iterates
# the other N-1 axes with a `CartesianIndices` loop; the `d`-th index is
# fixed at `1` (first) vs. `n_d` (last). Comparing scalars one-by-one
# avoids the view/broadcast allocation that `selectdim(arr, ::Int, ...)`
# would incur from the Union-return `ntuple` over a runtime `d`.
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
            i == d ? 1   : I[i]
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
    # Compile-time unroll over data axes. For each axis `d`, if the BC is
    # `PeriodicBC{:inclusive}` we run `_check_inclusive_seam_along!` on the
    # data array and every stored partial. Runtime branch on the bc type is
    # type-stable (single tuple element type per slot), so the body
    # specializes per-axis and the per-element index tuples stay inferable.
    blocks = Expr[]
    for d in 1:N
        push!(blocks, quote
            if bcs[$d] isa PeriodicBC{:inclusive}
                _check_inclusive_seam_along!(data, Val($d)) ||
                    _throw_inclusive_seam_mismatch("data", $d)
                @inbounds for m in 1:K
                    _check_inclusive_seam_along!(partials.partials[m], Val($d)) ||
                        _throw_inclusive_seam_mismatch("partials[mask=$m]", $d)
                end
            end
        end)
    end
    return Expr(:block, blocks..., :(return nothing))
end

# ========================================
# Pack and extend
# ========================================

"""
    _pack_and_extend_nodal_derivs(grids, data, partials, bcs)
        -> (grids_ext, nodal_derivs, bcs_post_extend)

In one pass:

1. Compute post-extension axis lengths: `n_ext[d] = n[d] + 1` if
   `bcs[d] isa PeriodicBC{:exclusive}`, else `n[d]`.
2. Allocate `buf::Array{Tv, N+1}` of shape `(2^N, n_ext_1, …, n_ext_N)`.
3. Fill `buf[1, …]` from `data` (mask=0 slot) and `buf[m+1, …]` from
   `partials.partials[m]` for `m ∈ 1:2^N-1`. For each `:exclusive` axis,
   append a wrap row equal to the first slice along that axis.
4. Extend grids: Range → `range(first(g); step=step(g), length=n+1)`;
   Vector → `vcat(g, x_end)`.
5. Promote each `:exclusive` BC to `:extended` via `_bc_after_extend`.

Returns the extended grids, the packed `_NodalDerivativesND{Tv, N, N+1}`,
and the per-axis BC tuple post-extension.
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

    # 3. Fill slots — mask=0 from data, mask=m from partials.partials[m]
    _fill_mask_slot_with_wrap!(buf, 1, data, extended)
    @inbounds for m in 1:K
        _fill_mask_slot_with_wrap!(buf, m + 1, partials.partials[m], extended)
    end

    nodal_derivs = _NodalDerivativesND{Tv, N, NP1}(buf)

    # 4. Extend grids (Range type preserved when possible)
    grids_ext = ntuple(Val(N)) do d
        bc_d = bcs[d]
        if bc_d isa PeriodicBC{:exclusive}
            _extend_grid_one_axis(grids[d], bc_d, Tg)
        else
            grids[d]
        end
    end

    # 5. Promote `:exclusive` BCs to `:extended`, with period materialized
    #    from the extended grid span. Inclusive / NoBC pass through (with
    #    period materialized for inclusive too, mirroring CubicInterpolantND).
    bcs_post = ntuple(Val(N)) do d
        bc_d = bcs[d]
        bc_eff = _bc_after_extend(bc_d)
        if bc_eff isa PeriodicBC
            _with_resolved_period(bc_eff, last(grids_ext[d]) - first(grids_ext[d]))
        else
            bc_eff
        end
    end

    return grids_ext, nodal_derivs, bcs_post
end

# Wrap-aware fill into one mask slot. After this call:
# - `buf[slot, 1:n_src[1], …, 1:n_src[N]] == src`
# - For every axis d with `extended[d]==true`, `buf[slot, …, n_ext_d, …]`
#   along that axis duplicates `buf[slot, …, 1, …]` (wrap row).
#
# Sequential wrap is safe: after axis 1 wrap, the corner cell at index
# `(n_ext_1, n_ext_2, …)` is reached by axis 2 wrap pulling from index
# `(n_ext_1, 1, …)` which was already set from `src[1, …]`.
function _fill_mask_slot_with_wrap!(
        buf::AbstractArray{Tv, NP1},
        slot::Int,
        src::AbstractArray{Tv, N},
        extended::NTuple{N, Bool},
    ) where {Tv, NP1, N}
    n_src = size(src)
    # Copy `src` into `buf[slot, 1:n_src[1], …, 1:n_src[N]]`.
    src_view = view(buf, slot, ntuple(d -> Base.OneTo(n_src[d]), Val(N))...)
    src_view .= src

    # Wrap rows for every `:exclusive` axis (no-op if none). Routed through
    # a `@generated` unroll so each per-axis `d` is a compile-time Val, the
    # SubArray index tuples are inferable, and the views are stack-elidable.
    # Mirrors `_extend_all_slices!` in core/periodic.jl.
    _wrap_all_axes_in_buf_slot!(buf, slot, extended, Val(N))
    return buf
end

# Apply ONE wrap row along data-axis `d` (= buf-dim `d+1`). Compile-time `d`
# via Val keeps the index-tuple `ntuple` closures inferable, so the SubArray
# headers produced by `view(buf, slot, ...)` are stack-allocated.
@inline function _wrap_one_axis_in_buf_slot!(
        buf::AbstractArray{Tv, NP1}, slot::Int,
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
    copyto!(view(buf, slot, dst_inds...), view(buf, slot, src_inds...))
    return nothing
end

# Compile-time unroll of per-axis wrap calls. Emits N straight-line dispatches
# to `_wrap_one_axis_in_buf_slot!` with literal `Val($d)`. The runtime
# `extended[d] || return nothing` short-circuit then compiles to a constant
# branch when the caller's `extended` tuple is constant-folded.
@generated function _wrap_all_axes_in_buf_slot!(
        buf::AbstractArray{Tv, NP1}, slot::Int,
        extended::NTuple{N, Bool}, ::Val{N},
    ) where {Tv, NP1, N}
    calls = [
        :(_wrap_one_axis_in_buf_slot!(buf, slot, extended, Val($d), Val(N)))
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
        throw(ArgumentError(
            "CubicHermiteInterpolantND: exclusive-axis grid last point $(last(grid)) " *
            "≥ extended endpoint $(x_end); check the period.",
        ))
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
# Mirrors `_pack_and_extend_nodal_derivs` but uses `acquire!(pool, ...)` for
# the packed buffer and any Vector-grid extension. Returns the same triple
# (grids_ext, packed_buf, bcs_post) where `packed_buf` is a *pool-owned*
# `Array{Tv, N+1}` of shape `(2^N, n_ext_1, …, n_ext_N)`.
#
# Safety: the returned `packed_buf` must NOT escape the enclosing
# `@with_pool` scope — it is reclaimed on pool rewind. Caller is the
# `@with_pool`-decorated oneshot entry, which only uses the buffer in-line
# through `_eval_nd_cell` before returning a scalar value.
@inline function _pack_and_extend_nodal_derivs_pooled(
        pool::AbstractArrayPool,
        grids::Tuple{Vararg{AbstractVector{Tg}, N}},
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        bcs::Tuple{Vararg{AbstractBC, N}},
    ) where {Tg, Tv, N, K}
    extended = ntuple(d -> bcs[d] isa PeriodicBC{:exclusive}, Val(N))
    n_orig = size(data)
    n_ext = ntuple(d -> extended[d] ? n_orig[d] + 1 : n_orig[d], Val(N))

    K_total = 1 << N
    buf = acquire!(pool, Tv, (K_total, n_ext...))

    _fill_mask_slot_with_wrap!(buf, 1, data, extended)
    @inbounds for m in 1:K
        _fill_mask_slot_with_wrap!(buf, m + 1, partials.partials[m], extended)
    end

    # `map` over tuples dispatches per-element with concrete (grid_d, bc_d)
    # types, so the `isa PeriodicBC{:exclusive}` branch specializes
    # per-axis and the return type is inferred without Union boxing. The
    # `ntuple(d -> ...)` form with `bcs[d]` runtime indexing into a
    # heterogeneous tuple would otherwise box the Union (~5 KB/call).
    # See MEMORY.md "ND Constructor Inferrability Pattern".
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
# (alloc-free); Vector case acquires a pool buffer and copies + writes the
# wrap endpoint (mirrors `_PoolGridExtender` from core/periodic.jl).
@inline function _extend_grid_one_axis_pooled(
        pool::AbstractArrayPool,
        grid::AbstractVector{Tg}, bc::PeriodicBC{:exclusive}, ::Type{Tg},
    ) where {Tg}
    period = _resolve_exclusive_period(grid, bc)
    _validate_exclusive_period(grid, period)
    x_end = first(grid) + Tg(period)
    last(grid) < x_end ||
        throw(ArgumentError(
            "CubicHermiteInterpolantND: exclusive-axis grid last point $(last(grid)) " *
            "≥ extended endpoint $(x_end); check the period.",
        ))
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
