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

# Compare slabs along axis `d`: array[..., 1, ...] vs array[..., end, ...].
# Uses `selectdim` so the check works for any N without explicit dispatch.
# Tolerance matches `_check_periodic_endpoints` for AbstractFloat in
# `src/core/periodic.jl` (`atol = 8 * eps(T), rtol = sqrt(eps(T))`).
function _check_inclusive_seam_one_axis(arr::AbstractArray{T}, d::Int) where {T}
    n = size(arr, d)
    n >= 2 || return true   # length-1 axis trivially "matches"
    slab_first = selectdim(arr, d, 1)
    slab_last  = selectdim(arr, d, n)
    return _seam_isapprox(slab_first, slab_last, T)
end

# Tolerance-aware comparison matching `_check_periodic_endpoints` per-eltype.
@inline function _seam_isapprox(a, b, ::Type{T}) where {T <: AbstractFloat}
    return isapprox(a, b; atol = 8 * eps(T), rtol = sqrt(eps(T)))
end
@inline function _seam_isapprox(a, b, ::Type{Complex{T}}) where {T <: AbstractFloat}
    return isapprox(a, b; atol = 8 * eps(T), rtol = sqrt(eps(T)))
end
@inline _seam_isapprox(a, b, ::Type) = a == b   # integer / duck-type fallback

function _validate_inclusive_seams(
        data::AbstractArray{Tv, N},
        partials::HermitePartials{N, Tv, K},
        bcs::Tuple{Vararg{AbstractBC, N}},
    ) where {Tv, N, K}
    @inbounds for d in 1:N
        bcs[d] isa PeriodicBC{:inclusive} || continue
        _check_inclusive_seam_one_axis(data, d) ||
            _throw_inclusive_seam_mismatch("data", d)
        for m in 1:K
            _check_inclusive_seam_one_axis(partials.partials[m], d) ||
                _throw_inclusive_seam_mismatch("partials[mask=$m]", d)
        end
    end
    return nothing
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

    # For every extended axis, write wrap row.
    mask_view = view(buf, slot, ntuple(_ -> Colon(), Val(N))...)
    @inbounds for d in 1:N
        if extended[d]
            n_ext_d = size(mask_view, d)
            last_slab  = selectdim(mask_view, d, n_ext_d)
            first_slab = selectdim(mask_view, d, 1)
            last_slab .= first_slab
        end
    end
    return buf
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
