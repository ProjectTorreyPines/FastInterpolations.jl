# ============================================================================
# _AxisAnchor — shared per-axis anchor backbone for gridded (separable) eval
# ============================================================================
#
# Shared RESOLUTION scaffold (searcher selection + hint chaining, InBounds lean
# arm, NoExtrap throwing BEFORE any output-sized allocation). Each method
# defines its own isbits payload via `_resolve_anchor`, and consuming kernels
# stay per-family.
#
# `M` is the method type (phantom tag): it routes the consuming pass primitive
# and keeps same-shape payloads of different methods from cross-wiring.

struct _AxisAnchor{M, P}
    idx::Int      # left node index (constant: the side-resolved SELECTED node)
    payload::P    # method-defined isbits tuple
end

# Anchor vector eltype per (method, axis types, pass value type). `Tvals` is
# the eltype of the array the consuming pass READS (raw data for pass 1 /
# raw-local methods; the promoted intermediate for later Hermite passes) —
# it feeds the value-matched width type exactly like point-wise eval.
@inline function _axis_anchor_type(
        m::ConstantInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        ::Type{Tvals}
    ) where {Tvals}
    Tone = promote_type(eltype(grid), eltype(targets))
    return _AxisAnchor{typeof(m), Tuple{Tone}}
end

@inline function _axis_anchor_type(
        m::LinearInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tinv = _promote_eltype(_inv_op, Tw)
    Tα = promote_type(eltype(grid), eltype(targets), Tinv)
    return _AxisAnchor{typeof(m), Tuple{Tα, Tinv}}
end

@inline function _axis_anchor_type(
        m::LinearInterp,
        grid::_ExclusivePeriodicAxis,
        targets::AbstractVector,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tinv = _promote_eltype(_inv_op, Tw)
    Tα = promote_type(eltype(grid), eltype(targets), Tinv)
    return _AxisAnchor{typeof(m), Tuple{Int, Tα, Tinv}}
end

@inline function _axis_anchor_type(
        m::AbstractLocalHermiteInterp,
        grid::AbstractVector,
        targets::AbstractVector,
        ::Type{Tvals}
    ) where {Tvals}
    Tw = _promote_grid_float(eltype(grid), Tvals)
    Tdl = promote_type(eltype(grid), eltype(targets), Tw)
    return _AxisAnchor{typeof(m), Tuple{Tdl, Tw, Tw}}
end

# ---- per-method payload resolution -------------------------------------------
# Constant: the side offset is folded into `idx` at build (the gather reads ONE
# node). Clamp/Fill folds the coordinate into the boundary cell (`dL ∈ [0, h]`)
# — the point-wise ND surface clamps the query COORDINATE before its kernel, so
# an OOB-left raw `dL < 0` would flip `RightSide`'s `iszero` test to the wrong
# node. Wrap folds in `_anchor_loc`, Fill's slabs are a post-pass, NoExtrap
# throws in the loop. The payload carries `one(dL)` for the point-wise kernel's
# carrier contract (`data * one(dL_1) * …`).
@inline function _resolve_anchor(
        m::ConstantInterp,
        ::Type{_AxisAnchor{M, Tuple{Tone}}},
        grid::AbstractVector,
        idx::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {M, Tone}
    h = _get_h(grid, idx, xL, xR)
    dL = xq - xL
    if extrap isa Union{ClampExtrap, FillExtrap}
        dL = clamp(dL, zero(dL), oftype(dL, h))
    end
    off = _compute_single_offset(m.side, h, dL)
    return _AxisAnchor{M, Tuple{Tone}}(idx + off, (Tone(one(dL)),))
end

@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{M, Tuple{Tα, Tinv}}},
        grid::AbstractVector,
        idx::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {M, Tα, Tinv}
    inv_h = Tinv(_get_inv_h(grid, idx, xL, xR))
    α = Tα(_alpha_of(xq, xL, inv_h))
    if extrap isa Union{ClampExtrap, FillExtrap}
        α = clamp(α, zero(Tα), one(Tα))
    end
    return _AxisAnchor{M, Tuple{Tα, Tinv}}(idx, (α, inv_h))
end

@inline function _resolve_anchor(
        ::LinearInterp,
        ::Type{_AxisAnchor{M, Tuple{Int, Tα, Tinv}}},
        grid::AbstractVector,
        idx::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {M, Tα, Tinv}
    inv_h = Tinv(_get_inv_h(grid, idx, xL, xR))
    α = Tα(_alpha_of(xq, xL, inv_h))
    if extrap isa Union{ClampExtrap, FillExtrap}
        α = clamp(α, zero(Tα), one(Tα))
    end
    return _AxisAnchor{M, Tuple{Int, Tα, Tinv}}(idx, (idxR, α, inv_h))
end

# Local Hermite: location-only payload `(dL, h, inv_h)` — slopes are
# data-dependent (and nonlinear), computed in-pass; geometry runs at the
# value-matched `Tw` exactly like the point-wise 1D arm. Clamp/Fill folds the
# coordinate into the cell (`dL ∈ [0, h]`), matching the point-wise ND
# surface, which clamps the query coordinate and still evaluates the kernel
# (a Clamp-OOB derivative is the boundary cell polynomial's slope): t = 0 is
# basis-exact; t = 1 is `h·inv_h` (≤ ~1 ulp).
@inline function _resolve_anchor(
        m::AbstractLocalHermiteInterp,
        ::Type{_AxisAnchor{M, Tuple{Tdl, Tw, Tw2}}},
        grid::AbstractVector,
        idx::Int,
        idxR::Int,
        xq,
        xL,
        xR,
        extrap::AbstractExtrap
    ) where {M, Tdl, Tw, Tw2}
    h = _get_h(Tw, grid, idx)
    inv_h = _get_inv_h(Tw, grid, idx)
    dL = Tdl(xq - xL)
    if extrap isa Union{ClampExtrap, FillExtrap}
        dL = clamp(dL, zero(Tdl), Tdl(h))
    end
    return _AxisAnchor{M, Tuple{Tdl, Tw, Tw2}}(idx, (dL, Tw(h), Tw2(inv_h)))
end

# ---- shared resolution loop ---------------------------------------------------
# Mirrors `_gridded_anchors_loop!` (see its notes): the searcher union splits at
# this call; @inline keeps the hinted arm's fresh `RefHint` inside the caller's
# compiled unit (allocation-elided). NoExtrap throws the canonical axis-named
# error on the FIRST OOB coordinate — before any output-sized buffer exists.
@inline function _axis_anchors_loop!(
        anchors::Vector{A},
        m::AbstractInterpMethod,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        searcher::Searcher
    ) where {A <: _AxisAnchor}
    k = 0
    @inbounds for xq in targets
        k += 1
        if extrap isa InBounds
            # caller-asserted in-domain: lean search, same call the default
            # path reaches after its `_oob_state` check (bit-identical anchors).
            idx, idxR, xL, xR = search_interval(searcher, grid, xq, extrap)
            anchors[k] = _resolve_anchor(m, A, grid, idx, idxR, xq, xL, xR, extrap)
            continue
        end
        loc = _anchor_loc(grid, xq, extrap isa WrapExtrap, searcher)
        if extrap isa NoExtrap && loc.state != IN_DOMAIN
            _throw_domain_error(xq, grid, dim)
        end
        anchors[k] = _resolve_anchor(m, A, grid, loc.idxL, loc.idxR, loc.xq, loc.xL, loc.xR, extrap)
    end
    return anchors
end

function _axis_anchors_pooled(
        pool,
        m::AbstractInterpMethod,
        grid::AbstractVector,
        targets::AbstractVector,
        extrap::AbstractExtrap,
        dim::Int,
        ::Type{Tvals}
    ) where {Tvals}
    A = _axis_anchor_type(m, grid, targets, Tvals)
    anchors = acquire!(pool, A, length(targets))
    return _axis_anchors_loop!(anchors, m, grid, targets, extrap, dim, _gridded_build_searcher(grid, targets))
end

# All-axes anchor tuple via literal-index construction — an ntuple closure over
# the heterogeneous (methods, grids, targets) tuples boxes on some type
# combinations (the `_gridded_hulls` lesson). `T1`/`Trest` split the pass value
# type: Hermite's pass 1 reads raw data (`Tv`) while later passes read the
# promoted intermediate (`Tmid`); raw-local single-pass methods pass the same
# type twice.
@generated function _axis_anchors_all(
        pool,
        methods,
        grids,
        targets,
        extraps,
        ::Type{T1},
        ::Type{Trest},
        ::Val{N}
    ) where {T1, Trest, N}
    exprs = [
        :(
                _axis_anchors_pooled(
                    pool, methods[$d], grids[$d], targets[$d], extraps[$d], $d,
                    $(d == 1 ? :T1 : :Trest)
                )
            )
            for d in 1:N
    ]
    return :(($(exprs...),))
end
