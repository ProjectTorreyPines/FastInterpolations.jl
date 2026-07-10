# ============================================================================
# _AxisAnchor — shared per-axis anchor RESOLUTION for gridded (separable) eval
# ============================================================================
#
# The `_AxisAnchor{I,P}` struct, its virtual properties, `_interval_type`, and
# the generic `_StatefulPayload` wrapper live in core/axis_anchor_types.jl
# (included before the method modules so 1D method files can consume them).
# This file holds the gridded-specific machinery: the shared resolution loop
# (`_axis_anchors_loop!`/`_pooled`/`_all`).
#
# Each method owns its own payload type + `_axis_anchor_type` + `_resolve_anchor`
# + consuming kernels in its own file (gridded_linear.jl, gridded_constant.jl,
# gridded_hermite.jl, gridded_partials.jl). The backbone only calls those through
# the generic functions, so it never names a payload.

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
        op::AbstractEvalOp,
        ::Type{Tvals}
    ) where {Tvals}
    A = _axis_anchor_type(m, grid, targets, op, Tvals)
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
        ops,
        ::Type{T1},
        ::Type{Trest},
        ::Val{N}
    ) where {T1, Trest, N}
    exprs = [
        :(
                _axis_anchors_pooled(
                    pool, methods[$d], grids[$d], targets[$d], extraps[$d], $d,
                    ops[$d], $(d == 1 ? :T1 : :Trest)
                )
            )
            for d in 1:N
    ]
    return :(($(exprs...),))
end
