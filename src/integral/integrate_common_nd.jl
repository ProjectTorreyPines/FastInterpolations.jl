@inline function _normalize_bounds_nd(lo::Tuple{Vararg{Real, N}}, hi::Tuple{Vararg{Real, N}}) where {N}
    nflips = 0
    @inbounds for d in 1:N
        nflips += (lo[d] > hi[d])
    end
    sign = iseven(nflips) ? 1 : -1
    lo2 = ntuple(d -> @inbounds(min(lo[d], hi[d])), Val(N))
    hi2 = ntuple(d -> @inbounds(max(lo[d], hi[d])), Val(N))
    lo2 == hi2 && return (0, lo2, hi2)
    return sign, lo2, hi2
end

# Locate cell index ranges for ND integration bounds.
# Uses 3-arg `search_interval(s, grid, q)` which dispatches correctly for raw
# Range/Vector and wrapped grids alike — `_get_h` is read from the wrapped
# axis directly when available.
# @generated per-axis unroll (not `ntuple(Val(N)) do d`): the `do` closure takes
# `d` as a runtime Int, so `grids[d]` on a MIXED grid tuple (e.g. Vector × Range)
# returns the element Union and the downstream `search_interval` dispatches
# dynamically → boxing. Emitting literal `grids[1]`, `grids[2]` keeps each axis
# concrete. Pinned by the mixed-grid 0-alloc testitem in test_integral_nd_separable.jl.
@generated function _nd_cell_ranges(
        grids::NTuple{N, AbstractVector},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}},
        search_tuple,
        hint
    ) where {N}
    hd(d) = hint === Nothing ? :nothing : :(hint[$d])
    lo_e = [:(_nd_cell_index(grids[$d], lo[$d], search_tuple[$d], $(hd(d)))) for d in 1:N]
    hi_e = [:(_nd_cell_index(grids[$d], hi[$d], search_tuple[$d], $(hd(d)))) for d in 1:N]
    return quote
        Base.@_inline_meta
        (($(lo_e...),), ($(hi_e...),))
    end
end

@inline function _nd_cell_index(g, q, s, h)
    searcher = _resolve_search(g, q, s, h)
    i, _, _, _ = search_interval(searcher, g, q)
    return i
end

# ND domain check for integration bounds.
@inline _check_nd_integrate_domain(x::AbstractVector, xi::Real, ::NoExtrap) =
    _check_domain(x, xi, NoExtrap())

@inline function _check_nd_integrate_domain(x::AbstractVector, xi::Real, ::AbstractExtrap)
    x_min, x_max = first(x), last(x)
    (xi < x_min || xi > x_max) && throw(
        ArgumentError(
            "ND integration only supports in-domain bounds (extrapolation is not yet implemented). " *
                "Bound $xi is outside the grid domain [$x_min, $x_max]."
        )
    )
    return nothing
end

# Per-axis domain checks, @generated so `grids[d]` stays concrete on mixed grid
# tuples (a runtime-`d` loop leaks the element Union → dynamic dispatch → alloc,
# same class as `_nd_cell_ranges`).
@generated function _check_nd_bounds(grids, lo2, hi2, extraps, ::Val{N}) where {N}
    checks = Expr[]
    for d in 1:N
        push!(checks, :(_check_nd_integrate_domain(grids[$d], lo2[$d], extraps[$d])))
        push!(checks, :(_check_nd_integrate_domain(grids[$d], hi2[$d], extraps[$d])))
    end
    return quote
        Base.@_inline_meta
        @inbounds begin
            $(checks...)
        end
        nothing
    end
end

# Shared ND preamble: normalize bounds, domain checks, cell range computation.
@inline function _integrate_nd_preamble(
        grids, extraps, lo::Tuple{Vararg{Real, N}}, hi::Tuple{Vararg{Real, N}},
        search, hint
    ) where {N}
    sign, lo2, hi2 = _normalize_bounds_nd(lo, hi)
    _check_nd_bounds(grids, lo2, hi2, extraps, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N), lo)  # NTuple{N,Real} <: Tuple → BinarySearch/axis
    idx_lo, idx_hi = _nd_cell_ranges(grids, lo2, hi2, search_tuple, hint)
    return (sign, lo2, hi2, idx_lo, idx_hi)
end
