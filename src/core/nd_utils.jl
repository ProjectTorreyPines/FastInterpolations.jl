# ========================================
# ND Shared Utilities
# ========================================
#
# Common utility functions for N-dimensional interpolation.
# These are shared across all ND algorithms (constant, linear, cubic)
# and dimensions.
#
# _resolve_* pattern: Convert flexible user input to canonical N-tuple form.
# - Single value → broadcast to all N axes
# - NTuple{N} → passthrough (with optional validation)
# - Wrong-sized tuple → ArgumentError


# ========================================
# Dimension Mismatch (non-empty Vararg guard)
# ========================================

@noinline _throw_ndims_mismatch(label::String, expected::Int, got::Int) =
    throw(DimensionMismatch("expected $expected $label, got $got"))

# ========================================
# Extrapolation Resolution
# ========================================
#
# _resolve_extrap_nd(extrap, bcs, Val(N)):
# Converts user-facing extrap input to NTuple{N, AbstractExtrap}.
# The `bcs` argument enables periodic BC validation and override:
# - `nothing` for constant/linear (no BCs)
# - NTuple{N, AbstractBC} for quadratic/cubic

# ── Fill-value FillExtrap validation for ND ────────────────────────────
# All fill-value axes must have the same value (prevents ambiguity).
# ClampExtrap (boundary clamp) is always OK and not counted.
@noinline _throw_conflicting_fill_values() = throw(
    ArgumentError(
        "All FillExtrap fill values in ND must be identical; " *
            "got conflicting fill values on different axes"
    )
)

@generated function _validate_fill_values_nd(extraps::E) where {E <: Tuple{Vararg{AbstractExtrap}}}
    N = fieldcount(E)
    fill_dims = [d for d in 1:N if fieldtype(E, d) <: FillExtrap]
    length(fill_dims) <= 1 && return :(nothing)
    first_d = fill_dims[1]
    checks = [:(extraps[$first_d].fill_value === extraps[$d].fill_value || _throw_conflicting_fill_values()) for d in fill_dims[2:end]]
    return quote
        $(checks...)
        nothing
    end
end

# ── FillExtrap OOB short-circuit for ND ────────────────────────────────
#
# When any FillExtrap axis is OOB, the interpolant returns its fill value
# (for EvalValue) or zero (for any derivative op).
# ClampExtrap just clamps via _handle_axis_extrap — no special OOB logic.
# All helpers are @generated for compile-time dead-code elimination.

"""
    _is_fill_oob(query, grids, extraps) -> Bool

Compile-time selective OOB check for FillExtrap axes only.
Returns `false` at compile time when no axis has FillExtrap (dead-code eliminated).
"""
@generated function _is_fill_oob(
        query::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        extraps::E
    ) where {N, E <: Tuple{Vararg{AbstractExtrap, N}}}
    fill_dims = [d for d in 1:N if fieldtype(E, d) <: FillExtrap]
    isempty(fill_dims) && return :(false)

    oob_checks = [
        :(
                let qp = _extract_primal(query[$d])
                    qp < first(grids[$d]) || qp > last(grids[$d])
            end
            ) for d in fill_dims
    ]
    oob_expr = length(oob_checks) == 1 ? oob_checks[1] :
        foldl((a, b) -> :($a || $b), oob_checks)

    return quote
        Base.@_inline_meta
        @inbounds $oob_expr
    end
end

"""
    _try_fill_oob(query, grids, extraps, ops, zero_ref) -> Union{Nothing, result}

FillExtrap OOB short-circuit for ND evaluation. Returns `nothing` to proceed
normally, or the fill/zero result to short-circuit.

Compile-time eliminated when no axis has FillExtrap.
Accepts both scalar `ops::AbstractEvalOp` and tuple `ops::Tuple{Vararg{AbstractEvalOp}}`.
"""
@generated function _try_fill_oob(
        query::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        extraps::E,
        ops,
        zero_ref
    ) where {N, E <: Tuple{Vararg{AbstractExtrap, N}}}
    fill_dims = [d for d in 1:N if fieldtype(E, d) <: FillExtrap]
    isempty(fill_dims) && return :(nothing)

    oob_checks = [
        :(
                let qp = _extract_primal(query[$d])
                    qp < first(grids[$d]) || qp > last(grids[$d])
            end
            ) for d in fill_dims
    ]
    oob_expr = length(oob_checks) == 1 ? oob_checks[1] :
        foldl((a, b) -> :($a || $b), oob_checks)

    fill_d = fill_dims[1]
    return quote
        Base.@_inline_meta
        @inbounds if $oob_expr
            return _fill_extrap_result(ops, extraps[$fill_d].fill_value, zero_ref, query[1])
        end
        return nothing
    end
end

# FillExtrap result dispatch: value for EvalValue, zero for any derivative.
# Uses `0 * zero_ref` (not `0 * fill_val`) to handle NaN fill values correctly.
# 4th arg `qe` (query element) promotes result to kernel return type for mixed-precision.
# Uses _promote_extrap_val/_promote_extrap_zero from core/utils.jl (Number dispatch).
@inline _fill_extrap_result(::EvalValue, fill_val, _, qe) = _promote_extrap_val(fill_val, qe)
@inline _fill_extrap_result(::AbstractEvalOp, _, zero_ref, qe) = _promote_extrap_zero(zero_ref, qe)
@inline function _fill_extrap_result(ops::Tuple{Vararg{AbstractEvalOp}}, fill_val, zero_ref, qe)
    for i in 1:length(ops)
        @inbounds ops[i] isa EvalValue || return _promote_extrap_zero(zero_ref, qe)
    end
    return _promote_extrap_val(fill_val, qe)
end

# Extract fill_value from the first FillExtrap in extraps tuple.
# Only called on OOB cold path (guarded by _is_fill_oob).
@inline function _first_fill_value(extraps::Tuple)
    for e in extraps
        e isa FillExtrap && return e.fill_value
    end
    error("unreachable: _first_fill_value called without FillExtrap")
end

# ── Mode → Mode tuple, then promote fill values ───────────────────────

@inline function _resolve_extrap_nd(extrap::AbstractExtrap, ::Nothing, ::Val{N}, ::Type{Tv}) where {N, Tv}
    result = ntuple(_ -> extrap, Val(N))
    _validate_fill_values_nd(result)
    return _promote_extraps_nd(result, Tv)
end

@inline function _resolve_extrap_nd(extrap::AbstractExtrap, bcs::Tuple{Vararg{AbstractBC, N}}, ::Val{N}, ::Type{Tv}) where {N, Tv}
    _check_mode_periodic_compat(extrap, bcs, Val(N))
    result = _mode_to_modes_with_periodic(extrap, bcs)
    _validate_fill_values_nd(result)
    return _promote_extraps_nd(result, Tv)
end

@inline function _resolve_extrap_nd(extrap::Tuple{Vararg{AbstractExtrap, N}}, ::Nothing, ::Val{N}, ::Type{Tv}) where {N, Tv}
    _validate_fill_values_nd(extrap)
    return _promote_extraps_nd(extrap, Tv)
end

@inline function _resolve_extrap_nd(extrap::Tuple{Vararg{AbstractExtrap, N}}, bcs::Tuple{Vararg{AbstractBC, N}}, ::Val{N}, ::Type{Tv}) where {N, Tv}
    _check_modes_periodic_compat(extrap, bcs, Val(N))
    result = _modes_to_modes_with_periodic(extrap, bcs)
    _validate_fill_values_nd(result)
    return _promote_extraps_nd(result, Tv)
end

@noinline function _resolve_extrap_nd(extrap::Tuple{Vararg{AbstractExtrap}}, ::Any, ::Val{N}, ::Type) where {N}
    throw(ArgumentError("extrap tuple must have $N elements to match grid dimensions, got $(length(extrap))"))
end

@generated function _promote_extraps_nd(extraps::E, ::Type{Tv}) where {E <: Tuple{Vararg{AbstractExtrap}}, Tv}
    N = fieldcount(E)
    exprs = [:(FastInterpolations._promote_extrap(extraps[$d], Tv)) for d in 1:N]
    return :(($(exprs...),))
end

# ── Periodic BC compatibility checks for Mode types ──────────────────

@inline function _check_mode_periodic_compat(extrap::AbstractExtrap, bcs::Tuple{Vararg{AbstractBC, N}}, ::Val{N}) where {N}
    # NoExtrap and WrapExtrap are always compatible with PeriodicBC
    (extrap isa NoExtrap || extrap isa WrapExtrap) && return nothing
    for d in 1:N
        _is_periodic_bc(bcs[d]) && _throw_periodic_extrap_mismatch(d, extrap)
    end
    return nothing
end

@inline function _check_modes_periodic_compat(extraps::Tuple{Vararg{AbstractExtrap, N}}, bcs::Tuple{Vararg{AbstractBC, N}}, ::Val{N}) where {N}
    for d in 1:N
        if _is_periodic_bc(bcs[d]) && !(extraps[d] isa NoExtrap || extraps[d] isa WrapExtrap)
            _throw_periodic_extrap_mismatch(d, extraps[d])
        end
    end
    return nothing
end

# ── @generated periodic override (compile-time Mode tuple construction) ──

@generated function _mode_to_modes_with_periodic(extrap::M, bcs::B) where {M <: AbstractExtrap, B <: Tuple{Vararg{AbstractBC}}}
    N = fieldcount(B)
    exprs = map(1:N) do d
        if fieldtype(B, d) <: PeriodicBC
            :(WrapExtrap())
        else
            :(extrap)
        end
    end
    return :(($(exprs...),))
end

@generated function _modes_to_modes_with_periodic(extraps::E, bcs::B) where {E <: Tuple{Vararg{AbstractExtrap}}, B <: Tuple{Vararg{AbstractBC}}}
    N = fieldcount(E)
    exprs = map(1:N) do d
        if fieldtype(B, d) <: PeriodicBC
            :(WrapExtrap())
        else
            :(extraps[$d])
        end
    end
    return :(($(exprs...),))
end


# ========================================
# Search Policy Resolution
# ========================================

"""
    _resolve_search_nd(search, Val(N)) -> NTuple{N, AbstractSearchPolicy}

Resolve search policy input to canonical N-tuple (broadcast only, no AutoSearch resolution).
- Single `AbstractSearchPolicy` → broadcast to all N axes
- `NTuple{N, AbstractSearchPolicy}` → passthrough

    _resolve_search_nd(search, Val(N), query_sample) -> NTuple{N, AbstractSearchPolicy}

Broadcast + resolve AutoSearch in one step. Pass the query container directly — no `first()` extraction needed:
- `query_sample::NTuple{N,Real}` (scalar ND query) → `Tuple` arm → `BinarySearch()` per axis
- `query_sample::NTuple{N,AbstractVector}` (SoA batch) → `Tuple{Vararg{AbstractVector}}` arm → `LinearBinarySearch()` per axis
- `query_sample::AbstractVector{<:Tuple}` (AoS batch) → `AbstractVector` arm → `LinearBinarySearch()` per axis
- Explicit policies pass through unchanged.
"""
@inline _resolve_search_nd(s::AbstractSearchPolicy, ::Val{N}) where {N} = ntuple(_ -> s, Val(N))

@inline _resolve_search_nd(s::NTuple{N, AbstractSearchPolicy}, ::Val{N}) where {N} = s

@noinline function _resolve_search_nd(s::Tuple{Vararg{AbstractSearchPolicy}}, ::Val{N}) where {N}
    throw(ArgumentError("search tuple must have $N elements to match grid dimensions, got $(length(s))"))
end

# 3-arg: broadcast + resolve AutoSearch per-axis in one step
@inline function _resolve_search_nd(s, ::Val{N}, query_sample) where {N}
    tuple = _resolve_search_nd(s, Val(N))
    return map(p -> _resolve_search_policy(p, query_sample), tuple)
end

# 4-arg form: adaptive ND resolution with hint awareness.
# Hinted cases fall through to the 3-arg form above (no monotonicity check).
@inline _resolve_search_nd(s, vn, queries, hints) = _resolve_search_nd(s, vn, queries)

# SoA Real vectors + no hint → per-axis adaptive resolution (cache-friendly).
# Each axis independently checks monotonicity via 1D _resolve_search_policy(policy, vec, nothing):
#   AutoSearch axes → _is_likely_monotone per axis → BinarySearch or LinearBinarySearch
#   Explicit policy axes → passthrough unchanged
# Uses map (not ntuple closure) to avoid closure heap allocation.
@inline function _resolve_search_nd(
        s, ::Val{N},
        queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
        ::Nothing
    ) where {N}
    tuple = _resolve_search_nd(s, Val(N))
    return map(_resolve_search_nohint, tuple, queries)
end

# Named helper for map — avoids closure capture in _resolve_search_nd.
# Forwards to 3-arg _resolve_search_policy with hint=nothing, triggering adaptive
# monotonicity check for AutoSearch axes.
@inline _resolve_search_nohint(p, q) = _resolve_search_policy(p, q, nothing)

# Generic queries + no hint → per-axis adaptive using query protocol.
# Fallback for non-SoA queries — SoA has a more specific method above (line ~278).
# Uses _is_axis_likely_monotone (protocol-based) instead of _is_likely_monotone (vector-based).
@inline function _resolve_search_nd(s, vn::Val{N}, queries, ::Nothing) where {N}
    tuple = _resolve_search_nd(s, vn)
    any(p -> p isa AutoSearch, tuple) || return tuple
    return ntuple(Val(N)) do d
        p = tuple[d]
        if p isa AutoSearch
            _is_axis_likely_monotone(queries, d, vn) ? LinearBinarySearch() : BinarySearch()
        else
            p
        end
    end
end

# ── Protocol-based per-axis monotonicity check ──
#
# Uses _query_extract to check first K elements along axis d.
# Works for any query type implementing the protocol (AoS, SVector, custom).
# SoA has a more cache-friendly specialization via _is_likely_monotone(q[d]).

@inline function _is_axis_likely_monotone(
        queries, d::Int, ::Val{N}, ::Val{K} = Val(8)
    ) where {N, K}
    nq = _query_length(queries)
    nq < K && return false
    @inbounds begin
        v1 = _query_extract(queries, 1)[d]
        v2 = _query_extract(queries, 2)[d]
        ascending = v2 >= v1
        prev = v2
        if ascending
            for i in 3:K
                curr = _query_extract(queries, i)[d]
                curr < prev && return false
                prev = curr
            end
        else
            for i in 3:K
                curr = _query_extract(queries, i)[d]
                curr > prev && return false
                prev = curr
            end
        end
    end
    return true
end

# ----------------------------------------
# All-or-Nothing Adaptive Resolution (Oneshot)
# ----------------------------------------
#
# For oneshot paths, per-axis adaptive creates Tuple{Union{BinarySearch,LB}, ...}
# — per-element Union that Julia boxes during tuple construction (144+ bytes).
#
# Solution: all-or-nothing — check all AutoSearch axes, return uniform type.
# If ALL AutoSearch axes are monotone → all AutoSearch → LinearBinarySearch.
# If ANY AutoSearch axis is non-monotone → all AutoSearch → BinarySearch.
# Explicit (non-AutoSearch) policies pass through unchanged.
#
# Return type is Union{ConcreteA, ConcreteB} — a 2-way Union of concrete tuple
# types that Julia union-splits at the function barrier.

# Named helpers for map — avoid closure capture.
@inline _autosearch_to_lb(::AutoSearch) = LinearBinarySearch()
@inline _autosearch_to_lb(p::AbstractSearchPolicy) = p
@inline _autosearch_to_binary(::AutoSearch) = BinarySearch()
@inline _autosearch_to_binary(p::AbstractSearchPolicy) = p
@inline _check_axis_monotone(::AutoSearch, q) = _is_likely_monotone(q)
@inline _check_axis_monotone(::AbstractSearchPolicy, _) = true

# SoA Real vectors + no hint → all-or-nothing adaptive resolution.
@inline function _resolve_search_nd_uniform(
        s, ::Val{N},
        queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
        ::Nothing
    ) where {N}
    tuple = _resolve_search_nd(s, Val(N))  # broadcast only, no resolution
    all_mono = all(map(_check_axis_monotone, tuple, queries))
    return all_mono ? map(_autosearch_to_lb, tuple) : map(_autosearch_to_binary, tuple)
end

# Generic queries + no hint → protocol-based all-or-nothing adaptive resolution.
# Fallback for non-SoA queries — SoA has a more specific method above (line ~365).
@inline function _resolve_search_nd_uniform(s, vn::Val{N}, queries, ::Nothing) where {N}
    tuple = _resolve_search_nd(s, vn)
    any(p -> p isa AutoSearch, tuple) || return tuple
    all_mono = all(
        ntuple(Val(N)) do d
            p = tuple[d]
            !isa(p, AutoSearch) || _is_axis_likely_monotone(queries, d, vn)
        end
    )
    return all_mono ? map(_autosearch_to_lb, tuple) : map(_autosearch_to_binary, tuple)
end

# Hinted → standard 3-arg type-based (already concrete, no monotonicity check).
@inline _resolve_search_nd_uniform(s, vn, queries, hints) = _resolve_search_nd(s, vn, queries)

# ========================================
# Boundary Condition Resolution
# ========================================

"""
    _resolve_bcs_nd(bc, Val(N)) -> NTuple{N, AbstractBC}

Resolve boundary condition input to canonical N-tuple.
- Single `AbstractBC` → broadcast to all N axes
- `NTuple{N, AbstractBC}` → passthrough
"""
@inline _resolve_bcs_nd(bc::AbstractBC, ::Val{N}) where {N} = ntuple(_ -> bc, Val(N))

@inline _resolve_bcs_nd(bc::NTuple{N, AbstractBC}, ::Val{N}) where {N} = bc

@noinline function _resolve_bcs_nd(bc::Tuple{Vararg{AbstractBC}}, ::Val{N}) where {N}
    throw(ArgumentError("bc tuple must have $N elements to match grid dimensions, got $(length(bc))"))
end

# ========================================
# Side Resolution (for Constant ND)
# ========================================

"""
    _resolve_side_nd(side, Val(N)) -> NTuple{N, AbstractSide}

Resolve side selection to canonical N-tuple.
- Single `AbstractSide` → broadcast to all N axes
- `NTuple{N, AbstractSide}` → passthrough
"""
@inline _resolve_side_nd(side::AbstractSide, ::Val{N}) where {N} = ntuple(_ -> side, Val(N))

@inline _resolve_side_nd(side::NTuple{N, AbstractSide}, ::Val{N}) where {N} = side

@noinline function _resolve_side_nd(side::Tuple{Vararg{AbstractSide}}, ::Val{N}) where {N}
    throw(ArgumentError("side tuple must have $N elements to match grid dimensions, got $(length(side))"))
end

# ========================================
# Derivative Order Resolution (ND)
# ========================================

"""
    _resolve_deriv_nd(deriv, Val(N)) -> NTuple{N, DerivOp}

Resolve derivative specification to canonical N-tuple of DerivOp singletons.
- Single `DerivOp` → broadcast to all N axes
- `Tuple{Vararg{DerivOp, N}}` → passthrough

# Examples
```julia
_resolve_deriv_nd(DerivOp(1), Val(2))        # → (DerivOp{1}(), DerivOp{1}())
_resolve_deriv_nd(DerivOp(1, 0), Val(2))     # → (DerivOp{1}(), DerivOp{0}())  passthrough
```
"""
@inline _resolve_deriv_nd(op::DerivOp, ::Val{N}) where {N} = ntuple(_ -> op, Val(N))

@inline _resolve_deriv_nd(ops::Tuple{Vararg{DerivOp, N}}, ::Val{N}) where {N} = ops

@noinline function _resolve_deriv_nd(ops::Tuple{Vararg{DerivOp}}, ::Val{N}) where {N}
    throw(ArgumentError("deriv tuple must have $N elements to match grid dimensions, got $(length(ops))"))
end

# ========================================
# Grid Validation Helpers
# ========================================

"""
    _validate_nd_grids(grids::NTuple{N}, data::AbstractArray{<:Any,N})

Validate that grid lengths match data dimensions.

Uses @generated to avoid closure boxing when iterating over heterogeneous grid tuples.
"""
@generated function _validate_nd_grids(grids::NTuple{N, AbstractVector}, data::AbstractArray{<:Any, N}) where {N}
    checks = [
        quote
                ng = length(grids[$i])
                nd = size(data, $i)
                if ng != nd
                    throw(
                        DimensionMismatch(
                            "Grid $($i) has " * string(ng) * " points but data dimension $($i) has size " * string(nd)
                        )
                    )
            end
                if ng < 2
                    throw(ArgumentError("Grid $($i) must have at least 2 points, got " * string(ng)))
            end
            end for i in 1:N
    ]

    return quote
        $(checks...)
        nothing
    end
end

# ========================================
# Per-Axis Extrapolation Handling
# ========================================
#
# Shared extrapolation logic for all ND interpolation methods.
# Handles domain boundary conditions before interval search.

"""
    _handle_all_extraps(queries, grids, extraps) -> NTuple{N}

Apply extrapolation handling to all query coordinates.
Returns tuple of processed query values ready for interpolation.

Accepts heterogeneous tuples (e.g., mixed grid types, per-axis extrap modes).
Uses map over named helper so each axis receives its concrete type directly,
avoiding ntuple-closure boxing on heterogeneous tuple inputs.
"""
# GridIdx: in-domain by construction (bounds-checked at resolution), skip extrap entirely
@inline _extrap_axis(q::GridIdx, grid, extrap) = q
@inline _extrap_axis(q, grid, extrap) = @inbounds _handle_axis_extrap(q, grid, extrap)

@inline function _handle_all_extraps(
        queries::Tuple{Vararg{Real, N}}, grids::Tuple{Vararg{AbstractVector, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}}
    ) where {N}
    return map(_extrap_axis, queries, grids, extraps)
end

# Per-extrap type handling (unconstrained q for duck-type compatibility: Dual, etc.)
@inline function _handle_axis_extrap(q, axis::AbstractVector, ::NoExtrap)
    @boundscheck _check_domain(axis, q, NoExtrap())
    return q
end

@inline function _handle_axis_extrap(q, axis::AbstractVector{Tg}, ::_ClampOrFill) where {Tg}
    q_primal = _extract_primal(q)
    lo, hi = first(axis), last(axis)
    q_primal < lo && return oftype(q, lo)
    q_primal > hi && return oftype(q, hi)
    return q
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::ExtendExtrap)
    return q
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::WrapExtrap)
    return _wrap_to_domain(q, first(axis), last(axis))
end

# ========================================
# Per-Axis Interval Search
# ========================================
#
# Shared interval search logic for all ND interpolation methods.
# Finds the cell containing each query coordinate.

"""
    _search_all_intervals(q_evals, grids, spacings, searches) -> (indices, Ls, Rs)

Perform interval search on all axes.
Returns tuples of: indices (cell index), Ls (left bounds), Rs (right bounds).

Accepts heterogeneous tuples (e.g., mixed grid types, spacing types, search policies).
Uses map over named helpers so each axis receives its concrete type directly,
avoiding ntuple-closure boxing on heterogeneous tuple inputs.
"""

# Named helpers for map-based search — each receives concrete types per axis.
# search_interval returns (idx, L, R) with the same concrete element type regardless
# of spacing type (ScalarSpacing or VectorSpacing), so results is homogeneous.
@inline _search_axis(q, grid, spacing, search) =
    @inbounds search_interval(_resolve_search(grid, q, search, nothing), grid, spacing, q)
@inline _search_axis_hint(q, grid, spacing, search, hint) =
    @inbounds search_interval(_resolve_search(grid, q, search, hint), grid, spacing, q)
@inline _getidx(r) = r[1]
@inline _getL(r) = r[2]
@inline _getR(r) = r[3]

@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}}, grids::Tuple{Vararg{AbstractVector, N}},
        spacings::Tuple{Vararg{AbstractGridSpacing, N}}, searches::Tuple{Vararg{AbstractSearchPolicy, N}}
    ) where {N}
    results = map(_search_axis, q_evals, grids, spacings, searches)
    return (map(_getidx, results), map(_getL, results), map(_getR, results))
end

# ----------------------------------------
# Hint-aware overloads for persistent search state
# ----------------------------------------

"""
    _get_axis_hint(hints, d) -> Nothing or Base.RefValue{Int}

Extract per-axis hint from a hint tuple or Nothing.
Used by N=2 specializations that destructure manually.
"""
@inline _get_axis_hint(::Nothing, d) = nothing
@inline _get_axis_hint(hints::Tuple, d) = @inbounds hints[d]

# Nothing hint → delegate to existing 4-arg (zero overhead)
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}}, grids::Tuple{Vararg{AbstractVector, N}},
        spacings::Tuple{Vararg{AbstractGridSpacing, N}}, searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        ::Nothing
    ) where {N}
    return _search_all_intervals(q_evals, grids, spacings, searches)
end

# Tuple hint → use 2-arg _to_searcher(policy, hint) per axis
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}}, grids::Tuple{Vararg{AbstractVector, N}},
        spacings::Tuple{Vararg{AbstractGridSpacing, N}}, searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}}
    ) where {N}
    results = map(_search_axis_hint, q_evals, grids, spacings, searches, hints)
    return (map(_getidx, results), map(_getL, results), map(_getR, results))
end

# ========================================
# N=2 Specialized Cell Location Preamble
# ========================================
#
# Shared 2D preamble for all N=2 _locate_cell specializations.
# Extracts query, handles extrapolation, and performs interval search.
# Returns raw (x_eval, y_eval, ix, iy, xL, yL) for type-specific post-processing.

"""
    _locate_cell_2d_preamble(query, grids, spacings, extraps, search, hints)

Shared preamble for all N=2 `_locate_cell` specializations.
Destructures 2D query, applies per-axis extrapolation, and performs interval search.

Returns `(x_eval, y_eval, ix, iy, xL, yL)` — the 6 raw values that each
interpolant type then post-processes into its kernel-specific cell tuple.
"""
@inline function _locate_cell_2d_preamble(
        query::Tuple{Vararg{Real, 2}},
        grids, spacings, extraps,
        search::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
        hints
    )
    xq, yq = query
    grid_x, grid_y = grids
    spacing_x, spacing_y = spacings
    extrap_x, extrap_y = extraps
    search_x, search_y = search

    x_eval = _handle_axis_extrap(xq, grid_x, extrap_x)
    y_eval = _handle_axis_extrap(yq, grid_y, extrap_y)

    hint_x = _get_axis_hint(hints, 1)
    hint_y = _get_axis_hint(hints, 2)
    searcher_x = _resolve_search(grid_x, x_eval, search_x, hint_x)
    searcher_y = _resolve_search(grid_y, y_eval, search_y, hint_y)
    ix, xL, _ = search_interval(searcher_x, grid_x, spacing_x, x_eval)
    iy, yL, _ = search_interval(searcher_y, grid_y, spacing_y, y_eval)

    return (x_eval, y_eval, ix, iy, xL, yL)
end

# ========================================
# Zero-Allocation Grid Type Helpers
# ========================================
#
# @generated functions to avoid closure boxing in tuple operations.
# These replace map/ntuple patterns that cause allocations due to
# capturing heterogeneous tuple variables.

# ========================================
# Shared ND Coefficient Storage
# ========================================

"""
    _NodalDerivativesND{Tv, N, NP1}

Storage for precomputed N-dimensional partial derivatives at grid nodes.

This structure stores the function value f and all mixed partial derivatives
(∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁∂x₂, etc.) at each grid node, enabling O(1) evaluation
via tensor-product interpolation (Hermite for cubic, quadratic kernel for quadratic).

# Type Parameters
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `NP1`: N + 1 (array dimensionality, Julia can't compute N+1 in type definition)

# Fields
- `partials::Array{Tv, NP1}`: Partial derivatives array of shape (2^N, n₁, n₂, ..., nₙ)

# Partials Indexing Convention (bit-encoding of derivatives)
The first index `p` encodes which partial derivative via binary representation:
- Bit d set → differentiate with respect to dimension d
- p=1 (binary 0...0): f (no derivatives)
- p=2 (binary 0...1): ∂f/∂x₁
- p=3 (binary 0..10): ∂f/∂x₂
- p=4 (binary 0..11): ∂²f/∂x₁∂x₂
- etc.

# Examples
For N=2 (4 partials per point):
- `partials[1, i, j]` = f(xᵢ, yⱼ)         (p=1, binary 00)
- `partials[2, i, j]` = ∂f/∂x             (p=2, binary 01)
- `partials[3, i, j]` = ∂f/∂y             (p=3, binary 10)
- `partials[4, i, j]` = ∂²f/∂x∂y          (p=4, binary 11)

For N=3 (8 partials per point):
- `partials[1, i, j, k]` = f              (p=1, binary 000)
- `partials[2, i, j, k]` = ∂f/∂x          (p=2, binary 001)
- `partials[3, i, j, k]` = ∂f/∂y          (p=3, binary 010)
- `partials[4, i, j, k]` = ∂²f/∂x∂y       (p=4, binary 011)
- `partials[5, i, j, k]` = ∂f/∂z          (p=5, binary 100)
- `partials[6, i, j, k]` = ∂²f/∂x∂z       (p=6, binary 101)
- `partials[7, i, j, k]` = ∂²f/∂y∂z       (p=7, binary 110)
- `partials[8, i, j, k]` = ∂³f/∂x∂y∂z     (p=8, binary 111)
"""
struct _NodalDerivativesND{Tv, N, NP1} <: AbstractArray{Tv, NP1}
    partials::Array{Tv, NP1}

    function _NodalDerivativesND{Tv, N, NP1}(partials::Array{Tv, NP1}) where {Tv, N, NP1}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1, got NP1=$NP1, N=$N"))
        size(partials, 1) == (1 << N) || throw(
            DimensionMismatch(
                "First dimension must be 2^N=$(1 << N), got $(size(partials, 1))"
            )
        )
        return new{Tv, N, NP1}(partials)
    end
end

# Convenience constructor that computes NP1 automatically
function _NodalDerivativesND{Tv, N}(partials::Array{Tv, NP1}) where {Tv, N, NP1}
    return _NodalDerivativesND{Tv, N, NP1}(partials)
end

# AbstractArray interface — size + getindex + setindex! gives everything else for free
Base.size(nd::_NodalDerivativesND) = size(nd.partials)
Base.getindex(nd::_NodalDerivativesND, i...) = nd.partials[i...]
Base.setindex!(nd::_NodalDerivativesND, v, i...) = (nd.partials[i...] = v)
Base.similar(nd::_NodalDerivativesND) = _NodalDerivativesND{eltype(nd.partials), ndims(nd.partials) - 1}(similar(nd.partials))
Base.similar(nd::_NodalDerivativesND, ::Type{T}, dims::Dims) where {T} = similar(nd.partials, T, dims)

# ========================================
# Shared ND Local Parameter Computation
# ========================================

"""
    _compute_all_local_params(q_evals, spacings, indices, Ls) -> (hs, inv_hs, dLs)

Compute local cell parameters for all axes.
Returns tuples of: hs (cell widths), inv_hs (reciprocals), dLs (left deltas).

Used by both CubicInterpolantND and QuadraticInterpolantND evaluation.
"""
@inline function _compute_all_local_params(
        q_evals::Tuple{Vararg{Real, N}},  # Allow heterogeneous/AD types (Dual) and GridIdx
        spacings::Tuple{Vararg{AbstractGridSpacing, N}},  # Allow heterogeneous spacing types (VectorSpacing, ScalarSpacing)
        indices::NTuple{N, Int},
        Ls::Tuple{Vararg{Real, N}}  # Grid boundary (allow heterogeneous Real types)
    ) where {N}
    hs = ntuple(Val(N)) do d
        @inbounds _get_h(spacings[d], indices[d])
    end
    inv_hs = ntuple(Val(N)) do d
        @inbounds _get_inv_h(spacings[d], indices[d])
    end
    dLs = ntuple(Val(N)) do d
        @inbounds q_evals[d] - Ls[d]
    end
    return (hs, inv_hs, dLs)
end

# ========================================
# @generated Tensor-Product Code Generation Helpers
# ========================================
#
# Used by @generated tensor-product kernels (cubic_nd_eval.jl, quadratic_nd_eval.jl)
# to build AST at compile time. These are NOT called at runtime.

"""
    _varname(stage, corner, deriv) -> Symbol

Generate a variable name for the tensor-product dimension-collapsing stages.
E.g., `_varname(2, 0, 1)` → `:g_2_0_1` (stage 2, corner 0, deriv 1).
"""
_varname(stage::Int, corner::Int, deriv::Int) = Symbol("g_$(stage)_$(corner)_$(deriv)")

"""
    _partial_index(deriv_bits) -> Int

Convert derivative bitmask to 1-based partials array index.
The bitmask encodes which dimensions are differentiated (bit d set → ∂/∂xd).
"""
_partial_index(deriv_bits::Int) = deriv_bits + 1

"""
    _corner_offset_expr(corner_bits, N) -> Vector{Int}

Convert corner bitmask to per-dimension 0/1 offsets.
Used to index into the 2^N corners of an N-dimensional cell.
E.g., for N=3, corner_bits=5 (binary 101) → [1, 0, 1].
"""
function _corner_offset_expr(corner_bits::Int, N::Int)
    return [((corner_bits >> (d - 1)) & 1) for d in 1:N]
end

# ========================================
# @generated Grid Type Promotion
# ========================================

"""
    _promote_grid_eltype(grids::NTuple{N, AbstractVector}) -> Type

Zero-allocation promoted element type extraction from grid tuple.
Generates unrolled `promote_type(eltype(grids[1]), eltype(grids[2]), ...)` at compile time.
"""
@generated function _promote_grid_eltype(grids::NTuple{N, AbstractVector}) where {N}
    types = [:(eltype(grids[$i])) for i in 1:N]
    return :(promote_type($(types...)))
end

"""
    _convert_grid(x, Tg) -> AbstractVector{Tg}

Convert grid to target float type, preserving Range type where possible.
Used by all ND interpolation methods via `_convert_grids_typed`.
"""
function _convert_grid(x::AbstractRange, ::Type{Tg}) where {Tg}
    # Always normalize to _CachedRange via _to_float (identity for _CachedRange{Tg}).
    return _to_float(x, Tg)
end

function _convert_grid(x::AbstractVector, ::Type{Tg}) where {Tg}
    eltype(x) === Tg && return x
    return Tg.(x)
end

"""
    _convert_grids_typed(grids::NTuple{N, AbstractVector}, ::Type{Tg}) -> NTuple{N}

Zero-allocation grid conversion to target element type.
Generates unrolled `(_convert_grid(grids[1], Tg), _convert_grid(grids[2], Tg), ...)` at compile time.
"""
@generated function _convert_grids_typed(grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    exprs = [:(FastInterpolations._convert_grid(grids[$i], Tg)) for i in 1:N]
    return :(($(exprs...),))
end

"""
    _create_spacings_typed(grids::NTuple{N, AbstractVector}) -> NTuple{N}

Zero-allocation spacing creation from grid tuple.
Generates unrolled `(_create_spacing(grids[1]), _create_spacing(grids[2]), ...)` at compile time.

Avoids closure boxing that occurs with `ntuple(d -> _create_spacing(grids[d]), Val(N))`
when grids is a heterogeneous tuple (e.g., mix of Range and Vector).
"""
@generated function _create_spacings_typed(grids::NTuple{N, AbstractVector}) where {N}
    exprs = [:(FastInterpolations._create_spacing(grids[$i])) for i in 1:N]
    return :(($(exprs...),))
end

# ========================================
# Pool-Based Spacing Creation (ND Oneshot)
# ========================================
#
# Variants of _create_spacing that acquire h/inv_h vectors from an
# AdaptiveArrayPools pool instead of heap-allocating.
# Used exclusively inside @with_pool scopes in ND oneshot paths.
# Range grids (ScalarSpacing) are zero-alloc regardless.

"""
    _create_spacing_pooled(pool, x::AbstractRange{T}) -> ScalarSpacing{T}

Pool-aware spacing for Range grids. Delegates to `_create_spacing` since
ScalarSpacing is already zero-allocation (two scalar values).
"""
@inline _create_spacing_pooled(pool::AbstractArrayPool, x::AbstractRange{T}) where {T <: AbstractFloat} = _create_spacing(x)

"""
    _create_spacing_pooled(pool, x::AbstractVector{T}) -> VectorSpacing{T}

Pool-aware spacing for Vector grids. Acquires `h` and `inv_h` arrays
from the pool instead of heap-allocating. The pool buffers are released
automatically when the enclosing `@with_pool` scope exits.
"""
@inline function _create_spacing_pooled(pool::AbstractArrayPool, x::AbstractVector{T}) where {T}
    n = length(x)
    h = acquire!(pool, T, n - 1)
    inv_h = acquire!(pool, T, n - 1)

    @inbounds for i in 1:(n - 1)
        h[i] = x[i + 1] - x[i]
        inv_h[i] = inv(h[i])
    end

    return VectorSpacing{T}(h, inv_h)
end

"""
    _create_spacings_pooled(pool, grids::NTuple{N, AbstractVector}) -> NTuple{N}

Pool-aware spacing creation from grid tuple.
Generates unrolled per-axis calls to `_create_spacing_pooled` at compile time.
For Range grids, no pool touch (ScalarSpacing is zero-alloc).
For Vector grids, h/inv_h are acquired from pool (zero heap alloc).
"""
@generated function _create_spacings_pooled(pool::AbstractArrayPool, grids::NTuple{N, AbstractVector}) where {N}
    exprs = [:(FastInterpolations._create_spacing_pooled(pool, grids[$i])) for i in 1:N]
    return :(($(exprs...),))
end
