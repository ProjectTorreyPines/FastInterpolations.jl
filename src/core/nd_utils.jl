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

    # Per-axis OOB via the shared `_oob_state` (widened `_CachedRange` bracket →
    # a query at the true endpoint is in-domain, matching the 1D paths).
    oob_checks = [:(_oob_state(grids[$d], query[$d]) != IN_DOMAIN) for d in fill_dims]
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

    # Per-axis OOB via the shared `_oob_state` (widened `_CachedRange` bracket →
    # a query at the true endpoint is in-domain, matching the 1D paths).
    oob_checks = [:(_oob_state(grids[$d], query[$d]) != IN_DOMAIN) for d in fill_dims]
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
# Both branches source from `fill_val` — the OOB cell's "data" IS the
# fill_value (cell-local-as-data contract): finite fill_value × 0 = 0 for
# deriv, NaN fill_value propagates through IEEE multiplication.
# 4th arg `qe` (query element) promotes result to kernel return type.
# `zero_ref` arg retained for signature stability; intentionally unused.
@inline _fill_extrap_result(::EvalValue, fill_val, _, qe) = _promote_extrap_val(fill_val, qe)
@inline _fill_extrap_result(::AbstractEvalOp, fill_val, _, qe) = _promote_extrap_zero(fill_val, qe)
@inline function _fill_extrap_result(ops::Tuple{Vararg{AbstractEvalOp}}, fill_val, _, qe)
    for i in 1:length(ops)
        @inbounds ops[i] isa EvalValue || return _promote_extrap_zero(fill_val, qe)
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

# ── _resolve_extrap: ND variants (expand + promote [+ per-axis resolve]) ──
#
# Continues the `_resolve_extrap` family from `src/core/periodic.jl` (primitive
# per-axis + 1D bundled + ND bundled-with-data). These ND methods handle the
# scalar→NTuple expansion, periodic-BC override, and FillExtrap value-type
# promotion. Two shapes by arity:
#
# - 4-arg (extrap, bcs, Val(N), Tv): expand + promote. Returns NTuple per axis,
#   forcing `WrapExtrap` where `bcs[d]` is periodic. Used by callers that
#   resolve against the grid separately (post-extension persistent paths
#   where the BC-aware check would trip the pre-extension `<` invariant).
#
# - 5-arg (extrap, bcs, grids, Val(N), Tv): above + per-axis passthrough
#   against `grids`. `bcs::NTuple` → 3-arg primitive (bc-aware);
#   `bcs::Nothing` → 2-arg primitive (no periodic override).

# ── 4-arg: expand + promote (no materialize) ──

@inline function _resolve_extrap(extrap::AbstractExtrap, ::Nothing, ::Val{N}, ::Type{Tv}) where {N, Tv}
    result = ntuple(_ -> extrap, Val(N))
    _validate_fill_values_nd(result)
    return _promote_extraps_nd(result, Tv)
end

@inline function _resolve_extrap(extrap::AbstractExtrap, bcs::Tuple{Vararg{AbstractBC, N}}, ::Val{N}, ::Type{Tv}) where {N, Tv}
    _check_mode_periodic_compat(extrap, bcs, Val(N))
    result = _mode_to_modes_with_periodic(extrap, bcs)
    _validate_fill_values_nd(result)
    return _promote_extraps_nd(result, Tv)
end

@inline function _resolve_extrap(extrap::Tuple{Vararg{AbstractExtrap, N}}, ::Nothing, ::Val{N}, ::Type{Tv}) where {N, Tv}
    _validate_fill_values_nd(extrap)
    return _promote_extraps_nd(extrap, Tv)
end

@inline function _resolve_extrap(extrap::Tuple{Vararg{AbstractExtrap, N}}, bcs::Tuple{Vararg{AbstractBC, N}}, ::Val{N}, ::Type{Tv}) where {N, Tv}
    _check_modes_periodic_compat(extrap, bcs, Val(N))
    result = _modes_to_modes_with_periodic(extrap, bcs)
    _validate_fill_values_nd(result)
    return _promote_extraps_nd(result, Tv)
end

@noinline function _resolve_extrap(extrap::Tuple{Vararg{AbstractExtrap}}, ::Any, ::Val{N}, ::Type) where {N}
    throw(ArgumentError("extrap tuple must have $N elements to match grid dimensions, got $(length(extrap))"))
end

# ── 5-arg: above + per-axis materialize against `grids` ──
#
# `bcs::NTuple{N,AbstractBC}` → per-axis 3-arg primitive (bc-aware, used
# pre-extension for adjoints / hetero OnTheFly where exclusive period matters).
# `bcs::Nothing` → per-axis 2-arg primitive (no periodic concept, used by
# adjoints without BC support).

@inline function _resolve_extrap(
        extrap::AbstractExtrap, ::Nothing,
        grids::NTuple{N, AbstractVector}, ::Val{N}, ::Type{Tv}
    ) where {N, Tv}
    result = ntuple(_ -> extrap, Val(N))
    _validate_fill_values_nd(result)
    promoted = _promote_extraps_nd(result, Tv)
    return map(_resolve_extrap, promoted, grids)
end

@inline function _resolve_extrap(
        extrap::AbstractExtrap, bcs::NTuple{N, AbstractBC},
        grids::NTuple{N, AbstractVector}, ::Val{N}, ::Type{Tv}
    ) where {N, Tv}
    _check_mode_periodic_compat(extrap, bcs, Val(N))
    result = _mode_to_modes_with_periodic(extrap, bcs)
    _validate_fill_values_nd(result)
    promoted = _promote_extraps_nd(result, Tv)
    return map(_resolve_extrap, promoted, bcs, grids)
end

@inline function _resolve_extrap(
        extrap::NTuple{N, AbstractExtrap}, ::Nothing,
        grids::NTuple{N, AbstractVector}, ::Val{N}, ::Type{Tv}
    ) where {N, Tv}
    _validate_fill_values_nd(extrap)
    promoted = _promote_extraps_nd(extrap, Tv)
    return map(_resolve_extrap, promoted, grids)
end

@inline function _resolve_extrap(
        extrap::NTuple{N, AbstractExtrap}, bcs::NTuple{N, AbstractBC},
        grids::NTuple{N, AbstractVector}, ::Val{N}, ::Type{Tv}
    ) where {N, Tv}
    _check_modes_periodic_compat(extrap, bcs, Val(N))
    result = _modes_to_modes_with_periodic(extrap, bcs)
    _validate_fill_values_nd(result)
    promoted = _promote_extraps_nd(result, Tv)
    return map(_resolve_extrap, promoted, bcs, grids)
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
    # Single-pass direction tracking (same algorithm as _is_likely_monotone).
    state = 0
    @inbounds begin
        prev = _query_extract(queries, 1)[d]
        for i in 2:K
            curr = _query_extract(queries, i)[d]
            diff = curr - prev
            s = (diff > 0) - (diff < 0)
            if state == 0
                state = s
            else
                diff * state < 0 && return false
            end
            prev = curr
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
# Single per-axis gateway: promote the query to the grid eltype (`_promote_coord`, as
# 1D does) so handlers see one concrete coordinate type — no OOB/in-domain `Union` (→ `Any`
# for Hermite ND). No-op for matched/non-float grids (Int-grid Int queries stay Int); GridIdx skips it.
@inline _extrap_axis(q::GridIdx, grid, extrap) = q
@inline _extrap_axis(q, grid, extrap) =
    @inbounds _handle_axis_extrap(_promote_coord(q, eltype(grid)), grid, extrap)

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

# Branchless per-axis ClampExtrap: one `min/max` clamp (vs the `_oob_state` classify + 2
# branches) — removes the boundary-OOB misprediction hump. Clamp to the real endpoints
# `first/last` (geometry, matching `_clamp_to_grid`); the widened `domain_lo/hi` are for OOB
# *classification* only (Fill/NoExtrap). `min/max` promote → symmetric for a Dual query/grid.
@inline _handle_axis_extrap(q, axis::AbstractVector, ::ClampExtrap) =
    _clamp(q, first(axis), last(axis))

# FillExtrap keeps the branchy `_oob_state` clamp: Fill needs `_oob_state` anyway for the
# `_try_fill_oob` decision, and the compiler CSEs it with this coordinate classify (so
# `return q` is free). Branchless `min/max` here can't be CSEd → adds ops (measured slower).
@inline function _handle_axis_extrap(q, axis::AbstractVector, ::FillExtrap)
    q_primal = _extract_primal(q)
    st = _oob_state(axis, q_primal)
    st == OOB_LEFT && return _promote_extrap_val(first(axis), q)
    st == OOB_RIGHT && return _promote_extrap_val(last(axis), q)
    return q
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::ExtendExtrap)
    return q
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::WrapExtrap)
    return _wrap_to_domain(q, axis)
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::InBounds)
    return q
end

# ========================================
# Per-Axis Interval Search
# ========================================
#
# Shared interval search logic for all ND interpolation methods.
# Finds the cell containing each query coordinate.

"""
    _search_all_intervals(q_evals, grids, searches) -> (indices, Ls, Rs)

Perform interval search on all axes.
Returns tuples of: indices (cell index), Ls (left bounds), Rs (right bounds).

Accepts heterogeneous tuples (e.g., mixed grid types and search policies).
Uses map over named helpers so each axis receives its concrete type directly,
avoiding ntuple-closure boxing on heterogeneous tuple inputs.
"""
# Named helpers for oneshot map-based search — each receives concrete types per axis.
# search_interval returns `(idx_L, idx_R, L, R)`. Persistent batch paths use
# `_search_axis_adaptive` instead (Bool-flag, no Union boxing).
@inline _getidx(r) = r[1]
@inline _getL(r) = r[3]
@inline _getR(r) = r[4]

# Stencil-valued interval tuple per axis: `stencils[d] = _IdxStencil{2}((idx_L_d, idx_R_d))`.
# Consumers (periodic-aware ND kernels) read corner addresses via
# `stencils[d][bit_d + 1]` — `bit=0 → idx_L`, `bit=1 → idx_R`. For non-periodic
# axes `idx_R == idx_L + 1`; for periodic-exclusive axes at the seam `idx_R == 1`
# (wrap) so eval reads the periodic neighbor without data extension.
# The `_IdxStencil{K}` wrapper (src/core/idx_stencil.jl) unifies this shape across
# method families and carries K as a type parameter for future K > 2 variants
# (ND Hermite bicubic, etc.).
@inline _getstencil(r) = _IdxStencil((r[1], r[2]))

# Shared projector for all `_search_all_intervals*` overloads. Every variant
# boils down to "run `map(search_fn, ...)` then extract `(indices, Ls, Rs)`
# from each result". Only the index extractor differs:
#   - `_getidx`     → `NTuple{N, Int}`             (single corner per axis)
#   - `_getstencil` → `NTuple{N, _IdxStencil{2}}`  (left/right pair per axis)
# Centralizing the `(map(_getL, ...), map(_getR, ...))` tail keeps all variants
# in sync when the 4-tuple `search_interval` return shape evolves.
@inline _project_search_results(results, proj::F) where {F} =
    (map(proj, results), map(_getL, results), map(_getR, results))

# ----------------------------------------
# Hint-aware overloads for persistent search state
# ----------------------------------------

"""
    _ensure_hint_nd(hint, Val(N)) -> NTuple{N, Base.RefValue{Int}}

Create persistent hint tuple for ND evaluation. User-provided hints pass
through; `nothing` creates N fresh `Ref(1)` objects. Must be a named function
(not a lambda) because callers include `@generated` bodies where closures
are forbidden.
"""
@inline _ensure_hint_nd(hint::NTuple{N, Base.RefValue{Int}}, ::Val{N}) where {N} = hint
@inline _ensure_hint_nd(::Nothing, ::Val{N}) where {N} = ntuple(_ref1, Val(N))
@inline _ref1(_) = Ref(1)
@inline _true_flag(_) = true
@inline _false_flag(_) = false

# Scalar mono flag: hint provided → assume locality (LB), no hint → stateless (Binary)
@inline _scalar_mono(::Nothing, ::Val{N}) where {N} = ntuple(_false_flag, Val(N))
@inline _scalar_mono(::NTuple{N, Base.RefValue{Int}}, ::Val{N}) where {N} = ntuple(_true_flag, Val(N))

# ----------------------------------------
# Monotonicity Flags (batch-level, once per call)
# ----------------------------------------
# Pre-compute per-axis monotonicity as NTuple{N, Bool} — concrete, zero-alloc.
# Used by _search_axis_adaptive to select LB vs Binary inside a function barrier,
# avoiding Union boxing from materializing a Tuple{Union{Binary,LB},...} policy tuple.

@inline _check_axis_mono(::AutoSearch, q) = _is_likely_monotone(q)
@inline _check_axis_mono(::AbstractSearchPolicy, _) = true  # explicit policies: flag unused

# AoS per-axis helper: dispatches on policy type (concrete per-element via map).
@inline _check_axis_mono_aos(::AutoSearch, d, queries, vn) =
    _is_axis_likely_monotone(queries, d, vn)
@inline _check_axis_mono_aos(::AbstractSearchPolicy, _d, _queries, _vn) = true

"""
    _check_mono_nd(policies, queries) -> NTuple{N, Bool}

Check per-axis monotonicity for AutoSearch axes.  Returns `NTuple{N, Bool}`,
a concrete tuple type that avoids the `Union{BinarySearch, LinearBinarySearch}`
boxing that per-axis policy resolution would produce.

For explicit (non-AutoSearch) policies, the returned flag is ignored by
`_search_axis_adaptive`; callers should not rely on it having any particular
value.  Short queries (< 8 elements) return all-false as a conservative
fallback.  The mono flags are only effective when hints are present —
without hints, `_search_all_intervals` delegates to the stateless 4-arg
path regardless of mono values.
"""
# SoA queries: per-axis monotonicity check (each query vector checked independently)
@inline _check_mono_nd(
    policies::Tuple{Vararg{AbstractSearchPolicy, N}},
    queries::Tuple{Vararg{AbstractVector, N}}
) where {N} =
    map(_check_axis_mono, policies, queries)

# AoS queries (Vector of point-like elements: Tuple, SVector, etc.):
# per-axis monotonicity via protocol-based _query_extract (no allocation).
# Wraps queries + Val(N) in _AoSMonoChecker to avoid closure capture in map.
struct _AoSMonoChecker{Q, VN}
    queries::Q
    vn::VN
end
@inline (c::_AoSMonoChecker)(p, d) = _check_axis_mono_aos(p, d, c.queries, c.vn)

@inline function _check_mono_nd(
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        queries::AbstractVector
    ) where {N}
    checker = _AoSMonoChecker(queries, Val(N))
    return map(checker, policies, ntuple(identity, Val(N)))
end

# Generic queries (custom protocol containers): per-axis monotonicity via
# _is_axis_likely_monotone (same protocol-based check as AoS).
# Covers any container implementing _query_length/_query_extract.
@inline function _check_mono_nd(
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        queries
    ) where {N}
    _query_length(queries) < 8 && return ntuple(_false_flag, Val(N))
    checker = _AoSMonoChecker(queries, Val(N))
    return map(checker, policies, ntuple(identity, Val(N)))
end

# ----------------------------------------
# Per-axis adaptive search (function barrier)
# ----------------------------------------
# Creates Searcher per-axis inside a function barrier. The Union{LB{8}, Binary+RefHint}
# is resolved by Julia's union-splitting INSIDE the barrier — the concrete
# return type (Int, Tg, Tg) never escapes as Union.
#
# mono=true  → LB{8} with persistent hint (walk + locality)
# mono=false → Binary+RefHint (pure binary search + hint write-back, no walk overhead)

# Range grid: always DirectSearch O(1) regardless of policy/mono.
@inline function _search_axis_adaptive(q, grid::AbstractRange, ::AbstractSearchPolicy, hint, _)
    searcher = _to_searcher(DirectSearch(), hint)
    return @inbounds search_interval(searcher, grid, q)
end

@inline function _search_axis_adaptive(q, grid::AbstractRange, ::AutoSearch, hint, _)
    searcher = _to_searcher(DirectSearch(), hint)
    return @inbounds search_interval(searcher, grid, q)
end

@inline function _search_axis_adaptive(q, grid::AbstractVector, ::AutoSearch, hint, is_mono)
    searcher = is_mono ?
        _to_searcher(LinearBinarySearch(), hint) :
        _to_searcher(BinarySearch(), hint)
    return @inbounds search_interval(searcher, grid, q)
end

@inline function _search_axis_adaptive(q, grid::AbstractVector, policy::AbstractSearchPolicy, hint, _)
    searcher = _to_searcher(policy, hint)
    return @inbounds search_interval(searcher, grid, q)
end

# ── Extrap-aware per-axis search (6-arg) ──────────────────────────────────────
# The 5-arg form above plus a trailing per-axis `extrap` — trailing because it is the
# new modifier and matches the other per-axis extrap helpers (`_extrap_axis(q, grid,
# extrap)`, `_handle_axis_extrap`, `_check_domain`, which all put `extrap` last), so the
# first five arguments keep the 5-arg positions verbatim. An `InBounds` axis on a
# normalized range takes the lean `_search_direct_inbounds` (one-sided clamp — the lower
# `max(·,1)` is dead when the query is in-domain); the interval is bit-identical to the
# generic path. The persistent hint is still written back (symmetry with the standard
# `_search_direct!` / `_search_grididx!` path): an explicitly-provided hint must report
# the found interval regardless of extrap. PER-AXIS — a mixed `(InBounds, ClampExtrap)`
# query leans only the InBounds axis. Every other `(extrap, grid)` pair delegates to the
# 5-arg form unchanged. Any ND `_locate_cell` that threads its `extraps` into the search
# picks this up; 5-arg callers are unaffected. `hint` is always a concrete `Ref{Int}` here
# (the 6-arg `_search_all_intervals` is reached only from persistent `_locate_cell`).
@inline function _search_axis_adaptive(q, grid::_CachedRange, ::AbstractSearchPolicy, hint::Base.RefValue{Int}, _mono, e::InBounds)
    idx, xL, xR = _search_direct_inbounds(grid, q, e)
    hint[] = idx
    return idx, idx + 1, xL, xR
end
@inline function _search_axis_adaptive(q::GridIdx, grid::_CachedRange, ::AbstractSearchPolicy, hint::Base.RefValue{Int}, _mono, ::InBounds)
    @boundscheck (grid.len >= 2 && 1 <= q.idx <= grid.len) || throw(BoundsError(grid, q.idx))
    idx = min(q.idx, grid.len - 1)
    hint[] = idx
    return idx, idx + 1, (@inbounds grid[idx]), (@inbounds grid[idx + 1])
end
@inline _search_axis_adaptive(q, grid::AbstractVector, policy::AbstractSearchPolicy, hint, mono, ::AbstractExtrap) =
    _search_axis_adaptive(q, grid, policy, hint, mono)
# InBounds vector grid: reuse the guarded path's Searcher choice (mono → LinearBinarySearch walk,
# else BinarySearch), but route through the 4-arg lean `search_interval` so a BinarySearch axis drops
# the `first`/`last` guards; LinearBinarySearch/Linear fall through (no lean). GridIdx is handled by
# the 4-arg dispatch (index short-circuit). Range keeps the more-specific `_CachedRange` method above;
# `::AbstractSearchPolicy` (not `::AutoSearch`) keeps this from tying with it on dispatch.
@inline _axis_vec_searcher(::AutoSearch, hint, is_mono) =
    is_mono ? _to_searcher(LinearBinarySearch(), hint) : _to_searcher(BinarySearch(), hint)
@inline _axis_vec_searcher(policy::AbstractSearchPolicy, hint, _is_mono) = _to_searcher(policy, hint)
@inline _search_axis_adaptive(q, grid::AbstractVector, policy::AbstractSearchPolicy, hint, is_mono, ::InBounds) =
    @inbounds search_interval(_axis_vec_searcher(policy, hint, is_mono), grid, q, InBounds())

# 5-arg `_search_all_intervals` (mono variant, no spacings).
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {N}
    results = map(_search_axis_adaptive, q_evals, grids, policies, hints, mono)
    return _project_search_results(results, _getidx)
end

# 6-arg extrap-aware variant: the 5-arg form plus a trailing `extraps` (positions of the
# first five match the 5-arg form), threaded per-axis into `_search_axis_adaptive` so
# InBounds range axes take the lean direct search. Callers with `extraps` in hand (every
# persistent `_locate_cell`) route here; the 5-arg form stays for callers that do not.
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
    ) where {N}
    results = map(_search_axis_adaptive, q_evals, grids, policies, hints, mono, extraps)
    return _project_search_results(results, _getidx)
end

# 5-arg variant with Nothing hint (used by oneshot scalar paths that have no
# persistent hint storage — same as the 4-arg with-hints form, plus mono).
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        policies::Tuple{Vararg{AbstractSearchPolicy, N}},
        ::Nothing,
        ::NTuple{N, Bool},
    ) where {N}
    return _search_all_intervals(q_evals, grids, policies)
end

# Per-axis oneshot variants (no spacing). Used by `hetero_nointerp.jl`
# real-axis-reduced search where spacings have been removed alongside the
# `spacings::S` field on HeteroInterpolantND.
@inline _search_axis_oneshot(q, grid, search) =
    @inbounds search_interval(_resolve_search(grid, q, search, nothing), grid, q)
@inline _search_axis_oneshot_hint(q, grid, search, hint) =
    @inbounds search_interval(_resolve_search(grid, q, search, hint), grid, q)

# 3-arg `_search_all_intervals` (no spacings, no hints, no mono).
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
    ) where {N}
    results = map(_search_axis_oneshot, q_evals, grids, searches)
    return _project_search_results(results, _getidx)
end

# 4-arg `_search_all_intervals` (no spacings, Nothing hint).
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        ::Nothing,
    ) where {N}
    return _search_all_intervals(q_evals, grids, searches)
end

# 4-arg `_search_all_intervals` (no spacings, Tuple hint).
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
    ) where {N}
    results = map(_search_axis_oneshot_hint, q_evals, grids, searches, hints)
    return _project_search_results(results, _getidx)
end

# Extrap-aware per-axis oneshot search (indices path, used by cubic/quad/hermite oneshot).
# An InBounds axis on a normalized range takes the lean `_search_direct_inbounds`
# (one-sided clamp), bit-identical to the standard search for an in-bounds query, and
# still writes the hint (symmetry). EVERY other `(q, grid, extrap)` — GridIdx queries,
# vector grids, and periodic axes (always WrapExtrap, never InBounds; the `_ExclusivePeriodicAxis`
# seam wrap in `search_interval` is untouched) — delegates to the 4-arg form verbatim.
@inline function _search_axis_oneshot_hint(q::Real, grid::_CachedRange, search, hint::Base.RefValue{Int}, e::InBounds)
    idx, xL, xR = _search_direct_inbounds(grid, q, e)
    hint[] = idx
    return idx, idx + 1, xL, xR
end
# GridIdx → 4-arg short-circuit (exact .idx), not the coordinate lean (which can pick an off-by-one
# cell on non-unit-step ranges). Mirrors the persistent `_search_axis_adaptive(q::GridIdx, …)`.
@inline _search_axis_oneshot_hint(q::GridIdx, grid::_CachedRange, search, hint::Base.RefValue{Int}, ::InBounds) =
    _search_axis_oneshot_hint(q, grid, search, hint)
@inline _search_axis_oneshot_hint(q, grid, search, hint, ::AbstractExtrap) =
    _search_axis_oneshot_hint(q, grid, search, hint)
# InBounds vector grid: thread InBounds into the resolved-searcher search so a BinarySearch axis
# leans; GridIdx routed by the 4-arg dispatch. Range uses the `_CachedRange` method above.
@inline _search_axis_oneshot_hint(q, grid::AbstractVector, search, hint::Base.RefValue{Int}, ::InBounds) =
    @inbounds search_interval(_resolve_search(grid, q, search, hint), grid, q, InBounds())

# 5-arg extrap-aware `_search_all_intervals` (oneshot indices path): the 4-arg hint form
# plus a trailing per-axis `extraps`, threaded into `_search_axis_oneshot_hint` so InBounds
# range axes take the lean direct search. The trailing `AbstractExtrap`-tuple is type-distinct
# from the 5-arg `mono::NTuple{N,Bool}` form, so they never collide for N >= 1.
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
    ) where {N}
    results = map(_search_axis_oneshot_hint, q_evals, grids, searches, hints, extraps)
    return _project_search_results(results, _getidx)
end

# Nothing-hint 5-arg extrap-aware (scalar oneshot): throwaway Refs (stack-elided), then
# the with-hint extrap-aware search. The InBounds hint write lands on the throwaway Ref
# and is DCE'd, so the lean fast path is still 0-alloc.
@inline function _search_all_intervals(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        ::Nothing,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
    ) where {N}
    return _search_all_intervals(q_evals, grids, searches, _ensure_hint_nd(nothing, Val(N)), extraps)
end

# Aqua-required N=0 disambiguators. The grid-only and spacings-based 4-arg
# / 5-arg variants of `_search_all_intervals` bind to `Tuple{}` ambiguously
# when N=0 (slot 3+ is `NTuple{0, X}` regardless of `X`). ND interpolants
# require N >= 1 at runtime — these methods exist solely to make
# `Aqua.test_ambiguities` happy.
@inline _search_all_intervals(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) =
    ((), (), ())
@inline _search_all_intervals(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}) =
    ((), (), ())
# Nothing-hint 5-arg N=0: the `(…, ::Nothing, mono::NTuple{N,Bool})` and
# `(…, ::Nothing, extraps::NTuple{N,AbstractExtrap})` forms both collapse to a `Tuple{}`
# 5th slot at N=0 — this `::Nothing`-4th disambiguator resolves the pair.
@inline _search_all_intervals(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Nothing, ::Tuple{}) =
    ((), (), ())

# ────────────────────────────────────────────────────────
# BC-aware per-axis search (Phase 6 — zero-copy periodic ND)
# ────────────────────────────────────────────────────────
# Parallel in purpose to `_search_all_intervals`, but threads per-axis `bcs`
# into each `Searcher` so `PeriodicBC{:exclusive}` axes return
# `(n, 1, x[n], x[1]+L)` at seam cells via the BC-aware `search_interval`
# dispatch. Returns `(stencils, Ls, Rs)` where
# `stencils[d] = _IdxStencil{2}((idx_L_d, idx_R_d))` — non-periodic axes have
# `idx_R == idx_L + 1`; periodic-exclusive axes at seam have `idx_R == 1` (wrap).
#
# Structurally mirrors persistent's `_search_all_intervals`: one `map` that
# resolves and searches per axis in a single body (persistent's
# `_search_axis_adaptive` pattern). Hint preparation reuses `_ensure_hint_nd` —
# the same helper persistent uses, so scalar vs batch semantics are identical:
#   - batch with AutoSearch → monotone queries hit LinearBinarySearch walk
#     (Ref tracks walk position across queries → intra-batch locality)
#   - scalar / non-monotone → BinarySearch, Refs are unused and stack-elided
# by Julia escape analysis → 0 heap bytes/call.

# Per-axis inline: build Searcher + run search_interval in one body.
# Callers pre-wrap grids via `_resolve_axis` (or pass already-wrapped axes),
# so seam handling is via axis-level dispatch in `periodic_axis.jl`.
@inline _search_axis_stencil(grid, q, search, hint) =
    @inbounds search_interval(_resolve_search(grid, q, search, hint), grid, q)

@inline function _search_all_intervals_stencil(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
    ) where {N}
    results = map(_search_axis_stencil, grids, q_evals, searches, hints)
    return _project_search_results(results, _getstencil)
end

# Nothing-hint overload — scalar oneshot entries only. Batch must use the
# 4-arg NTuple form (hint allocation hoisted via `_resolve_oneshot_search_nd`).
@inline function _search_all_intervals_stencil(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        ::Nothing,
    ) where {N}
    hints = _ensure_hint_nd(nothing, Val(N))
    return _search_all_intervals_stencil(q_evals, grids, searches, hints)
end

# Extrap-aware per-axis stencil search (stencil path, used by linear/constant oneshot).
# An InBounds axis on a normalized range takes the lean `_search_direct_inbounds`
# (one-sided clamp) and emits the non-seam stencil `(idx, idx+1, …)` — bit-identical to
# the standard stencil for an in-bounds query. EVERY other `(grid, q, extrap)` delegates
# to the 4-arg `_search_axis_stencil`: crucially `_ExclusivePeriodicAxis` (always
# WrapExtrap, never InBounds, and `<: AbstractVector` not `_CachedRange`) keeps its
# `search_interval` seam wrap `(n, 1, …)` untouched, as do GridIdx and vector grids.
@inline function _search_axis_stencil(grid::_CachedRange, q::Real, search, hint::Base.RefValue{Int}, e::InBounds)
    idx, xL, xR = _search_direct_inbounds(grid, q, e)
    hint[] = idx
    return idx, idx + 1, xL, xR
end
# GridIdx → short-circuit (exact `.idx`), not the coordinate lean — see the same overload on
# `_search_axis_oneshot_hint` above for the off-by-one rationale.
@inline _search_axis_stencil(grid::_CachedRange, q::GridIdx, search, hint::Base.RefValue{Int}, ::InBounds) =
    _search_axis_stencil(grid, q, search, hint)
@inline _search_axis_stencil(grid, q, search, hint, ::AbstractExtrap) =
    _search_axis_stencil(grid, q, search, hint)
# InBounds vector grid: thread InBounds into the resolved-searcher search (BinarySearch axis leans).
# Range uses the `_CachedRange` method above; periodic axes are `<: AbstractVector` but never arrive
# InBounds (always WrapExtrap), so they stay on the `::AbstractExtrap` seam path.
@inline _search_axis_stencil(grid::AbstractVector, q, search, hint::Base.RefValue{Int}, ::InBounds) =
    @inbounds search_interval(_resolve_search(grid, q, search, hint), grid, q, InBounds())

# 5-arg extrap-aware `_search_all_intervals_stencil`: the 4-arg hint form plus a trailing
# per-axis `extraps`, threaded into `_search_axis_stencil` so InBounds range axes take the
# lean direct search. Non-InBounds / non-range / periodic axes are unaffected.
@inline function _search_all_intervals_stencil(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
    ) where {N}
    results = map(_search_axis_stencil, grids, q_evals, searches, hints, extraps)
    return _project_search_results(results, _getstencil)
end

# Nothing-hint 5-arg extrap-aware stencil (scalar oneshot): throwaway Refs (stack-elided).
@inline function _search_all_intervals_stencil(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        searches::Tuple{Vararg{AbstractSearchPolicy, N}},
        ::Nothing,
        extraps::Tuple{Vararg{AbstractExtrap, N}},
    ) where {N}
    return _search_all_intervals_stencil(q_evals, grids, searches, _ensure_hint_nd(nothing, Val(N)), extraps)
end


"""
    _resolve_oneshot_search_nd(search, queries, hint, Val(N)) -> (policies, hints)

Per-axis adaptive policy + persistent hint tuple. Call once outside the
per-query loop in every ND oneshot batch path. `hint::Nothing` produces a
fresh `Ref{Int}` per axis; user-supplied Refs pass through unchanged.
"""
@inline function _resolve_oneshot_search_nd(
        search, queries, hint, ::Val{N}
    ) where {N}
    policies = _resolve_search_nd(search, Val(N), queries, hint)
    hints = _ensure_hint_nd(hint, Val(N))
    return policies, hints
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
    _compute_all_local_params(q_evals, grids, indices, Ls) -> (hs, inv_hs, dLs)

Compute local cell parameters for all axes via `_get_h(grid, idx)` /
`_get_inv_h(grid, idx)` on wrapped axes (`_CachedRange`, `_CachedVector`,
`_ExclusivePeriodicAxis`) — or on-the-fly `float(x[i+1] - x[i])` for raw
`Vector`. Returns tuples of: `hs` (cell widths), `inv_hs` (reciprocals),
`dLs` (left deltas).
"""
@inline function _compute_all_local_params(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        indices::NTuple{N, Int},
        Ls::Tuple{Vararg{Real, N}},
    ) where {N}
    # Promote `hs`/`inv_hs` to one common float `Tg`: `_eval_nd_*_cell` is `@generated`
    # and couples them as `NTuple{N, Tg}`, so heterogeneous/mixed-precision raw grids
    # need them unified. Bit-identical for the homogeneous Float/Dual callers.
    # (N=0 edge: `float(promote_type())` = `float(Union{})` throws; the ntuples are
    # empty there, so this placeholder is never used.)
    Tg = N == 0 ? Float64 : float(_promote_grid_eltype(grids))
    return _compute_all_local_params(q_evals, grids, indices, Ls, Tg)
end

# Data-aware form: the caller supplies the width type `Tg` (value-matched, e.g.
# `_promote_grid_float(grid eltype, Tv)` — Int grid + Float32 data → Float32), so raw
# Int axes don't widen the eval to Float64 via `inv(Int)`. `dLs` deliberately keep
# their natural `q - L` promotion — converting them would strip Dual-query partials.
@inline function _compute_all_local_params(
        q_evals::Tuple{Vararg{Real, N}},
        grids::Tuple{Vararg{AbstractVector, N}},
        indices::NTuple{N, Int},
        Ls::Tuple{Vararg{Real, N}},
        ::Type{Tg},
    ) where {N, Tg}
    hs = ntuple(Val(N)) do d
        @inbounds convert(Tg, _get_h(grids[d], indices[d]))
    end
    inv_hs = ntuple(Val(N)) do d
        @inbounds convert(Tg, _get_inv_h(grids[d], indices[d]))
    end
    dLs = ntuple(Val(N)) do d
        @inbounds q_evals[d] - Ls[d]
    end
    return hs, inv_hs, dLs
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

# ── Static-Tg tuple maps for the one-shot hot paths (@generated) ─────────────
# `map(f, grids, ntuple(_ -> Tg, Val(N)))` re-captures the Type witness in the
# ntuple closure: under a degraded-inference context (a long-lived test worker)
# the tuple elements decay to `DataType` and every per-axis call goes through
# dynamic dispatch — 32 B/axis on Julia 1.12 CI, 60-90 KB downstream on LTS.
# These unroll at codegen with `Tg` as a STATIC signature parameter: no closure
# and no runtime `Type` value exist on any Julia version.

# Pooled value-matched wrap (cubic/quadratic PreCompute scalar backends).
@generated function _cache_axes_pooled(pool, grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    exprs = [:(_cache_axis_pooled(pool, grids[$i], Tg)) for i in 1:N]
    return :(($(exprs...),))
end

# BC-aware one-shot resolve (linear/constant/hetero OnTheFly surfaces).
@generated function _resolve_axes(grids::NTuple{N, AbstractVector}, bcs, ::Type{Tg}) where {N, Tg}
    exprs = [:(_resolve_axis(grids[$i], bcs[$i], Tg)) for i in 1:N]
    return :(($(exprs...),))
end

# Raw-form bridge (hetero global-solve path): matching axes stay RAW (1D cache
# identity); only float-mismatched axes convert. The branch is decided in the
# GENERATOR — per-axis, from types alone.
@generated function _bridge_axes_raw(grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    exprs = map(1:N) do i
        float(eltype(fieldtype(grids, i))) === Tg ? :(grids[$i]) : :(_convert_grid(grids[$i], Tg))
    end
    return :(($(exprs...),))
end

# Width-typed reciprocal spans from search results (linear ND scalar one-shot).
# Keeps today's convert-after form bit-for-bit (span-first harmonization is a
# separate value-visible change).
@generated function _convert_inv_hs(grids::NTuple{N, AbstractVector}, idxs, Ls, Rs, ::Type{Tg}) where {N, Tg}
    exprs = [:(convert(Tg, _get_inv_h(grids[$i], idxs[$i], Ls[$i], Rs[$i]))) for i in 1:N]
    return :(($(exprs...),))
end

"""
    _nd_promote_grids(grids, data) -> (grids_typed, Tg, Tv, Tz)

Unified ND type promotion: compute grid type, convert grids, determine value
and coefficient types in a single call.

Returns:
- `grids_typed`: grids converted to unified float type
- `Tg`: promoted grid element type (always a float type; preserves Dual)
- `Tv`: value type (data eltype promoted to grid precision for standard numerics)
- `Tz`: coefficient/output type (`data × grid` — equals `Tv` for Float grids, `Dual` for Dual grids)

Callers destructure only what they need:
```julia
grids_typed, _, _, _ = _nd_promote_grids(grids, data)   # grid-only (batch dispatch)
grids_typed, Tg, Tv, Tz = _nd_promote_grids(grids, data) # full (oneshot/build)
```
"""
@inline _nd_promote_grids(
    grids::NTuple{N, AbstractVector},
    data::AbstractArray{Tv_raw, N}
) where {Tv_raw, N} = _nd_promote_grids(grids, data, Tv_raw)

# 3-arg form: `Tv_extra` widens the value space beyond `eltype(data)` BEFORE the grid
# value-match — Hermite's value space is data ∪ partials (Float32 data + Float64
# partials must give a Float64 grid, matching the one-shot rule). 2-arg delegates
# with `Tv_extra = eltype(data)` (neutral).
@inline function _nd_promote_grids(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv_raw, N},
        ::Type{Tv_extra}
    ) where {Tv_raw, Tv_extra, N}
    Tv_all = promote_type(Tv_raw, Tv_extra)
    # Value-matched grid float (1D rule): Int/OneTo grid + Float32 data → Float32 grid, so the
    # cheap grid converts and the O(nᴺ) data aliases under copy=false. The old grid-eltype-only
    # `float(...)` gave Float64 and dragged Tv (and the data, via `Tv.(data)`) up with it.
    Tg = _promote_grid_float(_promote_grid_eltype(grids), Tv_all)
    grids_typed = _convert_grids_typed(grids, Tg)
    Tv = _value_type(Tv_all, Tg)
    Tz = _promote_eltype(Tv, Tg)
    return grids_typed, Tg, Tv, Tz
end

"""
    _nd_promote_grids_raw(grids, data) -> (grids_typed, Tg, Tv)

Raw-eltype variant of `_nd_promote_grids`: skips the `float()` widening, keeps
`Tv = eltype(data)`. Used by selection-kernel methods (Constant) where there
is no x·y arithmetic and the output contract follows `eltype(data)` directly.

- `Tg = promote_type(eltype.(grids)...)` (via `@generated` `_promote_grid_eltype`,
  unrolled at compile time — zero alloc).
- `Tv = eltype(data)` — no promotion.
- `grids_typed`: each axis converted to share `Tg` (container heterogeneity
  preserved — Range stays Range, Vector stays Vector).

Arithmetic batch/constructor paths keep `_nd_promote_grids` (Float-widened Tg, value-promoted Tv).
"""
@inline function _nd_promote_grids_raw(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N}
    ) where {Tv, N}
    Tg = _promote_grid_eltype(grids)
    grids_typed = _convert_grids_typed(grids, Tg)
    return grids_typed, Tg, Tv
end
