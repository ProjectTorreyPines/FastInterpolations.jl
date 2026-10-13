# ========================================
# PHSInterpolantND — Constructor & Callables
# ========================================

# ======================================================
# Helper: compute blend_a and blend_r_idx
# ======================================================

function _phs_blend_params(grids, blend_factor::Real)
    N = length(grids)
    Tg = eltype(first(grids))
    # Mean grid spacing per axis
    h_max_per_axis = ntuple(N) do d
        g = grids[d]
        n = length(g)
        Tg((last(g) - first(g)) / (n - 1))
    end
    h_max = maximum(h_max_per_axis)
    blend_a = Tg(blend_factor) * h_max

    # Half-width in index space per axis (ceiling so we cover blend_a)
    blend_r_idx = ntuple(N) do d
        h_d = h_max_per_axis[d]
        h_d > zero(Tg) ? max(1, ceil(Int, blend_a / h_d)) : 1
    end

    return blend_a, blend_r_idx
end

# ======================================================
# Constructor
# ======================================================

"""
    phs_interp(grids, data; kwargs...) -> PHSInterpolantND

Create an N-dimensional polyharmonic spline interpolant.

# Arguments
- `grids`: `NTuple{N, AbstractVector}` — one grid vector per dimension. Each axis must be
    uniformly spaced: the stencil geometry uses a single spacing per axis, so non-uniform
    axes are accepted but give inaccurate results.
- `data`:  `AbstractArray{Tv, N}` — data values at grid nodes

# Keyword Arguments
- `stencil_size::Int = 8`:
    Number of stencil nodes per axis (total = stencil_size^N).
    Reduce for high dimensions (e.g. 4 for N≥4).
- `degree::Int = 3`:
    PHS radial function degree (odd positive integer: 1, 3, 5, …).
    Higher degree → smoother interpolant, larger condition number.
- `blend_factor::Real = 1.0`:
    Blend range = blend_factor × max_grid_spacing.
    Larger values → wider blending neighbourhood → smoother but more expensive.
    Default 1.0 provides good balance (3× faster than 2.0, ~2× error increase).
- `extrap=NoExtrap()`:
    Extrapolation mode (scalar or per-axis tuple).
- `search=AutoSearch()`:
    Search policy (scalar or per-axis tuple; used for OOB checking).
- `log_reference=nothing`:
    Enables the log transform: `data` is stored as `log(ρ/ρ₀)` and evaluation returns
    `ρ₀ · exp(f)` with matching derivatives. `ρ₀` is a nonzero constant, used as given
    and with the sign of the data, or a callable `ρ₀(q)` on an `NTuple{N}` point.
    Value queries only call `ρ₀(q)`; derivative queries also call `ρ₀(q; deriv = ops)`,
    as for a FastInterpolations interpolant, so a callable reference answers them
    through the `deriv` keyword. Every `data/ρ₀` must be positive and finite. Arrays are not accepted: a reference sampled on the data grid adds no
    information.

# Returns
`PHSInterpolantND{Tg, Tv, N, degree}` — callable interpolant.

# Examples
```julia
x = range(0.0, 1.0, 20)
y = range(0.0, 1.0, 20)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

itp = phs_interp((x, y), data)
itp((0.5, 0.3))                              # scalar query
itp((0.5, 0.3); deriv=DerivOp(1, 0))        # ∂f/∂x
itp(([0.1, 0.5, 0.9], [0.2, 0.4, 0.6]))     # batch SoA
```
"""
function phs_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv_raw, N};
        stencil_size::Int = 8,
        degree::Int = 3,
        blend_factor::Real = 1.0,
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        log_reference = nothing,
    ) where {N, Tv_raw}
    isodd(degree) && degree >= 1 || throw(ArgumentError("PHS degree must be odd and ≥ 1, got $degree"))
    stencil_size >= 1 || throw(ArgumentError("stencil_size must be ≥ 1, got $stencil_size"))

    _validate_nd_grids(grids, data)
    grids_typed, Tg, Tv, _ = _nd_promote_grids(grids, data)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    grids_c = _convert_cache_axes(grids_typed, ntuple(_ -> NoBC(), Val(N)), Tg)
    searches = _resolve_search_nd(search, Val(N))
    extrap_vals = _resolve_extrap(extrap, ntuple(_ -> NoBC(), N), Val(N), Tv)

    blend_a, blend_r_idx = _phs_blend_params(grids_c, blend_factor)

    # Build single canonical stencil + boundary shift cache
    stencil_offsets, phi_inv, hs, stencil_lo, stencil_hi, shift_cache =
        _phs_build_stencil(grids_c, stencil_size, degree)

    stencil_phys_offsets = [ntuple(d -> Tg(off[d]) * hs[d], Val(N)) for off in stencil_offsets]

    # Optional log transform: store log(ρ/ρ₀) and keep the reference for evaluation
    transform, data_store = _resolve_phs_log_reference(log_reference, grids_c, data_typed, Tv, Tg)

    blend_a3 = blend_a^3
    # Use maxthreadid() to account for interactive thread pools
    coeff_caches = Dict{NTuple{N, Int}, Vector{Tg}}[Dict{NTuple{N, Int}, Vector{Tg}}() for _ in 1:Threads.maxthreadid()]
    return PHSInterpolantND{
        Tg, Tv, N, degree,
        typeof(grids_c), typeof(transform), typeof(extrap_vals), typeof(searches),
    }(
        grids_c, data_store,
        stencil_offsets, stencil_phys_offsets, phi_inv, stencil_lo, stencil_hi, shift_cache, hs,
        blend_a, blend_a3, blend_r_idx,
        transform, extrap_vals, searches, coeff_caches
    )
end

# ======================================================
# Callable interface
# ======================================================

# ---- Helpers ----

@inline function _phs_check_domain(itp::PHSInterpolantND{Tg, Tv, N}, query::NTuple{N, <:Real}) where {Tg, Tv, N}
    return _validate_nd_domain(itp.grids, query, itp.extraps)
end

@inline function _phs_resolve_ops(
        deriv::Union{DerivOp, NTuple{N, DerivOp}},
        ::Val{N},
    ) where {N}
    return _resolve_deriv_nd(deriv, Val(N))
end

# ---- Scalar query (NTuple) ----

"""
    (itp::PHSInterpolantND)(query::NTuple{N,Real}; deriv=EvalValue()) -> scalar

Evaluate the PHS interpolant at a single N-tuple query point.
"""
# Shared implementation — always receives concrete `ops` tuple, zero-alloc.
@inline function _phs_callable_impl(
        itp::PHSInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N}
    _phs_check_domain(itp, query)
    # Handle out-of-bounds (fills FillExtrap, etc.)
    oob = _try_fill_oob(query, itp.grids, itp.extraps, ops, first(itp.data))
    oob !== nothing && return oob
    return _phs_eval(itp, query, ops)
end

# Single callable — Union{DerivOp, Tuple} is handled by Julia's union-splitting
# at the _phs_resolve_ops call site inside _phs_callable_impl.
@inline function (itp::PHSInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kw...,  # absorb search/hint passed by AbstractInterpolantND protocol
    ) where {Tg, Tv, N}
    return _phs_callable_impl(itp, query, _phs_resolve_ops(deriv, Val(N)))
end

"""
    (itp::PHSInterpolantND)(out::AbstractVector, queries; deriv=EvalValue())

In-place batch evaluation. `queries` can be:
  - `Tuple{Vararg{AbstractVector,N}}` (SoA)
  - `AbstractVector{<:NTuple{N}}` or `AbstractVector{<:AbstractVector}` (AoS)

Evaluates serially for maximum compute, memory, and allocation efficiency.
Per-thread caches (indexed by Threads.threadid()) and pool buffers make it safe to call from externally-threaded loops.
"""
# Shared batch implementation — receives concrete ops tuple.
function _phs_batch_impl!(
        itp::PHSInterpolantND{Tg, Tv, N},
        out::AbstractVector,
        queries,
        ops::NTuple{N, AbstractEvalOp},
    ) where {Tg, Tv, N}
    nq = _query_length(queries)
    length(out) == nq || _throw_query_output_mismatch(nq, length(out))
    _query_validate(queries)

    @inbounds for k in 1:nq
        q = _extract_query_point(queries, k, Val(N))
        oob = _try_fill_oob(q, itp.grids, itp.extraps, ops, first(itp.data))
        if oob !== nothing
            out[k] = oob
        else
            out[k] = _phs_eval(itp, q, ops)
        end
    end
    return out
end

function (itp::PHSInterpolantND{Tg, Tv, N})(
        out::AbstractVector,
        queries::Union{Tuple{Vararg{AbstractVector, N}}, AbstractVector};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kw...,  # absorb search/hint forwarded by AbstractInterpolantND protocol
    ) where {Tg, Tv, N}
    return _phs_batch_impl!(itp, out, queries, _phs_resolve_ops(deriv, Val(N)))
end

# Allocating batch evaluation is handled by AbstractInterpolantND protocol,
# which forwards to our in-place callable above via dynamic dispatch.

# N = 1 GriddedQuery forwards. A 1-axis `GriddedQuery` is an `AbstractVector`, so
# the generic in-place gridded functor (`AbstractInterpolantND{,,1}` × `GriddedQuery`)
# splits with the batch callable above (`PHSInterpolantND` × `AbstractVector`): each
# wins one argument. PHS has no separable gridded kernel, so mirror the 1-D
# contract — a 1-axis GriddedQuery is its coordinate vector — and forward both
# forms to the SoA batch path.
@inline (itp::PHSInterpolantND{Tg, Tv, 1})(
    out::AbstractVector,
    gq::GriddedQuery{<:Tuple{Any}};
    kwargs...,
) where {Tg, Tv} = itp(out, (gq.axes[1],); kwargs...)
@inline (itp::PHSInterpolantND{Tg, Tv, 1})(gq::GriddedQuery{<:Tuple{Any}}; kwargs...) where {Tg, Tv} =
    itp((gq.axes[1],); kwargs...)
