# ========================================
# PHSInterpolantND — Constructor & Callables
# ========================================

# ======================================================
# Helper: compute blend_a and blend_r_idx
# ======================================================

function _phs_blend_params(grids, spacings, blend_factor::Real)
    N = length(grids)
    Tg = eltype(first(grids))
    # Maximum h per axis
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
- `grids`: `NTuple{N, AbstractVector}` — one grid vector per dimension
- `data`:  `AbstractArray{Tv, N}` — data values at grid nodes

# Keyword Arguments
- `stencil_size::Int = 8`:
    Number of stencil nodes per axis (total = stencil_size^N).
    Reduce for high dimensions (e.g. 4 for N≥4).
- `degree::Int = 3`:
    PHS radial function degree (odd positive integer: 1, 3, 5, …).
    Higher degree → smoother interpolant, larger condition number.
- `blend_factor::Real = 2.0`:
    Blend range = blend_factor × max_grid_spacing.
    Larger values → wider blending neighbourhood → smoother but more expensive.
- `extrap=NoExtrap()`:
    Extrapolation mode (scalar or per-axis tuple).
- `search=AutoSearch()`:
    Search policy (scalar or per-axis tuple; used for OOB checking).
- `reference_interp=nothing`:
    If provided, enables the log-density smoothing transform:
    `data` is stored as `log(ρ/ρ₀)` where ρ₀ values come from `reference_data`
    (if given) or from evaluating `reference_interp` at each grid node.
    Evaluation returns ρ̃ = ρ₀ * exp(f) with correct derivative transforms.
- `reference_data=nothing`:
    Pre-computed ρ₀ values at all grid nodes (same shape as `data`).
    When provided alongside `reference_interp`, avoids evaluating `reference_interp`
    at every grid node during construction — useful when `reference_interp` is a
    nested PHS (expensive per-node) but the raw ρ₀ array is already available.
    Example:
    ```julia
    # itp_rho0 is a log-space PHS — accurate derivatives but slow to query 592K nodes
    itp_rho0 = phs_interp(grids, rho0; reference_interp = ConstantRef(1.0))
    # Pass rho0 directly so construction stays O(N·log N); eval uses itp_rho0 for ∂ρ₀
    itp_phs  = phs_interp(grids, rho; reference_interp = itp_rho0,
                          reference_data = rho0)
    ```

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
        blend_factor::Real = 2.0,
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
        reference_interp = nothing,
        reference_data = nothing,
    ) where {N, Tv_raw}
    isodd(degree) && degree >= 1 || throw(ArgumentError("PHS degree must be odd and ≥ 1, got $degree"))
    stencil_size >= 1 || throw(ArgumentError("stencil_size must be ≥ 1, got $stencil_size"))

    _validate_nd_grids(grids, data)
    grids_typed, Tg, Tv, _ = _nd_promote_grids(grids, data)
    data_typed = Tv === Tv_raw ? data : Tv.(data)

    spacings = _create_spacings_typed(grids_typed)
    searches = _resolve_search_nd(search, Val(N))
    extrap_vals = _resolve_extrap(extrap, ntuple(_ -> NoBC(), N), Val(N), Tv)

    blend_a, blend_r_idx = _phs_blend_params(grids_typed, spacings, blend_factor)

    # Build single canonical stencil + boundary shift cache
    stencil_offsets, phi_inv, hs, stencil_lo, stencil_hi, shift_cache =
        _phs_build_stencil(grids_typed, spacings, stencil_size, degree)

    # Optionally apply log-density smoothing transform
    transform, data_store = if reference_interp === nothing
        nothing, Array{Tv}(data_typed)
    else
        # Determine ρ₀ at each grid node.
        # `reference_data` lets the caller bypass per-node evaluation of `reference_interp`
        # (useful when reference_interp is a nested PHS — expensive at 592K+ nodes).
        rho0_nodes = if reference_data !== nothing
            # Pre-computed ρ₀ array supplied — use directly (fast path)
            size(reference_data) == size(data_typed) ||
                throw(DimensionMismatch("reference_data size $(size(reference_data)) must match data size $(size(data_typed))"))
            Tv.(reference_data)
        else
            # Evaluate reference_interp at each grid node in parallel.
            # Scalar queries to CubicInterpolant / PromolecularRef are read-only after
            # construction, so this is safe for Threads.@threads.
            _len  = length(data_typed)
            _size = size(data_typed)
            _rho0 = Vector{Tv}(undef, _len)
            Threads.@threads :static for k in 1:_len
                idx = CartesianIndices(_size)[k]
                @inbounds _rho0[k] = reference_interp(ntuple(d -> grids_typed[d][idx[d]], N))
            end
            reshape(_rho0, _size)
        end
        log_data = Array{Tv}(log.(data_typed ./ rho0_nodes))
        PHSLogTransform{N, typeof(reference_interp)}(reference_interp), log_data
    end

    blend_a3 = blend_a^3
    return PHSInterpolantND{
        Tg, Tv, N, degree,
        typeof(grids_typed), typeof(spacings), typeof(transform), typeof(extrap_vals), typeof(searches),
    }(
        grids_typed, spacings, data_store,
        stencil_offsets, phi_inv, stencil_lo, stencil_hi, shift_cache, hs,
        blend_a, blend_a3, blend_r_idx,
        transform, extrap_vals, searches,
    )
end

# ======================================================
# Callable interface
# ======================================================

# ---- Helpers ----

@inline function _phs_check_domain(itp::PHSInterpolantND{Tg, Tv, N}, query::NTuple{N, <:Real}) where {Tg, Tv, N}
    _validate_nd_domain(itp.grids, query, itp.extraps)
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
@inline function (itp::PHSInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kw...,  # absorb search/hint passed by AbstractInterpolantND protocol
    ) where {Tg, Tv, N}
    _phs_check_domain(itp, query)
    # Handle out-of-bounds (fills FillExtrap, etc.)
    oob = _try_fill_oob(query, itp.grids, itp.extraps, _phs_resolve_ops(deriv, Val(N)), first(itp.data))
    oob !== nothing && return oob
    ops = _phs_resolve_ops(deriv, Val(N))
    return _phs_eval(itp, query, ops)
end

# ---- Batch in-place (SoA + AoS via query protocol) ----

"""
    (itp::PHSInterpolantND)(out::AbstractVector, queries; deriv=EvalValue())

In-place batch evaluation. `queries` can be:
  - `Tuple{Vararg{AbstractVector,N}}` (SoA)
  - `AbstractVector{<:NTuple{N}}` or `AbstractVector{<:AbstractVector}` (AoS)

Uses `Threads.@threads :static` for parallelism.
All workspace is thread-local (AdaptiveArrayPools).
"""
function (itp::PHSInterpolantND{Tg, Tv, N})(
        out::AbstractVector,
        queries::Union{Tuple{Vararg{AbstractVector, N}}, AbstractVector};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kw...,  # absorb search/hint forwarded by AbstractInterpolantND protocol
    ) where {Tg, Tv, N}
    nq = _query_length(queries)
    length(out) == nq || _throw_query_output_mismatch(nq, length(out))
    _query_validate(queries)
    ops = _phs_resolve_ops(deriv, Val(N))

    Threads.@threads :static for k in 1:nq
        q = _extract_query_point(queries, k, Val(N))
        oob = _try_fill_oob(q, itp.grids, itp.extraps, ops, first(itp.data))
        if oob !== nothing
            @inbounds out[k] = oob
        else
            @inbounds out[k] = _phs_eval(itp, q, ops)
        end
    end
    return out
end

# Allocating batch evaluation is handled by AbstractInterpolantND protocol,
# which forwards to our in-place callable above via dynamic dispatch.
