# ========================================
# CubicHermiteInterpolantND — outer ctor, callable, protocol traits
# ========================================
#
# Wires `hermite_interp(grids, data, partials; …)` into the build path,
# implements `_locate_cell` / `_eval_at_cell` (the `AbstractInterpolantND`
# protocol traits — they let the shared callable in `interpolant_protocol.jl`
# do scalar evaluation, batch evaluation, gradient/hessian/laplacian, etc.
# with no further work), and provides the direct scalar callable.
#
# The eval kernel `_eval_nd_cell` lives in `src/cubic/nd/cubic_nd_eval.jl` —
# it operates on the packed `_NodalDerivativesND.partials` we share with
# `CubicInterpolantND`, so no kernel duplication is required.

# ========================================
# OUTER CONSTRUCTOR — public build entry
# ========================================

"""
    CubicHermiteInterpolantND(grids, data, partials::HermitePartials; bc, extrap, search)

Build a cubic Hermite ND interpolant from grid vectors, function values, and
user-supplied mixed partials. Mirrors the option shape of `cubic_interp` /
`quadratic_interp` (per-axis `bc` / `extrap` / `search`).

# Phase 1a BC restrictions
Only `NoBC()`, `PeriodicBC(...; endpoint=:inclusive)`, and
`PeriodicBC(...; endpoint=:exclusive)` are accepted. User partials supersede
any BC-derived ones, so richer BC families (BCPair, CubicFit, ...) have no
role here — use `cubic_interp` or the auto-slope methods if you need them.
"""
function CubicHermiteInterpolantND(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N, Tv_part, K};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}} = NoBC(),
        extrap::Union{AbstractExtrap, NTuple{N, AbstractExtrap}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, NTuple{N, AbstractSearchPolicy}} = AutoSearch(),
    ) where {N, Tv_part, K}
    # Phase 1a only accepts full mixed partials (K = 2^N - 1). Future
    # FirstOnly support would dispatch on K (or reintroduce a Completeness
    # marker) — for now we hard-reject any other K so the contract is loud.
    K == (1 << N) - 1 || _throw_partials_not_full_mixed(N, K)

    # Promote across (grid, data, partials) to a single Tv.
    grids_typed, _, Tv_promoted, _ = _nd_promote_grids(grids, data)
    Tv = promote_type(Tv_promoted, Tv_part)

    data_typed = _coerce_data_eltype(data, Tv, Val(N))
    partials_typed = _coerce_partials_eltype(partials, Tv, Val(N))

    _validate_nd_grids(grids_typed, data_typed)

    bcs = _resolve_bcs_nd(bc, Val(N))
    searches = _resolve_search_nd(search, Val(N))

    _validate_hermite_nd_bcs_phase1a(bcs)
    _validate_partial_sizes(data_typed, partials_typed)
    _validate_inclusive_seams(data_typed, partials_typed, bcs)

    grids_ext, nodal_derivs, bcs_post =
        _pack_and_extend_nodal_derivs(grids_typed, data_typed, partials_typed, bcs)

    extraps_val = _resolve_extrap(extrap, bcs_post, Val(N), Tv)

    return CubicHermiteInterpolantND(grids_ext, nodal_derivs, bcs_post, extraps_val, searches)
end

@noinline function _throw_partials_not_full_mixed(N::Int, K::Int)
    expected = (1 << N) - 1
    throw(
        ArgumentError(
            "CubicHermiteInterpolantND (Phase 1a): partials must contain every " *
                "non-zero multiindex in {0,1}^N (K = 2^N - 1 = $expected for N=$N), got K=$K. " *
                "Construct via `HermitePartials(...)`.",
        )
    )
end

# Zero-copy when eltype already matches; broadcast-convert otherwise.
@inline _coerce_data_eltype(data::AbstractArray{Tv, N}, ::Type{Tv}, ::Val{N}) where {Tv, N} = data
@inline _coerce_data_eltype(data::AbstractArray{<:Any, N}, ::Type{Tv}, ::Val{N}) where {Tv, N} = Tv.(data)

# Zero-copy when eltype already matches; rebuild HermitePartials with
# converted arrays otherwise. The constructor itself already enforces a
# single common Tv across all stored arrays, so a single type check suffices.
@inline _coerce_partials_eltype(p::HermitePartials{N, Tv, K}, ::Type{Tv}, ::Val{N}) where {N, Tv, K} = p
@inline function _coerce_partials_eltype(
        p::HermitePartials{N, Tv_in, K}, ::Type{Tv}, ::Val{N},
    ) where {N, Tv_in, K, Tv}
    converted = ntuple(k -> convert(Array{Tv, N}, p.partials[k]), Val(K))
    return HermitePartials{N, Tv, K, typeof(first(converted))}(converted)
end

# ========================================
# PUBLIC API — hermite_interp(grids, data, partials; …)
# ========================================

"""
    hermite_interp(grids::Tuple, data, partials::HermitePartials; bc, extrap, search) -> CubicHermiteInterpolantND

ND cubic Hermite interpolant from user-supplied data + mixed partials. See
[`CubicHermiteInterpolantND`](@ref) for option semantics and Phase 1a BC
restrictions.

# Example (N=2)
```julia
x = range(0.0, 1.0, 11)
y = range(0.0, 2π, 21)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
partials = HermitePartials(
    (1, 0) => [ cos(xi) * cos(yj) for xi in x, yj in y],
    (0, 1) => [-sin(xi) * sin(yj) for xi in x, yj in y],
    (1, 1) => [-cos(xi) * sin(yj) for xi in x, yj in y],
)
itp = hermite_interp((x, y), data, partials)
itp((0.3, 1.7))
```
"""
@inline function hermite_interp(
        grids::Tuple{Vararg{AbstractVector, N}},
        data::AbstractArray{<:Any, N},
        partials::HermitePartials{N};
        kwargs...,
    ) where {N}
    return CubicHermiteInterpolantND(grids, data, partials; kwargs...)
end

# ========================================
# CALLABLE INTERFACE
# ========================================

"""
    (itp::CubicHermiteInterpolantND)(query; deriv=EvalValue(), search=itp.searches)

Evaluate ND cubic Hermite at `query::NTuple{N, Real}`. Supports `deriv` as
`DerivOp` (same order all axes) or `NTuple{N, DerivOp}` (per-axis).
"""
@inline function (itp::CubicHermiteInterpolantND{Tg, Tv, N})(
        query::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing,
    ) where {Tg, Tv, N}
    resolved = map(_resolve_grididx, query, itp.grids)
    ops = _resolve_deriv_nd(deriv, Val(N))
    policies = _resolve_search_nd(search, Val(N))
    hints = _ensure_hint_nd(hint, Val(N))
    mono = _scalar_mono(hint, Val(N))
    return _eval_nd_at_point(itp, resolved, ops, policies, hints, mono)
end

# ========================================
# PROTOCOL TRAITS — _locate_cell + _eval_at_cell
# ========================================
#
# These mirror the CubicInterpolantND implementations verbatim because the
# storage layout (packed `_NodalDerivativesND.partials`) and the eval kernel
# (`_eval_nd_cell` from `cubic_nd_eval.jl`) are identical. The only
# difference between the two interpolants is *how* `nodal_derivs` was filled
# — auto-solved (Cubic) vs. user-supplied (CubicHermite).

@inline function _locate_cell(
        itp::CubicHermiteInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}},
        extraps::Tuple{Vararg{AbstractExtrap, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
        mono::NTuple{N, Bool},
    ) where {Tg, Tv, N}
    q_evals = _handle_all_extraps(query, itp.grids, extraps)
    indices, Ls, _ = _search_all_intervals(q_evals, itp.grids, policies, hints, mono)
    hs, inv_hs, dLs = _compute_all_local_params(q_evals, itp.grids, indices, Ls)
    return (itp.nodal_derivs.partials, indices, hs, inv_hs, dLs)
end

@inline function _eval_at_cell(
        ::CubicHermiteInterpolantND,
        cell::Tuple,
        ops::NTuple{N, AbstractEvalOp},
    ) where {N}
    partials, indices, hs, inv_hs, dLs = cell
    return _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)
end

# Per-method sample-of-Tv used by fill-extrap paths.
@inline _sample_data(itp::CubicHermiteInterpolantND) = @inbounds first(itp.nodal_derivs.partials)
