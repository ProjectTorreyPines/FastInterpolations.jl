# ═══════════════════════════════════════════════════════════════
# Local Cubic Hermite — ND convenience forwarders to the unified `interp` API
# ═══════════════════════════════════════════════════════════════
#
# `pchip_interp` / `cardinal_interp` / `akima_interp` are 1D-native and have
# no dedicated ND types. The local Hermite family's ND path is provided
# through `HeteroInterpolantND` via the unified `interp((grids...), data;
# method=...)` entry point (see `src/hetero/`).
#
# To avoid forcing users to rewrite `pchip_interp((x, y), data)` as
# `interp((x, y), data; method=PchipInterp())` by hand, the three per-method
# functions also accept ND signatures that simply forward to `interp` with the
# right `method=` kwarg. The forwarders cover all four ND call shapes:
#
#   1. Interpolant construction: `pchip_interp((x, y), data)`
#   2. Scalar one-shot:          `pchip_interp((x, y), data, (qx, qy))`
#   3. Batch one-shot:           `pchip_interp((x, y), data, queries)`
#   4. Batch in-place:           `pchip_interp!(out, (x, y), data, queries)`
#
# `bc` / `bcs` kwargs:
#   - `bc::AbstractBC`            — broadcast same BC to all axes (convenience)
#   - `bcs::NTuple{N, AbstractBC}` — explicit per-axis BC tuple
#   - omit both                   — defaults to NoBC() per axis (backward-compat)
#
# `CubicHermiteInterp` (user-supplied slopes) is intentionally NOT forwarded:
# user-supplied `dydx` is a 1D concept, and per-axis slope arrays for ND
# require a separate design.

# ────────────────────────────────────────────
# BC normalization helper
# ────────────────────────────────────────────
# Pick which method-tuple to send into `interp` based on `bc`/`bcs`.
# Users can specify per-axis BC via `bcs=(bc1, bc2, ...)`, broadcast a single
# BC across all axes via `bc=PeriodicBC(...)`, or omit both and inherit
# NoBC() defaults from `MethodCtor()` (backward-compat).
@inline function _build_local_hermite_method_tuple(
        ::Type{MethodCtor}, ::Val{N}, bc, bcs, args...,
    ) where {MethodCtor, N}
    if bcs !== nothing && bc !== nothing
        throw(ArgumentError("Specify either `bc` (broadcast) or `bcs` (per-axis), not both."))
    elseif bcs !== nothing
        length(bcs) == N || throw(ArgumentError("`bcs` length must equal grid dim N=$N, got $(length(bcs))."))
        return ntuple(d -> MethodCtor(args..., bcs[d]), Val(N))
    elseif bc !== nothing
        return ntuple(_ -> MethodCtor(args..., bc), Val(N))
    else
        # No explicit BC — broadcast a single backward-compat constructor (NoBC default).
        return ntuple(_ -> MethodCtor(args...), Val(N))
    end
end

# ── PCHIP ──

@inline function pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc, bcs)
    return interp(grids, data; method = methods, kwargs...)
end

@inline function pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc, bcs)
    return interp(grids, data, query; method = methods, kwargs...)
end

@inline function pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc, bcs)
    return interp(grids, data, queries; method = methods, kwargs...)
end

@inline function pchip_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc, bcs)
    return interp!(output, grids, data, queries; method = methods, kwargs...)
end

# ── Cardinal (forwards `tension` into `CardinalInterp(tension, bc)`) ──

@inline function cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        tension = 0.0,
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, bcs, tension)
    return interp(grids, data; method = methods, kwargs...)
end

@inline function cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        tension = 0.0,
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, bcs, tension)
    return interp(grids, data, query; method = methods, kwargs...)
end

@inline function cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        tension = 0.0,
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, bcs, tension)
    return interp(grids, data, queries; method = methods, kwargs...)
end

@inline function cardinal_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        tension = 0.0,
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, bcs, tension)
    return interp!(output, grids, data, queries; method = methods, kwargs...)
end

# ── Akima ──

@inline function akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc, bcs)
    return interp(grids, data; method = methods, kwargs...)
end

@inline function akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc, bcs)
    return interp(grids, data, query; method = methods, kwargs...)
end

@inline function akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc, bcs)
    return interp(grids, data, queries; method = methods, kwargs...)
end

@inline function akima_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, Nothing} = nothing,
        bcs::Union{Tuple, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc, bcs)
    return interp!(output, grids, data, queries; method = methods, kwargs...)
end
