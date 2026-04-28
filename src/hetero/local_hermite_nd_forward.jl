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
# `bc` kwarg (unified shape, mirrors `linear_interp` / `constant_interp`):
#   - `bc::AbstractBC`              — broadcast same BC to all axes
#   - `bc::NTuple{N, AbstractBC}`   — explicit per-axis BC tuple
#   - omit                          — defaults to NoBC() per axis (backward-compat)
#
# `CubicHermiteInterp` (user-supplied slopes) is intentionally NOT forwarded:
# user-supplied `dydx` is a 1D concept, and per-axis slope arrays for ND
# require a separate design.

# ────────────────────────────────────────────
# BC normalization helper
# ────────────────────────────────────────────
# Build the per-axis method tuple from the user-facing `bc` kwarg. Single
# `AbstractBC` broadcasts to every axis; `NTuple{N, AbstractBC}` is consumed
# axis-wise; `nothing` falls back to the method ctor's NoBC() default. Length
# and tuple-eltype validation happens at the kwarg boundary via the
# `Union{AbstractBC, NTuple{N, AbstractBC}, Nothing}` signature on each
# forwarder, so this helper assumes a well-typed `bc`.
@inline function _build_local_hermite_method_tuple(
        ::Type{MethodCtor}, ::Val{N}, bc, args...,
    ) where {MethodCtor, N}
    if bc isa Tuple
        return ntuple(d -> MethodCtor(args..., bc[d]), Val(N))
    elseif bc isa AbstractBC
        return ntuple(_ -> MethodCtor(args..., bc), Val(N))
    else
        # `bc === nothing` — broadcast the backward-compat zero-arg ctor (NoBC default).
        return ntuple(_ -> MethodCtor(args...), Val(N))
    end
end

# ── PCHIP ──

@inline function pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc)
    return interp(grids, data; method = methods, kwargs...)
end

@inline function pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc)
    return interp(grids, data, query; method = methods, kwargs...)
end

@inline function pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc)
    return interp(grids, data, queries; method = methods, kwargs...)
end

@inline function pchip_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(PchipInterp, Val(N), bc)
    return interp!(output, grids, data, queries; method = methods, kwargs...)
end

# ── Cardinal (forwards `tension` into `CardinalInterp(tension, bc)`) ──

@inline function cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        tension = 0.0,
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, tension)
    return interp(grids, data; method = methods, kwargs...)
end

@inline function cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        tension = 0.0,
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, tension)
    return interp(grids, data, query; method = methods, kwargs...)
end

@inline function cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        tension = 0.0,
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, tension)
    return interp(grids, data, queries; method = methods, kwargs...)
end

@inline function cardinal_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        tension = 0.0,
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(CardinalInterp, Val(N), bc, tension)
    return interp!(output, grids, data, queries; method = methods, kwargs...)
end

# ── Akima ──

@inline function akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc)
    return interp(grids, data; method = methods, kwargs...)
end

@inline function akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc)
    return interp(grids, data, query; method = methods, kwargs...)
end

@inline function akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc)
    return interp(grids, data, queries; method = methods, kwargs...)
end

@inline function akima_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        bc::Union{AbstractBC, NTuple{N, AbstractBC}, Nothing} = nothing,
        kwargs...,
    ) where {N}
    methods = _build_local_hermite_method_tuple(AkimaInterp, Val(N), bc)
    return interp!(output, grids, data, queries; method = methods, kwargs...)
end
