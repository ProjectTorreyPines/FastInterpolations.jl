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

# N=1 collapse: a 1-axis grid tuple forwards to the native 1D pchip (skips the
# HeteroInterpolantND wrapper entirely). Per-axis 1-tuple kwargs unwrap to scalar;
# omitting `bc` lets the 1D NoBC() default apply. More specific than the `NTuple{N}`
# forwarder above, so it only claims N=1.
@inline pchip_interp(grids::Tuple{AbstractVector}, data::AbstractVector; kwargs...) =
    pchip_interp(only(grids), data; _unwrap_nd_kwargs(values(kwargs))...)

# N=1 scalar one-shot: bare scalar → scalar query `(q,)` → ND scalar one-shot
# (scalar output, not `[val]`). See linear_nd_interpolant.jl.
@inline pchip_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Real; kwargs...) =
    pchip_interp(grids, data, (q,); kwargs...)

# N=1 batch one-shot → lean 1D batch one-shot. See linear_nd_interpolant.jl.
@inline pchip_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; kwargs...) =
    pchip_interp(only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
@inline pchip_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; kwargs...) =
    pchip_interp!(output, only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
# Single-axis SoA `(xv,)` → 1D batch. See linear_nd_interpolant.jl.
@inline pchip_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    pchip_interp(only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)
@inline pchip_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    pchip_interp!(output, only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)

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

# N=1 collapse: forward to native 1D cardinal (`tension`/per-axis 1-tuple kwargs
# unwrap to scalar). More specific than the `NTuple{N}` forwarder above.
@inline cardinal_interp(grids::Tuple{AbstractVector}, data::AbstractVector; kwargs...) =
    cardinal_interp(only(grids), data; _unwrap_nd_kwargs(values(kwargs))...)

# N=1 scalar one-shot: bare scalar → scalar query `(q,)` → ND scalar one-shot.
@inline cardinal_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Real; kwargs...) =
    cardinal_interp(grids, data, (q,); kwargs...)

# N=1 batch one-shot → lean 1D batch one-shot. See linear_nd_interpolant.jl.
@inline cardinal_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; kwargs...) =
    cardinal_interp(only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
@inline cardinal_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; kwargs...) =
    cardinal_interp!(output, only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
# Single-axis SoA `(xv,)` → 1D batch. See linear_nd_interpolant.jl.
@inline cardinal_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    cardinal_interp(only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)
@inline cardinal_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    cardinal_interp!(output, only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)

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

# N=1 collapse: forward to native 1D akima (per-axis 1-tuple kwargs unwrap to
# scalar). More specific than the `NTuple{N}` forwarder above.
@inline akima_interp(grids::Tuple{AbstractVector}, data::AbstractVector; kwargs...) =
    akima_interp(only(grids), data; _unwrap_nd_kwargs(values(kwargs))...)

# N=1 scalar one-shot: bare scalar → scalar query `(q,)` → ND scalar one-shot.
@inline akima_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Real; kwargs...) =
    akima_interp(grids, data, (q,); kwargs...)

# N=1 batch one-shot → lean 1D batch one-shot. See linear_nd_interpolant.jl.
@inline akima_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; kwargs...) =
    akima_interp(only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
@inline akima_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::AbstractVector{<:Real}; kwargs...) =
    akima_interp!(output, only(grids), data, q; _unwrap_nd_kwargs(values(kwargs))...)
# Single-axis SoA `(xv,)` → 1D batch. See linear_nd_interpolant.jl.
@inline akima_interp(grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    akima_interp(only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)
@inline akima_interp!(output::AbstractVector, grids::Tuple{AbstractVector}, data::AbstractVector, q::Tuple{AbstractVector}; kwargs...) =
    akima_interp!(output, only(grids), data, only(q); _unwrap_nd_kwargs(values(kwargs))...)

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
