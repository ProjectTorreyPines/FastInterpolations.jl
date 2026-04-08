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
# right `method=` kwarg. The forwarders cover all three ND call shapes:
#
#   1. Interpolant construction: `pchip_interp((x, y), data)`
#   2. Scalar one-shot:          `pchip_interp((x, y), data, (qx, qy))`
#   3. Batch one-shot:           `pchip_interp((x, y), data, queries)`
#   4. Batch in-place:           `pchip_interp!(out, (x, y), data, queries)`
#
# `CubicHermiteInterp` (user-supplied slopes) is intentionally NOT forwarded:
# user-supplied `dydx` is a 1D concept, and per-axis slope arrays for ND
# require a separate design.

# ── PCHIP ──

@inline pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        kwargs...,
    ) where {N} =
    interp(grids, data; method = PchipInterp(), kwargs...)

@inline pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...,
    ) where {N} =
    interp(grids, data, query; method = PchipInterp(), kwargs...)

@inline pchip_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        kwargs...,
    ) where {N} =
    interp(grids, data, queries; method = PchipInterp(), kwargs...)

@inline pchip_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        kwargs...,
    ) where {N} =
    interp!(output, grids, data, queries; method = PchipInterp(), kwargs...)

# ── Cardinal (forwards `tension` into `CardinalInterp(tension)`) ──

@inline cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        tension = 0.0,
        kwargs...,
    ) where {N} =
    interp(grids, data; method = CardinalInterp(tension), kwargs...)

@inline cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        tension = 0.0,
        kwargs...,
    ) where {N} =
    interp(grids, data, query; method = CardinalInterp(tension), kwargs...)

@inline cardinal_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        tension = 0.0,
        kwargs...,
    ) where {N} =
    interp(grids, data, queries; method = CardinalInterp(tension), kwargs...)

@inline cardinal_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        tension = 0.0,
        kwargs...,
    ) where {N} =
    interp!(output, grids, data, queries; method = CardinalInterp(tension), kwargs...)

# ── Akima ──

@inline akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N};
        kwargs...,
    ) where {N} =
    interp(grids, data; method = AkimaInterp(), kwargs...)

@inline akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...,
    ) where {N} =
    interp(grids, data, query; method = AkimaInterp(), kwargs...)

@inline akima_interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        kwargs...,
    ) where {N} =
    interp(grids, data, queries; method = AkimaInterp(), kwargs...)

@inline akima_interp!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        kwargs...,
    ) where {N} =
    interp!(output, grids, data, queries; method = AkimaInterp(), kwargs...)
