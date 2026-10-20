# ========================================
# GPUArrays Extension for FastInterpolations.jl
# ========================================
# GPU storage recognition and the GPU batch runner, for every GPUArrays backend (CUDA, Metal,
# AMDGPU, oneAPI, JLArrays):
#   ditp = adapt(MtlArray, linear_interp(x, y))
#   ditp(xq_gpu)          # allocating
#   ditp(out_gpu, xq_gpu) # in-place
#
# Storage: GPU arrays, including wrapped ones such as strided views (`AnyGPUArray`), are
# device storage; interpolants built from or moved to them hold it in `_DeviceArray`.
#
# Batch calls on GPU queries take this runner at the API entry, before any CPU planning
# (host output allocation, query-order search probing, host domain scans). Planning keeps
# the CPU batch contracts — call-time extrap override, NoExtrap throws before any output is
# written — while the work runs on the device: the NoExtrap domain check is one reduction
# whose bounds are computed in the kernel, and the evaluation is one broadcast of the
# family's scalar evaluation (`_itp_eval_scalar`), each element resolving its own searcher
# as the scalar call does. Families are admitted one by one, once their scalar evaluation
# is verified to compile in GPU kernels.

module FastInterpolationsGPUArraysExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant1D, LinearInterpolant, AbstractExtrap,
    AbstractSearchPolicy, DerivOp, EvalValue, _DeviceArray, _DeviceVector, _itp_grid,
    _itp_extrap, _itp_search, _itp_eval_scalar, _resolve_extrap_override, _promote_eltype,
    _resolve_search, _domain_bounds, _extract_primal, _ge, _le, _throw_batch_oob, NoBC, AbstractBC,
    StorePolicy, PeriodicBC, periodic_check, _PlainAxisBC, _resolve_exclusive_period,
    _throw_excl_endpoint_too_small, AbstractInterpolantND, LinearInterpolantND,
    _resolve_extrap_override_nd, _eval_nd_scalar_query, _query_eltype, _check_domain,
    _ExclusivePeriodicAxis, _ExclusivePeriodicData, _RawFields, _resolve_bc_period,
    _throw_excl_axis_period_too_small, _linear_eval_at_point, Searcher, NoHint, RefHint,
    AbstractEvalOp
import Adapt
using GPUArrays: AbstractGPUVector, AnyGPUArray, AnyGPUVector

FastInterpolations._is_device_storage(::AnyGPUArray) = true

# ── GPU-native construction ──
# Device inputs reach the shared normalization seams; these keep them on the device as
# `_DeviceArray` leaves — eltype conversion broadcast on the device, ownership per
# StorePolicy (copy by default, alias with `copy = false`), never a host `Vector`. A uniform
# grid still becomes a `_CachedRange`.
FastInterpolations._policy_axis(x::AnyGPUVector, ::_PlainAxisBC, ::Type{Tg}, ::StorePolicy{CG, CV, true}) where {Tg, CG, CV} =
    _DeviceArray(_device_convert(x, Tg))
FastInterpolations._policy_axis(x::AnyGPUVector, ::_PlainAxisBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} =
    _DeviceArray(_device_convert(x, Tg))

FastInterpolations._own_or_ref_values(y::AnyGPUVector, ::Type{Tv}, ::StorePolicy{CG, true, CA}) where {Tv, CG, CA} =
    _DeviceArray(_device_copy(y, Tv))
FastInterpolations._own_or_ref_values(y::AnyGPUVector{Tv}, ::Type{Tv}, ::StorePolicy{CG, false, CA}) where {Tv, CG, CA} =
    _DeviceArray(y)
FastInterpolations._own_or_ref_values(y::AnyGPUVector, ::Type{Tv}, ::StorePolicy{CG, false, CA}) where {Tv, CG, CA} =
    _DeviceArray(Tv.(y))

# ND factories normalize grids through the Tg-less seam (grids already value-promoted) and
# store data through `_own_or_ref_data`.
FastInterpolations._policy_axis(x::AnyGPUVector, ::_PlainAxisBC, ::StorePolicy{CG, CV, true}) where {CG, CV} = _DeviceArray(x)
FastInterpolations._policy_axis(x::AnyGPUVector, ::_PlainAxisBC, ::StorePolicy{CG, CV, false}) where {CG, CV} = _DeviceArray(x)
FastInterpolations._own_or_ref_data(data::AnyGPUArray, ::StorePolicy{CG, true, CA}) where {CG, CA} = _DeviceArray(copy(data))
FastInterpolations._own_or_ref_data(data::AnyGPUArray, ::StorePolicy{CG, false, CA}) where {CG, CA} = _DeviceArray(data)

_device_convert(a, ::Type{T}) where {T} = eltype(a) === T ? a : T.(a)
_device_copy(a, ::Type{T}) where {T} = eltype(a) === T ? copy(a) : T.(a)

# Periodic preparation on device data, before the grid is normalized. Inclusive: the
# endpoint-match check reads two values (one transfer at construction). Exclusive: the data
# gains its seam value by a device copy of `y[1]`; a GPU grid gains `x[1] + period` from its
# two endpoints, read once.
function FastInterpolations._check_periodic_endpoints(bc::PeriodicBC{:inclusive}, y::AnyGPUVector)
    periodic_check(bc) || return nothing
    FastInterpolations._check_periodic_endpoints(_host_endpoints(y))
    return nothing
end

FastInterpolations._extend_values(y::AnyGPUVector) = vcat(y, view(y, firstindex(y):firstindex(y)))

function FastInterpolations._extend_exclusive(x::AnyGPUVector, y::AbstractVector, bc::PeriodicBC)
    period = _resolve_exclusive_period(x, bc)
    x1, xn = _host_endpoints(x)
    x_end = x1 + eltype(x)(period)
    xn < x_end || _throw_excl_endpoint_too_small(period, x_end, xn)
    return vcat(x, fill!(similar(x, 1), x_end)), FastInterpolations._extend_values(y)
end

# ── Admission ──
# Admitted families list the arrays their evaluation reads; any other family is rejected
# before allocation or launch.
_gpu_batch_storage(itp::LinearInterpolant) = (itp.x, itp.y)
_gpu_batch_storage(itp::AbstractInterpolant1D) =
    throw(ArgumentError("GPU batch evaluation is not supported for $(nameof(typeof(itp))) yet"))

# Usable inside a kernel: device storage, or plain isbits data (e.g. `_CachedRange`).
_on_device(a) = a isa _DeviceArray || FastInterpolations._is_device_storage(a) || isbits(a)
_on_device(g::Union{_ExclusivePeriodicAxis, _ExclusivePeriodicData}) = _on_device(g.inner)

function _check_gpu_batch(itp)
    all(_on_device, _gpu_batch_storage(itp)) || throw(
        ArgumentError(
            "the interpolant's data is on the host; move it to the device first, e.g. `adapt(CuArray, itp)`"
        )
    )
    return nothing
end

@noinline _throw_mixed_storage() =
    throw(ArgumentError("GPU batch evaluation needs both the output and the queries on the device"))

# ── Per-element evaluator launched by broadcast ──
struct _DeviceEval{I, E, O, P}
    itp::I
    extrap::E
    op::O
    search::P
end

@inline function (f::_DeviceEval)(xq)
    searcher = _resolve_search(_itp_grid(f.itp), xq, f.search, nothing)
    return _itp_eval_scalar(f.itp, xq, f.extrap, f.op, searcher)
end

Adapt.adapt_structure(to, f::_DeviceEval) = _DeviceEval(Adapt.adapt(to, f.itp), f.extrap, f.op, f.search)

# ── Entry points ──
function (itp::AbstractInterpolant1D{Tg, Tv})(
        xq::AnyGPUVector{Tq};
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Number}
    _check_gpu_batch(itp)
    output = similar(xq, _promote_eltype(itp, Tq, deriv))
    return _gpu_eval!(output, itp, xq, deriv, extrap, search, hint)
end

(itp::AbstractInterpolant1D{Tg, Tv})(
    output::AbstractVector, xq::AnyGPUVector{Tq}; kwargs...
) where {Tg, Tv, Tq <: Number} = _gpu_inplace(itp, output, xq; kwargs...)

function _gpu_inplace(
        itp, output, xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap} = nothing,
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tq}
    _check_gpu_batch(itp)
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    length(output) == length(xq) ||
        throw(DimensionMismatch("output length $(length(output)) must match xq length $(length(xq))"))
    return _gpu_eval!(output, itp, xq, deriv, extrap, search, hint)
end

# GPU output with host queries would fall into the host loop (scalar writes). Host queries
# are matched as `DenseVector`: a broader `AbstractVector` would collide with the
# `GriddedQuery` callables (an `AbstractArray`), which keep their own path. Dense GPU
# queries match both this and the in-place method above; they take the device path.
(itp::AbstractInterpolant1D{Tg, Tv})(
    ::AnyGPUVector, ::DenseVector{Tq}; kwargs...
) where {Tg, Tv, Tq <: Number} = _throw_mixed_storage()
(itp::AbstractInterpolant1D{Tg, Tv})(
    output::AnyGPUVector, xq::AbstractGPUVector{Tq}; kwargs...
) where {Tg, Tv, Tq <: Number} = _gpu_inplace(itp, output, xq; kwargs...)

# ── One-shot ──
# One-shot calls on GPU queries take the core one-shot route (`linear_interp` /
# `linear_interp!` are the core methods): the storage leaves and the planning/execution
# seams below it are the only GPU methods. No interpolant is built and nothing is copied
# unless an eltype conversion asks for it.
FastInterpolations._resolve_axis(x::AnyGPUVector, ::_PlainAxisBC, ::Type{Tg}) where {Tg} = _DeviceArray(_device_convert(x, Tg))

# `:exclusive`: the shared lazy wrapper around a device leaf; its seam point `x[1] + period`
# comes from the two endpoints, read once on the host.
function FastInterpolations._resolve_axis(x::AnyGPUVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    inner = _DeviceArray(_device_convert(x, Tg))
    period = _resolve_bc_period(inner, bc).period
    Te = promote_type(Tg, typeof(period))
    inner_e = Te === Tg ? inner : _DeviceArray(Te.(parent(inner)))
    x1, xn = _host_endpoints(inner_e)
    x_max = x1 + Te(period)
    xn < x_max || _throw_excl_axis_period_too_small(period, x_max, xn)
    return _ExclusivePeriodicAxis{Te, typeof(inner_e), typeof(period)}(_RawFields(), inner_e, period, x_max)
end

FastInterpolations._resolve_data(y::AnyGPUArray, ::NoBC) = _DeviceArray(y)
function FastInterpolations._resolve_data(y::AnyGPUArray, bc::PeriodicBC{:inclusive})
    periodic_check(bc) && FastInterpolations._check_periodic_endpoints(bc, y)
    return _DeviceArray(y)
end
FastInterpolations._resolve_data(y::AnyGPUArray, ::PeriodicBC{:extended}) = _DeviceArray(y)
FastInterpolations._resolve_data(y::AnyGPUArray, ::PeriodicBC{:exclusive}) = _ExclusivePeriodicData(_DeviceArray(y))

FastInterpolations._check_periodic_endpoints(bc::PeriodicBC{:inclusive}, y::_DeviceVector) =
    FastInterpolations._check_periodic_endpoints(bc, parent(y))

FastInterpolations._alloc_query_output(::Type{T}, q::AnyGPUArray) where {T} = similar(q, T)

# Searching is planned before the loop: reject host grids here (before any query is read),
# and host hints (device threads cannot write one back).
function FastInterpolations._resolve_search(grid, q::AnyGPUArray, search, ::Nothing)
    _on_device(grid) || _throw_host_oneshot()
    return FastInterpolations._to_searcher(FastInterpolations._resolve_search_policy(grid, q, search, nothing), nothing)
end
FastInterpolations._resolve_search(grid, ::AnyGPUArray, search, ::Base.RefValue{Int}) =
    throw(ArgumentError("`hint` is not supported for GPU array queries (device threads cannot share a host Ref)"))
@noinline _throw_host_oneshot() =
    throw(ArgumentError("one-shot GPU queries need the grid and the data on the device too"))

function FastInterpolations._linear_interp_loop!(
        output::AbstractArray,
        x::AbstractVector,
        y::AbstractVector,
        x_targets::AnyGPUArray,
        extrap::AbstractExtrap,
        op::AbstractEvalOp,
        searcher::Searcher
    )
    (_on_device(x) && _on_device(y)) || _throw_host_oneshot()
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    isempty(x_targets) && return output
    extrap_eff = _gpu_domain_check(x, x_targets, extrap)
    output .= _OneShotEval{typeof(searcher)}(x, y, extrap_eff, op).(x_targets)
    return output
end

# The CPU one-shot point evaluation, per thread. A searcher carries no shared state across
# threads: each one starts fresh (a hint-carrying policy keeps a thread-local hint).
struct _OneShotEval{S, X, Y, E, O}
    x::X
    y::Y
    extrap::E
    op::O
end
_OneShotEval{S}(x, y, extrap, op) where {S} = _OneShotEval{S, typeof(x), typeof(y), typeof(extrap), typeof(op)}(x, y, extrap, op)
@inline (f::_OneShotEval{S})(q) where {S} = _linear_eval_at_point(f.x, f.y, q, f.extrap, f.op, _fresh_searcher(S))
Adapt.adapt_structure(to, f::_OneShotEval{S}) where {S} = _OneShotEval{S}(Adapt.adapt(to, f.x), Adapt.adapt(to, f.y), f.extrap, f.op)

@inline _fresh_searcher(::Type{Searcher{P, NoHint}}) where {P} = Searcher{P, NoHint}(NoHint())
@inline _fresh_searcher(::Type{Searcher{P, RefHint}}) where {P} = Searcher{P, RefHint}(RefHint())

function _gpu_eval!(output, itp, xq, deriv, extrap, search, hint)
    hint === nothing ||
        throw(ArgumentError("`hint` is not supported for GPU array queries (device threads cannot share a host Ref)"))
    isempty(xq) && return output
    extrap_eff = _gpu_domain_check(_itp_grid(itp), xq, _resolve_extrap_override(_itp_extrap(itp), extrap))
    # Broadcast, not `map!`: the backend's native broadcast kernel is leaner than the
    # generic KernelAbstractions kernel behind GPUArrays' `map!`.
    output .= _DeviceEval(itp, extrap_eff, deriv, search).(xq)
    return output
end

# ── Domain check ──
# Only NoExtrap needs a batch check; the kernel then runs with `InBounds`. The CPU's
# `InBounds` promotion for Clamp/Fill would cost a device reduction plus a sync per call,
# while the per-element boundary branch it removes is nearly free in a kernel. The domain
# bounds come from `_domain_bounds` inside the reduction kernel — the same code every axis
# type uses on the CPU — so no endpoint is read on the host unless the check fails.
function _gpu_domain_check(x, xq, ::NoExtrap)
    mapreduce(q -> _in_domain(x, q), &, xq) || _throw_gpu_batch_oob(x, xq)
    return InBounds()
end
@inline _gpu_domain_check(_, _, e::AbstractExtrap) = e

@inline function _in_domain(x, q)
    lo, hi = _domain_bounds(x)
    qp = _extract_primal(q)
    return _ge(qp, _extract_primal(lo)) & _le(qp, _extract_primal(hi))
end

# Failure path only: the CPU error (same bounds, same message) from a host copy of the
# grid's two endpoints; the query extremes are device reductions.
@noinline _throw_gpu_batch_oob(x, xq) = _throw_batch_oob(_host_endpoints(x), xq)
_host_endpoints(x) = x
_host_endpoints(x::_DeviceVector) = _host_endpoints(parent(x))
_host_endpoints(a::AnyGPUVector) = [_host_element(a, firstindex(a)), _host_element(a, lastindex(a))]
_host_element(a, i) = only(Array(view(a, i:i)))

# ── ND batch runner ──
# SoA queries (a tuple of GPU coordinate vectors) and AoS queries (a GPU vector of points:
# N-tuples or static vectors). As in 1D, the runner takes the call at the entry, checks the
# NoExtrap axes on the device, and broadcasts the family's scalar ND evaluation.
_gpu_batch_storage(itp::LinearInterpolantND) = (itp.grids..., itp.data)
_gpu_batch_storage(itp::AbstractInterpolantND) =
    throw(ArgumentError("GPU batch evaluation is not supported for $(nameof(typeof(itp))) yet"))

const _GPUPoints = AnyGPUVector{<:Union{Tuple, AbstractVector}}
const _GPUQueriesND = Union{Tuple{Vararg{AnyGPUVector}}, _GPUPoints}

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        queries::Tuple{Vararg{AnyGPUVector, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kwargs...
    ) where {Tg, Tv, N}
    _check_gpu_batch(itp)
    output = similar(first(queries), _promote_eltype(itp, _query_eltype(queries), deriv))
    return _gpu_eval_nd!(output, itp, queries; deriv, kwargs...)
end

function (itp::AbstractInterpolantND{Tg, Tv, N})(
        queries::_GPUPoints;
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kwargs...
    ) where {Tg, Tv, N}
    _check_gpu_batch(itp)
    output = similar(queries, _promote_eltype(itp, _query_eltype(queries), deriv))
    return _gpu_eval_nd!(output, itp, queries; deriv, kwargs...)
end

(itp::AbstractInterpolantND{Tg, Tv, N})(
    output::AbstractVector, queries::Tuple{Vararg{AnyGPUVector, N}}; kwargs...
) where {Tg, Tv, N} = _gpu_inplace_nd(itp, output, queries; kwargs...)
(itp::AbstractInterpolantND{Tg, Tv, N})(
    output::AbstractVector, queries::_GPUPoints; kwargs...
) where {Tg, Tv, N} = _gpu_inplace_nd(itp, output, queries; kwargs...)

function _gpu_inplace_nd(itp, output, queries; kwargs...)
    _check_gpu_batch(itp)
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    length(output) == _nd_length(queries) ||
        throw(DimensionMismatch("output length $(length(output)) must match the number of query points $(_nd_length(queries))"))
    return _gpu_eval_nd!(output, itp, queries; kwargs...)
end

_nd_length(q::Tuple) = length(first(q))
_nd_length(q) = length(q)

function _gpu_eval_nd!(
        output, itp::AbstractInterpolantND{Tg, Tv, N}, queries;
        deriv = EvalValue(),
        extrap::Union{Nothing, AbstractExtrap, Tuple} = nothing,
        search = itp.searches,
        hint = nothing
    ) where {Tg, Tv, N}
    hint === nothing ||
        throw(ArgumentError("`hint` is not supported for GPU array queries (device threads cannot share a host Ref)"))
    queries isa Tuple && (
        all(q -> length(q) == length(first(queries)), queries) ||
            throw(DimensionMismatch("all coordinate vectors of a SoA query must have the same length"))
    )
    _nd_length(queries) == 0 && return output
    overrides = _gpu_domain_check_nd(itp.grids, queries, _resolve_extrap_override_nd(itp, extrap))
    f = _DeviceEvalND(itp, deriv, overrides, search)
    if queries isa Tuple
        output .= f.(tuple.(queries...))
    else
        output .= f.(queries)
    end
    return output
end

# Per-axis call-time overrides for the kernel: `InBounds()` on each NoExtrap axis once the
# device check passed, `nothing` (stored extrap) elsewhere.
function _gpu_domain_check_nd(grids::NTuple{N, Any}, queries::Tuple, extraps) where {N}
    return map(_axis_override, grids, queries, extraps, ntuple(identity, Val(N)))
end
_axis_override(_, _, ::AbstractExtrap, _) = nothing
function _axis_override(g, q, ::NoExtrap, d)
    mapreduce(v -> _in_domain(g, v), &, q) || _throw_batch_oob(_host_endpoints(g), q, d)
    return InBounds()
end

function _gpu_domain_check_nd(grids::NTuple{N, Any}, queries, extraps) where {N}
    overrides = map(e -> e isa NoExtrap ? InBounds() : nothing, extraps)
    all(isnothing, overrides) && return overrides
    mapreduce(q -> _point_in_domain(grids, q, extraps), &, queries) || _throw_point_oob(grids, queries, extraps)
    return overrides
end

@inline _point_in_domain(grids::NTuple{N, Any}, q, extraps) where {N} =
    reduce(&, map((g, e, d) -> _axis_in_domain(g, @inbounds(q[d]), e), grids, extraps, ntuple(identity, Val(N))))
@inline _axis_in_domain(g, v, ::NoExtrap) = _in_domain(g, v)
@inline _axis_in_domain(_, _, ::AbstractExtrap) = true

# Failure path only: the first out-of-domain point (CPU order: point-major, then axis) is
# copied to the host and re-checked by the CPU code, so the error matches the CPU's.
@noinline function _throw_point_oob(grids::NTuple{N, Any}, queries, extraps) where {N}
    i = findfirst(q -> !_point_in_domain(grids, q, extraps), queries)
    q = _host_element(queries, i)
    map(grids, extraps, ntuple(identity, Val(N))) do g, e, d
        e isa NoExtrap && _check_domain(_host_endpoints(g), q[d], NoExtrap(), d)
        nothing
    end
    return nothing
end

struct _DeviceEvalND{N, I, O, E, P}
    itp::I
    deriv::O
    extrap::E
    search::P
end
_DeviceEvalND(itp::AbstractInterpolantND{Tg, Tv, N}, deriv, extrap, search) where {Tg, Tv, N} =
    _DeviceEvalND{N, typeof(itp), typeof(deriv), typeof(extrap), typeof(search)}(itp, deriv, extrap, search)

@inline (f::_DeviceEvalND{N})(q) where {N} =
    _eval_nd_scalar_query(f.itp, ntuple(d -> @inbounds(q[d]), Val(N)), f.deriv, f.extrap, f.search, nothing)

Adapt.adapt_structure(to, f::_DeviceEvalND) = _DeviceEvalND(Adapt.adapt(to, f.itp), f.deriv, f.extrap, f.search)

end # module
