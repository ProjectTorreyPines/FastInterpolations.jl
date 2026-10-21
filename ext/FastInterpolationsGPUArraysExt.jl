# ========================================
# GPUArrays Extension for FastInterpolations.jl
# ========================================
# GPU storage recognition and GPU batch execution, for every GPUArrays backend (CUDA, Metal,
# AMDGPU, oneAPI, JLArrays):
#   ditp = adapt(MtlArray, linear_interp(x, y))
#   ditp(xq_gpu)          # allocating
#   ditp(out_gpu, xq_gpu) # in-place
#
# Storage: GPU arrays, including wrapped ones such as strided views (`AnyGPUArray`), are
# device storage; interpolants built from or moved to them hold it in `_DeviceArray`.
#
# Batch calls on GPU queries (persistent and one-shot, any query shape) take the core API
# routes; the GPU methods sit at the seams below them: output allocation, search planning
# and the batch loop. Planning reads no query and keeps the CPU batch contracts (call-time
# extrap override, NoExtrap throws before any output is written), while the work runs on
# the device: the NoExtrap domain check is one reduction whose bounds are computed in the
# kernel, and the loop is one broadcast of the family's point evaluation. Families are
# admitted one by one, once their point evaluation is verified to compile in GPU kernels.

module FastInterpolationsGPUArraysExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant1D, LinearInterpolant, AbstractExtrap,
    _DeviceArray, _DeviceVector, _itp_grid, _itp_eval_scalar, _domain_bounds, _extract_primal, _ge, _le, _throw_batch_oob,
    StorePolicy, PeriodicBC, periodic_check, _PlainAxisBC, _resolve_exclusive_period,
    _throw_excl_endpoint_too_small, AbstractInterpolantND, LinearInterpolantND,
    _eval_nd_scalar_query, _check_domain, _query_validate, _query_length,
    _ExclusivePeriodicAxis, _ExclusivePeriodicData, _RawFields, _resolve_bc_period,
    _throw_excl_axis_period_too_small, _linear_eval_at_point, Searcher, NoHint, RefHint,
    AbstractEvalOp
import Adapt
using GPUArrays: AnyGPUArray, AnyGPUVector

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
# at its loop seam, before any query is read or any kernel launched.
_gpu_batch_storage(itp::LinearInterpolant) = (itp.x, itp.y)
_gpu_batch_storage(itp::LinearInterpolantND) = (itp.grids..., itp.data)
_gpu_batch_storage(itp::Union{AbstractInterpolant1D, AbstractInterpolantND}) =
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
@noinline _throw_gpu_hint() =
    throw(ArgumentError("`hint` is not supported for GPU array queries (device threads cannot share a host Ref)"))

# ── Planning seams (persistent and one-shot, 1D and ND) ──
# The output takes the query's shape on the device (SoA: the shape of its coordinate arrays).
FastInterpolations._alloc_query_output(::Type{T}, q::AnyGPUArray) where {T} = similar(q, T)
FastInterpolations._alloc_query_output(::Type{T}, q::Tuple{AnyGPUArray, Vararg{AnyGPUArray}}) where {T} =
    similar(first(q), T)

# Search is planned without reading a query: device threads share no hint, and a host
# grid's query-order probe would read the queries. A host grid is rejected by the loop seam,
# which knows the family.
function FastInterpolations._resolve_search(grid, q::AnyGPUArray, search, ::Nothing)
    policy = _on_device(grid) ? FastInterpolations._resolve_search_policy(grid, q, search, nothing) : BinarySearch()
    return FastInterpolations._to_searcher(policy, nothing)
end
FastInterpolations._resolve_search(_, ::AnyGPUArray, _, ::Base.RefValue{Int}) = _throw_gpu_hint()

# ── 1D batch loop seam ──
# Persistent batch calls reach the family-agnostic loop entry with the planned extrap and
# searcher. GPU output with host queries would run the host loop into device storage; it is
# rejected at the same seam (the third method resolves the GPU/GPU overlap).
FastInterpolations._itp_batch_loop!(output, itp::AbstractInterpolant1D, xq::AnyGPUArray, extrap, op, searcher) =
    _gpu_batch_loop!(output, itp, xq, extrap, op, searcher)
FastInterpolations._itp_batch_loop!(::AnyGPUArray, ::AbstractInterpolant1D, ::AbstractArray, _, _, _) = _throw_mixed_storage()
FastInterpolations._itp_batch_loop!(output::AnyGPUArray, itp::AbstractInterpolant1D, xq::AnyGPUArray, extrap, op, searcher) =
    _gpu_batch_loop!(output, itp, xq, extrap, op, searcher)

function _gpu_batch_loop!(output, itp, xq, extrap, op, searcher)
    _check_gpu_batch(itp)
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    isempty(xq) && return output
    extrap_eff = _gpu_domain_check(_itp_grid(itp), xq, extrap)
    # Broadcast, not `map!`: the backend's native broadcast kernel is leaner than the
    # generic KernelAbstractions kernel behind GPUArrays' `map!`.
    output .= _DeviceEval{typeof(searcher)}(itp, extrap_eff, op).(xq)
    return output
end

# The family's scalar evaluation, per thread, with the planned searcher started fresh.
struct _DeviceEval{S, I, E, O}
    itp::I
    extrap::E
    op::O
end
_DeviceEval{S}(itp, extrap, op) where {S} = _DeviceEval{S, typeof(itp), typeof(extrap), typeof(op)}(itp, extrap, op)
@inline (f::_DeviceEval{S})(q) where {S} = _itp_eval_scalar(f.itp, q, f.extrap, f.op, _fresh_searcher(S))
Adapt.adapt_structure(to, f::_DeviceEval{S}) where {S} = _DeviceEval{S}(Adapt.adapt(to, f.itp), f.extrap, f.op)

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

# ── ND batch loop seam ──
# SoA queries (a tuple of GPU coordinate arrays) and AoS queries (a GPU array of points:
# N-tuples or static vectors) reach the point-wise batch core every ND family shares, with
# the per-axis ops and the override-resolved extraps. As in 1D: device-side NoExtrap check,
# then one broadcast of the family's scalar ND evaluation.
const _GPUPoints = AnyGPUArray{<:Union{Tuple, AbstractVector}}
const _GPUQueriesND = Union{Tuple{AnyGPUArray, Vararg{AnyGPUArray}}, _GPUPoints}

FastInterpolations._nd_batch_pointwise!(
    output::AbstractArray, itp::AbstractInterpolantND{Tg, Tv, N}, queries::_GPUQueriesND,
    ops::Tuple{Vararg{AbstractEvalOp, N}}, extraps, search, hint
) where {Tg, Tv, N} = _gpu_nd_batch!(output, itp, queries, ops, extraps, search, hint)
FastInterpolations._nd_batch_pointwise!(
    ::AnyGPUArray, ::AbstractInterpolantND{Tg, Tv, N}, _, ::Tuple{Vararg{AbstractEvalOp, N}}, _, _, _
) where {Tg, Tv, N} = _throw_mixed_storage()
FastInterpolations._nd_batch_pointwise!(
    output::AnyGPUArray, itp::AbstractInterpolantND{Tg, Tv, N}, queries::_GPUQueriesND,
    ops::Tuple{Vararg{AbstractEvalOp, N}}, extraps, search, hint
) where {Tg, Tv, N} = _gpu_nd_batch!(output, itp, queries, ops, extraps, search, hint)

function _gpu_nd_batch!(output, itp, queries, ops, extraps, search, hint)
    _check_gpu_batch(itp)
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    hint === nothing || _throw_gpu_hint()
    _query_validate(queries)
    _query_length(queries) == 0 && return output
    f = _DeviceEvalND(itp, ops, _gpu_domain_check_nd(itp.grids, queries, extraps), search)
    if queries isa Tuple
        output .= f.(tuple.(queries...))
    else
        output .= f.(queries)
    end
    return output
end

# Per-axis call-time overrides for the kernel: `InBounds()` on NoExtrap axes once the device
# check passed and on call-time `InBounds` axes, `nothing` (stored extrap) elsewhere.
@inline _kernel_override(::Union{NoExtrap, InBounds}) = InBounds()
@inline _kernel_override(::AbstractExtrap) = nothing

# SoA vectors: per-axis checks, as the CPU's per-axis batch validation (its error names the
# axis and that axis's batch extreme).
function _gpu_domain_check_nd(grids::NTuple{N, Any}, queries::Tuple{AnyGPUVector, Vararg{AnyGPUVector}}, extraps) where {N}
    map(_check_axis_domain, grids, queries, extraps, ntuple(identity, Val(N)))
    return map(_kernel_override, extraps)
end
_check_axis_domain(_, _, ::AbstractExtrap, _) = nothing
function _check_axis_domain(g, q, ::NoExtrap, d)
    _axis_in_domain_all(g, q, NoExtrap()) || _throw_batch_oob(_host_endpoints(g), q, d)
    return nothing
end
_axis_in_domain_all(g, q, ::NoExtrap) = mapreduce(v -> _in_domain(g, v), &, q)
_axis_in_domain_all(_, _, ::AbstractExtrap) = true

# Shaped SoA and AoS: the CPU validates point by point, so a failure names the first
# out-of-domain point. Shaped SoA is checked one axis at a time (every axis in domain ⇔ every
# point in domain); AoS by one reduction over the points.
function _gpu_domain_check_nd(grids::NTuple{N, Any}, queries, extraps) where {N}
    any(e -> e isa NoExtrap, extraps) &&
        (_points_in_domain(grids, queries, extraps) || _throw_point_oob(grids, queries, extraps))
    return map(_kernel_override, extraps)
end
_points_in_domain(grids, queries::Tuple, extraps) = all(map(_axis_in_domain_all, grids, queries, extraps))
_points_in_domain(grids, queries, extraps) = mapreduce(q -> _point_in_domain(grids, q, extraps), &, queries)

@inline _point_in_domain(grids::NTuple{N, Any}, q, extraps) where {N} =
    reduce(&, map((g, e, d) -> _axis_in_domain(g, @inbounds(q[d]), e), grids, extraps, ntuple(identity, Val(N))))
@inline _axis_in_domain(g, v, ::NoExtrap) = _in_domain(g, v)
@inline _axis_in_domain(_, _, ::AbstractExtrap) = true

# Failure path only: the first out-of-domain point (CPU order: point-major in linear index
# order, then axis) is copied to the host and re-checked by the CPU code, so the error
# matches the CPU's.
@noinline function _throw_point_oob(grids::NTuple{N, Any}, queries, extraps) where {N}
    points = _points(queries)
    i = findfirst(q -> !_point_in_domain(grids, q, extraps), points)
    q = _host_element(points, i)
    map(grids, extraps, ntuple(identity, Val(N))) do g, e, d
        e isa NoExtrap && _check_domain(_host_endpoints(g), q[d], NoExtrap(), d)
        nothing
    end
    return nothing
end
# The points as one device vector in linear index order (SoA: materialized, failure path only).
_points(queries::Tuple) = tuple.(map(vec, queries)...)
_points(queries) = vec(queries)

# The family's scalar ND evaluation, per thread: per-axis ops (resolved `DerivOp`s pass
# through) and per-axis extrap overrides.
struct _DeviceEvalND{N, I, O, E, P}
    itp::I
    ops::O
    extrap::E
    search::P
end
_DeviceEvalND(itp::AbstractInterpolantND{Tg, Tv, N}, ops, extrap, search) where {Tg, Tv, N} =
    _DeviceEvalND{N, typeof(itp), typeof(ops), typeof(extrap), typeof(search)}(itp, ops, extrap, search)

@inline (f::_DeviceEvalND{N})(q) where {N} =
    _eval_nd_scalar_query(f.itp, ntuple(d -> @inbounds(q[d]), Val(N)), f.ops, f.extrap, f.search, nothing)

Adapt.adapt_structure(to, f::_DeviceEvalND) = _DeviceEvalND(Adapt.adapt(to, f.itp), f.ops, f.extrap, f.search)

end # module
