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
# kernel, and the loop is one broadcast of the family's point evaluation. A family is
# admitted by declaring its storage (`_storage` / `_rebuild`), once its point evaluation is
# verified to compile in GPU kernels.

module FastInterpolationsGPUArraysExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant1D, LinearInterpolant, AbstractExtrap,
    _DeviceArray, _DeviceVector, _itp_grid, _itp_eval_scalar, _domain_bounds, _extract_primal, _ge, _le, _throw_batch_oob,
    StorePolicy, PeriodicBC, periodic_check, _PlainAxisBC, _resolve_exclusive_period,
    _throw_excl_endpoint_too_small, AbstractInterpolantND,
    _eval_nd_scalar_query, _check_domain, _query_validate, _query_length,
    _ExclusivePeriodicAxis, _ExclusivePeriodicData, _RawFields, _resolve_bc_period,
    _throw_excl_axis_period_too_small, _storage, _device_eltype, Searcher, NoHint, RefHint,
    AbstractEvalOp, DerivativeView
import Adapt
using Base.Broadcast: Broadcasted
using GPUArrays: AnyGPUArray, AnyGPUVector, AbstractGPUArrayStyle

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
# A family that declares its storage is admitted; the grids and values its evaluation reads
# must be on the device. Any other family is rejected at its loop seam, before any query is
# read or any kernel launched.
_check_gpu_batch(itp) = _check_gpu_storage(itp, _storage(itp))
_check_gpu_storage(itp, ::Nothing) =
    throw(ArgumentError("GPU batch evaluation is not supported for $(nameof(typeof(itp))) yet"))
function _check_gpu_storage(_, s::NamedTuple)
    (all(_on_device, s.grids) && all(_on_device, s.values)) || throw(
        ArgumentError(
            "GPU batch evaluation needs the grid and the data on the device: build from device arrays, or move the interpolant first, e.g. `adapt(CuArray, itp)`"
        )
    )
    return nothing
end

# Usable inside a kernel: device storage, or plain isbits data (e.g. `_CachedRange`).
_on_device(a) = FastInterpolations._is_device_storage(a) || isbits(a)
_on_device(g::Union{_ExclusivePeriodicAxis, _ExclusivePeriodicData}) = _on_device(g.inner)

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
    _gpu_eval!(output, itp, xq, extrap, op, searcher)
    return output
end

# Broadcast, not `map!`: the backend's native broadcast kernel is leaner than the generic
# KernelAbstractions kernel behind GPUArrays' `map!`.
_gpu_eval!(output, itp, xq, extrap, op, ::S) where {S} = (output .= _DeviceEval{S}(itp, extrap, op).(xq))

# NoExtrap: the evaluation kernel checks the domain itself (see "Domain check" below); the
# host reads the flag once afterwards and throws the CPU's error if it is raised.
function _gpu_eval!(output, itp, xq, ::NoExtrap, op, ::S) where {S}
    _noextrap_notice()
    flag = _device_flag(xq)
    output .= _CheckedEval{S}(itp, op, flag).(xq)
    _flag_raised(flag) && _throw_gpu_batch_oob(_itp_grid(itp), xq)
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

# NoExtrap's evaluation, per thread: an in-domain query as with `InBounds`, any other one
# raises the flag (see `_guarded`).
struct _CheckedEval{S, I, O, F}
    itp::I
    op::O
    flag::F
end
_CheckedEval{S}(itp, op, flag) where {S} = _CheckedEval{S, typeof(itp), typeof(op), typeof(flag)}(itp, op, flag)
@inline function (f::_CheckedEval{S})(q) where {S}
    ok, qe = _guarded(_itp_grid(f.itp), q, NoExtrap())
    ok || (@inbounds f.flag[1] = true)
    return _itp_eval_scalar(f.itp, qe, InBounds(), f.op, _fresh_searcher(S))
end
Adapt.adapt_structure(to, f::_CheckedEval{S}) where {S} = _CheckedEval{S}(Adapt.adapt(to, f.itp), f.op, Adapt.adapt(to, f.flag))

# ── Broadcasts ──
# A broadcast over device arrays evaluates the interpolant inside a kernel, where NoExtrap's
# out-of-domain error cannot be built. A NoExtrap interpolant broadcast over device queries
# therefore takes the batch call (device domain check, the CPU's DomainError): the same
# values, materialized instead of fused into the surrounding broadcast. ND SoA broadcasts
# `f.(xs, ys)` over equal-shaped device arrays are the SoA batch call.
const _Evaluable = Union{AbstractInterpolant1D, AbstractInterpolantND, DerivativeView}

function Base.Broadcast.broadcasted(style::AbstractGPUArrayStyle, f::_Evaluable, q::AnyGPUArray, qs::AnyGPUArray...)
    batch = _raises_in_kernel(f) ? _batch_form(f, q, qs) : nothing
    return batch === nothing ? Broadcasted{typeof(style)}(f, (q, qs...)) : f(batch)
end

_raises_in_kernel(d::DerivativeView) = _raises_in_kernel(d.parent)
_raises_in_kernel(itp) = _any_noextrap(_storage(itp))
_any_noextrap(::Nothing) = false
_any_noextrap(s::NamedTuple) = any(e -> e isa NoExtrap, s.extraps)

# The batch query for a broadcast's arguments (`nothing`: not a batch call).
_batch_form(_, q, ::Tuple{}) = q
_batch_form(::AbstractInterpolantND, q, qs::Tuple{Any, Vararg}) = all(a -> axes(a) == axes(q), qs) ? (q, qs...) : nothing
_batch_form(_, _, ::Tuple{Any, Vararg}) = nothing

# ── One-shot ──
# One-shot calls on GPU queries take the core one-shot route (`linear_interp` /
# `linear_interp!` are the core methods): the storage leaves and the planning/execution
# seams below it are the only GPU methods. Nothing is copied unless an eltype conversion
# asks for it.
FastInterpolations._resolve_axis(x::AnyGPUVector, ::_PlainAxisBC, ::Type{Tg}) where {Tg} = _DeviceArray(_device_convert(x, Tg))

# `:exclusive`: the shared lazy wrapper around a device leaf; its seam point `x[1] + period`
# comes from the two endpoints, read once on the host.
function FastInterpolations._resolve_axis(x::AnyGPUVector, bc::PeriodicBC{:exclusive}, ::Type{Tg}) where {Tg}
    inner = _DeviceArray(_device_convert(x, Tg))
    period = _resolve_bc_period(inner, bc).period
    Te = promote_type(Tg, typeof(period))
    inner_e = _device_eltype(inner, Te)
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

# The resolved storage, held by the raw interpolant (nothing is copied or checked), runs
# the persistent batch loop: one-shot and persistent calls share the GPU evaluation.
FastInterpolations._linear_interp_loop!(
    output::AbstractArray, x::AbstractVector, y::AbstractVector, x_targets::AnyGPUArray,
    extrap::AbstractExtrap, op::AbstractEvalOp, searcher::Searcher
) = _gpu_batch_loop!(output, LinearInterpolant(_RawFields(), x, y, extrap, _policy(searcher)), x_targets, extrap, op, searcher)
@inline _policy(::Searcher{P}) where {P} = P()      # search policies are singletons

# A searcher carries no shared state across threads: each one starts fresh (a hint-carrying
# policy keeps a thread-local hint).
@inline _fresh_searcher(::Type{Searcher{P, NoHint}}) where {P} = Searcher{P, NoHint}(NoHint())
@inline _fresh_searcher(::Type{Searcher{P, RefHint}}) where {P} = Searcher{P, RefHint}(RefHint())

# ── Domain check (NoExtrap) ──
# The evaluation kernel checks the domain as it goes: an out-of-domain (or NaN) query raises
# a one-element device flag and is evaluated at the domain's lower end instead, in bounds
# (its value is discarded). The host reads the flag once, after the kernel; a raised flag
# throws the CPU's `DomainError` (same bounds, same message), computed on this failure path
# only. Unlike on the CPU, the output may then hold partial results. The bounds come from
# `_domain_bounds`, the code every axis type uses on the CPU. Other extraps need no check:
# the per-element boundary branch they take is nearly free in a kernel.
@inline function _guarded(x, q, ::NoExtrap)
    ok = _in_domain(x, q)
    return ok, ifelse(ok, q, convert(typeof(q), first(_domain_bounds(x))))
end
@inline _guarded(_, q, ::AbstractExtrap) = (true, q)

_device_flag(like) = fill!(similar(like, Bool, 1), false)
_flag_raised(flag) = _host_element(flag, 1)

@noinline _noextrap_notice() = @warn """
NoExtrap() on GPU arrays (the default extrapolation): each batch call waits for the device once \
to check the domain, and an out-of-domain query throws after the kernel has run, so an in-place \
output may hold partial results. For queries known to lie in the domain, `extrap = InBounds()` \
skips the check and the wait; `ClampExtrap()` and `FillExtrap(v)` never throw. Inside your own \
kernels, NoExtrap does not compile.""" maxlog = 1

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
# the per-axis ops and the override-resolved extraps. As in 1D: one broadcast of the family's
# scalar ND evaluation, NoExtrap axes checked in the kernel.
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
    overrides = map(_kernel_override, extraps)
    if any(e -> e isa NoExtrap, extraps)
        _noextrap_notice()
        flag = _device_flag(queries isa Tuple ? first(queries) : queries)
        _nd_broadcast!(output, _CheckedEvalND(itp, ops, overrides, extraps, search, flag), queries)
        _flag_raised(flag) && _throw_nd_oob(itp.grids, queries, extraps)
    else
        _nd_broadcast!(output, _DeviceEvalND(itp, ops, overrides, search), queries)
    end
    return output
end
_nd_broadcast!(output, f, queries::Tuple) = (output .= f.(tuple.(queries...)))
_nd_broadcast!(output, f, queries) = (output .= f.(queries))

# Per-axis call-time overrides for the kernel: `InBounds()` on NoExtrap axes (checked in the
# kernel) and on call-time `InBounds` axes, `nothing` (stored extrap) elsewhere.
@inline _kernel_override(::Union{NoExtrap, InBounds}) = InBounds()
@inline _kernel_override(::AbstractExtrap) = nothing

# Failure path only (the kernel raised the flag): the CPU's error. SoA vectors: per axis, as
# the CPU's per-axis batch validation (its error names the axis and that axis's batch
# extreme).
@noinline function _throw_nd_oob(grids::NTuple{N, Any}, queries::Tuple{AnyGPUVector, Vararg{AnyGPUVector}}, extraps) where {N}
    map(_check_axis_domain, grids, queries, extraps, ntuple(identity, Val(N)))
    return nothing
end
_check_axis_domain(_, _, ::AbstractExtrap, _) = nothing
function _check_axis_domain(g, q, ::NoExtrap, d)
    mapreduce(v -> _in_domain(g, v), &, q) || _throw_batch_oob(_host_endpoints(g), q, d)
    return nothing
end

# Shaped SoA and AoS: the CPU validates point by point, so the error names the first
# out-of-domain point.
@noinline _throw_nd_oob(grids, queries, extraps) = _throw_point_oob(grids, queries, extraps)

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

# With NoExtrap axes: each one checked in the kernel (see `_guarded`), an out-of-domain
# coordinate raising the flag.
struct _CheckedEvalND{N, I, O, E, X, P, F}
    itp::I
    ops::O
    extrap::E
    extraps::X
    search::P
    flag::F
end
_CheckedEvalND(itp::AbstractInterpolantND{Tg, Tv, N}, ops, extrap, extraps, search, flag) where {Tg, Tv, N} =
    _CheckedEvalND{N, typeof(itp), typeof(ops), typeof(extrap), typeof(extraps), typeof(search), typeof(flag)}(
    itp, ops, extrap, extraps, search, flag
)

@inline function (f::_CheckedEvalND{N})(q) where {N}
    checked = map(_guarded, f.itp.grids, ntuple(d -> @inbounds(q[d]), Val(N)), f.extraps)
    all(first, checked) || (@inbounds f.flag[1] = true)
    return _eval_nd_scalar_query(f.itp, map(last, checked), f.ops, f.extrap, f.search, nothing)
end

Adapt.adapt_structure(to, f::_CheckedEvalND) =
    _CheckedEvalND(Adapt.adapt(to, f.itp), f.ops, f.extrap, f.extraps, f.search, Adapt.adapt(to, f.flag))

end # module
