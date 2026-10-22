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
# extrap override, the CPU's errors); the loop is the CPU loop's body broadcast over the
# queries, with NoExtrap's domain check folded into it. A family is admitted by declaring
# its storage (`_storage` / `_rebuild`), once its point evaluation is verified to compile in
# GPU kernels.

module FastInterpolationsGPUArraysExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant1D, LinearInterpolant, AbstractExtrap,
    _DeviceArray, _DeviceVector, _itp_grid, _itp_eval_scalar, _domain_bounds, _extract_primal, _ge, _le, _throw_batch_oob,
    StorePolicy, PeriodicBC, periodic_check, _PlainAxisBC, _resolve_exclusive_period,
    _throw_excl_endpoint_too_small, AbstractInterpolantND,
    _eval_nd_scalar_query, _check_domain, _query_validate, _query_length,
    _ExclusivePeriodicAxis, _ExclusivePeriodicData, _RawFields, _resolve_bc_period, _promote_coord,
    _throw_excl_axis_period_too_small, _storage, _device_eltype, Searcher, _resolve_search_nd,
    AbstractEvalOp, DerivativeView
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

# Search is planned without reading a query and without a hint: device threads evaluate
# queries independently, so no hint carries from one query to the next. A vector grid takes
# `BinarySearch` (AutoSearch's choice here); a policy built on a carried hint runs as
# `BinarySearch` too, with a one-time notice. A range grid takes the direct lookup whatever
# the policy, as on the CPU. A host grid's query-order probe would read the queries; a host
# grid is rejected by the loop seam, which knows the family.
function FastInterpolations._resolve_search(grid, q::AnyGPUArray, search, ::Nothing)
    policy = _on_device(grid) ? FastInterpolations._resolve_search_policy(grid, q, _device_policy(grid, search), nothing) : BinarySearch()
    return FastInterpolations._to_searcher(policy, nothing)
end
FastInterpolations._resolve_search(_, ::AnyGPUArray, _, ::Base.RefValue{Int}) = _throw_gpu_hint()

_device_policy(::AbstractRange, search) = search
_device_policy(_, search) = _hint_free(search)
_hint_free(::AutoSearch) = BinarySearch()
_hint_free(search::Union{LinearSearch, LinearBinarySearch, LinearGallopSearch}) = (_hint_notice(search); BinarySearch())
_hint_free(search::AbstractSearchPolicy) = search

@noinline _hint_notice(search) = @warn """
$search on GPU arrays: it carries a search hint from one query to the next, but device \
threads evaluate queries independently, so it runs as BinarySearch(). Omit `search`, or pass \
BinarySearch(), to silence this.""" maxlog = 1

# ── 1D batch loop seam ──
# Persistent batch calls reach the family-agnostic loop entry with the planned extrap and
# searcher. GPU output with host queries would run the host loop into device storage; it is
# rejected at the same seam (the third method resolves the GPU/GPU overlap).
FastInterpolations._itp_batch_loop!(output, itp::AbstractInterpolant1D, xq::AnyGPUArray, extrap, op, searcher) =
    _gpu_batch_loop!(output, itp, xq, extrap, op, searcher)
FastInterpolations._itp_batch_loop!(::AnyGPUArray, ::AbstractInterpolant1D, ::AbstractArray, _, _, _) = _throw_mixed_storage()
FastInterpolations._itp_batch_loop!(output::AnyGPUArray, itp::AbstractInterpolant1D, xq::AnyGPUArray, extrap, op, searcher) =
    _gpu_batch_loop!(output, itp, xq, extrap, op, searcher)

# The CPU loop's body (the family's point evaluation with the planned extrap and searcher),
# broadcast over the queries; NoExtrap's domain check rides along (see "Domain check").
# Broadcast, not `map!`: the backend's native broadcast kernel is leaner than the generic
# KernelAbstractions kernel behind GPUArrays' `map!`.
function _gpu_batch_loop!(output, itp, xq, extrap, op, searcher)
    _check_gpu_batch(itp)
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    isempty(xq) && return output
    flag = _domain_flag(extrap, xq)
    extrap_k = _kernel_extrap(extrap)
    qk = _kernel_queries(flag, output, xq)
    output .= (@inline q -> _itp_eval_scalar(itp, _admit(flag, _itp_grid(itp), q), extrap_k, op, searcher)).(qk)
    _raised(flag) && _throw_gpu_batch_oob(_itp_grid(itp), qk)
    return output
end

# ── Broadcasts ──
# A broadcast over device arrays evaluates the interpolant inside a kernel, where NoExtrap's
# out-of-domain error cannot be built. A NoExtrap interpolant (or a derivative view of one)
# broadcast over device data therefore takes the batch call (device domain check, the CPU's
# DomainError): the same values, materialized instead of fused into the surrounding
# broadcast. Its arguments become the batch query: a device array as it is (ND: equal-shaped
# coordinate arrays are the SoA batch); anything else, such as a nested broadcast, a scalar
# coordinate or a shape to broadcast, is materialized on the device first (ND: as points).
const _Evaluable = Union{AbstractInterpolant1D, AbstractInterpolantND, DerivativeView}

function Base.Broadcast.broadcasted(style::AbstractGPUArrayStyle, f::_Evaluable, q, qs...)
    itp = _underlying(f)
    batch = _raises_in_kernel(itp) ? _batch_form(itp, q, qs) : nothing
    return batch === nothing ? Broadcasted{typeof(style)}(f, (q, qs...)) : f(batch)
end

_underlying(d::DerivativeView) = _underlying(d.parent)
_underlying(itp) = itp
_raises_in_kernel(itp) = _any_noextrap(_storage(itp))
_any_noextrap(::Nothing) = false
_any_noextrap(s::NamedTuple) = any(e -> e isa NoExtrap, s.extraps)

# The batch query for a broadcast's arguments (`nothing`: not a batch call).
_batch_form(_, q, ::Tuple{}) = _device_query(q)
_batch_form(::AbstractInterpolantND, q, qs::Tuple{Any, Vararg}) = _soa_or_points(q, qs)
_batch_form(_, _, ::Tuple{Any, Vararg}) = nothing

_device_query(q::AnyGPUArray) = q
_device_query(q) = Base.materialize(Base.Broadcast.broadcasted(identity, q))
_soa_or_points(q, qs) = _is_soa(q, qs) ? (q, qs...) : Base.materialize(Base.Broadcast.broadcasted(tuple, q, qs...))
_is_soa(q::AnyGPUArray, qs) = all(a -> a isa AnyGPUArray && axes(a) == axes(q), qs)
_is_soa(_, _) = false

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

# ── Domain check (NoExtrap) ──
# The batch loop checks the domain as it evaluates: an out-of-domain (or NaN) query raises
# a one-element device flag and is evaluated at the domain's lower end instead, in bounds
# (its value is discarded), the evaluation itself taking `InBounds`. The host reads the flag
# once, after the kernel; a raised flag throws the CPU's `DomainError` (same bounds, same
# message), computed on this failure path only. Unlike on the CPU, the output may then hold
# partial results. The bounds come from `_domain_bounds`, the code every axis type uses on
# the CPU. Other extraps take no flag and no check: the per-element boundary branch they
# take is nearly free in a kernel.
_domain_flag(::NoExtrap, like) = (_noextrap_notice(); _device_flag(like))
_domain_flag(::AbstractExtrap, _) = nothing
_domain_flag(extraps::Tuple, like) = any(e -> e isa NoExtrap, extraps) ? _domain_flag(NoExtrap(), like) : nothing
_kernel_extrap(::NoExtrap) = InBounds()
_kernel_extrap(extrap::AbstractExtrap) = extrap

@inline _admit(::Nothing, _, q) = q
@inline function _admit(flag, x, q)
    ok, qe = _guarded(x, q, NoExtrap())
    ok || (@inbounds flag[1] = true)
    return qe
end

# ND: the point as an N-tuple, each NoExtrap axis checked.
@inline _admit_point(::Nothing, grids, _, q) = _point(grids, q)
@inline function _admit_point(flag, grids, extraps, q)
    checked = map(_guarded, grids, _point(grids, q), extraps)
    all(first, checked) || (@inbounds flag[1] = true)
    return map(last, checked)
end
@inline _point(::NTuple{N, Any}, q) where {N} = ntuple(d -> @inbounds(q[d]), Val(N))

# The query takes the grid's coordinate type first (an `Int` query on a `Float32` grid), so
# the substitute lower end is representable; the evaluation promotes it the same way.
@inline function _guarded(x, q, ::NoExtrap)
    qc = _promote_coord(q, eltype(x))
    ok = _in_domain(x, qc)
    return ok, ifelse(ok, qc, oftype(qc, first(_domain_bounds(x))))
end
@inline _guarded(_, q, ::AbstractExtrap) = (true, q)

_device_flag(like) = fill!(similar(like, Bool, 1), false)
_raised(::Nothing) = false
_raised(flag) = _host_element(flag, 1)

# The failure path rereads the queries after the kernel; an output sharing memory with them
# would have overwritten them by then, so the kernel reads a copy instead.
_kernel_queries(::Nothing, _, queries) = queries
_kernel_queries(_, output, queries) = _unaliased(output, queries)
_unaliased(output, queries::Tuple) = map(q -> _unaliased(output, q), queries)
_unaliased(output, q) = Base.mightalias(output, q) ? copy(q) : q

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
# the per-axis ops and the override-resolved extraps. As in 1D: the family's scalar ND
# evaluation broadcast over the points, per-axis policies planned without a hint, NoExtrap
# axes checked as they are evaluated.
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

function _gpu_nd_batch!(output, itp::AbstractInterpolantND{Tg, Tv, N}, queries, ops, extraps, search, hint) where {Tg, Tv, N}
    _check_gpu_batch(itp)
    FastInterpolations._is_device_storage(output) || _throw_mixed_storage()
    hint === nothing || _throw_gpu_hint()
    _query_validate(queries)
    _query_length(queries) == 0 && return output
    flag = _domain_flag(extraps, _flag_like(queries))
    overrides = map(_kernel_override, extraps)
    policies = map(_device_policy, itp.grids, _resolve_search_nd(search, Val(N)))
    qk = _kernel_queries(flag, output, queries)
    _nd_broadcast!(output, @inline(q -> _eval_nd_scalar_query(itp, _admit_point(flag, itp.grids, extraps, q), ops, overrides, policies, nothing)), qk)
    _raised(flag) && _throw_nd_oob(itp.grids, qk, extraps)
    return output
end
_nd_broadcast!(output, f, queries::Tuple) = (output .= f.(tuple.(queries...)))
_nd_broadcast!(output, f, queries) = (output .= f.(queries))
_flag_like(queries::Tuple) = first(queries)
_flag_like(queries) = queries

# Per-axis call-time overrides for the kernel: `InBounds()` on NoExtrap axes (checked in the
# kernel) and on call-time `InBounds` axes, `nothing` (stored extrap) elsewhere.
@inline _kernel_override(::Union{NoExtrap, InBounds}) = InBounds()
@inline _kernel_override(::AbstractExtrap) = nothing

# Failure path only (the kernel raised the flag): the CPU's error. SoA vectors: per axis, as
# the CPU's per-axis batch validation (its error names the axis and that axis's batch
# extreme).
@noinline _throw_nd_oob(grids::NTuple{N, Any}, queries::Tuple{AnyGPUVector, Vararg{AnyGPUVector}}, extraps) where {N} =
    foreach(_check_axis_domain, grids, queries, extraps, ntuple(identity, Val(N)))
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
    return foreach(grids, extraps, ntuple(identity, Val(N))) do g, e, d
        e isa NoExtrap && _check_domain(_host_endpoints(g), q[d], NoExtrap(), d)
    end
end
# The points as one device vector in linear index order (SoA: materialized, failure path only).
_points(queries::Tuple) = tuple.(map(vec, queries)...)
_points(queries) = vec(queries)

end # module
