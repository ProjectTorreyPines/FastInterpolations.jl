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
    StorePolicy
import Adapt
using GPUArrays: AbstractGPUVector, AnyGPUArray, AnyGPUVector

FastInterpolations._is_device_storage(::AnyGPUArray) = true

# ── GPU-native construction ──
# Device inputs reach the shared normalization seams; these keep them on the device as
# `_DeviceArray` leaves — eltype conversion broadcast on the device, ownership per
# StorePolicy (copy by default, alias with `copy = false`), never a host `Vector`. A uniform
# grid still becomes a `_CachedRange`; periodic boundary conditions are not supported yet.
FastInterpolations._policy_axis(x::AnyGPUVector, ::NoBC, ::Type{Tg}, ::StorePolicy{CG, CV, true}) where {Tg, CG, CV} =
    _DeviceArray(_device_convert(x, Tg))
FastInterpolations._policy_axis(x::AnyGPUVector, ::NoBC, ::Type{Tg}, ::StorePolicy{CG, CV, false}) where {Tg, CG, CV} =
    _DeviceArray(_device_convert(x, Tg))

FastInterpolations._own_or_ref_values(y::AnyGPUVector, ::Type{Tv}, ::StorePolicy{CG, true, CA}) where {Tv, CG, CA} =
    _DeviceArray(_device_copy(y, Tv))
FastInterpolations._own_or_ref_values(y::AnyGPUVector{Tv}, ::Type{Tv}, ::StorePolicy{CG, false, CA}) where {Tv, CG, CA} =
    _DeviceArray(y)
FastInterpolations._own_or_ref_values(y::AnyGPUVector, ::Type{Tv}, ::StorePolicy{CG, false, CA}) where {Tv, CG, CA} =
    _DeviceArray(Tv.(y))

_device_convert(a, ::Type{T}) where {T} = eltype(a) === T ? a : T.(a)
_device_copy(a, ::Type{T}) where {T} = eltype(a) === T ? copy(a) : T.(a)

# ── Admission ──
# Admitted families list the arrays their evaluation reads; any other family is rejected
# before allocation or launch.
_gpu_batch_storage(itp::LinearInterpolant) = (itp.x, itp.y)
_gpu_batch_storage(itp::AbstractInterpolant1D) =
    throw(ArgumentError("GPU batch evaluation is not supported for $(nameof(typeof(itp))) yet"))

# Usable inside a kernel: device storage, or plain isbits data (e.g. `_CachedRange`).
_on_device(a) = a isa _DeviceArray || FastInterpolations._is_device_storage(a) || isbits(a)

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
# On a device a linear one-shot call is a build that aliases its inputs (no copy, no spacing
# caches: nothing is precomputed) followed by the batch runner above — the same device
# paths at no extra cost. Host data with GPU queries is rejected by the runner's admission.
function FastInterpolations.linear_interp(
        x::AbstractVector,
        y::AbstractVector,
        xq::AnyGPUVector{Tq};
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tq <: Number}
    return _oneshot_itp(x, y, bc, extrap, search)(xq; deriv, hint)
end

function FastInterpolations.linear_interp!(
        output::AbstractVector,
        x::AbstractVector,
        y::AbstractVector,
        xq::AnyGPUVector{Tq};
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tq <: Number}
    return _oneshot_itp(x, y, bc, extrap, search)(output, xq; deriv, hint)
end

function _oneshot_itp(x, y, bc, extrap, search)
    bc isa NoBC || throw(ArgumentError("boundary condition $(bc) is not supported for GPU array queries yet"))
    return linear_interp(x, y; extrap, search, store = StorePolicy(copy = false))
end

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
_host_endpoints(x::_DeviceVector) = [_host_element(parent(x), firstindex(x)), _host_element(parent(x), lastindex(x))]
_host_element(a, i) = only(Array(view(a, i:i)))

end # module
