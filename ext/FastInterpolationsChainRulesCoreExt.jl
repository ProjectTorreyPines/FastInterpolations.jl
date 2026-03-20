# ========================================
# ChainRulesCore Extension for FastInterpolations.jl
# ========================================
# Provides analytical differentiation rules (frule/rrule) for:
#   - All 1D interpolants (CubicInterpolant, LinearInterpolant, etc.)
#   - All ND interpolants (CubicInterpolantND, LinearInterpolantND, etc.)
#
# These rules use built-in analytical derivatives (DerivOp) instead of
# propagating Dual numbers, yielding ~10-100x speedup.
#
# Usage:
#   using FastInterpolations, ChainRulesCore

module FastInterpolationsChainRulesCoreExt

using FastInterpolations
using ChainRulesCore

# ════════════════════════════════════════
# 1D Interpolants (scalar query → scalar output)
# ════════════════════════════════════════
# Excludes AbstractSeriesInterpolant (Vector output needs different pullback)
# and AbstractInterpolantND (dispatch on Real query already excludes them).

"""
Forward-mode rule for 1D scalar interpolants.

"""
function ChainRulesCore.frule(
        (_, Δx),
        itp::FastInterpolations.AbstractInterpolant{Tg, Tv},
        x::Real
    ) where {Tg, Tv}
    # Series interpolants have Vector output — skip (frule would work but
    # keep consistent with rrule exclusion)
    itp isa FastInterpolations.AbstractSeriesInterpolant && return nothing

    y = itp(x)
    Δx isa AbstractZero && return y, zero(y)
    ∂y = Δx * itp(x; deriv = DerivOp(1))
    return y, ∂y
end

"""
Reverse-mode rule for 1D scalar interpolants.

Enables `Zygote.gradient(itp, x)` to use analytical derivatives.
"""
function ChainRulesCore.rrule(
        itp::FastInterpolations.AbstractInterpolant{Tg, Tv},
        x::Real
    ) where {Tg, Tv}
    itp isa FastInterpolations.AbstractSeriesInterpolant && return nothing

    y = itp(x)

    function itp_1d_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent()
        # real(conj(Δy) * f'(x)) is the correct pullback for f: ℝ → ℂ
        # For all-real case this is a no-op: real(conj(r) * r') = r * r'
        return NoTangent(), real(conj(Δy) * itp(x; deriv = DerivOp(1)))
    end

    return y, itp_1d_pullback
end

# ════════════════════════════════════════
# ND Interpolants — Tuple query
# ════════════════════════════════════════
# All ND rules use FastInterpolations.gradient() for locate-once optimization:
# interval search is performed ONCE, then _eval_at_cell is called N times.

"""
Forward-mode rule for all ND interpolants with Tuple input.

"""
function ChainRulesCore.frule(
        (_, Δquery),
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}}
    ) where {Tg, Tv, N}
    y = itp(query)
    Δquery isa AbstractZero && return y, zero(y)

    # Locate once via gradient(), then dot product for directional derivative
    grad = FastInterpolations.gradient(itp, query)
    ∂y = sum(Δquery[i] * grad[i] for i in 1:N)
    return y, ∂y
end

"""
Forward-mode rule for all ND interpolants with Vector input.

"""
function ChainRulesCore.frule(
        (_, Δquery),
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real}
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "query length $(length(query)) does not match interpolant dimension $N"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    y = itp(query_tuple)
    Δquery isa AbstractZero && return y, zero(y)

    grad = FastInterpolations.gradient(itp, query_tuple)
    ∂y = sum(Δquery[i] * grad[i] for i in 1:N)
    return y, ∂y
end

# ════════════════════════════════════════
# ND Interpolants — Reverse-mode (rrule)
# ════════════════════════════════════════

"""
Reverse-mode rule for all ND interpolants with Tuple input.

Enables `Zygote.gradient(x -> itp((x[1], x[2])), ...)` to use
analytical derivatives via pullback.
"""
function ChainRulesCore.rrule(
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::AbstractVector{<:Real}
    ) where {Tg, Tv, N}
    length(query) == N || throw(
        DimensionMismatch(
            "query length $(length(query)) does not match interpolant dimension $N"
        )
    )
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    y = itp(query_tuple)

    function itp_nd_vec_pb(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent()
        grad = FastInterpolations.gradient(itp, query_tuple)
        ∂query = [real(conj(Δy) * grad[i]) for i in 1:N]
        return NoTangent(), ∂query
    end

    return y, itp_nd_vec_pb
end

# ════════════════════════════════════════
# 1D one-shot — data adjoint (∂/∂f) + query gradient (∂/∂xq)
# ════════════════════════════════════════
# Each interpolant type has ONE rrule covering both scalar and vec xq via
# Union{Real, AbstractVector}. The adjoint functions already have scalar
# convenience overloads (x_query::Real → [x_query] internally), so no
# Tg-typed wrapping is needed here.
#
# _adj_pullback dispatches on Δu type: Number → wraps to [Δu] for the adj
# callable (which requires AbstractVector input); AbstractVector → passes through.
# Δu .* d works for both: scalar .* scalar = scalar, vector .* vector = vector.
#
# linear/quadratic/constant share one unified rule (_RealInterp1D Union).
# cubic_interp: kept separate (real/complex disambiguation via Tg constraint).
# ∂/∂xq: linear/quadratic/cubic (real f) → Δu .* d via deriv=DerivOp(1).
#         constant → ZeroTangent() (API convention: derivative = 0).
#         cubic complex f → NoTangent() (Wirtinger; Zygote handles via source tracing).

@inline _adj_pullback(adj, Δu::Number; kwargs...) = adj([Δu]; kwargs...)
@inline _adj_pullback(adj, Δu; kwargs...) = adj(Δu; kwargs...)

# Trait: func → adjoint constructor (compile-time dispatch, zero runtime cost)
_adjoint_func(::typeof(linear_interp)) = linear_adjoint
_adjoint_func(::typeof(quadratic_interp)) = quadratic_adjoint
_adjoint_func(::typeof(constant_interp)) = constant_adjoint
_adjoint_func(::typeof(cubic_interp)) = cubic_adjoint

# Trait: interpolant struct → adjoint constructor + kwargs
_adjoint_func_from_itp(::FastInterpolations.CubicInterpolantND) = cubic_adjoint
_adjoint_func_from_itp(::FastInterpolations.QuadraticInterpolantND) = quadratic_adjoint
_adjoint_func_from_itp(::FastInterpolations.LinearInterpolantND) = linear_adjoint
_adjoint_func_from_itp(::FastInterpolations.ConstantInterpolantND) = constant_adjoint

_adjoint_kwargs_from_itp(itp::FastInterpolations.CubicInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::FastInterpolations.QuadraticInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::FastInterpolations.LinearInterpolantND) = (extrap = itp.extraps,)
_adjoint_kwargs_from_itp(itp::FastInterpolations.ConstantInterpolantND) = (side = itp.sides, extrap = itp.extraps)

# ── All 1D one-shot interpolants — unified rule ────────────────────────────────────────────
# Single rule for all 4 interpolant types, any f element type Tv.
# Julia specializes per (func, Tv) pair at compile time — no runtime overhead.
#
# ∂/∂f:  via adjoint operator (works for any Tv)
# ∂/∂xq: real.(conj.(Δu) .* d) — Wirtinger formula, correct for Real and Complex Tv

const _InterpMethod = Union{
    typeof(linear_interp), typeof(quadratic_interp),
    typeof(constant_interp), typeof(cubic_interp),
}

# `deriv` extracted explicitly — must not leak into adjoint or derivative computation.
# When deriv is non-EvalValue (evaluating a derivative), query gradient requires
# higher-order derivatives — not supported; returns NoTangent() in that case.
function ChainRulesCore.rrule(
        func::_InterpMethod,
        x::AbstractVector,
        f::AbstractVector{Tv},
        xq::Union{Real, AbstractVector};
        deriv::DerivOp = EvalValue(),
        kwargs...
    ) where {Tv}
    y = func(x, f, xq; deriv = deriv, kwargs...)
    adj = _adjoint_func(func)(x, xq; deriv = deriv, kwargs...)
    eval_value = deriv isa DerivOp{0}
    d = eval_value ? func(x, f, xq; deriv = DerivOp(1), kwargs...) : nothing
    function _interp1d_pb(Δy)
        Δy isa AbstractZero && return (NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent())
        Δu = unthunk(Δy)
        ∂xq = eval_value ? real.(conj.(Δu) .* d) : NoTangent()
        return (NoTangent(), NoTangent(), _adj_pullback(adj, Δu; deriv = deriv, kwargs...), ∂xq)
    end
    return y, _interp1d_pb
end

# ════════════════════════════════════════
# All ND one-shot interpolants — unified rules
# ════════════════════════════════════════
# Two dispatches by query type:
#
# 1. Single-point queries::Tuple{Vararg{Real,N}}
#    ∂/∂data via adjoint + ∂/∂queries via value_gradient (1 cell search, N+1 values)
#
# 2. Batch queries (SoA, Vector{NTuple}, etc.)
#    ∂/∂data via adjoint only; queries → NoTangent() (batch query-grad not supported)

# Single-point: build itp once, value_gradient for efficient ∂/∂queries.
# `deriv` extracted explicitly — constructor does not accept it; adjoint needs it.
# ∂/∂queries only when deriv == EvalValue (higher-order query-grad not yet supported).
function ChainRulesCore.rrule(
        func::_InterpMethod,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries::Tuple{Vararg{Real, N}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        kwargs...
    ) where {Tv, N}
    eval_value = deriv isa DerivOp{0} || (deriv isa Tuple && all(d -> d isa DerivOp{0}, deriv))
    if eval_value
        itp = func(grids, data; kwargs...)
        y, ds = value_gradient(itp, queries)
    else
        y = func(grids, data, queries; deriv = deriv, kwargs...)
        ds = nothing
    end
    adj = _adjoint_func(func)(grids, queries; deriv = deriv, kwargs...)
    function _interp_nd_scalar_pb(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
        Δu = unthunk(Δy)
        f_bar = adj(Δu; deriv = deriv, kwargs...)
        ∂queries = eval_value ? ntuple(i -> real(conj(Δu) * ds[i]), Val(N)) : NoTangent()
        return NoTangent(), NoTangent(), f_bar, ∂queries
    end
    return y, _interp_nd_scalar_pb
end

# Batch queries: ∂/∂data only
function ChainRulesCore.rrule(
        func::_InterpMethod,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries;
        kwargs...
    ) where {Tv, N}
    y = func(grids, data, queries; kwargs...)
    adj = _adjoint_func(func)(grids, queries; kwargs...)
    function _interp_nd_batch_pb(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
        f_bar = adj(unthunk(Δy); kwargs...)
        return NoTangent(), NoTangent(), f_bar, NoTangent()
    end
    return y, _interp_nd_batch_pb
end

# ════════════════════════════════════════
# CubicInterpolantND — constructor rrule (∂/∂data via interpolant API)
# ════════════════════════════════════════
# Enables the natural API pattern:
#   itp = cubic_interp((x, y), data)
#   loss = f(itp(x0))
#   Zygote.gradient(data -> f(cubic_interp((x,y), data)(x0)), data)
#
# The constructor pullback simply passes through the incoming tangent
# (computed by the eval/gradient/hessian/laplacian rrules below) as Δdata.
# This works because ChainRulesCore allows non-structural tangents.

"""
Constructor rrule for `cubic_interp(grids, data; ...)` → `CubicInterpolantND`.

The pullback receives the tangent accumulated from downstream eval/gradient/hessian
rrules (an Array of same shape as `data`) and passes it through as `Δdata`.
"""
function ChainRulesCore.rrule(
        func::_InterpMethod,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N};
        kwargs...
    ) where {Tv, N}
    itp = func(grids, data; kwargs...)

    function _interp_nd_ctor_pb(Δitp)
        Δitp isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent()
        return NoTangent(), NoTangent(), Δitp
    end

    return itp, _interp_nd_ctor_pb
end

# ════════════════════════════════════════
# CubicInterpolantND — eval rrule with ∂/∂data
# ════════════════════════════════════════
# More specific than the generic AbstractInterpolantND rrule (lines 140-155),
# so Julia dispatches here for CubicInterpolantND.
# Returns both ∂/∂query AND ∂/∂data (as the itp tangent).

"""
Eval rrule for `itp::AbstractInterpolantND(query)` with ∂/∂data support.

The pullback computes:
- `∂query`: via `gradient(itp, query)` (locate-once, analytical)
- `data_bar`: via adjoint operator — returned as the itp tangent,
  which flows back through the constructor rrule to become `Δdata`.
"""
function ChainRulesCore.rrule(
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}}
    ) where {Tg, Tv, N}
    y = itp(query)

    adj_fn = _adjoint_func_from_itp(itp)
    adj = adj_fn(itp.grids, (query,); _adjoint_kwargs_from_itp(itp)...)

    function _itp_nd_eval_pb(Δy)
        Δy isa AbstractZero && return ZeroTangent(), ZeroTangent()
        Δy_val = unthunk(Δy)

        grad = FastInterpolations.gradient(itp, query)
        ∂query = ntuple(i -> real(conj(Δy_val) * grad[i]), Val(N))

        data_bar = adj(Δy_val; deriv = EvalValue())

        return data_bar, ∂query
    end

    return y, _itp_nd_eval_pb
end

# ════════════════════════════════════════
# AbstractInterpolantND — gradient rrule with ∂/∂data
# ════════════════════════════════════════

function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.gradient),
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    grad = FastInterpolations.gradient(itp, query; kwargs...)

    adj_fn = _adjoint_func_from_itp(itp)
    adj = adj_fn(itp.grids, (query,); _adjoint_kwargs_from_itp(itp)...)

    function _gradient_itp_nd_pb(Δgrad_raw)
        Δgrad_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        Δgrad = unthunk(Δgrad_raw)

        T_out = promote_type(Tg, typeof(first(Δgrad)))
        data_bar = zeros(T_out, size(itp)...)
        for i in 1:N
            dg_i = Δgrad[i]
            iszero(dg_i) && continue
            ops_i = ntuple(j -> j == i ? DerivOp(1) : EvalValue(), Val(N))
            data_bar .+= adj(dg_i; deriv = ops_i)
        end

        H = FastInterpolations.hessian(itp, query)
        ∂query = ntuple(j -> sum(real(conj(Δgrad[i]) * H[i, j]) for i in 1:N), Val(N))

        return NoTangent(), data_bar, ∂query
    end

    return grad, _gradient_itp_nd_pb
end

# ════════════════════════════════════════
# AbstractInterpolantND — hessian rrule with ∂/∂data
# ════════════════════════════════════════

function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.hessian),
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    H = FastInterpolations.hessian(itp, query; kwargs...)

    adj_fn = _adjoint_func_from_itp(itp)
    adj = adj_fn(itp.grids, (query,); _adjoint_kwargs_from_itp(itp)...)

    function _hessian_itp_nd_pb(ΔH_raw)
        ΔH_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        ΔH = unthunk(ΔH_raw)

        T_out = promote_type(Tg, eltype(ΔH))
        data_bar = zeros(T_out, size(itp)...)

        for i in 1:N
            dh_ii = ΔH[i, i]
            iszero(dh_ii) && continue
            ops = ntuple(j -> j == i ? DerivOp(2) : EvalValue(), Val(N))
            data_bar .+= adj(dh_ii; deriv = ops)
        end

        for i in 1:N, j in (i + 1):N
            dh_ij = ΔH[i, j] + ΔH[j, i]
            iszero(dh_ij) && continue
            ops = ntuple(k -> (k == i || k == j) ? DerivOp(1) : EvalValue(), Val(N))
            data_bar .+= adj(dh_ij; deriv = ops)
        end

        return NoTangent(), data_bar, ZeroTangent()
    end

    return H, _hessian_itp_nd_pb
end

# ════════════════════════════════════════
# AbstractInterpolantND — laplacian rrule with ∂/∂data
# ════════════════════════════════════════

function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.laplacian),
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    lap = FastInterpolations.laplacian(itp, query; kwargs...)

    adj_fn = _adjoint_func_from_itp(itp)
    adj = adj_fn(itp.grids, (query,); _adjoint_kwargs_from_itp(itp)...)

    function _laplacian_itp_nd_pb(Δlap_raw)
        Δlap_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        Δlap = unthunk(Δlap_raw)

        T_out = promote_type(Tg, typeof(Δlap))
        data_bar = zeros(T_out, size(itp)...)
        for i in 1:N
            ops = ntuple(j -> j == i ? DerivOp(2) : EvalValue(), Val(N))
            data_bar .+= adj(Δlap; deriv = ops)
        end

        return NoTangent(), data_bar, ZeroTangent()
    end

    return lap, _laplacian_itp_nd_pb
end

# ════════════════════════════════════════
# AbstractInterpolantND — value_gradient rrule with ∂/∂data
# ════════════════════════════════════════

function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.value_gradient),
        itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    val, grad = FastInterpolations.value_gradient(itp, query; kwargs...)

    adj_fn = _adjoint_func_from_itp(itp)
    adj = adj_fn(itp.grids, (query,); _adjoint_kwargs_from_itp(itp)...)

    function _value_gradient_itp_nd_pb(Δ_raw)
        Δ_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        Δ = unthunk(Δ_raw)
        Δval, Δgrad = Δ[1], Δ[2]

        T_grad = Δgrad isa AbstractZero ? Tg : promote_type(map(typeof, Δgrad)...)
        T_out = promote_type(Tg, Tv, typeof(Δval), T_grad)
        data_bar = zeros(T_out, size(itp)...)
        ∂query_val = ntuple(_ -> zero(Tg), Val(N))
        ∂query_grad = ntuple(_ -> zero(Tg), Val(N))

        if !(Δval isa AbstractZero) && !iszero(Δval)
            data_bar .+= adj(Δval; deriv = EvalValue())
            ∂query_val = ntuple(i -> real(conj(Δval) * grad[i]), Val(N))
        end

        if !(Δgrad isa AbstractZero)
            for i in 1:N
                dg_i = Δgrad[i]
                iszero(dg_i) && continue
                ops_i = ntuple(j -> j == i ? DerivOp(1) : EvalValue(), Val(N))
                data_bar .+= adj(dg_i; deriv = ops_i)
            end
            H = FastInterpolations.hessian(itp, query)
            ∂query_grad = ntuple(j -> sum(real(conj(Δgrad[i]) * H[i, j]) for i in 1:N), Val(N))
        end

        ∂query = ntuple(i -> ∂query_val[i] + ∂query_grad[i], Val(N))
        return NoTangent(), data_bar, ∂query
    end

    return (val, grad), _value_gradient_itp_nd_pb
end

end # module
