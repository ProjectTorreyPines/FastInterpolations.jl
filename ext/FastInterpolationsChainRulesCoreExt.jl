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
# Adjoint OOB masking for FillExtrap
# ════════════════════════════════════════
# FillExtrap returns a constant for out-of-domain queries → ∂fill/∂f = 0.
# We zero out Δy entries for OOB queries so the adjoint doesn't accumulate
# spurious gradients from those positions.

@inline _needs_oob_masking(::FastInterpolations.AbstractExtrap) = false
@inline _needs_oob_masking(::FillExtrap) = true

@inline function _mask_oob_tangent(Δy, x, xq, extrap)
    _needs_oob_masking(extrap) || return Δy
    lo, hi = first(x), last(x)
    return [lo <= xq[i] <= hi ? Δy[i] : zero(Δy[i]) for i in eachindex(Δy)]
end

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
        query::Tuple{Vararg{Real, N}}
    ) where {Tg, Tv, N}
    y = itp(query)

    function itp_nd_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent()
        # Locate once via gradient(), then scale by Δy
        grad = FastInterpolations.gradient(itp, query)
        ∂query = ntuple(i -> real(conj(Δy) * grad[i]), Val(N))
        return NoTangent(), ∂query
    end

    return y, itp_nd_pullback
end

"""
Reverse-mode rule for all ND interpolants with Vector input.

Enables `Zygote.gradient(itp, [0.5, 0.5])` to use analytical derivatives.
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

    function itp_nd_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent()
        grad = FastInterpolations.gradient(itp, query_tuple)
        ∂query = [real(conj(Δy) * grad[i]) for i in 1:N]
        return NoTangent(), ∂query
    end

    return y, itp_nd_pullback
end

# ════════════════════════════════════════
# Cubic one-shot — data adjoint (∂/∂f)
# ════════════════════════════════════════
# Enables Zygote.gradient(f -> ...(cubic_interp(x, f, xq; ...))..., f)
# by using the pre-built CubicAdjoint operator for the pullback.
#
# ForwardDiff already works via Dual propagation; these rrules exist
# solely to unblock reverse-mode backends (Zygote) that cannot trace
# through the in-place tridiagonal solve in the one-shot path.

"""
Reverse-mode rule for `cubic_interp(x, f, xq; ...)` — vector query.

The pullback computes `∂L/∂f = Wᵀ · ∂L/∂y` via `CubicAdjoint`, where
`W = Eᵧ + E_z · A⁻¹ · R` is the full interpolation operator.

All extrap modes are supported. For `WrapExtrap`/`ClampExtrap`, queries
are preprocessed before adjoint construction. For `FillExtrap`, OOB
query sensitivities are zeroed (constant fill has zero gradient w.r.t. data).
"""
function ChainRulesCore.rrule(
        ::typeof(cubic_interp),
        x::AbstractVector{Tg},
        f::AbstractVector{Tv},
        xq::AbstractVector{Tg};
        kwargs...
    ) where {Tg <: AbstractFloat, Tv}
    y = cubic_interp(x, f, xq; kwargs...)

    bc = get(kwargs, :bc, CubicFit())
    extrap = get(kwargs, :extrap, NoExtrap())
    deriv = get(kwargs, :deriv, EvalValue())
    adj = cubic_adjoint(x, xq; bc, extrap)

    function cubic_interp_vec_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        Δy_eff = _mask_oob_tangent(unthunk(Δy), x, xq, extrap)
        f_bar = adj(Δy_eff; deriv = deriv)
        return NoTangent(), NoTangent(), f_bar, NoTangent()
    end

    return y, cubic_interp_vec_pullback
end

"""
Reverse-mode rule for `cubic_interp(x, f, xq; ...)` — scalar query.

Wraps the scalar query into a 1-element vector for `CubicAdjoint`,
then unwraps the scalar cotangent for the pullback.

All extrap modes are supported.
"""
function ChainRulesCore.rrule(
        ::typeof(cubic_interp),
        x::AbstractVector{Tg},
        f::AbstractVector{Tv},
        xq::Real;
        kwargs...
    ) where {Tg <: AbstractFloat, Tv}
    y = cubic_interp(x, f, xq; kwargs...)

    bc = get(kwargs, :bc, CubicFit())
    extrap = get(kwargs, :extrap, NoExtrap())
    deriv = get(kwargs, :deriv, EvalValue())
    adj = cubic_adjoint(x, Tg[xq]; bc, extrap)

    function cubic_interp_scalar_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        Δy_eff = _mask_oob_tangent(Tg[unthunk(Δy)], x, Tg[xq], extrap)
        f_bar = adj(Δy_eff; deriv = deriv)
        return NoTangent(), NoTangent(), f_bar, NoTangent()
    end

    return y, cubic_interp_scalar_pullback
end

# ════════════════════════════════════════
# Cubic ND one-shot — data adjoint (∂/∂data)
# ════════════════════════════════════════
# Enables Zygote.gradient(data -> ...(cubic_interp(grids, data, queries; ...))..., data)
# by using the pre-built CubicAdjointND operator for the pullback.
#
# Currently supports NoExtrap (default) and PeriodicBC. Other extrap modes
# (FillExtrap, ClampExtrap, WrapExtrap) will be added when the ND adjoint
# gains native extrap support.

"""
Reverse-mode rule for `cubic_interp(grids, data, queries; ...)` — SoA batch (ND).

The pullback computes `∂L/∂data = Wᵀ · ∂L/∂y` via `CubicAdjointND`.
Grids and queries are not differentiated.
"""
function ChainRulesCore.rrule(
        ::typeof(cubic_interp),
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        queries::Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}};
        kwargs...
    ) where {Tv, N}
    y = cubic_interp(grids, data, queries; kwargs...)

    bc = get(kwargs, :bc, CubicFit())
    deriv = get(kwargs, :deriv, EvalValue())
    adj = cubic_adjoint(grids, queries; bc)

    function cubic_interp_nd_soa_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        f_bar = adj(unthunk(Δy); deriv = deriv)
        return NoTangent(), NoTangent(), f_bar, NoTangent()
    end

    return y, cubic_interp_nd_soa_pullback
end

"""
Reverse-mode rule for `cubic_interp(grids, data, query; ...)` — single point (ND).

Wraps the scalar query tuple into 1-element vectors for `CubicAdjointND`,
then unwraps the scalar cotangent for the pullback.
"""
function ChainRulesCore.rrule(
        ::typeof(cubic_interp),
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tv, N}
    y = cubic_interp(grids, data, query; kwargs...)

    bc = get(kwargs, :bc, CubicFit())
    deriv = get(kwargs, :deriv, EvalValue())

    # Wrap scalar query into 1-element vectors for CubicAdjointND
    Tg = FastInterpolations._promote_grid_eltype(grids)
    Tg_f = Tg <: AbstractFloat ? Tg : Float64
    queries_vec = ntuple(d -> Tg_f[query[d]], Val(N))
    adj = cubic_adjoint(grids, queries_vec; bc)

    function cubic_interp_nd_scalar_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        f_bar = adj(Tg_f[unthunk(Δy)]; deriv = deriv)
        return NoTangent(), NoTangent(), f_bar, NoTangent()
    end

    return y, cubic_interp_nd_scalar_pullback
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
        ::typeof(cubic_interp),
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{Tv, N};
        kwargs...
    ) where {Tv, N}
    itp = cubic_interp(grids, data; kwargs...)

    function cubic_interp_nd_ctor_pullback(Δitp)
        Δitp isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent()
        # Δitp is an Array (data_bar) computed by the eval/gradient/hessian rrules
        return NoTangent(), NoTangent(), Δitp
    end

    return itp, cubic_interp_nd_ctor_pullback
end

# ════════════════════════════════════════
# CubicInterpolantND — eval rrule with ∂/∂data
# ════════════════════════════════════════
# More specific than the generic AbstractInterpolantND rrule (lines 140-155),
# so Julia dispatches here for CubicInterpolantND.
# Returns both ∂/∂query AND ∂/∂data (as the itp tangent).

"""
Eval rrule for `itp::CubicInterpolantND(query)` with ∂/∂data support.

The pullback computes:
- `∂query`: via `gradient(itp, query)` (locate-once, analytical)
- `data_bar`: via `CubicAdjointND(Δy)` — returned as the itp tangent,
  which flows back through the constructor rrule to become `Δdata`.
"""
function ChainRulesCore.rrule(
        itp::FastInterpolations.CubicInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}}
    ) where {Tg, Tv, N}
    y = itp(query)

    # Build adjoint for ∂/∂data (single query point → wrap in 1-element vectors)
    queries_vec = ntuple(d -> Tg[query[d]], Val(N))
    adj = cubic_adjoint(itp.grids, queries_vec; bc = itp.bcs)

    function cubic_itp_nd_eval_pullback(Δy)
        Δy isa AbstractZero && return ZeroTangent(), ZeroTangent()
        Δy_val = unthunk(Δy)

        # ∂/∂query via analytical gradient (locate-once)
        grad = FastInterpolations.gradient(itp, query)
        ∂query = ntuple(i -> real(conj(Δy_val) * grad[i]), Val(N))

        # ∂/∂data via adjoint operator
        data_bar = adj(Tg[Δy_val]; deriv = EvalValue())

        return data_bar, ∂query
    end

    return y, cubic_itp_nd_eval_pullback
end

# ════════════════════════════════════════
# CubicInterpolantND — gradient rrule with ∂/∂data
# ════════════════════════════════════════
# For `FastInterpolations.gradient(itp, query)` → NTuple{N} of partials.
# Single CubicAdjointND built once, applied N times with different deriv tuples.

"""
rrule for `gradient(itp::CubicInterpolantND, query)` with ∂/∂data support.

The pullback receives `Δgrad::NTuple{N}` (cotangents for each partial derivative).
For each axis `i` where `Δgrad[i] ≠ 0`:
- `data_bar += adj(Δgrad[i]; deriv = e_i)` where `e_i` = DerivOp(1) on axis i
- `∂query[j] += Δgrad[i] * H[i,j]` (Hessian row)

Uses a single `CubicAdjointND` for all N data-adjoint applications.
"""
function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.gradient),
        itp::FastInterpolations.CubicInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    grad = FastInterpolations.gradient(itp, query; kwargs...)

    # Build adjoint once for ∂/∂data
    queries_vec = ntuple(d -> Tg[query[d]], Val(N))
    adj = cubic_adjoint(itp.grids, queries_vec; bc = itp.bcs)

    function gradient_itp_nd_pullback(Δgrad_raw)
        Δgrad_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        Δgrad = unthunk(Δgrad_raw)

        # ∂/∂data: accumulate adjoint applications for each axis
        data_bar = zeros(Tg, size(itp)...)
        for i in 1:N
            dg_i = Δgrad[i]
            iszero(dg_i) && continue
            ops_i = ntuple(j -> j == i ? DerivOp(1) : EvalValue(), Val(N))
            data_bar .+= adj(Tg[dg_i]; deriv = ops_i)
        end

        # ∂/∂query: Hessian × Δgrad
        H = FastInterpolations.hessian(itp, query)
        ∂query = ntuple(j -> sum(real(conj(Δgrad[i]) * H[i, j]) for i in 1:N), Val(N))

        return NoTangent(), data_bar, ∂query
    end

    return grad, gradient_itp_nd_pullback
end

# ════════════════════════════════════════
# CubicInterpolantND — hessian rrule with ∂/∂data
# ════════════════════════════════════════
# For `FastInterpolations.hessian(itp, query)` → N×N Matrix.
# Builds one CubicAdjointND, applies N(N+1)/2 times (symmetry).

"""
rrule for `hessian(itp::CubicInterpolantND, query)` with ∂/∂data support.

The pullback receives `ΔH::Matrix` (N×N cotangent matrix).
For each unique (i,j) pair:
- `data_bar += adj(ΔH[i,j]; deriv = ops_ij)` where `ops_ij` has DerivOp on axes i,j
- `∂query` via third derivatives (not computed — returned as ZeroTangent for now)
"""
function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.hessian),
        itp::FastInterpolations.CubicInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    H = FastInterpolations.hessian(itp, query; kwargs...)

    queries_vec = ntuple(d -> Tg[query[d]], Val(N))
    adj = cubic_adjoint(itp.grids, queries_vec; bc = itp.bcs)

    function hessian_itp_nd_pullback(ΔH_raw)
        ΔH_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        ΔH = unthunk(ΔH_raw)

        data_bar = zeros(Tg, size(itp)...)

        # Diagonal: ∂²f/∂xᵢ²
        for i in 1:N
            dh_ii = ΔH[i, i]
            iszero(dh_ii) && continue
            ops = ntuple(j -> j == i ? DerivOp(2) : EvalValue(), Val(N))
            data_bar .+= adj(Tg[dh_ii]; deriv = ops)
        end

        # Off-diagonal (symmetry: ΔH[i,j] + ΔH[j,i])
        for i in 1:N, j in (i + 1):N
            dh_ij = ΔH[i, j] + ΔH[j, i]
            iszero(dh_ij) && continue
            ops = ntuple(k -> (k == i || k == j) ? DerivOp(1) : EvalValue(), Val(N))
            data_bar .+= adj(Tg[dh_ij]; deriv = ops)
        end

        # ∂/∂query requires third derivatives — omit for now
        return NoTangent(), data_bar, ZeroTangent()
    end

    return H, hessian_itp_nd_pullback
end

# ════════════════════════════════════════
# CubicInterpolantND — laplacian rrule with ∂/∂data
# ════════════════════════════════════════
# For `FastInterpolations.laplacian(itp, query)` → scalar.
# Builds one CubicAdjointND, applies N times (diagonal only).

"""
rrule for `laplacian(itp::CubicInterpolantND, query)` with ∂/∂data support.

The pullback receives scalar `Δlap`:
- `data_bar += adj(Δlap; deriv = (EvalValue,...,DerivOp(2),...))` for each axis
- `∂query` via third derivatives — omitted (ZeroTangent).
"""
function ChainRulesCore.rrule(
        ::typeof(FastInterpolations.laplacian),
        itp::FastInterpolations.CubicInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        kwargs...
    ) where {Tg, Tv, N}
    lap = FastInterpolations.laplacian(itp, query; kwargs...)

    queries_vec = ntuple(d -> Tg[query[d]], Val(N))
    adj = cubic_adjoint(itp.grids, queries_vec; bc = itp.bcs)

    function laplacian_itp_nd_pullback(Δlap_raw)
        Δlap_raw isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent()
        Δlap = unthunk(Δlap_raw)

        data_bar = zeros(Tg, size(itp)...)
        for i in 1:N
            ops = ntuple(j -> j == i ? DerivOp(2) : EvalValue(), Val(N))
            data_bar .+= adj(Tg[Δlap]; deriv = ops)
        end

        return NoTangent(), data_bar, ZeroTangent()
    end

    return lap, laplacian_itp_nd_pullback
end

end # module
