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
        bc::FastInterpolations.AbstractBC = CubicFit(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, Tv}
    y = cubic_interp(x, f, xq; bc, extrap, autocache, deriv, search)

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
        bc::FastInterpolations.AbstractBC = CubicFit(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat, Tv}
    y = cubic_interp(x, f, xq; bc, extrap, autocache, deriv, search)

    adj = cubic_adjoint(x, Tg[xq]; bc, extrap)

    function cubic_interp_scalar_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        Δy_eff = _mask_oob_tangent(Tg[unthunk(Δy)], x, Tg[xq], extrap)
        f_bar = adj(Δy_eff; deriv = deriv)
        return NoTangent(), NoTangent(), f_bar, NoTangent()
    end

    return y, cubic_interp_scalar_pullback
end

end # module
