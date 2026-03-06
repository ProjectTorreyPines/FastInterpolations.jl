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
#   # ForwardDiff uses frule, Zygote uses rrule — both pick up these rules.

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

Enables `ForwardDiff.derivative(itp, x)` to use analytical derivatives
instead of Dual number propagation.
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
    ∂y = Δx * itp(x; deriv=DerivOp(1))
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
        return NoTangent(), real(conj(Δy) * itp(x; deriv=DerivOp(1)))
    end

    return y, itp_1d_pullback
end

# ════════════════════════════════════════
# ND Interpolants — Tuple query
# ════════════════════════════════════════

"""
Forward-mode rule for all ND interpolants with Tuple input.

Enables `ForwardDiff.derivative(q -> itp((q, 0.5)), 0.5)` to use
analytical partial derivatives.
"""
function ChainRulesCore.frule(
    (_, Δquery),
    itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
    query::Tuple{Vararg{Real, N}}
) where {Tg, Tv, N}
    y = itp(query)
    Δquery isa AbstractZero && return y, zero(y)

    # Directional derivative: ∑ᵢ (∂f/∂xᵢ) * Δxᵢ
    ∂y = sum(
        Δquery[i] * itp(query; deriv=ntuple(j -> DerivOp(j == i ? 1 : 0), Val(N)))
        for i in 1:N
    )
    return y, ∂y
end

"""
Forward-mode rule for all ND interpolants with Vector input.

Enables `ForwardDiff.gradient(itp, [0.5, 0.5])` to use analytical derivatives.
"""
function ChainRulesCore.frule(
    (_, Δquery),
    itp::FastInterpolations.AbstractInterpolantND{Tg, Tv, N},
    query::AbstractVector{<:Real}
) where {Tg, Tv, N}
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    y = itp(query_tuple)
    Δquery isa AbstractZero && return y, zero(y)

    ∂y = sum(
        Δquery[i] * itp(query_tuple; deriv=ntuple(j -> DerivOp(j == i ? 1 : 0), Val(N)))
        for i in 1:N
    )
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
        ∂query = ntuple(Val(N)) do i
            real(conj(Δy) * itp(query; deriv=ntuple(j -> DerivOp(j == i ? 1 : 0), Val(N))))
        end
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
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    y = itp(query_tuple)

    function itp_nd_pullback(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent()
        ∂query = [
            real(conj(Δy) * itp(query_tuple; deriv=ntuple(j -> DerivOp(j == i ? 1 : 0), Val(N))))
            for i in 1:N
        ]
        return NoTangent(), ∂query
    end

    return y, itp_nd_pullback
end

end # module
