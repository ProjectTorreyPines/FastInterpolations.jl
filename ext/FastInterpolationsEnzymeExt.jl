# ========================================
# Enzyme Extension for FastInterpolations.jl
# ========================================
# Native EnzymeRules for cubic_interp reverse-mode differentiation w.r.t. data.
# Uses CubicAdjoint operator (W^T) in the reverse pass, bypassing the
# in-place tridiagonal solve that Enzyme cannot autodiff through.
# Loaded automatically when Enzyme is available.

module FastInterpolationsEnzymeExt

using FastInterpolations
using Enzyme
using Enzyme.EnzymeRules

# ════════════════════════════════════════
# OOB masking for FillExtrap (same logic as ChainRulesCore ext)
# ════════════════════════════════════════

@inline _needs_oob_masking(::FastInterpolations.AbstractExtrap) = false
@inline _needs_oob_masking(::FillExtrap) = true

@inline function _mask_oob_tangent(Δy, x, xq, extrap)
    _needs_oob_masking(extrap) || return Δy
    lo, hi = first(x), last(x)
    return [lo <= xq[i] <= hi ? Δy[i] : zero(Δy[i]) for i in eachindex(Δy)]
end

# ════════════════════════════════════════
# Vector query: cubic_interp(x, f, xq_vec) → Vector
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        bc::FastInterpolations.AbstractBC = CubicFit(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = cubic_interp(x.val, f.val, xq.val; bc, extrap, autocache, deriv, search)

    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing

    adj = cubic_adjoint(x.val, xq.val; bc, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow, extrap, x.val, xq.val))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        ::Type{RT},
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        bc::FastInterpolations.AbstractBC = CubicFit(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, RT}
    adj, deriv_op, dy, extrap_t, x_val, xq_val = tape
    if dy !== nothing
        dy_eff = _mask_oob_tangent(dy, x_val, xq_val, extrap_t)
        f_bar = adj(dy_eff; deriv = deriv_op)
        f.dval .+= f_bar
        dy .= zero(eltype(dy))
    end
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# Scalar query: cubic_interp(x, f, xq_scalar) → scalar
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        bc::FastInterpolations.AbstractBC = CubicFit(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat}
    y = cubic_interp(x.val, f.val, xq.val; bc, extrap, autocache, deriv, search)

    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing

    adj = cubic_adjoint(x.val, Tg[xq.val]; bc, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        dret::Active,
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        bc::FastInterpolations.AbstractBC = CubicFit(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch(),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg <: AbstractFloat}
    adj, deriv_op = tape
    f_bar = adj(Tg[dret.val]; deriv = deriv_op)
    f.dval .+= f_bar
    return (nothing, nothing, nothing)
end

end # module
