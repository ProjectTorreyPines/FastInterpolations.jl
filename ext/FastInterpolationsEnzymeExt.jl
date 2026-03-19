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
# AbstractInterpolant1D — eval rrule (∂/∂xq)
# ════════════════════════════════════════
# Handles: itp(xq::Real) for ALL 1D interpolant types.
# Replaces Enzyme's LLVM-level autodiff (which processes the full search +
# kernel + extrap dispatch chain) with a single analytical DerivOp(1) call.
#
# Without this rule: Enzyme processes full LLVM IR per concrete type (~100-12000ms TTFJ).
# With this rule: Enzyme calls itp(xq; deriv=DerivOp(1)) instead (~<1ms TTFJ).
#
# Note: This rule fires when Enzyme runs BEFORE ChainRulesCore is loaded.
# When CRC ext is loaded first, Enzyme uses the CRC rrule as fallback instead.

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
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
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
    adj, deriv_op, dy = tape
    if dy !== nothing
        f_bar = adj(dy; deriv = deriv_op)
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

# ════════════════════════════════════════
# ND SoA batch: cubic_interp(grids, data, queries_soa) → Vector
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        RT::Type{<:Annotation},
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        queries::Const{<:Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{FastInterpolations.AbstractBC, NTuple{N, FastInterpolations.AbstractBC}} = CubicFit(),
        extrap::Union{FastInterpolations.AbstractExtrap, NTuple{N, FastInterpolations.AbstractExtrap}} = NoExtrap(),
        search::Union{FastInterpolations.AbstractSearchPolicy, NTuple{N, FastInterpolations.AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = PreCompute(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {N}
    y = cubic_interp(grids.val, data.val, queries.val; deriv, bc, extrap, search, coeffs, hint)

    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing

    adj = cubic_adjoint(grids.val, queries.val; bc, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        ::Type{RT},
        tape,
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        queries::Const{<:Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{FastInterpolations.AbstractBC, NTuple{N, FastInterpolations.AbstractBC}} = CubicFit(),
        extrap::Union{FastInterpolations.AbstractExtrap, NTuple{N, FastInterpolations.AbstractExtrap}} = NoExtrap(),
        search::Union{FastInterpolations.AbstractSearchPolicy, NTuple{N, FastInterpolations.AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = PreCompute(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {N, RT}
    adj, deriv_op, dy = tape
    if dy !== nothing
        f_bar = adj(dy; deriv = deriv_op)
        data.dval .+= f_bar
        dy .= zero(eltype(dy))
    end
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# ND scalar query: cubic_interp(grids, data, query_tuple) → scalar
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        RT::Type{<:Annotation},
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{FastInterpolations.AbstractBC, NTuple{N, FastInterpolations.AbstractBC}} = CubicFit(),
        extrap::Union{FastInterpolations.AbstractExtrap, NTuple{N, FastInterpolations.AbstractExtrap}} = NoExtrap(),
        search::Union{FastInterpolations.AbstractSearchPolicy, NTuple{N, FastInterpolations.AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = PreCompute(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {N}
    y = cubic_interp(grids.val, data.val, query.val; deriv, bc, extrap, search, coeffs, hint)

    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing

    adj = cubic_adjoint(grids.val, (query.val,); bc, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(cubic_interp)},
        dret::Active,
        tape,
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        bc::Union{FastInterpolations.AbstractBC, NTuple{N, FastInterpolations.AbstractBC}} = CubicFit(),
        extrap::Union{FastInterpolations.AbstractExtrap, NTuple{N, FastInterpolations.AbstractExtrap}} = NoExtrap(),
        search::Union{FastInterpolations.AbstractSearchPolicy, NTuple{N, FastInterpolations.AbstractSearchPolicy}} = AutoSearch(),
        coeffs::AbstractCoeffStrategy = PreCompute(),
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {N}
    adj, deriv_op = tape
    f_bar = adj(dret.val; deriv = deriv_op)
    data.dval .+= f_bar
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# linear_interp — data adjoint (∂/∂f)
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(linear_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = linear_interp(x.val, f.val, xq.val; extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = linear_adjoint(x.val, xq.val; extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(linear_interp)},
        ::Type{RT},
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, RT}
    adj, deriv_op, dy = tape
    if dy !== nothing
        f_bar = adj(dy; deriv = deriv_op)
        f.dval .+= f_bar
        dy .= zero(eltype(dy))
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(linear_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = linear_interp(x.val, f.val, xq.val; extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = linear_adjoint(x.val, Tg[xq.val]; extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(linear_interp)},
        dret::Active,
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    adj, deriv_op = tape
    f_bar = adj(Tg[dret.val]; deriv = deriv_op)
    f.dval .+= f_bar
    return (nothing, nothing, nothing)
end

# linear_interp — query adjoint (∂/∂xq), f::Const
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(linear_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Const{<:AbstractVector},
        xq::Active{<:Real};
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = linear_interp(x.val, f.val, xq.val; extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = nothing
    d = linear_interp(x.val, f.val, Tg(xq.val); extrap, deriv = DerivOp(1), search)
    return EnzymeRules.AugmentedReturn(primal, shadow, d)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig,
        ::Const{typeof(linear_interp)},
        dret::Active,
        tape,
        ::Const{<:AbstractVector{Tg}},
        ::Const{<:AbstractVector},
        ::Active{<:Real};
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    d = tape
    return (nothing, nothing, dret.val * d)
end

# ════════════════════════════════════════
# quadratic_interp — data adjoint (∂/∂f)
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(quadratic_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        bc::FastInterpolations.QuadraticBC = Left(QuadraticFit()),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = quadratic_interp(x.val, f.val, xq.val; bc, extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = quadratic_adjoint(x.val, xq.val; bc, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(quadratic_interp)},
        ::Type{RT},
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        bc::FastInterpolations.QuadraticBC = Left(QuadraticFit()),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, RT}
    adj, deriv_op, dy = tape
    if dy !== nothing
        f_bar = adj(dy; deriv = deriv_op)
        f.dval .+= f_bar
        dy .= zero(eltype(dy))
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(quadratic_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        bc::FastInterpolations.QuadraticBC = Left(QuadraticFit()),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = quadratic_interp(x.val, f.val, xq.val; bc, extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = quadratic_adjoint(x.val, Tg[xq.val]; bc, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(quadratic_interp)},
        dret::Active,
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        bc::FastInterpolations.QuadraticBC = Left(QuadraticFit()),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    adj, deriv_op = tape
    f_bar = adj(Tg[dret.val]; deriv = deriv_op)
    f.dval .+= f_bar
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# constant_interp — data adjoint (∂/∂f)
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(constant_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        side::FastInterpolations.AbstractSide = NearestSide(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = constant_interp(x.val, f.val, xq.val; side, extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = constant_adjoint(x.val, xq.val; side, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(constant_interp)},
        ::Type{RT},
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        side::FastInterpolations.AbstractSide = NearestSide(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat, RT}
    adj, deriv_op, dy = tape
    if dy !== nothing
        f_bar = adj(dy; deriv = deriv_op)
        f.dval .+= f_bar
        dy .= zero(eltype(dy))
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(constant_interp)},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        side::FastInterpolations.AbstractSide = NearestSide(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    y = constant_interp(x.val, f.val, xq.val; side, extrap, deriv, search)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = constant_adjoint(x.val, Tg[xq.val]; side, extrap)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(constant_interp)},
        dret::Active,
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        side::FastInterpolations.AbstractSide = NearestSide(),
        extrap::FastInterpolations.AbstractExtrap = NoExtrap(),
        deriv::DerivOp = EvalValue(),
        search::FastInterpolations.AbstractSearchPolicy = AutoSearch()
    ) where {Tg <: AbstractFloat}
    adj, deriv_op = tape
    f_bar = adj(Tg[dret.val]; deriv = deriv_op)
    f.dval .+= f_bar
    return (nothing, nothing, nothing)
end

end # module
