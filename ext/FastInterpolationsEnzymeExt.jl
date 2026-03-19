# ========================================
# Enzyme Extension for FastInterpolations.jl
# ========================================
# Native EnzymeRules for all interpolant reverse-mode differentiation w.r.t. data.
# Uses adjoint operators (W^T) in the reverse pass, bypassing the in-place
# tridiagonal solve that Enzyme cannot autodiff through.
# Loaded automatically when Enzyme is available.

module FastInterpolationsEnzymeExt

using FastInterpolations
using Enzyme
using Enzyme.EnzymeRules

# ════════════════════════════════════════
# Trait dispatch — same pattern as CRC extension
# ════════════════════════════════════════

const _InterpMethod = Union{
    typeof(linear_interp), typeof(quadratic_interp),
    typeof(constant_interp), typeof(cubic_interp),
}

_adjoint_func(::typeof(linear_interp)) = linear_adjoint
_adjoint_func(::typeof(quadratic_interp)) = quadratic_adjoint
_adjoint_func(::typeof(constant_interp)) = constant_adjoint
_adjoint_func(::typeof(cubic_interp)) = cubic_adjoint

# ════════════════════════════════════════
# 1D Vector query: func(x, f, xq_vec) → Vector
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        kwargs...
    ) where {Tg <: AbstractFloat}
    y = func.val(x.val, f.val, xq.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = _adjoint_func(func.val)(x.val, xq.val; kwargs...)
    deriv = get(kwargs, :deriv, EvalValue())
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        ::Type{RT},
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:AbstractVector{Tg}};
        kwargs...
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
# 1D Scalar query: func(x, f, xq_scalar) → scalar
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        kwargs...
    ) where {Tg <: AbstractFloat}
    y = func.val(x.val, f.val, xq.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = _adjoint_func(func.val)(x.val, Tg[xq.val]; kwargs...)
    deriv = get(kwargs, :deriv, EvalValue())
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        dret::Active,
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Duplicated{<:AbstractVector},
        xq::Const{<:Real};
        kwargs...
    ) where {Tg <: AbstractFloat}
    adj, deriv_op = tape
    f_bar = adj(Tg[dret.val]; deriv = deriv_op)
    f.dval .+= f_bar
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# 1D Scalar query — ∂/∂xq (f::Const, xq::Active)
# ════════════════════════════════════════
# Generic for all 4 types: computes derivative via DerivOp(1).

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Const{<:AbstractVector},
        xq::Active{<:Real};
        kwargs...
    ) where {Tg <: AbstractFloat}
    y = func.val(x.val, f.val, xq.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = nothing
    d = func.val(x.val, f.val, Tg(xq.val); deriv = DerivOp(1), kwargs...)
    return EnzymeRules.AugmentedReturn(primal, shadow, d)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig,
        ::Const{<:_InterpMethod},
        dret::Active,
        tape,
        ::Const{<:AbstractVector{Tg}},
        ::Const{<:AbstractVector},
        ::Active{<:Real};
        kwargs...
    ) where {Tg <: AbstractFloat}
    d = tape
    return (nothing, nothing, real(conj(dret.val) * d))
end

# ════════════════════════════════════════
# 1D Vector query — ∂/∂xq (f::Const, xq::Duplicated)
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        x::Const{<:AbstractVector{Tg}},
        f::Const{<:AbstractVector},
        xq::Duplicated{<:AbstractVector{Tg}};
        kwargs...
    ) where {Tg <: AbstractFloat}
    y = func.val(x.val, f.val, xq.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    d = func.val(x.val, f.val, xq.val; deriv = DerivOp(1), kwargs...)
    return EnzymeRules.AugmentedReturn(primal, shadow, (d, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        ::Type{RT},
        tape,
        x::Const{<:AbstractVector{Tg}},
        f::Const{<:AbstractVector},
        xq::Duplicated{<:AbstractVector{Tg}};
        kwargs...
    ) where {Tg <: AbstractFloat, RT}
    d, dy = tape
    if dy !== nothing
        xq.dval .+= real.(conj.(dy) .* d)
        dy .= zero(eltype(dy))
    end
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# ND SoA batch: func(grids, data, queries_soa) → Vector
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        queries::Const{<:Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}}};
        kwargs...
    ) where {N}
    y = func.val(grids.val, data.val, queries.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = _adjoint_func(func.val)(grids.val, queries.val; kwargs...)
    deriv = get(kwargs, :deriv, EvalValue())
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        ::Type{RT},
        tape,
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        queries::Const{<:Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}}};
        kwargs...
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
# ND scalar query: func(grids, data, query_tuple) → scalar
# ════════════════════════════════════════

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {N}
    y = func.val(grids.val, data.val, query.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj = _adjoint_func(func.val)(grids.val, (query.val,); kwargs...)
    deriv = get(kwargs, :deriv, EvalValue())
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, deriv))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        dret::Active,
        tape,
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {N}
    adj, deriv_op = tape
    f_bar = adj(dret.val; deriv = deriv_op)
    data.dval .+= f_bar
    return (nothing, nothing, nothing)
end

# ════════════════════════════════════════
# ND struct API — constructor + eval rules
# ════════════════════════════════════════
# Enables: itp = func(grids, data); loss(itp(query))
#
# Mechanism: shadow struct as communication buffer between eval and constructor.
#   eval reverse:  data_bar = adj(dy) → stored in shadow's data-sized field
#   ctor reverse:  data.dval .+= shadow's data-sized field
#
# The adjoint operator returns ∂L/∂data directly, bypassing all internal
# transforms (tridiag solve, slope recurrence). The shadow field is just a buffer.

# Trait: struct → adjoint constructor + kwargs (same as CRC ext)
_adjoint_func_from_itp(::FastInterpolations.CubicInterpolantND) = cubic_adjoint
_adjoint_func_from_itp(::FastInterpolations.QuadraticInterpolantND) = quadratic_adjoint
_adjoint_func_from_itp(::FastInterpolations.LinearInterpolantND) = linear_adjoint
_adjoint_func_from_itp(::FastInterpolations.ConstantInterpolantND) = constant_adjoint

_adjoint_kwargs_from_itp(itp::FastInterpolations.CubicInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::FastInterpolations.QuadraticInterpolantND) = (bc = itp.bcs, extrap = itp.extraps)
_adjoint_kwargs_from_itp(itp::FastInterpolations.LinearInterpolantND) = (extrap = itp.extraps,)
_adjoint_kwargs_from_itp(itp::FastInterpolations.ConstantInterpolantND) = (side = itp.sides, extrap = itp.extraps)

# Trait: data-sized mutable buffer inside shadow struct for accumulating data_bar.
# Linear/Constant: .data is same size as input data.
# Cubic/Quadratic: .nodal_derivs.partials is (2^N, n1, n2, ...); use first slice as buffer.
_data_field(itp::FastInterpolations.LinearInterpolantND) = itp.data
_data_field(itp::FastInterpolations.ConstantInterpolantND) = itp.data
_data_field(itp::FastInterpolations.CubicInterpolantND) = selectdim(itp.nodal_derivs.partials, 1, 1)
_data_field(itp::FastInterpolations.QuadraticInterpolantND) = selectdim(itp.nodal_derivs.partials, 1, 1)

# ── Constructor: func(grids, data) → itp ──

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        RT::Type{<:Annotation},
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}};
        kwargs...
    ) where {N}
    itp = func.val(grids.val, data.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? itp : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Enzyme.make_zero(itp) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{<:_InterpMethod},
        ::Type{RT},
        tape,
        grids::Const{<:NTuple{N, AbstractVector}},
        data::Duplicated{<:AbstractArray{<:Any, N}};
        kwargs...
    ) where {N, RT}
    shadow_itp = tape
    if shadow_itp !== nothing
        data.dval .+= _data_field(shadow_itp)
    end
    return (nothing, nothing)
end

# ── Eval: itp(query) → scalar ──
# Enzyme uses MixedDuplicated for immutable structs with mutable fields.
# Shadow access: Duplicated → itp.dval, MixedDuplicated → itp.dval[] (Ref).

const _DupItp{T} = Union{Duplicated{T}, MixedDuplicated{T}}

@inline _get_shadow(itp::Duplicated) = itp.dval
@inline _get_shadow(itp::MixedDuplicated) = itp.dval[]

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        RT::Type{<:Annotation},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    y = itp.val(query.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    adj_fn = _adjoint_func_from_itp(itp.val)
    adj = adj_fn(itp.val.grids, (query.val,); _adjoint_kwargs_from_itp(itp.val)...)
    return EnzymeRules.AugmentedReturn(primal, shadow, adj)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        dret::Active,
        tape,
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    adj = tape
    data_bar = adj(dret.val; deriv = EvalValue())
    _data_field(_get_shadow(itp)) .+= data_bar
    return (nothing,)
end

# ── Eval: itp(query) → scalar, ∂/∂query only (itp::Const, query::Duplicated) ──

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        itp::Const{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        RT::Type{<:Annotation},
        query::Duplicated{<:AbstractVector{<:Real}};
        kwargs...
    ) where {Tg, Tv, N}
    q_tuple = ntuple(i -> @inbounds(query.val[i]), Val(N))
    y = itp.val(q_tuple; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    grad = FastInterpolations.gradient(itp.val, q_tuple)
    return EnzymeRules.AugmentedReturn(primal, shadow, grad)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        itp::Const{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        dret::Active,
        tape,
        query::Duplicated{<:AbstractVector{<:Real}};
        kwargs...
    ) where {Tg, Tv, N}
    grad = tape
    for i in 1:N
        query.dval[i] += real(conj(dret.val) * grad[i])
    end
    return (nothing,)
end

# ── gradient(itp, query) → NTuple{N} ──

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(FastInterpolations.gradient)},
        RT::Type{<:Annotation},
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    grad = FastInterpolations.gradient(itp.val, query.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? grad : nothing
    shadow = EnzymeRules.needs_shadow(config) ? ntuple(_ -> zero(Tg), Val(N)) : nothing
    adj_fn = _adjoint_func_from_itp(itp.val)
    adj = adj_fn(itp.val.grids, (query.val,); _adjoint_kwargs_from_itp(itp.val)...)
    return EnzymeRules.AugmentedReturn(primal, shadow, adj)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(FastInterpolations.gradient)},
        dret::Active,
        tape,
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    adj = tape
    Δgrad = dret.val
    shadow = _get_shadow(itp)
    buf = _data_field(shadow)
    for i in 1:N
        dg_i = Δgrad[i]
        iszero(dg_i) && continue
        ops_i = ntuple(j -> j == i ? DerivOp(1) : EvalValue(), Val(N))
        buf .+= adj(dg_i; deriv = ops_i)
    end
    return (nothing, nothing)
end

# ── hessian(itp, query) → Matrix{N×N} ──

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(FastInterpolations.hessian)},
        RT::Type{<:Annotation},
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    H = FastInterpolations.hessian(itp.val, query.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? H : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(H) : nothing
    adj_fn = _adjoint_func_from_itp(itp.val)
    adj = adj_fn(itp.val.grids, (query.val,); _adjoint_kwargs_from_itp(itp.val)...)
    return EnzymeRules.AugmentedReturn(primal, shadow, (adj, shadow))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(FastInterpolations.hessian)},
        ::Type{RT},
        tape,
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N, RT}
    adj, ΔH = tape
    if ΔH !== nothing
        shadow = _get_shadow(itp)
        buf = _data_field(shadow)
        # Diagonal
        for i in 1:N
            dh_ii = ΔH[i, i]
            iszero(dh_ii) && continue
            ops = ntuple(j -> j == i ? DerivOp(2) : EvalValue(), Val(N))
            buf .+= adj(dh_ii; deriv = ops)
        end
        # Off-diagonal (symmetry)
        for i in 1:N, j in (i + 1):N
            dh_ij = ΔH[i, j] + ΔH[j, i]
            iszero(dh_ij) && continue
            ops = ntuple(k -> (k == i || k == j) ? DerivOp(1) : EvalValue(), Val(N))
            buf .+= adj(dh_ij; deriv = ops)
        end
        ΔH .= zero(eltype(ΔH))
    end
    return (nothing, nothing)
end

# ── laplacian(itp, query) → scalar ──

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(FastInterpolations.laplacian)},
        RT::Type{<:Annotation},
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    lap = FastInterpolations.laplacian(itp.val, query.val; kwargs...)
    primal = EnzymeRules.needs_primal(config) ? lap : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(lap) : nothing
    adj_fn = _adjoint_func_from_itp(itp.val)
    adj = adj_fn(itp.val.grids, (query.val,); _adjoint_kwargs_from_itp(itp.val)...)
    return EnzymeRules.AugmentedReturn(primal, shadow, adj)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(FastInterpolations.laplacian)},
        dret::Active,
        tape,
        itp::_DupItp{<:FastInterpolations.AbstractInterpolantND{Tg, Tv, N}},
        query::Const{<:Tuple{Vararg{Real, N}}};
        kwargs...
    ) where {Tg, Tv, N}
    adj = tape
    Δlap = dret.val
    shadow = _get_shadow(itp)
    buf = _data_field(shadow)
    for i in 1:N
        ops = ntuple(j -> j == i ? DerivOp(2) : EvalValue(), Val(N))
        buf .+= adj(Δlap; deriv = ops)
    end
    return (nothing, nothing)
end

end # module
