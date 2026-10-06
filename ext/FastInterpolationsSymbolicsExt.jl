# ========================================
# Symbolics Extension for FastInterpolations.jl
# ========================================
# Registers FastInterpolations interpolant types with Symbolics.jl
# so they can be used in ModelingToolkit.jl contexts.
#
# Supports:
# - 1D interpolants (AbstractInterpolant1D): symbolic calling + derivative chain rules
# - ND interpolants (AbstractInterpolantND): symbolic calling + partial derivative chain rules
#
# Usage:
#   using FastInterpolations, Symbolics
#   @variables t
#   itp = cubic_interp(x, y; extrap=ExtendExtrap())
#   expr = itp(t)              # Symbolic expression
#   D = Differential(t)
#   D(itp(t))                  # Symbolic derivative: DerivativeView{1}(itp)(t)

module FastInterpolationsSymbolicsExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant, AbstractInterpolant1D, AbstractInterpolantND,
    LinearInterpolantND, CubicInterpolantND, QuadraticInterpolantND, ConstantInterpolantND,
    DerivOp, EvalValue, DerivativeView, deriv_view, deriv_order
using Symbolics
using Symbolics: Num, unwrap
import SymbolicUtils

# This extension targets the Symbolics 7 / SymbolicUtils 4 symbolic API
# (`SymbolicUtils.TypeT` / `ShapeT` / `@register_derivative`). On the older
# Symbolics 6 / SymbolicUtils 3 generation that API is absent, so the symbolic
# interpolation glue is compiled out and the extension loads as a no-op. Numeric
# interpolation in the FastInterpolations core is unaffected on either generation.
@static if isdefined(SymbolicUtils, :TypeT)

    # ========================================
    # Shared Helpers
    # ========================================

    # Symbolic call `f(args...)` with the callable `f` (an interpolant or its
    # DerivativeView) as the term's operation, never as an argument: the chain
    # rule differentiates every argument, and an interpolant has no symbolic zero.
    function _symbolic_call(f, t_args)
        return Num(SymbolicUtils.term(f, unwrap.(t_args)...; type = Real))
    end

    # Operation for a call's `deriv` request: the interpolant itself for a value,
    # otherwise its DerivativeView (ND: per-axis orders, or one broadcast to all axes).
    _symbolic_op(itp, ::DerivOp{0}) = itp
    _symbolic_op(itp, op::DerivOp) = deriv_view(itp, deriv_order(op))
    function _symbolic_op(itp, ops::Tuple)
        return all(op -> op isa DerivOp{0}, ops) ? itp : deriv_view(itp, ops)
    end

    # Derivative order of a view: Int for 1D, per-axis NTuple for ND.
    _view_order(::DerivativeView{Order}) where {Order} = Order

    Base.nameof(::AbstractInterpolant) = :FastInterpolation
    Base.nameof(::AbstractInterpolantND) = :FastInterpolationND
    Base.nameof(::DerivativeView) = :DifferentiatedFastInterpolation
    Base.nameof(::DerivativeView{<:Any, <:AbstractInterpolantND}) = :DifferentiatedFastInterpolationND

    # Symtype/shape promotion: interpolants and their derivative views return scalars.
    # Shape promotion must use the concrete ShapeT union type to avoid ambiguity
    # with the generic fallback.
    const _SymbolicCallable = Union{AbstractInterpolant1D, AbstractInterpolantND, DerivativeView}
    const _ShapeT = SymbolicUtils.ShapeT

    function SymbolicUtils.promote_symtype(::_SymbolicCallable, ::Vararg)
        return Real
    end

    function SymbolicUtils.promote_shape(::_SymbolicCallable, ::Vararg{_ShapeT})
        return SymbolicUtils.ShapeVecT()
    end

    # ========================================
    # 1D Registration
    # ========================================

    # One method for every 1D family: the core 1D entry is
    # `(itp::AbstractInterpolant1D)(xq::Number)`, which `Num <: Number` refines.
    function (itp::AbstractInterpolant1D)(t::Num; deriv::DerivOp = EvalValue(), kwargs...)
        return _symbolic_call(_symbolic_op(itp, deriv), (t,))
    end

    function (d::DerivativeView{Order, ITP})(
            t::Num; deriv = nothing, kwargs...
        ) where {Order, ITP <: AbstractInterpolant1D}
        FastInterpolations._check_no_deriv_override(Val(Order), deriv)
        return _symbolic_call(d, (t,))
    end

    # Derivative chain rules:
    # d/dt itp(t) = DerivativeView{1}(itp)(t)
    @register_derivative (itp::AbstractInterpolant1D)(t) 1 begin
        SymbolicUtils.term(deriv_view(itp, 1), t; type = Real)
    end
    # d/dt DerivativeView{n}(itp)(t) = DerivativeView{n + 1}(itp)(t)
    @register_derivative (d::DerivativeView{<:Any, <:AbstractInterpolant1D})(t) 1 begin
        SymbolicUtils.term(deriv_view(d.parent, _view_order(d) + 1), t; type = Real)
    end

    # ========================================
    # ND Registration
    # ========================================

    # Per concrete type: the core defines each family's ND entry
    # `(itp::CubicInterpolantND)(query::Tuple{Vararg{Number, N}})` separately, so a
    # method on AbstractInterpolantND would be ambiguous with every one of them.
    for NDT in [CubicInterpolantND, LinearInterpolantND, QuadraticInterpolantND, ConstantInterpolantND]
        # Tuple form: itp((u, v))
        @eval function (itp::$NDT{Tg, Tv, N})(
                t::NTuple{N, Num}; deriv = EvalValue(), kwargs...
            ) where {Tg, Tv, N}
            return _symbolic_call(_symbolic_op(itp, deriv), t)
        end

        # Varargs form: itp(u, v)
        @eval function (itp::$NDT{Tg, Tv, N})(
                t::Vararg{Num, N}; deriv = EvalValue(), kwargs...
            ) where {Tg, Tv, N}
            return _symbolic_call(_symbolic_op(itp, deriv), t)
        end
    end

    # DerivativeView of an ND interpolant: tuple and varargs forms
    function (d::DerivativeView{Order, ITP})(
            t::NTuple{N, Num}; deriv = nothing, kwargs...
        ) where {Order, Tg, Tv, N, ITP <: AbstractInterpolantND{Tg, Tv, N}}
        FastInterpolations._check_no_deriv_override(Val(Order), deriv)
        return _symbolic_call(d, t)
    end

    function (d::DerivativeView{Order, ITP})(
            t::Vararg{Num, N}; deriv = nothing, kwargs...
        ) where {Order, Tg, Tv, N, ITP <: AbstractInterpolantND{Tg, Tv, N}}
        FastInterpolations._check_no_deriv_override(Val(Order), deriv)
        return _symbolic_call(d, t)
    end

    # Derivative chain rules:
    # d/d(arg_I) itp(args...) = DerivativeView{(0,…,1,…,0)}(itp)(args...)
    @register_derivative (itp::AbstractInterpolantND)(args...) I begin
        orders = ntuple(j -> j == I ? 1 : 0, Val{Nargs}())
        SymbolicUtils.term(deriv_view(itp, orders), args...; type = Real)
    end
    # DerivativeViews accumulate the per-axis orders
    @register_derivative (d::DerivativeView{<:Any, <:AbstractInterpolantND})(args...) I begin
        orders = _view_order(d) .+ ntuple(j -> j == I ? 1 : 0, Val{Nargs}())
        SymbolicUtils.term(deriv_view(d.parent, orders), args...; type = Real)
    end

end # @static if isdefined(SymbolicUtils, :TypeT)

end # module
