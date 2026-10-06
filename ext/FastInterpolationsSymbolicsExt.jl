# ========================================
# Symbolics Extension for FastInterpolations.jl
# ========================================
# Registers FastInterpolations interpolant types with Symbolics.jl
# so they can be used in ModelingToolkit.jl contexts.
#
# Supports:
# - 1D interpolants (AbstractInterpolant): symbolic calling + derivative chain rules
# - ND interpolants (AbstractInterpolantND): symbolic calling + partial derivative chain rules
#
# Usage:
#   using FastInterpolations, Symbolics
#   @variables t
#   itp = cubic_interp(x, y; extrap=ExtendExtrap())
#   expr = itp(t)              # Symbolic expression
#   D = Differential(t)
#   D(itp(t))                  # Symbolic derivative (uses derivative chain rule)

module FastInterpolationsSymbolicsExt

using FastInterpolations
using FastInterpolations: AbstractInterpolant, AbstractInterpolant1D, AbstractInterpolantND,
    LinearInterpolant, CubicInterpolant, QuadraticInterpolant, ConstantInterpolant,
    LinearInterpolantND, CubicInterpolantND, QuadraticInterpolantND, ConstantInterpolantND,
    DerivOp
using Symbolics
using Symbolics: Num, unwrap, wrap
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

    # Symbolic call `f(args...)` with the callable `f` (an interpolant, or a derivative
    # wrapper around one) as the term's operation, never as an argument: the chain
    # rule differentiates every argument, and an interpolant has no symbolic zero.
    function _symbolic_call(f, t_args, is_num::Bool)
        args = is_num ? unwrap.(t_args) : t_args
        res = SymbolicUtils.term(f, args...; type = Real)
        return is_num ? Num(res) : res
    end

    # Shape promotion must use the concrete ShapeT union type to avoid ambiguity
    # with the generic fallback.
    const _ShapeT = SymbolicUtils.ShapeT

    # ========================================
    # 1D Interpolant Registration
    # ========================================

    # Wrapper struct for tracking the derivative order K of 1D interpolants.
    # Enables higher-order symbolic differentiation by accumulating the order.
    # K is a type parameter so compiled symbolic code resolves DerivOp(K) statically.
    struct DifferentiatedInterpolant{K, I <: AbstractInterpolant1D}
        interp::I
    end

    function DifferentiatedInterpolant(itp::AbstractInterpolant1D, order::Int)
        return DifferentiatedInterpolant{order, typeof(itp)}(itp)
    end

    _derivative_order(::DifferentiatedInterpolant{K}) where {K} = K

    function (d::DifferentiatedInterpolant{K})(t::Real) where {K}
        return d.interp(t; deriv = DerivOp(K))
    end

    Base.nameof(itp::AbstractInterpolant) = :FastInterpolation
    Base.nameof(::DifferentiatedInterpolant) = :DifferentiatedFastInterpolation

    # Compact display: symbolic expressions print their operation, and the default
    # show would spell out the interpolant's full type.
    function Base.show(io::IO, d::DifferentiatedInterpolant)
        print(io, "DifferentiatedInterpolant(")
        show(io, d.interp)
        print(io, ", ", _derivative_order(d), ")")
        return nothing
    end

    # Register 1D callables for Num and BasicSymbolic argument types.
    # Must define on concrete types: their (itp::ConcreteType)(xq; ...) methods
    # leave xq untyped, so a method on AbstractInterpolant1D would be ambiguous.
    for T in [LinearInterpolant, CubicInterpolant, QuadraticInterpolant, ConstantInterpolant]
        for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
            is_num = symT === Num
            @eval function (itp::$T)(t::$symT; kwargs...)
                return _symbolic_call(itp, (t,), $is_num)
            end
        end
    end

    # DifferentiatedInterpolant symbolic calls
    for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
        is_num = symT === Num
        @eval function (d::DifferentiatedInterpolant)(t::$symT)
            return _symbolic_call(d, (t,), $is_num)
        end
    end

    # Symtype/shape promotion: 1D interpolants and their derivatives return scalars.
    function SymbolicUtils.promote_symtype(::AbstractInterpolant1D, ::Vararg)
        return Real
    end

    function SymbolicUtils.promote_symtype(::DifferentiatedInterpolant, ::Vararg)
        return Real
    end

    function SymbolicUtils.promote_shape(::AbstractInterpolant1D, ::Vararg{_ShapeT})
        return SymbolicUtils.ShapeVecT()
    end

    function SymbolicUtils.promote_shape(::DifferentiatedInterpolant, ::Vararg{_ShapeT})
        return SymbolicUtils.ShapeVecT()
    end

    # Derivative chain rules:
    # d/dt itp(t) = DifferentiatedInterpolant(itp, 1)(t)
    @register_derivative (itp::AbstractInterpolant1D)(t) 1 begin
        SymbolicUtils.term(DifferentiatedInterpolant(itp, 1), t; type = Real)
    end
    # d/dt DifferentiatedInterpolant(itp, n)(t) = DifferentiatedInterpolant(itp, n + 1)(t)
    @register_derivative (d::DifferentiatedInterpolant)(t) 1 begin
        order = _derivative_order(d) + 1
        SymbolicUtils.term(DifferentiatedInterpolant(d.interp, order), t; type = Real)
    end

    # ========================================
    # ND Interpolant Registration
    # ========================================

    # Wrapper struct for tracking derivative orders of ND interpolants.
    # Enables higher-order symbolic differentiation by accumulating per-axis orders.
    struct DifferentiatedInterpolantND{N, I <: AbstractInterpolantND}
        interp::I
        derivative_orders::NTuple{N, Int}
    end

    function (d::DifferentiatedInterpolantND{N})(args::Vararg{Real, N}) where {N}
        deriv_ops = map(n -> DerivOp(n), d.derivative_orders)
        return d.interp(args; deriv = deriv_ops)
    end

    Base.nameof(::AbstractInterpolantND) = :FastInterpolationND
    Base.nameof(::DifferentiatedInterpolantND) = :DifferentiatedFastInterpolationND

    # Register ND callable for Num and BasicSymbolic argument types.
    # Must define on concrete types to avoid ambiguity with existing
    # (itp::ConcreteND)(query::Tuple{Vararg{Real, N}}) methods.
    #
    # Also add numeric varargs methods: build_function generates `itp(x, y)` (varargs)
    # but the numeric ND API uses `itp((x, y))` (tuple). This bridge enables compiled
    # symbolic expressions to call back into the numeric code correctly.
    for NDT in [CubicInterpolantND, LinearInterpolantND, QuadraticInterpolantND, ConstantInterpolantND]
        # Numeric varargs → tuple conversion for compiled symbolic code
        @eval function (itp::$NDT{Tg, Tv, N})(
                args::Vararg{Real, N}; kwargs...
            ) where {Tg, Tv, N}
            return itp(args; kwargs...)
        end

        for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
            is_num = symT === Num
            # ND interpolant call via tuple: itp((sym_x, sym_y, ...))
            @eval function (itp::$NDT{Tg, Tv, N})(
                    t::NTuple{N, $symT}; kwargs...
                ) where {Tg, Tv, N}
                return _symbolic_call(itp, t, $is_num)
            end

            # Varargs form: itp(sym_x, sym_y, ...)
            @eval function (itp::$NDT{Tg, Tv, N})(
                    t::Vararg{$symT, N}; kwargs...
                ) where {Tg, Tv, N}
                return _symbolic_call(itp, t, $is_num)
            end
        end
    end

    # DifferentiatedInterpolantND symbolic calls
    for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
        is_num = symT === Num
        @eval function (d::DifferentiatedInterpolantND{N})(
                t::Vararg{$symT, N}
            ) where {N}
            return _symbolic_call(d, t, $is_num)
        end
    end

    # Symtype promotion for ND interpolants
    function SymbolicUtils.promote_symtype(::AbstractInterpolantND, ::Vararg)
        return Real
    end

    function SymbolicUtils.promote_symtype(::DifferentiatedInterpolantND, ::Vararg)
        return Real
    end

    # Shape promotion: ND interpolants return scalars.
    function SymbolicUtils.promote_shape(::AbstractInterpolantND, ::Vararg{_ShapeT})
        return SymbolicUtils.ShapeVecT()
    end

    function SymbolicUtils.promote_shape(::DifferentiatedInterpolantND, ::Vararg{_ShapeT})
        return SymbolicUtils.ShapeVecT()
    end

    # Derivative rules for ND interpolants via @register_derivative.
    # d/d(arg_I) itp(args...) = DifferentiatedInterpolantND(itp, (0,...,1,...,0))(args...)
    for NDT in [CubicInterpolantND, LinearInterpolantND, QuadraticInterpolantND, ConstantInterpolantND]
        @eval @register_derivative (itp::$NDT)(args...) I begin
            orders = ntuple(j -> j == I ? 1 : 0, Val{Nargs}())
            dinterp = DifferentiatedInterpolantND{Nargs, typeof(itp)}(itp, orders)
            SymbolicUtils.term(dinterp, args...; type = Real)
        end
    end

    # Derivative rules for DifferentiatedInterpolantND: accumulate orders
    @register_derivative (d::DifferentiatedInterpolantND)(args...) I begin
        orders_offset = ntuple(j -> j == I ? 1 : 0, Val{Nargs}())
        orders = d.derivative_orders .+ orders_offset
        new_d = DifferentiatedInterpolantND{Nargs, typeof(d.interp)}(d.interp, orders)
        SymbolicUtils.term(new_d, args...; type = Real)
    end

end # @static if isdefined(SymbolicUtils, :TypeT)

end # module
