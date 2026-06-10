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
using FastInterpolations: AbstractInterpolant, AbstractInterpolantND,
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
# 1D Interpolant Registration
# ========================================

# Wrapper function for symbolic registration. Using a regular function
# (not a callable struct) avoids method ambiguity with concrete interpolant types.
_fast_interp_eval(itp::AbstractInterpolant, t) = itp(t)
_derivative_symbolic(itp::AbstractInterpolant, t, order::Integer) = itp(t; deriv = DerivOp(order))

# Register the wrapper and derivative functions with Symbolics
@register_symbolic _fast_interp_eval(itp::AbstractInterpolant, t)
@register_symbolic _derivative_symbolic(itp::AbstractInterpolant, t, order::Integer) false

Base.nameof(itp::AbstractInterpolant) = :FastInterpolation

# Type/shape promotions for _derivative_symbolic
function SymbolicUtils.promote_symtype(
        ::typeof(_derivative_symbolic), Ti::SymbolicUtils.TypeT,
        Tt::SymbolicUtils.TypeT,
        To::SymbolicUtils.TypeT
    )
    @assert Ti <: AbstractInterpolant
    @assert Tt <: Real
    @assert To <: Integer
    return Real
end

function SymbolicUtils.promote_shape(
        ::typeof(_derivative_symbolic),
        @nospecialize(shi::SymbolicUtils.ShapeT),
        @nospecialize(sht::SymbolicUtils.ShapeT),
        @nospecialize(sho::SymbolicUtils.ShapeT)
    )
    @assert !SymbolicUtils.is_array_shape(shi)
    @assert !SymbolicUtils.is_array_shape(sht)
    @assert !SymbolicUtils.is_array_shape(sho)
    return SymbolicUtils.ShapeVecT()
end

# Derivative chain rules:
# d/dt _fast_interp_eval(itp, t) = _derivative_symbolic(itp, t, 1)
@register_derivative _fast_interp_eval(itp, t) 2 _derivative_symbolic(itp, t, 1)
# d/dt _derivative_symbolic(itp, t, n) = _derivative_symbolic(itp, t, n+1)
@register_derivative _derivative_symbolic(itp, t, ord) 2 _derivative_symbolic(itp, t, ord + 1)

# Redirect concrete interpolant callable methods to the registered wrapper.
# This is needed because concrete types have (itp::ConcreteType)(xq; ...) methods
# where xq is untyped, creating ambiguity if we defined on AbstractInterpolant.
for T in [LinearInterpolant, CubicInterpolant, QuadraticInterpolant, ConstantInterpolant]
    for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
        @eval function (itp::$T)(t::$symT; kwargs...)
            return _fast_interp_eval(itp, t)
        end
    end
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

# Helper for ND symbolic term construction
function _symbolic_nd_call(itp, t_args, is_num::Bool)
    args = is_num ? unwrap.(t_args) : t_args
    res = SymbolicUtils.term(itp, args...; type = Real)
    return is_num ? Num(res) : res
end

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
            return _symbolic_nd_call(itp, t, $is_num)
        end

        # Varargs form: itp(sym_x, sym_y, ...)
        @eval function (itp::$NDT{Tg, Tv, N})(
                t::Vararg{$symT, N}; kwargs...
            ) where {Tg, Tv, N}
            return _symbolic_nd_call(itp, t, $is_num)
        end
    end
end

# DifferentiatedInterpolantND symbolic calls
for symT in [Num, SymbolicUtils.BasicSymbolic{<:Real}]
    is_num = symT === Num
    @eval function (d::DifferentiatedInterpolantND{N})(
            t::Vararg{$symT, N}
        ) where {N}
        return _symbolic_nd_call(d, t, $is_num)
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
# Must use the concrete ShapeT union type to avoid ambiguity with the generic fallback.
const _ShapeT = SymbolicUtils.ShapeT

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
