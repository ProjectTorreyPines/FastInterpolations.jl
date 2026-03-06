# ========================================
# DerivativeView: Wrapper for Broadcast Support
# ========================================
#
# This file contains:
# - DerivativeView struct for derivative evaluation
# - Factory functions for 1D and ND interpolants
# - Callable methods delegating to parent's deriv keyword
#
# The wrapper enables:
# - Broadcast: deriv1(itp).(xs)
# - HOF composition: map(deriv1(itp), xs)
# - Fused broadcast: @. coef * deriv1(itp)(xs)

# ========================================
# DerivativeView Struct
# ========================================

"""
    AbstractDerivativeView

Abstract supertype for all derivative view wrappers.

Useful for dispatch when you want to accept any derivative view regardless of order or interpolant type.

# Example
```julia
# Accept any derivative view
function integrate(d::AbstractDerivativeView, a, b)
    # works with deriv1, deriv2, deriv3 of any interpolant
end
```

See also: [`DerivativeView`](@ref), [`deriv1`](@ref), [`deriv2`](@ref), [`deriv3`](@ref), [`deriv_view`](@ref)
"""
abstract type AbstractDerivativeView end

"""
    DerivativeView{Order, ITP} <: AbstractDerivativeView

Lightweight callable wrapper for derivative evaluation with a fixed derivative order.
Enables broadcast and higher-order function composition.

# Type Parameters
- `Order`: Derivative order
  - `Int` for 1D derivatives (1, 2, or 3)
  - `NTuple{N,Int}` for ND partials (e.g., `(1,0)`)
- `ITP`: Parent interpolant type

# Construction
Use `deriv1(itp)`, `deriv2(itp)`, or `deriv3(itp)` to create instances for 1D.
Use `deriv_view(itp, (d1, d2, ...))` for ND mixed partials.

# Type Dispatch Patterns
```julia
# Match any derivative view
function foo(d::AbstractDerivativeView) ... end

# Match specific derivative order (any interpolant)
function foo(d::DerivativeView{1}) ... end

# Match specific order AND interpolant type
function foo(d::DerivativeView{1, <:CubicInterpolant}) ... end
```

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)     # DerivativeView{1, ...}
d2 = deriv2(itp)     # DerivativeView{2, ...}

# Broadcast evaluation
slopes = d1.(query_points)

# Fused broadcast (zero allocation)
result = @. coef * d1(xs) * other_func(xs)
```

See also: [`AbstractDerivativeView`](@ref), [`deriv1`](@ref), [`deriv2`](@ref), [`deriv3`](@ref), [`deriv_view`](@ref)
"""
struct DerivativeView{Order, ITP} <: AbstractDerivativeView
    parent::ITP
end

# ========================================
# Factory Functions
# ========================================

"""
    deriv1(itp::AbstractInterpolant)
    deriv2(itp::AbstractInterpolant)
    deriv3(itp::AbstractInterpolant)

    deriv_view(itp::AbstractInterpolant, order::Int)
    deriv_view(itp::AbstractInterpolantND, order::Int)
    deriv_view(itp::AbstractInterpolantND, order::NTuple{N,Int})
    deriv_view(itp::AbstractInterpolantND, ops::Tuple{Vararg{DerivOp, N}})

Create a callable, zero-allocation derivative view of the interpolant.

- 1D convenience: `deriv1/deriv2/deriv3` for the 1st/2nd/3rd derivatives.
- Generic: `deriv_view(itp, order)` for 1D derivative orders and ND mixed partials.
- DerivOp: `deriv_view(itp, DerivOp(1, 0))` for type-stable ND mixed partials.

`DerivativeView` is a lightweight wrapper that delegates all evaluation calls to the underlying interpolant
using the `deriv` keyword argument (e.g., `itp(xq; deriv=DerivOp(1))`). This enables a more functional syntax
without the overhead of full object copying.

# Keyword Arguments
All keyword arguments are forwarded to the parent interpolant. Common options include:
- `search`: Search policy for interval lookup (default: parent's `search_policy`)
- `hint::Union{Nothing,Base.RefValue{Int}}`: Mutable hint for sequential access patterns (default: `nothing`)

# Features
- **Functional API**: Pass derivative views into high-order functions like `map` or `find_zeros`.
- **Broadcast Fusion**: Enables evaluation without temporary allocations when combined with other operations using dot syntax (e.g., `@. d1(xs) + d2(xs)`).
- **Unified Interface**: Supports both single-series and multi-series (SeriesInterpolant) objects.
  - For `AbstractInterpolant`, returns a scalar derivative.
  - For `AbstractSeriesInterpolant`, returns a vector of derivatives.
- **In-place Evaluation**: Supports in-place evaluation for both single-series and multi-series interpolants.

# Notes
- **Order 3**: For cubic splines, the 3rd derivative is piecewise constant and may jump at knot points. For lower-order interpolants, it returns zero.

# Example
```julia
itp = cubic_interp(x, y)
d1 = deriv1(itp)

# Evaluate at a point
val = d1(0.5)

# Override search policy
val = d1(0.5; search=BinarySearch())

# Use hint for sequential access
hint = Ref(1)
for xq in sorted_queries
    val = d1(xq; hint=hint)
end

# Use in higher-order functions
using Roots
root = find_zero(d1, 0.5)

# Fused broadcast (no temporary allocations)
query_pts = range(0.0, 1.0, 100)
results = @. itp(query_pts) + 0.1 * d1(query_pts)

# In-place evaluation (single-series)
output = similar(query_pts)
d1(output, query_pts)
d1(output, query_pts; search=LinearBinarySearch())  # explicit: LinearBinarySearch for sorted vectors
```
"""
(deriv1, deriv2, deriv3)

@inline deriv1(itp::AbstractInterpolant) = DerivativeView{1, typeof(itp)}(itp)
@inline deriv2(itp::AbstractInterpolant) = DerivativeView{2, typeof(itp)}(itp)
@inline deriv3(itp::AbstractInterpolant) = DerivativeView{3, typeof(itp)}(itp)

"""
    deriv_view(itp::AbstractInterpolant, order::Int)
    deriv_view(itp::AbstractInterpolantND, order::Int)
    deriv_view(itp::AbstractInterpolantND, order::NTuple{N,Int})
    deriv_view(itp::AbstractInterpolantND, ops::Tuple{Vararg{DerivOp, N}})

Create a derivative view for 1D or N-dimensional interpolants.
- 1D: `order::Int` maps to `deriv=order`.
- ND: `order::Int` applies the same order to all axes (e.g., `1` → `(1,1,...,1)`), i.e. a mixed partial with order `order` along every axis.
- ND: `order::NTuple{N,Int}` specifies mixed partials.
- ND: `ops::Tuple{Vararg{DerivOp, N}}` specifies mixed partials via DerivOp tuples (e.g., `DerivOp(1, 0)`).
The ND view forwards `deriv=DerivOp(...)` to enable compile-time dispatch.

# Examples
```julia
itp = cubic_interp((x, y), data)

# In 2D, `order::Int` is shorthand for the mixed partial `(order, order)`.
dxy = deriv_view(itp, 1)              # same as (1, 1) => ∂²f/∂x∂y
dx = deriv_view(itp, (1, 0))
dy = deriv_view(itp, DerivOp(0, 1))   # DerivOp syntax

dx((0.5, 0.5))                         # ∂f/∂x
dxy.([(0.1, 0.2), (0.3, 0.4)])          # broadcast over points
```
"""
@inline function deriv_view(itp::AbstractInterpolant, order::Int)
    _make_derivop(order)  # validate order ∈ [0,3]
    return DerivativeView{order, typeof(itp)}(itp)
end

@inline function deriv_view(
        itp::AbstractInterpolantND{Tg, Tv, N},
        order::Int
    ) where {Tg, Tv, N}
    _make_derivop(order)  # validate order ∈ [0,3]
    return DerivativeView{ntuple(_ -> order, Val(N)), typeof(itp)}(itp)
end

@inline function deriv_view(
        itp::AbstractInterpolantND{Tg, Tv, N},
        order::Tuple{Int, Vararg{Int}}
    ) where {Tg, Tv, N}
    length(order) == N || _throw_ndims_mismatch("derivative orders", N, length(order))
    foreach(_make_derivop, order)  # validate each order ∈ [0,3]
    return DerivativeView{order, typeof(itp)}(itp)
end

@inline function deriv_view(
        itp::AbstractInterpolantND{Tg, Tv, N},
        ops::Tuple{DerivOp, Vararg{DerivOp}}
    ) where {Tg, Tv, N}
    length(ops) == N || _throw_ndims_mismatch("DerivOps", N, length(ops))
    order = ntuple(i -> deriv_order(ops[i]), Val(N))
    return DerivativeView{order, typeof(itp)}(itp)
end

# ND interpolants use tuple-based derivative API (via deriv_view)
@noinline function _nd_deriv_error(order::Int, N::Int)
    throw(
        ArgumentError(
            "deriv$order is not supported for $(N)D interpolants. " *
                "For N-dimensional interpolants, use:\n" *
                "  • deriv_view(itp, DerivOp(d1, d2, ...))  for mixed partial derivatives\n" *
                "  • itp(x; deriv=DerivOp(1,0,...))          for mixed partial derivatives\n" *
                "  • gradient(itp, x)                for ∇f\n" *
                "  • hessian(itp, x)                 for H(f)\n" *
                "  • laplacian(itp, x)               for ∇²f"
        )
    )
end

@inline deriv1(itp::AbstractInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = _nd_deriv_error(1, N)
@inline deriv2(itp::AbstractInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = _nd_deriv_error(2, N)
@inline deriv3(itp::AbstractInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} = _nd_deriv_error(3, N)

# ========================================
# Callable Methods (kwargs forwarding for future-proofing)
# ========================================
# Note: `deriv` keyword is captured and rejected - DerivativeView always uses
# its compile-time Order parameter. This check is zero-cost due to constant folding.

@inline _order_label(order::Int) = order == 1 ? "1st" : order == 2 ? "2nd" : order == 3 ? "3rd" : "$(order)th"
@inline _order_label(order::Tuple) = "mixed $(order)"

@inline _check_no_deriv_override(::Val, ::Nothing) = nothing
@noinline _check_no_deriv_override(::Val{Order}, ::Any) where {Order} = throw(
    ArgumentError(
        "This DerivativeView already evaluates the $(_order_label(Order)) derivative (deriv=$(Order)). " *
            "The `deriv` keyword argument is not accepted. " *
            "To evaluate a different derivative order, create a new view: deriv1/deriv2/deriv3 for 1D, " *
            "or deriv_view(itp, (d1, d2, ...)) for ND."
    )
)

@inline _deriv_kw(::Val{Order}) where {Order} = Order isa Tuple ? DerivOp(Order...) : DerivOp(Order)

# Out-of-place calls (Scalar or Vector)
@inline function (d::DerivativeView{Order, ITP})(
        xq::Union{Real, AbstractArray{<:Real}}; deriv = nothing, kwargs...
    ) where {Order, ITP}
    _check_no_deriv_override(Val(Order), deriv)
    return d.parent(xq; deriv = _deriv_kw(Val(Order)), kwargs...)
end

# ND queries with Real/AbstractArray (tie-breaker for 1D-vs-ND dispatch)
@inline function (d::DerivativeView{Order, ITP})(
        xq::Union{Real, AbstractArray{<:Real}}; deriv = nothing, kwargs...
    ) where {Order, ITP <: AbstractInterpolantND}
    _check_no_deriv_override(Val(Order), deriv)
    return d.parent(xq; deriv = _deriv_kw(Val(Order)), kwargs...)
end

# ND queries (tuples, SoA/AoS, etc.)
@inline function (d::DerivativeView{Order, ITP})(
        xq; deriv = nothing, kwargs...
    ) where {Order, ITP <: AbstractInterpolantND}
    _check_no_deriv_override(Val(Order), deriv)
    return d.parent(xq; deriv = _deriv_kw(Val(Order)), kwargs...)
end

# In-place vector query => vector output (single-series interpolants)
# Note: No element type constraint - parent handles type checking/conversion
@inline function (d::DerivativeView{Order, ITP})(
        output::AbstractVector, xq::AbstractVector{<:Real}; deriv = nothing, kwargs...
    ) where {Order, ITP}
    _check_no_deriv_override(Val(Order), deriv)
    return d.parent(output, xq; deriv = _deriv_kw(Val(Order)), kwargs...)
end

# In-place scalar query => array output (SeriesInterpolant)
# Note: No element type constraint - parent handles type checking/conversion
@inline function (d::DerivativeView{Order, ITP})(
        out::AbstractArray, xq::Real; deriv = nothing, kwargs...
    ) where {Order, ITP}
    _check_no_deriv_override(Val(Order), deriv)
    return d.parent(out, xq; deriv = _deriv_kw(Val(Order)), kwargs...)
end
