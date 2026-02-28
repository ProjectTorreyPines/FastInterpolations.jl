# ========================================
# Symbolic-Compatible Derivative Function
# ========================================
# Provides a `derivative(itp, t, order)` function that can be registered
# with Symbolics.jl for symbolic differentiation chain rules.
#
# This is a thin wrapper around the `deriv` keyword argument API.

"""
    derivative(itp::AbstractInterpolant, t, order::Integer=1)

Evaluate the `order`-th derivative of 1D interpolant `itp` at point `t`.

This function wraps the `deriv` keyword API (`itp(t; deriv=DerivOp(order))`)
in a form suitable for Symbolics.jl registration.

# Arguments
- `itp`: A 1D interpolant (any `AbstractInterpolant`)
- `t`: Evaluation point
- `order`: Derivative order (1, 2, or 3). Default: 1.

# Examples
```julia
itp = cubic_interp(x, y)
derivative(itp, 0.5)       # First derivative at 0.5
derivative(itp, 0.5, 2)    # Second derivative at 0.5
```

See also: [`deriv1`](@ref), [`deriv2`](@ref), [`deriv3`](@ref), [`DerivOp`](@ref)
"""
function derivative end

@inline function derivative(itp::AbstractInterpolant, t, order::Integer=1)
    itp(t; deriv=DerivOp(order))
end
