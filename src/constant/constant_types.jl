# ========================================
# Constant Interpolant Types
# ========================================
# Type definition for ConstantInterpolant.
# Constructor and callable methods are in constant_interpolant.jl.
# Internal evaluation functions are in constant_oneshot.jl.

"""
    ConstantInterpolant{T,X,Y}

Lightweight callable interpolant for constant (step) interpolation.
Returned by `constant_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `mode::ExtrapVal`: Extrapolation mode
- `side::SideVal`: Side selection (:nearest, :left, :right)

# Usage
```julia
itp = constant_interp(x, y)  # default: extrap=:none, side=:nearest
val = itp(0.5)               # scalar evaluation
vals = itp.(query_points)    # broadcast

# With options
itp_left = constant_interp(x, y; side=:left)
itp_wrap = constant_interp(x, y; extrap=:wrap, side=:right)
```
"""
struct ConstantInterpolant{T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}} <: AbstractInterpolant{T}
    x::X
    y::Y
    mode::ExtrapVal
    side::SideVal

    function ConstantInterpolant(
        x::X, y::Y;
        extrap::Symbol=:none,
        side::Symbol=:nearest
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"

        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                return new{T,X,Y}(x, y, ev, sv)
            end
        end
    end
end
