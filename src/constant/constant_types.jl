# ========================================
# Constant Interpolant Types
# ========================================
# Type definition for ConstantInterpolant.
# Constructor and callable methods are in constant_interpolant.jl.
# Internal evaluation functions are in constant_oneshot.jl.

"""
    ConstantInterpolant{T,X,Y,P}

Lightweight callable interpolant for constant (step) interpolation.
Returned by `constant_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `extrap::ExtrapVal`: Extrapolation mode
- `side::SideVal`: Side selection (:nearest, :left, :right)
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = constant_interp(x, y)  # default: extrap=:none, side=:nearest
val = itp(0.5)               # scalar evaluation
vals = itp.(query_points)    # broadcast

# With options
itp_left = constant_interp(x, y; side=:left)
itp_wrap = constant_interp(x, y; extrap=:wrap, side=:right)

# Custom search policy
itp = constant_interp(x, y; search=LinearBinary())
val = itp(0.5)               # uses LinearBinary() by default
val = itp(0.5; search=Binary())  # override with Binary()
```
"""
struct ConstantInterpolant{T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}, P<:AbstractSearchPolicy} <: AbstractInterpolant{T}
    x::X
    y::Y
    extrap::ExtrapVal
    side::SideVal
    search_policy::P  # Default search policy (immutable, thread-safe)

    function ConstantInterpolant(
        x::X, y::Y;
        extrap::Symbol=:none,
        side::Symbol=:nearest,
        search::P=Binary()
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"

        @_dispatch_extrap extrap => ev begin
            @_dispatch_side side => sv begin
                return new{T,X,Y,P}(x, y, ev, sv, search)
            end
        end
    end
end
