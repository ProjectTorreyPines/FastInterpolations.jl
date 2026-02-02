# ========================================
# Constant Interpolant Types
# ========================================
# Type definition for ConstantInterpolant.
# Constructor and callable methods are in constant_interpolant.jl.
# Internal evaluation functions are in constant_oneshot.jl.

"""
    ConstantInterpolant{Tg,Tv,X,Y,P}

Lightweight callable interpolant for constant (step) interpolation.
Returned by `constant_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg<:AbstractFloat`: Grid type (Float32, Float64) for x-coordinates
- `Tv`: Value type for y-values (can be Tg, Complex{Tg}, or other Number)
- `X<:AbstractVector{Tg}`: Type of x-coordinates
- `Y<:AbstractVector{Tv}`: Type of y-values
- `P<:AbstractSearchPolicy`: Search policy type

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

# With Complex values
x = [0.0, 1.0, 2.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im]
itp = constant_interp(x, y)
val = itp(0.5)  # returns ComplexF64

# With options
itp_left = constant_interp(x, y; side=:left)
itp_wrap = constant_interp(x, y; extrap=:wrap, side=:right)

# Custom search policy
itp = constant_interp(x, y; search=LinearBinary())
val = itp(0.5)               # uses LinearBinary() by default
val = itp(0.5; search=Binary())  # override with Binary()
```
"""
struct ConstantInterpolant{Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy} <: AbstractInterpolant{Tg, Tv}
    x::X
    y::Y
    extrap::ExtrapVal
    side::SideVal
    search_policy::P  # Default search policy (immutable, thread-safe)

    # Inner constructor: parametric, only calls new (handles validation only)
    function ConstantInterpolant{Tg,Tv,X,Y,P}(
        x::X, y::Y, ev::ExtrapVal, sv::SideVal, search::P
    ) where {Tg<:AbstractFloat, Tv, X<:AbstractVector{Tg}, Y<:AbstractVector{Tv}, P<:AbstractSearchPolicy}
        @assert length(x) == length(y) "x and y must have same length"
        @assert length(x) >= 2 "x must have at least 2 elements"
        new{Tg,Tv,X,Y,P}(x, y, ev, sv, search)
    end
end

# ========================================
# Outer Constructor: handles all logic
# ========================================
# - Type conversion (_promote_itp_inputs)
# - Symbol → Val dispatch
# - Call inner constructor
function ConstantInterpolant(
    x::AbstractVector,
    y::AbstractVector;
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    search::AbstractSearchPolicy=Binary()
)
    x_p, y_p = _promote_itp_inputs(x, y)
    X = typeof(x_p)
    Y = typeof(y_p)
    Tg = eltype(x_p)
    Tv = eltype(y_p)
    P = typeof(search)

    @_dispatch_extrap extrap => ev begin
        @_dispatch_side side => sv begin
            return ConstantInterpolant{Tg,Tv,X,Y,P}(x_p, y_p, ev, sv, search)
        end
    end
end
