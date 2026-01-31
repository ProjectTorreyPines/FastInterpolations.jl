# ========================================
# Constant Interpolant Callable Methods
# ========================================
# Callable methods for ConstantInterpolant and 2-arg API.
# Type definition is in constant_types.jl.
# Internal evaluation and oneshot API (constant_interp!, constant_interp 3-arg)
# are in constant_oneshot.jl.

# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# Default search is now the stored policy in itp.search_policy
# Type parameters: Tg = grid type, Tv = value type (can be Complex)
# ─────────────────────────────────────────────────────────────
@inline function (itp::ConstantInterpolant{Tg,Tv,X,Y,P})(xi::Tg; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P}
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        _constant_eval_at_point(itp.x, itp.y, xi, itp.extrap, itp.side, op, searcher)
    end
end

# Real scalar wrapper - delegates to Tg method
@inline function (itp::ConstantInterpolant{Tg,Tv,X,Y,P})(xi::S; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    itp(Tg(xi); deriv=deriv, search=search, hint=hint)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# Now supports hint for ODE/streaming patterns
# Output type is Tv (value type), not Tg (grid type)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{Tg,Tv,X,Y,P})(xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    xi_typed = _to_float(xi, Tg)
    output = Vector{Tv}(undef, length(xi_typed))
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.extrap)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.extrap, itp.side, op, searcher)
        end
    end
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call (zero allocation)
# Output type is Tv (value type)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{Tg,Tv,X,Y,P})(output::AbstractVector{Tv}, xi::AbstractVector{Tg}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P}
    @assert length(output) == length(xi) "output length must match xi length"
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi, itp.extrap)
        @inbounds for i in eachindex(xi, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi[i], itp.extrap, itp.side, op, searcher)
        end
    end
    return output
end

# In-place with type conversion
function (itp::ConstantInterpolant{Tg,Tv,X,Y,P})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = _to_float(xi, Tg)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.extrap)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.extrap, itp.side, op, searcher)
        end
    end
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    constant_interp(x, y; extrap=:none, side=:nearest, search=Binary()) -> ConstantInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (can be Real or Complex)
- `extrap::Symbol`: Extrapolation mode
- `side::Symbol`: Side selection
- `search::AbstractSearchPolicy`: Default search policy (default: `Binary()`)

# Returns
`ConstantInterpolant{Tg, Tv}` object for scalar/broadcast evaluation.
- `Tg`: Grid type (Float32/Float64)
- `Tv`: Value type (Tg for real values, Complex{Tg} for complex values)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

itp = constant_interp(x, y)
itp(0.5)           # 10.0
itp.([0.5, 1.5])   # [10.0, 20.0]

# Complex values
x = [0.0, 1.0, 2.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im]
itp = constant_interp(x, y)
itp(0.5)           # 1.0+2.0im (ComplexF64)

# Create with custom search policy
itp = constant_interp(x, y; search=LinearBinary())
val = itp(0.5)     # uses LinearBinary() by default

# Fused broadcast (optimal)
result = @. coef * itp(query)

# Vector call with hint for ODE/streaming patterns
hint = Ref(1)
for batch in batches
    vals = itp(batch; hint=hint)
end
```
"""
# Hot path: x is AbstractFloat, y can be Tg or Complex{Tg}
function constant_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Auto-promote x/y types (zero allocation if already compatible)
    x_p, y_p = _ensure_promoted_xy(x, y)
    return ConstantInterpolant(x_p, y_p; extrap, side, search)
end

# ========================================
# 2-arg Generic Constructor (Type Promotion Wrapper)
# Handles: Int grid, Real values, Complex values
# ========================================

function constant_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    search::P=Binary()
) where {TX<:Real, TY, P<:AbstractSearchPolicy}
    x_p, y_p = _ensure_promoted_xy(x, y)
    return ConstantInterpolant(x_p, y_p; extrap, side, search)
end
