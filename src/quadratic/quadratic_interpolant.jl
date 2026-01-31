# ========================================
# Quadratic Interpolant Callable Methods
# ========================================
# Callable methods for QuadraticInterpolant and 2-arg API.
# Type definition is in quadratic_types.jl.
# Internal evaluation and oneshot API (quadratic_interp!, quadratic_interp 3-arg)
# are in quadratic_oneshot.jl.

# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# Default search is now the stored policy in itp.search_policy
# Type parameters: Tg = grid type, Tv = value type (can be Complex)
# ─────────────────────────────────────────────────────────────
@inline function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(xi::Tg; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P}
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi, itp.extrap, op, searcher)
    end
end

# Real scalar wrapper - delegates to Tg method
@inline function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(xi::S; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    itp(Tg(xi); deriv=deriv, search=search, hint=hint)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# Now supports hint for ODE/streaming patterns
# Output type is Tv (value type), not Tg (grid type)
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    xi_typed = _to_float(xi, Tg)
    output = Vector{Tv}(undef, length(xi_typed))
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.extrap)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi_typed[i], itp.extrap, op, searcher)
        end
    end
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call (zero allocation)
# Output type is Tv (value type)
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(output::AbstractVector{Tv}, xi::AbstractVector{Tg}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P}
    @assert length(output) == length(xi) "output length must match xi length"
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi, itp.extrap)
        @inbounds for i in eachindex(xi, output)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi[i], itp.extrap, op, searcher)
        end
    end
    return output
end

# In-place with type conversion
function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = _to_float(xi, Tg)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.extrap)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi_typed[i], itp.extrap, op, searcher)
        end
    end
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    quadratic_interp(x, y; bc=Left(QuadraticFit()), extrap=:none, search=Binary()) -> QuadraticInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (can be Real or Complex)
- `bc`: Boundary condition (Left, Right, MinCurvFit, or Left/Right with QuadraticFit)
- `extrap::Symbol`: Extrapolation mode
- `search::AbstractSearchPolicy`: Default search policy (default: `Binary()`)

# Returns
`QuadraticInterpolant{Tg, Tv}` object for scalar/broadcast evaluation.
- `Tg`: Grid type (Float32/Float64)
- `Tv`: Value type (Tg for real values, Complex{Tg} for complex values)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2

# Default BC (QuadraticFit) gives exact polynomial reproduction
itp = quadratic_interp(x, y)
itp(1.5)           # 2.25 (exact)
itp.([0.5, 1.5])   # [0.25, 2.25]

# Complex values
x = [0.0, 1.0, 2.0, 3.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]
itp = quadratic_interp(x, y)
itp(0.5)           # returns ComplexF64

# Create with custom search policy
itp = quadratic_interp(x, y; search=LinearBinary())
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
function quadratic_interp(
    x::AbstractVector{Tg},
    y::AbstractVector{Tv};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::Symbol=:none,
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Auto-promote x/y types (zero allocation if already compatible)
    x_p, y_p = _ensure_promoted_xy(x, y)
    bc_promoted = _promote_bc(bc, eltype(x_p))
    return QuadraticInterpolant(x_p, y_p; bc=bc_promoted, extrap, search)
end

# ========================================
# 2-arg Generic Constructor (Type Promotion Wrapper)
# Handles: Int grid, Real values, Complex values
# ========================================

function quadratic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::Symbol=:none,
    search::P=Binary()
) where {TX<:Real, TY, P<:AbstractSearchPolicy}
    x_p, y_p = _ensure_promoted_xy(x, y)
    bc_promoted = _promote_bc(bc, eltype(x_p))
    return QuadraticInterpolant(x_p, y_p; bc=bc_promoted, extrap, search)
end
