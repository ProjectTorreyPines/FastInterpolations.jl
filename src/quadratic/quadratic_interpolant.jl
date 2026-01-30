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
    bc::QuadraticBC{Tg}=Left(QuadraticFit{Tg}()),
    extrap::Symbol=:none,
    search::P=Binary()
) where {Tg<:AbstractFloat, Tv, P<:AbstractSearchPolicy}
    # Check if Tv's real part requires promotion of Tg
    Tv_real = _real_eltype(Tv)
    if Tv_real !== Tg && Tv_real <: AbstractFloat
        # Promote Tg to match the wider value type
        Tg_new = promote_type(Tg, Tv_real)
        x_typed = _to_float(x, Tg_new)
        _, y_typed = _promote_value_type(y, Tg_new)
        bc_promoted = _promote_bc(bc, Tg_new)
        return QuadraticInterpolant(x_typed, y_typed; bc=bc_promoted, extrap, search)
    end
    # No promotion needed - types are compatible
    return QuadraticInterpolant(x, y; bc, extrap, search)
end

# ========================================
# 2-arg Generic Constructor (Type Promotion Wrapper)
# Handles: Int grid, Real values, Complex values
# ========================================

function quadratic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY};
    bc::QuadraticBC{<:AbstractFloat}=Left(QuadraticFit{Float64}()),
    extrap::Symbol=:none,
    search::P=Binary()
) where {TX<:Real, TY, P<:AbstractSearchPolicy}
    # Determine grid type from x and real part of y
    Tg = float(promote_type(TX, _real_eltype(TY)))
    x_typed = _to_float(x, Tg)

    # Promote y to appropriate type (handles both Real and Complex)
    _, y_typed = _promote_value_type(y, Tg)

    # Promote BC to grid type
    bc_promoted = _promote_bc(bc, Tg)

    return QuadraticInterpolant(x_typed, y_typed; bc=bc_promoted, extrap, search)
end
