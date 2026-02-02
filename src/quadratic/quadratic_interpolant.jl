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
# Unified method: accepts any query type (Tg, Real, or Dual for AD)
# ─────────────────────────────────────────────────────────────
@inline function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(xq; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P}
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        # Pass original xq to preserve Dual type for AD
        _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xq, itp.extrap, op, searcher)
    end
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# Now supports hint for ODE/streaming patterns
# Output type is promoted to wider type for precision preservation
# Uses xi_search for domain check only; arithmetic uses original xi
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    xi_search = _to_float(xi, Tg)  # For domain check only
    T_out = promote_type(Tv, S)    # Lossless: wider type to avoid precision loss
    output = Vector{T_out}(undef, length(xi))
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_search, itp.extrap)
        @inbounds for i in eachindex(xi, output)
            # Use original xi[i] for arithmetic (preserves precision and Dual types)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi[i], itp.extrap, op, searcher)
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

# In-place with mixed types
# Uses xi_search for domain check only; arithmetic uses original xi
function (itp::QuadraticInterpolant{Tg,Tv,X,Y,P})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_search = _to_float(xi, Tg)  # For domain check only
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_search, itp.extrap)
        @inbounds for i in eachindex(xi, output)
            # Use original xi[i] for arithmetic (preserves precision and Dual types)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi[i], itp.extrap, op, searcher)
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
# Generic constructor (forwarding to outer constructor)
# Handles all Real grid types (Int, Float32, Float64, etc.)
# Type promotion is handled by QuadraticInterpolant outer constructor
quadratic_interp(x::AbstractVector{<:Real}, y::AbstractVector; bc::QuadraticBC=Left(QuadraticFit()), extrap::Symbol=:none, search::AbstractSearchPolicy=Binary()) =
    QuadraticInterpolant(x, y; bc, extrap, search)
