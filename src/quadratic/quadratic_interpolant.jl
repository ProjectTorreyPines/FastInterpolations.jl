# ========================================
# Quadratic Interpolant Callable Methods
# ========================================
# Callable methods for QuadraticInterpolant and 2-arg API.
# Type definition is in quadratic_types.jl.
# Internal evaluation and oneshot API (quadratic_interp!, quadratic_interp 3-arg)
# are in quadratic_oneshot.jl.

# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# ─────────────────────────────────────────────────────────────
@inline function (itp::QuadraticInterpolant{T})(xi::T; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat}
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi, itp.mode, op, searcher)
    end
end

# Real scalar wrapper - delegates to T method
@inline function (itp::QuadraticInterpolant{T})(xi::S; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); deriv=deriv, search=search)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{T})(xi::AbstractVector{S}; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat, S<:Real}
    xi_typed = _to_float(xi, T)
    output = Vector{T}(undef, length(xi_typed))
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi_typed[i], itp.mode, op, searcher)
        end
    end
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call (zero allocation)
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi, itp.mode)
        @inbounds for i in eachindex(xi, output)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi[i], itp.mode, op, searcher)
        end
    end
    return output
end

# In-place with type conversion
function (itp::QuadraticInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = _to_float(xi, T)
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xi_typed[i], itp.mode, op, searcher)
        end
    end
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    quadratic_interp(x, y; bc=Left(ParabolaFit()), extrap=:none) -> QuadraticInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values
- `bc`: Boundary condition (Left, Right, MinCurvFit, or Left/Right with ParabolaFit)
- `extrap::Symbol`: Extrapolation mode

# Returns
`QuadraticInterpolant` object for scalar/broadcast evaluation.

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = x.^2

# Default BC (ParabolaFit) gives exact polynomial reproduction
itp = quadratic_interp(x, y)
itp(1.5)           # 2.25 (exact)
itp.([0.5, 1.5])   # [0.25, 2.25]

# Fused broadcast (optimal)
result = @. coef * itp(query)

# Vector call with search policy
sorted_queries = sort(rand(1000))
vals = itp(sorted_queries; search=LinearBounded(max_steps=8))
```
"""
function quadratic_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT};
    bc::QuadraticBC{FT}=Left(ParabolaFit{FT}()),
    extrap::Symbol=:none
) where {FT<:AbstractFloat}
    return QuadraticInterpolant(x, y; bc, extrap)
end

# Real wrapper for 2-argument form
function quadratic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    bc::QuadraticBC{<:AbstractFloat}=Left(ParabolaFit{Float64}()),
    extrap::Symbol=:none
) where {T<:Real}
    FT = float(T)
    bc_promoted = _promote_bc(bc, FT)
    return QuadraticInterpolant(_to_float(x, FT), _to_float(y, FT); bc=bc_promoted, extrap)
end
