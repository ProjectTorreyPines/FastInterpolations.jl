# ========================================
# Constant Interpolant Callable Methods
# ========================================
# Callable methods for ConstantInterpolant and 2-arg API.
# Type definition is in constant_types.jl.
# Internal evaluation and oneshot API (constant_interp!, constant_interp 3-arg)
# are in constant_oneshot.jl.

# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# ─────────────────────────────────────────────────────────────
@inline function (itp::ConstantInterpolant{T})(xi::T; deriv::Int=0, search=Binary(), hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat}
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        _constant_eval_at_point(itp.x, itp.y, xi, itp.mode, itp.side, op, searcher)
    end
end

# Real scalar wrapper - delegates to T method
@inline function (itp::ConstantInterpolant{T})(xi::S; deriv::Int=0, search=Binary(), hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); deriv=deriv, search=search, hint=hint)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T})(xi::AbstractVector{S}; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat, S<:Real}
    xi_typed = _to_float(xi, T)
    output = Vector{T}(undef, length(xi_typed))
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.mode, itp.side, op, searcher)
        end
    end
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call (zero allocation)
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi, itp.mode)
        @inbounds for i in eachindex(xi, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi[i], itp.mode, itp.side, op, searcher)
        end
    end
    return output
end

# In-place with type conversion
function (itp::ConstantInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0, search::AbstractSearchPolicy=Binary()) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = _to_float(xi, T)
    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = _constant_eval_at_point(itp.x, itp.y, xi_typed[i], itp.mode, itp.side, op, searcher)
        end
    end
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    constant_interp(x, y; extrap=:none, side=:nearest) -> ConstantInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values
- `extrap::Symbol`: Extrapolation mode
- `side::Symbol`: Side selection

# Returns
`ConstantInterpolant` object for scalar/broadcast evaluation.

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

itp = constant_interp(x, y)
itp(0.5)           # 10.0
itp.([0.5, 1.5])   # [10.0, 20.0]

# Fused broadcast (optimal)
result = @. coef * itp(query)

# Vector call with search policy
sorted_queries = sort(rand(1000))
vals = itp(sorted_queries; search=LinearBounded(max_steps=8))
```
"""
function constant_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT};
    extrap::Symbol=:none,
    side::Symbol=:nearest
) where {FT<:AbstractFloat}
    return ConstantInterpolant(x, y; extrap, side)
end

# ========================================
# 2-arg Callable Real → Float wrapper
# ========================================

function constant_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrap::Symbol=:none,
    side::Symbol=:nearest
) where {T<:Real}
    FT = float(T)
    return ConstantInterpolant(_to_float(x, FT), _to_float(y, FT); extrap, side)
end
