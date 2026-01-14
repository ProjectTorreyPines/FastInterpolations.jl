# ========================================
# Linear Interpolant Callable Methods
# ========================================
# Callable methods for LinearInterpolant and 2-arg API.
# Type definition is in linear_types.jl.
# Oneshot API (linear_interp!, linear_interp 3-arg) is in linear_oneshot.jl.

# Scalar call - hot path (inlined for broadcast fusion)
# Supports deriv keyword for derivative evaluation
@inline function (itp::LinearInterpolant{T})(xq::T; deriv::Int=0) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    @_dispatch_deriv deriv => op begin
        _linear_with_extrap(itp.x, itp.y, xq, itp.mode, op)
    end
end

# Real scalar wrapper - delegates to T method with deriv keyword
@inline function (itp::LinearInterpolant{T})(xq::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xq); deriv=deriv)
end

# Vector call with deriv keyword support
function (itp::LinearInterpolant{T,X,Y})(xq::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, X, Y, S<:Real}
    xi_typed = S === T ? xq : T.(xq)
    @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
    output = Vector{T}(undef, length(xi_typed))
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.mode, op)
        end
    end
    return output
end

# Optimized path when xq element type matches T (zero conversion)
function (itp::LinearInterpolant{T,X,Y})(xq::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat, X, Y}
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    output = Vector{T}(undef, length(xq))
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq, output)
            output[i] = linear_interp(itp.x, itp.y, xq[i], itp.mode, op)
        end
    end
    return output
end

# In-place vector call with deriv keyword support - zero allocation
function (itp::LinearInterpolant{T,X,Y})(output::AbstractVector{T}, xq::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat, X, Y}
    @assert length(output) == length(xq) "output length must match xq length"
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq, output)
            output[i] = linear_interp(itp.x, itp.y, xq[i], itp.mode, op)
        end
    end
    return output
end

# In-place with type conversion and deriv keyword
function (itp::LinearInterpolant{T,X,Y})(output::AbstractVector, xq::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, X, Y, S<:Real}
    @assert length(output) == length(xq) "output length must match xq length"
    xi_typed = T.(xq)
    @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.mode, op)
        end
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    linear_interp(x, y; extrap=:none) -> LinearInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`

# Returns
`LinearInterpolant` object that can be:
- Called with scalar: `itp(0.5)`
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)`
- Reused multiple times without re-creating

# Examples
```julia
# Create once, reuse multiple times
itp = linear_interp(x_data, y_data)

# Scalar call
val = itp(0.5)

# Broadcast (creates array)
vals = itp.(query_points)

# Fused broadcast (optimal - no intermediate arrays)
result = @. coefficient * itp(rho) * ne / Te^2

# Wrap to domain (for periodic-like data)
itp_wrap = linear_interp(x_data, y_data; extrap=:wrap)
val_wrap = itp_wrap(2.5)  # wraps to domain

# Compare with 3-argument form (returns array immediately)
vals_direct = linear_interp(x_data, y_data, query_points)
```

# Performance Notes
- Returns lightweight callable (~48 bytes), best for reuse and broadcast fusion
- 3-argument form returns array immediately, best for single use
"""
function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    return LinearInterpolant(x, y; extrap)
end

# Real wrapper for 2-argument form (allows different container types)
# Uses _to_float from utils.jl to preserve Range structure
function linear_interp(
    x::X,
    y::Y;
    extrap::Symbol=:none
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    T = promote_type(TX, TY)
    FT = float(T)
    return LinearInterpolant(_to_float(x, FT), FT.(y); extrap)
end
