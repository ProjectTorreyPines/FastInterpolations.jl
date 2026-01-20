# ========================================
# Linear Interpolant Callable Methods
# ========================================
# Callable methods for LinearInterpolant and 2-arg API.
# Type definition is in linear_types.jl.
# Oneshot API (linear_interp!, linear_interp 3-arg) is in linear_oneshot.jl.

# Scalar call - hot path (inlined for broadcast fusion)
# Supports deriv, search, and hint keywords for derivative evaluation and search policy
# Default search is now the stored policy in itp.search_policy
@inline function (itp::LinearInterpolant{T,X,Y,P})(xq::T; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P}
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        _linear_with_extrap(itp.x, itp.y, xq, itp.mode, op, searcher)
    end
end

# Real scalar wrapper - delegates to T method with deriv keyword
@inline function (itp::LinearInterpolant{T,X,Y,P})(xq::S; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P, S<:Real}
    itp(T(xq); deriv=deriv, search=search, hint=hint)
end

# Vector call with deriv and search keyword support
# Now supports hint for ODE/streaming patterns - hint is updated during loop
function (itp::LinearInterpolant{T,X,Y,P})(xq::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P, S<:Real}
    xi_typed = S === T ? xq : T.(xq)
    @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
    output = Vector{T}(undef, length(xi_typed))
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.mode, op, searcher)
        end
    end
    return output
end

# Optimized path when xq element type matches T (zero conversion)
function (itp::LinearInterpolant{T,X,Y,P})(xq::AbstractVector{T}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P}
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    output = Vector{T}(undef, length(xq))
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq, output)
            output[i] = linear_interp(itp.x, itp.y, xq[i], itp.mode, op, searcher)
        end
    end
    return output
end

# In-place vector call with deriv and search keyword support - zero allocation
function (itp::LinearInterpolant{T,X,Y,P})(output::AbstractVector{T}, xq::AbstractVector{T}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P}
    @assert length(output) == length(xq) "output length must match xq length"
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq, output)
            output[i] = linear_interp(itp.x, itp.y, xq[i], itp.mode, op, searcher)
        end
    end
    return output
end

# In-place with type conversion and deriv keyword
function (itp::LinearInterpolant{T,X,Y,P})(output::AbstractVector, xq::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P, S<:Real}
    @assert length(output) == length(xq) "output length must match xq length"
    xi_typed = T.(xq)
    @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.mode, op, searcher)
        end
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    linear_interp(x, y; extrap=:none, search=Binary()) -> LinearInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`
- `search::AbstractSearchPolicy`: Default search policy for interval lookup (default: `Binary()`)

# Returns
`LinearInterpolant` object that can be:
- Called with scalar: `itp(0.5)` (uses stored search policy)
- Called with search override: `itp(0.5; search=Binary())` (override stored policy)
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)`
- Reused multiple times without re-creating

# Examples
```julia
# Create with default Binary() search policy
itp = linear_interp(x_data, y_data)

# Create with LinearBounded() as default policy (optimal for sorted queries)
itp = linear_interp(x_data, y_data; search=LinearBounded())

# Scalar call (uses stored policy)
val = itp(0.5)

# Scalar call with search policy override
val = itp(0.5; search=Binary())

# Vector call with hint for ODE/streaming patterns
hint = Ref(1)
for batch in batches
    vals = itp(batch; hint=hint)  # hint persists across calls
end

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
- Returns lightweight callable (~56 bytes), best for reuse and broadcast fusion
- 3-argument form returns array immediately, best for single use
- Use `search=LinearBounded()` for sorted query sequences
- Use `hint=Ref(idx)` for ODE/streaming patterns with persistent hint
"""
function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrap::Symbol=:none,
    search::P=Binary()
) where {T<:AbstractFloat, P<:AbstractSearchPolicy}
    return LinearInterpolant(x, y; extrap, search)
end

# Real wrapper for 2-argument form (allows different container types)
# Uses _to_float from utils.jl to preserve Range structure
function linear_interp(
    x::X,
    y::Y;
    extrap::Symbol=:none,
    search::P=Binary()
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}, P<:AbstractSearchPolicy}
    T = promote_type(TX, TY)
    FT = float(T)
    return LinearInterpolant(_to_float(x, FT), FT.(y); extrap, search)
end
