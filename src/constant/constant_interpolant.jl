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
# ─────────────────────────────────────────────────────────────
@inline function (itp::ConstantInterpolant{T,X,Y,P})(xi::T; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P}
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        _constant_eval_at_point(itp.x, itp.y, xi, itp.extrap, itp.side, op, searcher)
    end
end

# Real scalar wrapper - delegates to T method
@inline function (itp::ConstantInterpolant{T,X,Y,P})(xi::S; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P, S<:Real}
    itp(T(xi); deriv=deriv, search=search, hint=hint)
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# Now supports hint for ODE/streaming patterns
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T,X,Y,P})(xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P, S<:Real}
    xi_typed = _to_float(xi, T)
    output = Vector{T}(undef, length(xi_typed))
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
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{T,X,Y,P})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P}
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
function (itp::ConstantInterpolant{T,X,Y,P})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {T<:AbstractFloat, X, Y, P, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = _to_float(xi, T)
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
- `y::AbstractVector`: y-values
- `extrap::Symbol`: Extrapolation mode
- `side::Symbol`: Side selection
- `search::AbstractSearchPolicy`: Default search policy (default: `Binary()`)

# Returns
`ConstantInterpolant` object for scalar/broadcast evaluation.

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

itp = constant_interp(x, y)
itp(0.5)           # 10.0
itp.([0.5, 1.5])   # [10.0, 20.0]

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
function constant_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    search::P=Binary()
) where {FT<:AbstractFloat, P<:AbstractSearchPolicy}
    return ConstantInterpolant(x, y; extrap, side, search)
end

# ========================================
# 2-arg Callable Real → Float wrapper
# ========================================

function constant_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrap::Symbol=:none,
    side::Symbol=:nearest,
    search::P=Binary()
) where {T<:Real, P<:AbstractSearchPolicy}
    FT = float(T)
    return ConstantInterpolant(_to_float(x, FT), _to_float(y, FT); extrap, side, search)
end
