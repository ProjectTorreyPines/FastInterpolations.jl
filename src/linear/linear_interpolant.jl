# ========================================
# Linear Interpolant Callable Methods
# ========================================
# Callable methods for LinearInterpolant and 2-arg API.
# Type definition is in linear_types.jl.
# Oneshot API (linear_interp!, linear_interp 3-arg) is in linear_oneshot.jl.

# ========================================
# Scalar Call - Hot Path
# ========================================
# Supports deriv, search, and hint keywords for derivative evaluation and search policy.
# Default search is the stored policy in itp.search_policy.
#
# TYPE PARAMETERS:
# - Tg: Grid type (Float32/Float64)
# - Tv: Value type (Tg, Complex{Tg}, etc.)
# - Tq: Query type (Tg or any Real, including Dual for AD)

# Primary scalar call - accepts any query type (Tg, Real, or Dual for AD)
# This unified method handles:
# - Tg queries (hot path)
# - Int/Float32 queries (type promotion)
# - ForwardDiff.Dual queries (automatic differentiation)
@inline function (itp::LinearInterpolant{Tg,Tv,X,Y,P})(xq; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P}
    # For domain check, extract primal value (works for Float and Dual)
    xq_primal = _extract_primal(xq)
    @boundscheck _check_domain(itp.x, Tg(xq_primal), itp.extrap)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        # Pass original xq to preserve Dual type for AD
        _linear_with_extrap(itp.x, itp.y, xq, itp.extrap, op, searcher)
    end
end

# ========================================
# Vector Call - Allocating
# ========================================
# Output type is Tv (value type), not Tg (grid type).
# Handles type conversion via ternary (no-op when Tq === Tg).
function (itp::LinearInterpolant{Tg,Tv,X,Y,P})(xq::AbstractVector{Tq}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, Tq<:Real}
    xq_typed = Tq === Tg ? xq : Tg.(xq)
    @boundscheck _check_domain(itp.x, xq_typed, itp.extrap)
    output = Vector{Tv}(undef, length(xq_typed))
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq_typed, output)
            output[i] = _linear_with_extrap(itp.x, itp.y, xq_typed[i], itp.extrap, op, searcher)
        end
    end
    return output
end

# ========================================
# In-Place Vector Call
# ========================================
# Handles type conversion via _to_float (no-op when already Tg)
function (itp::LinearInterpolant{Tg,Tv,X,Y,P})(output::AbstractVector, xq::AbstractVector{Tq}; deriv::Int=0, search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, P, Tq<:Real}
    @assert length(output) == length(xq) "output length must match xq length"
    xq_typed = _to_float(xq, Tg)
    @boundscheck _check_domain(itp.x, xq_typed, itp.extrap)
    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq_typed, output)
            output[i] = _linear_with_extrap(itp.x, itp.y, xq_typed[i], itp.extrap, op, searcher)
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
- `y::AbstractVector`: y-values (can be real or complex)
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`
- `search::AbstractSearchPolicy`: Default search policy for interval lookup (default: `Binary()`)

# Type Handling
- x: Grid coordinates → converted to AbstractFloat, Range structure preserved
- y: Value type determines return type:
  - Real types → promoted to float(eltype(y))
  - Complex{Real} → promoted to Complex{Tg} where Tg = float(real(eltype(y)))
  - Already matching types → no conversion

# Returns
`LinearInterpolant{Tg, Tv}` object where:
- `Tg`: Grid type (Float32 or Float64)
- `Tv`: Value type (Tg for real, Complex{Tg} for complex)

Can be:
- Called with scalar: `itp(0.5)` (uses stored search policy)
- Called with search override: `itp(0.5; search=Binary())` (override stored policy)
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)`
- Reused multiple times without re-creating

# Examples
```julia
# Create with default Binary() search policy
itp = linear_interp(x_data, y_data)

# Create with LinearBinary() as default policy (optimal for sorted queries)
itp = linear_interp(x_data, y_data; search=LinearBinary())

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
- Use `search=LinearBinary()` for sorted query sequences
- Use `hint=Ref(idx)` for ODE/streaming patterns with persistent hint
"""
function linear_interp end

# ========================================
# Generic Constructor
# ========================================
# Handles all Real grid types (Int, Float32, Float64, etc.)
# _ensure_promoted_xy handles type promotion (no-op when already compatible)
function linear_interp(
    x::AbstractVector{Tx},
    y::AbstractVector{Ty};
    extrap::Symbol=:none,
    search::P=Binary()
) where {Tx<:Real, Ty, P<:AbstractSearchPolicy}
    x_p, y_p = _ensure_promoted_xy(x, y)
    return LinearInterpolant(x_p, y_p; extrap, search)
end
