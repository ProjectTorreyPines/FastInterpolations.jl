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
@inline function (itp::LinearInterpolant{Tg,Tv,X,Y,E,P})(xq; deriv::DerivOp=EvalValue(), search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, E, P}
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    resolved = _resolve_search(search, xq)
    searcher = _to_searcher(resolved, hint)
    # Pass original xq to preserve Dual type for AD
    _linear_with_extrap(itp.x, itp.y, xq, itp.extrap, deriv, searcher)
end

# ========================================
# Vector Call - Allocating
# ========================================
# Output type is promoted to wider type for precision preservation.
function (itp::LinearInterpolant{Tg,Tv,X,Y,E,P})(xq::AbstractVector{Tq}; deriv::DerivOp=EvalValue(), search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, E, P, Tq<:Real}
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    T_out = promote_type(Tv, Tq)   # Lossless: wider type to avoid precision loss
    output = Vector{T_out}(undef, length(xq))
    resolved = _resolve_search(search, xq)
    searcher = _to_searcher(resolved, hint)
    @inbounds for i in eachindex(xq, output)
        output[i] = _linear_with_extrap(itp.x, itp.y, xq[i], itp.extrap, deriv, searcher)
    end
    return output
end

# ========================================
# In-Place Vector Call
# ========================================
function (itp::LinearInterpolant{Tg,Tv,X,Y,E,P})(output::AbstractVector, xq::AbstractVector{Tq}; deriv::DerivOp=EvalValue(), search=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, X, Y, E, P, Tq<:Real}
    @assert length(output) == length(xq) "output length must match xq length"
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    resolved = _resolve_search(search, xq)
    searcher = _to_searcher(resolved, hint)
    @inbounds for i in eachindex(xq, output)
        output[i] = _linear_with_extrap(itp.x, itp.y, xq[i], itp.extrap, deriv, searcher)
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    linear_interp(x, y; extrap=NoExtrap(), search=AutoSearch()) -> LinearInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values (can be real or complex)
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ConstExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `search::AbstractSearchPolicy`: Default search policy for interval lookup (default: `AutoSearch()`)

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
# Create with default AutoSearch() search policy
itp = linear_interp(x_data, y_data)

# Default AutoSearch: scalar→Binary, vector→LinearBinary
itp = linear_interp(x_data, y_data)

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
itp_wrap = linear_interp(x_data, y_data; extrap=WrapExtrap())
val_wrap = itp_wrap(2.5)  # wraps to domain

# Compare with 3-argument form (returns array immediately)
vals_direct = linear_interp(x_data, y_data, query_points)
```

# Performance Notes
- Returns lightweight callable (~56 bytes), best for reuse and broadcast fusion
- 3-argument form returns array immediately, best for single use
- Default `AutoSearch()` adapts: scalar→`Binary()`, vector→`LinearBinary()`
- Use `search=LinearBinary()` to force linear-binary for all query types
- Use `hint=Ref(idx)` for ODE/streaming patterns with persistent hint
"""
function linear_interp end

# ========================================
# Generic Constructor (User API)
# ========================================
# Handles all Real grid types (Int, Float32, Float64, etc.)
# Type promotion done here, then forwards to typed LinearInterpolant constructor.
#
# PERFORMANCE: Typed signature enables compile-time specialization.
# _promote_itp_inputs becomes no-op when types already match (Float64 → Float64).
@inline function linear_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY};
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {TX<:Real, TY}
    x_p, y_p = _promote_itp_inputs(x, y)
    return LinearInterpolant(x_p, y_p; extrap, search)
end
