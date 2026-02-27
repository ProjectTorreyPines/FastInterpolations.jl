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
@inline function (itp::QuadraticInterpolant{Tg,Tv})(xq; deriv::DerivOp=EvalValue(), search::AbstractSearchPolicy=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv}
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    searcher = _resolve_search(itp.x, xq, search, hint)
    # Pass original xq to preserve Dual type for AD
    _quadratic_eval_at_point(itp.x, itp.y, itp.h, itp.a, itp.d, xq, itp.extrap, deriv, searcher)
end

# ─────────────────────────────────────────────────────────────
# Vector loop (function barrier)
# Julia specializes on concrete Searcher type P, eliminating Union-split
# overhead when adaptive AutoSearch resolves to Binary or LinearBinary.
# CRITICAL: All arguments must be fully typed — untyped args prevent SROA
# of RefHint's Ref, causing 16-byte heap allocation per call.
# ─────────────────────────────────────────────────────────────
@inline function _quadratic_vector_loop!(
    output::AbstractVector,
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    h::AbstractVector{Tg},
    a::AbstractVector{Tv},
    d::AbstractVector{Tv},
    xq::AbstractVector{<:Real},
    extrap::E,
    deriv::O,
    searcher::P
) where {Tg<:AbstractFloat, Tv, E<:AbstractExtrap, O<:AbstractEvalOp, P<:Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    @inbounds for i in eachindex(xq, output)
        output[i] = _quadratic_eval_at_point(x, y, h, a, d, xq[i], extrap, deriv, searcher)
    end
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# Now supports hint for ODE/streaming patterns
# Output type is promoted to wider type for precision preservation
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{Tg,Tv})(xi::AbstractVector{S}; deriv::DerivOp=EvalValue(), search::AbstractSearchPolicy=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, S<:Real}
    T_out = promote_type(Tv, S)    # Lossless: wider type to avoid precision loss
    output = Vector{T_out}(undef, length(xi))
    searcher = _resolve_search(itp.x, xi, search, hint)
    _quadratic_vector_loop!(output, itp.x, itp.y, itp.h, itp.a, itp.d, xi, itp.extrap, deriv, searcher)
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call
# Unified: accepts any Real query type (Tg, Float32, Dual, etc.)
# ─────────────────────────────────────────────────────────────
function (itp::QuadraticInterpolant{Tg,Tv})(output::AbstractVector, xi::AbstractVector{<:Real}; deriv::DerivOp=EvalValue(), search::AbstractSearchPolicy=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv}
    @assert length(output) == length(xi) "output length must match xi length"
    searcher = _resolve_search(itp.x, xi, search, hint)
    _quadratic_vector_loop!(output, itp.x, itp.y, itp.h, itp.a, itp.d, xi, itp.extrap, deriv, searcher)
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    quadratic_interp(x, y; bc=Left(QuadraticFit()), extrap=NoExtrap(), search=AutoSearch()) -> QuadraticInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (can be Real or Complex)
- `bc`: Boundary condition (Left, Right, MinCurvFit, or Left/Right with QuadraticFit)
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ConstExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

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

# Search policy: AutoSearch adapts to query type (scalar→Binary, vector→LinearBinary)
itp = quadratic_interp(x, y)
val = itp(0.5)     # AutoSearch resolves to Binary() for scalar
itp = quadratic_interp(x, y; search=LinearBinary())  # explicit override

# Fused broadcast (optimal)
result = @. coef * itp(query)

# Vector call with hint for ODE/streaming patterns
hint = Ref(1)
for batch in batches
    vals = itp(batch; hint=hint)
end
```
"""
# ========================================
# Generic Constructor (User API)
# ========================================
# Handles all Real grid types (Int, Float32, Float64, etc.)
# Type promotion, BC conversion, and coefficient computation done here,
# then forwards to typed QuadraticInterpolant constructor.
#
# PERFORMANCE: Typed signature enables compile-time specialization.
# _promote_itp_inputs becomes no-op when types already match (Float64 → Float64).
@inline function quadratic_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY};
    bc::QuadraticBC=Left(QuadraticFit()),
    extrap::AbstractExtrap=NoExtrap(),
    search::AbstractSearchPolicy=AutoSearch()
) where {TX<:Real, TY}
    x_p, y_p = _promote_itp_inputs(x, y)
    bc_p = _promote_bc(bc, eltype(x_p))

    # Validate PolyFit{D} point requirements (e.g., CubicFit needs 4+ points)
    validate_polyfit_points(bc_p, length(x_p))

    # Compute coefficients (h::Tg, d::Tv, a::Tv)
    h, d, a = _compute_quadratic_coeffs(x_p, y_p, bc_p)

    return QuadraticInterpolant(x_p, y_p, h, a, d; extrap, search)
end
