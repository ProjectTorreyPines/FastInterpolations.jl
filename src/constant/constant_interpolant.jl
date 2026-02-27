# ========================================
# Constant Interpolant Callable Methods
# ========================================
# Callable methods for ConstantInterpolant and 2-arg API.
# Type definition is in constant_types.jl.
# Internal evaluation and oneshot API (constant_interp!, constant_interp 3-arg)
# are in constant_oneshot.jl.

# ─────────────────────────────────────────────────────────────
# Scalar call - hot path (inlined for broadcast fusion)
# AD Support: xq can be any Real (including ForwardDiff.Dual)
# Type parameters: Tg = grid type, Tv = value type, Tq = query type
# ─────────────────────────────────────────────────────────────
@inline function (itp::ConstantInterpolant{Tg,Tv})(xq::Tq; deriv::DerivOp=EvalValue(), search::AbstractSearchPolicy=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    searcher = _resolve_search(itp.x, xq, search, hint)
    _constant_eval_at_point(itp.x, itp.y, xq, itp.extrap, itp.side, deriv, searcher)
end

# ─────────────────────────────────────────────────────────────
# Vector loop (function barrier)
# Julia specializes on concrete Searcher type P, eliminating Union-split
# overhead when adaptive AutoSearch resolves to BinarySearch or LinearBinarySearch.
# CRITICAL: All arguments must be fully typed — untyped args prevent SROA
# of RefHint's Ref, causing 16-byte heap allocation per call.
# ─────────────────────────────────────────────────────────────
@inline function _constant_vector_loop!(
    output::AbstractVector,
    x::AbstractVector{Tg},
    y::AbstractVector{Tv},
    xq::AbstractVector{<:Real},
    extrap::E,
    side::SD,
    deriv::O,
    searcher::P
) where {Tg<:AbstractFloat, Tv, E<:AbstractExtrap, SD<:AbstractSide, O<:AbstractEvalOp, P<:Searcher}
    @inbounds for i in eachindex(xq, output)
        output[i] = _constant_eval_at_point(x, y, xq[i], extrap, side, deriv, searcher)
    end
end

# ─────────────────────────────────────────────────────────────
# Vector call (allocating)
# Supports hint for ODE/streaming patterns
# Output type is promoted to wider type for precision preservation
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{Tg,Tv})(xq::AbstractVector{Tq}; deriv::DerivOp=EvalValue(), search::AbstractSearchPolicy=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    T_out = promote_type(Tv, Tq)   # Lossless: wider type to avoid precision loss
    output = Vector{T_out}(undef, length(xq))
    searcher = _resolve_search(itp.x, xq, search, hint)
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    _constant_vector_loop!(output, itp.x, itp.y, xq, itp.extrap, itp.side, deriv, searcher)
    return output
end

# ─────────────────────────────────────────────────────────────
# In-place vector call
# ─────────────────────────────────────────────────────────────
function (itp::ConstantInterpolant{Tg,Tv})(output::AbstractVector, xq::AbstractVector{Tq}; deriv::DerivOp=EvalValue(), search::AbstractSearchPolicy=itp.search_policy, hint::Union{Nothing,Base.RefValue{Int}}=nothing) where {Tg<:AbstractFloat, Tv, Tq<:Real}
    @assert length(output) == length(xq) "output length must match xq length"
    searcher = _resolve_search(itp.x, xq, search, hint)
    @boundscheck _check_domain(itp.x, xq, itp.extrap)
    _constant_vector_loop!(output, itp.x, itp.y, xq, itp.extrap, itp.side, deriv, searcher)
    return output
end


# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    constant_interp(x, y; extrap=NoExtrap(), side=NearestSide(), search=AutoSearch()) -> ConstantInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values (can be Real or Complex)
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ConstExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `side::AbstractSide`: Side selection (NearestSide(), LeftSide(), RightSide())
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Returns
`ConstantInterpolant{Tg, Tv}` object for scalar/broadcast evaluation.
- `Tg`: Grid type (Float32/Float64)
- `Tv`: Value type (Tg for real values, Complex{Tg} for complex values)

# Example
```julia
x = [0.0, 1.0, 2.0, 3.0]
y = [10.0, 20.0, 30.0, 40.0]

itp = constant_interp(x, y)
itp(0.5)           # 10.0
itp.([0.5, 1.5])   # [10.0, 20.0]

# Complex values
x = [0.0, 1.0, 2.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im]
itp = constant_interp(x, y)
itp(0.5)           # 1.0+2.0im (ComplexF64)

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = constant_interp(x, y)
val = itp(0.5)     # AutoSearch resolves to BinarySearch() for scalar
itp = constant_interp(x, y; search=LinearBinarySearch())  # explicit override

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
# Type promotion done here, then forwards to typed ConstantInterpolant constructor.
#
# PERFORMANCE: Typed signature enables compile-time specialization.
# _promote_itp_inputs becomes no-op when types already match (Float64 → Float64).
@inline function constant_interp(
    x::AbstractVector{TX},
    y::AbstractVector{TY};
    extrap::AbstractExtrap=NoExtrap(),
    side::AbstractSide=NearestSide(),
    search::AbstractSearchPolicy=AutoSearch()
) where {TX<:Real, TY}
    x_p, y_p = _promote_itp_inputs(x, y)
    return ConstantInterpolant(x_p, y_p; extrap, side, search)
end
