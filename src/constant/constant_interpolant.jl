# ========================================
# Constant Interpolant Callable Methods
# ========================================
# Callable methods for ConstantInterpolant and 2-arg API.
# Type definition is in constant_types.jl.
# Internal evaluation and oneshot API (constant_interp!, constant_interp 3-arg)
# are in constant_oneshot.jl.

# ========================================
# Protocol Trait Implementations
# ========================================
# Generic callables inherited from AbstractInterpolant1D (interpolant_protocol.jl).
# _itp_grid, _itp_extrap, _itp_search use defaults (itp.x, itp.extrap, itp.search_policy).

@inline function _itp_eval_scalar(itp::ConstantInterpolant, xq, extrap, op, searcher)
    return _constant_eval_at_point(itp.x, itp.y, xq, extrap, itp.side, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::ConstantInterpolant, xq, extrap, op, searcher)
    return _constant_vector_loop!(output, itp.x, itp.y, xq, extrap, itp.side, op, searcher)
end

# ========================================
# Callable Overrides — selection-kernel output type
# ========================================
# The shared 1D protocol allocates `Vector{_output_eltype(Tv, Tg, Tq)}` which
# widens Tv via promote_type. That's correct for arithmetic kernels (Linear,
# Cubic, …) but violates Constant's raw-Tv contract — `itp([0.5])` would
# return `Vector{Float64}` while `itp(0.5)` and the oneshot return raw `Tv`.
#
# These overrides keep raw `Tv` for plain numeric queries (`_PromotableValue`)
# and widen only for duck-typed queries (Dual, Measurement, …) so AD carriers
# round-trip without being stripped. Scalar / 1D-batch / ND-batch share the
# same rule for consistency with the oneshot family in `constant_oneshot.jl`.

@inline _constant_out_eltype(::Type{Tv}, ::Type{Tq}) where {Tv, Tq} =
    Tq <: _PromotableValue ? Tv : promote_type(Tv, Tq)

@inline function (itp::ConstantInterpolant{Tg, Tv})(
        xq::Tq;
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    @boundscheck _check_domain(grid, xq, extrap)
    searcher = _resolve_search(grid, xq, search, hint)
    val = _constant_eval_at_point(itp.x, itp.y, xq, extrap, itp.side, deriv, searcher)
    return convert(_constant_out_eltype(Tv, Tq), val)
end

function (itp::ConstantInterpolant{Tg, Tv})(
        xq::AbstractVector{Tq};
        deriv::DerivOp = EvalValue(),
        search::AbstractSearchPolicy = _itp_search(itp),
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv, Tq <: Real}
    grid = _itp_grid(itp)
    extrap = _itp_extrap(itp)
    T_out = _constant_out_eltype(Tv, Tq)
    output = Vector{T_out}(undef, length(xq))
    searcher = _resolve_search(grid, xq, search, hint)
    _itp_vector_loop!(output, itp, xq, extrap, deriv, searcher)
    return output
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
    ) where {Tg, Tv, E <: AbstractExtrap, SD <: AbstractSide, O <: AbstractEvalOp, P <: Searcher}
    extrap = _check_domain(x, xq, extrap)
    return @inbounds for i in eachindex(xq, output)
        output[i] = _constant_eval_at_point(x, y, xq[i], extrap, side, deriv, searcher)
    end
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    constant_interp(x, y; bc=NoBC(), side=NearestSide(), extrap=NoExtrap(), search=AutoSearch()) -> ConstantInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` (no BC). Pass
  `PeriodicBC(endpoint=:inclusive)` or `PeriodicBC(endpoint=:exclusive, period=L)`
  to build a periodic interpolant (extrap is forced to `WrapExtrap()`).
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `side::AbstractSide`: Side selection (NearestSide(), LeftSide(), RightSide())
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Returns
`ConstantInterpolant{Tg, Tv}` object for scalar/broadcast evaluation.
- `Tg`: Grid type (Float32/Float64)
- `Tv`: Value type (unconstrained)

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
# Selection kernel → raw eltype contract (Int in → Int out). Signature
# parametrized directly on `{Tg, Tv}` — no `_promote_grid_float` indirection.
@inline function constant_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv};
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    x_eff = _cache_axis(x, bc, Tg)
    y_eff = _resolve_data(y, bc)
    extrap_eff = _resolve_extrap(extrap, bc, x_eff, y_eff)
    extrap_p = _promote_extrap(extrap_eff, Tv)
    return ConstantInterpolant(x_eff, y_eff, extrap_p, side, search; bc = bc)
end
