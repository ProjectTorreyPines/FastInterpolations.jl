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

# Constant declares its kernel shape — selection (`y * one(dL)`, no division).
# Args mirror the real kernel: `(xL, yv, xq)` so `xq - xL` exposes the actual
# `dL` carrier (e.g. `Dual` grid + `Float` xq → `Dual` dL). Trait infers the
# exact return type via `promote_op`, so scalar/batch agree (Int×Int×Int → Int;
# SVector × Dual → SVector{Dual}; Float y × Dual grid → Dual; etc.).
@inline _select_op(xL, yv, xq) = yv * one(xq - xL)

@inline _promote_eltype(::ConstantInterpolant{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq} =
    _promote_eltype(_select_op, Tg, Tv, Tq)

# ─────────────────────────────────────────────────────────────
# Vector loop (function barrier)
# Julia specializes on concrete Searcher type P, eliminating Union-split
# overhead when adaptive AutoSearch resolves to BinarySearch or LinearBinarySearch.
# CRITICAL: All arguments must be fully typed — untyped args prevent SROA
# of RefHint's Ref, causing 16-byte heap allocation per call.
# ─────────────────────────────────────────────────────────────
@inline function _constant_vector_loop!(
        output::AbstractArray,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::AbstractArray,
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
# Storage parametrized on `{Tg, Tv}` directly — no `_promote_grid_float`
# indirection. Return type widens via the kernel's `* one(dL)` carrier
# propagation (handled per-callable, not at construction).
@inline function constant_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv};
        bc::AbstractBC = NoBC(),
        side::AbstractSide = NearestSide(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch(),
        store::StorePolicy = StorePolicy()
    ) where {Tg, Tv}
    _check_grid_orderable(Tg)
    # Persistent: extend-promote for `:exclusive` (matches PCHIP/Cardinal/Akima/Cubic/Linear).
    # OneShot path continues to use the lazy wrapper (constant_oneshot.jl).
    x_ext, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    x_eff = _policy_axis(x_ext, bc_eff, Tg, store)
    extrap_p = _promote_extrap(extrap_eff, Tv)
    return ConstantInterpolant(x_eff, y_ext, extrap_p, side, search; bc = bc_eff, store = store)
end
