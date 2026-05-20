# ========================================
# Linear Interpolant Callable Methods
# ========================================
# Callable methods for LinearInterpolant and 2-arg API.
# Type definition is in linear_types.jl.
# Oneshot API (linear_interp!, linear_interp 3-arg) is in linear_oneshot.jl.

# ========================================
# Protocol Trait Implementations
# ========================================
# Generic callables inherited from AbstractInterpolant1D (interpolant_protocol.jl).
# _itp_grid, _itp_extrap, _itp_search use defaults (itp.x, itp.extrap, itp.search_policy).

@inline function _itp_eval_scalar(itp::LinearInterpolant, xq, extrap, op, searcher)
    return _linear_eval_at_point(itp.x, itp.y, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::LinearInterpolant, xq, extrap, op, searcher)
    return _linear_vector_loop!(output, itp.x, itp.y, xq, extrap, op, searcher)
end

# Linear's kernel divides by `h` (`y0 + (y1-y0)*(dL/h)`); shared
# `_arithmetic_kernel_shape` drives type inference for the output buffer.
@inline _output_eltype(::LinearInterpolant{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq} =
    _output_eltype(_arithmetic_kernel_shape, Tg, Tv, Tq)

# ========================================
# Vector Loop (Function Barrier)
# ========================================
# Outer resolves the `_check_domain` Union; inner sees concrete `extrap`,
# union-splitting the per-iter dispatch. Args must be fully typed — untyped
# blocks SROA of RefHint's Ref (16 B/call alloc).
@inline function _linear_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    extrap_eff = _check_domain(x, xq, extrap)
    return _linear_vector_loop_inner!(output, x, y, xq, extrap_eff, deriv, searcher)
end

@inline function _linear_vector_loop_inner!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    @inbounds for i in eachindex(xq, output)
        output[i] = _linear_eval_at_point(x, y, xq[i], extrap, deriv, searcher)
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    linear_interp(x, y; bc=NoBC(), extrap=NoExtrap(), search=AutoSearch()) -> LinearInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `bc::AbstractBC`: Boundary condition. Default `NoBC()` means "use the built-in
  linear-interpolation behavior (no BC needed)". Pass `PeriodicBC(endpoint=:inclusive)`
  or `PeriodicBC(endpoint=:exclusive, period=L)` to build a periodic interpolant
  (extrap is forced to `WrapExtrap()` in that case).
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
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
- `Tv`: Value type (unconstrained)

Can be:
- Called with scalar: `itp(0.5)` (uses stored search policy)
- Called with search override: `itp(0.5; search=BinarySearch())` (override stored policy)
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)`
- Reused multiple times without re-creating

# Examples
```julia
# Create with default AutoSearch() search policy
itp = linear_interp(x_data, y_data)

# Default AutoSearch: scalar→BinarySearch, vector→LinearBinarySearch
itp = linear_interp(x_data, y_data)

# Scalar call (uses stored policy)
val = itp(0.5)

# Scalar call with search policy override
val = itp(0.5; search=BinarySearch())

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
- Default `AutoSearch()` adapts: scalar→`BinarySearch()`, vector→`LinearBinarySearch()`
- Use `search=LinearBinarySearch()` to force linear-binary for all query types
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
        bc::AbstractBC = NoBC(),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX, TY}
    Tg = _promote_grid_float(TX, TY)
    # Persistent: extend-promote for `:exclusive` (matches PCHIP/Cardinal/Akima/Cubic).
    # OneShot path continues to use the lazy wrapper (linear_oneshot.jl).
    x_ext, y_ext, bc_eff, extrap_eff = _periodic_extend_1d(x, y, bc, extrap)
    x_eff = _cache_axis(x_ext, bc_eff, Tg)
    extrap_p = _promote_extrap(extrap_eff, _value_type(TY, Tg))
    return LinearInterpolant(x_eff, y_ext, extrap_p, search; bc = bc_eff)
end
