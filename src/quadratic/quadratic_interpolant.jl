# ========================================
# Quadratic Interpolant Callable Methods
# ========================================
# Callable methods for QuadraticInterpolant and 2-arg API.
# Type definition is in quadratic_types.jl.
# Internal evaluation and oneshot API (quadratic_interp!, quadratic_interp 3-arg)
# are in quadratic_oneshot.jl.

# ========================================
# Protocol Trait Implementations
# ========================================
# Generic callables inherited from AbstractInterpolant1D (interpolant_protocol.jl).
# _itp_grid, _itp_extrap, _itp_search use defaults (itp.x, itp.extrap, itp.search_policy).

@inline function _itp_eval_scalar(itp::QuadraticInterpolant, xq, extrap, op, searcher)
    return _quadratic_eval_at_point(itp.x, itp.y, itp.a, itp.d, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::QuadraticInterpolant, xq, extrap, op, searcher)
    return _quadratic_vector_loop!(output, itp.x, itp.y, itp.a, itp.d, xq, extrap, op, searcher)
end

# ─────────────────────────────────────────────────────────────
# Vector loop (function barrier)
# Julia specializes on concrete Searcher type P, eliminating Union-split
# overhead when adaptive AutoSearch resolves to BinarySearch or LinearBinarySearch.
# CRITICAL: All arguments must be fully typed — untyped args prevent SROA
# of RefHint's Ref, causing 16-byte heap allocation per call.
# ─────────────────────────────────────────────────────────────
@inline function _quadratic_vector_loop!(
        output::AbstractVector,
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        a::AbstractVector{Tc},
        d::AbstractVector{Tc},
        xq::AbstractVector{<:Real},
        extrap::E,
        deriv::O,
        searcher::P
    ) where {Tg, Tv, Tc, E <: AbstractExtrap, O <: AbstractEvalOp, P <: Searcher}
    extrap = _check_domain(x, xq, extrap)
    return @inbounds for i in eachindex(xq, output)
        output[i] = _quadratic_eval_at_point(x, y, a, d, xq[i], extrap, deriv, searcher)
    end
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    quadratic_interp(x, y; bc=Left(QuadraticFit()), extrap=NoExtrap(), search=AutoSearch()) -> QuadraticInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (sorted, length ≥ 2)
- `y::AbstractVector`: y-values
- `bc`: Boundary condition (Left, Right, MinCurvFit, or Left/Right with QuadraticFit)
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Returns
`QuadraticInterpolant` object for scalar/broadcast evaluation.
- `Tg`: Grid type (unconstrained — supports duck types like ForwardDiff.Dual)
- `Tv`: Value type (unconstrained)

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

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = quadratic_interp(x, y)
val = itp(0.5)     # AutoSearch resolves to BinarySearch() for scalar
itp = quadratic_interp(x, y; search=LinearBinarySearch())  # explicit override

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
        bc::QuadraticBC = Left(QuadraticFit()),
        extrap::AbstractExtrap = NoExtrap(),
        search::AbstractSearchPolicy = AutoSearch()
    ) where {TX, TY}
    x_p = _promote_grid_only(x, y)
    bc_p = _normalize_bc(bc, first(y))

    # Validate PolyFit{D} point requirements (e.g., CubicFit needs 4+ points)
    validate_polyfit_points(bc_p, length(x_p))

    # Compute spacing once: used for both coefficients and struct storage
    spacing = _create_spacing(x_p)

    # Compute coefficients (d::Tc, a::Tc where Tc = _output_eltype(Tv, Tg))
    d, a = _compute_quadratic_coeffs(x_p, y, bc_p, spacing)

    extrap_p = _promote_extrap(extrap, _value_type(TY, eltype(x_p)))
    return QuadraticInterpolant(x_p, y, spacing, a, d, extrap_p, search, bc_p)
end
