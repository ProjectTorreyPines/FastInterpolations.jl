# ========================================
# CubicInterpolant: 2-Argument Form API
# ========================================
#
# This file contains:
# - CubicInterpolant callable methods (scalar, vector, in-place)
# - 2-argument form `cubic_interp(x, y; ...)` that returns CubicInterpolant
# - Internal build helpers for type-stable interpolant construction
#
# Dependencies (from cubic_interp.jl):
# - _is_periodic_bc(bc)
# - _get_cubic_cache(x, bc, autocache)
# - _solve_system!(out_z, cache, y, bc_config)
# - _check_periodic_endpoints(y)
# - _cubic_vector_loop!(output, cache, y, z, x_query, extrap, op, searcher)

# ========================================
# Protocol Trait Implementations
# ========================================
# Generic callables inherited from AbstractInterpolant1D (interpolant_protocol.jl).
# Cubic overrides _itp_grid because grid lives in itp.cache.x, not itp.x.

@inline _itp_grid(itp::CubicInterpolant) = itp.cache.x

@inline function _itp_eval_scalar(itp::CubicInterpolant, xq, extrap, op, searcher)
    return _eval_with_bc(itp.cache, itp.y, itp.z, xq, extrap, op, searcher)
end

@inline function _itp_vector_loop!(output, itp::CubicInterpolant, xq, extrap, op, searcher)
    return _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xq, extrap, op, searcher)
end

# ========================================
# Anchored Query Evaluation
# ========================================

"""
    (itp::CubicInterpolant)(aq::_CubicAnchoredQuery; deriv::DerivOp=EvalValue()) -> Tv

Evaluate cubic spline using precomputed anchor weights.

This is the ultra-fast evaluation path that skips interval search and
geometry computation. The caller must ensure the anchor was built for
the same grid as the interpolant. Derivatives are selected at evaluation time via
the `deriv` keyword, matching `itp(xq; deriv=...)`.

# Performance
- 4 loads (yL, yR, zL, zR)
- 4 weight loads (from anchor struct, likely in registers)
- 3 FMAs + 1 mul (dot product)
- Total: ~8 FP ops vs ~14 for standard path + no interval search

# Example
```julia
x = collect(range(0.0, 1.0, 101))
itp = cubic_interp(x, sin.(2π .* x))
aq = _anchor_query(x, 0.35, Val(:cubic))

itp(aq)  # Ultra-fast evaluation
```
"""
@inline function (itp::CubicInterpolant{Tg, Tv})(aq::_CubicAnchoredQuery{Tg, Tq}; deriv::DerivOp = EvalValue()) where {Tg, Tv, Tq <: Real}
    # Fast path: inside domain (most common case)
    if aq.state == IN_DOMAIN
        return _eval_anchored_kernel(itp, aq, deriv)
    end

    # Outside domain: dispatch on extrapolation mode
    return _eval_anchored_extrap(itp, aq, itp.extrap, deriv)
end

"""
    _eval_anchored_kernel(itp, aq, op) -> Tv

Thin wrapper: delegates to shared `_cubic_eval_kernel(y, z, aq, op)` in cubic_anchor.jl.
"""
@inline _eval_anchored_kernel(itp::CubicInterpolant, aq::_CubicAnchoredQuery, op::AbstractEvalOp) =
    _cubic_eval_kernel(itp.y, itp.z, aq, op)

# ========================================
# Anchored Extrapolation Handlers (thin wrappers)
# ========================================

# NoExtrap - throw DomainError with enriched bounds
@inline function _eval_anchored_extrap(itp::CubicInterpolant{Tg, Tv}, aq::_CubicAnchoredQuery{Tg, Tq}, ::NoExtrap, ::AbstractEvalOp) where {Tg, Tv, Tq <: Real}
    x_min, x_max = first(itp.cache.x), last(itp.cache.x)
    throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
end

# ClampExtrap / FillExtrap - delegate to shared
@inline _eval_anchored_extrap(itp::CubicInterpolant, aq::_CubicAnchoredQuery, extrap::_ClampOrFill, op::AbstractEvalOp) =
    _cubic_eval_at_anchor(itp.y, itp.z, aq, op, extrap)

# ExtendExtrap - delegate to shared kernel
@inline _eval_anchored_extrap(itp::CubicInterpolant, aq::_CubicAnchoredQuery, ::ExtendExtrap, op::AbstractEvalOp) =
    _cubic_eval_kernel(itp.y, itp.z, aq, op)

# WrapExtrap - delegate to shared kernel
@inline _eval_anchored_extrap(itp::CubicInterpolant, aq::_CubicAnchoredQuery, ::WrapExtrap, op::AbstractEvalOp) =
    _cubic_eval_kernel(itp.y, itp.z, aq, op)

# ========================================
# Vector Anchored Query Evaluation
# ========================================

"""
    _eval_anchored_vector_loop!(output, itp, aq, op) -> output

Internal kernel for batch anchored evaluation.

Iterates through anchor vector, dispatching each to existing scalar kernels:
- `_eval_anchored_kernel` for inside-domain (state == IN_DOMAIN)
- `_eval_anchored_extrap` for outside-domain (state != IN_DOMAIN)

For extrap=NoExtrap(), throws DomainError on first out-of-domain anchor.
"""
@inline function _eval_anchored_vector_loop!(
        output::AbstractVector{Tv},
        itp::CubicInterpolant{Tg, Tv},
        aq::AbstractVector{<:_CubicAnchoredQuery{Tg}},
        op::AbstractEvalOp
    ) where {Tg, Tv}
    @inbounds for k in eachindex(aq, output)
        aq_k = aq[k]
        if aq_k.state == IN_DOMAIN
            # Fast path: inside domain
            output[k] = _eval_anchored_kernel(itp, aq_k, op)
        else
            # Extrapolation path (may throw for NoExtrap)
            output[k] = _eval_anchored_extrap(itp, aq_k, itp.extrap, op)
        end
    end
    return output
end

"""
    (itp::CubicInterpolant{Tg,Tv})(aq::AbstractVector{<:_CubicAnchoredQuery{Tg}}; deriv::DerivOp=EvalValue()) -> Vector{Tv}

Evaluate cubic spline at multiple anchored query points (allocating).

# Extrapolation Behavior
- `NoExtrap()`: Throws `DomainError` on **first** out-of-domain anchor
- `ClampExtrap()`: Returns boundary value (or zero for derivatives)
- `ExtendExtrap()`: Uses boundary polynomial extrapolation
- `WrapExtrap()`: Uses pre-wrapped coordinates from anchor construction

# Example
```julia
x = collect(range(0.0, 1.0, 101))
itp = cubic_interp(x, sin.(2π .* x))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.5], Val(:cubic))

vals = itp(aq_vec)            # Value
derivs = itp(aq_vec; deriv=DerivOp(1)) # First derivative
```
"""
function (itp::CubicInterpolant{Tg, Tv})(
        aq::AbstractVector{<:_CubicAnchoredQuery{Tg, Tq}};
        deriv::DerivOp = EvalValue()
    ) where {Tg, Tv, Tq <: Real}
    T_out = promote_type(Tv, Tq)  # Lossless: wider type to avoid precision loss from anchor
    output = Vector{T_out}(undef, length(aq))
    _eval_anchored_vector_loop!(output, itp, aq, deriv)
    return output
end

"""
    (itp::CubicInterpolant{Tg,Tv})(output::AbstractVector{Tv}, aq::AbstractVector{<:_CubicAnchoredQuery{Tg}}; deriv::DerivOp=EvalValue()) -> AbstractVector{Tv}

Evaluate cubic spline at multiple anchored query points (in-place, zero-allocation).

# Example
```julia
output = Vector{Float64}(undef, length(aq_vec))
itp(output, aq_vec; deriv=DerivOp(1))  # Zero allocation after warmup
```
"""
function (itp::CubicInterpolant{Tg, Tv})(
        output::AbstractVector{Tv},
        aq::AbstractVector{<:_CubicAnchoredQuery{Tg}};
        deriv::DerivOp = EvalValue()
    ) where {Tg, Tv}
    @assert length(output) == length(aq) "output length ($(length(output))) must match aq length ($(length(aq)))"
    _eval_anchored_vector_loop!(output, itp, aq, deriv)
    return output
end

# ========================================
# Internal Build Helpers
# ========================================
# These helpers unify the interpolant construction logic,
# using the cache helpers from cubic_interp.jl.

"""
    _build_interpolant_bcpair(x, y, bc_pair, extrap, autocache, search) -> CubicInterpolant

Build a CubicInterpolant for BCPair boundary conditions.
Tg = grid type, Tv = value type (can be Complex)

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@inline @with_pool pool function _build_interpolant_bcpair(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc_pair::BCPair{L, R},
        extrap::AbstractExtrap,
        autocache::Bool,
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv, L <: PointBC, R <: PointBC}
    # Cache uses structural equivalent (PolyFit → Deriv1 via _cache_bc_pair internally)
    cache = _get_cubic_cache(x, bc_pair, _effective_autocache(autocache, Tg))
    Tz = _output_eltype(Tv, eltype(cache.x))
    tmp_z = acquire!(pool, Tz, length(y))
    # Solve uses original BC for proper RHS materialization
    _solve_system!(tmp_z, cache, y, bc_pair)
    extrap_p = _promote_extrap(extrap, Tv)
    return CubicInterpolant(cache, y, tmp_z, bc_pair, extrap_p, search)
end

"""
    _build_interpolant_periodic(x, y, autocache, search) -> CubicInterpolant

Build a CubicInterpolant for PeriodicBC boundary conditions.
Periodic BC always uses WrapExtrap extrapolation.
Tg = grid type, Tv = value type (can be Complex)

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@inline @with_pool pool function _build_interpolant_periodic(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::PeriodicBC,
        autocache::Bool,
        search::AbstractSearchPolicy = AutoSearch()
    ) where {Tg, Tv}
    x, y = _prepare_periodic(x, y, bc)
    _check_periodic_endpoints(bc, y)
    cache = _get_cubic_cache(x, PeriodicBC(), _effective_autocache(autocache, Tg))
    Tz = _output_eltype(eltype(y), eltype(cache.x))
    tmp_z = acquire!(pool, Tz, length(y))
    _solve_system!(tmp_z, cache, y, cache.bc_config)
    bc_display = _with_resolved_period(bc, cache.bc_config.period)
    return CubicInterpolant(cache, y, tmp_z, bc_display, WrapExtrap(), search)
end

# ========================================
# BC Type Promotion for Complex Support
# ========================================

"""
    _promote_bc(bc::AbstractBC, ::Type{Tv}) -> AbstractBC

Promote BC to target value type Tv for cubic splines.
Handles conversion of Real BC values to Complex when needed.
"""
@inline _promote_bc(bc::BCPair, ::Type{Tv}) where {Tv} = _normalize_bc(bc, Tv)
@inline _promote_bc(::ZeroCurvBC, ::Type{Tv}) where {Tv} = ZeroCurvBC()
@inline _promote_bc(::ZeroSlopeBC, ::Type{Tv}) where {Tv} = ZeroSlopeBC()
@inline _promote_bc(bc::PeriodicBC, ::Type{Tv}) where {Tv} = bc
@inline _promote_bc(bc::PointBC, ::Type{Tv}) where {Tv} = _promote_pointbc(bc, Tv)
@inline _promote_bc(bcs::AbstractVector{<:AbstractBC}, ::Type{Tv}) where {Tv} =
    [_promote_bc(bc, Tv) for bc in bcs]

# ========================================
# 2-Argument Form: Return CubicInterpolant
# ========================================

"""
    cubic_interp(x, y; bc=CubicFit(), extrap=NoExtrap(), autocache=true, search=AutoSearch()) -> CubicInterpolant

Create a callable interpolant for broadcast fusion and reuse.

Pre-computes second derivative coefficients z ONCE at construction time,
enabling true zero-allocation scalar evaluations in broadcast operations.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `bc::AbstractBC`: Boundary condition (default: `CubicFit()`)
- `extrap::AbstractExtrap`: `NoExtrap()` (default), `ClampExtrap()`, `ExtendExtrap()`, or `WrapExtrap()`
- `autocache::Bool`: Enable automatic caching (default: `true`)
- `search::AbstractSearchPolicy`: Default search policy (default: `AutoSearch()`)

# Example
```julia
itp = cubic_interp(x, y)           # Pre-computes z coefficients
val = itp(0.5)                      # Scalar (zero-allocation)
vals = itp.(query_points)           # Broadcast
result = @. coef * itp(rho) * ne    # Fused broadcast

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = cubic_interp(x, y)
val = itp(0.5)                      # AutoSearch resolves to BinarySearch() for scalar
itp = cubic_interp(x, y; search=LinearBinarySearch())  # explicit override
val = itp(0.5; search=BinarySearch())     # per-call override

# Complex values
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im, 9.0+10.0im]
itp = cubic_interp(x, y)
val = itp(0.5)  # returns ComplexF64
```
"""
# Internal implementation - takes AbstractBC only (type-stable)
# Tg = grid type, Tv = value type (can be Complex)
@inline function _cubic_interp_impl(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv},
        bc::AbstractBC,
        extrap::AbstractExtrap,
        autocache::Bool,
        search::P = AutoSearch()
    ) where {Tg, Tv, P <: AbstractSearchPolicy}
    if _is_periodic_bc(bc)
        return _build_interpolant_periodic(x, y, bc, autocache, search)
    else
        bc_pair = _normalize_bc(bc, first(y))
        return _build_interpolant_bcpair(x, y, bc_pair, extrap, autocache, search)
    end
end

# Unified entry: handles all grid types including duck-typed (Dual).
function cubic_interp(
        x::AbstractVector{Tg},
        y::AbstractVector{Tv};
        bc::AbstractBC = CubicFit(),
        extrap::AbstractExtrap = NoExtrap(),
        autocache::Bool = true,
        search::P = AutoSearch()
    ) where {Tg, Tv, P <: AbstractSearchPolicy}
    x_p, y_p = _promote_itp_inputs(x, y)
    bc_promoted = _promote_bc(bc, eltype(y_p))
    return _cubic_interp_impl(x_p, y_p, bc_promoted, extrap, autocache, search)
end

"""
    cubic_interp(cache, y; extrap=NoExtrap(), search=AutoSearch()) -> CubicInterpolant

Create a callable interpolant from a pre-built cache.

# Note on BC Values
Uses `cache.bc_config` for boundary condition values. This is correct when:
- Cache was built via `CubicSplineCache(x; bc=...)` with actual BC values
- BC is ZeroCurvBC/ZeroSlopeBC/PeriodicBC (values are always zero)

**Warning**: Caches from `get_cubic_cache` contain placeholder zeros in `bc_config`.
For non-zero BC values, use the full API: `cubic_interp(x, y; bc=Deriv1(val))`.

Tg = grid type, Tv = value type (can be Complex)

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@with_pool pool function cubic_interp(
        cache::CubicSplineCache{Tg},
        y::AbstractVector{Tv};
        extrap::AbstractExtrap = NoExtrap(),
        search::P = AutoSearch()
    ) where {Tg, Tv, P <: AbstractSearchPolicy}
    Tz = _output_eltype(Tv, eltype(cache.x))
    tmp_z = acquire!(pool, Tz, length(y))
    _solve_system!(tmp_z, cache, y, cache.bc_config)

    if cache.bc_config isa PeriodicData
        _check_periodic_endpoints(y)
        return CubicInterpolant(cache, y, tmp_z, PeriodicBC(), WrapExtrap(), search)
    end

    # cache.bc_config is BCPair - use it directly
    extrap_p = _promote_extrap(extrap, Tv)
    return CubicInterpolant(cache, y, tmp_z, cache.bc_config, extrap_p, search)
end


# Note: Real wrapper (TX <: Real) removed — unified entry above handles
# all grid types including ForwardDiff.Dual via _promote_itp_inputs.
