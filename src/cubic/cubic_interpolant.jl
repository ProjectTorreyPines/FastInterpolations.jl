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
# - _cubic_vector_loop!(output, cache, y, z, x_query, ev)

# ========================================
# CubicInterpolant Callable Methods
# ========================================

# CubicInterpolant struct defined in cubic_types.jl

# Scalar call - hot path (zero-allocation)
# Supports deriv keyword for derivative evaluation
@inline function (itp::CubicInterpolant{T})(xi::T; deriv::Int=0) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.cache.x, xi, itp.extrap)
    @_dispatch_deriv deriv => op begin
        _eval_with_bc(itp.cache, itp.y, itp.z, xi, itp.extrap, op)
    end
end

# Real scalar wrapper - delegates to T method with deriv keyword
@inline function (itp::CubicInterpolant{T})(xi::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); deriv=deriv)
end

# Vector call with deriv keyword support
function (itp::CubicInterpolant{T})(xi::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    xi_typed = S === T ? xi : T.(xi)
    output = Vector{T}(undef, length(xi_typed))
    @_dispatch_deriv deriv => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi_typed, itp.extrap, op)
    end
    return output
end

function (itp::CubicInterpolant{T})(xi::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(xi))
    @_dispatch_deriv deriv => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap, op)
    end
    return output
end

# In-place vector call with deriv keyword support
function (itp::CubicInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    @_dispatch_deriv deriv => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap, op)
    end
    return output
end

function (itp::CubicInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = T.(xi)
    @_dispatch_deriv deriv => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi_typed, itp.extrap, op)
    end
    return output
end

# ========================================
# Anchored Query Evaluation
# ========================================

"""
    (itp::CubicInterpolant)(aq::_CubicAnchoredQuery; deriv::Int=0) -> T

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
aq = _anchor_query(x, 0.35)

itp(aq)  # Ultra-fast evaluation
```
"""
@inline function (itp::CubicInterpolant{T})(aq::_CubicAnchoredQuery{T}; deriv::Int=0) where {T<:AbstractFloat}
    @_dispatch_deriv deriv => op begin
        # Fast path: inside domain (most common case)
        if aq.side == 0x00
            return _eval_anchored_kernel(itp, aq, op)
        end

        # Outside domain: dispatch on extrapolation mode
        return _eval_anchored_extrap(itp, aq, itp.extrap, op)
    end
end

"""
    _eval_anchored_kernel(itp, aq, op) -> T

Core anchored evaluation kernel. Computes the 4-term dot product.

S(xq) = wyL*yL + wyR*yR + wzL*zL + wzR*zR
"""
@inline function _eval_anchored_kernel(itp::CubicInterpolant{T}, aq::_CubicAnchoredQuery{T}, op::AbstractEvalOp) where {T}
    @inbounds begin
        yL = itp.y[aq.idx]
        yR = itp.y[aq.idx + 1]
        zL = itp.z[aq.idx]
        zR = itp.z[aq.idx + 1]
    end
    wyL, wyR, wzL, wzR = _anchored_weights(aq, op)
    # Optimal FMA chain: 3 FMAs + 1 mul
    return muladd(wyR, yR, muladd(wyL, yL, muladd(wzR, zR, wzL * zL)))
end

# Anchored weights by op
@inline _anchored_weights(aq::_CubicAnchoredQuery, ::EvalValue) = aq.w0
@inline _anchored_weights(aq::_CubicAnchoredQuery, ::EvalDeriv1) = aq.w1
@inline _anchored_weights(aq::_CubicAnchoredQuery, ::EvalDeriv2) = aq.w2
@inline _anchored_weights(aq::_CubicAnchoredQuery, ::EvalDeriv3) = aq.w3

# ========================================
# Anchored Extrapolation Handlers
# ========================================

# :none - throw DomainError
@inline function _eval_anchored_extrap(itp::CubicInterpolant{T}, aq::_CubicAnchoredQuery{T}, ::Val{:none}, ::AbstractEvalOp) where {T}
    x_min, x_max = first(itp.cache.x), last(itp.cache.x)
    throw(DomainError(aq.xq, "query point outside domain [$x_min, $x_max]"))
end

# :constant for value - return boundary y value
@inline function _eval_anchored_extrap(itp::CubicInterpolant{T}, aq::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalValue) where {T}
    return aq.side == 0x01 ? @inbounds(itp.y[1]) : @inbounds(itp.y[end])
end

# :constant for derivatives - return zero
@inline function _eval_anchored_extrap(::CubicInterpolant{T}, ::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalDeriv1) where {T}
    return zero(T)
end

@inline function _eval_anchored_extrap(::CubicInterpolant{T}, ::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalDeriv2) where {T}
    return zero(T)
end

@inline function _eval_anchored_extrap(::CubicInterpolant{T}, ::_CubicAnchoredQuery{T}, ::Val{:constant}, ::EvalDeriv3) where {T}
    return zero(T)
end

# :extension - use precomputed weights (boundary polynomial extrapolation)
@inline function _eval_anchored_extrap(itp::CubicInterpolant{T}, aq::_CubicAnchoredQuery{T}, ::Val{:extension}, op::AbstractEvalOp) where {T}
    return _eval_anchored_kernel(itp, aq, op)
end

# :wrap - use precomputed weights (already wrapped at anchor construction if needed)
@inline function _eval_anchored_extrap(itp::CubicInterpolant{T}, aq::_CubicAnchoredQuery{T}, ::Val{:wrap}, op::AbstractEvalOp) where {T}
    return _eval_anchored_kernel(itp, aq, op)
end

# ========================================
# Vector Anchored Query Evaluation
# ========================================

"""
    _eval_anchored_vector_loop!(output, itp, aq, op) -> output

Internal kernel for batch anchored evaluation.

Iterates through anchor vector, dispatching each to existing scalar kernels:
- `_eval_anchored_kernel` for inside-domain (side == 0x00)
- `_eval_anchored_extrap` for outside-domain (side != 0x00)

For extrap=:none, throws DomainError on first out-of-domain anchor.
"""
@inline function _eval_anchored_vector_loop!(
    output::AbstractVector{T},
    itp::CubicInterpolant{T},
    aq::AbstractVector{<:_CubicAnchoredQuery{T}},
    op::AbstractEvalOp
) where {T<:AbstractFloat}
    @inbounds for k in eachindex(aq, output)
        aq_k = aq[k]
        if aq_k.side == 0x00
            # Fast path: inside domain
            output[k] = _eval_anchored_kernel(itp, aq_k, op)
        else
            # Extrapolation path (may throw for :none)
            output[k] = _eval_anchored_extrap(itp, aq_k, itp.extrap, op)
        end
    end
    return output
end

"""
    (itp::CubicInterpolant{T})(aq::AbstractVector{<:_CubicAnchoredQuery{T}}; deriv::Int=0) -> Vector{T}

Evaluate cubic spline at multiple anchored query points (allocating).

# Extrapolation Behavior
- `:none`: Throws `DomainError` on **first** out-of-domain anchor
- `:constant`: Returns boundary value (or zero for derivatives)
- `:extension`: Uses boundary polynomial extrapolation
- `:wrap`: Uses pre-wrapped coordinates from anchor construction

# Example
```julia
x = collect(range(0.0, 1.0, 101))
itp = cubic_interp(x, sin.(2π .* x))
aq_vec = _anchor_query(x, [0.15, 0.35, 0.5])

vals = itp(aq_vec)            # Value
derivs = itp(aq_vec; deriv=1) # First derivative
```
"""
function (itp::CubicInterpolant{T})(
    aq::AbstractVector{<:_CubicAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(aq))
    @_dispatch_deriv deriv => op begin
        _eval_anchored_vector_loop!(output, itp, aq, op)
    end
    return output
end

"""
    (itp::CubicInterpolant{T})(output::AbstractVector{T}, aq::AbstractVector{<:_CubicAnchoredQuery{T}}; deriv::Int=0) -> AbstractVector{T}

Evaluate cubic spline at multiple anchored query points (in-place, zero-allocation).

# Example
```julia
output = Vector{Float64}(undef, length(aq_vec))
itp(output, aq_vec; deriv=1)  # Zero allocation after warmup
```
"""
function (itp::CubicInterpolant{T})(
    output::AbstractVector{T},
    aq::AbstractVector{<:_CubicAnchoredQuery{T}};
    deriv::Int=0
) where {T<:AbstractFloat}
    @assert length(output) == length(aq) "output length ($(length(output))) must match aq length ($(length(aq)))"
    @_dispatch_deriv deriv => op begin
        _eval_anchored_vector_loop!(output, itp, aq, op)
    end
    return output
end

# ========================================
# Internal Build Helpers
# ========================================
# These helpers unify the interpolant construction logic,
# using the cache helpers from cubic_interp.jl.

"""
    _symbol_to_extrap_val(extrap::Symbol) -> Val

Convert extrapolation symbol to Val type for storage in CubicInterpolant.
"""
@inline function _symbol_to_extrap_val(extrap::Symbol)
    extrap === :none && return Val(:none)
    extrap === :constant && return Val(:constant)
    extrap === :extension && return Val(:extension)
    extrap === :wrap && return Val(:wrap)
    throw(ArgumentError("`extrap` must be :none, :constant, :extension, or :wrap, got :$extrap"))
end

"""
    _build_interpolant_bcpair(x, y, bc_pair, extrap, autocache) -> CubicInterpolant

Build a CubicInterpolant for BCPair boundary conditions.
Uses task-local pool for thread-safe workspace allocation.

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@inline @with_pool pool function _build_interpolant_bcpair(
    x::AbstractVector{T},
    y::AbstractVector{T},
    bc_pair::BCPair{T,L,R},
    extrap::Symbol,
    autocache::Bool
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    cache = _get_cubic_cache(x, bc_pair, autocache)
    ev = _symbol_to_extrap_val(extrap)
    tmp_z = similar!(pool, y)
    _solve_system!(tmp_z, cache, y, bc_pair)
    # tmp_z is copied by CubicInterpolant constructor - safe to return to pool
    return CubicInterpolant(cache, y, tmp_z, ev)
end

"""
    _build_interpolant_periodic(x, y, autocache) -> CubicInterpolant

Build a CubicInterpolant for PeriodicBC boundary conditions.
Uses task-local pool for thread-safe workspace allocation.
Periodic BC always uses :wrap extrapolation.

# Thread-Safety
Pool-allocated `tmp_z` is copied by the CubicInterpolant constructor,
so the pool memory can be safely reused after this function returns.
"""
@inline @with_pool pool function _build_interpolant_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    autocache::Bool
) where {T<:AbstractFloat}
    _check_periodic_endpoints(y)
    cache = _get_cubic_cache(x, PeriodicBC(), autocache)
    tmp_z = similar!(pool, y)
    _solve_system!(tmp_z, cache, y, cache.bc_config)
    # tmp_z is copied by CubicInterpolant constructor - safe to return to pool
    return CubicInterpolant(cache, y, tmp_z, Val(:wrap))
end

# ========================================
# 2-Argument Form: Return CubicInterpolant
# ========================================

"""
    cubic_interp(x, y; bc=NaturalBC(), extrap=:none, autocache=true) -> CubicInterpolant

Create a callable interpolant for broadcast fusion and reuse.

Pre-computes second derivative coefficients z ONCE at construction time,
enabling true zero-allocation scalar evaluations in broadcast operations.

# Example
```julia
itp = cubic_interp(x, y)           # Pre-computes z coefficients
val = itp(0.5)                      # Scalar (zero-allocation)
vals = itp.(query_points)           # Broadcast
result = @. coef * itp(rho) * ne    # Fused broadcast
```
"""
# Internal implementation - takes AbstractBC only (type-stable)
@inline function _cubic_interp_impl(
    x::AbstractVector{T},
    y::AbstractVector{T},
    bc::AbstractBC,
    extrap::Symbol,
    autocache::Bool
) where {T<:AbstractFloat}
    if _is_periodic_bc(bc)
        return _build_interpolant_periodic(x, y, autocache)
    else
        bc_pair = _normalize_bc(bc, T)
        return _build_interpolant_bcpair(x, y, bc_pair, extrap, autocache)
    end
end

# Public API - AbstractBC only (type-stable)
function cubic_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true
) where {T<:AbstractFloat}
    return _cubic_interp_impl(x, y, bc, extrap, autocache)
end

"""
    cubic_interp(cache, y; extrap=:none) -> CubicInterpolant

Create a callable interpolant from a pre-built cache.

# Note on BC Values
Uses `cache.bc_config` for boundary condition values. This is correct when:
- Cache was built via `CubicSplineCache(x; bc=...)` with actual BC values
- BC is NaturalBC/ClampedBC/PeriodicBC (values are always zero)

**Warning**: Caches from `get_cubic_cache` contain placeholder zeros in `bc_config`.
For non-zero BC values, use the full API: `cubic_interp(x, y; bc=Deriv1(val))`.

# Thread-Safety
Uses task-local pool for workspace allocation. Pool memory is copied
by the CubicInterpolant constructor.
"""
@with_pool pool function cubic_interp(
    cache::CubicSplineCache{T},
    y::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    tmp_z = similar!(pool, y)
    _solve_system!(tmp_z, cache, y, cache.bc_config)

    if cache.bc_config isa PeriodicData
        _check_periodic_endpoints(y)
        return CubicInterpolant(cache, y, tmp_z, Val(:wrap))
    end

    ev = _symbol_to_extrap_val(extrap)
    return CubicInterpolant(cache, y, tmp_z, ev)
end

# Real wrapper for 2-argument form
function cubic_interp(
    x::X,
    y::Y;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    T = promote_type(TX, TY)
    FT = float(T)
    x_float = _to_float(x, FT)
    y_float = FT.(y)

    if _is_periodic_bc(bc)
        return _build_interpolant_periodic(x_float, y_float, autocache)
    else
        bc_pair = _normalize_bc(bc, FT)
        return _build_interpolant_bcpair(x_float, y_float, bc_pair, extrap, autocache)
    end
end
