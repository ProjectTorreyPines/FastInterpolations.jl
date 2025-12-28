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
# - _get_cache_and_solve!(x, y, bc_pair, autocache)
# - _get_cache_and_solve_periodic!(x, y, autocache)
# - _check_periodic_endpoints(y)
# - _cubic_vector_loop!(output, cache, y, z, x_query, ev)

# ========================================
# CubicInterpolant Callable Methods
# ========================================

# CubicInterpolant struct defined in cubic_types.jl

# Scalar call - hot path (zero-allocation)
# Supports order keyword for derivative evaluation
@inline function (itp::CubicInterpolant{T})(xi::T; order::Int=0) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.cache.x, xi, itp.extrap)
    @_dispatch_order order => op begin
        _eval_with_bc(itp.cache, itp.y, itp.cache.h, itp.z, xi, itp.extrap, op)
    end
end

# Real scalar wrapper - delegates to T method with order keyword
@inline function (itp::CubicInterpolant{T})(xi::S; order::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xi); order=order)
end

# Vector call with order keyword support
function (itp::CubicInterpolant{T})(xi::AbstractVector{S}; order::Int=0) where {T<:AbstractFloat, S<:Real}
    xi_typed = S === T ? xi : T.(xi)
    output = Vector{T}(undef, length(xi_typed))
    @_dispatch_order order => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi_typed, itp.extrap, op)
    end
    return output
end

function (itp::CubicInterpolant{T})(xi::AbstractVector{T}; order::Int=0) where {T<:AbstractFloat}
    output = Vector{T}(undef, length(xi))
    @_dispatch_order order => op begin
        _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap, op)
    end
    return output
end

# In-place vector call
function (itp::CubicInterpolant{T})(output::AbstractVector{T}, xi::AbstractVector{T}) where {T<:AbstractFloat}
    @assert length(output) == length(xi) "output length must match xi length"
    _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi, itp.extrap)
    return output
end

function (itp::CubicInterpolant{T})(output::AbstractVector, xi::AbstractVector{S}) where {T<:AbstractFloat, S<:Real}
    @assert length(output) == length(xi) "output length must match xi length"
    xi_typed = T.(xi)
    _cubic_vector_loop!(output, itp.cache, itp.y, itp.z, xi_typed, itp.extrap)
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
Uses `_get_cache_and_solve!` for unified cache handling.
"""
@inline function _build_interpolant_bcpair(
    x::AbstractVector{T},
    y::AbstractVector{T},
    bc_pair::BCPair{T,L,R},
    extrap::Symbol,
    autocache::Bool
) where {T<:AbstractFloat, L<:PointBC{T}, R<:PointBC{T}}
    cache = _get_cache_and_solve!(x, y, bc_pair, autocache)
    ev = _symbol_to_extrap_val(extrap)
    # Pass z_workspace directly - constructor will copy
    return CubicInterpolant(cache, y, cache.z_workspace, ev)
end

"""
    _build_interpolant_periodic(x, y, autocache) -> CubicInterpolant

Build a CubicInterpolant for PeriodicBC boundary conditions.
Uses `_get_cache_and_solve_periodic!` for unified cache handling.
Periodic BC always uses :wrap extrapolation.
"""
@inline function _build_interpolant_periodic(
    x::AbstractVector{T},
    y::AbstractVector{T},
    autocache::Bool
) where {T<:AbstractFloat}
    _check_periodic_endpoints(y)
    cache = _get_cache_and_solve_periodic!(x, y, autocache)
    # Pass z_workspace directly - constructor will copy
    return CubicInterpolant(cache, y, cache.z_workspace, Val(:wrap))
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
Uses `cache.bc_data` for boundary condition values. This is correct when:
- Cache was built via `CubicSplineCache(x; bc=...)` with actual BC values
- BC is NaturalBC/ClampedBC/PeriodicBC (values are always zero)

**Warning**: Caches from `get_cubic_cache` contain placeholder zeros in `bc_data`.
For non-zero BC values, use the full API: `cubic_interp(x, y; bc=D1(val))`.
"""
function cubic_interp(
    cache::CubicSplineCache{T},
    y::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    _solve_system!(cache, y, cache.bc_data)
    # Pass z_workspace directly - constructor will copy

    if cache.bc_data isa PeriodicData
        _check_periodic_endpoints(y)
        return CubicInterpolant(cache, y, cache.z_workspace, Val(:wrap))
    end

    ev = _symbol_to_extrap_val(extrap)
    return CubicInterpolant(cache, y, cache.z_workspace, ev)
end

# Real wrapper for 2-argument form
function cubic_interp(
    x::X,
    y::Y;
    bc::AbstractBC=NaturalBC(),
    extrap::Symbol=:none,
    autocache::Bool=true
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    _validate_bc(bc)

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
