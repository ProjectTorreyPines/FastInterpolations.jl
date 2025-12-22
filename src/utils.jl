# Internal utility functions for FastInterpolations.jl

"""
    _find_interval_with_bounds(x::AbstractRange{FT}, xi::FT) where {FT<:AbstractFloat}

Find interpolation interval using O(1) direct calculation for uniform grids.

Returns `(idx, x0, x1)` where:
- `idx`: interval index in [1, length(x)-1]
- `x0`: left boundary value x[idx]
- `x1`: right boundary value x[idx+1]
"""
@inline function _find_interval_with_bounds(
    x::AbstractRange{FT},
    xi::FT
) where {FT<:AbstractFloat}
    n = length(x)
    x_min = first(x)
    dx = Base.step(x)

    # epsilon handles floating point errors (e.g., 1.999999 should map to index 2, not 1)
    idx = clamp(floor(Int, (xi - x_min) / dx + 1 + 10*eps(FT)), 1, n - 1)

    # Direct calculation to avoid expensive TwicePrecision indexing
    x0 = x_min + (idx - 1) * dx
    x1 = x0 + dx
    return idx, x0, x1
end

"""
    _find_interval_with_bounds(x::AbstractVector{FT}, xi::FT) where {FT<:AbstractFloat}

Find interpolation interval using O(log n) binary search for non-uniform grids.

Returns `(idx, x0, x1)` where:
- `idx`: interval index in [1, length(x)-1]
- `x0`: left boundary value x[idx]
- `x1`: right boundary value x[idx+1]
"""
@inline function _find_interval_with_bounds(
    x::AbstractVector{FT},
    xi::FT
) where {FT<:AbstractFloat}
    n = length(x)

    # Find interval using binary search
    @inbounds begin
        if xi <= x[1]
            idx = 1
        elseif xi >= x[end]
            idx = n - 1
        else
            lo, hi = 1, n
            while hi - lo > 1
                mid = (lo + hi) >> 1 # Fast division by 2
                if x[mid] <= xi
                    lo = mid
                else
                    hi = mid
                end
            end
            idx = lo
        end
    end

    # Return idx and boundary values for alpha calculation
    @inbounds x0, x1 = x[idx], x[idx + 1]
    return idx, x0, x1
end

# ========================================
# Type Conversion Helpers
# ========================================

"""
    _to_float(x::AbstractRange, ::Type{FT}) where {FT<:AbstractFloat}

Convert a Range to a float type while preserving Range structure for O(1) index lookup.
Using `FT.(x)` would convert Range to Vector, losing the O(1) optimization.
"""
_to_float(x::AbstractRange, ::Type{FT}) where {FT<:AbstractFloat} =
    range(FT(first(x)), FT(last(x)), length(x))

"""
    _to_float(x::AbstractVector, ::Type{FT}) where {FT<:AbstractFloat}

Convert a Vector to a float type (element-wise broadcast).
"""
_to_float(x::AbstractVector, ::Type{FT}) where {FT<:AbstractFloat} = FT.(x)

# ========================================
# Periodic Boundary Helpers
# ========================================

"""
    _wrap_to_domain(xi::FT, x_min::FT, x_max::FT) where {FT<:AbstractFloat}

Wrap a query point `xi` to the domain [x_min, x_max).
Used for periodic boundary conditions and extrap=:wrap.

Optimized: skips expensive `mod()` when xi is already in domain.
"""
@inline function _wrap_to_domain(xi::FT, x_min::FT, x_max::FT) where {FT<:AbstractFloat}
    # Single-branch check: outside domain → slow path
    if xi < x_min || xi >= x_max
        period = x_max - x_min
        return x_min + mod(xi - x_min, period)
    end
    # Fast path: already in domain (most common case)
    return xi
end

# ========================================
# Periodic BC Validation
# ========================================

# Tolerance for periodic endpoint check
# Use sqrt(eps) which is ~1e-8 for Float64, allowing for typical floating point errors
# (e.g., sin(2π) ≈ 2.45e-16 should pass)
const _PERIODIC_ATOL_F64 = 1e-12
const _PERIODIC_ATOL_F32 = 1f-6

"""
    _check_periodic_endpoints(y::AbstractVector)

Validate that y[1] ≈ y[end] for periodic boundary conditions.
Called once at construction time (zero runtime overhead).

Throws `ArgumentError` if endpoints differ significantly.
"""
@inline function _check_periodic_endpoints(y::AbstractVector{T}) where {T<:AbstractFloat}
    y1, yn = first(y), last(y)
    atol = T === Float32 ? _PERIODIC_ATOL_F32 : _PERIODIC_ATOL_F64
    if !isapprox(y1, yn; atol=atol)
        throw(ArgumentError(
            "Periodic BC requires y[1] ≈ y[end], got y[1]=$y1, y[end]=$yn (diff=$(abs(yn-y1)))"
        ))
    end
    return nothing
end


# ========================================
# Domain Validation Helpers
# ========================================

"""
    _check_domain(x, xi::T, ::Val{:none}) where {T<:AbstractFloat}

Check if scalar query point is within domain for `:none` extrapolation mode.
Throws `DomainError` if `xi` is outside `[first(x), last(x)]`.

Uses `@boundscheck` so it's skipped in `@inbounds` blocks for vector paths
that do a single upfront check via the vector dispatch.
"""
@inline function _check_domain(x::AbstractVector{FT}, xi::FT, ::Val{:none}) where {FT<:AbstractFloat}
    x_min, x_max = first(x), last(x)
    (xi < x_min || xi > x_max) && throw(DomainError(xi, "query point outside interpolation domain [$x_min, $x_max]"))
    return nothing
end

"""
    _check_domain(x, xi::T, ::Val) where {T<:AbstractFloat}

No-op domain check for extrapolation modes other than `:none`.
"""
@inline _check_domain(::AbstractVector{FT}, ::FT, ::Val) where {FT<:AbstractFloat} = nothing

"""
    _check_domain(x, xi::AbstractVector{T}, ::Val{:none}) where {T<:AbstractFloat}

Vector-level domain check using minimum/maximum (faster than extrema due to SIMD).
Called once before vector loop, then scalar `_check_domain` is skipped via `@inbounds`.
"""
@inline function _check_domain(x::AbstractVector{FT}, xi::AbstractVector{FT}, ::Val{:none}) where {FT<:AbstractFloat}
    x_min, x_max = first(x), last(x)
    # NOTE: Using minimum/maximum for potential SIMD optimization over extrema
    # extrema can be ~30x slower than minimum/maximum
    xq_min, xq_max = minimum(xi), maximum(xi)
    (xq_min < x_min || xq_max > x_max) && throw(DomainError(
        xq_min < x_min ? xq_min : xq_max,
        "query point outside interpolation domain [$x_min, $x_max]"
    ))
    return nothing
end

"""
    _check_domain(x, xi::AbstractVector{T}, ::Val) where {T<:AbstractFloat}

No-op vector domain check for extrapolation modes other than `:none`.
"""
@inline _check_domain(::AbstractVector{FT}, ::AbstractVector{FT}, ::Val) where {FT<:AbstractFloat} = nothing

# ========================================
# Validation Utilities
# ========================================
#
# Centralized validation for keyword arguments.
# @inline ensures zero overhead - compiler inlines the check.

"""
    _validate_extrap(extrap::Symbol) -> Nothing

Validate extrapolation mode symbol. Throws `ArgumentError` if invalid.

Valid options: `:none`, `:constant`, `:extension`, `:wrap`
"""
@inline function _validate_extrap(extrap::Symbol)
    extrap in (:none, :constant, :extension, :wrap) && return nothing
    throw(ArgumentError("`extrap` must be :none, :constant, :extension, or :wrap, got :$extrap"))
end

"""
    _validate_bc(bc::Symbol) -> Nothing

Validate boundary condition symbol for cubic interpolation.
Throws `ArgumentError` if invalid.

Valid options: `:natural`, `:periodic`
"""
@inline function _validate_bc(bc::Symbol)
    bc in (:natural, :periodic) && return nothing
    throw(ArgumentError("bc must be :natural or :periodic, got :$bc"))
end

# ========================================
# Dispatch Macros (Zero-Allocation Branching)
# ========================================
#
# These macros expand to manual if-elseif blocks that avoid union-splitting issues.
# Each branch calls with a concrete Val(:literal), ensuring zero-allocation dispatch.

"""
    @_dispatch_extrap sym => varname body

Dispatch on runtime extrapolation symbol, executing body with concrete Val type.

# Arguments
- `sym => varname`: Pair of symbol variable and binding name for Val type
- `body`: Expression to execute with `varname` bound to concrete Val

# Example
```julia
@_dispatch_extrap extrap => ev begin
    _cubic_interp_impl!(output, cache, y, x_query, ev)
end
```

Expands to:
```julia
let _mode = extrap
    if _mode === :none
        ev = Val(:none)
        _cubic_interp_impl!(output, cache, y, x_query, ev)
    elseif _mode === :constant
        ...
    end
end
```
"""
macro _dispatch_extrap(pair, body)
    # Parse pair: extrap => ev becomes Expr(:call, :(=>), :extrap, :ev)
    pair.head === :call && pair.args[1] === :(=>) ||
        error("@_dispatch_extrap expects `sym => varname`, got: $pair")
    sym = pair.args[2]
    varname = pair.args[3]
    evs = esc(varname)
    quote
        let _mode = $(esc(sym))
            if _mode === :none
                $evs = Val(:none)
                $(esc(body))
            elseif _mode === :constant
                $evs = Val(:constant)
                $(esc(body))
            elseif _mode === :extension
                $evs = Val(:extension)
                $(esc(body))
            elseif _mode === :wrap
                $evs = Val(:wrap)
                $(esc(body))
            else
                throw(ArgumentError("`extrap` must be :none, :constant, :extension, or :wrap, got :$_mode"))
            end
        end
    end
end

