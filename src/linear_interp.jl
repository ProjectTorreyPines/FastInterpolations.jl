# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH - OPTIMIZED CORE                         ║
# ║                  All arguments have same FT<:AbstractFloat                ║
# ║                      Zero type conversion overhead                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation (in-place, zero-allocation)
# ========================================

"""
    linear_interp!(output, x, y, x_targets; extrapolation=:extension)

Zero-allocation linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: O(log n) binary search

# Arguments
- `output`: Pre-allocated output vector (must be floating-point type)
- `extrapolation::Symbol`: `:extension` (default, linear extrapolation) or `:constant`

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
out = Vector{Float64}(undef, 2)
linear_interp!(out, rho, y, [0.55, 0.77])  # linear extrapolation (default)
linear_interp!(out, rho, y, [-0.1, 1.2]; extrapolation=:constant)  # constant extrapolation
```

# Implementation Note
- Optimized core works with `AbstractFloat` types (calls optimized scalar version)
- Integer/Real inputs automatically promoted via wrapper methods
"""
function linear_interp! end

# Unified method for AbstractVector{FT} (includes AbstractRange via dispatch)
function linear_interp!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    extrapolation::Symbol=:extension
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert extrapolation in (:constant, :extension) "extrapolation must be :constant or :extension"

    extrap_val = Val(extrapolation)  # Create Val once outside loop

    # Calls optimized scalar version with Val dispatch - zero runtime branches
    @inbounds for i in eachindex(x_targets, output)
        output[i] = linear_interp(x, y, x_targets[i], extrap_val)
    end

    return output
end

# Specific method for AbstractRange{FT} (resolves ambiguity with Real wrappers)
@inline function linear_interp!(
    output::AbstractVector{FT},
    x::AbstractRange{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    extrapolation::Symbol=:extension
) where {FT<:AbstractFloat}
    # @assert length(y) == length(x) "x and y must have same length"
    # @assert length(output) == length(x_targets) "output must match x_targets length"
    # @assert extrapolation in (:constant, :extension) "extrapolation must be :constant or :extension"

    extrap_val = Val(extrapolation)  # Create Val once outside loop
    # Calls optimized scalar version with Val dispatch
    @inbounds for i in eachindex(output)
        output[i] = linear_interp(x, y, x_targets[i], extrap_val)
    end

    return output
end

# ========================================
# Scalar interpolation (zero-allocation)
# ========================================

"""
    linear_interp(x, y, xi::Real; extrapolation=:extension) -> AbstractFloat

Zero-allocation scalar linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: O(log n) binary search

# Arguments
- `xi::Real`: Single interpolation point
- `extrapolation::Symbol`: `:extension` (default, linear extrapolation) or `:constant`

# Returns
- Always returns a floating-point type (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
value = linear_interp(rho, y, 0.55)  # Returns Float64, zero allocation

# Integer inputs auto-promoted to Float
x_int = 0:10
y_int = x_int.^2
value = linear_interp(x_int, y_int, 5.5)  # Returns Float64 (not Int)
```

# Implementation Note
- Optimized core works with `AbstractFloat` types only (zero conversion overhead)
- Integer/Real inputs automatically promoted to Float via wrapper methods
- Uses Val dispatch for extrapolation to eliminate runtime branches
"""

# Core implementation with Val dispatch (compile-time specialization)
@inline function linear_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xi::FT,
    extrap::Val
)::FT where {FT<:AbstractFloat}
    idx, x0, x1 = _find_idx(x, xi)
    α = _compute_alpha(x0, x1, xi, extrap)
    @inbounds return y[idx] * (one(FT) - α) + y[idx + 1] * α
end

# Public API - Symbol dispatch (converts to Val)
@inline function linear_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xi::FT;
    extrapolation::Symbol=:extension
)::FT where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))
    @boundscheck extrapolation in (:constant, :extension) || throw(ArgumentError("extrapolation must be :constant or :extension"))
    return linear_interp(x, y, xi, Val(extrapolation))
end

# Specific method for AbstractRange{FT} (resolves ambiguity with Real wrappers)
@inline function linear_interp(
    x::AbstractRange{FT},
    y::AbstractVector{FT},
    xi::FT;
    extrapolation::Symbol=:extension
)::FT where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))
    @boundscheck extrapolation in (:constant, :extension) || throw(ArgumentError("extrapolation must be :constant or :extension"))
    return linear_interp(x, y, xi, Val(extrapolation))
end

# ========================================
# Helper functions (2-step dispatch pattern)
# ========================================
# Step 1: _find_idx - Dispatches on grid type (Range O(1) vs Vector O(log n))
# Step 2: _compute_alpha - Dispatches on extrapolation (constant clamps, extension doesn't)

# Find interpolation index - O(1) for AbstractRange (direct indexing)
@inline function _find_idx(
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

# Find interpolation index - O(log n) for AbstractVector (binary search)
@inline function _find_idx(
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
                mid = (lo + hi) >> 1
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

# Compute alpha with constant extrapolation (clamp to [0, 1])
@inline function _compute_alpha(
    x0::FT,
    x1::FT,
    xi::FT,
    ::Val{:constant}
) where {FT<:AbstractFloat}
    α = (xi - x0) / (x1 - x0)
    return clamp(α, zero(FT), one(FT))
end

# Compute alpha with extension extrapolation (no clamping)
@inline function _compute_alpha(
    x0::FT,
    x1::FT,
    xi::FT,
    ::Val{:extension}
) where {FT<:AbstractFloat}
    return (xi - x0) / (x1 - x0)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        GENERIC WRAPPERS - CONVENIENCE                     ║
# ║              Auto-promote Real types to Float (type conversion)           ║
# ║                     Integer inputs → Float64 outputs                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation - Allocating version
# ========================================

"""
    linear_interp(x, y, x_targets; extrapolation=:extension)

Linear interpolation with automatic dispatch (allocating version):
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: O(log n) binary search

# Arguments
- `extrapolation::Symbol`: `:extension` (default, linear extrapolation) or `:constant`

# Returns
- Always returns a floating-point vector (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
result = linear_interp(rho, y, [0.55, 0.77])  # linear extrapolation (default)
result = linear_interp(rho, y, [-0.1, 1.2]; extrapolation=:constant)  # constant extrapolation

# Integer inputs auto-promoted to Float
x_int = 0:10
y_int = x_int.^2
result = linear_interp(x_int, y_int, [5.5, 7.3])  # Returns Vector{Float64}
```
"""
function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrapolation::Symbol=:extension
) where {T<:Real, S<:Real}
    FT = float(T)
    output = Vector{FT}(undef, length(x_targets))
    linear_interp!(output, x, y, x_targets; extrapolation)
    return output
end

# ========================================
# Vector interpolation - Real wrappers (in-place)
# ========================================

# Wrapper for AbstractRange with Real types (requires conversion)
function linear_interp!(
    output::AbstractVector,
    x::AbstractRange{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrapolation::Symbol=:extension
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert extrapolation in (:constant, :extension) "extrapolation must be :constant or :extension"

    FT = float(T)
    x_float = range(FT(first(x)), FT(last(x)), length(x))
    y_float = FT.(y)  # Allocate once
    extrap_val = Val(extrapolation)  # Create Val once outside loop

    # Call optimized scalar version for each point (with Val dispatch)
    @inbounds for i in eachindex(x_targets, output)
        output[i] = linear_interp(x_float, y_float, FT(x_targets[i]), extrap_val)
    end

    return output
end

# Wrapper for AbstractVector with Real types (requires conversion)
function linear_interp!(
    output::AbstractVector,
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrapolation::Symbol=:extension
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"
    @assert extrapolation in (:constant, :extension) "extrapolation must be :constant or :extension"

    FT = float(T)
    x_float = FT.(x)  # Allocate once
    y_float = FT.(y)  # Allocate once
    extrap_val = Val(extrapolation)  # Create Val once outside loop

    # Call optimized scalar version for each point (with Val dispatch)
    @inbounds for i in eachindex(x_targets, output)
        output[i] = linear_interp(x_float, y_float, FT(x_targets[i]), extrap_val)
    end

    return output
end

# ========================================
# Scalar interpolation - Real wrappers
# ========================================

# Wrapper for AbstractRange with Real types (requires conversion)
@inline function linear_interp(
    x::AbstractRange{T},
    y::AbstractVector{T},
    xi::S;
    extrapolation::Symbol=:extension
) where {T<:Real, S<:Real}
    FT = float(T)
    x_float = range(FT(first(x)), FT(last(x)), length(x))
    return linear_interp(x_float, FT.(y), FT(xi); extrapolation)
end

function linear_interp(
    x::AbstractRange{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrapolation::Symbol=:extension
) where {T<:Real, S<:Real}
    output = Vector{float(T)}(undef, length(x_targets))
    return linear_interp!(output, x, y, x_targets; extrapolation)
end


# Wrapper for AbstractVector with Real types (requires conversion)
@inline function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xi::S;
    extrapolation::Symbol=:extension
) where {T<:Real, S<:Real}
    FT = float(T)
    return linear_interp(FT.(x), FT.(y), FT(xi); extrapolation)
end

# ========================================
# Callable Interpolator for Broadcast Fusion
# ========================================

"""
    LinearInterpCallable{T,X,Y}

Lightweight callable interpolator for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `extrap::Val`: Extrapolation method (Val(:extension) or Val(:constant))

# Usage
```julia
# Create interpolator (minimal allocation)
itp = linear_interp(x, y)

# Use in broadcast (fused, no intermediate arrays)
result = @. coef * itp(rho) * other_terms

# Reuse interpolator multiple times
vals1 = itp.(query_points1)
vals2 = @. compute(itp(query_points2))
```
"""
struct LinearInterpCallable{T<:AbstractFloat,X<:AbstractVector{T},Y<:AbstractVector{T}}
    x::X
    y::Y
    extrap::Val

    function LinearInterpCallable(
        x::X,
        y::Y;
        extrapolation::Symbol=:extension
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        @assert length(x) == length(y) "x and y must have same length"
        @assert extrapolation in (:constant, :extension) "extrapolation must be :constant or :extension"
        new{T,X,Y}(x, y, Val(extrapolation))
    end
end

# Scalar call - hot path (inlined for broadcast fusion)
@inline function (itp::LinearInterpCallable{T})(xi::T) where {T<:AbstractFloat}
    linear_interp(itp.x, itp.y, xi, itp.extrap)
end

# Real scalar wrapper for convenience
@inline function (itp::LinearInterpCallable{T})(xi::S) where {T<:AbstractFloat, S<:Real}
    linear_interp(itp.x, itp.y, T(xi), itp.extrap)
end

# Vector call - optimized to avoid type reflection
function (itp::LinearInterpCallable{T,X,Y})(xi::AbstractVector{S}) where {T<:AbstractFloat, X, Y, S<:Real}
    output = Vector{T}(undef, length(xi))
    xi_typed = S === T ? xi : T.(xi)
    @inbounds for i in eachindex(xi, output)
        output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.extrap)
    end
    return output
end

# Optimized path when xi element type matches T (zero conversion)
function (itp::LinearInterpCallable{T,X,Y})(xi::AbstractVector{T}) where {T<:AbstractFloat, X, Y}
    output = Vector{T}(undef, length(xi))
    @inbounds for i in eachindex(xi, output)
        output[i] = linear_interp(itp.x, itp.y, xi[i], itp.extrap)
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    linear_interp(x, y; extrapolation=:extension) -> LinearInterpCallable

Create a callable interpolator for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `extrapolation::Symbol`: `:extension` (default) or `:constant`

# Returns
`LinearInterpCallable` object that can be:
- Called with scalar: `itp(0.5)`
- Broadcasted: `itp.(rho)` or `@. coef * itp(rho)`
- Reused multiple times without re-creating

# Examples
```julia
# Create once, reuse multiple times
itp = linear_interp(x_data, y_data)

# Scalar call
val = itp(0.5)

# Broadcast (creates array)
vals = itp.(query_points)

# Fused broadcast (optimal - no intermediate arrays)
result = @. coefficient * itp(rho) * ne / Te^2

# Compare with 3-argument form (returns array immediately)
vals_direct = linear_interp(x_data, y_data, query_points)
```

# Performance Notes
- 2-argument: Returns lightweight callable (~48 bytes on 64-bit)
- Best for: Reuse, broadcast fusion in complex expressions
- 3-argument: Returns array, best for single immediate use
- Callable eliminates closure overhead (4x faster than anonymous functions)
"""
function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrapolation::Symbol=:extension
) where {T<:AbstractFloat}
    return LinearInterpCallable(x, y; extrapolation)
end

# ========================================
# Type Conversion Helpers
# ========================================

# Convert to float type while preserving Range structure (O(1) index lookup)
# FT.(x) would convert Range to Vector, losing O(1) optimization
_to_float(x::AbstractRange, ::Type{FT}) where {FT<:AbstractFloat} =
    range(FT(first(x)), FT(last(x)), length(x))
_to_float(x::AbstractVector, ::Type{FT}) where {FT<:AbstractFloat} = FT.(x)

# Real wrapper for 2-argument form (allows different container types)
function linear_interp(
    x::X,
    y::Y;
    extrapolation::Symbol=:extension
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    T = promote_type(TX, TY)
    FT = float(T)
    return LinearInterpCallable(_to_float(x, FT), FT.(y); extrapolation)
end
