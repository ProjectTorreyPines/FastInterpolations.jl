# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH - OPTIMIZED CORE                         ║
# ║                  All arguments have same FT<:AbstractFloat                ║
# ║                      Zero type conversion overhead                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation (in-place, zero-allocation)
# ========================================

"""
    linear_interp!(output, x, y, x_targets; extrap=:none)

Zero-allocation linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: O(log n) binary search

# Arguments
- `output`: Pre-allocated output vector (must be floating-point type)
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
out = Vector{Float64}(undef, 2)
linear_interp!(out, rho, y, [0.55, 0.77])  # throws error if outside domain
linear_interp!(out, rho, y, [-0.1, 1.2]; extrap=:extension)  # linear extrapolation
linear_interp!(out, rho, y, [-0.1, 1.2]; extrap=:constant)  # clamp to boundary values
linear_interp!(out, rho, y, [1.5, 2.5]; extrap=:wrap)  # wrap to domain (periodic)
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
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x, x_targets, ev)
            _linear_interp_loop!(output, x, y, x_targets, ev, op)
        end
    end
end

# Internal loop with Val dispatch (type-stable)
@inline function _linear_interp_loop!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT},
    extrap_val::Val,
    op::O
) where {FT<:AbstractFloat, O<:AbstractEvalOp}
    @inbounds for i in eachindex(x_targets, output)
        output[i] = linear_interp(x, y, x_targets[i], extrap_val, op)
    end
    return output
end


# Optimized loop for :wrap - uses 2-stage strategy
# Stage 1: Check if ALL queries are inside domain (cheap: ~150ns for 1000 elements)
# Stage 2: If all inside, use extension path (no wrap needed); otherwise per-element wrap
@inline function _linear_interp_loop!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT},
    ::Val{:wrap},
    op::O
) where {FT<:AbstractFloat, O<:AbstractEvalOp}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(x_targets), maximum(x_targets)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain - use extension (no wrap overhead)
        @inbounds for i in eachindex(x_targets, output)
            output[i] = linear_interp(x, y, x_targets[i], Val(:extension), op)
        end
    else
        # Slow path: some queries outside - per-element wrap
        @inbounds for i in eachindex(x_targets, output)
            output[i] = linear_interp(x, y, x_targets[i], Val(:wrap), op)
        end
    end
    return output
end

# Same optimization for AbstractRange (O(1) indexing path)
@inline function _linear_interp_loop!(
    output::AbstractVector{FT},
    x::AbstractRange{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT},
    ::Val{:wrap},
    op::O
) where {FT<:AbstractFloat, O<:AbstractEvalOp}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(x_targets), maximum(x_targets)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(x_targets, output)
            output[i] = linear_interp(x, y, x_targets[i], Val(:extension), op)
        end
    else
        @inbounds for i in eachindex(x_targets, output)
            output[i] = linear_interp(x, y, x_targets[i], Val(:wrap), op)
        end
    end
    return output
end

# Specific method for AbstractRange{FT} (resolves ambiguity with Real wrappers)
@inline function linear_interp!(
    output::AbstractVector{FT},
    x::AbstractRange{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT};
    extrap::Symbol=:none,
    deriv::Int=0
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x, x_targets, ev)
            _linear_interp_loop!(output, x, y, x_targets, ev, op)
        end
    end
end

# ========================================
# Scalar interpolation (zero-allocation)
# ========================================

"""
    linear_interp(x, y, xq::Real; extrap=:none) -> AbstractFloat

Zero-allocation scalar linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: O(log n) binary search

# Arguments
- `xq::Real`: Single interpolation query point
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`

# Returns
- Always returns a floating-point type (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
value = linear_interp(rho, y, 0.55)  # Returns Float64, zero allocation
value = linear_interp(rho, y, 1.5; extrap=:wrap)  # wraps to domain

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

# ========================================
# Internal evaluation with op parameter
# ========================================

"""
    _linear_eval_at_point(x, y, xq, extrap, op) -> FT

Core linear interpolation evaluation using kernel function.
Supports value (EvalValue), first derivative (EvalDeriv1), and second derivative (EvalDeriv2).
"""
@inline function _linear_eval_at_point(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    extrap::Val,
    op::O
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp}
    @boundscheck _check_domain(x, xq, extrap)
    idx, xL, xR = _find_interval_with_bounds(x, xq)
    h = xR - xL
    dL = xq - xL
    @inbounds return _linear_kernel(op, y[idx], y[idx + 1], h, dL)
end


"""
    _linear_eval_constant_extrap(y, is_left, op) -> FT

Handle constant extrapolation: returns boundary value for EvalValue, zero for derivatives.
"""
@inline function _linear_eval_constant_extrap(
    y::AbstractVector{FT},
    is_left::Bool,
    ::EvalValue
)::FT where {FT<:AbstractFloat}
    @inbounds return is_left ? y[1] : y[end]
end

@inline function _linear_eval_constant_extrap(
    ::AbstractVector{FT},
    ::Bool,
    ::EvalDeriv1
)::FT where {FT<:AbstractFloat}
    return zero(FT)
end

@inline function _linear_eval_constant_extrap(
    ::AbstractVector{FT},
    ::Bool,
    ::EvalDeriv2
)::FT where {FT<:AbstractFloat}
    return zero(FT)
end

"""
    _linear_with_extrap(x, y, xq, extrap, op) -> FT

Linear interpolation with extrapolation handling and op parameter.
"""
@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:none},
    op::O
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp}
    _linear_eval_at_point(x, y, xq, Val(:none), op)
end

@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:extension},
    op::O
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp}
    _linear_eval_at_point(x, y, xq, Val(:extension), op)
end

@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:constant},
    op::O
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp}
    x_min, x_max = first(x), last(x)
    if xq < x_min
        return _linear_eval_constant_extrap(y, true, op)
    elseif xq > x_max
        return _linear_eval_constant_extrap(y, false, op)
    else
        return _linear_eval_at_point(x, y, xq, Val(:extension), op)
    end
end

@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:wrap},
    op::O
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp}
    xi_wrapped = _wrap_to_domain(xq, first(x), last(x))
    _linear_eval_at_point(x, y, xi_wrapped, Val(:extension), op)
end


# ========================================
# Core implementation with Val dispatch
# ========================================

# Core implementation with Val + op dispatch
@inline function linear_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    extrap::Val,
    op::O
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp}
    _linear_with_extrap(x, y, xq, extrap, op)
end

# Public API - Symbol dispatch (converts to Val)
@inline function linear_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT;
    extrap::Symbol=:none,
    deriv::Int=0
)::FT where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            linear_interp(x, y, xq, ev, op)
        end
    end
end

# Specific method for AbstractRange{FT} (resolves ambiguity with Real wrappers)
@inline function linear_interp(
    x::AbstractRange{FT},
    y::AbstractVector{FT},
    xq::FT;
    extrap::Symbol=:none,
    deriv::Int=0
)::FT where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            linear_interp(x, y, xq, ev, op)
        end
    end
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
    linear_interp(x, y, x_targets; extrap=:none)

Linear interpolation with automatic dispatch (allocating version):
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: O(log n) binary search

# Arguments
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`

# Returns
- Always returns a floating-point vector (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
result = linear_interp(rho, y, [0.55, 0.77])  # throws error if outside domain
result = linear_interp(rho, y, [-0.1, 1.2]; extrap=:extension)  # linear extrap
result = linear_interp(rho, y, [-0.1, 1.2]; extrap=:constant)  # clamp to boundary values
result = linear_interp(rho, y, [1.5, 2.5]; extrap=:wrap)  # wrap to domain

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
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    output = Vector{FT}(undef, length(x_targets))
    linear_interp!(output, x, y, x_targets; extrap, deriv)
    return output
end

# ========================================
# Vector interpolation - Real wrappers (in-place)
# ========================================

# Internal helper for Real wrappers (type-stable)
@inline function _linear_interp_real_loop!(output, x_float, y_float, x_targets_float, extrap_val::Val, op::O) where {O<:AbstractEvalOp}
    @boundscheck _check_domain(x_float, x_targets_float, extrap_val)
    @inbounds for i in eachindex(x_targets_float, output)
        output[i] = linear_interp(x_float, y_float, x_targets_float[i], extrap_val, op)
    end
    return output
end


# Wrapper for AbstractRange with Real types (requires conversion)
function linear_interp!(
    output::AbstractVector,
    x::AbstractRange{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    FT = float(T)
    x_float = range(FT(first(x)), FT(last(x)), length(x))
    y_float = FT.(y)  # Allocate once
    x_targets_float = FT.(x_targets)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _linear_interp_real_loop!(output, x_float, y_float, x_targets_float, ev, op)
        end
    end
end

# Wrapper for AbstractVector with Real types (requires conversion)
function linear_interp!(
    output::AbstractVector,
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    FT = float(T)
    x_float = FT.(x)  # Allocate once
    y_float = FT.(y)  # Allocate once
    x_targets_float = FT.(x_targets)

    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _linear_interp_real_loop!(output, x_float, y_float, x_targets_float, ev, op)
        end
    end
end

# ========================================
# Scalar interpolation - Real wrappers
# ========================================

# Wrapper for AbstractRange with Real types (requires conversion)
@inline function linear_interp(
    x::AbstractRange{T},
    y::AbstractVector{T},
    xq::S;
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    x_float = range(FT(first(x)), FT(last(x)), length(x))
    return linear_interp(x_float, FT.(y), FT(xq); extrap, deriv)
end

function linear_interp(
    x::AbstractRange{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    output = Vector{float(T)}(undef, length(x_targets))
    return linear_interp!(output, x, y, x_targets; extrap, deriv)
end


# Wrapper for AbstractVector with Real types (requires conversion)
@inline function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xq::S;
    extrap::Symbol=:none,
    deriv::Int=0
) where {T<:Real, S<:Real}
    FT = float(T)
    return linear_interp(FT.(x), FT.(y), FT(xq); extrap, deriv)
end

# ========================================
# Callable Interpolator for Broadcast Fusion
# ========================================

"""
    LinearInterpolant{T,X,Y}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `linear_interp(x, y)` (2-argument form).

# Fields
- `x::X`: x-coordinates (sorted)
- `y::Y`: y-values
- `mode::Val`: Evaluation mode (Val(:none), Val(:extension), Val(:constant), or Val(:wrap))

# Usage
```julia
# Create interpolator (minimal allocation)
itp = linear_interp(x, y)  # default extrap=:none (throws error if outside domain)

# Use in broadcast (fused, no intermediate arrays)
result = @. coef * itp(rho) * other_terms

# Reuse interpolator multiple times
vals1 = itp.(query_points1)
vals2 = @. compute(itp(query_points2))

# Extrapolation options
itp_ext = linear_interp(x, y; extrap=:extension)  # linear extrap
itp_const = linear_interp(x, y; extrap=:constant)  # clamp to boundary values
itp_wrap = linear_interp(x, y; extrap=:wrap)  # wrap to domain
val = itp_wrap(2.5)  # wraps to domain
```
"""
struct LinearInterpolant{T<:AbstractFloat,X<:AbstractVector{T},Y<:AbstractVector{T}}
    x::X
    y::Y
    mode::ExtrapVal  # Evaluation mode (concrete union for union-splitting)

    function LinearInterpolant(
        x::X,
        y::Y;
        extrap::Symbol=:none
    ) where {T<:AbstractFloat, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        @assert length(x) == length(y) "x and y must have same length"

        # Manual dispatch to avoid union-splitting with 4 Val types
        @_dispatch_extrap extrap => ev begin
            return new{T,X,Y}(x, y, ev)
        end
    end
end

# Scalar call - hot path (inlined for broadcast fusion)
# Supports deriv keyword for derivative evaluation
@inline function (itp::LinearInterpolant{T})(xq::T; deriv::Int=0) where {T<:AbstractFloat}
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    @_dispatch_deriv deriv => op begin
        _linear_with_extrap(itp.x, itp.y, xq, itp.mode, op)
    end
end

# Real scalar wrapper - delegates to T method with deriv keyword
@inline function (itp::LinearInterpolant{T})(xq::S; deriv::Int=0) where {T<:AbstractFloat, S<:Real}
    itp(T(xq); deriv=deriv)
end

# Vector call with deriv keyword support
function (itp::LinearInterpolant{T,X,Y})(xq::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, X, Y, S<:Real}
    xi_typed = S === T ? xq : T.(xq)
    @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
    output = Vector{T}(undef, length(xi_typed))
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.mode, op)
        end
    end
    return output
end

# Optimized path when xq element type matches T (zero conversion)
function (itp::LinearInterpolant{T,X,Y})(xq::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat, X, Y}
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    output = Vector{T}(undef, length(xq))
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq, output)
            output[i] = linear_interp(itp.x, itp.y, xq[i], itp.mode, op)
        end
    end
    return output
end

# In-place vector call with deriv keyword support - zero allocation
function (itp::LinearInterpolant{T,X,Y})(output::AbstractVector{T}, xq::AbstractVector{T}; deriv::Int=0) where {T<:AbstractFloat, X, Y}
    @assert length(output) == length(xq) "output length must match xq length"
    @boundscheck _check_domain(itp.x, xq, itp.mode)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xq, output)
            output[i] = linear_interp(itp.x, itp.y, xq[i], itp.mode, op)
        end
    end
    return output
end

# In-place with type conversion and deriv keyword
function (itp::LinearInterpolant{T,X,Y})(output::AbstractVector, xq::AbstractVector{S}; deriv::Int=0) where {T<:AbstractFloat, X, Y, S<:Real}
    @assert length(output) == length(xq) "output length must match xq length"
    xi_typed = T.(xq)
    @boundscheck _check_domain(itp.x, xi_typed, itp.mode)
    @_dispatch_deriv deriv => op begin
        @inbounds for i in eachindex(xi_typed, output)
            output[i] = linear_interp(itp.x, itp.y, xi_typed[i], itp.mode, op)
        end
    end
    return output
end

# ========================================
# 2-Argument Form: Return Callable
# ========================================

"""
    linear_interp(x, y; extrap=:none) -> LinearInterpolant

Create a callable interpolant for broadcast fusion and reuse.

# Arguments
- `x::AbstractVector`: x-coordinates (must be sorted)
- `y::AbstractVector`: y-values
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`

# Returns
`LinearInterpolant` object that can be:
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

# Wrap to domain (for periodic-like data)
itp_wrap = linear_interp(x_data, y_data; extrap=:wrap)
val_wrap = itp_wrap(2.5)  # wraps to domain

# Compare with 3-argument form (returns array immediately)
vals_direct = linear_interp(x_data, y_data, query_points)
```

# Performance Notes
- Returns lightweight callable (~48 bytes), best for reuse and broadcast fusion
- 3-argument form returns array immediately, best for single use
"""
function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T};
    extrap::Symbol=:none
) where {T<:AbstractFloat}
    return LinearInterpolant(x, y; extrap)
end

# Real wrapper for 2-argument form (allows different container types)
# Uses _to_float from utils.jl to preserve Range structure
function linear_interp(
    x::X,
    y::Y;
    extrap::Symbol=:none
) where {TX<:Real, TY<:Real, X<:AbstractVector{TX}, Y<:AbstractVector{TY}}
    T = promote_type(TX, TY)
    FT = float(T)
    return LinearInterpolant(_to_float(x, FT), FT.(y); extrap)
end
