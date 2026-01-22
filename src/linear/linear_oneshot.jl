# ========================================
# Linear Interpolation Oneshot API
# ========================================
# Zero-allocation linear interpolation functions.
# Type definitions in linear_types.jl.
# Callable methods (2-arg form) in linear_interpolant.jl.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         HOT PATH - OPTIMIZED CORE                         ║
# ║                  All arguments have same FT<:AbstractFloat                ║
# ║                      Zero type conversion overhead                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ========================================
# Vector interpolation (in-place, zero-allocation)
# ========================================

"""
    linear_interp!(output, x, y, x_targets; extrap=:none, deriv=0, search=Binary())

Zero-allocation linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `output`: Pre-allocated output vector (must be floating-point type)
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`
- `deriv::Int`: Derivative order (0=value, 1=first derivative, 2=second derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `Binary()` (default): O(log n) binary search, stateless
  - `HintedBinary()`: O(1) if hint valid, O(log n) fallback
  - `LinearBinary(linear_window=8)`: Linear search within window, then binary fallback

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
out = Vector{Float64}(undef, 2)
linear_interp!(out, rho, y, [0.55, 0.77])  # throws error if outside domain
linear_interp!(out, rho, y, [-0.1, 1.2]; extrap=:extension)  # linear extrapolation

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
output = zeros(1000)
linear_interp!(output, x_vec, y_vec, sorted_queries; search=LinearBinary(linear_window=8))
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
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x, x_targets, ev)
            _linear_interp_loop!(output, x, y, x_targets, ev, op, searcher)
        end
    end
end

# Internal loop with Val dispatch and Searcher (type-stable)
@inline function _linear_interp_loop!(
    output::AbstractVector{FT},
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    x_targets::AbstractVector{FT},
    extrap_val::Val,
    op::O,
    searcher::S
) where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    @inbounds for i in eachindex(x_targets, output)
        output[i] = linear_interp(x, y, x_targets[i], extrap_val, op, searcher)
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
    op::O,
    searcher::S
) where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(x_targets), maximum(x_targets)

    if qmin >= x_min && qmax < x_max
        # Fast path: all queries inside domain - use extension (no wrap overhead)
        @inbounds for i in eachindex(x_targets, output)
            output[i] = _linear_eval_at_point(x, y, x_targets[i], Val(:extension), op, searcher)
        end
    else
        # Slow path: some queries outside - per-element wrap
        @inbounds for i in eachindex(x_targets, output)
            xi_wrapped = _wrap_to_domain(x_targets[i], x_min, x_max)
            output[i] = _linear_eval_at_point(x, y, xi_wrapped, Val(:extension), op, searcher)
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
    op::O,
    searcher::S
) where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    x_min, x_max = first(x), last(x)
    qmin, qmax = minimum(x_targets), maximum(x_targets)

    if qmin >= x_min && qmax < x_max
        @inbounds for i in eachindex(x_targets, output)
            output[i] = _linear_eval_at_point(x, y, x_targets[i], Val(:extension), op, searcher)
        end
    else
        @inbounds for i in eachindex(x_targets, output)
            xi_wrapped = _wrap_to_domain(x_targets[i], x_min, x_max)
            output[i] = _linear_eval_at_point(x, y, xi_wrapped, Val(:extension), op, searcher)
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
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {FT<:AbstractFloat}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            @boundscheck _check_domain(x, x_targets, ev)
            _linear_interp_loop!(output, x, y, x_targets, ev, op, searcher)
        end
    end
end

# ========================================
# Scalar interpolation (zero-allocation)
# ========================================

"""
    linear_interp(x, y, xq::Real; extrap=:none, deriv=0, search=Binary()) -> AbstractFloat

Zero-allocation scalar linear interpolation with automatic dispatch:
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `xq::Real`: Single interpolation query point
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`
- `deriv::Int`: Derivative order (0=value, 1=first derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding
  - `Binary()` (default): O(log n) binary search, stateless
  - `HintedBinary()`: O(1) if hint valid, O(log n) fallback
  - `LinearBinary(linear_window=8)`: Linear search within window, then binary fallback

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
    _linear_eval_at_point(x, y, xq, extrap, op, searcher) -> FT

Core linear interpolation evaluation using kernel function and search policy.
Supports value (EvalValue), first derivative (EvalDeriv1), and second derivative (EvalDeriv2).
"""
@inline function _linear_eval_at_point(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    extrap::Val,
    op::O,
    searcher::S
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    @boundscheck _check_domain(x, xq, extrap)
    idx, xL, xR = search_interval(searcher, x, xq)
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

@inline function _linear_eval_constant_extrap(
    ::AbstractVector{FT},
    ::Bool,
    ::EvalDeriv3
)::FT where {FT<:AbstractFloat}
    return zero(FT)
end

"""
    _linear_with_extrap(x, y, xq, extrap, op, searcher) -> FT

Linear interpolation with extrapolation handling, op parameter, and search policy.
"""
@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:none},
    op::O,
    searcher::S
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    _linear_eval_at_point(x, y, xq, Val(:none), op, searcher)
end

@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:extension},
    op::O,
    searcher::S
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    _linear_eval_at_point(x, y, xq, Val(:extension), op, searcher)
end

@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:constant},
    op::O,
    searcher::S
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    x_min, x_max = first(x), last(x)
    if xq < x_min
        return _linear_eval_constant_extrap(y, true, op)
    elseif xq > x_max
        return _linear_eval_constant_extrap(y, false, op)
    else
        return _linear_eval_at_point(x, y, xq, Val(:extension), op, searcher)
    end
end

@inline function _linear_with_extrap(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    ::Val{:wrap},
    op::O,
    searcher::S
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    xi_wrapped = _wrap_to_domain(xq, first(x), last(x))
    _linear_eval_at_point(x, y, xi_wrapped, Val(:extension), op, searcher)
end


# ========================================
# Core implementation with Val dispatch
# ========================================

# Core implementation with Val + op + searcher dispatch
@inline function linear_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT,
    extrap::Val,
    op::O,
    searcher::S
)::FT where {FT<:AbstractFloat, O<:AbstractEvalOp, S<:Searcher}
    _linear_with_extrap(x, y, xq, extrap, op, searcher)
end

# Public API - Symbol dispatch (converts to Val)
@inline function linear_interp(
    x::AbstractVector{FT},
    y::AbstractVector{FT},
    xq::FT;
    extrap::Symbol=:none,
    deriv::Int=0,
    search=Binary(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
)::FT where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            linear_interp(x, y, xq, ev, op, searcher)
        end
    end
end

# Specific method for AbstractRange{FT} (resolves ambiguity with Real wrappers)
@inline function linear_interp(
    x::AbstractRange{FT},
    y::AbstractVector{FT},
    xq::FT;
    extrap::Symbol=:none,
    deriv::Int=0,
    search=Binary(),
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
)::FT where {FT<:AbstractFloat}
    @boundscheck length(y) == length(x) || throw(ArgumentError("x and y must have same length"))

    searcher = _to_searcher(search, hint)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            linear_interp(x, y, xq, ev, op, searcher)
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
    linear_interp(x, y, x_targets; extrap=:none, deriv=0, search=Binary())

Linear interpolation with automatic dispatch (allocating version):
- For `AbstractRange` x: O(1) direct indexing
- For general `AbstractVector` x: Search algorithm determined by `search` parameter

# Arguments
- `extrap::Symbol`: `:none` (default, throws DomainError), `:constant`, `:extension`, or `:wrap`
- `deriv::Int`: Derivative order (0=value, 1=first derivative)
- `search::AbstractSearchPolicy`: Search algorithm for interval finding

# Returns
- Always returns a floating-point vector (Integer inputs auto-promoted to Float)

# Example
```julia
rho = 0.0:0.01:1.0  # Uniform grid → fast O(1) path
y = sin.(rho)
result = linear_interp(rho, y, [0.55, 0.77])  # throws error if outside domain
result = linear_interp(rho, y, [-0.1, 1.2]; extrap=:extension)  # linear extrap

# Optimized for sorted queries
sorted_queries = sort(rand(1000))
vals = linear_interp(x_vec, y_vec, sorted_queries; search=LinearBinary(linear_window=8))
```
"""
function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:Real, S<:Real}
    FT = float(T)
    output = Vector{FT}(undef, length(x_targets))
    linear_interp!(output, x, y, x_targets; extrap, deriv, search)
    return output
end

# ========================================
# Vector interpolation - Real wrappers (in-place)
# ========================================

# Internal helper for Real wrappers (type-stable)
@inline function _linear_interp_real_loop!(
    output, x_float, y_float, x_targets_float, extrap_val::Val, op::O, searcher::S
) where {O<:AbstractEvalOp, S<:Searcher}
    @boundscheck _check_domain(x_float, x_targets_float, extrap_val)
    @inbounds for i in eachindex(x_targets_float, output)
        output[i] = linear_interp(x_float, y_float, x_targets_float[i], extrap_val, op, searcher)
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
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    FT = float(T)
    x_float = range(FT(first(x)), FT(last(x)), length(x))
    y_float = FT.(y)  # Allocate once
    x_targets_float = FT.(x_targets)

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _linear_interp_real_loop!(output, x_float, y_float, x_targets_float, ev, op, searcher)
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
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:Real, S<:Real}
    @assert length(y) == length(x) "x and y must have same length"
    @assert length(output) == length(x_targets) "output must match x_targets length"

    FT = float(T)
    x_float = FT.(x)  # Allocate once
    y_float = FT.(y)  # Allocate once
    x_targets_float = FT.(x_targets)

    searcher = _to_searcher(search)
    @_dispatch_deriv deriv => op begin
        @_dispatch_extrap extrap => ev begin
            _linear_interp_real_loop!(output, x_float, y_float, x_targets_float, ev, op, searcher)
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
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:Real, S<:Real}
    FT = float(T)
    x_float = range(FT(first(x)), FT(last(x)), length(x))
    return linear_interp(x_float, FT.(y), FT(xq); extrap, deriv, search)
end

function linear_interp(
    x::AbstractRange{T},
    y::AbstractVector{T},
    x_targets::AbstractVector{S};
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:Real, S<:Real}
    output = Vector{float(T)}(undef, length(x_targets))
    return linear_interp!(output, x, y, x_targets; extrap, deriv, search)
end


# Wrapper for AbstractVector with Real types (requires conversion)
@inline function linear_interp(
    x::AbstractVector{T},
    y::AbstractVector{T},
    xq::S;
    extrap::Symbol=:none,
    deriv::Int=0,
    search::AbstractSearchPolicy=Binary()
) where {T<:Real, S<:Real}
    FT = float(T)
    return linear_interp(FT.(x), FT.(y), FT(xq); extrap, deriv, search)
end
