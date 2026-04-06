# ========================================
# Coefficient Strategy Types
# ========================================
# Abstract and concrete types for PreCompute/OnTheFly coefficient strategies.
# Defined in core/ so all interpolant families can reference them.
# Resolution logic is in coeff_policy.jl (included later, after method types).

"""
    AbstractCoeffStrategy

Abstract supertype for coefficient computation strategies in ND interpolation.

# Implemented Strategies
- [`PreCompute`](@ref): Precompute all partial derivatives at construction (O(1) query)
- [`OnTheFly`](@ref): Compute coefficients lazily at query time (O(n) query)
- [`AutoCoeffs`](@ref): Automatic selection based on context
"""
abstract type AbstractCoeffStrategy end

"""
    PreCompute <: AbstractCoeffStrategy

Precompute all partial derivatives at construction time.

For N-dimensional interpolation, stores 2^N partial derivatives per grid point.

# Trade-offs
- **Memory**: O(2^N × n^N) - higher than OnTheFly
- **Construction**: O(N × n^N) - expensive (N passes of 1D spline solving)
- **Query**: O(1) - ultra-fast (just polynomial evaluation)
"""
struct PreCompute <: AbstractCoeffStrategy end

"""
    OnTheFly <: AbstractCoeffStrategy

Compute coefficients lazily at query time using tensor-product (separable) approach.

# Trade-offs
- **Memory**: O(n^N) - minimal (only original data)
- **Construction**: O(1) - instant (just store data reference)
- **Query**: O(n) per axis - slower (must solve 1D systems)
"""
struct OnTheFly <: AbstractCoeffStrategy end

"""
    AutoCoeffs <: AbstractCoeffStrategy

Automatic coefficient strategy selection based on context.

# Rules
- 1D scalar query → `OnTheFly()` (O(1) local slopes beats O(n) bulk)
- 1D vector query → `length(xq) > length(x)` ? `PreCompute()` : `OnTheFly()`
- 1D interpolant → `PreCompute()` (amortize for reuse)
- ND trivial methods (Linear/Constant) → `PreCompute()` (no slopes)
- ND N ≥ 3 → `OnTheFly()` (2^N memory savings)
- ND any local Hermite method → `OnTheFly()`
"""
struct AutoCoeffs <: AbstractCoeffStrategy end
