# ========================================
# Abstract Type Hierarchy
# ========================================
# Abstract types for interpolant type hierarchy.
# Enables generic programming over different interpolation methods.
#
# DESIGN: Type hierarchy with minimal shared interface.
# Only universally applicable methods (eltype, grid_type, value_type, eval_type) are defined here.
#
# TYPE PARAMETERS:
# - Tg: Grid/coordinate type (AbstractFloat) - used for x-coordinates, spacing, search
# - Tv: Value type (unconstrained) - used for y-values, coefficients, return values
#       Any type supporting +, -, scalar * (see Custom Value Types guide)
#
# Include order: FIRST - before all other interp files
#
# ========================================

"""
    AbstractInterpolant{Tg<:AbstractFloat, Tv}

Abstract supertype for all interpolant objects.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64) - used for x-coordinates, spacing, search
- `Tv`: Value type (unconstrained) - used for y-values, coefficients, return values.
        Any type supporting the 5 core operations (+, -, Tg*Tv, Tv*Tg, Int*Tv).

# Design Invariant
- Grid operations (search, spacing) always use Tg
- Value operations (kernel, coefficients) use Tv
- Evaluation returns type based on promote_type(Tv, query_type)

# Subtypes
- `AbstractInterpolant1D{Tg, Tv}`: 1D interpolants (shared callable protocol)
- `AbstractSeriesInterpolant{Tg, Tv}`: Multi-series interpolants
- `AbstractInterpolantND{Tg, Tv, N}`: N-dimensional interpolants
"""
abstract type AbstractInterpolant{Tg <: AbstractFloat, Tv} end

"""
    AbstractInterpolant1D{Tg<:AbstractFloat, Tv} <: AbstractInterpolant{Tg, Tv}

Abstract supertype for 1-dimensional interpolant objects.

All four 1D interpolant types inherit from this, enabling shared callable
dispatch via the `interpolant_protocol.jl` interface.

# Protocol (interpolant_protocol.jl)
Subtypes automatically inherit 3 callable overloads (scalar, vector-alloc, vector-inplace)
by implementing the required interface:

    _itp_eval_scalar(itp, xq, extrap, op, searcher)   — core scalar evaluation (required)
    _itp_vector_loop!(out, itp, xq, extrap, op, searcher) — core vector loop (required)
    _itp_grid(itp)                  — grid vector access (default: itp.x)
    _itp_extrap(itp)                — extrapolation mode (default: itp.extrap)
    _itp_search(itp)                — default search policy (default: itp.search_policy)

# Subtypes
- `LinearInterpolant{Tg, Tv}`: Piecewise linear interpolation
- `ConstantInterpolant{Tg, Tv}`: Piecewise constant (step) interpolation
- `QuadraticInterpolant{Tg, Tv}`: C1 piecewise quadratic spline
- `CubicInterpolant{Tg, Tv}`: C2 cubic spline
- `AbstractHermiteInterpolant1D{Tg, Tv}`: Cubic Hermite family (see subtypes below)
"""
abstract type AbstractInterpolant1D{Tg <: AbstractFloat, Tv} <: AbstractInterpolant{Tg, Tv} end

"""
    AbstractHermiteInterpolant1D{Tg, Tv} <: AbstractInterpolant1D{Tg, Tv}

Abstract supertype for 1D cubic Hermite family interpolants.

All subtypes store `(x, y, dy, spacing, extrap, search_policy)` and share
the same evaluation kernel (`_cubic_hermite_eval_at_point`) and integration
kernel (`_hermite_integral_kernel_1d`). This enables a single dispatch point
for `_itp_eval_scalar`, `_itp_vector_loop!`, and `integrate`.

# Subtypes
- `CubicHermiteInterpolant1D`: User-supplied slopes
- `PchipInterpolant1D`: Fritsch-Carlson monotone-preserving slopes
- `CardinalInterpolant1D`: Central finite difference (+ tension parameter)
- `AkimaInterpolant1D`: Weighted-average outlier-robust slopes
"""
abstract type AbstractHermiteInterpolant1D{Tg <: AbstractFloat, Tv} <: AbstractInterpolant1D{Tg, Tv} end

"""
    AbstractSeriesInterpolant{Tg<:AbstractFloat, Tv}

Abstract supertype for multi-series interpolant objects.
Series interpolants handle multiple y-series sharing the same x-grid.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (unconstrained)

# Subtypes
- `LinearSeriesInterpolant{Tg, Tv}`: Multiple linear interpolants sharing x-grid
- `ConstantSeriesInterpolant{Tg, Tv}`: Multiple constant interpolants sharing x-grid
- `QuadraticSeriesInterpolant{Tg, Tv}`: Multiple quadratic interpolants sharing x-grid
- `CubicSeriesInterpolant{Tg, Tv}`: Multiple cubic splines sharing x-grid

# Key Features
- Anchor optimization: compute interval once, evaluate all series
- Matrix storage: unified storage for optimal SIMD vectorization
- Zero-allocation batch evaluation with pre-built anchors

# Usage
```julia
x = collect(range(0.0, 1.0, 101))
y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-x)

sitp = cubic_interp(x, [y1, y2, y3])  # Creates CubicSeriesInterpolant

vals = sitp(0.5)            # Returns Vector of 3 values
sitp(output, 0.5)           # In-place evaluation
```

# Note
This is a pure type hierarchy - no methods are defined on `AbstractSeriesInterpolant` itself.
All functionality is implemented in concrete subtypes.
"""
abstract type AbstractSeriesInterpolant{Tg <: AbstractFloat, Tv} <: AbstractInterpolant{Tg, Tv} end

"""
    AbstractInterpolantND{Tg<:AbstractFloat, Tv, N}

Abstract supertype for N-dimensional interpolant objects.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions

# API Differences from 1D Interpolants
- **Evaluation**: `itp(x::NTuple{N})` or `itp(x::AbstractVector)` instead of `itp(x::Real)`
- **Derivatives**: Use `deriv` keyword (e.g., `itp(x; deriv=(1,0))`) or `deriv_view(itp, (1,0))`
- **Vector Calculus**: Supports `gradient`, `hessian`, `laplacian`

# Subtypes
- `CubicInterpolantND{Tg, Tv, N}`: N-dimensional cubic Hermite interpolation

# Example
```julia
x, y = range(0, 1, 50), range(0, 1, 50)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
itp = cubic_interp((x, y), data)  # Returns CubicInterpolantND{..., 2}

itp((0.5, 0.5))                    # Evaluate
itp((0.5, 0.5); deriv=(1, 0))      # ∂f/∂x
gradient(itp, (0.5, 0.5))          # (∂f/∂x, ∂f/∂y)
```
"""
abstract type AbstractInterpolantND{Tg <: AbstractFloat, Tv, N} <: AbstractInterpolant{Tg, Tv} end

Base.size(itp::AbstractInterpolantND) = map(length, itp.grids)

"""
    AbstractAdjoint{Tg<:AbstractFloat}

Abstract supertype for adjoint (transpose) operators of interpolation.

These operators compute `f̄ = Wᵀȳ` where `W` is the forward interpolation weight matrix.
They are **query-baked, data-free**: constructed from grid + query points, then applied
to any value vector `ȳ` regardless of its element type.

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)

Note: Only `Tg` is needed (no `Tv`) because adjoint operators are value-type independent.
The same operator works for Float64, ComplexF64, or any custom value type.

# 1D Protocol (adjoint_protocol.jl)
Subtypes automatically inherit 6 callable overloads (Vector/Real/Tuple × alloc/in-place),
`Base.size`, `Base.Matrix`, and exclusive periodic in-place handling by implementing:

    _n_queries(adj)::Int                    — number of baked query points (required)
    _adjoint_output_length(adj)::Int        — user-facing output length (required)
    _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)  — core scatter/solve (required)
    _adjoint_internal_length(adj)::Int      — alloc size (default: output length)
    _adjoint_1d_has_exclusive_periodic(adj)::Bool  — (default: false)
    _adjoint_1d_finalize(f_bar, adj)        — fold+truncate (default: identity)

# Subtypes
- [`AbstractAdjoint1D`](@ref): 1D adjoint operators
- [`AbstractAdjointND`](@ref): N-dimensional adjoint operators
"""
abstract type AbstractAdjoint{Tg <: AbstractFloat} end

"""
    AbstractAdjoint1D{Tg<:AbstractFloat} <: AbstractAdjoint{Tg}

Abstract supertype for 1-dimensional adjoint operators.

Subtypes automatically inherit 6 callable overloads (Vector/Real/Tuple × alloc/in-place),
`Base.size`, `Base.Matrix`, and exclusive periodic in-place handling from
`adjoint_protocol.jl` by implementing the required interface:

    _n_queries(adj)::Int                    — number of baked query points (required)
    _adjoint_output_length(adj)::Int        — user-facing output length (required)
    _adjoint_1d_apply!(f_bar, adj, y_bar, deriv)  — core scatter/solve (required)
    _adjoint_internal_length(adj)::Int      — alloc size (default: output length)
    _adjoint_1d_has_exclusive_periodic(adj)::Bool  — (default: false)
    _adjoint_1d_finalize(f_bar, adj)        — fold+truncate (default: identity)

# Subtypes
- `LinearAdjoint{Tg, EP}`: Adjoint of linear interpolation (1D, pure scatter)
- `CubicAdjoint{Tg, C, BC}`: Adjoint of cubic spline interpolation (1D)
"""
abstract type AbstractAdjoint1D{Tg <: AbstractFloat} <: AbstractAdjoint{Tg} end

"""
    AbstractAdjointND{Tg<:AbstractFloat, N} <: AbstractAdjoint{Tg}

Abstract supertype for N-dimensional adjoint operators.

Subtypes automatically inherit shared callable dispatch from `nd_adjoint_protocol.jl`
(allocating, in-place, scalar, tuple) by implementing the required interface:

    _n_queries(adj)::Int                          — number of baked query points
    _grid_size(adj)::NTuple{N,Int}                — internal grid size
    _adjoint_bcs(adj)                             — boundary conditions tuple
    _adjoint_nd_apply!(f_bar, adj, y_bar, ops)    — scatter y_bar into f_bar (accumulate)

# Subtypes
- `CubicAdjointND{Tg, N, ...}`: Adjoint of cubic spline interpolation (ND)
- `LinearAdjointND{Tg, N, ...}`: Adjoint of linear interpolation (ND)
"""
abstract type AbstractAdjointND{Tg <: AbstractFloat, N} <: AbstractAdjoint{Tg} end

# ========================================
# Type Helper Functions
# ========================================

"""
    Base.eltype(::AbstractInterpolant{Tg, Tv}) -> Type{Tv}

Element type of an interpolant: the value type `Tv` produced by evaluation.
"""
@inline Base.eltype(::AbstractInterpolant{Tg, Tv}) where {Tg, Tv} = Tv

"""
    Base.eltype(::AbstractAdjoint{Tg}) -> Type{Tg}

Weight/grid scalar type of an adjoint operator.
This is the coordinate type used internally for weights, not necessarily the
result type of applying the adjoint (which may promote with the sensitivity
vector's element type, e.g., Complex sensitivities produce Complex output).
"""
@inline Base.eltype(::AbstractAdjoint{Tg}) where {Tg} = Tg

"""
    grid_type(::AbstractInterpolant{Tg, Tv}) -> Type{Tg}

Get the grid/coordinate type of an interpolant.
"""
@inline grid_type(::AbstractInterpolant{Tg, Tv}) where {Tg, Tv} = Tg

"""
    grid_type(::AbstractAdjoint{Tg}) -> Type{Tg}

Get the grid/coordinate type of an adjoint operator.
"""
@inline grid_type(::AbstractAdjoint{Tg}) where {Tg} = Tg

"""
    value_type(::AbstractInterpolant{Tg, Tv}) -> Type{Tv}

Get the value type of an interpolant.
"""
@inline value_type(::AbstractInterpolant{Tg, Tv}) where {Tg, Tv} = Tv

"""
    eval_type(::AbstractInterpolant{Tg, Tv}, ::Type{Tq}) -> Type

Compute the output type when evaluating an interpolant with query type `Tq`.
This is `promote_type(Tv, Tq)`, accounting for value type
and query type (standard float or ForwardDiff.Dual).

# Examples
```julia
itp = linear_interp([0.0, 1.0], [1.0, 2.0])
eval_type(itp, Float64)  # Float64

itp_c = linear_interp([0.0, 1.0], [1.0+0im, 2.0+0im])
eval_type(itp_c, Float64)  # ComplexF64
```
"""
@inline eval_type(::AbstractInterpolant{Tg, Tv}, ::Type{Tq}) where {Tg, Tv, Tq} = promote_type(Tv, Tq)
