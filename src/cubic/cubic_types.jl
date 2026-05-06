# ========================================
# Cubic Spline Type Definitions
# ========================================
# Structs for cubic spline interpolation.
# Separated from cubic_interp.jl for clarity.
# Include order: utils.jl → bc_types.jl → cubic_types.jl → cubic_solver.jl → cubic_interp.jl

# Boundary condition types (AbstractBC, PointBC, Deriv1, Deriv2, BCPair, PeriodicBC) are defined in bc_types.jl

"""
    CubicSplineCache{Tg, X, F, BC}

Cache structure for cubic spline interpolation with reusable Thomas factorization.

`cache.x` is the **single source of truth** for grid geometry — the wrapped
axis (`_CachedRange`, `_CachedVector`, or `_ExclusivePeriodicAxis`) returned by
`_resolve_axis_copied(x, bc, T)`. Cell widths read via `_get_h(cache.x, idx)`
(cached lookup); wrap-domain bounds via `(first(cache.x), last(cache.x))`.
Periodic period / seam-cell info travels with the wrapper, so the cache needs
no separate `period` field.

# Type Parameters
- `Tg`: Grid element type (Float32, Float64, or duck e.g. `ForwardDiff.Dual`).
- `X`: Wrapped axis type (`_CachedRange`/`_CachedVector`/`_ExclusivePeriodicAxis`).
- `F`: Thomas factorization type (`ThomasFactorization{Tg, Vector{Tg}}`).
- `BC`: User's boundary condition (`BCPair{L,R}`, `PeriodicBC{E,P,C}`, etc.).
  Carries the resolved period for `:exclusive` periodic so display / cache pool
  comparison works without a separate field.

# Fields
- `x::X`: Wrapped axis. `length(x) - 1 == n_cells` uniformly across all forms.
- `bc::BC`: Boundary condition (drives solver dispatch).
- `thomas::F`: Thomas factorization of the (modified) tridiagonal A.
- `q::Vector{Tg}`: Sherman-Morrison `A'⁻¹ u` vector for periodic; empty
  (length 0) for non-periodic caches.

# Notes
- `n_cells = length(cache.x) - 1` works uniformly: inclusive periodic / non-
  periodic has user-supplied n+1 grid → n cells; exclusive periodic wrapper
  exposes virtual length n+1 → same n cells.
- Workspaces (d, z) are allocated from task-local pools via `@with_pool`; this
  cache holds no per-call mutable state and is thread-safe by design.

# Boundary Conditions
- `bc=CubicFit()` (default): 4-point polynomial fit at endpoints
- `bc=ZeroCurvBC()`: Zero-curvature spline with z[1] = z[n+1] = 0
- `bc=PeriodicBC()`: Periodic spline with C2 continuity at boundaries
"""
struct CubicSplineCache{Tg, X <: AbstractVector{Tg}, F, BC <: AbstractBC}
    x::X
    bc::BC
    thomas::F
    q::Vector{Tg}   # Sherman-Morrison q (length n_cells for periodic; empty otherwise)
end

# AbstractExtrap types are defined in eval_ops.jl (shared across all interpolants)

"""
    CubicInterpolant{Tg, Tv, C, E, P, BC, Tz}

Lightweight callable interpolant for broadcast fusion optimization.
Returned by `cubic_interp(x, y)` (2-argument form).

# Type Parameters
- `Tg`: Grid element type (Float32, Float64, or duck-typed e.g. ForwardDiff.Dual)
- `Tv`: Value type (unconstrained)
- `C`: CubicSplineCache type (preserves grid type info for O(1) vs O(log n) lookup)
- `E`: Extrapolation mode type (compile-time specialized)
- `P`: Search policy type (AutoSearch, BinarySearch, LinearBinarySearch, etc.)
- `BC`: Boundary condition type (BCPair or PeriodicBC)
- `Tz`: Element type of z coefficients (`= _output_eltype(Tv, Tg)` — Dual when grid is Dual)

# Fields
- `cache::C`: Pre-computed CubicSplineCache (LU factorization)
- `y::Vector{Tv}`: y-values (function values at grid points)
- `z::Vector{Tz}`: Pre-computed second derivative coefficients (solves system once!)
- `bc::BC`: Boundary condition used for this interpolant
- `extrap::E`: Extrapolation mode (compile-time specialized via type parameter)
- `search_policy::P`: Default search policy for interval lookup

# Usage
```julia
itp = cubic_interp(x, y)
result = @. coef * itp(rho) * other_terms  # fused, zero-allocation per call
val = itp(0.5)                              # scalar (zero-allocation)

# Search policy: AutoSearch adapts to query type (scalar→BinarySearch, vector→LinearBinarySearch)
itp = cubic_interp(x, y)
val = itp(0.5)                              # AutoSearch resolves to BinarySearch() for scalar
itp = cubic_interp(x, y; search=LinearBinarySearch())  # explicit override
val = itp(0.5; search=BinarySearch())             # per-call override

# Complex values
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im, 9.0+10.0im]
itp = cubic_interp(x, y)
val = itp(0.5)  # returns ComplexF64
```

# Performance Notes
- System solved ONCE at construction -> z coefficients pre-computed
- Each scalar call just evaluates cubic polynomial (zero-allocation!)
- Broadcast operations are perfectly fused (no intermediate arrays)
- Extrapolation mode uses type-parametrized dispatch for zero overhead
"""
struct CubicInterpolant{Tg, Tv, C <: CubicSplineCache{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy, BC <: CubicBC, Tz} <: AbstractInterpolant1D{Tg, Tv}
    cache::C
    y::Vector{Tv}
    z::Vector{Tz}  # Second derivative coefficients: Tz = _output_eltype(Tv, Tg)
    bc::BC  # Boundary condition used for this interpolant
    extrap::E  # Extrapolation mode (compile-time specialized via type parameter)
    search_policy::P  # Default search policy (immutable, thread-safe)
    function CubicInterpolant(
            cache::C,
            y::AbstractVector,
            z::AbstractVector,
            bc::BC,
            extrap::E,
            search::P = AutoSearch()
        ) where {Tg, C <: CubicSplineCache{Tg}, E <: AbstractExtrap, P <: AbstractSearchPolicy, BC <: CubicBC}
        length(cache.x) == length(y) || _throw_length_mismatch(length(cache.x), length(y))
        length(cache.x) == length(z) || _throw_length_mismatch(length(cache.x), length(z), "grid", "z")
        Tv = _value_type(eltype(y), Tg)
        Tz = eltype(z)
        return new{Tg, Tv, C, E, P, BC, Tz}(cache, _convert_copy(y, Tv), Vector{Tz}(z), bc, extrap, search)
    end
end

# ========================================
# TransposeSnapshot Type (shared between multi-series interpolants)
# ========================================

"""
    TransposeSnapshot{Tv}

Immutable snapshot of point-contiguous (transposed) matrices.

Used for atomic swap in multi-series cubic interpolants to ensure thread-safe
lazy initialization of point-contiguous layout.

# Type Parameters
- `Tv`: Value type (unconstrained)

# Fields
- `y_point::Union{Nothing, Matrix{Tv}}`: Point-contiguous y values (n_series × n_points)
- `z_point::Union{Nothing, Matrix{Tv}}`: Point-contiguous z values (n_series × n_points)

# Thread Safety
Used with atomic operations for lock-free lazy initialization.
Multiple threads may compute the transpose simultaneously (benign duplication).
"""
struct TransposeSnapshot{Tv}
    y_point::Union{Nothing, Matrix{Tv}}
    z_point::Union{Nothing, Matrix{Tv}}
end

# Empty snapshot constructor
TransposeSnapshot{Tv}() where {Tv} = TransposeSnapshot{Tv}(nothing, nothing)
