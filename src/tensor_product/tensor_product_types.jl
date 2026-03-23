# ========================================
# Tensor Product ND Interpolant Type
# ========================================
# N-dimensional interpolant with per-axis method specification.
# Supports two strategies via the D type parameter:
# - OnTheFly: D = Array{Tv, N} — stores raw data, builds 1D interps per query
# - PreCompute: D = NodalDerivativesND{Tv, N, N+1} — precomputed partials, O(1) eval
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (AbstractFloat)
# - Tv: Value type (unconstrained)
# - N:  Number of dimensions

"""
    TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D} <: AbstractInterpolantND{Tg, Tv, N}

N-dimensional interpolant with per-axis method specification.

Supports heterogeneous methods across dimensions (e.g., cubic on axis 1,
linear on axis 2) with two evaluation strategies:

- **`OnTheFly()`**: Sequential 1D interpolation per query (O(n) per query, zero build cost)
- **`PreCompute()`**: Precomputed partial derivatives with local kernel eval (O(1) per query)

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `G`: Tuple type for grids
- `S`: Tuple type for spacings
- `M`: Tuple type for per-axis methods (heterogeneous)
- `E`: Tuple type for extrapolation modes
- `P`: Tuple type for search policies
- `D`: Data storage type — `Array{Tv,N}` (OnTheFly) or `NodalDerivativesND` (PreCompute)

# Example
```julia
x, y = range(0, 1, 50), range(0, 1, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# On-the-fly (default)
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()))

# Precomputed (fast eval)
itp = interp((x, y), data; method=(CubicInterp(), LinearInterp()), coeffs=PreCompute())
itp((0.5, 0.3))
```
"""
struct TensorProductInterpolantND{
        Tg <: AbstractFloat,
        Tv,
        N,
        G <: Tuple{Vararg{AbstractVector, N}},
        S <: Tuple{Vararg{AbstractGridSpacing, N}},
        M <: Tuple{Vararg{AbstractInterpMethod, N}},
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
        D,
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    spacings::S
    data::D
    methods::M
    extraps::E
    searches::P
end

"""
    HeteroPartials{Tv, N, NP1}

Compact storage for heterogeneous partial derivatives.

Stores `prod(sizes)` partials per grid point using mixed-radix indexing,
where `sizes[d] = 2` for derivative axes (Cubic/Quadratic), `1` for others.
Saves memory vs `NodalDerivativesND` (which always stores `2^N`) when some
axes don't require derivatives.

Layout: `partials[p, i₁, i₂, ..., iₙ]` where `p ∈ 1:prod(sizes)`.
"""
struct HeteroPartials{Tv, N, NP1}
    partials::Array{Tv, NP1}
end
