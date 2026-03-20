# ========================================
# Tensor Product ND Interpolant Type
# ========================================
# N-dimensional interpolant with per-axis method specification.
# Evaluates via sequential 1D interpolation (on-the-fly tensor product).
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (AbstractFloat)
# - Tv: Value type (unconstrained)
# - N:  Number of dimensions

"""
    TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P} <: AbstractInterpolantND{Tg, Tv, N}

N-dimensional interpolant with per-axis method specification.

Evaluates via sequential 1D interpolation (on-the-fly tensor product),
supporting heterogeneous methods across dimensions (e.g., cubic on axis 1,
linear on axis 2).

# Type Parameters
- `Tg`: Grid/coordinate type (Float32 or Float64)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `G`: Tuple type for grids
- `S`: Tuple type for spacings
- `M`: Tuple type for per-axis methods (heterogeneous)
- `E`: Tuple type for extrapolation modes
- `P`: Tuple type for search policies

# Fields
- `grids`: Per-axis grid vectors
- `spacings`: Per-axis spacing objects
- `data`: N-dimensional data array
- `methods`: Per-axis interpolation method (`CubicInterp`, `LinearMethod`, etc.)
- `extraps`: Per-axis extrapolation modes
- `searches`: Per-axis search policies

# Example
```julia
x, y = range(0, 1, 50), range(0, 1, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]
itp = interp_nd((x, y), data; methods=(CubicInterp(), LinearMethod()))
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
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    spacings::S
    data::Array{Tv, N}
    methods::M
    extraps::E
    searches::P
end
