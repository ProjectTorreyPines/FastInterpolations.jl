# ========================================
# Tensor Product ND Interpolant Type
# ========================================
# N-dimensional interpolant with per-axis method specification.
# Supports two strategies via the D type parameter:
# - OnTheFly: D = Array{Tv, N} — stores raw data, builds 1D interps per query
# - PreCompute: D = _HeteroPartials{Tv, N, N+1} — precomputed partials, O(1) eval
#
# Type Parameters Convention:
# - Tg: Grid/coordinate type (unconstrained — supports duck types like ForwardDiff.Dual)
# - Tv: Value type (unconstrained)
# - N:  Number of dimensions

# Method-aware `_cache_axis` for HeteroND. `NoInterp` axes are length-1
# markers — bypass `_cache_axis` (which requires ≥ 2 points). 3-arg form
# is used by outer factories; 4-arg threads `Tg` for the inner ctor.
@inline _cache_axis_for_method(g, ::AbstractBC, ::NoInterp) = g
@inline _cache_axis_for_method(g, bc::AbstractBC, ::AbstractInterpMethod) = _cache_axis(g, bc)
@inline _cache_axis_for_method(g, ::AbstractBC, ::Type, ::NoInterp) = g
@inline _cache_axis_for_method(g, bc::AbstractBC, ::Type{Tg}, ::AbstractInterpMethod) where {Tg} =
    _cache_axis(g, bc, Tg)

"""
    HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D} <: AbstractInterpolantND{Tg, Tv, N}

N-dimensional interpolant with per-axis method specification.

Supports heterogeneous methods across dimensions (e.g., cubic on axis 1,
linear on axis 2) with two evaluation strategies:

- **`OnTheFly()`**: Sequential 1D interpolation per query (O(n) per query, zero build cost)
- **`PreCompute()`**: Precomputed partial derivatives with local kernel eval (O(1) per query)

# Type Parameters
- `Tg`: Grid/coordinate type (unconstrained — supports duck types like ForwardDiff.Dual)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `G`: Tuple type for grids (wrapped grids carry cached `h`/`inv_h`)
- `M`: Tuple type for per-axis methods (heterogeneous)
- `E`: Tuple type for extrapolation modes
- `P`: Tuple type for search policies
- `D`: Data storage type — `Array{Tv,N}` (OnTheFly) or `_HeteroPartials{Tv,N,N+1}` (PreCompute)

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
struct HeteroInterpolantND{
        Tg,
        Tv,
        N,
        G <: Tuple{Vararg{AbstractVector, N}},
        M <: Tuple{Vararg{AbstractInterpMethod, N}},
        E <: Tuple{Vararg{AbstractExtrap, N}},
        P <: Tuple{Vararg{AbstractSearchPolicy, N}},
        D,
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    data::D
    methods::M
    extraps::E
    searches::P

    # Inner ctor: type params inferred from arg signature. Tv extracted via
    # `eltype(data)` (works for both `Array{Tv,N}` and `_HeteroPartials{Tv,N,NP1}`).
    function HeteroInterpolantND(
            grids::Tuple{Vararg{AbstractVector{Tg}, N}},
            data,
            methods::Tuple{Vararg{AbstractInterpMethod, N}},
            extraps::Tuple{Vararg{AbstractExtrap, N}},
            searches::Tuple{Vararg{AbstractSearchPolicy, N}};
            bcs::NTuple{N, AbstractBC} = ntuple(_ -> NoBC(), Val(N))
        ) where {Tg, N}
        Tv = eltype(data)
        grids_c = map((g, bc, m) -> _convert_copy(_cache_axis_for_method(g, bc, Tg, m), Tg), grids, bcs, methods)
        return new{Tg, Tv, N, typeof(grids_c), typeof(methods), typeof(extraps), typeof(searches), typeof(data)}(
            grids_c, data, methods, extraps, searches
        )
    end
end

"""
    _HeteroPartials{Tv, N, NP1}

Compact storage for heterogeneous partial derivatives.

Stores `prod(sizes)` partials per grid point using mixed-radix indexing,
where `sizes[d] = 2` for derivative axes (Cubic/Quadratic), `1` for others.
Saves memory vs `_NodalDerivativesND` (which always stores `2^N`) when some
axes don't require derivatives.

Layout: `partials[p, i₁, i₂, ..., iₙ]` where `p ∈ 1:prod(sizes)`.
"""
struct _HeteroPartials{Tv, N, NP1}
    partials::Array{Tv, NP1}
end

@inline Base.eltype(::Type{<:_HeteroPartials{Tv}}) where {Tv} = Tv
@inline Base.eltype(p::_HeteroPartials) = eltype(typeof(p))
