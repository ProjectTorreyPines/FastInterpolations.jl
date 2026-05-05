# ========================================
# ConstantInterpolantND Type Definition
# ========================================
#
# N-dimensional constant (step) interpolation.
# Each axis independently selects left or right neighbor based on side mode.

"""
    ConstantInterpolantND{Tg,Tv,N,G,E,SD,P}

N-dimensional constant (step) interpolation with per-axis configuration.

# Type Parameters
- `Tg`: Grid coordinate type (unconstrained)
- `Tv`: Value type (unconstrained)
- `N`: Number of dimensions
- `G<:NTuple{N, AbstractVector{Tg}}`: Grid tuple type. Wrapped grids
  (`_CachedRange`/`_CachedVector`/`_ExclusivePeriodicAxis`) carry cached
  `h`/`inv_h` directly — no separate spacings field needed.
- `E<:Tuple{Vararg{AbstractExtrap, N}}`: Extrapolation mode tuple type
- `SD<:Tuple{Vararg{AbstractSide, N}}`: Side selection tuple type
- `P<:NTuple{N, AbstractSearchPolicy}`: Search policy tuple type

# Fields
- `grids`: Tuple of (wrapped) grid vectors, one per dimension
- `data`: N-dimensional data array
- `extraps`: Per-axis extrapolation modes
- `sides`: Per-axis side selection (NearestSide(), LeftSide(), RightSide())
- `searches`: Per-axis search policies

# Usage
```julia
# 2D constant interpolation
x = [0.0, 1.0, 2.0]
y = [0.0, 1.0, 2.0, 3.0]
data = rand(3, 4)

itp = constant_interp((x, y), data)  # Default: side=NearestSide(), extrap=NoExtrap()
val = itp((0.5, 1.5))                # Single query

# With configuration
itp = constant_interp((x, y), data; side=LeftSide(), extrap=ClampExtrap())

# Per-axis configuration
itp = constant_interp((x, y), data; side=(LeftSide(), RightSide()), extrap=(NoExtrap(), WrapExtrap()))
```
"""
struct ConstantInterpolantND{
        Tg,
        Tv,
        N,
        G <: NTuple{N, AbstractVector{Tg}},
        E <: Tuple{Vararg{AbstractExtrap, N}},
        SD <: Tuple{Vararg{AbstractSide, N}},
        P <: NTuple{N, AbstractSearchPolicy},
    } <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    data::Array{Tv, N}
    extraps::E
    sides::SD
    searches::P

    function ConstantInterpolantND{Tg, Tv, N, G, E, SD, P}(
            grids::Tuple{Vararg{AbstractVector, N}}, data::AbstractArray{Tv, N}, extraps::E, sides::SD, searches::P;
            bcs::NTuple{N, AbstractBC} = ntuple(_ -> NoBC(), Val(N))
        ) where {
            Tg, Tv, N, G <: NTuple{N, AbstractVector{Tg}},
            E, SD <: Tuple{Vararg{AbstractSide, N}}, P <: NTuple{N, AbstractSearchPolicy},
        }
        # Per-axis `_cache_axis` (insurance) + `_convert_copy` (ownership).
        # Array() converts AbstractArray→Array AND copies data in one step.
        grids_c = map((g, bc) -> _convert_copy(_cache_axis(g, bc, Tg), Tg), grids, bcs)
        return new{Tg, Tv, N, typeof(grids_c), E, SD, P}(grids_c, Array(data), extraps, sides, searches)
    end
end

# Type introspection
@inline grid_type(::ConstantInterpolantND{Tg}) where {Tg} = Tg
@inline value_type(::ConstantInterpolantND{Tg, Tv}) where {Tg, Tv} = Tv
