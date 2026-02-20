# ========================================
# ConstantInterpolantND Type Definition
# ========================================
#
# N-dimensional constant (step) interpolation.
# Each axis independently selects left or right neighbor based on side mode.

"""
    ConstantInterpolantND{Tg,Tv,N,G,S,E,SD,P}

N-dimensional constant (step) interpolation with per-axis configuration.

# Type Parameters
- `Tg<:AbstractFloat`: Grid coordinate type
- `Tv`: Value type (can be Real or Complex)
- `N`: Number of dimensions
- `G<:NTuple{N, AbstractVector{Tg}}`: Grid tuple type
- `S<:NTuple{N, AbstractGridSpacing{Tg}}`: Spacing tuple type
- `E<:Tuple{Vararg{AbstractExtrap, N}}`: Extrapolation mode tuple type
- `SD<:Tuple{Vararg{AbstractSide, N}}`: Side selection tuple type
- `P<:NTuple{N, AbstractSearchPolicy}`: Search policy tuple type

# Fields
- `grids`: Tuple of grid vectors, one per dimension
- `spacings`: Tuple of spacing objects for efficient interval lookup
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
itp = constant_interp((x, y), data; side=LeftSide(), extrap=ConstExtrap())

# Per-axis configuration
itp = constant_interp((x, y), data; side=(LeftSide(), RightSide()), extrap=(NoExtrap(), WrapExtrap()))
```
"""
struct ConstantInterpolantND{
    Tg<:AbstractFloat,
    Tv,
    N,
    G<:NTuple{N, AbstractVector{Tg}},
    S<:NTuple{N, AbstractGridSpacing{Tg}},
    E<:Tuple{Vararg{AbstractExtrap, N}},
    SD<:Tuple{Vararg{AbstractSide, N}},
    P<:NTuple{N, AbstractSearchPolicy},
} <: AbstractInterpolantND{Tg, Tv, N}
    grids::G
    spacings::S
    data::Array{Tv, N}
    extraps::E
    sides::SD
    searches::P

    function ConstantInterpolantND{Tg,Tv,N,G,S,E,SD,P}(
        grids::G, spacings::S, data::Array{Tv,N}, extraps::E, sides::SD, searches::P
    ) where {Tg<:AbstractFloat, Tv, N, G<:NTuple{N,AbstractVector{Tg}},
             S<:NTuple{N,AbstractGridSpacing{Tg}}, E,
             SD<:Tuple{Vararg{AbstractSide, N}}, P<:NTuple{N,AbstractSearchPolicy}}
        new{Tg,Tv,N,G,S,E,SD,P}(grids, spacings, data, extraps, sides, searches)
    end
end

# Type introspection
@inline grid_type(::ConstantInterpolantND{Tg}) where {Tg} = Tg
@inline value_type(::ConstantInterpolantND{Tg,Tv}) where {Tg,Tv} = Tv
