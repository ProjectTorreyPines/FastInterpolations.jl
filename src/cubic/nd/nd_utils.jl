# ========================================
# ND Shared Utilities
# ========================================
#
# Common utility functions for N-dimensional interpolation.
# These are shared across all ND algorithms and dimensions.
#
# _resolve_* pattern: Convert flexible user input to canonical N-tuple form.
# - Single value → broadcast to all N axes
# - NTuple{N} → passthrough (with optional validation)
# - Wrong-sized tuple → ArgumentError

"""
    _resolve_extrap_nd(extrap, Val(N)) -> NTuple{N, Symbol}

Resolve extrapolation mode to canonical N-tuple.
- Single `Symbol` → broadcast to all N axes (validated)
- `NTuple{N, Symbol}` → validate each and passthrough
"""
@inline function _resolve_extrap_nd(extrap::Symbol, ::Val{N}) where {N}
    _validate_extrap(extrap)
    return ntuple(_ -> extrap, Val(N))
end

@inline function _resolve_extrap_nd(extrap::NTuple{N,Symbol}, ::Val{N}) where {N}
    @inbounds for i in 1:N
        _validate_extrap(extrap[i])
    end
    return extrap
end

@inline function _resolve_extrap_nd(extrap::Tuple{Vararg{Symbol}}, ::Val{N}) where {N}
    throw(ArgumentError("extrap tuple must have $N elements to match grid dimensions, got $(length(extrap))"))
end

"""
    _resolve_search_nd(search, Val(N)) -> NTuple{N, AbstractSearchPolicy}

Resolve search policy input to canonical N-tuple.
- Single `AbstractSearchPolicy` → broadcast to all N axes
- `NTuple{N, AbstractSearchPolicy}` → passthrough
"""
@inline _resolve_search_nd(s::AbstractSearchPolicy, ::Val{N}) where {N} = ntuple(_ -> s, Val(N))

@inline _resolve_search_nd(s::NTuple{N,AbstractSearchPolicy}, ::Val{N}) where {N} = s

@inline function _resolve_search_nd(s::Tuple{Vararg{AbstractSearchPolicy}}, ::Val{N}) where {N}
    throw(ArgumentError("search tuple must have $N elements to match grid dimensions, got $(length(s))"))
end

"""
    _resolve_bcs_nd(bc, Val(N)) -> NTuple{N, AbstractBC}

Resolve boundary condition input to canonical N-tuple.
- Single `AbstractBC` → broadcast to all N axes
- `NTuple{N, AbstractBC}` → passthrough
"""
@inline _resolve_bcs_nd(bc::AbstractBC, ::Val{N}) where {N} = ntuple(_ -> bc, Val(N))

@inline _resolve_bcs_nd(bc::NTuple{N,AbstractBC}, ::Val{N}) where {N} = bc

@inline function _resolve_bcs_nd(bc::Tuple{Vararg{AbstractBC}}, ::Val{N}) where {N}
    throw(ArgumentError("bc tuple must have $N elements to match grid dimensions, got $(length(bc))"))
end

# ========================================
# Derivative Order → EvalOp Conversion
# ========================================

"""
    _int_to_evalop(d::Int) -> AbstractEvalOp

Convert derivative order `d ∈ {0,1,2,3}` to evaluation operation singleton.
"""
@inline function _int_to_evalop(d::Int)
    if d == 0
        EvalValue()
    elseif d == 1
        EvalDeriv1()
    elseif d == 2
        EvalDeriv2()
    elseif d == 3
        EvalDeriv3()
    else
        throw(ArgumentError("deriv must be 0, 1, 2, or 3; got $d"))
    end
end

"""
    _resolve_deriv_nd(deriv, Val(N)) -> NTuple{N, AbstractEvalOp}

Resolve derivative specification to canonical N-tuple of evaluation operations.
- Single `Int` → broadcast to all N axes, convert to EvalOp
- `NTuple{N, Int}` → convert each to EvalOp
"""
@inline _resolve_deriv_nd(d::Int, ::Val{N}) where {N} = ntuple(_ -> _int_to_evalop(d), Val(N))

@inline _resolve_deriv_nd(d::NTuple{N,Int}, ::Val{N}) where {N} = ntuple(i -> @inbounds(_int_to_evalop(d[i])), Val(N))

@inline function _resolve_deriv_nd(d::Tuple{Vararg{Int}}, ::Val{N}) where {N}
    throw(ArgumentError("deriv tuple must have $N elements to match grid dimensions, got $(length(d))"))
end

# ========================================
# PolyFit BC Helpers
# ========================================

"""
    _get_polyfit_bc(bc::AbstractBC, deg::Int) -> PolyFit{D}

Get a PolyFit BC for use in derivative estimation.
If bc is already a PolyFit, returns it. Otherwise constructs PolyFit{deg}.
"""
_get_polyfit_bc(bc::PolyFit{D}, ::Int) where {D} = bc
_get_polyfit_bc(bc::BCPair{L,R}, deg::Int) where {L<:PolyFit,R} = bc.left
_get_polyfit_bc(bc::BCPair{L,R}, deg::Int) where {L,R<:PolyFit} = bc.right
_get_polyfit_bc(::AbstractBC, deg::Int) = _make_polyfit(Val(deg))

# Construct PolyFit at runtime from degree (common cases are type-stable)
_make_polyfit(::Val{1}) = PolyFit{1}()
_make_polyfit(::Val{2}) = PolyFit{2}()
_make_polyfit(::Val{3}) = PolyFit{3}()
_make_polyfit(::Val{4}) = PolyFit{4}()
_make_polyfit(::Val{5}) = PolyFit{5}()
@generated function _make_polyfit(::Val{D}) where {D}
    :(PolyFit{$D}())
end

# ========================================
# Grid Validation Helpers
# ========================================

"""
    _validate_nd_grids(grids::NTuple{N}, data::AbstractArray{<:Any,N})

Validate that grid lengths match data dimensions.
"""
function _validate_nd_grids(grids::NTuple{N,AbstractVector}, data::AbstractArray{<:Any,N}) where {N}
    data_size = size(data)
    for (i, g) in enumerate(grids)
        ng = length(g)
        nd = data_size[i]
        if ng != nd
            throw(DimensionMismatch(
                "Grid $i has $ng points but data dimension $i has size $nd"
            ))
        end
        if ng < 2
            throw(ArgumentError("Grid $i must have at least 2 points, got $ng"))
        end
    end
    return nothing
end

"""
    _validate_2d_grids(x, y, data)

Validate 2D grid and data dimensions.
"""
function _validate_2d_grids(
    x::AbstractVector{Tg},
    y::AbstractVector{Tg},
    data::AbstractMatrix{Tv}
) where {Tg, Tv}
    nx, ny = length(x), length(y)
    dnx, dny = size(data)

    dnx == nx || throw(DimensionMismatch(
        "data rows ($dnx) must match length(x) ($nx)"
    ))
    dny == ny || throw(DimensionMismatch(
        "data columns ($dny) must match length(y) ($ny)"
    ))
    nx >= 2 || throw(ArgumentError("x must have at least 2 points, got $nx"))
    ny >= 2 || throw(ArgumentError("y must have at least 2 points, got $ny"))

    return nothing
end
