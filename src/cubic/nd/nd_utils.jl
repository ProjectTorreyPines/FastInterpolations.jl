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
#
# Design: Strict API for Performance
# -----------------------------------
# Accepts only:
#   1. Int (0-3): Broadcast to all axes via @_dispatch_deriv
#   2. Val{D}: Compile-time spec → type-stable ntuple
#
# Raw Tuple rejected to prevent Union type performance traps.

"""
    _int_to_evalop(::Val{d}) -> AbstractEvalOp

Convert compile-time derivative order to evaluation operation singleton.
Val-based dispatch ensures type stability.
"""
@inline _int_to_evalop(::Val{0}) = EvalValue()
@inline _int_to_evalop(::Val{1}) = EvalDeriv1()
@inline _int_to_evalop(::Val{2}) = EvalDeriv2()
@inline _int_to_evalop(::Val{3}) = EvalDeriv3()

"""
    _resolve_deriv_nd(deriv, Val(N)) -> NTuple{N, AbstractEvalOp}

Resolve derivative specification to N-tuple of EvalOp singletons.

# Accepted
- `Int` (0-3): Broadcast to all axes
- `Val{Int}`: Compile-time broadcast (e.g., `Val(1)`)
- `Val{Tuple}`: Mixed partials, compile-time (e.g., `Val((1,0))` for ∂f/∂x)
- `NTuple{N,Int}`: Mixed partials, runtime (e.g., `(1,0)` for ∂f/∂x)

Note: `Val((1,0))` is slightly faster than `(1,0)` due to compile-time dispatch,
but the difference is negligible (~0.3 KiB extra allocation per batch call).
"""
# Int path: macro dispatch at call site ensures concrete type
@inline function _resolve_deriv_nd(d::Int, ::Val{N}) where {N}
    if d == 0
        return ntuple(_ -> EvalValue(), Val(N))
    elseif d == 1
        return ntuple(_ -> EvalDeriv1(), Val(N))
    elseif d == 2
        return ntuple(_ -> EvalDeriv2(), Val(N))
    elseif d == 3
        return ntuple(_ -> EvalDeriv3(), Val(N))
    else
        throw(ArgumentError(
            "Integer deriv must be 0, 1, 2, or 3. " *
            "For higher derivatives or mixed partials, use Val(d) or Val((d1,d2,...))."
        ))
    end
end

# Val{Int}: Compile-time broadcast
@inline function _resolve_deriv_nd(::Val{D}, ::Val{N}) where {D, N}
    if D isa Int
        # Val(1) → broadcast to all axes
        return ntuple(_ -> _int_to_evalop(Val(D)), Val(N))
    elseif D isa Tuple
        # Val((1,0)) → per-axis specification
        length(D) == N || throw(ArgumentError(
            "Val tuple must have $N elements, got $(length(D))"
        ))
        return ntuple(i -> _int_to_evalop(Val(D[i])), Val(N))
    else
        throw(ArgumentError("Val parameter must be Int or Tuple{Vararg{Int}}, got $(typeof(D))"))
    end
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

# ========================================
# Zero-Allocation Grid Type Helpers
# ========================================
#
# @generated functions to avoid closure boxing in tuple operations.
# These replace map/ntuple patterns that cause allocations due to
# capturing heterogeneous tuple variables.

"""
    _promote_grid_eltype(grids::NTuple{N, AbstractVector}) -> Type

Zero-allocation promoted element type extraction from grid tuple.
Generates unrolled `promote_type(eltype(grids[1]), eltype(grids[2]), ...)` at compile time.
"""
@generated function _promote_grid_eltype(grids::NTuple{N, AbstractVector}) where {N}
    types = [:(eltype(grids[$i])) for i in 1:N]
    :(promote_type($(types...)))
end

"""
    _convert_grids_typed(grids::NTuple{N, AbstractVector}, ::Type{Tg}) -> NTuple{N}

Zero-allocation grid conversion to target element type.
Generates unrolled `(_convert_grid(grids[1], Tg), _convert_grid(grids[2], Tg), ...)` at compile time.

Requires `_convert_grid(grid, Type)` to be defined.
"""
@generated function _convert_grids_typed(grids::NTuple{N, AbstractVector}, ::Type{Tg}) where {N, Tg}
    exprs = [:(FastInterpolations._convert_grid(grids[$i], Tg)) for i in 1:N]
    :(($(exprs...),))
end
