# ========================================
# ND Shared Utilities
# ========================================
#
# Common utility functions for N-dimensional interpolation.
# These are shared across all ND algorithms (constant, linear, cubic)
# and dimensions.
#
# _resolve_* pattern: Convert flexible user input to canonical N-tuple form.
# - Single value → broadcast to all N axes
# - NTuple{N} → passthrough (with optional validation)
# - Wrong-sized tuple → ArgumentError

# ========================================
# Side Validation (for Constant ND)
# ========================================

"""
    _validate_side(side::Symbol) -> Nothing

Validate side selection symbol. Throws `ArgumentError` if invalid.

Valid options: `:nearest`, `:left`, `:right`
"""
@inline function _validate_side(side::Symbol)
    side in (:nearest, :left, :right) && return nothing
    throw(ArgumentError("`side` must be :nearest, :left, or :right, got :$side"))
end

# ========================================
# Extrapolation Resolution
# ========================================

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

# ========================================
# Search Policy Resolution
# ========================================

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

# ========================================
# Boundary Condition Resolution
# ========================================

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
# Side Resolution (for Constant ND)
# ========================================

"""
    _resolve_side_nd(side, Val(N)) -> NTuple{N, Symbol}

Resolve side selection to canonical N-tuple.
- Single `Symbol` → broadcast to all N axes (validated)
- `NTuple{N, Symbol}` → validate each and passthrough

Valid side options: `:nearest`, `:left`, `:right`
"""
@inline function _resolve_side_nd(side::Symbol, ::Val{N}) where {N}
    _validate_side(side)
    return ntuple(_ -> side, Val(N))
end

@inline function _resolve_side_nd(side::NTuple{N,Symbol}, ::Val{N}) where {N}
    @inbounds for i in 1:N
        _validate_side(side[i])
    end
    return side
end

@inline function _resolve_side_nd(side::Tuple{Vararg{Symbol}}, ::Val{N}) where {N}
    throw(ArgumentError("side tuple must have $N elements to match grid dimensions, got $(length(side))"))
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

# Tuple{Int,...} path: runtime per-axis specification (e.g., (1, 0) for ∂f/∂x)
# Wraps in Val and delegates to the compile-time path (consistent with cubic_nd_eval.jl)
@inline _resolve_deriv_nd(d::Tuple{Vararg{Int,N}}, ::Val{N}) where {N} = _resolve_deriv_nd(Val(d), Val(N))

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

Uses @generated to avoid closure boxing when iterating over heterogeneous grid tuples.
"""
@generated function _validate_nd_grids(grids::NTuple{N,AbstractVector}, data::AbstractArray{<:Any,N}) where {N}
    checks = [quote
        ng = length(grids[$i])
        nd = size(data, $i)
        if ng != nd
            throw(DimensionMismatch(
                "Grid $($i) has " * string(ng) * " points but data dimension $($i) has size " * string(nd)
            ))
        end
        if ng < 2
            throw(ArgumentError("Grid $($i) must have at least 2 points, got " * string(ng)))
        end
    end for i in 1:N]

    quote
        $(checks...)
        nothing
    end
end

# ========================================
# Per-Axis Extrapolation Handling
# ========================================
#
# Shared extrapolation logic for all ND interpolation methods.
# Handles domain boundary conditions before interval search.

"""
    _handle_all_extraps(queries, grids, extraps) -> NTuple{N}

Apply extrapolation handling to all query coordinates.
Returns tuple of processed query values ready for interpolation.

Accepts heterogeneous tuples (e.g., mixed grid types, per-axis extrap modes).
"""
@inline function _handle_all_extraps(
    queries::Tuple{Vararg{Real,N}}, grids::Tuple{Vararg{AbstractVector,N}}, extraps::Tuple{Vararg{Val,N}}
) where {N}
    ntuple(Val(N)) do d
        @inbounds _handle_axis_extrap(queries[d], grids[d], extraps[d])
    end
end

# Extrapolation handlers for each mode
# Note: Preserve query type (don't convert to Tg) for AD support (ForwardDiff.Dual)

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::Val{:none})
    @boundscheck _check_domain(axis, q, Val(:none))
    return q  # preserve original type for AD
end

@inline function _handle_axis_extrap(q, axis::AbstractVector{Tg}, ::Val{:constant}) where {Tg}
    # For AD: use primal for comparison, preserve type when in domain
    q_primal = _extract_primal(q)
    lo, hi = first(axis), last(axis)
    q_primal < lo && return oftype(q, lo)  # outside domain: return boundary
    q_primal > hi && return oftype(q, hi)
    return q  # in domain: preserve original type for AD
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::Val{:extension})
    return q  # Allow queries outside domain (for polynomial/constant extension)
end

@inline function _handle_axis_extrap(q, axis::AbstractVector, ::Val{:wrap})
    return _wrap_to_domain(q, first(axis), last(axis))  # already handles AD via _extract_primal
end

# ========================================
# Per-Axis Interval Search
# ========================================
#
# Shared interval search logic for all ND interpolation methods.
# Finds the cell containing each query coordinate.

"""
    _search_all_intervals(q_evals, grids, spacings, searches) -> (indices, Ls, Rs)

Perform interval search on all axes.
Returns tuples of: indices (cell index), Ls (left bounds), Rs (right bounds).

Accepts heterogeneous tuples (e.g., mixed grid types, spacing types, search policies).
"""
@inline function _search_all_intervals(
    q_evals::Tuple{Vararg{Real,N}}, grids::Tuple{Vararg{AbstractVector,N}},
    spacings::Tuple{Vararg{AbstractGridSpacing,N}}, searches::Tuple{Vararg{AbstractSearchPolicy,N}}
) where {N}
    results = ntuple(Val(N)) do d
        searcher = @inbounds _to_searcher(searches[d])
        @inbounds search_interval(searcher, grids[d], spacings[d], q_evals[d])
    end
    # Restructure (idx, L, R) tuples into separate tuples
    indices = ntuple(d -> @inbounds(results[d][1]), Val(N))
    Ls = ntuple(d -> @inbounds(results[d][2]), Val(N))
    Rs = ntuple(d -> @inbounds(results[d][3]), Val(N))
    return (indices, Ls, Rs)
end

# ----------------------------------------
# Hint-aware overloads for persistent search state
# ----------------------------------------

"""
    _get_axis_hint(hints, d) -> Nothing or Base.RefValue{Int}

Extract per-axis hint from a hint tuple or Nothing.
Used by N=2 specializations that destructure manually.
"""
@inline _get_axis_hint(::Nothing, d) = nothing
@inline _get_axis_hint(hints::Tuple, d) = @inbounds hints[d]

# Nothing hint → delegate to existing 4-arg (zero overhead)
@inline function _search_all_intervals(
    q_evals::Tuple{Vararg{Real,N}}, grids::Tuple{Vararg{AbstractVector,N}},
    spacings::Tuple{Vararg{AbstractGridSpacing,N}}, searches::Tuple{Vararg{AbstractSearchPolicy,N}},
    ::Nothing
) where {N}
    return _search_all_intervals(q_evals, grids, spacings, searches)
end

# Tuple hint → use 2-arg _to_searcher(policy, hint) per axis
@inline function _search_all_intervals(
    q_evals::Tuple{Vararg{Real,N}}, grids::Tuple{Vararg{AbstractVector,N}},
    spacings::Tuple{Vararg{AbstractGridSpacing,N}}, searches::Tuple{Vararg{AbstractSearchPolicy,N}},
    hints::Tuple{Vararg{Base.RefValue{Int},N}}
) where {N}
    results = ntuple(Val(N)) do d
        searcher = @inbounds _to_searcher(searches[d], hints[d])
        @inbounds search_interval(searcher, grids[d], spacings[d], q_evals[d])
    end
    indices = ntuple(d -> @inbounds(results[d][1]), Val(N))
    Ls = ntuple(d -> @inbounds(results[d][2]), Val(N))
    Rs = ntuple(d -> @inbounds(results[d][3]), Val(N))
    return (indices, Ls, Rs)
end

# ========================================
# N=2 Specialized Cell Location Preamble
# ========================================
#
# Shared 2D preamble for all N=2 _locate_cell specializations.
# Extracts query, handles extrapolation, and performs interval search.
# Returns raw (x_eval, y_eval, ix, iy, xL, yL) for type-specific post-processing.

"""
    _locate_cell_2d_preamble(query, grids, spacings, extraps, search, hints)

Shared preamble for all N=2 `_locate_cell` specializations.
Destructures 2D query, applies per-axis extrapolation, and performs interval search.

Returns `(x_eval, y_eval, ix, iy, xL, yL)` — the 6 raw values that each
interpolant type then post-processes into its kernel-specific cell tuple.
"""
@inline function _locate_cell_2d_preamble(
    query::Tuple{Vararg{Real, 2}},
    grids, spacings, extraps,
    search::Tuple{<:AbstractSearchPolicy, <:AbstractSearchPolicy},
    hints
)
    xq, yq = query
    grid_x, grid_y = grids
    spacing_x, spacing_y = spacings
    extrap_x, extrap_y = extraps
    search_x, search_y = search

    x_eval = _handle_axis_extrap(xq, grid_x, extrap_x)
    y_eval = _handle_axis_extrap(yq, grid_y, extrap_y)

    searcher_x = _to_searcher(search_x, _get_axis_hint(hints, 1))
    searcher_y = _to_searcher(search_y, _get_axis_hint(hints, 2))
    ix, xL, _ = search_interval(searcher_x, grid_x, spacing_x, x_eval)
    iy, yL, _ = search_interval(searcher_y, grid_y, spacing_y, y_eval)

    return (x_eval, y_eval, ix, iy, xL, yL)
end

# ========================================
# Zero-Allocation Grid Type Helpers
# ========================================
#
# @generated functions to avoid closure boxing in tuple operations.
# These replace map/ntuple patterns that cause allocations due to
# capturing heterogeneous tuple variables.

# ========================================
# Shared ND Coefficient Storage
# ========================================

"""
    NodalDerivativesND{Tv, N, NP1}

Storage for precomputed N-dimensional partial derivatives at grid nodes.

This structure stores the function value f and all mixed partial derivatives
(∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁∂x₂, etc.) at each grid node, enabling O(1) evaluation
via tensor-product interpolation (Hermite for cubic, quadratic kernel for quadratic).

# Type Parameters
- `Tv`: Value type (Float64, ComplexF64, etc.)
- `N`: Number of dimensions
- `NP1`: N + 1 (array dimensionality, Julia can't compute N+1 in type definition)

# Fields
- `partials::Array{Tv, NP1}`: Partial derivatives array of shape (2^N, n₁, n₂, ..., nₙ)

# Partials Indexing Convention (bit-encoding of derivatives)
The first index `p` encodes which partial derivative via binary representation:
- Bit d set → differentiate with respect to dimension d
- p=1 (binary 0...0): f (no derivatives)
- p=2 (binary 0...1): ∂f/∂x₁
- p=3 (binary 0..10): ∂f/∂x₂
- p=4 (binary 0..11): ∂²f/∂x₁∂x₂
- etc.

# Examples
For N=2 (4 partials per point):
- `partials[1, i, j]` = f(xᵢ, yⱼ)         (p=1, binary 00)
- `partials[2, i, j]` = ∂f/∂x             (p=2, binary 01)
- `partials[3, i, j]` = ∂f/∂y             (p=3, binary 10)
- `partials[4, i, j]` = ∂²f/∂x∂y          (p=4, binary 11)

For N=3 (8 partials per point):
- `partials[1, i, j, k]` = f              (p=1, binary 000)
- `partials[2, i, j, k]` = ∂f/∂x          (p=2, binary 001)
- `partials[3, i, j, k]` = ∂f/∂y          (p=3, binary 010)
- `partials[4, i, j, k]` = ∂²f/∂x∂y       (p=4, binary 011)
- `partials[5, i, j, k]` = ∂f/∂z          (p=5, binary 100)
- `partials[6, i, j, k]` = ∂²f/∂x∂z       (p=6, binary 101)
- `partials[7, i, j, k]` = ∂²f/∂y∂z       (p=7, binary 110)
- `partials[8, i, j, k]` = ∂³f/∂x∂y∂z     (p=8, binary 111)
"""
struct NodalDerivativesND{Tv, N, NP1}
    partials::Array{Tv, NP1}

    function NodalDerivativesND{Tv, N, NP1}(partials::Array{Tv, NP1}) where {Tv, N, NP1}
        NP1 == N + 1 || throw(ArgumentError("NP1 must equal N+1, got NP1=$NP1, N=$N"))
        size(partials, 1) == (1 << N) || throw(DimensionMismatch(
            "First dimension must be 2^N=$(1 << N), got $(size(partials, 1))"
        ))
        new{Tv, N, NP1}(partials)
    end
end

# Convenience constructor that computes NP1 automatically
function NodalDerivativesND{Tv, N}(partials::Array{Tv, NP1}) where {Tv, N, NP1}
    NodalDerivativesND{Tv, N, NP1}(partials)
end

# ========================================
# Shared ND Local Parameter Computation
# ========================================

"""
    _compute_all_local_params(q_evals, spacings, indices, Ls) -> (hs, inv_hs, dLs)

Compute local cell parameters for all axes.
Returns tuples of: hs (cell widths), inv_hs (reciprocals), dLs (left deltas).

Used by both CubicInterpolantND and QuadraticInterpolantND evaluation.
"""
@inline function _compute_all_local_params(
    q_evals::Tuple{Vararg{Real, N}},  # Allow heterogeneous/AD types (Dual)
    spacings::Tuple{Vararg{AbstractGridSpacing, N}},  # Allow heterogeneous spacing types (VectorSpacing, ScalarSpacing)
    indices::NTuple{N, Int},
    Ls::Tuple{Vararg{Real, N}}  # Grid boundary (allow heterogeneous Real types)
) where {N}
    hs = ntuple(Val(N)) do d
        @inbounds _get_h(spacings[d], indices[d])
    end
    inv_hs = ntuple(Val(N)) do d
        @inbounds _get_inv_h(spacings[d], indices[d])
    end
    dLs = ntuple(Val(N)) do d
        @inbounds q_evals[d] - Ls[d]
    end
    return (hs, inv_hs, dLs)
end

# ========================================
# @generated Tensor-Product Code Generation Helpers
# ========================================
#
# Used by @generated tensor-product kernels (cubic_nd_eval.jl, quadratic_nd_eval.jl)
# to build AST at compile time. These are NOT called at runtime.

"""
    _varname(stage, corner, deriv) -> Symbol

Generate a variable name for the tensor-product dimension-collapsing stages.
E.g., `_varname(2, 0, 1)` → `:g_2_0_1` (stage 2, corner 0, deriv 1).
"""
_varname(stage::Int, corner::Int, deriv::Int) = Symbol("g_$(stage)_$(corner)_$(deriv)")

"""
    _partial_index(deriv_bits) -> Int

Convert derivative bitmask to 1-based partials array index.
The bitmask encodes which dimensions are differentiated (bit d set → ∂/∂xd).
"""
_partial_index(deriv_bits::Int) = deriv_bits + 1

"""
    _corner_offset_expr(corner_bits, N) -> Vector{Int}

Convert corner bitmask to per-dimension 0/1 offsets.
Used to index into the 2^N corners of an N-dimensional cell.
E.g., for N=3, corner_bits=5 (binary 101) → [1, 0, 1].
"""
function _corner_offset_expr(corner_bits::Int, N::Int)
    [((corner_bits >> (d-1)) & 1) for d in 1:N]
end

# ========================================
# @generated Grid Type Promotion
# ========================================

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

"""
    _create_spacings_typed(grids::NTuple{N, AbstractVector}) -> NTuple{N}

Zero-allocation spacing creation from grid tuple.
Generates unrolled `(_create_spacing(grids[1]), _create_spacing(grids[2]), ...)` at compile time.

Avoids closure boxing that occurs with `ntuple(d -> _create_spacing(grids[d]), Val(N))`
when grids is a heterogeneous tuple (e.g., mix of Range and Vector).
"""
@generated function _create_spacings_typed(grids::NTuple{N, AbstractVector}) where {N}
    exprs = [:(FastInterpolations._create_spacing(grids[$i])) for i in 1:N]
    :(($(exprs...),))
end

# ========================================
# In-Place Batch Evaluation (Generic ND)
# ========================================
#
# Generic inner loops for in-place batch evaluation.
# Works for ALL AbstractInterpolantND subtypes via _locate_cell + _eval_at_cell dispatch.
# Callables in each type's eval file handle deriv dispatch and call these.

"""
    _batch_nd_soa!(output, itp, queries, ops, search, hints=nothing)

In-place SoA batch evaluation. Writes results into `output`.
"""
@inline function _batch_nd_soa!(
    output::AbstractVector,
    itp::AbstractInterpolantND{Tg, Tv, N},
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}},
    ops::NTuple{N, AbstractEvalOp},
    search::Tuple{Vararg{AbstractSearchPolicy, N}},
    hints=nothing
) where {Tg, Tv, N}
    @inbounds for k in 1:length(queries[1])
        query_k = ntuple(d -> queries[d][k], Val(N))
        cell = _locate_cell(itp, query_k, search, hints)
        output[k] = _eval_at_cell(itp, cell, ops)
    end
    return output
end

"""
    _batch_nd_aos!(output, itp, queries, ops, search, hints=nothing)

In-place AoS batch evaluation. Writes results into `output`.
"""
@inline function _batch_nd_aos!(
    output::AbstractVector,
    itp::AbstractInterpolantND{Tg, Tv, N},
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}},
    ops::NTuple{N, AbstractEvalOp},
    search::Tuple{Vararg{AbstractSearchPolicy, N}},
    hints=nothing
) where {Tg, Tv, N}
    @inbounds for k in 1:length(queries)
        cell = _locate_cell(itp, queries[k], search, hints)
        output[k] = _eval_at_cell(itp, cell, ops)
    end
    return output
end

# ========================================
# Shared ND Callable Interface
# ========================================
#
# Vector query, SoA batch, and AoS batch callables are identical across all
# ND interpolant types. Define once on AbstractInterpolantND.
# Each concrete type only needs: scalar tuple callable + in-place batch callables.

# Vector query → tuple conversion for ForwardDiff/Optim compatibility
@inline function (itp::AbstractInterpolantND{Tg, Tv, N})(
    query::AbstractVector{<:Real};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    length(query) == N || throw(DimensionMismatch(
        "expected $N-element vector, got $(length(query))-element vector"
    ))
    query_tuple = ntuple(i -> @inbounds(query[i]), Val(N))
    return itp(query_tuple; deriv=deriv, search=search, hint=hint)
end

# SoA batch: allocate output + delegate to in-place
function (itp::AbstractInterpolantND{Tg, Tv, N})(
    queries::Tuple{Vararg{AbstractVector{<:Real}, N}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    Tq = _query_eltype(queries)
    output = Vector{promote_type(Tv, Tg, Tq)}(undef, length(queries[1]))
    return itp(output, queries; deriv=deriv, search=search, hint=hint)
end

# AoS batch: allocate output + delegate to in-place
function (itp::AbstractInterpolantND{Tg, Tv, N})(
    queries::AbstractVector{<:Tuple{Vararg{Real, N}}};
    deriv::Union{Int, Val, NTuple{N,Int}} = 0,
    search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
) where {Tg, Tv, N}
    Tq = _query_eltype(queries)
    output = Vector{promote_type(Tv, Tg, Tq)}(undef, length(queries))
    return itp(output, queries; deriv=deriv, search=search, hint=hint)
end

# ========================================
# Query Element Type Extraction
# ========================================

@inline _query_eltype(queries::Tuple{Vararg{AbstractVector}}) =
    promote_type(map(eltype, queries)...)

@inline _query_eltype(::AbstractVector{T}) where {T<:Tuple} =
    promote_type(fieldtypes(T)...)
