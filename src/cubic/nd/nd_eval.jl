# ========================================
# Generic ND Hermite Evaluation
# ========================================
#
# Callable interface and evaluation functions for N-dimensional cubic
# Hermite interpolation.
#
# Key features:
# - Tg/Tv type separation (grid vs value types)
# - @generated tensor product collapse for zero-allocation O(1) evaluation
# - Support for derivatives via AbstractEvalOp dispatch
# - Tuple-based query interface for type stability
#
# The @generated function unrolls the tensor product at compile time,
# generating specialized code for each dimension count N.

# Debug flag for @generated function code inspection
# Usage: FastInterpolations._DEBUG_GENERATED_CELL[] = true
const _DEBUG_GENERATED_CELL = Ref(false)

# ========================================
# CALLABLE INTERFACE
# ========================================
#
# Design: Strict API for Performance
# -----------------------------------
# deriv accepts only:
#   - Int (0-3): broadcast to all axes, fast-path via @_dispatch_deriv
#   - Val{D}: compile-time specification for zero-allocation
#
# Raw Tuple (1,0) is rejected to prevent Union type performance traps.

"""
    (itp::CubicInterpolantND)(query; deriv=0, search=itp.searches)

Evaluate N-dimensional cubic Hermite interpolant at query point.

# Arguments
- `query::NTuple{N, Real}`: Query coordinates as N-tuple

# Keywords
- `deriv`: Derivative order (0-3) or `Val` for mixed partials
  - `Int`: 0=value, 1=∇f, 2=∇²f, 3=∇³f (broadcast to all axes)
  - `Val{D}`: e.g., `Val((1,0,0))` for ∂f/∂x only
  - Raw tuple `(1,0)` NOT accepted (use `Val((1,0))`)
- `search`: Search policy override
  - Single `AbstractSearchPolicy`: applied to all axes
  - `NTuple{N}` of policies: per-axis override

# Returns
- Interpolated value (or derivative) at query point

# Examples
```julia
itp = cubic_interp((x, y, z), data)
itp((1.0, 0.5, 0.3))                  # Value at point
itp((1.0, 0.5, 0.3); deriv=1)         # All first derivatives (∇f)
itp((1.0, 0.5, 0.3); deriv=Val((1,0,0))) # ∂f/∂x only (type-stable)
itp((1.0, 0.5, 0.3); deriv=Val((0,1,0))) # ∂f/∂y only (type-stable)
itp((1.0, 0.5, 0.3); deriv=Val(2))       # All second derivatives
```

# Performance Notes
- `deriv=0,1,2,3` literals: constant-propagation → zero-allocation
- `Val((1,0,...))`: compile-time dispatch → zero-allocation
"""
# Single-point evaluation
@inline function (itp::CubicInterpolantND{Tg, Tv, N})(
    query::NTuple{N, <:Real};
    deriv::Union{Int, Val}=0,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=itp.searches
) where {Tg, Tv, N}
    # Note: Don't convert to Tg - preserve query type for AD support
    search_tuple = _resolve_search_nd(search, Val(N))

    if deriv isa Int
        # Int path: macro dispatch ensures concrete op type
        @_dispatch_deriv deriv => op begin
            ops = ntuple(_ -> op, Val(N))
            return _eval_nd_hermite(itp, query, ops, search_tuple)
        end
    else
        # Val path: compile-time resolution
        ops = _resolve_deriv_nd(deriv, Val(N))
        return _eval_nd_hermite(itp, query, ops, search_tuple)
    end
end

# ========================================
# BATCH EVALUATION: Tuple of Vectors (SoA)
# ========================================
# Standard format for separated coordinate arrays (DataFrames, CSV columns)

"""
    (itp::CubicInterpolantND)(queries::NTuple{N,Vector}; deriv=0, search=...)

Batch evaluation with Structure-of-Arrays (SoA) input.

# Arguments
- `queries::NTuple{N, AbstractVector}`: Coordinate vectors per axis

# Example
```julia
xs, ys = rand(1000), rand(1000)
results = itp((xs, ys))  # Evaluate at all 1000 points
```
"""
function (itp::CubicInterpolantND{Tg, Tv, N})(
    queries::NTuple{N, <:AbstractVector{<:Real}};
    deriv::Union{Int, Val}=0,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=itp.searches
) where {Tg, Tv, N}
    n_queries = length(queries[1])
    for d in 2:N
        length(queries[d]) == n_queries || throw(DimensionMismatch(
            "query vectors must have same length: dim 1 has $n_queries, dim $d has $(length(queries[d]))"
        ))
    end
    search_tuple = _resolve_search_nd(search, Val(N))

    if deriv isa Int
        @_dispatch_deriv deriv => op begin
            ops = ntuple(_ -> op, Val(N))
            return _eval_nd_batch_soa(itp, queries, ops, search_tuple)
        end
    else
        ops = _resolve_deriv_nd(deriv, Val(N))
        return _eval_nd_batch_soa(itp, queries, ops, search_tuple)
    end
end

# ========================================
# BATCH EVALUATION: Vector of Tuples (AoS)
# ========================================
# Standard format for point lists (StaticArrays, geometry libraries)

"""
    (itp::CubicInterpolantND)(queries::Vector{<:NTuple{N}}; deriv=0, search=...)

Batch evaluation with Array-of-Structures (AoS) input.

# Arguments
- `queries::AbstractVector{<:NTuple{N}}`: Vector of coordinate tuples

# Example
```julia
points = [(rand(), rand()) for _ in 1:1000]
results = itp(points)  # Evaluate at all 1000 points
```
"""
function (itp::CubicInterpolantND{Tg, Tv, N})(
    queries::AbstractVector{<:NTuple{N, <:Real}};
    deriv::Union{Int, Val}=0,
    search::Union{AbstractSearchPolicy, NTuple{N,AbstractSearchPolicy}}=itp.searches
) where {Tg, Tv, N}
    search_tuple = _resolve_search_nd(search, Val(N))

    if deriv isa Int
        @_dispatch_deriv deriv => op begin
            ops = ntuple(_ -> op, Val(N))
            return _eval_nd_batch_aos(itp, queries, ops, search_tuple)
        end
    else
        ops = _resolve_deriv_nd(deriv, Val(N))
        return _eval_nd_batch_aos(itp, queries, ops, search_tuple)
    end
end

# ========================================
# BATCH EVALUATION INNER FUNCTIONS
# ========================================

"""
    _eval_nd_batch_soa(itp, queries, ops, search)

Batch evaluation for SoA input. Query type preserved for AD support.
"""
@inline function _eval_nd_batch_soa(
    itp::CubicInterpolantND{Tg, Tv, N},
    queries::NTuple{N, <:AbstractVector{Tq}},
    ops::OPS,
    search::SEARCH
) where {Tg, Tv, Tq<:Real, N, OPS<:NTuple{N,AbstractEvalOp}, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    n_queries = length(queries[1])
    # Include Tq for AD support (Dual numbers propagate through)
    Tout = promote_type(Tv, Tg, Tq)
    results = Vector{Tout}(undef, n_queries)

    @inbounds for k in 1:n_queries
        # Don't convert to Tg - preserve query type for AD
        query_k = ntuple(d -> queries[d][k], Val(N))
        results[k] = _eval_nd_hermite(itp, query_k, ops, search)
    end

    return results
end

"""
    _eval_nd_batch_aos(itp, queries, ops, search)

Batch evaluation for AoS input. Query type preserved for AD support.
"""
@inline function _eval_nd_batch_aos(
    itp::CubicInterpolantND{Tg, Tv, N},
    queries::AbstractVector{<:NTuple{N, Tq}},
    ops::OPS,
    search::SEARCH
) where {Tg, Tv, Tq<:Real, N, OPS<:NTuple{N,AbstractEvalOp}, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    n_queries = length(queries)
    # Include Tq for AD support (Dual numbers propagate through)
    Tout = promote_type(Tv, Tg, Tq)
    results = Vector{Tout}(undef, n_queries)

    @inbounds for k in 1:n_queries
        # Don't convert to Tg - preserve query type for AD
        results[k] = _eval_nd_hermite(itp, queries[k], ops, search)
    end

    return results
end

# ========================================
# CORE EVALUATION
# ========================================

"""
    _eval_nd_hermite(itp, query, ops, search)

Core N-dimensional Hermite evaluation using generic helpers and @generated cell kernel.

All helper functions use ntuple(Val(N)) pattern for zero-allocation operation.
"""
@inline function _eval_nd_hermite(
    itp::CubicInterpolantND{Tg, Tv, N},
    query::NTuple{N, Tq},
    ops::OPS,
    search::SEARCH
) where {Tg, Tv, Tq<:Real, N, OPS<:NTuple{N,AbstractEvalOp}, SEARCH<:NTuple{N,AbstractSearchPolicy}}
    # Extract all fields using @generated helpers (zero allocation)
    grids = _get_grids(itp)       # NTuple{N}
    spacings = _get_spacings(itp) # NTuple{N}
    extraps = _get_extraps(itp)   # NTuple{N}

    # Per-axis setup using ntuple pattern
    q_evals = _handle_all_extraps(query, grids, extraps)

    indices, Ls, _ = _search_all_intervals(q_evals, grids, spacings, search)

    hs, inv_hs, dLs = _compute_all_local_params(q_evals, spacings, indices, Ls)

    # Tensor product collapse using @generated kernel
    return _eval_nd_cell(itp.nodal_derivs.partials, indices, hs, inv_hs, dLs, ops)
end

"""
    _eval_nd_hermite(itp::CubicInterpolantND{Tg,Tv,2}, query, ops, search)

Specialized 2D Hermite evaluation that eliminates ntuple closure overhead.
Uses direct destructuring and inline operations for better LLVM optimization.
"""
@inline function _eval_nd_hermite(
    itp::CubicInterpolantND{Tg, Tv, 2},
    query::NTuple{2, Tq},
    ops::NTuple{2, <:AbstractEvalOp},
    search::NTuple{2, <:AbstractSearchPolicy}
) where {Tg, Tv, Tq<:Real}
    # Direct destructuring (no @generated helper overhead)
    xq, yq = query
    grid_x, grid_y = itp.grids
    spacing_x, spacing_y = itp.spacings
    extrap_x, extrap_y = itp.extraps
    op_x, op_y = ops
    search_x, search_y = search

    # Inline extrapolation handling (no ntuple closure)
    x_eval = _handle_axis_extrap(xq, grid_x, extrap_x)
    y_eval = _handle_axis_extrap(yq, grid_y, extrap_y)

    # Inline interval search (no ntuple closure)
    searcher_x = _to_searcher(search_x)
    searcher_y = _to_searcher(search_y)
    ix, xL, _ = search_interval(searcher_x, grid_x, spacing_x, x_eval)
    iy, yL, _ = search_interval(searcher_y, grid_y, spacing_y, y_eval)

    # Inline local params (no ntuple closure)
    hx = _get_h(spacing_x, ix)
    hy = _get_h(spacing_y, iy)
    inv_hx = _get_inv_h(spacing_x, ix)
    inv_hy = _get_inv_h(spacing_y, iy)
    dLx = x_eval - xL
    dLy = y_eval - yL

    # Same @generated tensor product kernel
    return _eval_nd_cell(
        itp.nodal_derivs.partials,
        (ix, iy), (hx, hy), (inv_hx, inv_hy), (dLx, dLy), (op_x, op_y)
    )
end

# ========================================
# @GENERATED TENSOR PRODUCT EVALUATION
# ========================================
#
# Algorithm: Collapse dimensions 1 to N sequentially via tensor product
#
# Notation:
#   - corner_bits: binary encoding of cell corner for dims d+1..N
#   - deriv_bits: binary encoding of remaining derivatives for dims d+1..N
#   - g_{stage}_{corner}_{deriv}: intermediate value after collapsing dim 'stage'
#
# For dimension d collapse:
#   - Input: 2^(N-d+1) values at 2^(N-d+1) locations (corner × deriv combos)
#   - Output: 2^(N-d) values at 2^(N-d) locations
#   - Each output = hermite_kernel_1d(4 inputs from adjacent corners in dim d)

"""
    _varname(stage, corner, deriv) -> Symbol

Generate variable name for intermediate value.
- stage=0: partials access (not a variable)
- stage>0: g_{stage}_{corner}_{deriv}
"""
_varname(stage::Int, corner::Int, deriv::Int) = Symbol("g_$(stage)_$(corner)_$(deriv)")

"""
    _partial_index(deriv_bits) -> Int

Convert derivative bit pattern to 1-based partial index.
Bit d set means differentiated w.r.t. dimension d.
"""
_partial_index(deriv_bits::Int) = deriv_bits + 1

"""
    _corner_offset_expr(corner_bits, N) -> Vector{Int}

Generate index offset tuple for a cell corner.
For dimension d, bit (d-1) in corner_bits determines +0 or +1 offset.
"""
function _corner_offset_expr(corner_bits::Int, N::Int)
    offsets = [((corner_bits >> (d-1)) & 1) for d in 1:N]
    return offsets
end

"""
    _eval_nd_cell(partials, indices, hs, inv_hs, dLs, ops)

@generated tensor product collapse for N-dimensional Hermite evaluation.

Generates specialized code at compile time for each N, unrolling all
2^N kernel calls into a flat sequence of operations.

# Type Parameters
- `Tv`: Value type (from partials array)
- `Tg`: Grid type (from hs, inv_hs, dLs)
- `N`: Number of dimensions
- `NP1`: N + 1 (partials array dimensionality)

# Arguments
- `partials::Array{Tv, NP1}`: Precomputed partial derivatives
- `indices::NTuple{N, Int}`: Cell indices for each dimension
- `hs::NTuple{N, Tg}`: Cell widths for each dimension
- `inv_hs::NTuple{N, Tg}`: Inverse cell widths
- `dLs::NTuple{N, Tg}`: Distances from left cell boundary
- `ops::NTuple{N, AbstractEvalOp}`: Evaluation operations per dimension
"""
@inline @generated function _eval_nd_cell(
    partials::Array{Tv, NP1},
    indices::NTuple{N, Int},
    hs::NTuple{N, Tg},
    inv_hs::NTuple{N, Tg},
    dLs::NTuple{N, Tq},
    ops::NTuple{N, AbstractEvalOp}
) where {Tv, Tg, Tq, N, NP1}
    # Validate dimensions
    NP1 == N + 1 || error("NP1 must equal N+1")

    # Generate all statements
    stmts = Expr[]

    # Unpack tuples using destructuring (efficient AST)
    for (prefix, source) in [("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
                              ("dL_", :dLs), ("op_", :ops)]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    # Collapse each dimension
    for stage in 1:N
        # After collapsing dim 'stage', we have 2^(N-stage) corners and derivs
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    # Read from partials array
                    function make_partial_access(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)

                        # Build index expression: partials[p_idx, idx_1 + off_1, ...]
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_partial_access(0, 0)
                    fR = make_partial_access(1, 0)
                    dfL = make_partial_access(0, 1)
                    dfR = make_partial_access(1, 1)
                else
                    # Read from previous stage variables
                    prev_stage = stage - 1
                    fL = _varname(prev_stage, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev_stage, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev_stage, 0 | (corner << 1), 1 | (deriv << 1))
                    dfR = _varname(prev_stage, 1 | (corner << 1), 1 | (deriv << 1))
                end

                h = Symbol("h_", stage)
                inv_h = Symbol("inv_h_", stage)
                dL = Symbol("dL_", stage)
                op = Symbol("op_", stage)

                kernel_call = :(_hermite_kernel_1d($op, $fL, $fR, $dfL, $dfR, $h, $inv_h, $dL))
                push!(stmts, :($out_var = $kernel_call))
            end
        end
    end

    # Final result is g_{N}_{0}_{0}
    final_var = _varname(N, 0, 0)
    push!(stmts, :(return $final_var))

    # Wrap in quote block with @inbounds and @inline_meta
    result = quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
    end

    # Debug output (controlled by _DEBUG_GENERATED_CELL flag)
    if _DEBUG_GENERATED_CELL[]
        function count_ast_nodes(ex)
            if ex isa Expr
                return 1 + sum(count_ast_nodes(arg) for arg in ex.args; init=0)
            else
                return 1
            end
        end

        println("=" ^ 60)
        println("Generated _eval_nd_cell for N=$N, Tv=$Tv, Tg=$Tg")
        println("AST nodes: ", count_ast_nodes(result))
        println("=" ^ 60)
        println(Base.remove_linenums!(deepcopy(result)))
        println("=" ^ 60)
    end

    return result
end
