# ========================================
# NoInterp / GridIdx Support
# ========================================
#
# Two independent concepts:
#   GridIdx(k)  — query-time: "slice this axis at index k"
#   NoInterp()  — build-time: "this axis is discrete, skip precomputation"
#
# Strategy: detect GridIdx positions in query tuple TYPE at compile time,
# pre-slice data and filter all per-axis tuples to Real-only axes,
# then delegate to existing reduced-dim APIs (zero internal kernel changes).
#
# Include order: after tensor_product_oneshot.jl (needs interp, _eval_hetero_nd_cell, etc.)

# ========================================
# Compile-Time Traits
# ========================================

# _has_nointerp_method is defined in tensor_product_interpolant.jl (included before this file)

# ========================================
# @generated Tuple Operations
# ========================================

"""
    _slice_grididx(data, query) -> view

Slice `data` at `GridIdx` positions in `query`, keeping `:` for Real positions.
Result is a reduced-dimension view. Dispatch on query TYPE (zero compile explosion).

Example: `data` is 3D, `query = (0.5, GridIdx(5), 0.3)` → `@view data[:, 5, :]`
"""
@generated function _slice_grididx(data::AbstractArray{Tv, N}, query::Q) where {Tv, N, Q}
    grididx_dims = [d for d in 1:N if fieldtype(Q, d) <: GridIdx]
    idx_exprs = [d in grididx_dims ? :(query[$d].idx) : :(:) for d in 1:N]
    bounds_checks = [
        :(
                1 <= query[$d].idx <= size(data, $d) ||
                _throw_grididx_oob($d, query[$d].idx, size(data, $d))
            )
            for d in grididx_dims
    ]
    return quote
        Base.@_inline_meta
        @boundscheck begin
            $(bounds_checks...)
        end
        @inbounds @view data[$(idx_exprs...)]
    end
end

"""
    _filter_real_axes(vals, ::Type{Q}) -> filtered tuple

Keep only elements of `vals` at positions where `Q` has a `Real` (non-GridIdx) field type.
"""
@generated function _filter_real_axes(vals::Tuple{Vararg{Any, N}}, ::Type{Q}) where {N, Q}
    kept = [i for i in 1:N if !(fieldtype(Q, i) <: GridIdx)]
    isempty(kept) && return :(())
    exprs = [:(vals[$i]) for i in kept]
    return :(tuple($(exprs...)))
end

# ========================================
# Validation Helpers
# ========================================

"""
    _any_grididx_has_nonzero_deriv(query, deriv_tuple) -> Bool

Runtime check: returns true if any GridIdx axis has a non-zero DerivOp.
Used to short-circuit to zero for derivatives on discrete axes.
"""
@generated function _any_grididx_has_nonzero_deriv(query::Q, deriv_tuple::Tuple{Vararg{DerivOp, N}}) where {Q, N}
    checks = [:(deriv_order(deriv_tuple[$d]) != 0) for d in 1:N if fieldtype(Q, d) <: GridIdx]
    isempty(checks) && return :(false)
    cond = length(checks) == 1 ? checks[1] : Expr(:||, checks...)
    return :($cond)
end

@noinline _throw_grididx_oob(d, idx, n) =
    throw(ArgumentError("GridIdx axis $d: index $idx out of range 1:$n"))

@noinline _throw_nointerp_needs_grididx(d) =
    throw(ArgumentError("Axis $d uses NoInterp but query provides Real; use GridIdx(k) for NoInterp axes"))

@noinline _throw_grididx_on_interp_axis(d, method) =
    throw(
    ArgumentError(
        "Axis $d uses $method but query provides GridIdx; " *
            "GridIdx is only valid for NoInterp axes in interpolant queries"
    )
)

"""
    _validate_grididx_query_oneshot(query, data)

Validate a one-shot query containing GridIdx: check that every element is either
`Real` or `GridIdx`, and that GridIdx indices are in bounds.
"""
@generated function _validate_grididx_query_oneshot(query::Q, data::AbstractArray{<:Any, N}) where {Q, N}
    fieldcount(Q) == N || return :(throw(DimensionMismatch("query has $(fieldcount($Q)) elements but data is $($N)-dimensional")))
    checks = Expr[]
    for d in 1:N
        ft = fieldtype(Q, d)
        if ft <: GridIdx
            push!(
                checks, :(
                    1 <= query[$d].idx <= size(data, $d) ||
                        _throw_grididx_oob($d, query[$d].idx, size(data, $d))
                )
            )
        elseif !(ft <: Real)
            push!(checks, :(throw(ArgumentError("Axis $($d): query element must be Real or GridIdx, got " * string(typeof(query[$d]))))))
        end
    end
    return quote
        $(checks...)
        nothing
    end
end

"""
    _validate_nointerp_grididx(methods, query)

Validate interpolant query: NoInterp axes must have GridIdx, non-NoInterp axes must have Real.
Bounds checking on GridIdx values is deferred to `_eval_nointerp` / `_slice_grididx`.
"""
@generated function _validate_nointerp_grididx(methods::M, query::Q) where {M <: Tuple, Q <: Tuple}
    N = fieldcount(M)
    checks = Expr[]
    for d in 1:N
        is_nointerp = fieldtype(M, d) <: NoInterp
        is_grididx = fieldtype(Q, d) <: GridIdx
        if is_nointerp && !is_grididx
            push!(checks, :(_throw_nointerp_needs_grididx($d)))
        elseif !is_nointerp && is_grididx
            push!(checks, :(_throw_grididx_on_interp_axis($d, methods[$d])))
        end
    end
    isempty(checks) && return :(nothing)
    return quote
        $(checks...)
        nothing
    end
end

# ========================================
# NoInterp-Aware Interpolant Eval
# ========================================

"""
    _eval_nointerp(itp, query, ops, search, hint)

@generated eval for interpolant with NoInterp axes. Slices partials/data at GridIdx
positions, filters all per-axis tuples to Real-only axes, delegates to existing pipeline.
"""
@generated function _eval_nointerp(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q, ops_full, search, hint,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q}
    nointerp_dims = [d for d in 1:N if fieldtype(M, d) <: NoInterp]
    real_dims = [d for d in 1:N if !(fieldtype(M, d) <: NoInterp)]
    N_r = length(real_dims)

    # Build slice expressions: `:` for Real axes, `query[d].idx` for NoInterp
    slice_args = [d in nointerp_dims ? :(query[$d].idx) : :(:) for d in 1:N]

    # Bounds checks for GridIdx on NoInterp axes (before @inbounds slicing)
    # Use grid length as the canonical size (works for both HeteroPartials and raw Array)
    bounds_checks = [
        :(
                1 <= query[$d].idx <= size(itp.grids[$d], 1) ||
                _throw_grididx_oob($d, query[$d].idx, size(itp.grids[$d], 1))
            )
            for d in nointerp_dims
    ]

    # Build reduced tuple expressions
    r_grids = [:(itp.grids[$d]) for d in real_dims]
    r_spacings = [:(itp.spacings[$d]) for d in real_dims]
    r_methods = [:(itp.methods[$d]) for d in real_dims]
    r_extraps = [:(itp.extraps[$d]) for d in real_dims]
    r_searches = [:(search[$d]) for d in real_dims]
    r_query = [:(query[$d]) for d in real_dims]
    r_ops = [:(ops_full[$d]) for d in real_dims]

    # Check if any NoInterp axis has a non-zero DerivOp → return zero
    # (discrete axes have zero derivatives by definition)
    nointerp_deriv_checks = [:(deriv_order(ops_full[$d]) != 0) for d in nointerp_dims]
    nointerp_deriv_guard = if isempty(nointerp_deriv_checks)
        :()
    else
        cond = length(nointerp_deriv_checks) == 1 ? nointerp_deriv_checks[1] :
            Expr(:||, nointerp_deriv_checks...)
        :(
            if $cond
                return 0 * _zero_ref(itp)
            end
        )
    end

    if N_r == 0
        # All-NoInterp: pure table lookup (or zero if any deriv requested)
        if D <: HeteroPartials
            return quote
                Base.@_inline_meta
                $(bounds_checks...)
                $nointerp_deriv_guard
                @inbounds itp.data.partials[1, $(slice_args...)]
            end
        else
            return quote
                Base.@_inline_meta
                $(bounds_checks...)
                $nointerp_deriv_guard
                @inbounds itp.data[$(slice_args...)]
            end
        end
    end

    if D <: HeteroPartials
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
            $nointerp_deriv_guard
            # Slice partials at GridIdx positions
            p_sliced = @inbounds @view itp.data.partials[:, $(slice_args...)]

            # Reduced tuples (all types compile-time known)
            rq = ($(r_query...),)
            rg = ($(r_grids...),)
            rs = ($(r_spacings...),)
            re = ($(r_extraps...),)
            ro = ($(r_ops...),)
            rm = ($(r_methods...),)

            # Standard eval pipeline on reduced dims
            _validate_nd_domain(rg, rq, re)
            oob = _try_fill_oob(rq, rg, re, ro, @inbounds p_sliced[1])
            oob !== nothing && return oob

            q_eval = _handle_all_extraps(rq, rg, re)
            rsrc = ($(r_searches...),)
            search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
            idxs, Ls, _ = _search_all_intervals(q_eval, rg, rs, search_r, nothing)
            hs, inv_hs, dLs = _compute_all_local_params(q_eval, rs, idxs, Ls)
            return _eval_hetero_nd_cell(p_sliced, idxs, hs, inv_hs, dLs, ro, rm)
        end
    else
        # OnTheFly: slice data, delegate to reduced-dim _collapse_dims
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
            $nointerp_deriv_guard
            d_sliced = @inbounds @view itp.data[$(slice_args...)]
            rq = ($(r_query...),)
            rg = ($(r_grids...),)
            rm = ($(r_methods...),)
            re = ($(r_extraps...),)
            ro = ($(r_ops...),)

            # FillExtrap check (same as PreCompute path)
            _validate_nd_domain(rg, rq, re)
            oob = _try_fill_oob(rq, rg, re, ro, @inbounds d_sliced[1])
            oob !== nothing && return oob

            q_eval = _handle_all_extraps(rq, rg, re)
            rsrc = ($(r_searches...),)
            search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
            return _collapse_dims(d_sliced, rg, rm, re, q_eval, ro, search_r, nothing)
        end
    end
end

# ========================================
# One-Shot interp with GridIdx
# ========================================
# Pre-slices data at GridIdx positions, filters all tuples to Real-only axes,
# then delegates to the existing reduced-dim interp one-shot API.
#
# Dispatch: Tuple{Float64, GridIdx} does NOT match Tuple{Vararg{Real, N}},
# so Julia selects this method only when at least one GridIdx is present.

"""
    interp(grids, data, query; method, ...) where query contains GridIdx

One-shot interpolation with GridIdx elements in the query.
`GridIdx` axes are sliced out of `data` before delegating to the
existing reduced-dimension one-shot API.

# Examples
```julia
x, y = range(0, 1, 50), range(0, 1, 30)
data = [sin(xi) * cos(yj) for xi in x, yj in y]

# Slice y at index 5, interpolate x with cubic
val = interp((x, y), data, (0.5, GridIdx(5)); method=(CubicInterp(), NoInterp()))

# Works with any method (one-shot pre-slices regardless)
val = interp((x, y), data, (0.5, GridIdx(5)); method=(CubicInterp(), LinearInterp()))

# All-GridIdx: pure table lookup
val = interp((x, y), data, (GridIdx(3), GridIdx(5)); method=(NoInterp(), NoInterp()))
```
"""
function interp(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Q;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy}}} = AutoSearch(),
        hint = nothing,
    ) where {N, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    # Only reach here when at least one GridIdx (all-Real → more specific existing method)
    _validate_grididx_query_oneshot(query, data)
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method

    # Check if any GridIdx axis has non-zero deriv → return zero
    # (discrete axes have zero derivatives by definition)
    deriv_t = deriv isa DerivOp ? ntuple(_ -> deriv, Val(N)) : deriv
    if _any_grididx_has_nonzero_deriv(query, deriv_t)
        Tg = _promote_grid_eltype(grids)
        Tg = Tg <: AbstractFloat ? Tg : Float64
        return zero(_output_eltype(eltype(data), Tg, typeof(first(query))))
    end

    # Pre-slice data and filter all tuples to Real-only axes
    QT = typeof(query)
    data_r = _slice_grididx(data, query)
    grids_r = _filter_real_axes(grids, QT)
    query_r = _filter_real_axes(query, QT)
    methods_r = _filter_real_axes(method_tuple, QT)

    # All-GridIdx edge case: pure table lookup (0-dim view → extract scalar)
    if grids_r === ()
        return data_r[]
    end

    # Resolve remaining per-axis kwargs to N-tuples, then filter to Real axes
    deriv_t = deriv isa DerivOp ? ntuple(_ -> deriv, Val(N)) : deriv
    extrap_t = extrap isa AbstractExtrap ? ntuple(_ -> extrap, Val(N)) : extrap
    search_t = search isa AbstractSearchPolicy ? ntuple(_ -> search, Val(N)) : search
    deriv_r = _filter_real_axes(deriv_t, QT)
    extrap_r = _filter_real_axes(extrap_t, QT)
    search_r = _filter_real_axes(search_t, QT)

    return interp(
        grids_r, data_r, query_r;
        method = methods_r, deriv = deriv_r, extrap = extrap_r,
        search = search_r, hint = nothing
    )
end

# ========================================
# Vector Calculus: gradient / hessian / laplacian with NoInterp
# ========================================
#
# Strategy: reuse `_eval_nointerp` for each derivative component.
# NoInterp axes contribute 0 to all derivatives.
# Simpler than adding _locate_cell/_eval_at_cell overloads at the cost of
# repeated search per component (acceptable: NoInterp reduces N, so N_r is small).
#
# Disambiguation: We define methods for (TensorProductInterpolantND, Tuple{Vararg{Union{Real,GridIdx},N}}).
# This overlaps with the standard (AbstractInterpolantND, Tuple{Vararg{Real,N}}) methods.
# Julia resolves this via explicit disambiguating methods for (TensorProductInterpolantND, Tuple{Vararg{Real,N}})
# that delegate to the standard vector_calculus implementations.

# --- Disambiguation methods (all-Real query on TensorProductInterpolantND) ---
# These are more specific than BOTH the generic (AbstractInterpolantND, NTuple{N,Real})
# and our (TensorProductInterpolantND, NTuple{N, Union{Real,GridIdx}}) methods.
@generated function gradient(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q;
        hint = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
    # Delegate to standard gradient via _locate_cell/_eval_at_cell
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    zero_tuple = [:(0 * zref) for _ in 1:N]
    return quote
        search = _resolve_search_nd(itp.searches, Val($N), query)
        if _is_fill_oob(query, itp.grids, itp.extraps)
            zref = _zero_ref(itp)
            return tuple($(zero_tuple...))
        end
        cell = _locate_cell(itp, query, search, hint)
        return tuple($(deriv_calls...))
    end
end

@generated function hessian(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q;
        hint = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
    stmts = Expr[]
    for i in 1:N
        ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
        push!(stmts, :(H[$i, $i] = _eval_at_cell(itp, cell, $ops)))
    end
    for i in 1:N, j in (i + 1):N
        ops = ntuple(k -> (k == i || k == j) ? DerivOp{1}() : DerivOp{0}(), N)
        push!(
            stmts, quote
                val = _eval_at_cell(itp, cell, $ops)
                H[$i, $j] = val
                H[$j, $i] = val
            end
        )
    end
    return quote
        Tq = promote_type(eltype(query), $Tg, $Tv)
        H = Matrix{Tq}(undef, $N, $N)
        search = _resolve_search_nd(itp.searches, Val($N), query)
        if _is_fill_oob(query, itp.grids, itp.extraps)
            fill!(H, zero(Tq))
            return H
        end
        cell = _locate_cell(itp, query, search, hint)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

@generated function laplacian(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q;
        hint = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    return quote
        search = _resolve_search_nd(itp.searches, Val($N), query)
        if _is_fill_oob(query, itp.grids, itp.extraps)
            return 0 * _zero_ref(itp)
        end
        cell = _locate_cell(itp, query, search, hint)
        return +($(deriv_calls...))
    end
end

"""
    gradient(itp::TensorProductInterpolantND, query::Tuple{..., GridIdx, ...})

Gradient with NoInterp support. Returns N-tuple with zeros at NoInterp positions.

# Examples
```julia
itp = interp((x, y), data; method=(CubicInterp(), NoInterp()))
gradient(itp, (0.5, GridIdx(3)))  # → (∂f/∂x, 0.0)
```
"""
@generated function gradient(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q;
        hint = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    grad_exprs = []
    for i in 1:N
        if i in nointerp_dims
            push!(grad_exprs, :(0 * zref))
        else
            ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
            push!(grad_exprs, :(_eval_nointerp(itp, query, $ops, itp.searches, hint)))
        end
    end

    return quote
        Base.@_inline_meta
        _validate_nointerp_grididx(itp.methods, query)
        zref = _zero_ref(itp)
        return tuple($(grad_exprs...))
    end
end

"""
    hessian(itp::TensorProductInterpolantND, query::Tuple{..., GridIdx, ...})

Hessian with NoInterp support. Returns N×N matrix with zero rows/columns at NoInterp positions.
"""
@generated function hessian(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q;
        hint = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    stmts = Expr[]
    for i in 1:N
        i in nointerp_dims && continue
        # Diagonal: ∂²f/∂xᵢ²
        ops_diag = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
        push!(stmts, :(H[$i, $i] = _eval_nointerp(itp, query, $ops_diag, itp.searches, hint)))
        # Off-diagonal: ∂²f/∂xᵢ∂xⱼ (only Real×Real pairs)
        for j in (i + 1):N
            j in nointerp_dims && continue
            ops_mixed = ntuple(k -> (k == i || k == j) ? DerivOp{1}() : DerivOp{0}(), N)
            push!(
                stmts, quote
                    val = _eval_nointerp(itp, query, $ops_mixed, itp.searches, hint)
                    H[$i, $j] = val
                    H[$j, $i] = val
                end
            )
        end
    end

    # Compute output type from Real query elements only (GridIdx is not numeric)
    real_query_types = [fieldtype(Q, d) for d in 1:N if !(d in nointerp_dims)]
    Tq_expr = isempty(real_query_types) ? Tv : :(promote_type($(real_query_types...), $Tg, $Tv))

    return quote
        _validate_nointerp_grididx(itp.methods, query)
        Tq = $Tq_expr
        H = zeros(Tq, $N, $N)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

"""
    laplacian(itp::TensorProductInterpolantND, query::Tuple{..., GridIdx, ...})

Laplacian with NoInterp support. Sums ∂²f/∂xᵢ² only over interpolated axes.
"""
@generated function laplacian(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q;
        hint = nothing,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    terms = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
                :(_eval_nointerp(itp, query, $ops, itp.searches, hint))
            end for i in 1:N if !(i in nointerp_dims)
    ]

    isempty(terms) && return :(0 * _zero_ref(itp))
    return quote
        Base.@_inline_meta
        _validate_nointerp_grididx(itp.methods, query)
        return +($(terms...))
    end
end

# ========================================
# Batch interp! / interp with GridIdx (Fixed-Slice)
# ========================================
#
# Supports SoA queries with scalar GridIdx at NoInterp positions:
#   (xvec, GridIdx(5), yvec) — all queries share the same GridIdx indices
#
# Strategy: pre-slice data once at GridIdx positions, filter all per-axis tuples
# to Real-only axes, then delegate to existing batch `interp!`.

"""
    _filter_grididx_batch(queries, ::Type{Q}) -> (queries_real, grididx_template)

For SoA batch queries with GridIdx, extract:
- `queries_real`: tuple of vectors (only Real axes)
- Used to build the pre-slice query template
"""
@generated function _build_grididx_template(queries::Q) where {Q <: Tuple}
    N = fieldcount(Q)
    # Build a template query: vectors → first element as Real, GridIdx → as-is
    template_exprs = [
        fieldtype(Q, d) <: GridIdx ? :(queries[$d]) : :(queries[$d][1])
            for d in 1:N
    ]
    return :(tuple($(template_exprs...)))
end

@generated function _filter_real_batch_queries(queries::Q) where {Q <: Tuple}
    kept = [i for i in 1:fieldcount(Q) if !(fieldtype(Q, i) <: GridIdx)]
    isempty(kept) && return :(())
    return :(tuple($([:(queries[$i]) for i in kept]...)))
end

@generated function _has_grididx_in_batch(::Type{Q}) where {Q <: Tuple}
    has_vec = any(i -> fieldtype(Q, i) <: AbstractVector, 1:fieldcount(Q))
    has_gidx = any(i -> fieldtype(Q, i) <: GridIdx, 1:fieldcount(Q))
    return :($(has_vec && has_gidx))
end

"""
    _interp_batch_grididx!(output, grids, data, queries; method, ...)

Internal: batch one-shot with fixed GridIdx slice. Pre-slices data once,
filters all tuples to Real-only axes, then delegates to existing `interp!`.
Separated from `interp!` to avoid dispatch conflicts with the standard batch path.
"""
function _interp_batch_grididx!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy}}} = AutoSearch(),
        hint = nothing,
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method

    # Build a template query point for data slicing
    template = _build_grididx_template(queries)
    QT = typeof(template)

    # Pre-slice data at GridIdx positions (ONE-TIME)
    _validate_grididx_query_oneshot(template, data)
    data_r = _slice_grididx(data, template)

    # Filter to Real-only axes
    grids_r = _filter_real_axes(grids, QT)
    methods_r = _filter_real_axes(method_tuple, QT)
    queries_r = _filter_real_batch_queries(queries)

    # Resolve and filter kwargs
    deriv_t = deriv isa DerivOp ? ntuple(_ -> deriv, Val(N)) : deriv
    extrap_t = extrap isa AbstractExtrap ? ntuple(_ -> extrap, Val(N)) : extrap
    search_t = search isa AbstractSearchPolicy ? ntuple(_ -> search, Val(N)) : search
    deriv_r = _filter_real_axes(deriv_t, QT)
    extrap_r = _filter_real_axes(extrap_t, QT)
    search_r = _filter_real_axes(search_t, QT)

    return interp!(
        output, grids_r, data_r, queries_r;
        method = methods_r, deriv = deriv_r, extrap = extrap_r,
        search = search_r, hint = nothing
    )
end

"""
    interp_batch_grididx!(output, grids, data, (xvec, GridIdx(k), yvec); method, ...)

Batch one-shot with fixed GridIdx slice. All query points share the same GridIdx indices.
Query format (SoA): `(xvec, GridIdx(5), yvec)` — vectors for interpolated axes,
scalar `GridIdx` for NoInterp axes.

Pre-slices data once at GridIdx positions, then delegates to existing `interp!` on reduced dims.

# Examples
```julia
x, y, z = range(0,1,50), range(0,1,30), range(0,1,20)
data = rand(50, 30, 20)
xq, zq = rand(100), rand(100)
output = zeros(100)
interp_batch_grididx!(output, (x, y, z), data, (xq, GridIdx(5), zq);
    method=(CubicInterp(), NoInterp(), LinearInterp()))
```
"""
function interp_batch_grididx!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy}}} = AutoSearch(),
        hint = nothing,
    ) where {N}
    return _interp_batch_grididx!(
        output, grids, data, queries;
        method = method, deriv = deriv, extrap = extrap, search = search, hint = hint
    )
end
