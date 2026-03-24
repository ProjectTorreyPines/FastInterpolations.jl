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

@noinline function _throw_nointerp_needs_grididx(nointerp_dims::Tuple, N::Int)
    parts = [d in nointerp_dims ? "GridIdx(k$d::Int)" : "x$d::Real" for d in 1:N]
    usage = "itp((" * join(parts, ", ") * "))"
    nointerp_str = join(["axis $d" for d in nointerp_dims], ", ")
    throw(
        ArgumentError(
            "NoInterp on $nointerp_str — use GridIdx(k) for discrete axes.\n\n" *
                "  Usage:\n" *
                "    $usage\n\n" *
                "  GridIdx(k) selects the k-th grid point on a NoInterp axis.\n" *
                "  Other axes take real-valued coordinates as usual."
        )
    )
end

"""
    _update_grididx_hints!(hint, query)

Write `hint[d][] = query[d].idx` for each GridIdx position in `query`.
No-op when `hint === nothing`. Used by one-shot and batch paths.
"""
@generated function _update_grididx_hints!(hint, query::Q) where {Q}
    grididx_dims = [d for d in 1:fieldcount(Q) if fieldtype(Q, d) <: GridIdx]
    isempty(grididx_dims) && return :(nothing)
    writes = [:(hint[$d][] = query[$d].idx) for d in grididx_dims]
    return quote
        Base.@_inline_meta
        $(writes...)
        return nothing
    end
end

"""
    _convert_non_nointerp_grididx(methods, grids, query) -> resolved_query

Convert GridIdx on non-NoInterp axes to `grids[d][k]` (Real values).
GridIdx on NoInterp axes are kept as-is. Returns a tuple that may be all-Real
(if no NoInterp axes have GridIdx) or mixed (if NoInterp axes remain GridIdx).

Used by TensorProductInterpolantND callable to accept GridIdx on ANY axis,
while routing NoInterp axes through the optimized pre-slice path.
"""
@generated function _convert_non_nointerp_grididx(
        methods::M, grids, query::Q,
    ) where {M, Q}
    N = fieldcount(Q)
    bounds_checks = Expr[]
    query_exprs = Expr[]
    for d in 1:N
        if fieldtype(Q, d) <: GridIdx
            push!(
                bounds_checks, :(
                    1 <= query[$d].idx <= length(grids[$d]) ||
                        _throw_grididx_oob($d, query[$d].idx, length(grids[$d]))
                )
            )
            if fieldtype(M, d) <: NoInterp
                push!(query_exprs, :(query[$d]))  # keep GridIdx for NoInterp
            else
                push!(query_exprs, :(@inbounds grids[$d][query[$d].idx]))  # convert to Real
            end
        else
            push!(query_exprs, :(query[$d]))
        end
    end
    return quote
        Base.@_inline_meta
        $(bounds_checks...)
        return ($(query_exprs...),)
    end
end

"""
    _check_nointerp_needs_grididx(methods, query)

Check at the all-Real callable entry point: if any axis has NoInterp,
the query must use GridIdx for that axis (not a plain Real value).
Generates a clear error message instead of the cryptic InBounds MethodError.
"""
@generated function _check_nointerp_needs_grididx(methods::M, ::Tuple{Vararg{Real, N}}) where {M, N}
    nointerp_dims = [d for d in 1:N if fieldtype(M, d) <: NoInterp]
    isempty(nointerp_dims) && return :(nothing)
    dims_tuple = Expr(:tuple, nointerp_dims...)
    return :(_throw_nointerp_needs_grididx($dims_tuple, $N))
end

# --- Dispatch helpers for resolved GridIdx queries ---
# After _convert_non_nointerp_grididx, the resolved query is either:
# (a) all-Real (no NoInterp axes, or NoInterp axes also got Real → will error)
# (b) mixed with GridIdx only on NoInterp axes → _eval_nointerp
#
# Julia dispatch resolves this at compile time (Tuple{Vararg{Real,N}} is more specific).

@inline function _eval_grididx_resolved(
        itp::TensorProductInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}}, deriv, search, hint,
    ) where {Tg, Tv, N}
    # All GridIdx converted to Real → delegate to standard all-Real callable
    return itp(query; deriv = deriv, search = search, hint = hint)
end

@inline function _eval_grididx_resolved(
        itp::TensorProductInterpolantND{Tg, Tv, N},
        query::Q, deriv, search, hint,
    ) where {Tg, Tv, N, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    # Still has GridIdx on NoInterp axes → resolve kwargs, validate, and eval
    ops = _resolve_deriv_nd(deriv, Val(N))
    search_tuple = _resolve_search_nd(search, Val(N))
    _validate_nointerp_grididx(itp.methods, query)
    return _eval_nointerp(itp, query, ops, search_tuple, hint)
end

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
    missing_nointerp_dims = [d for d in 1:N if fieldtype(M, d) <: NoInterp && !(fieldtype(Q, d) <: GridIdx)]
    checks = Expr[]
    # NoInterp axes missing GridIdx → clear error with usage hint
    isempty(missing_nointerp_dims) || push!(
        checks,
        :(_throw_nointerp_needs_grididx($(Expr(:tuple, missing_nointerp_dims...)), $N))
    )
    # Non-NoInterp axes with GridIdx → error (should have been resolved by _convert_non_nointerp_grididx)
    for d in 1:N
        if !(fieldtype(M, d) <: NoInterp) && fieldtype(Q, d) <: GridIdx
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
    # NOTE: guard runs AFTER domain validation so OOB queries still error.
    nointerp_deriv_checks = [:(deriv_order(ops_full[$d]) != 0) for d in nointerp_dims]
    nointerp_deriv_cond = if isempty(nointerp_deriv_checks)
        nothing
    elseif length(nointerp_deriv_checks) == 1
        nointerp_deriv_checks[1]
    else
        Expr(:||, nointerp_deriv_checks...)
    end
    # Use promoted type for zero (fixes Float32 data + Float64 query type mismatch)
    real_query_types = [fieldtype(Q, d) for d in real_dims]
    Tz_expr = isempty(real_query_types) ? Tv : :(promote_type($(real_query_types...), $Tg, $Tv))
    nointerp_deriv_guard = nointerp_deriv_cond === nothing ? :() :
        :(
            if $nointerp_deriv_cond
                return zero($Tz_expr)
        end
        )

    # Write GridIdx index into hint for NoInterp axes (consistent hint semantics)
    hint_nointerp_writes = [:(hint[$d][] = query[$d].idx) for d in nointerp_dims]
    hint_nointerp_update = isempty(hint_nointerp_writes) ? :() :
        :(
            if hint !== nothing
                $(hint_nointerp_writes...)
        end
        )

    if N_r == 0
        # All-NoInterp: pure table lookup (or zero if any deriv requested)
        if D <: HeteroPartials
            return quote
                Base.@_inline_meta
                $(bounds_checks...)
                $hint_nointerp_update
                $nointerp_deriv_guard
                @inbounds itp.data.partials[1, $(slice_args...)]
            end
        else
            return quote
                Base.@_inline_meta
                $(bounds_checks...)
                $hint_nointerp_update
                $nointerp_deriv_guard
                @inbounds itp.data[$(slice_args...)]
            end
        end
    end

    if D <: HeteroPartials
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
            $hint_nointerp_update
            # Slice partials at GridIdx positions
            p_sliced = @inbounds @view itp.data.partials[:, $(slice_args...)]

            # Reduced tuples (all types compile-time known)
            rq = ($(r_query...),)
            rg = ($(r_grids...),)
            rs = ($(r_spacings...),)
            re = ($(r_extraps...),)
            ro = ($(r_ops...),)
            rm = ($(r_methods...),)

            # Domain validation BEFORE deriv zero check
            _validate_nd_domain(rg, rq, re)
            oob = _try_fill_oob(rq, rg, re, ro, @inbounds p_sliced[1])
            oob !== nothing && return oob
            $nointerp_deriv_guard

            q_eval = _handle_all_extraps(rq, rg, re)
            rsrc = ($(r_searches...),)
            search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
            hint_r = hint === nothing ? nothing : ($([:(hint[$d]) for d in real_dims]...),)
            idxs, Ls, _ = _search_all_intervals(q_eval, rg, rs, search_r, hint_r)
            hs, inv_hs, dLs = _compute_all_local_params(q_eval, rs, idxs, Ls)
            return _eval_hetero_nd_cell(p_sliced, idxs, hs, inv_hs, dLs, ro, rm)
        end
    else
        # OnTheFly: slice data, delegate to reduced-dim _collapse_dims
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
            $hint_nointerp_update
            d_sliced = @inbounds @view itp.data[$(slice_args...)]
            rq = ($(r_query...),)
            rg = ($(r_grids...),)
            rm = ($(r_methods...),)
            re = ($(r_extraps...),)
            ro = ($(r_ops...),)

            # Domain validation + FillExtrap BEFORE deriv zero check
            _validate_nd_domain(rg, rq, re)
            oob = _try_fill_oob(rq, rg, re, ro, @inbounds d_sliced[1])
            oob !== nothing && return oob
            $nointerp_deriv_guard

            q_eval = _handle_all_extraps(rq, rg, re)
            rsrc = ($(r_searches...),)
            search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
            hint_r = hint === nothing ? nothing : ($([:(hint[$d]) for d in real_dims]...),)
            return _collapse_dims(d_sliced, rg, rm, re, q_eval, ro, search_r, hint_r)
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
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = AutoSearch(),
        hint = nothing,
    ) where {N, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    # Only reach here when at least one GridIdx (all-Real → more specific existing method)
    _validate_grididx_query_oneshot(query, data)
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method

    # Pre-slice data and filter all tuples to Real-only axes
    QT = typeof(query)
    data_r = _slice_grididx(data, query)
    grids_r = _filter_real_axes(grids, QT)
    query_r = _filter_real_axes(query, QT)
    methods_r = _filter_real_axes(method_tuple, QT)

    # Resolve per-axis kwargs to N-tuples
    deriv_t = deriv isa DerivOp ? ntuple(_ -> deriv, Val(N)) : deriv

    # All-GridIdx edge case: pure table lookup (or zero if any deriv requested)
    if grids_r === ()
        if _any_grididx_has_nonzero_deriv(query, deriv_t)
            Tg = _promote_grid_eltype(grids)
            Tg = Tg <: AbstractFloat ? Tg : Float64
            return zero(_output_eltype(eltype(data), Tg))
        end
        return data_r[]
    end

    # Domain validation on Real axes BEFORE deriv zero check
    # (so OOB on Real axes raises DomainError even when NoInterp axis has deriv)
    extrap_t = extrap isa AbstractExtrap ? ntuple(_ -> extrap, Val(N)) : extrap
    extrap_r = _filter_real_axes(extrap_t, QT)
    _validate_nd_domain(grids_r, query_r, extrap_r)

    # Check if any GridIdx axis has non-zero deriv → return zero (with promoted type)
    if _any_grididx_has_nonzero_deriv(query, deriv_t)
        Tg = _promote_grid_eltype(grids)
        Tg = Tg <: AbstractFloat ? Tg : Float64
        return zero(_output_eltype(eltype(data), Tg, typeof.(query_r)...))
    end

    # Filter remaining kwargs to Real axes
    search_t = search isa AbstractSearchPolicy ? ntuple(_ -> search, Val(N)) : search
    deriv_r = _filter_real_axes(deriv_t, QT)
    search_r = _filter_real_axes(search_t, QT)

    # Update hints for GridIdx axes, then filter to Real axes
    hint !== nothing && _update_grididx_hints!(hint, query)
    hint_r = hint === nothing ? nothing : _filter_real_axes(hint, QT)

    return interp(
        grids_r, data_r, query_r;
        method = methods_r, deriv = deriv_r, extrap = extrap_r,
        search = search_r, hint = hint_r
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
# The Union{Real,GridIdx} dispatch methods live in vector_calculus.jl at the
# AbstractInterpolantND level (gradient, hessian, laplacian). They call
# _gradient_with_grididx / _hessian_with_grididx / _laplacian_with_grididx.
#
# The generic fallback (AbstractInterpolantND) converts GridIdx → grid value and
# delegates to the standard all-Real method. TensorProductInterpolantND overrides
# to call _gradient_nointerp / _hessian_nointerp / _laplacian_nointerp, which use
# the optimized pre-slice path for NoInterp axes.
#
# No disambiguation methods needed: both the NTuple{N,Real} and the
# Union{Real,GridIdx} methods share the same arg1 type (AbstractInterpolantND),
# so Julia resolves by arg2 specificity alone (NTuple{N,Real} wins for all-Real).

# --- TensorProductInterpolantND overrides: redirect to _*_nointerp ---
@inline _gradient_with_grididx(itp::TensorProductInterpolantND, query, hint) =
    _gradient_nointerp(itp, query, hint)
@inline _hessian_with_grididx(itp::TensorProductInterpolantND, query, hint) =
    _hessian_nointerp(itp, query, hint)
@inline _laplacian_with_grididx(itp::TensorProductInterpolantND, query, hint) =
    _laplacian_nointerp(itp, query, hint)

"""
    _gradient_nointerp(itp::TensorProductInterpolantND, query, hint)

Gradient with NoInterp support. Returns N-tuple with zeros at NoInterp positions.
Uses the pre-slice strategy: slices data at GridIdx positions, evaluates on reduced dims.
"""
@generated function _gradient_nointerp(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    # Use promoted type for zero (handles Float32 data + Float64 query)
    real_query_types_g = [fieldtype(Q, d) for d in 1:N if !(d in nointerp_dims)]
    Tz_g = isempty(real_query_types_g) ? Tv : :(promote_type($(real_query_types_g...), $Tg, $Tv))

    grad_exprs = []
    for i in 1:N
        if i in nointerp_dims
            push!(grad_exprs, :(zero($Tz_g)))
        else
            ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
            push!(grad_exprs, :(_eval_nointerp(itp, query, $ops, itp.searches, hint)))
        end
    end

    return quote
        Base.@_inline_meta
        _validate_nointerp_grididx(itp.methods, query)
        return tuple($(grad_exprs...))
    end
end

"""
    _hessian_nointerp(itp::TensorProductInterpolantND, query, hint)

Hessian with NoInterp support. Returns N×N matrix with zero rows/columns at NoInterp positions.
"""
@generated function _hessian_nointerp(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q, hint,
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
    _hessian_nointerp!(H, itp::TensorProductInterpolantND, query, hint)

In-place Hessian with NoInterp support. Fills H with zeros at NoInterp positions.
"""
@generated function _hessian_nointerp!(
        H::AbstractMatrix,
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    stmts = Expr[]
    for i in 1:N
        if i in nointerp_dims
            # Zero out entire NoInterp row/column
            push!(stmts, :(H[$i, $i] = zero(eltype(H))))
            for j in 1:N
                i == j && continue
                push!(stmts, :(H[$i, $j] = zero(eltype(H))))
                push!(stmts, :(H[$j, $i] = zero(eltype(H))))
            end
            continue
        end
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

    return quote
        _validate_nointerp_grididx(itp.methods, query)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

# TensorProduct override: use in-place NoInterp-aware path
@inline _hessian_with_grididx!(H::AbstractMatrix, itp::TensorProductInterpolantND, query, hint) =
    _hessian_nointerp!(H, itp, query, hint)

# Generic fallback: convert GridIdx(k) → grids[d][k], delegate to all-Real hessian!
@generated function _hessian_with_grididx!(
        H::AbstractMatrix,
        itp::AbstractInterpolantND{Tg, Tv, N}, query::Q, hint,
    ) where {Tg, Tv, N, Q}
    grididx_dims = [d for d in 1:N if fieldtype(Q, d) <: GridIdx]
    bounds_checks = [
        :(
                1 <= query[$d].idx <= length(itp.grids[$d]) ||
                _throw_grididx_oob($d, query[$d].idx, length(itp.grids[$d]))
            )
            for d in grididx_dims
    ]
    real_query_exprs = [
        d in grididx_dims ? :(@inbounds itp.grids[$d][query[$d].idx]) : :(query[$d])
            for d in 1:N
    ]
    return quote
        $(bounds_checks...)
        return hessian!(H, itp, ($(real_query_exprs...),); hint = hint)
    end
end

"""
    _laplacian_nointerp(itp::TensorProductInterpolantND, query, hint)

Laplacian with NoInterp support. Sums ∂²f/∂xᵢ² only over interpolated axes.
"""
@generated function _laplacian_nointerp(
        itp::TensorProductInterpolantND{Tg, Tv, N, G, S, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, S, M, E, P, D, Q <: Tuple{Vararg{Union{Real, GridIdx}, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    terms = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
                :(_eval_nointerp(itp, query, $ops, itp.searches, hint))
            end for i in 1:N if !(i in nointerp_dims)
    ]

    if isempty(terms)
        # All-NoInterp: validate query before returning zero
        bounds_checks_lap = [
            :(
                    1 <= query[$d].idx <= size(itp.grids[$d], 1) ||
                    _throw_grididx_oob($d, query[$d].idx, size(itp.grids[$d], 1))
                )
                for d in nointerp_dims
        ]
        # Use promoted type for zero (handles Float32 data + Float64 query)
        real_query_types_lap = [fieldtype(Q, d) for d in 1:N if !(d in nointerp_dims)]
        Tz_lap = isempty(real_query_types_lap) ? Tv : :(promote_type($(real_query_types_lap...), $Tg, $Tv))
        return quote
            Base.@_inline_meta
            _validate_nointerp_grididx(itp.methods, query)
            $(bounds_checks_lap...)
            return zero($Tz_lap)
        end
    end
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
    _build_grididx_template(queries) -> template_tuple

Build a scalar template query point from SoA batch queries.
Vectors → first element (as Real), GridIdx → passed through.
Used for data slicing and type-based dispatch.
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

"""
    _has_grididx(queries) -> Bool

Compile-time check: returns true if any element of the tuple type is `GridIdx`.
Used by `interp!` to detect mixed batch queries and delegate to the GridIdx path.
"""
@generated function _has_grididx(::Type{Q}) where {Q <: Tuple}
    result = any(i -> fieldtype(Q, i) <: GridIdx, 1:fieldcount(Q))
    return :($result)
end

@generated function _filter_real_batch_queries(queries::Q) where {Q <: Tuple}
    kept = [i for i in 1:fieldcount(Q) if !(fieldtype(Q, i) <: GridIdx)]
    isempty(kept) && return :(())
    return :(tuple($([:(queries[$i]) for i in kept]...)))
end

"""
    _interp_batch_with_grididx!(output, grids, data, queries; method, ...)

Internal batch path for queries containing `GridIdx` elements.
Called by `interp!` when it detects `GridIdx` in the query tuple.

Pre-slices data once at GridIdx positions, filters all per-axis tuples
to Real-only axes, then delegates to existing `interp!` on reduced dims.
"""
function _interp_batch_with_grididx!(
        output::AbstractVector,
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        queries;
        method::Union{AbstractInterpMethod, Tuple{Vararg{AbstractInterpMethod, N}}},
        deriv::Union{DerivOp, Tuple{Vararg{DerivOp, N}}} = EvalValue(),
        extrap::Union{AbstractExtrap, Tuple{Vararg{AbstractExtrap, N}}} = NoExtrap(),
        search::Union{AbstractSearchPolicy, Tuple{Vararg{AbstractSearchPolicy, N}}} = AutoSearch(),
        hint = nothing,
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method

    # Filter queries to Real-only axes first (for length check)
    queries_r = _filter_real_batch_queries(queries)

    # All-GridIdx: no Real axes to batch over
    if queries_r === ()
        return output
    end

    # Empty batch fast path
    nq = _query_length(queries_r)
    if nq == 0
        return output
    end

    # Build a template query point for data slicing
    template = _build_grididx_template(queries)
    QT = typeof(template)

    # Pre-slice data at GridIdx positions (ONE-TIME)
    _validate_grididx_query_oneshot(template, data)
    data_r = _slice_grididx(data, template)

    # Filter to Real-only axes
    grids_r = _filter_real_axes(grids, QT)
    methods_r = _filter_real_axes(method_tuple, QT)

    # Resolve deriv/extrap to N-tuples and filter to Real axes
    deriv_t = deriv isa DerivOp ? ntuple(_ -> deriv, Val(N)) : deriv
    extrap_t = extrap isa AbstractExtrap ? ntuple(_ -> extrap, Val(N)) : extrap
    extrap_r = _filter_real_axes(extrap_t, QT)

    # Domain validation on Real axes BEFORE deriv zero check
    # (so OOB on Real axes raises DomainError even when NoInterp axis has deriv)
    _validate_nd_domain(grids_r, queries_r, extrap_r)

    # Check for GridIdx axis derivatives → zero fill
    if _any_grididx_has_nonzero_deriv(template, deriv_t)
        fill!(output, zero(eltype(output)))
        return output
    end

    # Resolve and filter remaining kwargs
    search_t = search isa AbstractSearchPolicy ? ntuple(_ -> search, Val(N)) : search
    deriv_r = _filter_real_axes(deriv_t, QT)
    search_r = _filter_real_axes(search_t, QT)
    # Update hints for GridIdx axes, then filter to Real axes
    hint !== nothing && _update_grididx_hints!(hint, template)
    hint_r = hint === nothing ? nothing : _filter_real_axes(hint, QT)

    return interp!(
        output, grids_r, data_r, queries_r;
        method = methods_r, deriv = deriv_r, extrap = extrap_r,
        search = search_r, hint = hint_r
    )
end
