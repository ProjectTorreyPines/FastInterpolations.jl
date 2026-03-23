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

    if N_r == 0
        # All-NoInterp: pure table lookup
        if D <: HeteroPartials
            return quote
                Base.@_inline_meta
                $(bounds_checks...)
                @inbounds itp.data.partials[1, $(slice_args...)]
            end
        else
            return quote
                Base.@_inline_meta
                $(bounds_checks...)
                @inbounds itp.data[$(slice_args...)]
            end
        end
    end

    if D <: HeteroPartials
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
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
            d_sliced = @inbounds @view itp.data[$(slice_args...)]
            rq = ($(r_query...),)
            rg = ($(r_grids...),)
            rm = ($(r_methods...),)
            re = ($(r_extraps...),)
            ro = ($(r_ops...),)
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
