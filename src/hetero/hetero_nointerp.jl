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
# Include order: after hetero_oneshot.jl (needs interp, _eval_hetero_nd_cell, etc.)

# ========================================
# Compile-Time Traits
# ========================================

# _has_nointerp_method is defined in hetero_interpolant.jl (included before this file)

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
# GridIdx → NoInterp Auto-Promotion (one-shot only)
# ========================================
# When all derivs are EvalValue, GridIdx axes need no interpolation —
# pre-slicing at the grid point gives the exact value. Replacing the
# method with NoInterp() triggers dimension reduction, avoiding the
# full ND build (e.g., 3D cubic spline build → 1D: ~5000x speedup).

"""
    _all_eval_value(deriv) -> Bool

Compile-time check: true if `deriv` is `EvalValue()` (scalar) or a tuple of all `EvalValue()`.
Used to guard GridIdx → NoInterp auto-promotion (safe only when no derivatives requested).
"""
_all_eval_value(::EvalValue) = true
_all_eval_value(::DerivOp) = false
@generated function _all_eval_value(::D) where {D <: Tuple}
    return all(d -> fieldtype(D, d) <: EvalValue, 1:fieldcount(D)) ? :(true) : :(false)
end

"""
    _promote_grididx_to_nointerp(methods, query) -> promoted_methods

Replace methods at GridIdx query positions with `NoInterp()`.
Compile-time: inspects query tuple field types, zero runtime cost.
Returns `methods` unchanged if no GridIdx is present.
"""
@generated function _promote_grididx_to_nointerp(
        methods::M, query::Q,
    ) where {M <: Tuple, Q <: Tuple}
    N = fieldcount(Q)
    any_grididx = any(d -> fieldtype(Q, d) <: GridIdx, 1:N)
    !any_grididx && return :(methods)
    exprs = [fieldtype(Q, d) <: GridIdx ? :(NoInterp()) : :(methods[$d]) for d in 1:N]
    return :(tuple($(exprs...)))
end

# ========================================
# Validation Helpers
# ========================================

"""
    _any_nointerp_grididx_has_nonzero_deriv(query, methods, deriv_tuple) -> Bool

Compile-time check: returns true if any **NoInterp** GridIdx axis has a non-zero DerivOp.
Only NoInterp axes have zero derivatives by definition; non-NoInterp GridIdx axes
hold valid grid coordinates and support normal derivatives.
"""
@generated function _any_nointerp_grididx_has_nonzero_deriv(
        query::Q, methods::M, deriv_tuple::Tuple{Vararg{DerivOp, N}},
    ) where {Q, M, N}
    checks = [
        :(deriv_order(deriv_tuple[$d]) != 0)
            for d in 1:N if fieldtype(Q, d) <: GridIdx && fieldtype(M, d) <: NoInterp
    ]
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
    # NoInterp axes missing GridIdx → clear error with usage hint
    # Non-NoInterp axes with GridIdx are fine (resolved GridIdx flows through search short-circuit)
    missing_nointerp_dims = [d for d in 1:N if fieldtype(M, d) <: NoInterp && !(fieldtype(Q, d) <: GridIdx)]
    isempty(missing_nointerp_dims) && return :(nothing)
    return :(_throw_nointerp_needs_grididx($(Expr(:tuple, missing_nointerp_dims...)), $N))
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
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D},
        query::Q, ops_full, search, hint,
    ) where {Tg, Tv, N, G, M, E, P, D, Q}
    nointerp_dims = [d for d in 1:N if fieldtype(M, d) <: NoInterp]
    real_dims = [d for d in 1:N if !(fieldtype(M, d) <: NoInterp)]
    N_r = length(real_dims)

    # Build slice expressions: `:` for Real axes, `query[d].idx` for NoInterp
    slice_args = [d in nointerp_dims ? :(query[$d].idx) : :(:) for d in 1:N]

    # Bounds checks for GridIdx on NoInterp axes (before @inbounds slicing)
    # Use grid length as the canonical size (works for both _HeteroPartials and raw Array)
    bounds_checks = [
        :(
                1 <= query[$d].idx <= size(itp.grids[$d], 1) ||
                _throw_grididx_oob($d, query[$d].idx, size(itp.grids[$d], 1))
            )
            for d in nointerp_dims
    ]

    # Build reduced tuple expressions
    r_grids = [:(itp.grids[$d]) for d in real_dims]
    r_methods = [:(itp.methods[$d]) for d in real_dims]
    r_extraps = [:(itp.extraps[$d]) for d in real_dims]
    r_searches = [:(search[$d]) for d in real_dims]
    r_query = [:(query[$d]) for d in real_dims]
    r_ops = [:(ops_full[$d]) for d in real_dims]

    # Per-real-axis full windows (static @generated form — no runtime closure).
    # Real axis k of d_sliced corresponds to enumerate index, so size(d_sliced, k).
    r_full_windows = [:(Base.OneTo(size(d_sliced, $k))) for k in 1:N_r]

    # Per-real-axis cell-local window expressions (Phase 5b). Built statically — each
    # expression closes over the runtime `idxs_r` (computed by `_search_all_intervals`).
    # Position `k` in idxs_r corresponds to real axis at original dim `d`.
    r_axis_window_exprs = [
        :(_axis_window(itp.methods[$d], idxs_r[$k], length(itp.grids[$d])))
            for (k, d) in enumerate(real_dims)
    ]
    # Compile-time check: any windowable axis remaining in the filtered methods?
    # `_eval_nointerp` is called on a persistent `HeteroInterpolantND`. Spacings
    # are derived from grid wrappers on demand (no separate field), so the
    # asymmetric persistent-path rule (use `_has_any_windowable_method`, not the
    # narrower `_has_any_local_method`) still applies — see hetero_window.jl for
    # rationale.
    # Mirror the `_is_windowable_method` set: PCHIP/Cardinal/Akima/Linear/Constant.
    rm_has_local = any(
        M -> M <: Union{PchipInterp, CardinalInterp, AkimaInterp, LinearInterp, ConstantInterp},
        [fieldtype(M, d) for d in real_dims]
    )

    # Check if any NoInterp axis has a non-zero DerivOp → cell-local zero
    # (discrete axes have zero derivatives by definition). Guard runs AFTER
    # domain validation so OOB queries still error.
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

    # Write GridIdx index into hint for NoInterp axes (consistent hint semantics)
    hint_nointerp_writes = [:(hint[$d][] = query[$d].idx) for d in nointerp_dims]
    hint_nointerp_update = isempty(hint_nointerp_writes) ? :() :
        :(
            if hint !== nothing
                $(hint_nointerp_writes...)
        end
        )

    if N_r == 0
        # All-NoInterp: pure table lookup; deriv on NoInterp axis → multiply
        # cell-local data by `zero(Tz)` so NaN in the queried cell propagates
        # while still returning a type-promoted zero (mirrors `_constant_nd_evaluate`'s
        # `kernel * 0` dispatch).
        data_expr = D <: _HeteroPartials ?
            :(itp.data.partials[1, $(slice_args...)]) :
            :(itp.data[$(slice_args...)])
        nointerp_deriv_guard = nointerp_deriv_cond === nothing ? :() :
            :(
                if $nointerp_deriv_cond
                    return @inbounds $data_expr * zero($Tz_expr)
            end
            )
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
            $hint_nointerp_update
            $nointerp_deriv_guard
            @inbounds $data_expr
        end
    end

    # Mixed case (N_r > 0): run the Real-axis interp normally, then multiply
    # by `0` if any NoInterp axis has a non-zero deriv. `result * 0` preserves
    # the carrier (Tg/Tq/Tv) from the cell-local Real-axis computation —
    # mirrors `_constant_nd_evaluate`'s `kernel * 0` dispatch.
    deriv_zero_wrap = nointerp_deriv_cond === nothing ?
        :(return result) :
        :(return $nointerp_deriv_cond ? result * 0 : result)

    # OOB short-circuit wrap: `_try_fill_oob` returns the cell-local fill_value
    # under the option-B contract (see `_fill_extrap_result` in `nd_utils.jl`).
    # If a NoInterp axis carries a non-zero deriv, multiply by `0` so the
    # mathematical "NoInterp deriv ⇒ 0" rule applies — `fill_value * 0` keeps
    # NaN propagation (NaN × 0 = NaN), finite × 0 = 0.
    oob_wrap = nointerp_deriv_cond === nothing ?
        :(oob !== nothing && return oob) :
        :(
            if oob !== nothing
                return $nointerp_deriv_cond ? oob * 0 : oob
        end
        )

    if D <: _HeteroPartials
        return quote
            Base.@_inline_meta
            $(bounds_checks...)
            $hint_nointerp_update
            # Slice partials at GridIdx positions
            p_sliced = @inbounds @view itp.data.partials[:, $(slice_args...)]

            # Reduced tuples (all types compile-time known)
            rq = ($(r_query...),)
            rg = ($(r_grids...),)
            re = ($(r_extraps...),)
            ro = ($(r_ops...),)
            rm = ($(r_methods...),)

            _validate_nd_domain(rg, rq, re)
            oob = _try_fill_oob(rq, rg, re, ro, @inbounds p_sliced[1])
            $oob_wrap

            q_eval = _handle_all_extraps(rq, rg, re)
            rsrc = ($(r_searches...),)
            search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
            hint_r = hint === nothing ? nothing : ($([:(hint[$d]) for d in real_dims]...),)
            idxs, Ls, _ = _search_all_intervals(q_eval, rg, search_r, hint_r)
            hs, inv_hs, dLs = _compute_all_local_params(q_eval, rg, idxs, Ls)
            result = _eval_hetero_nd_cell(p_sliced, idxs, hs, inv_hs, dLs, ro, rm)
            $deriv_zero_wrap
        end
    else
        # OnTheFly: slice data, delegate to reduced-dim _collapse_dims.
        # When at least one real axis is local-Hermite, pre-search the cell once in
        # the real-axis-only coordinate system, build per-axis cell-local windows,
        # slice further, and call the kernel with relative windows. Pure global-solve
        # real-axis tuples skip the pre-search and fall through to the full-windows path.
        if rm_has_local
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

                _validate_nd_domain(rg, rq, re)
                oob = _try_fill_oob(rq, rg, re, ro, @inbounds d_sliced[1])
                $oob_wrap

                q_eval = _handle_all_extraps(rq, rg, re)
                rsrc = ($(r_searches...),)
                search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
                hint_r = hint === nothing ? nothing : ($([:(hint[$d]) for d in real_dims]...),)
                Tr = _promote_eltype(eltype(d_sliced), $Tg, typeof.(q_eval)...)

                # Pre-search: mutates user hints (real-axis subset) to absolute indices.
                idxs_r, _, _ = _search_all_intervals(q_eval, rg, search_r, hint_r)
                # Per-real-axis cell-local windows. Static tuple literal — no closure.
                windows_r = ($(r_axis_window_exprs...),)
                # Slice d_sliced (already NoInterp-sliced) further to the cell-local stencil.
                d_local = view(d_sliced, windows_r...)
                rg_local = map(view, rg, windows_r)
                rel_windows_r = map(Base.OneTo ∘ length, windows_r)
                result = _collapse_dims(Tr, d_local, rg_local, rm, re, q_eval, ro, search_r, nothing, rel_windows_r)
                $deriv_zero_wrap
            end
        end

        # Pure global-solve real-axis tuple: full-window path.
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

            _validate_nd_domain(rg, rq, re)
            oob = _try_fill_oob(rq, rg, re, ro, @inbounds d_sliced[1])
            $oob_wrap

            q_eval = _handle_all_extraps(rq, rg, re)
            rsrc = ($(r_searches...),)
            search_r = _resolve_search_nd(rsrc, Val($N_r), rq)
            hint_r = hint === nothing ? nothing : ($([:(hint[$d]) for d in real_dims]...),)
            Tr = _promote_eltype(eltype(d_sliced), $Tg, typeof.(q_eval)...)
            full_windows_r = ($(r_full_windows...),)
            result = _collapse_dims(Tr, d_sliced, rg, rm, re, q_eval, ro, search_r, hint_r, full_windows_r)
            $deriv_zero_wrap
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
# Internal: one-shot with NoInterp/GridIdx pre-slicing.
# Called from unified interp() when _has_nointerp_method is true.
# Query is already resolved (_resolve_grididx applied at entry).
function _interp_nointerp_oneshot(
        grids::NTuple{N, AbstractVector},
        data::AbstractArray{<:Any, N},
        query::Q,
        method_tuple::Tuple{Vararg{AbstractInterpMethod, N}},
        deriv,
        extrap,
        search,
        hint,
    ) where {N, Q <: Tuple{Vararg{Real, N}}}
    _validate_grididx_query_oneshot(query, data)

    # Pre-slice data and filter all tuples to Real-only axes
    QT = typeof(query)
    data_r = _slice_grididx(data, query)
    grids_r = _filter_real_axes(grids, QT)
    query_r = _filter_real_axes(query, QT)
    methods_r = _filter_real_axes(method_tuple, QT)

    # Resolve per-axis kwargs to N-tuples
    deriv_t = deriv isa DerivOp ? ntuple(_ -> deriv, Val(N)) : deriv

    # All-GridIdx edge case: pure table lookup (deriv on NoInterp axis → zero,
    # but cell-local — `data_r[] * zero(Tz)` propagates NaN at the queried
    # cell while still returning a type-promoted zero).
    if grids_r === ()
        if _any_nointerp_grididx_has_nonzero_deriv(query, method_tuple, deriv_t)
            Tg = float(_promote_grid_eltype(grids))
            return data_r[] * zero(_promote_eltype(eltype(data), Tg))
        end
        return data_r[]
    end

    # Domain validation on Real axes (so OOB on Real axes raises DomainError
    # regardless of NoInterp deriv).
    extrap_t = extrap isa AbstractExtrap ? ntuple(_ -> extrap, Val(N)) : extrap
    extrap_r = _filter_real_axes(extrap_t, QT)
    _validate_nd_domain(grids_r, query_r, extrap_r)

    # Filter remaining kwargs to Real axes
    search_t = search isa AbstractSearchPolicy ? ntuple(_ -> search, Val(N)) : search
    deriv_r = _filter_real_axes(deriv_t, QT)
    search_r = _filter_real_axes(search_t, QT)

    # Update hints for GridIdx axes, then filter to Real axes
    hint !== nothing && _update_grididx_hints!(hint, query)
    hint_r = hint === nothing ? nothing : _filter_real_axes(hint, QT)

    # Run Real-axis interp normally. `result * 0` when a NoInterp axis carries
    # non-zero deriv: applies the "NoInterp deriv ⇒ 0" rule while preserving
    # the Real-axis carrier. `result` already reflects the OOB cell-local
    # contract from `_eval_extrapolation` / `_fill_extrap_result` (fill_value-
    # as-data for FillExtrap, boundary y for ClampExtrap), so NaN propagation
    # — whether from cell data or from a NaN fill_value — flows through `* 0`.
    result = interp(
        grids_r, data_r, query_r;
        method = methods_r, deriv = deriv_r, extrap = extrap_r,
        search = search_r, hint = hint_r
    )
    return _any_nointerp_grididx_has_nonzero_deriv(query, method_tuple, deriv_t) ?
        result * 0 : result
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
# GridIdx <: Real: no separate Union dispatch needed.
# HeteroInterpolantND overrides vector_calculus.jl methods (more specific arg1 type)
# and delegates to _*_nointerp for the optimized pre-slice path.

# --- HeteroInterpolantND vector calculus: NoInterp-aware overrides ---
# Only override when NoInterp is present (compile-time _has_nointerp_method check).
# Without NoInterp, delegate to _*_generic helpers in vector_calculus.jl which use
# the optimal locate-once path (_locate_cell once → _eval_at_cell per component).
@inline function gradient(
        itp::HeteroInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint = nothing,
    ) where {Tg, Tv, N}
    _has_nointerp_method(typeof(itp.methods)) || return _gradient_generic(itp, query, hint)
    resolved = map(_resolve_grididx, query, itp.grids)
    return _gradient_nointerp(itp, resolved, hint)
end

@inline function hessian(
        itp::HeteroInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint = nothing,
    ) where {Tg, Tv, N}
    _has_nointerp_method(typeof(itp.methods)) || return _hessian_generic(itp, query, hint)
    resolved = map(_resolve_grididx, query, itp.grids)
    return _hessian_nointerp(itp, resolved, hint)
end

@inline function hessian!(
        H::AbstractMatrix,
        itp::HeteroInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint = nothing,
    ) where {Tg, Tv, N}
    _has_nointerp_method(typeof(itp.methods)) || return _hessian_generic!(H, itp, query, hint)
    resolved = map(_resolve_grididx, query, itp.grids)
    return _hessian_nointerp!(H, itp, resolved, hint)
end

@inline function laplacian(
        itp::HeteroInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint = nothing,
    ) where {Tg, Tv, N}
    _has_nointerp_method(typeof(itp.methods)) || return _laplacian_generic(itp, query, hint)
    resolved = map(_resolve_grididx, query, itp.grids)
    return _laplacian_nointerp(itp, resolved, hint)
end

@inline function value_gradient(
        itp::HeteroInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint = nothing,
    ) where {Tg, Tv, N}
    _has_nointerp_method(typeof(itp.methods)) || return _value_gradient_generic(itp, query, hint)
    resolved = map(_resolve_grididx, query, itp.grids)
    val = itp(resolved; hint = hint)
    g = _gradient_nointerp(itp, resolved, hint)
    return (val, g)
end

@inline function gradient!(
        G::AbstractVector,
        itp::HeteroInterpolantND{Tg, Tv, N},
        query::Tuple{Vararg{Real, N}};
        hint = nothing,
    ) where {Tg, Tv, N}
    _has_nointerp_method(typeof(itp.methods)) || return _gradient_generic!(G, itp, query, hint)
    @boundscheck length(G) >= N || throw(
        DimensionMismatch(
            "gradient output vector must have at least $N elements, got $(length(G))"
        )
    )
    resolved = map(_resolve_grididx, query, itp.grids)
    g = _gradient_nointerp(itp, resolved, hint)
    @inbounds for i in 1:N
        G[i] = g[i]
    end
    return G
end

"""
    _gradient_nointerp(itp::HeteroInterpolantND, query, hint)

Gradient with NoInterp support. Returns N-tuple with zeros at NoInterp positions.
Uses the pre-slice strategy: slices data at GridIdx positions, evaluates on reduced dims.
"""
@generated function _gradient_nointerp(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
    nointerp_dims = Set(d for d in 1:N if fieldtype(M, d) <: NoInterp)

    # Use promoted type for zero (handles Float32 data + Float64 query)
    real_query_types_g = [fieldtype(Q, d) for d in 1:N if !(d in nointerp_dims)]
    Tz_g = isempty(real_query_types_g) ? Tv : :(promote_type($(real_query_types_g...), $Tg, $Tv))

    grad_exprs = []
    all_nointerp = length(nointerp_dims) == N
    for i in 1:N
        if i in nointerp_dims
            push!(grad_exprs, :(zero($Tz_g)))
        else
            ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
            push!(grad_exprs, :(_eval_nointerp(itp, query, $ops, itp.searches, hint)))
        end
    end

    if all_nointerp
        # All-NoInterp: _eval_nointerp is never called, so validate GridIdx bounds explicitly
        bounds_checks = [
            :(
                    1 <= query[$d].idx <= size(itp.grids[$d], 1) ||
                    _throw_grididx_oob($d, query[$d].idx, size(itp.grids[$d], 1))
                )
                for d in nointerp_dims
        ]
        return quote
            Base.@_inline_meta
            _validate_nointerp_grididx(itp.methods, query)
            $(bounds_checks...)
            return tuple($(grad_exprs...))
        end
    end

    return quote
        Base.@_inline_meta
        _validate_nointerp_grididx(itp.methods, query)
        return tuple($(grad_exprs...))
    end
end

"""
    _hessian_nointerp(itp::HeteroInterpolantND, query, hint)

Hessian with NoInterp support. Returns N×N matrix with zero rows/columns at NoInterp positions.
"""
@generated function _hessian_nointerp(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
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
    all_nointerp = length(nointerp_dims) == N

    if all_nointerp
        # All-NoInterp: _eval_nointerp is never called, so validate GridIdx bounds explicitly
        bounds_checks = [
            :(
                    1 <= query[$d].idx <= size(itp.grids[$d], 1) ||
                    _throw_grididx_oob($d, query[$d].idx, size(itp.grids[$d], 1))
                )
                for d in nointerp_dims
        ]
        return quote
            _validate_nointerp_grididx(itp.methods, query)
            $(bounds_checks...)
            Tq = $Tq_expr
            return zeros(Tq, $N, $N)
        end
    end

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
    _hessian_nointerp!(H, itp::HeteroInterpolantND, query, hint)

In-place Hessian with NoInterp support. Fills H with zeros at NoInterp positions.
"""
@generated function _hessian_nointerp!(
        H::AbstractMatrix,
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
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

    all_nointerp = length(nointerp_dims) == N
    if all_nointerp
        # All-NoInterp: _eval_nointerp is never called, so validate GridIdx bounds explicitly
        bounds_checks = [
            :(
                    1 <= query[$d].idx <= size(itp.grids[$d], 1) ||
                    _throw_grididx_oob($d, query[$d].idx, size(itp.grids[$d], 1))
                )
                for d in nointerp_dims
        ]
        return quote
            @boundscheck size(H) == ($N, $N) || throw(
                DimensionMismatch(
                    "Hessian output matrix must be $($N)×$($N), got $(size(H))"
                )
            )
            _validate_nointerp_grididx(itp.methods, query)
            $(bounds_checks...)
            fill!(H, zero(eltype(H)))
            return H
        end
    end

    return quote
        @boundscheck size(H) == ($N, $N) || throw(
            DimensionMismatch(
                "Hessian output matrix must be $($N)×$($N), got $(size(H))"
            )
        )
        _validate_nointerp_grididx(itp.methods, query)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end


"""
    _laplacian_nointerp(itp::HeteroInterpolantND, query, hint)

Laplacian with NoInterp support. Sums ∂²f/∂xᵢ² only over interpolated axes.
"""
@generated function _laplacian_nointerp(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D},
        query::Q, hint,
    ) where {Tg, Tv, N, G, M, E, P, D, Q <: Tuple{Vararg{Real, N}}}
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
    _expand_grididx_queries(queries, grids, methods, nq) -> expanded queries

Convert non-NoInterp GridIdx elements to Real vectors (repeated grid values).
NoInterp GridIdx elements are kept as-is. Used when non-NoInterp GridIdx axes
have nonzero derivatives — pre-slice can't compute those derivatives, so we
expand to standard batch format and let the normal path handle them.
"""
@generated function _expand_grididx_queries(queries::Q, grids, methods::M, nq) where {Q <: Tuple, M <: Tuple}
    N = fieldcount(Q)
    exprs = [
        if fieldtype(Q, d) <: GridIdx && !(fieldtype(M, d) <: NoInterp)
                :(fill(grids[$d][queries[$d].idx], nq))
        else
                :(queries[$d])
        end
            for d in 1:N
    ]
    return :(tuple($(exprs...)))
end

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
        coeffs::AbstractCoeffStrategy = AutoCoeffs(),
    ) where {N}
    method_tuple = method isa AbstractInterpMethod ? ntuple(_ -> method, Val(N)) : method

    # Quick check: filter Real-only axes for batch length
    queries_r_initial = _filter_real_batch_queries(queries)

    # All-GridIdx: no Real axes to batch over → early return
    if queries_r_initial === ()
        return output
    end

    # Empty batch fast path
    nq = _query_length(queries_r_initial)
    if nq == 0
        return output
    end

    # Mirror scalar API's GridIdx auto-promotion via `_promote_grididx_to_nointerp`.
    # When all requested derivs are EvalValue(), a GridIdx on a non-NoInterp axis
    # is equivalent to slicing the data at that index and dropping the axis — so
    # we can promote the method at that position to NoInterp(), keeping the axis
    # on the pre-slice path below. Without this, the fallback path expands the
    # GridIdx to a constant Real vector and recurses with the original method
    # tuple, which would (incorrectly) reject `coeffs=PreCompute()` for any
    # local-Hermite axis: e.g. `(CubicInterp, PchipInterp)` + `(qx, GridIdx(k))`
    # + `coeffs=PreCompute()` used to throw "PreCompute not supported for Pchip
    # in ND" even though the reduced 1-D problem is pure Cubic.
    if _all_eval_value(deriv)
        method_tuple = _promote_grididx_to_nointerp(method_tuple, queries)
    end

    # Non-NoInterp GridIdx: convert to Real vectors before pre-slicing.
    # Pre-slice only makes sense for NoInterp axes (discrete, zero derivative).
    # Non-NoInterp GridIdx axes hold valid coordinates that need full interpolation.
    # After the promotion above, any GridIdx axis with EvalValue deriv is already
    # NoInterp and will be kept as-is by `_expand_grididx_queries`; only GridIdx
    # axes with nonzero derivatives (which the pre-slice path can't handle) fall
    # through to the expansion.
    queries = _expand_grididx_queries(queries, grids, method_tuple, nq)

    # After expansion, if no GridIdx remains, delegate to standard batch path
    if !_has_grididx(typeof(queries))
        return interp!(
            output, grids, data, queries;
            method = method, deriv = deriv, extrap = extrap,
            search = search, hint = hint, coeffs = coeffs,
        )
    end

    # From here, only NoInterp GridIdx axes remain.
    queries_r = _filter_real_batch_queries(queries)

    # Build a template query point for data slicing
    template = _build_grididx_template(queries)
    QT = typeof(template)

    # Pre-slice data at NoInterp GridIdx positions (ONE-TIME)
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
    _validate_nd_domain(grids_r, queries_r, extrap_r)

    # NoInterp GridIdx with nonzero deriv → zero fill (discrete axes have no derivative)
    if _any_nointerp_grididx_has_nonzero_deriv(template, method_tuple, deriv_t)
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
        search = search_r, hint = hint_r, coeffs = coeffs,
    )
end
