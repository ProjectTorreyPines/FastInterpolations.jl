# ========================================
# Vector Calculus: PeriodicBC ND wrap-aware specialization
# ========================================
#
# Specializes `_gradient_generic` / `_hessian_generic` / `_laplacian_generic`
# (and their in-place variants) for `HeteroInterpolantND{...,<:Array}` so that
# PeriodicBC axes route through the wrap-aware cell construction instead of
# the persistent path's full-axis fallback in `_locate_cell` / `_build_windowed_cell`.
#
# Why this lives in a separate file (not vector_calculus.jl):
#   - The wrap-aware path is HeteroInterpolantND-specific (uses
#     `_axis_window_pooled`, `_strip_periodic_bc`, etc.); keeping the
#     specialization next to the type's other helpers is more cohesive than
#     polluting the type-agnostic `vector_calculus.jl` with hetero-only code.
#   - This file is included AFTER `vector_calculus.jl` in `FastInterpolations.jl`
#     so the specialized methods shadow the generic ones for the relevant type.
#
# Design (per-op):
#   1. `_<op>_generic[!]` (specialized for HeteroInterpolantND{<:Array}):
#      Standard setup (resolve grididx, search policies, hints, fill-OOB
#      check). When `_has_any_periodic_method(itp.methods)` is true, route to
#      `_wrap_aware_eval` with the op-specific collapser; otherwise fall
#      through to the same body as the generic method.
#   2. `_wrap_aware_eval(collapser, ...)`: single shared `@with_pool` shim
#      that builds the wrap-aware cell ONCE via `_build_wrap_aware_cell_components`
#      (in hetero_eval.jl) and then calls `collapser` to expand the
#      per-derivative `_collapse_dims` calls. Type-stable on `collapser::F`.
#   3. `_<op>_collapse_all[!]`: generated function that splices the per-axis
#      (or per-pair, for hessian) `_collapse_dims` calls. Pure body (no
#      closures or comprehensions in the returned `quote`) so Julia's
#      `@generated` purity check accepts it.
#
# Closures over `pool` are isolated to `_wrap_aware_eval` (a regular function),
# never inside generated bodies — that's why we can't fold the shim into the
# generated dispatch.

# ────────────────────────────────────────
# Shared @with_pool shim — used by all 5 dispatch entries below.
# ────────────────────────────────────────

@inline @with_pool pool function _wrap_aware_eval(
        collapser::F,
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query_r::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        hints::Tuple{Vararg{Base.RefValue{Int}, N}},
    ) where {F, Tg, Tv, N, G, S, M, E, P}
    q_eval = _handle_all_extraps(query_r, itp.grids, itp.extraps)
    windows, grids_local, methods_inner, extraps_inner =
        _build_wrap_aware_cell_components(pool, itp, q_eval, policies, hints)
    Tr = _output_eltype(Tv, Tg, typeof.(q_eval)...)
    return collapser(itp, q_eval, policies, windows, grids_local, methods_inner, extraps_inner, Tr)
end

# ────────────────────────────────────────
# GRADIENT
# ────────────────────────────────────────

@generated function _gradient_collapse_all(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        windows, grids_local, methods_inner, extraps_inner,
        ::Type{Tr},
    ) where {Tg, Tv, N, G, S, M, E, P, Tr}
    calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(_collapse_dims(Tr, itp.data, grids_local, methods_inner, extraps_inner, q_eval, $ops, policies, nothing, windows))
            end for i in 1:N
    ]
    return :(tuple($(calls...)))
end

@generated function _gradient_collapse_all!(
        G::AbstractVector,
        itp::HeteroInterpolantND{Tg, Tv, N, G_, S, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        windows, grids_local, methods_inner, extraps_inner,
        ::Type{Tr},
    ) where {Tg, Tv, N, G_, S, M, E, P, Tr}
    stmts = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(G[$i] = _collapse_dims(Tr, itp.data, grids_local, methods_inner, extraps_inner, q_eval, $ops, policies, nothing, windows))
            end for i in 1:N
    ]
    return quote
        @inbounds begin
            $(stmts...)
        end
        return G
    end
end

@generated function _gradient_generic(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        hint,
    ) where {Tg, Tv, N, G, S, M, E, P}
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    zero_tuple = [:(0 * zref) for _ in 1:N]
    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _zero_ref(itp)
            return tuple($(zero_tuple...))
        end
        if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query_r))
            return _wrap_aware_eval(_gradient_collapse_all, itp, query_r, policies, hints)
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        return tuple($(deriv_calls...))
    end
end

@generated function _gradient_generic!(
        G::AbstractVector,
        itp::HeteroInterpolantND{Tg, Tv, N, G_, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        hint,
    ) where {Tg, Tv, N, G_, S, M, E, P}
    stmts = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{1}() : DerivOp{0}(), N)
                :(G[$i] = _eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        @boundscheck length(G) >= $N || throw(
            DimensionMismatch(
                "gradient output vector must have at least $($N) elements, got $(length(G))"
            )
        )
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            zref = _zero_ref(itp)
            @inbounds for i in 1:$N
                G[i] = 0 * zref
            end
            return G
        end
        if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query_r))
            _wrap_aware_eval(Base.Fix1(_gradient_collapse_all!, G), itp, query_r, policies, hints)
            return G
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        @inbounds begin
            $(stmts...)
        end
        return G
    end
end

# ────────────────────────────────────────
# HESSIAN
# ────────────────────────────────────────

@generated function _hessian_collapse_all!(
        H::AbstractMatrix,
        itp::HeteroInterpolantND{Tg, Tv, N, G_, S, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        windows, grids_local, methods_inner, extraps_inner,
        ::Type{Tr},
    ) where {Tg, Tv, N, G_, S, M, E, P, Tr}
    stmts = Expr[]
    for i in 1:N
        ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
        push!(stmts, :(H[$i, $i] = _collapse_dims(Tr, itp.data, grids_local, methods_inner, extraps_inner, q_eval, $ops, policies, nothing, windows)))
    end
    for i in 1:N, j in (i + 1):N
        ops = ntuple(k -> (k == i || k == j) ? DerivOp{1}() : DerivOp{0}(), N)
        push!(
            stmts, quote
                val = _collapse_dims(Tr, itp.data, grids_local, methods_inner, extraps_inner, q_eval, $ops, policies, nothing, windows)
                H[$i, $j] = val
                H[$j, $i] = val
            end
        )
    end
    return quote
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

@generated function _hessian_generic(
        itp::HeteroInterpolantND{Tg, Tv, N, G_, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        hint,
    ) where {Tg, Tv, N, G_, S, M, E, P}
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
        query_r = map(_resolve_grididx, query, itp.grids)
        Tq = promote_type(eltype(map(float, query_r)), $Tg, $Tv)
        H = Matrix{Tq}(undef, $N, $N)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            fill!(H, zero(Tq))
            return H
        end
        if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query_r))
            _wrap_aware_eval(Base.Fix1(_hessian_collapse_all!, H), itp, query_r, policies, hints)
            return H
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

@generated function _hessian_generic!(
        H::AbstractMatrix,
        itp::HeteroInterpolantND{Tg, Tv, N, G_, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        hint,
    ) where {Tg, Tv, N, G_, S, M, E, P}
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
        query_r = map(_resolve_grididx, query, itp.grids)
        @boundscheck size(H) == ($N, $N) || throw(
            DimensionMismatch(
                "Hessian output matrix must be $($N)×$($N), got $(size(H))"
            )
        )
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            fill!(H, zero(eltype(H)))
            return H
        end
        if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query_r))
            _wrap_aware_eval(Base.Fix1(_hessian_collapse_all!, H), itp, query_r, policies, hints)
            return H
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        @inbounds begin
            $(stmts...)
        end
        return H
    end
end

# ────────────────────────────────────────
# LAPLACIAN
# ────────────────────────────────────────

@generated function _laplacian_collapse_all(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        q_eval::Tuple{Vararg{Real, N}},
        policies::NTuple{N, AbstractSearchPolicy},
        windows, grids_local, methods_inner, extraps_inner,
        ::Type{Tr},
    ) where {Tg, Tv, N, G, S, M, E, P, Tr}
    calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
                :(_collapse_dims(Tr, itp.data, grids_local, methods_inner, extraps_inner, q_eval, $ops, policies, nothing, windows))
            end for i in 1:N
    ]
    return :(+($(calls...)))
end

@generated function _laplacian_generic(
        itp::HeteroInterpolantND{Tg, Tv, N, G, S, M, E, P, <:Array},
        query::Tuple{Vararg{Real, N}},
        hint,
    ) where {Tg, Tv, N, G, S, M, E, P}
    deriv_calls = [
        begin
                ops = ntuple(j -> j == i ? DerivOp{2}() : DerivOp{0}(), N)
                :(_eval_at_cell(itp, cell, $ops))
            end for i in 1:N
    ]
    return quote
        query_r = map(_resolve_grididx, query, itp.grids)
        policies = _resolve_search_nd(itp.searches, Val($N))
        hints = _ensure_hint_nd(hint, Val($N))
        mono = _scalar_mono(hint, Val($N))
        if _is_fill_oob(query_r, itp.grids, itp.extraps)
            return 0 * _zero_ref(itp)
        end
        if _has_any_periodic_method(itp.methods) && !_has_grididx(typeof(query_r))
            return _wrap_aware_eval(_laplacian_collapse_all, itp, query_r, policies, hints)
        end
        cell = _locate_cell(itp, query_r, policies, hints, mono)
        return +($(deriv_calls...))
    end
end
