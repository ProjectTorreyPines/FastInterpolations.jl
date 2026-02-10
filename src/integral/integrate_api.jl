# ── 1D grid accessor (CubicInterpolant stores x in cache) ──
@inline _grid_1d(itp::CubicInterpolant) = itp.cache.x
@inline _grid_1d(itp::AbstractInterpolant) = itp.x

# ── Unified 1D full-domain: replaces 4 identical per-type methods ──
@inline function integrate(
    itp::AbstractInterpolant{Tg,Tv};
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = _grid_1d(itp)
    return integrate(itp, first(x), last(x); search=search, hint=hint)
end

# ── Fallback stub (bounded 1D) ──
function integrate(itp::AbstractInterpolant, x0::Real, x1::Real; search=nothing, hint=nothing)
    throw(ArgumentError("integrate(itp, x0, x1) is not implemented for $(typeof(itp)) yet"))
end

# ── CubicInterpolant 1D ──

@inline function integrate(
    itp::CubicInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.cache.x
    y = itp.y
    z = itp.z
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _to_searcher(search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _cubic_integral_kernel(
            _EvalIntegralPartial(), z[i], z[i+1], y[i], y[i+1], h, a2 - xL, b2 - xL
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _cubic_integral_kernel(
            _EvalIntegralCell(), z[i], z[i+1], y[i], y[i+1], h
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── LinearInterpolant 1D ──

@inline function integrate(
    itp::LinearInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.x
    y = itp.y
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _to_searcher(search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _linear_integral_kernel(
            _EvalIntegralPartial(), y[i], y[i+1], h, a2 - xL, b2 - xL
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _linear_integral_kernel(
            _EvalIntegralCell(), y[i], y[i+1], h
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, y[1], y[end], x0, x1, Tout)
end

# ── QuadraticInterpolant 1D ──

@inline function integrate(
    itp::QuadraticInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.x
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _to_searcher(search, hint)

    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _quadratic_integral_kernel(
            _EvalIntegralPartial(), itp.a[i], itp.d[i], itp.y[i], a2 - xL, b2 - xL
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _quadratic_integral_kernel(
            _EvalIntegralCell(), itp.a[i], itp.d[i], itp.y[i], h
        )
    end

    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)

    return _dispatch_extrap_integrate_1d(itp.extrap, in_domain, x, itp.y[1], itp.y[end], x0, x1, Tout)
end

# ── ConstantInterpolant 1D ──

@inline function integrate(
    itp::ConstantInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    searcher = _to_searcher(search, hint)
    # Manual if-else on side to avoid >4 union combinations.
    # Each branch has concrete side → _integrate_constant_1d_impl is fully typed.
    side = itp.side
    if side === Val(:left)
        return _integrate_constant_1d_impl(itp.x, itp.y, Val(:left), itp.extrap, x0, x1, searcher, Tg, Tout)
    elseif side === Val(:right)
        return _integrate_constant_1d_impl(itp.x, itp.y, Val(:right), itp.extrap, x0, x1, searcher, Tg, Tout)
    else
        return _integrate_constant_1d_impl(itp.x, itp.y, Val(:nearest), itp.extrap, x0, x1, searcher, Tg, Tout)
    end
end

# Receives concrete side Val{S} + union extrap (4-way, within union-split limit).
# Uses the generic _integrate_1d_cellwise path — side is already concrete here.
@inline function _integrate_constant_1d_impl(
    x::AbstractVector, y::AbstractVector, side::Val{S}, extrap::ExtrapVal,
    x0::Real, x1::Real, searcher::SR, ::Type{Tg}, ::Type{Tout}
) where {S, SR<:Searcher, Tg, Tout}
    partial = @inline (i, xL, h, a2, b2) -> begin
        @inbounds _constant_integral_kernel(
            _EvalIntegralPartial(), y[i], y[i+1], h, a2 - xL, b2 - xL, side
        )
    end
    full = @inline (i, h) -> begin
        @inbounds _constant_integral_kernel(
            _EvalIntegralPartial(), y[i], y[i+1], h, zero(Tg), h, side
        )
    end
    in_domain = @inline (a, b) -> _integrate_1d_cellwise(x, a, b, searcher, partial, full, Tout)
    y_left = side === Val(:right) ? (@inbounds y[2]) : (@inbounds y[1])
    y_right = side === Val(:left) ? (@inbounds y[end-1]) : (@inbounds y[end])
    return _dispatch_extrap_integrate_1d(extrap, in_domain, x, y_left, y_right, x0, x1, Tout)
end

# ═══════════════════════════════════════════════════════════════
# ND Integration
# ═══════════════════════════════════════════════════════════════

# ── Unified ND full-domain: replaces 4 identical per-type methods ──
@inline function integrate(
    itp::AbstractInterpolantND{Tg,Tv,N};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    lo = ntuple(d -> first(itp.grids[d]), Val(N))
    hi = ntuple(d -> last(itp.grids[d]), Val(N))
    return integrate(itp, lo, hi; search=search, hint=hint)
end

# ── Fallback stub (bounded ND) ──
function integrate(
    itp::AbstractInterpolantND{Tg,Tv,N},
    lo::Tuple{Vararg{Real,N}},
    hi::Tuple{Vararg{Real,N}};
    search=nothing,
    hint=nothing
) where {Tg,Tv,N}
    throw(ArgumentError("integrate(itp_nd, lo, hi) is not implemented for $(typeof(itp)) yet"))
end

# ═══════════════════════════════════════════════════════════════
# ND Cubic Integration
# ═══════════════════════════════════════════════════════════════

# @generated tensor-product cell integral kernel
# Mirrors _eval_nd_cell but replaces _hermite_kernel_1d with
# _hermite_integral_kernel_1d for each dimension collapse stage.
@inline @generated function _integrate_nd_cubic_cell(
    partials::Array{Tv, NP1},
    indices::NTuple{N, Int},
    hs::NTuple{N, Tg},
    inv_hs::NTuple{N, Tg},
    ulos::NTuple{N, <:Real},
    uhis::NTuple{N, <:Real}
) where {Tv, Tg, N, NP1}
    NP1 == N + 1 || error("NP1 must equal N+1")

    stmts = Expr[]

    # Unpack tuples
    for (prefix, source) in [("idx_", :indices), ("h_", :hs), ("inv_h_", :inv_hs),
                              ("ulo_", :ulos), ("uhi_", :uhis)]
        syms = ntuple(d -> Symbol(prefix, d), N)
        lhs = Expr(:tuple, syms...)
        push!(stmts, :($lhs = $source))
    end

    # Collapse each dimension via integral kernel
    for stage in 1:N
        num_corners = 1 << (N - stage)
        num_derivs = 1 << (N - stage)

        for corner in 0:(num_corners - 1)
            for deriv in 0:(num_derivs - 1)
                out_var = _varname(stage, corner, deriv)

                if stage == 1
                    function make_partial_access(c_dim1::Int, d_dim1::Int)
                        corner_full = c_dim1 | (corner << 1)
                        deriv_full = d_dim1 | (deriv << 1)
                        p_idx = _partial_index(deriv_full)
                        offsets = _corner_offset_expr(corner_full, N)
                        idx_exprs = [:($(Symbol("idx_", d)) + $(offsets[d])) for d in 1:N]
                        return :(partials[$p_idx, $(idx_exprs...)])
                    end

                    fL = make_partial_access(0, 0)
                    fR = make_partial_access(1, 0)
                    dfL = make_partial_access(0, 1)
                    dfR = make_partial_access(1, 1)
                else
                    prev_stage = stage - 1
                    fL = _varname(prev_stage, 0 | (corner << 1), 0 | (deriv << 1))
                    fR = _varname(prev_stage, 1 | (corner << 1), 0 | (deriv << 1))
                    dfL = _varname(prev_stage, 0 | (corner << 1), 1 | (deriv << 1))
                    dfR = _varname(prev_stage, 1 | (corner << 1), 1 | (deriv << 1))
                end

                h = Symbol("h_", stage)
                inv_h = Symbol("inv_h_", stage)
                ulo = Symbol("ulo_", stage)
                uhi = Symbol("uhi_", stage)

                kernel_call = :(_hermite_integral_kernel_1d($fL, $fR, $dfL, $dfR, $h, $inv_h, $ulo, $uhi))
                push!(stmts, :($out_var = $kernel_call))
            end
        end
    end

    final_var = _varname(N, 0, 0)
    push!(stmts, :(return $final_var))

    return quote
        Base.@_inline_meta
        @inbounds begin
            $(stmts...)
        end
    end
end

# ── CubicInterpolantND bounded ──

@inline function integrate(
    itp::CubicInterpolantND{Tg,Tv,N},
    lo::Tuple{Vararg{Real,N}},
    hi::Tuple{Vararg{Real,N}};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.spacings, lo, hi, search, hint
    )
    sign == 0 && return zero(promote_type(Tv, Tg))

    total = zero(promote_type(Tv, Tg, map(typeof, lo2)..., map(typeof, hi2)...))
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.spacings[d], idx[d])), Val(N))
            total += _integrate_nd_cubic_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, uhis)
        end
    end
    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# ND Linear Integration
# ═══════════════════════════════════════════════════════════════

@inline function integrate(
    itp::LinearInterpolantND{Tg,Tv,N},
    lo::Tuple{Vararg{Real,N}},
    hi::Tuple{Vararg{Real,N}};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.spacings, lo, hi, search, hint
    )
    sign == 0 && return zero(promote_type(Tv, Tg))

    total = zero(promote_type(Tv, Tg, map(typeof, lo2)..., map(typeof, hi2)...))
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            total += _integrate_linear_nd_cell(itp.data, idx, hs, ulos, uhis)
        end
    end
    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# ND Quadratic Integration
# ═══════════════════════════════════════════════════════════════

@inline function integrate(
    itp::QuadraticInterpolantND{Tg,Tv,N},
    lo::Tuple{Vararg{Real,N}},
    hi::Tuple{Vararg{Real,N}};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.spacings, lo, hi, search, hint
    )
    sign == 0 && return zero(promote_type(Tv, Tg))

    total = zero(promote_type(Tv, Tg, map(typeof, lo2)..., map(typeof, hi2)...))
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.spacings[d], idx[d])), Val(N))
            total += _integrate_nd_quad_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, uhis)
        end
    end
    return sign * total
end

# ═══════════════════════════════════════════════════════════════
# ND Constant Integration
# ═══════════════════════════════════════════════════════════════

@inline function integrate(
    itp::ConstantInterpolantND{Tg,Tv,N},
    lo::Tuple{Vararg{Real,N}},
    hi::Tuple{Vararg{Real,N}};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.spacings, lo, hi, search, hint
    )
    sign == 0 && return zero(promote_type(Tv, Tg))

    total = zero(promote_type(Tv, Tg, map(typeof, lo2)..., map(typeof, hi2)...))
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx, hs, ulos, uhis = _nd_cell_geom(itp.grids, itp.spacings, lo2, hi2, I, Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            total += _integrate_constant_nd_cell(itp.data, idx, hs, ulos, uhis, itp.sides)
        end
    end
    return sign * total
end
