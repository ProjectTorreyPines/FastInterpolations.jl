# ── Fallback stubs (not-yet-implemented methods) ──

function integrate(itp::AbstractInterpolant)
    throw(ArgumentError("integrate(itp) is not implemented for $(typeof(itp)) yet"))
end

function integrate(itp::AbstractInterpolant, x0::Real, x1::Real; search=nothing, hint=nothing)
    throw(ArgumentError("integrate(itp, x0, x1) is not implemented for $(typeof(itp)) yet"))
end

# ── CubicInterpolant 1D ──

@inline function integrate(
    itp::CubicInterpolant{Tg,Tv};
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.cache.x
    return integrate(itp, first(x), last(x); search=search, hint=hint)
end

@inline function integrate(
    itp::CubicInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.cache.x
    spacing = itp.cache.spacing
    y = itp.y
    z = itp.z

    @boundscheck _check_domain(x, min(x0, x1), itp.extrap)
    @boundscheck _check_domain(x, max(x0, x1), itp.extrap)

    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    return _integrate_1d_cellwise(
        x, spacing, x0, x1;
        search=search,
        hint=hint,
        partial_fn=@inline((i, xL, a, b) -> begin
            h = _get_h(spacing, i)
            @inbounds _cubic_integral_kernel(
                _EvalIntegralPartial(), z[i], z[i+1], y[i], y[i+1], h, a - xL, b - xL
            )
        end),
        full_fn=@inline((i) -> begin
            h = _get_h(spacing, i)
            @inbounds _cubic_integral_kernel(
                _EvalIntegralCell(), z[i], z[i+1], y[i], y[i+1], h
            )
        end),
        Tout=Tout
    )
end

# ── LinearInterpolant 1D ──

@inline function integrate(
    itp::LinearInterpolant{Tg,Tv};
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    return integrate(itp, first(itp.x), last(itp.x); search=search, hint=hint)
end

@inline function integrate(
    itp::LinearInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.x
    y = itp.y
    spacing = _create_spacing(x)

    @boundscheck _check_domain(x, min(x0, x1), itp.extrap)
    @boundscheck _check_domain(x, max(x0, x1), itp.extrap)

    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    return _integrate_1d_cellwise(
        x, spacing, x0, x1;
        search=search,
        hint=hint,
        partial_fn=@inline((i, xL, a, b) -> begin
            h = _get_h(spacing, i)
            @inbounds _linear_integral_kernel(
                _EvalIntegralPartial(), y[i], y[i+1], h, a - xL, b - xL
            )
        end),
        full_fn=@inline((i) -> begin
            h = _get_h(spacing, i)
            @inbounds _linear_integral_kernel(
                _EvalIntegralCell(), y[i], y[i+1], h
            )
        end),
        Tout=Tout
    )
end

# ── QuadraticInterpolant 1D ──

@inline function integrate(
    itp::QuadraticInterpolant{Tg,Tv};
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    return integrate(itp, first(itp.x), last(itp.x); search=search, hint=hint)
end

@inline function integrate(
    itp::QuadraticInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.x
    spacing = _create_spacing(x)

    @boundscheck _check_domain(x, min(x0, x1), itp.extrap)
    @boundscheck _check_domain(x, max(x0, x1), itp.extrap)

    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    return _integrate_1d_cellwise(
        x, spacing, x0, x1;
        search=search,
        hint=hint,
        partial_fn=@inline((i, xL, a, b) -> begin
            @inbounds _quadratic_integral_kernel(
                _EvalIntegralPartial(), itp.a[i], itp.d[i], itp.y[i], a - xL, b - xL
            )
        end),
        full_fn=@inline((i) -> begin
            h = _get_h(spacing, i)
            @inbounds _quadratic_integral_kernel(
                _EvalIntegralCell(), itp.a[i], itp.d[i], itp.y[i], h
            )
        end),
        Tout=Tout
    )
end

# ── ConstantInterpolant 1D ──

@inline function integrate(
    itp::ConstantInterpolant{Tg,Tv};
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    return integrate(itp, first(itp.x), last(itp.x); search=search, hint=hint)
end

@inline function integrate(
    itp::ConstantInterpolant{Tg,Tv},
    x0::Real, x1::Real;
    search=itp.search_policy,
    hint::Union{Nothing,Base.RefValue{Int}}=nothing
) where {Tg<:AbstractFloat, Tv}
    x = itp.x
    y = itp.y
    side = itp.side
    spacing = _create_spacing(x)

    @boundscheck _check_domain(x, min(x0, x1), itp.extrap)
    @boundscheck _check_domain(x, max(x0, x1), itp.extrap)

    Tout = promote_type(Tv, Tg, typeof(x0), typeof(x1))
    return _integrate_1d_cellwise(
        x, spacing, x0, x1;
        search=search,
        hint=hint,
        partial_fn=@inline((i, xL, a, b) -> begin
            h = _get_h(spacing, i)
            @inbounds _constant_integral_kernel(
                _EvalIntegralPartial(), y[i], y[i+1], h, a - xL, b - xL, side
            )
        end),
        full_fn=@inline((i) -> begin
            h = _get_h(spacing, i)
            @inbounds _constant_integral_kernel(
                _EvalIntegralPartial(), y[i], y[i+1], h, zero(Tg), h, side
            )
        end),
        Tout=Tout
    )
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

# In-domain ND cubic integration over a bounding box
@inline function _integrate_cubic_nd_in_domain(
    itp::CubicInterpolantND{Tg,Tv,N},
    lo::NTuple{N,<:Real},
    hi::NTuple{N,<:Real};
    search=itp.searches,
    hint=nothing
) where {Tg,Tv,N}
    sign, lo2, hi2 = _normalize_bounds_nd(lo, hi)
    sign == 0 && return zero(promote_type(Tv, Tg))

    @inbounds for d in 1:N
        _check_domain(itp.grids[d], lo2[d], Val(:none))
        _check_domain(itp.grids[d], hi2[d], Val(:none))
    end

    search_tuple = _resolve_search_nd(search, Val(N))
    idx_lo, idx_hi = _nd_cell_ranges(itp.grids, itp.spacings, lo2, hi2, search_tuple, hint)

    total = zero(promote_type(Tv, Tg, map(typeof, lo2)..., map(typeof, hi2)...))
    for I in CartesianIndices(ntuple(d -> idx_lo[d]:idx_hi[d], Val(N)))
        idx = ntuple(d -> I[d], Val(N))
        hs = ntuple(d -> _get_h(itp.spacings[d], idx[d]), Val(N))
        inv_hs = ntuple(d -> _get_inv_h(itp.spacings[d], idx[d]), Val(N))
        Ls = ntuple(d -> itp.grids[d][idx[d]], Val(N))
        Rs = ntuple(d -> itp.grids[d][idx[d] + 1], Val(N))
        ulos = ntuple(d -> max(lo2[d], Ls[d]) - Ls[d], Val(N))
        uhis = ntuple(d -> min(hi2[d], Rs[d]) - Ls[d], Val(N))
        if all(d -> uhis[d] > ulos[d], 1:N)
            total += _integrate_nd_cubic_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, uhis)
        end
    end
    return sign * total
end

# ── CubicInterpolantND API ──

@inline function integrate(
    itp::CubicInterpolantND{Tg,Tv,N};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    lo = ntuple(d -> first(itp.grids[d]), Val(N))
    hi = ntuple(d -> last(itp.grids[d]), Val(N))
    return integrate(itp, lo, hi; search=search, hint=hint)
end

@inline function integrate(
    itp::CubicInterpolantND{Tg,Tv,N},
    lo::NTuple{N,<:Real},
    hi::NTuple{N,<:Real};
    search=itp.searches,
    hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}}=nothing
) where {Tg,Tv,N}
    return _integrate_cubic_nd_in_domain(itp, lo, hi; search=search, hint=hint)
end

# ── ND Fallback stubs ──

function integrate(itp::AbstractInterpolantND{Tg,Tv,N}) where {Tg,Tv,N}
    throw(ArgumentError("integrate(itp_nd) is not implemented for $(typeof(itp)) yet"))
end

function integrate(
    itp::AbstractInterpolantND{Tg,Tv,N},
    lo::NTuple{N,<:Real},
    hi::NTuple{N,<:Real};
    search=nothing,
    hint=nothing
) where {Tg,Tv,N}
    throw(ArgumentError("integrate(itp_nd, lo, hi) is not implemented for $(typeof(itp)) yet"))
end
