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
