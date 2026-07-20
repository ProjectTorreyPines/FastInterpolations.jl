# ═══════════════════════════════════════════════════════════════
# Full-domain fast path + cumulative integration
#
# When integrating over the entire domain, ALL cells are full cells:
# no search, no partial-cell handling, no bound normalization.
# This file defines:
#   - _full_cell_fn(itp, ...) trait: per-type closure for one cell's integral
#   - integrate(itp)          fast path: scalar accumulator (1D, Series, ND)
#   - cumulative_integrate    prefix-sum of per-cell integrals (1D, Series)
# ═══════════════════════════════════════════════════════════════

# ── 1D scalar trait: _full_cell_fn(itp) → (i, h) -> value ──

@inline function _full_cell_fn(itp::CubicInterpolant)
    y, z = itp.y, itp.z
    return @inline (i, h) -> @inbounds _cubic_integral_kernel(_EvalIntegralCell(), z[i], z[i + 1], y[i], y[i + 1], h)
end

# Named functor (not an anonymous closure) so the engine can dispatch on it — see
# the uniform-grid closed form `_integrate_1d_fulldomain(::_CachedRange, ::_LinearFullCell, …)`.
struct _LinearFullCell{Y <: AbstractVector} <: Function
    y::Y
end
@inline (f::_LinearFullCell)(i, h) = @inbounds _linear_integral_kernel(_EvalIntegralCell(), f.y[i], f.y[i + 1], h)
@inline _full_cell_fn(itp::LinearInterpolant) = _LinearFullCell(itp.y)

@inline function _full_cell_fn(itp::QuadraticInterpolant)
    a, d, y = itp.a, itp.d, itp.y
    return @inline (i, h) -> @inbounds _quadratic_integral_kernel(_EvalIntegralCell(), a[i], d[i], y[i], h)
end

@inline _full_cell_fn(itp::AbstractHermiteInterpolant1D) =
    _hermite_full_cell_fn(itp, itp.dy)

# PreCompute branch: slopes live in `itp.dy::AbstractVector`, indexed directly.
# `inv_h` is pulled from the wrapped grid's precomputed cache via `_get_inv_h`
# — single field load for `_CachedRange`, indexed load for `_CachedVector` —
# always cheaper than computing `inv(h)` per cell.
@inline function _hermite_full_cell_fn(itp::AbstractHermiteInterpolant1D, dy::AbstractVector)
    x = _grid_1d(itp)
    y = itp.y
    return @inline (i, h) -> begin
        inv_h = _get_inv_h(x, i)
        @inbounds _hermite_integral_kernel_1d(y[i], y[i + 1], dy[i], dy[i + 1], h, inv_h, zero(h), h)
    end
end

# OnTheFly branch: slopes computed locally per call via `_local_slope`.
# Each cell independently computes its two slopes — no sliding-window state,
# so the closure stays immutable and heap-free. The PCHIP/Cardinal/Akima stencils
# are O(1), so the extra recomputation (2(n-1) calls vs n for a sliding window)
# adds only a constant factor to full-domain integration; the code simplification
# removes three dedicated kernels.
@inline function _hermite_full_cell_fn(itp::AbstractHermiteInterpolant1D, sm::AbstractSlopeMethod)
    x, y = _grid_1d(itp), itp.y
    n = length(x)
    return @inline (i, h) -> begin
        dy_L = _local_slope(sm, x, y, i, n)
        dy_R = _local_slope(sm, x, y, i + 1, n)
        inv_h = _get_inv_h(x, i)
        @inbounds _hermite_integral_kernel_1d(y[i], y[i + 1], dy_L, dy_R, h, inv_h, zero(h), h)
    end
end

@inline function _full_cell_fn(itp::ConstantInterpolant{Tg}, side::AbstractSide) where {Tg}
    y = itp.y
    return @inline (i, h) -> @inbounds _constant_integral_kernel(_EvalIntegralPartial(), y[i], y[i + 1], h, zero(Tg), h, side)
end

# ── 1D series trait: _full_cell_fn(sitp, k) → (i, h) -> value ──

@inline function _full_cell_fn(sitp::CubicSeriesInterpolant, k::Int)
    y, z = sitp.y, sitp.z
    return @inline (i, h) -> @inbounds _cubic_integral_kernel(_EvalIntegralCell(), z[i, k], z[i + 1, k], y[i, k], y[i + 1, k], h)
end

@inline function _full_cell_fn(sitp::LinearSeriesInterpolant, k::Int)
    y = sitp.y
    return @inline (i, h) -> @inbounds _linear_integral_kernel(_EvalIntegralCell(), y[i, k], y[i + 1, k], h)
end

@inline function _full_cell_fn(sitp::QuadraticSeriesInterpolant, k::Int)
    a, d, y = sitp.a, sitp.d, sitp.y
    return @inline (i, h) -> @inbounds _quadratic_integral_kernel(_EvalIntegralCell(), a[i, k], d[i, k], y[i, k], h)
end

@inline function _full_cell_fn(sitp::ConstantSeriesInterpolant{Tg}, k::Int, side::AbstractSide) where {Tg}
    y = sitp.y
    return @inline (i, h) -> @inbounds _constant_integral_kernel(_EvalIntegralPartial(), y[i, k], y[i + 1, k], h, zero(Tg), h, side)
end

# ═══════════════════════════════════════════════════════════════
# integrate(itp) — 1D full-domain fast path
# ═══════════════════════════════════════════════════════════════

# Generic 1D: catches any interpolant whose `_full_cell_fn(itp)` trait is the
# single-arg form — Cubic, Linear, Quadratic, and the entire Hermite family
# (PreCompute + OnTheFly via `_full_cell_fn`'s internal dispatch on `itp.dy`).
# Constant has its own override below because its full-cell trait depends on `side`.
"""
    # persistent — integrate an interpolant you already built
    integrate(itp)                          # full-domain
    integrate(itp, a, b)                    # 1-D, over [a, b]
    integrate(itp, lo::NTuple, hi::NTuple)  # ND, over the box [lo, hi]

    # one-shot — build the `method` interpolant from raw data, then integrate
    integrate(x, y; method)                 # 1-D full-domain
    integrate(x, y, a, b; method)           # 1-D, over [a, b]
    integrate(grids, data; method)          # ND full-domain
    integrate(grids, data, lo, hi; method)  # ND, over the box [lo, hi]

Definite integral of an interpolant.

The **persistent** forms integrate an interpolant you built earlier.
`integrate(itp)` covers the whole domain through a specialized search-free
summation; the bounded forms integrate over `[a, b]` (1-D) or the
hyper-rectangle `[lo, hi]` (ND).

The **one-shot** forms build the `method` interpolant from raw data with
`copy=false` reference storage — nothing is copied — then integrate in a single
call. 1-D integrates every method; ND only the tensor-product types (Linear,
Cubic, Quadratic, Constant), as the Hermite family has no ND integral.

```julia
integrate(cubic_interp(x, y))                        # persistent, full-domain
integrate(x, y; method = CubicInterp())              # one-shot,   full-domain
integrate(x, y, 0.2, 1.5; method = LinearInterp())   # one-shot,   ∫ from 0.2 to 1.5
integrate((xs, ys), data; method = LinearInterp())   # one-shot,   2-D full-domain
```
"""
@inline function integrate(
        itp::AbstractInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: Real, Tv}
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    return _integrate_1d_fulldomain(_grid_1d(itp), _full_cell_fn(itp), Tout)
end

# Uniform-grid linear closed form: the trapezoid telescopes to h·Σy_interior + one
# end-cell kernel call — the kernel keeps the `_fieldsum` widen / h-only-division
# duck contracts; Float16 accumulates in Float32 (raw Σy overflows 65504).
# Summation order ≠ cellwise engine → ULP-level shifts.
@inline function _integrate_1d_fulldomain(
        x::_CachedRange, f::_LinearFullCell, ::Type{Tout}
    ) where {Tout}
    y = f.y
    n = length(x)
    n < 2 && return zero(Tout)
    h = _get_h(x)
    s = zero(Tout === Float16 ? Float32 : Tout)
    @inbounds begin
        @simd for i in 2:(n - 1)
            s += y[i]
        end
        return convert(Tout, h * s + _linear_integral_kernel(_EvalIntegralCell(), y[1], y[n], h))
    end
end

# Constant override: side is parametric → compiler knows concrete type,
# so `_full_cell_fn(itp, side)` is a distinct method from the generic single-arg form.
@inline function integrate(
        itp::ConstantInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: Real, Tv}
    x = _grid_1d(itp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    return _integrate_1d_fulldomain(x, _full_cell_fn(itp, itp.side), Tout)
end

# Hermite family is handled by the generic path above — `_full_cell_fn`
# internally dispatches on `itp.dy` (PreCompute vs OnTheFly).

# ═══════════════════════════════════════════════════════════════
# integrate(sitp) — 1D Series full-domain fast path
# ═══════════════════════════════════════════════════════════════

# Generic Series: catches Cubic, Linear, Quadratic series
@inline function integrate(
        sitp::AbstractSeriesInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: Real, Tv}
    x = _grid_1d(sitp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        results[k] = _integrate_1d_fulldomain(x, _full_cell_fn(sitp, k), Tout)
    end
    return results
end

# Constant Series override: side is parametric → compiler knows concrete type
@inline function integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv};
        search = nothing, hint = nothing
    ) where {Tg <: Real, Tv}
    x = _grid_1d(sitp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        results[k] = _integrate_1d_fulldomain(x, _full_cell_fn(sitp, k, sitp.side), Tout)
    end
    return results
end

# ═══════════════════════════════════════════════════════════════
# integrate(itp) — ND full-domain fast path
# ═══════════════════════════════════════════════════════════════

# ND per-type trait: _full_cell_integral_nd(itp, idx, hs)
# Each calls the existing ND kernel with ulos = zeros, uhis = hs. Methods whose
# kernel needs reciprocal spacings (cubic/quadratic hermite-form collapse) fetch
# inv_hs themselves — mirrors 1D, where `_hermite_full_cell_fn` pulls `_get_inv_h`
# inside the closure; linear/constant never pay for it (their kernels take no inv_hs).

@inline function _full_cell_integral_nd(
        itp::CubicInterpolantND{Tg, Tv, N}, idx, hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.grids[d], idx[d])), Val(N))
    return _integrate_nd_cubic_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, hs)
end

@inline function _full_cell_integral_nd(
        itp::LinearInterpolantND{Tg, Tv, N}, idx, hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    return _integrate_linear_nd_cell(itp.data, idx, hs, ulos, hs)
end

@inline function _full_cell_integral_nd(
        itp::QuadraticInterpolantND{Tg, Tv, N}, idx, hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    inv_hs = ntuple(d -> @inbounds(_get_inv_h(itp.grids[d], idx[d])), Val(N))
    return _integrate_nd_quad_cell(itp.nodal_derivs.partials, idx, hs, inv_hs, ulos, hs)
end

@inline function _full_cell_integral_nd(
        itp::ConstantInterpolantND{Tg, Tv, N}, idx, hs
    ) where {Tg, Tv, N}
    ulos = ntuple(d -> zero(Tg), Val(N))
    return _integrate_constant_nd_cell(itp.data, idx, hs, ulos, hs, itp.sides)
end

# Sample Tv value for duck-typing safe zero initialization
@inline _nd_sample_value(itp::LinearInterpolantND) = @inbounds itp.data[1]
@inline _nd_sample_value(itp::ConstantInterpolantND) = @inbounds itp.data[1]
@inline _nd_sample_value(itp::CubicInterpolantND) = @inbounds itp.nodal_derivs.partials[1]
@inline _nd_sample_value(itp::QuadraticInterpolantND) = @inbounds itp.nodal_derivs.partials[1]

# HeteroInterpolantND: explicit not-implemented error. Beats the MethodError
# the generic dispatch below would hit inside `_full_cell_integral_nd`, and
# tells the user about the actual workaround (per-axis 1D integration, or
# switching to a homogeneous specialized ND type).
function integrate(
        itp::HeteroInterpolantND;
        search = nothing,
        hint = nothing,
    )
    _throw_hetero_nd_integrate_unsupported(itp.methods)
end

# Bounded ND integrate never existed; add a HeteroInterpolantND-specific
# overload too so `integrate(itp, a, b)` doesn't surface as a MethodError
# asking the user about Real vs Tuple bound types. Must use `::Real, ::Real`
# to win dispatch against the `(::AbstractInterpolant, ::Real, ::Real)`
# fallback in integrate_api.jl (more specific first arg + equal specificity
# on bounds).
function integrate(
        itp::HeteroInterpolantND, ::Real, ::Real;
        search = nothing, hint = nothing,
    )
    _throw_hetero_nd_integrate_unsupported(itp.methods)
end

@noinline function _throw_hetero_nd_integrate_unsupported(methods)
    throw(
        ArgumentError(
            "integrate is not yet implemented for HeteroInterpolantND " *
                "(method tuple: $(methods)). ND tensor-product integration over " *
                "mixed/Hermite axes is not yet supported. Workarounds: " *
                "(1) use a homogeneous specialized ND type " *
                "(`cubic_interp`, `quadratic_interp`, `linear_interp`, `constant_interp`) " *
                "which do support `integrate`; " *
                "(2) integrate axis-by-axis via 1D `integrate` calls on per-fiber " *
                "1D interpolants."
        )
    )
end

# ── Separable ND engine: full-domain + bounded, all tensor-product families ──
#
# On full cells every per-axis basis integral collapses to constants that are
# LINEAR in the node payload, so the domain integral is a tensor product of 1D
# node-weight rules contracted against the payload — each node read once,
# instead of the 2^N corner reads per cell the generic engine repeats:
#
#   rank 1 — payload `data[i…]`, one weight per axis:
#     linear    w = ½h at the ends, ½(h₋+h₊) interior (composite trapezoid)
#     constant  w = side-selected h (NearestSide ≡ the trapezoid weights)
#   rank 2 — payload `partials[1+mask, i…]` (bit d of mask = ∂ along axis d),
#             a (w, v) = (value, deriv) weight pair per axis:
#     cubic     ∫cell = h/2(fL+fR) + h²/12(dfL−dfR) → v telescopes interior,
#               vanishing on uniform axes
#     quad      ∫cell = 2h/3·fL + h/3·fR + h²/6·dfL → left-anchored, no right v
#
# The per-axis weight spec reuses the existing vocabulary — the method tags
# (`LinearInterp`/`CubicInterp`/`QuadraticInterp`) or the constant `sides` — and
# the @generated kernels pick the rank from the payload dimensionality (N vs
# N+1), so one engine serves every family; only the weight methods differ.

# 1D composite-trapezoid node weight on `grid` (length `n`): ½h at the endpoints,
# ½(h₋+h₊) in the interior. Only the grid is divided — values stay multiply-only.
@inline _nd_trap_weight_interior(grid, k::Int) = (_get_h(grid, k - 1) + _get_h(grid, k)) / 2
@inline function _nd_trap_weight(grid, k::Int, n::Int)
    k == 1 && return _get_h(grid, 1) / 2
    k == n && return _get_h(grid, n - 1) / 2
    return _nd_trap_weight_interior(grid, k)
end

# ── Full-cell node weights: `_nd_node_weights(tag, g, k, n) -> NTuple{R}` ──

@inline _nd_node_weights(::Union{LinearInterp, NearestSide}, g, k::Int, n::Int) =
    (_nd_trap_weight(g, k, n),)
@inline _nd_node_weights_interior(::Union{LinearInterp, NearestSide}, g, k::Int) =
    (_nd_trap_weight_interior(g, k),)
@inline _nd_node_weights(::LeftSide, g, k::Int, n::Int) =
    (k == n ? zero(_get_h(g, 1)) : _get_h(g, k),)
@inline _nd_node_weights_interior(::LeftSide, g, k::Int) = (_get_h(g, k),)
@inline _nd_node_weights(::RightSide, g, k::Int, n::Int) =
    (k == 1 ? zero(_get_h(g, 1)) : _get_h(g, k - 1),)
@inline _nd_node_weights_interior(::RightSide, g, k::Int) = (_get_h(g, k - 1),)

@inline function _nd_node_weights(::CubicInterp, g, k::Int, n::Int)
    if k == 1
        h1 = _get_h(g, 1)
        return (h1 / 2, h1 * h1 / 12)
    elseif k == n
        hm = _get_h(g, n - 1)
        return (hm / 2, -hm * hm / 12)
    end
    return _nd_node_weights_interior(CubicInterp(), g, k)
end
@inline function _nd_node_weights_interior(::CubicInterp, g, k::Int)
    hm = _get_h(g, k - 1)
    hk = _get_h(g, k)
    return ((hm + hk) / 2, (hk * hk - hm * hm) / 12)
end

@inline function _nd_node_weights(::QuadraticInterp, g, k::Int, n::Int)
    if k == 1
        h1 = _get_h(g, 1)
        return (2 * h1 / 3, h1 * h1 / 6)
    elseif k == n
        hm = _get_h(g, n - 1)
        return (hm / 3, zero(hm * hm))
    end
    return _nd_node_weights_interior(QuadraticInterp(), g, k)
end
@inline function _nd_node_weights_interior(::QuadraticInterp, g, k::Int)
    hm = _get_h(g, k - 1)
    hk = _get_h(g, k)
    return ((2 * hk + hm) / 3, hk * hk / 6)
end

# ── Partial-cell node shares: `_nd_cell_weights0/1(tag, u0, u1, h) -> NTuple{R}` ──
# Share of ∫_{u0}^{u1} within one cell landing on the left (0) / right (1) node;
# full cells reduce to the closed full-cell forms.

@inline _nd_cell_weights0(::LinearInterp, u0, u1, h) = (_w0_int(u0, u1, h),)
@inline _nd_cell_weights1(::LinearInterp, u0, u1, h) = (_w1_int(u0, u1, h),)
@inline _nd_cell_weights0(s::AbstractSide, u0, u1, h) = (_cw0(u0, u1, h, s),)
@inline _nd_cell_weights1(s::AbstractSide, u0, u1, h) = (_cw1(u0, u1, h, s),)

# The ΔH antiderivatives carry Float64 coefficients, so the cubic pair converts
# back to the promoted input precision (`oftype`) — same numerics as the generic
# engine's per-cell `convert(Tout, …)`, keeping Float32 outputs Float32.
@inline function _nd_cell_weights0(::CubicInterp, u0, u1, h)
    t0 = u0 / h
    t1 = u1 / h
    return (
        oftype(h * one(t1), h * (_IH00(t1) - _IH00(t0))),
        oftype(h * h * one(t1), h * h * (_IH10(t1) - _IH10(t0))),
    )
end
@inline function _nd_cell_weights1(::CubicInterp, u0, u1, h)
    t0 = u0 / h
    t1 = u1 / h
    return (
        oftype(h * one(t1), h * (_IH01(t1) - _IH01(t0))),
        oftype(h * h * one(t1), h * h * (_IH11(t1) - _IH11(t0))),
    )
end
@inline function _nd_cell_weights0(::QuadraticInterp, u0, u1, h)
    du = u1 - u0
    D2 = du * (u1 + u0)
    D3 = u1 * u1 * u1 - u0 * u0 * u0
    return (du - D3 / (3 * h * h), D2 / 2 - D3 / (3 * h))
end
@inline function _nd_cell_weights1(::QuadraticInterp, u0, u1, h)
    D3 = u1 * u1 * u1 - u0 * u0 * u0
    return (D3 / (3 * h * h), zero(h * h))
end

# ── Bounded per-axis machinery ──

# Per-axis spec: covered cell range [ilo, ihi] plus the clipped local start
# inside cell `ilo` (u0 ≥ 0) and clipped local end inside cell `ihi` (u1 ≤ h).
struct _BoundedAxisSpec{T}
    ilo::Int
    ihi::Int
    u0::T
    u1::T
end

@generated function _nd_bounded_axis_specs(
        grids::NTuple{N, Any}, lo2, hi2, idx_lo, idx_hi
    ) where {N}
    exprs = [
        quote
                let g = grids[$d], il = idx_lo[$d], ih = idx_hi[$d]
                    xLl = @inbounds g[il]
                    xLh = @inbounds g[ih]
                    xRh = @inbounds g[ih + 1]
                    u0, u1 = promote(max(lo2[$d], xLl) - xLl, min(hi2[$d], xRh) - xLh)
                    _BoundedAxisSpec(il, ih, u0, u1)
            end
            end for d in 1:N
    ]
    return :(($(exprs...),))
end

# Clipped composite node weights: right-end share of cell k−1 plus left-end
# share of cell k, each clipped to the covered range. Only the ≤ 4 nodes
# touching the two boundary cells differ from the full-cell weights. The seed
# is an empty-span cell share — typed zeros of the tag's weight tuple.
@inline function _nd_bounded_node_weights(w, spec::_BoundedAxisSpec, g, k::Int)
    acc = _nd_cell_weights0(w, spec.u0, spec.u0, _get_h(g, spec.ilo))
    c = k - 1
    if spec.ilo <= c <= spec.ihi
        h = _get_h(g, c)
        acc = map(
            +, acc,
            _nd_cell_weights1(w, c == spec.ilo ? spec.u0 : zero(h), c == spec.ihi ? spec.u1 : h, h)
        )
    end
    c = k
    if spec.ilo <= c <= spec.ihi
        h = _get_h(g, c)
        acc = map(
            +, acc,
            _nd_cell_weights0(w, c == spec.ilo ? spec.u0 : zero(h), c == spec.ihi ? spec.u1 : h, h)
        )
    end
    return acc
end

# ── The two @generated kernels (full-domain / bounded), rank-aware ──
#
# Emission per rank: rank 1 hoists the plain outer weight product `wp` and
# multiplies once per inner sweep; rank 2 hoists the Kronecker partial products
# of the (w, v) pairs (axis 2 = fastest mask bit, matching the slot order) and
# folds each node's contiguous slot pair with one muladd per outer mask. The
# inner axis peels its boundary nodes so the interior runs branch-free @simd.

function _separable_emit_common(N::Int, R::Int)
    ivars = [Symbol(:i_, d) for d in 1:N]
    rest = ivars[2:N]
    M = 1 << (N - 1)
    pidx(slot, i1) = R == 1 ? :(payload[$i1, $(rest...)]) : :(payload[$slot, $i1, $(rest...)])
    wnames(ws) = R == 1 ? [Symbol(ws, :_1)] : [Symbol(ws, :_1), Symbol(ws, :_2)]
    wdecl(ws, call) = Expr(:(=), Expr(:tuple, wnames(ws)...), call)
    contrib(ws, i1) = R == 1 ? :($(Symbol(ws, :_1)) * $(pidx(0, i1))) :
        foldl(
            (a, b) -> :($a + $b),
            [
                :(
                    $(N == 1 ? 1 : Symbol(:ko2_, om)) * muladd(
                        $(Symbol(ws, :_2)), $(pidx(2om, i1)),
                        $(Symbol(ws, :_1)) * $(pidx(2om - 1, i1))
                    )
                ) for om in 1:M
            ]
        )
    close_expr = N == 1 ? :(total += s) : R == 1 ? :(total += wp_2 * s) : :(total += s)
    return ivars, rest, pidx, wdecl, contrib, close_expr
end

# Outer-axis loop wrapper: rank 1 chains the scalar product `wp_d`; rank 2
# builds the Kronecker partial products `ko{d}_{j}` of the (w, v) pairs.
function _separable_emit_outer(body, N::Int, R::Int, ivars, wcall::Function, ranges::Function)
    for d in 2:N
        id = ivars[d]
        assigns = Expr[]
        if R == 1
            wexpr = d == N ? :($(wcall(d))[1]) : :($(wcall(d))[1] * $(Symbol(:wp_, d + 1)))
            push!(assigns, :($(Symbol(:wp_, d)) = $wexpr))
        else
            push!(assigns, Expr(:(=), Expr(:tuple, Symbol(:w_, d), Symbol(:v_, d)), wcall(d)))
            if d == N
                push!(assigns, :($(Symbol(:ko, d, :_, 1)) = $(Symbol(:w_, d))))
                push!(assigns, :($(Symbol(:ko, d, :_, 2)) = $(Symbol(:v_, d))))
            else
                for j in 1:(1 << (N - d))
                    push!(assigns, :($(Symbol(:ko, d, :_, 2j - 1)) = $(Symbol(:w_, d)) * $(Symbol(:ko, d + 1, :_, j))))
                    push!(assigns, :($(Symbol(:ko, d, :_, 2j)) = $(Symbol(:v_, d)) * $(Symbol(:ko, d + 1, :_, j))))
                end
            end
        end
        body = quote
            for $id in $(ranges(d))
                $(assigns...)
                $body
            end
        end
    end
    return body
end

@generated function _integrate_separable_nd_fulldomain(
        wspec::NTuple{N, Any}, grids::NTuple{N, Any},
        payload::AbstractArray{Tv, NP}, ::Type{Tout}
    ) where {N, Tv, NP, Tout}
    R = NP - N + 1
    R in (1, 2) || error("payload must have N or N+1 dimensions")
    ivars, _, _, wdecl, contrib, close_expr = _separable_emit_common(N, R)
    inner = quote
        $(wdecl(:w1a, :(_nd_node_weights(w1, g1, 1, n_1))))
        $(wdecl(:w1b, :(_nd_node_weights(w1, g1, n_1, n_1))))
        s = $(contrib(:w1a, 1)) + $(contrib(:w1b, :n_1))
        @simd for i_1 in 2:(n_1 - 1)
            $(wdecl(:w1i, :(_nd_node_weights_interior(w1, g1, i_1))))
            s += $(contrib(:w1i, :i_1))
        end
        $close_expr
    end
    body = _separable_emit_outer(
        inner, N, R, ivars,
        d -> :(_nd_node_weights(wspec[$d], grids[$d], $(ivars[d]), $(Symbol(:n_, d)))),
        d -> :(1:$(Symbol(:n_, d)))
    )
    nassign = Expr(:block, [:($(Symbol(:n_, d)) = size(payload, $(d + R - 1))) for d in 1:N]...)
    return quote
        Base.@_inline_meta
        $nassign
        g1 = grids[1]
        w1 = wspec[1]
        total = zero(Tout)
        @inbounds begin
            $body
        end
        return total
    end
end

@generated function _integrate_separable_nd_bounded(
        wspec::NTuple{N, Any}, specs::NTuple{N, _BoundedAxisSpec},
        grids::NTuple{N, Any}, payload::AbstractArray{Tv, NP}, ::Type{Tout}
    ) where {N, Tv, NP, Tout}
    R = NP - N + 1
    R in (1, 2) || error("payload must have N or N+1 dimensions")
    ivars, _, _, wdecl, contrib, close_expr = _separable_emit_common(N, R)
    inner = quote
        s = zero(Tout)
        if ihi1 >= ilo1 + 2
            $(wdecl(:w1a, :(_nd_bounded_node_weights(w1, spec1, g1, ilo1))))
            $(wdecl(:w1b, :(_nd_bounded_node_weights(w1, spec1, g1, ilo1 + 1))))
            $(wdecl(:w1c, :(_nd_bounded_node_weights(w1, spec1, g1, ihi1))))
            $(wdecl(:w1d, :(_nd_bounded_node_weights(w1, spec1, g1, ihi1 + 1))))
            s += $(contrib(:w1a, :ilo1)) + $(contrib(:w1b, :(ilo1 + 1))) +
                $(contrib(:w1c, :ihi1)) + $(contrib(:w1d, :(ihi1 + 1)))
            @simd for i_1 in (ilo1 + 2):(ihi1 - 1)
                $(wdecl(:w1i, :(_nd_node_weights_interior(w1, g1, i_1))))
                s += $(contrib(:w1i, :i_1))
            end
        else
            for i_1 in ilo1:(ihi1 + 1)
                $(wdecl(:w1x, :(_nd_bounded_node_weights(w1, spec1, g1, i_1))))
                s += $(contrib(:w1x, :i_1))
            end
        end
        $close_expr
    end
    body = _separable_emit_outer(
        inner, N, R, ivars,
        d -> :(_nd_bounded_node_weights(wspec[$d], specs[$d], grids[$d], $(ivars[d]))),
        d -> :((specs[$d].ilo):(specs[$d].ihi + 1))
    )
    return quote
        Base.@_inline_meta
        g1 = grids[1]
        w1 = wspec[1]
        spec1 = specs[1]
        ilo1 = spec1.ilo
        ihi1 = spec1.ihi
        total = zero(Tout)
        @inbounds begin
            $body
        end
        return total
    end
end

# ── Dispatches: numeric values take the separable engine; duck values fall
# through to the generic per-cell methods. The weight spec per family is the
# existing vocabulary: a method-tag tuple, or the constant `sides` as-is. ──

@inline _separable_wspec(itp::LinearInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (ntuple(_ -> LinearInterp(), Val(N)), itp.data)
@inline _separable_wspec(itp::ConstantInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (itp.sides, itp.data)
@inline _separable_wspec(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (ntuple(_ -> CubicInterp(), Val(N)), itp.nodal_derivs.partials)
@inline _separable_wspec(itp::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (ntuple(_ -> QuadraticInterp(), Val(N)), itp.nodal_derivs.partials)

const _SeparableND{Tg, Tv, N} = Union{
    LinearInterpolantND{Tg, Tv, N}, ConstantInterpolantND{Tg, Tv, N},
    CubicInterpolantND{Tg, Tv, N}, QuadraticInterpolantND{Tg, Tv, N},
}

@inline function integrate(
        itp::_SeparableND{Tg, Tv, N};
        search = nothing,
        hint = nothing
    ) where {Tg, Tv <: Number, N}
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    wspec, payload = _separable_wspec(itp)
    return _integrate_separable_nd_fulldomain(wspec, itp.grids, payload, Tout)
end

# Bounded impl shared by the four thin dispatches below. The dispatches stay
# per-type (not the `_SeparableND` union): the legacy per-cell bounded methods
# in integrate_api.jl are per-type with unconstrained `Tv`, and a union-typed
# `Tv <: Number` method would cross specificities with them (ambiguity); a
# per-type `Tv <: Number` method is strictly more specific instead.
@inline function _separable_bounded(itp, lo, hi, search, hint, ::Type{Tg}, ::Type{Tv}) where {Tg, Tv}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    sign == 0 && return zero(Tout)
    specs = _nd_bounded_axis_specs(itp.grids, lo2, hi2, idx_lo, idx_hi)
    wspec, payload = _separable_wspec(itp)
    total = _integrate_separable_nd_bounded(wspec, specs, itp.grids, payload, Tout)
    return sign * total
end

for T in (:LinearInterpolantND, :ConstantInterpolantND, :CubicInterpolantND, :QuadraticInterpolantND)
    @eval @inline function integrate(
            itp::$T{Tg, Tv, N},
            lo::Tuple{Vararg{Real, N}},
            hi::Tuple{Vararg{Real, N}};
            search = itp.searches,
            hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
        ) where {Tg, Tv <: Number, N}
        return _separable_bounded(itp, lo, hi, search, hint, Tg, Tv)
    end
end

# Generic ND full-domain: catches all ND types
@inline function integrate(
        itp::AbstractInterpolantND{Tg, Tv, N};
        search = nothing,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    total = Tout <: Number ? zero(Tout) : 0 * _nd_sample_value(itp)
    cell_ranges = ntuple(d -> 1:(length(itp.grids[d]) - 1), Val(N))
    for I in CartesianIndices(cell_ranges)
        idx = ntuple(d -> I[d], Val(N))
        hs = ntuple(d -> @inbounds(_get_h(itp.grids[d], idx[d])), Val(N))
        total += _full_cell_integral_nd(itp, idx, hs)
    end
    return total
end

# ═══════════════════════════════════════════════════════════════
# cumulative_integrate / cumulative_integrate! — 1D prefix-sum
# ═══════════════════════════════════════════════════════════════
#
# Architecture: the in-place `cumulative_integrate!(out, itp)` is the real
# implementation; the allocating `cumulative_integrate(itp)` is a thin
# wrapper that allocates `out` and forwards. Every interpolant type gets
# *one* method on `cumulative_integrate!` — Hermite OnTheFly joins the
# generic path via `_full_cell_fn(itp)`'s internal dispatch on `itp.dy`.

@noinline function _throw_cumulative_length_mismatch(got::Int, expected::Int)
    throw(
        DimensionMismatch(
            "cumulative_integrate! output buffer length $got does not match grid length $expected"
        )
    )
end

@inline function _check_cumulative_out(out::AbstractVector, n::Int)
    length(out) == n || _throw_cumulative_length_mismatch(length(out), n)
    return nothing
end

# Generic 1D: catches Cubic, Linear, Quadratic, and the entire Hermite family
# (PreCompute + OnTheFly via `_full_cell_fn`'s trait dispatch on `itp.dy`).
"""
    cumulative_integrate!(out, itp) -> out

In-place [`cumulative_integrate`](@ref): fills `out` with the running integral.
`out` must satisfy `length(out) == length(grid)`.
"""
function cumulative_integrate!(
        out::AbstractVector, itp::AbstractInterpolant{Tg, Tv}
    ) where {Tg <: Real, Tv}
    x = _grid_1d(itp)
    _check_cumulative_out(out, length(x))
    return _cumulative_integrate_1d!(out, x, _full_cell_fn(itp))
end

# Constant override: side is parametric → compiler knows concrete type,
# so `_full_cell_fn(itp, side)` dispatches to a distinct specialized method.
function cumulative_integrate!(
        out::AbstractVector, itp::ConstantInterpolant{Tg, Tv}
    ) where {Tg <: Real, Tv}
    x = _grid_1d(itp)
    _check_cumulative_out(out, length(x))
    return _cumulative_integrate_1d!(out, x, _full_cell_fn(itp, itp.side))
end

# Allocating wrappers: allocate output vector then forward to the in-place path.
"""
    cumulative_integrate(itp)          # persistent — Vector (Matrix for a Series)
    cumulative_integrate(x, y; method) # one-shot   — build from raw data

Running integral at every grid node: `out[i]` is the integral from the first
node up to node `i`, so `out[1] == 0` and `out[end] == integrate(itp)`. The
one-shot form builds the `method` interpolant (reference storage) first. 1-D
only — ND cumulative integration has no unambiguous definition.
"""
function cumulative_integrate(itp::AbstractInterpolant{Tg, Tv}) where {Tg <: Real, Tv}
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    out = Vector{Tout}(undef, length(_grid_1d(itp)))
    return cumulative_integrate!(out, itp)
end

# Generic Series: catches Cubic, Linear, Quadratic series
function cumulative_integrate(
        sitp::AbstractSeriesInterpolant{Tg, Tv}
    ) where {Tg <: Real, Tv}
    x = _grid_1d(sitp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    n_pts = length(x)
    n_ser = n_series(sitp)
    result = Matrix{Tout}(undef, n_pts, n_ser)
    @inbounds for k in 1:n_ser
        _cumulative_integrate_1d!(@view(result[:, k]), x, _full_cell_fn(sitp, k))
    end
    return result
end

# Constant Series override: side is parametric → compiler knows concrete type
function cumulative_integrate(
        sitp::ConstantSeriesInterpolant{Tg, Tv}
    ) where {Tg <: Real, Tv}
    x = _grid_1d(sitp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    n_pts = length(x)
    n_ser = n_series(sitp)
    result = Matrix{Tout}(undef, n_pts, n_ser)
    @inbounds for k in 1:n_ser
        _cumulative_integrate_1d!(@view(result[:, k]), x, _full_cell_fn(sitp, k, sitp.side))
    end
    return result
end

# ND override: more specific than AbstractInterpolant, throws clear error
function cumulative_integrate(itp::AbstractInterpolantND)
    throw(
        ArgumentError(
            "cumulative_integrate is not supported for $(typeof(itp)). " *
                "Only 1D interpolants and 1D series interpolants are supported."
        )
    )
end
