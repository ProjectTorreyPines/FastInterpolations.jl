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

# ── Linear/Constant ND full-domain: separable node-weighted path ──
#
# On full cells the per-axis basis weights collapse to constants (multilinear →
# h/2 each end; constant → side-selected h), so the whole domain integral is a
# tensor product of 1D node-weight rules:  I = Σ_nodes (Π_d w_d(i_d))·data[i].
# Iterating nodes (not cells) reads each value once instead of the 2^N corner
# reads the generic per-cell kernel repeats — an N-D generalization of the 1D
# telescoped path.

# 1D composite-trapezoid node weight on `grid` (length `n`): ½h at the endpoints,
# ½(h₋+h₊) in the interior. Only the grid is divided — values stay multiply-only.
@inline _nd_trap_weight_interior(grid, k::Int) = (_get_h(grid, k - 1) + _get_h(grid, k)) / 2
@inline function _nd_trap_weight(grid, k::Int, n::Int)
    k == 1 && return _get_h(grid, 1) / 2
    k == n && return _get_h(grid, n - 1) / 2
    return _nd_trap_weight_interior(grid, k)
end

# Per-axis weight provider: `nothing` = trapezoid (linear); a side = the constant
# family's collapsed weights (NearestSide splits each cell at h/2 — identical to
# the trapezoid rule; Left/Right put the whole h on the selected node).
@inline _nd_node_weight(::Union{Nothing, NearestSide}, g, k::Int, n::Int) = _nd_trap_weight(g, k, n)
@inline _nd_node_weight_interior(::Union{Nothing, NearestSide}, g, k::Int) = _nd_trap_weight_interior(g, k)
@inline _nd_node_weight(::LeftSide, g, k::Int, n::Int) = k == n ? zero(_get_h(g, 1)) : _get_h(g, k)
@inline _nd_node_weight_interior(::LeftSide, g, k::Int) = _get_h(g, k)
@inline _nd_node_weight(::RightSide, g, k::Int, n::Int) = k == 1 ? zero(_get_h(g, 1)) : _get_h(g, k - 1)
@inline _nd_node_weight_interior(::RightSide, g, k::Int) = _get_h(g, k - 1)

# Nested reduction: outer axes hoist their weight product `wp`, the contiguous
# inner axis peels its endpoints so the interior is a branch-free @simd loop.
# `wspec` = per-axis weight provider tuple (see `_nd_node_weight`).
@generated function _integrate_separable_nd_fulldomain(
        wspec::NTuple{N, Any}, grids::NTuple{N, Any}, data::AbstractArray, ::Type{Tout}
    ) where {N, Tout}
    ivars = [Symbol(:i_, d) for d in 1:N]
    rest = ivars[2:N]
    inner = quote
        s = _nd_node_weight(w1, g1, 1, n_1) * data[1, $(rest...)] +
            _nd_node_weight(w1, g1, n_1, n_1) * data[n_1, $(rest...)]
        @simd for i_1 in 2:(n_1 - 1)
            s += _nd_node_weight_interior(w1, g1, i_1) * data[i_1, $(rest...)]
        end
        total += wp_2 * s
    end
    body = inner
    for d in 2:N
        id = ivars[d]
        nd = Symbol(:n_, d)
        wexpr = d == N ? :(_nd_node_weight(wspec[$d], grids[$d], $id, $nd)) :
            :(_nd_node_weight(wspec[$d], grids[$d], $id, $nd) * $(Symbol(:wp_, d + 1)))
        body = quote
            for $id in 1:$nd
                $(Symbol(:wp_, d)) = $wexpr
                $body
            end
        end
    end
    nassign = Expr(:block, [:($(Symbol(:n_, d)) = size(data, $d)) for d in 1:N]...)
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

# LinearInterpolantND with numeric values: the separable path. Non-numeric (duck)
# value types fall through to the generic per-cell method below.
@inline function integrate(
        itp::LinearInterpolantND{Tg, Tv, N};
        search = nothing,
        hint = nothing
    ) where {Tg, Tv <: Number, N}
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    return _integrate_separable_nd_fulldomain(ntuple(_ -> nothing, Val(N)), itp.grids, itp.data, Tout)
end

# ConstantInterpolantND with numeric values: same separable engine, per-axis side
# weights (mixed sides compose axis-wise). Duck values fall through to generic.
@inline function integrate(
        itp::ConstantInterpolantND{Tg, Tv, N};
        search = nothing,
        hint = nothing
    ) where {Tg, Tv <: Number, N}
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    return _integrate_separable_nd_fulldomain(itp.sides, itp.grids, itp.data, Tout)
end

# ── Bounded ND (linear/constant): separable clipped-composite path ──
#
# Separability holds on any axis-aligned box, not just the full domain: the
# per-axis node weights become "clipped composites" — zero outside the covered
# cells, the partial-cell integral weights at the two boundary cells, and the
# full-cell weights in between — so the sum visits the node sub-box once with
# no per-cell clip geometry. Bounds/sign/extrap semantics come from the shared
# `_integrate_nd_preamble`, identical to the generic cell-wise engine.

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

# Partial-cell end-weight pair through the provider (`nothing` = linear basis,
# a side = the constant selection weights) — full cells reduce to the closed forms.
@inline _nd_cell_w0(::Nothing, u0, u1, h) = _w0_int(u0, u1, h)
@inline _nd_cell_w1(::Nothing, u0, u1, h) = _w1_int(u0, u1, h)
@inline _nd_cell_w0(s::AbstractSide, u0, u1, h) = _cw0(u0, u1, h, s)
@inline _nd_cell_w1(s::AbstractSide, u0, u1, h) = _cw1(u0, u1, h, s)

# Clipped composite node weight: right-end share of cell k−1 plus left-end share
# of cell k, each clipped to the covered range. Only the ≤ 4 nodes touching the
# two boundary cells differ from the full-domain weights.
@inline function _nd_bounded_node_weight(w, spec::_BoundedAxisSpec, g, k::Int)
    total = zero(spec.u0)
    c = k - 1
    if spec.ilo <= c <= spec.ihi
        h = _get_h(g, c)
        total += _nd_cell_w1(w, c == spec.ilo ? spec.u0 : zero(h), c == spec.ihi ? spec.u1 : h, h)
    end
    c = k
    if spec.ilo <= c <= spec.ihi
        h = _get_h(g, c)
        total += _nd_cell_w0(w, c == spec.ilo ? spec.u0 : zero(h), c == spec.ihi ? spec.u1 : h, h)
    end
    return total
end

# Nested reduction over the node sub-box. Wide inner spans peel the four
# boundary-cell nodes so the interior runs the branch-free full-cell weights
# under @simd; narrow spans (≤ 2 cells) take the scalar clipped loop.
@generated function _integrate_separable_nd_bounded(
        wspec::NTuple{N, Any}, specs::NTuple{N, _BoundedAxisSpec},
        grids::NTuple{N, Any}, data::AbstractArray, ::Type{Tout}
    ) where {N, Tout}
    ivars = [Symbol(:i_, d) for d in 1:N]
    rest = ivars[2:N]
    inner = quote
        s = zero(Tout)
        if ihi1 >= ilo1 + 2
            s += _nd_bounded_node_weight(w1, spec1, g1, ilo1) * data[ilo1, $(rest...)] +
                _nd_bounded_node_weight(w1, spec1, g1, ilo1 + 1) * data[ilo1 + 1, $(rest...)] +
                _nd_bounded_node_weight(w1, spec1, g1, ihi1) * data[ihi1, $(rest...)] +
                _nd_bounded_node_weight(w1, spec1, g1, ihi1 + 1) * data[ihi1 + 1, $(rest...)]
            @simd for i_1 in (ilo1 + 2):(ihi1 - 1)
                s += _nd_node_weight_interior(w1, g1, i_1) * data[i_1, $(rest...)]
            end
        else
            for i_1 in ilo1:(ihi1 + 1)
                s += _nd_bounded_node_weight(w1, spec1, g1, i_1) * data[i_1, $(rest...)]
            end
        end
        total += wp_2 * s
    end
    body = inner
    for d in 2:N
        id = ivars[d]
        wexpr = d == N ? :(_nd_bounded_node_weight(wspec[$d], specs[$d], grids[$d], $id)) :
            :(_nd_bounded_node_weight(wspec[$d], specs[$d], grids[$d], $id) * $(Symbol(:wp_, d + 1)))
        body = quote
            for $id in specs[$d].ilo:(specs[$d].ihi + 1)
                $(Symbol(:wp_, d)) = $wexpr
                $body
            end
        end
    end
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

# LinearInterpolantND bounded, numeric values: separable clipped path. Duck
# values keep the generic per-cell method in integrate_api.jl.
@inline function integrate(
        itp::LinearInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv <: Number, N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    sign == 0 && return zero(Tout)
    specs = _nd_bounded_axis_specs(itp.grids, lo2, hi2, idx_lo, idx_hi)
    total = _integrate_separable_nd_bounded(
        ntuple(_ -> nothing, Val(N)), specs, itp.grids, itp.data, Tout
    )
    return sign * total
end

# ConstantInterpolantND bounded, numeric values: same path with side weights.
@inline function integrate(
        itp::ConstantInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv <: Number, N}
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    sign == 0 && return zero(Tout)
    specs = _nd_bounded_axis_specs(itp.grids, lo2, hi2, idx_lo, idx_hi)
    total = _integrate_separable_nd_bounded(itp.sides, specs, itp.grids, itp.data, Tout)
    return sign * total
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
