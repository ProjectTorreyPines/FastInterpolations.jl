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
# Forward the single-arg trait to the side-aware form so the generic
# integrate/cumulative paths serve Constant too — `side` is a type param, so this
# fully specializes with no runtime branch (no dedicated Constant overrides needed).
@inline _full_cell_fn(itp::ConstantInterpolant) = _full_cell_fn(itp, itp.side)

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
@inline _full_cell_fn(sitp::ConstantSeriesInterpolant, k::Int) = _full_cell_fn(sitp, k, sitp.side)

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
hyper-rectangle `[lo, hi]` (ND). ND covers every tensor-product interpolant —
homogeneous (`linear_interp`, `cubic_interp`, …) and heterogeneous mixes
(`interp(grids, data; method=(CubicInterp(), LinearInterp()))`); only the
Hermite family (local-slope / user-slope) has no ND integral.

The **one-shot** forms build the `method` interpolant from raw data, then
integrate in a single call, storing the input by reference where the method
allows (`copy=false`): the trivial families (Linear/Constant) never copy, the
1-D coefficient builds reference the data (Cubic copies only its grid), and the
ND `PreCompute` methods (Cubic/Quadratic) copy grids and data. 1-D integrates
every method; ND takes a single tensor-product `method`
(Linear, Cubic, Quadratic, Constant).

```julia
integrate(cubic_interp(x, y))                        # persistent, full-domain
integrate(x, y; method = CubicInterp())              # one-shot,   full-domain
integrate(x, y, 0.2, 1.5; method = LinearInterp())   # one-shot,   ∫ from 0.2 to 1.5
integrate((xs, ys), data; method = LinearInterp())   # one-shot,   2-D full-domain
```
"""
@inline function integrate(itp::AbstractInterpolant{Tg, Tv}) where {Tg <: Real, Tv}
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

# Constant and the Hermite family are handled by the generic path above —
# `_full_cell_fn` forwards Constant to its side-aware form and dispatches the
# Hermite family on `itp.dy` (PreCompute vs OnTheFly).

# ═══════════════════════════════════════════════════════════════
# integrate(sitp) — 1D Series full-domain fast path
# ═══════════════════════════════════════════════════════════════

# Generic Series: catches Cubic, Linear, Quadratic series
@inline function integrate(sitp::AbstractSeriesInterpolant{Tg, Tv}) where {Tg <: Real, Tv}
    x = _grid_1d(sitp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    n = n_series(sitp)
    results = Vector{Tout}(undef, n)
    @inbounds for k in 1:n
        results[k] = _integrate_1d_fulldomain(x, _full_cell_fn(sitp, k), Tout)
    end
    return results
end

# ═══════════════════════════════════════════════════════════════
# integrate(itp[, lo, hi]) — ND separable engine (single entry point)
# ═══════════════════════════════════════════════════════════════
#
# ── Separable ND engine: full-domain + bounded, all tensor-product families ──
#
# On full cells every per-axis basis integral collapses to a constant that is
# LINEAR in the node payload, so the domain integral is a tensor product of 1D
# node-weight rules contracted against the payload — each node read once, vs the
# 2^N corner reads per cell of the generic engine. Rank-1 axes (linear/constant)
# carry one weight per node; rank-2 axes (cubic/quadratic) a (value, deriv) pair.
# The per-axis weight functions below hold the closed forms; the @generated
# kernels pick the rank from the payload dimensionality (N vs N+1), reusing the
# method tags (`LinearInterp`/`CubicInterp`/…) or constant `sides` as the spec.

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

# Constant axes carry the side in the `ConstantInterp` method tag, so the whole
# engine speaks one vocabulary (method tags); the weight is the side's.
@inline _nd_node_weights(m::ConstantInterp, g, k::Int, n::Int) = _nd_node_weights(m.side, g, k, n)
@inline _nd_node_weights_interior(m::ConstantInterp, g, k::Int) = _nd_node_weights_interior(m.side, g, k)

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
@inline _nd_cell_weights0(m::ConstantInterp, u0, u1, h) = _nd_cell_weights0(m.side, u0, u1, h)
@inline _nd_cell_weights1(m::ConstantInterp, u0, u1, h) = _nd_cell_weights1(m.side, u0, u1, h)

# Per-axis weight rank (type-keyed for the @generated kernels; value-keyed for
# the mixed-radix payload-storage checks): 1 = value-only axis (Linear/Constant),
# 2 = (value, deriv) pair (Cubic/Quadratic). Matches `_deriv_size` in the hetero
# build, so the slot layout the kernels index is exactly the stored layout.
@inline _nd_weight_rank(::Type{<:Union{LinearInterp, ConstantInterp}}) = 1
@inline _nd_weight_rank(::Type{<:Union{CubicInterp, QuadraticInterp}}) = 2

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

# `rs[d] ∈ {1, 2}` — the per-axis weight rank: 1 for value-only axes
# (Linear/Constant), 2 for the (value, deriv) pairs (Cubic/Quadratic). Slot
# layout is the mixed-radix convention `_HeteroPartials` uses: `stride_d =
# prod(rs[1:d-1])`, so `slot = 1 + Σ_d off_d·stride_d` with the inner axis (d=1)
# varying fastest. `has_slot` (payload rank N+1 vs N) is a *storage* fact, kept
# separate from `prod(rs)`: an all-trivial hetero PreCompute payload carries a
# size-1 leading slot axis even though `prod(rs) == 1`.
function _separable_emit_common(rs::NTuple{N, Int}, has_slot::Bool) where {N}
    # Both the inner contraction here and the outer Kronecker fold in
    # `_separable_emit_outer` hard-code rank ≤ 2 (value only, or a value+deriv pair),
    # so a rank-≥3 family would silently drop slots. This helper runs first at
    # generation, so guarding here covers both — but extending to rank ≥ 3 means
    # touching both sites, not just this one.
    all(≤(2), rs) || error("separable ND integrate kernel supports per-axis weight rank ≤ 2; got ranks $rs")
    P = prod(rs)
    r1 = rs[1]
    Mout = P ÷ r1                      # outer-axis slot configurations
    ivars = [Symbol(:i_, d) for d in 1:N]
    rest = ivars[2:N]
    pidx(slot, i1) = has_slot ? :(payload[$slot, $i1, $(rest...)]) : :(payload[$i1, $(rest...)])
    wnames(ws, r) = r == 1 ? [Symbol(ws, :_1)] : [Symbol(ws, :_1), Symbol(ws, :_2)]
    wdecl(ws, r, call) = Expr(:(=), Expr(:tuple, wnames(ws, r)...), call)
    # Inner contraction for outer configuration `m`: contract the inner axis' `r1`
    # weight components against its contiguous slot pair (slots `r1·(m−1)+1 … r1·m`).
    function inner_term(m, i1, ws)
        w = wnames(ws, r1)
        base = r1 * (m - 1)
        return r1 == 2 ?
            :(muladd($(w[2]), $(pidx(base + 2, i1)), $(w[1]) * $(pidx(base + 1, i1)))) :
            :($(w[1]) * $(pidx(base + 1, i1)))
    end
    # Sum the `Mout` outer configs, each scaled by its Kronecker product `ko2_m`,
    # as a right-nested muladd chain so every `ko2_m · inner_m` fuses (FMA) rather
    # than emitting a separate multiply and add (absent when N == 1: one config).
    function contrib(ws, i1)
        N == 1 && return inner_term(1, i1, ws)
        acc = :($(Symbol(:ko2_, Mout)) * $(inner_term(Mout, i1, ws)))
        for m in (Mout - 1):-1:1
            acc = :(muladd($(Symbol(:ko2_, m)), $(inner_term(m, i1, ws)), $acc))
        end
        return acc
    end
    return ivars, rest, pidx, wdecl, contrib
end

# Outer-axis loops (i_N outermost … i_2 just outside the inner sweep). Each level
# folds axis d's rank-`rs[d]` weight components into the Kronecker partial
# products `ko{d}_m` of the outer axes d…N (axis 2 = fastest index, matching the
# slot order). Uniform `rs` reproduces the homogeneous scalar chain (all rank 1)
# or the dense 2^N Kronecker chain (all rank 2) bit-for-bit.
function _separable_emit_outer(body, rs::NTuple{N, Int}, ivars, wcall::Function, ranges::Function) where {N}
    for d in 2:N
        id = ivars[d]
        rd = rs[d]
        wc(o) = Symbol(o == 0 ? :w_ : :v_, d)
        assigns = Expr[Expr(:(=), Expr(:tuple, ntuple(o -> wc(o - 1), rd)...), wcall(d))]
        if d == N
            for o in 0:(rd - 1)
                push!(assigns, :($(Symbol(:ko, d, :_, o + 1)) = $(wc(o))))
            end
        else
            for mp in 1:prod(rs[(d + 1):N]), o in 0:(rd - 1)
                push!(
                    assigns,
                    :($(Symbol(:ko, d, :_, o + rd * (mp - 1) + 1)) = $(wc(o)) * $(Symbol(:ko, d + 1, :_, mp))),
                )
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
        payload::AbstractArray{Tv, NP}, ::Type{Tout}, z
    ) where {N, Tv, NP, Tout}
    rs = ntuple(d -> _nd_weight_rank(wspec.parameters[d]), N)
    slotoff = NP - N          # 0 = raw N-d value array, 1 = leading slot axis
    slotoff in (0, 1) || error("payload rank $NP inconsistent with N=$N")
    r1 = rs[1]
    ivars, _, _, wdecl, contrib = _separable_emit_common(rs, slotoff == 1)
    inner = quote
        $(wdecl(:w1a, r1, :(_nd_node_weights(w1, g1, 1, n_1))))
        $(wdecl(:w1b, r1, :(_nd_node_weights(w1, g1, n_1, n_1))))
        s = $(contrib(:w1a, 1)) + $(contrib(:w1b, :n_1))
        @simd for i_1 in 2:(n_1 - 1)
            $(wdecl(:w1i, r1, :(_nd_node_weights_interior(w1, g1, i_1))))
            s += $(contrib(:w1i, :i_1))
        end
        total += s
    end
    body = _separable_emit_outer(
        inner, rs, ivars,
        d -> :(_nd_node_weights(wspec[$d], grids[$d], $(ivars[d]), $(Symbol(:n_, d)))),
        d -> :(1:$(Symbol(:n_, d)))
    )
    nassign = Expr(:block, [:($(Symbol(:n_, d)) = size(payload, $(d + slotoff))) for d in 1:N]...)
    return quote
        Base.@_inline_meta
        $nassign
        g1 = grids[1]
        w1 = wspec[1]
        total = z
        @inbounds begin
            $body
        end
        return total
    end
end

@generated function _integrate_separable_nd_bounded(
        wspec::NTuple{N, Any}, specs::NTuple{N, _BoundedAxisSpec},
        grids::NTuple{N, Any}, payload::AbstractArray{Tv, NP}, ::Type{Tout}, z
    ) where {N, Tv, NP, Tout}
    rs = ntuple(d -> _nd_weight_rank(wspec.parameters[d]), N)
    slotoff = NP - N          # 0 = raw N-d value array, 1 = leading slot axis
    slotoff in (0, 1) || error("payload rank $NP inconsistent with N=$N")
    r1 = rs[1]
    ivars, _, _, wdecl, contrib = _separable_emit_common(rs, slotoff == 1)
    inner = quote
        s = z
        if ihi1 >= ilo1 + 2
            $(wdecl(:w1a, r1, :(_nd_bounded_node_weights(w1, spec1, g1, ilo1))))
            $(wdecl(:w1b, r1, :(_nd_bounded_node_weights(w1, spec1, g1, ilo1 + 1))))
            $(wdecl(:w1c, r1, :(_nd_bounded_node_weights(w1, spec1, g1, ihi1))))
            $(wdecl(:w1d, r1, :(_nd_bounded_node_weights(w1, spec1, g1, ihi1 + 1))))
            s += $(contrib(:w1a, :ilo1)) + $(contrib(:w1b, :(ilo1 + 1))) +
                $(contrib(:w1c, :ihi1)) + $(contrib(:w1d, :(ihi1 + 1)))
            @simd for i_1 in (ilo1 + 2):(ihi1 - 1)
                $(wdecl(:w1i, r1, :(_nd_node_weights_interior(w1, g1, i_1))))
                s += $(contrib(:w1i, :i_1))
            end
        else
            for i_1 in ilo1:(ihi1 + 1)
                $(wdecl(:w1x, r1, :(_nd_bounded_node_weights(w1, spec1, g1, i_1))))
                s += $(contrib(:w1x, :i_1))
            end
        end
        total += s
    end
    body = _separable_emit_outer(
        inner, rs, ivars,
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
        total = z
        @inbounds begin
            $body
        end
        return total
    end
end

# ── Single entry point. Every tensor-product ND interpolant — homogeneous or
# heterogeneous, numeric or duck-typed — routes through the separable engine.
# `_separable_spec(itp)` returns `(per-axis method-tag tuple, payload)` in the
# existing vocabulary (the method tuple the build already carries), or throws a
# clear error for the non-tensor families (local-Hermite / user-slope Hermite /
# NoInterp) that have no separable node-weight rule. ──

@inline _separable_spec(itp::LinearInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (_method_tuple(LinearInterp(), Val(N)), itp.data)
@inline _separable_spec(itp::ConstantInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (map(ConstantInterp, itp.sides), itp.data)
@inline _separable_spec(itp::CubicInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (_method_tuple(CubicInterp(), Val(N)), itp.nodal_derivs.partials)
@inline _separable_spec(itp::QuadraticInterpolantND{Tg, Tv, N}) where {Tg, Tv, N} =
    (_method_tuple(QuadraticInterp(), Val(N)), itp.nodal_derivs.partials)

# HeteroInterpolantND: the stored per-axis methods ARE the weight-tag tuple; the
# payload is the mixed-radix partials (PreCompute) or the raw data (OnTheFly /
# all-trivial). @generated so the separability + storage gate is a compile-time
# branch — no runtime method scan, no boxing.
@generated function _separable_spec(
        itp::HeteroInterpolantND{Tg, Tv, N, G, M, E, P, D}
    ) where {Tg, Tv, N, G, M, E, P, D}
    ms = (M.parameters...,)
    all(_is_separable_method_type, ms) || return :(_throw_nd_integrate_unsupported(itp))
    if D <: _HeteroPartials
        return :((itp.methods, itp.data.partials))
    elseif any(t -> _nd_weight_rank(t) == 2, ms)
        return :(_throw_hetero_precompute_required(itp.methods))
    else
        return :((itp.methods, itp.data))
    end
end

# Non-tensor ND families (CubicHermiteInterpolantND, and any future type): clean
# error instead of a MethodError deep in the kernel.
@inline _separable_spec(itp::AbstractInterpolantND) = _throw_nd_integrate_unsupported(itp)

@inline _is_separable_method_type(
    ::Type{<:Union{LinearInterp, ConstantInterp, CubicInterp, QuadraticInterp}}
) = true
@inline _is_separable_method_type(::Type{<:AbstractInterpMethod}) = false

# Duck-safe integral zero: `zero(Tout)` for numbers, `0 * sample` otherwise.
@inline _nd_int_zero(::Type{Tout}, payload) where {Tout <: Number} = zero(Tout)
@inline _nd_int_zero(::Type{Tout}, payload) where {Tout} = 0 * @inbounds(payload[begin])

@inline function integrate(itp::AbstractInterpolantND{Tg, Tv, N}) where {Tg, Tv, N}
    tags, payload = _separable_spec(itp)
    Tout = _promote_eltype(_integrate_op, Tg, Tv, Tg)
    z = _nd_int_zero(Tout, payload)
    return _integrate_separable_nd_fulldomain(tags, itp.grids, payload, Tout, z)
end

@inline function integrate(
        itp::AbstractInterpolantND{Tg, Tv, N},
        lo::Tuple{Vararg{Real, N}},
        hi::Tuple{Vararg{Real, N}};
        search = itp.searches,
        hint::Union{Nothing, NTuple{N, Base.RefValue{Int}}} = nothing
    ) where {Tg, Tv, N}
    tags, payload = _separable_spec(itp)
    sign, lo2, hi2, idx_lo, idx_hi = _integrate_nd_preamble(
        itp.grids, itp.extraps, lo, hi, search, hint
    )
    Tout = _integrate_nd_output_type(Tv, Tg, lo2, hi2)
    z = _nd_int_zero(Tout, payload)
    sign == 0 && return z
    specs = _nd_bounded_axis_specs(itp.grids, lo2, hi2, idx_lo, idx_hi)
    total = _integrate_separable_nd_bounded(tags, specs, itp.grids, payload, Tout, z)
    return sign * total
end

# Real bounds on an ND interpolant → point at the tuple form (otherwise the 1D
# `(::AbstractInterpolant, ::Real, ::Real)` stub fires with a 1D-flavoured message).
@inline function integrate(::AbstractInterpolantND, ::Real, ::Real; search = nothing, hint = nothing)
    throw(
        ArgumentError(
            "ND `integrate` needs tuple bounds — pass lo/hi as `NTuple{N,Real}`, " *
                "e.g. `integrate(itp, (a1, a2), (b1, b2))`."
        )
    )
end

@noinline function _throw_nd_integrate_unsupported(itp)
    throw(
        ArgumentError(
            "integrate is not implemented for $(nameof(typeof(itp))). ND tensor-product " *
                "integration is defined for Linear/Cubic/Quadratic/Constant axes (homogeneous " *
                "or heterogeneous). Local-Hermite (Pchip/Cardinal/Akima), user-slope Hermite, " *
                "and NoInterp axes have no separable node-weight rule — integrate axis-by-axis " *
                "via 1D `integrate` on per-fiber 1D interpolants instead."
        )
    )
end

@noinline function _throw_hetero_precompute_required(methods)
    throw(
        ArgumentError(
            "integrate over a HeteroInterpolantND with a derivative axis (Cubic/Quadratic) " *
                "requires precomputed partials: rebuild with `coeffs=PreCompute()` " *
                "(method tuple: $(methods)). The OnTheFly build stores only raw data, " *
                "which has no derivative slots."
        )
    )
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

# ND override: more specific than AbstractInterpolant, throws clear error
function cumulative_integrate(itp::AbstractInterpolantND)
    throw(
        ArgumentError(
            "cumulative_integrate is not supported for $(typeof(itp)). " *
                "Only 1D interpolants and 1D series interpolants are supported."
        )
    )
end
