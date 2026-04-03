# ========================================
# AkimaAdjoint1D: Adjoint (Transpose) Operator
# ========================================
#
# Computes f_bar = W^T * y_bar where W is the implicit forward Akima
# interpolation matrix.
#
# Akima forward pipeline:
#   1. Slope computation:  dy[k] = akima_slopes(y, x, k)  (Akima 1970)
#   2. Hermite evaluation: P(t) = h00*yL + h10*h*dyL + h01*yR + h11*h*dyR
#
# Adjoint pipeline reverses both stages:
#   1. Hermite scatter -> (f_bar_direct, dy_bar)  [shared core from hermite_adjoint.jl]
#   2. Akima slope J^T * dy_bar -> f_bar_slope    [Akima-specific, this file]
#   3. Final: f_bar_total = f_bar_direct + f_bar_slope
#
# Unlike Cardinal (where slopes are linear in y everywhere), Akima has
# branch conditions (|Δm| weights) that depend on the data y. The adjoint
# is computed at the specific y passed to the constructor (linearization
# at that operating point). The dot-product identity holds exactly at
# the given y.
#
# Dependencies (already included before this file):
# - _HermiteAdjointAnchor1D, _bake_hermite_adjoint_anchors (hermite_adjoint.jl)
# - _scatter_hermite_adjoint! (hermite_adjoint.jl)
# - AbstractAdjoint1D, _throw_adjoint_grid_too_small (adjoint_protocol.jl)
# - _promote_grid_float, _to_float (promotion helpers)
# - _create_spacing (grid_spacing.jl)

# ========================================
# AkimaAdjoint1D Struct
# ========================================

"""
    AkimaAdjoint1D{Tg, Tv, EP}

Adjoint (transpose) operator for Akima interpolation.
Computes `f_bar = W^T * y_bar` where `W` is the forward Akima interpolation
weight matrix, including the slope-from-data dependence.

Because Akima slopes involve data-dependent branching (absolute-value weights
on secant differences), this adjoint is constructed at a specific `y`
operating point. The dot-product identity holds exactly at that `y`:

    dot(akima_interp(x, y).(xq), y_bar) == dot(y, adj(y_bar))

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `Tv`: Value type (must support sign, abs, zero, division)
- `EP`: Extrapolation policy type

# Usage
```julia
adj = akima_adjoint(x, y, xq)

f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # derivative adjoint
adj(f_bar, y_bar)                       # in-place
```
"""
struct AkimaAdjoint1D{Tg <: AbstractFloat, Tv <: Real, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_HermiteAdjointAnchor1D{Tg}}
    grid::Vector{Tg}       # Grid points (needed for slope adjoint secant computation)
    data::Vector{Tv}       # y values (needed for slope weight conditions)
    grid_size::Int
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================

@inline _n_queries(adj::AkimaAdjoint1D) = length(adj.anchors)
@inline _adjoint_output_length(adj::AkimaAdjoint1D) = adj.grid_size

# ========================================
# Akima Slope Adjoint: J^T * dy_bar -> f_bar update
# ========================================
#
# Akima slopes depend on data y through secants m[k] = (y[k+1]-y[k])/h[k].
# The slope formula uses |Δm| weights:
#   w1 = |m[k+1] - m[k]|, w2 = |m[k-1] - m[k-2]|
#   dy[k] = (w1*m[k-1] + w2*m[k]) / (w1+w2)  (or fallback to average)
#
# For each point k, the transpose scatters dy_bar[k] to f_bar[j] for j in stencil.
#
# The adjoint chains:  ∂dy/∂m_j  ×  ∂m_j/∂y_i  ×  dy_bar[k]
#
# Derivatives of dy w.r.t. the 4 secants (m_km2, m_km1, m_k, m_kp1):
#
# Let s1 = sign(m_kp1 - m_k), s2 = sign(m_km1 - m_km2).
#
# If wsum > 0 (weighted branch):
#   ∂dy/∂m_km2 = s2*(dy - m_k) / wsum
#   ∂dy/∂m_km1 = (w1 + s2*(m_k - dy)) / wsum
#   ∂dy/∂m_k   = (w2 + s1*(dy - m_km1)) / wsum
#   ∂dy/∂m_kp1 = s1*(m_km1 - dy) / wsum
#
# If wsum == 0 (equal-weight fallback):
#   ∂dy/∂m_km1 = 1/2, ∂dy/∂m_k = 1/2, others = 0

"""
    _akima_slope_adjoint_weighted(dy, m_km1, m_k, w1, w2, wsum, s1, s2)

Compute ∂dy/∂m for the weighted-average Akima slope branch.
Returns (∂dy/∂m_km2, ∂dy/∂m_km1, ∂dy/∂m_k, ∂dy/∂m_kp1).
"""
@inline function _akima_slope_adjoint_weighted(dy, m_km1, m_k, w1, w2, wsum, s1, s2)
    inv_wsum = inv(wsum)
    ddy_dm_km2 = s2 * (dy - m_k) * inv_wsum
    ddy_dm_km1 = (w1 + s2 * (m_k - dy)) * inv_wsum
    ddy_dm_k = (w2 + s1 * (dy - m_km1)) * inv_wsum
    ddy_dm_kp1 = s1 * (m_km1 - dy) * inv_wsum
    return ddy_dm_km2, ddy_dm_km1, ddy_dm_k, ddy_dm_kp1
end

"""
    _akima_scatter_secant_adjoint!(f_bar, coeff, m_idx, x, n)

Scatter `coeff` (= dy_bar[k] * ∂dy/∂m[j]) through the secant m[j] = (y[j+1]-y[j])/h[j]
to f_bar. Handles virtual secant indices (j < 1 or j > n-1) via the boundary
extrapolation formulas.

Secant index convention: m[j] is the secant of interval [j, j+1], for j=1..n-1.
Virtual secants for j=0,-1,n,n+1 are linear combinations of real secants.
"""
@inline function _akima_scatter_secant_adjoint!(
        f_bar::AbstractVector, coeff, m_idx::Int,
        x::AbstractVector{Tg}, n::Int
    ) where {Tg}
    # Real secant: m[j] = (y[j+1] - y[j]) / h[j]
    # ∂m[j]/∂y[j] = -1/h[j], ∂m[j]/∂y[j+1] = +1/h[j]
    if 1 <= m_idx <= n - 1
        @inbounds begin
            inv_h = one(Tg) / (x[m_idx + 1] - x[m_idx])
            c = coeff * inv_h
            f_bar[m_idx] -= c
            f_bar[m_idx + 1] += c
        end
    elseif m_idx == 0
        # m[0] = 2*m[1] - m[2]
        # ∂/∂y via m[1]: coeff * 2 * ∂m[1]/∂y
        # ∂/∂y via m[2]: coeff * (-1) * ∂m[2]/∂y
        @inbounds begin
            inv_h1 = one(Tg) / (x[2] - x[1])
            inv_h2 = one(Tg) / (x[3] - x[2])
            c1 = coeff * 2 * inv_h1
            c2 = coeff * inv_h2
            f_bar[1] -= c1
            f_bar[2] += c1
            f_bar[2] += c2
            f_bar[3] -= c2
        end
    elseif m_idx == -1
        # m[-1] = 3*m[1] - 2*m[2]
        @inbounds begin
            inv_h1 = one(Tg) / (x[2] - x[1])
            inv_h2 = one(Tg) / (x[3] - x[2])
            c1 = coeff * 3 * inv_h1
            c2 = coeff * 2 * inv_h2
            f_bar[1] -= c1
            f_bar[2] += c1
            f_bar[2] += c2
            f_bar[3] -= c2
        end
    elseif m_idx == n
        # m[n] = 2*m[n-1] - m[n-2]
        @inbounds begin
            inv_hn1 = one(Tg) / (x[n] - x[n - 1])
            inv_hn2 = one(Tg) / (x[n - 1] - x[n - 2])
            c1 = coeff * 2 * inv_hn1
            c2 = coeff * inv_hn2
            f_bar[n - 1] -= c1
            f_bar[n] += c1
            f_bar[n - 2] += c2
            f_bar[n - 1] -= c2
        end
    elseif m_idx == n + 1
        # m[n+1] = 3*m[n-1] - 2*m[n-2]
        @inbounds begin
            inv_hn1 = one(Tg) / (x[n] - x[n - 1])
            inv_hn2 = one(Tg) / (x[n - 1] - x[n - 2])
            c1 = coeff * 3 * inv_hn1
            c2 = coeff * 2 * inv_hn2
            f_bar[n - 1] -= c1
            f_bar[n] += c1
            f_bar[n - 2] += c2
            f_bar[n - 1] -= c2
        end
    end
    return nothing
end

@inline function _akima_slope_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        x::AbstractVector{Tg}, y::AbstractVector{Tv}
    ) where {Tg, Tv}
    n = length(x)

    # Special case: n=2 → dy[1] = dy[2] = δ = (y[2]-y[1])/(x[2]-x[1])
    if n == 2
        @inbounds begin
            inv_h = one(Tg) / (x[2] - x[1])
            c1 = Tv(inv_h) * dy_bar[1]
            c2 = Tv(inv_h) * dy_bar[2]
            f_bar[1] -= c1 + c2
            f_bar[2] += c1 + c2
        end
        return nothing
    end

    # Special case: n=3 → dy[1] = m[1], dy[2] = (m[1]+m[2])/2, dy[3] = m[2]
    if n == 3
        @inbounds begin
            inv_h1 = one(Tg) / (x[2] - x[1])
            inv_h2 = one(Tg) / (x[3] - x[2])

            # dy[1] = m[1]: ∂/∂y[1] = -1/h1, ∂/∂y[2] = +1/h1
            c1 = Tv(inv_h1) * dy_bar[1]
            f_bar[1] -= c1
            f_bar[2] += c1

            # dy[2] = (m[1]+m[2])/2: ∂/∂m[1] = 1/2, ∂/∂m[2] = 1/2
            db2 = dy_bar[2]
            c2a = Tv(inv_h1 / 2) * db2
            f_bar[1] -= c2a
            f_bar[2] += c2a
            c2b = Tv(inv_h2 / 2) * db2
            f_bar[2] -= c2b
            f_bar[3] += c2b

            # dy[3] = m[2]: ∂/∂y[2] = -1/h2, ∂/∂y[3] = +1/h2
            c3 = Tv(inv_h2) * dy_bar[3]
            f_bar[2] -= c3
            f_bar[3] += c3
        end
        return nothing
    end

    # General case: n ≥ 4
    # Recompute secants on the fly (same approach as forward pass)
    @inbounds m1 = Tv((y[2] - y[1]) / (x[2] - x[1]))
    @inbounds m2 = Tv((y[3] - y[2]) / (x[3] - x[2]))
    @inbounds m3 = Tv((y[4] - y[3]) / (x[4] - x[3]))

    # Virtual secants for left boundary
    m_neg1 = 3 * m1 - 2 * m2
    m_0 = 2 * m1 - m2

    # ── Point k=1: uses m[-1], m[0], m[1], m[2] ─────────────────────
    # Secant indices: km2=-1, km1=0, k=1, kp1=2
    @inbounds begin
        w1 = abs(m2 - m1)
        w2 = abs(m_0 - m_neg1)
        wsum = w1 + w2
        if wsum == zero(wsum)
            # Fallback: dy = (m_0 + m1)/2 → ∂/∂m_0 = 1/2, ∂/∂m1 = 1/2
            db = dy_bar[1]
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, 0, x, n)
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, 1, x, n)
        else
            dy_k = (w1 * m_0 + w2 * m1) / wsum
            s1 = sign(m2 - m1)
            s2 = sign(m_0 - m_neg1)
            d_km2, d_km1, d_k, d_kp1 = _akima_slope_adjoint_weighted(
                dy_k, m_0, m1, w1, w2, wsum, s1, s2
            )
            db = dy_bar[1]
            _akima_scatter_secant_adjoint!(f_bar, d_km2 * db, -1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_km1 * db, 0, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_k * db, 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_kp1 * db, 2, x, n)
        end
    end

    # ── Point k=2: uses m[0], m[1], m[2], m[3] ──────────────────────
    @inbounds begin
        w1 = abs(m3 - m2)
        w2 = abs(m1 - m_0)
        wsum = w1 + w2
        if wsum == zero(wsum)
            db = dy_bar[2]
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, 2, x, n)
        else
            dy_k = (w1 * m1 + w2 * m2) / wsum
            s1 = sign(m3 - m2)
            s2 = sign(m1 - m_0)
            d_km2, d_km1, d_k, d_kp1 = _akima_slope_adjoint_weighted(
                dy_k, m1, m2, w1, w2, wsum, s1, s2
            )
            db = dy_bar[2]
            _akima_scatter_secant_adjoint!(f_bar, d_km2 * db, 0, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_km1 * db, 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_k * db, 2, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_kp1 * db, 3, x, n)
        end
    end

    # ── Interior points k=3..n-2: uses m[k-2], m[k-1], m[k], m[k+1] ─
    # Rolling window from the forward pass
    m_km2 = m1
    m_km1 = m2
    m_k = m3

    @inbounds for k in 3:(n - 2)
        m_kp1 = Tv((y[k + 2] - y[k + 1]) / (x[k + 2] - x[k + 1]))
        w1 = abs(m_kp1 - m_k)
        w2 = abs(m_km1 - m_km2)
        wsum = w1 + w2
        if wsum == zero(wsum)
            db = dy_bar[k]
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, k - 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, k, x, n)
        else
            dy_kv = (w1 * m_km1 + w2 * m_k) / wsum
            s1 = sign(m_kp1 - m_k)
            s2 = sign(m_km1 - m_km2)
            d_km2, d_km1, d_k, d_kp1 = _akima_slope_adjoint_weighted(
                dy_kv, m_km1, m_k, w1, w2, wsum, s1, s2
            )
            db = dy_bar[k]
            _akima_scatter_secant_adjoint!(f_bar, d_km2 * db, k - 2, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_km1 * db, k - 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_k * db, k, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_kp1 * db, k + 1, x, n)
        end

        # Advance window
        m_km2 = m_km1
        m_km1 = m_k
        m_k = m_kp1
    end

    # After loop: m_km2 = m[n-3], m_km1 = m[n-2], m_k = m[n-1]
    # Virtual secants for right boundary
    m_np1 = 2 * m_k - m_km1       # m[n] = 2*m[n-1] - m[n-2]
    m_np2 = 3 * m_k - 2 * m_km1   # m[n+1] = 3*m[n-1] - 2*m[n-2]

    # ── Point k=n-1: uses m[n-3], m[n-2], m[n-1], m[n] ──────────────
    @inbounds begin
        w1 = abs(m_np1 - m_k)
        w2 = abs(m_km1 - m_km2)
        wsum = w1 + w2
        if wsum == zero(wsum)
            db = dy_bar[n - 1]
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, n - 2, x, n)
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, n - 1, x, n)
        else
            dy_kv = (w1 * m_km1 + w2 * m_k) / wsum
            s1 = sign(m_np1 - m_k)
            s2 = sign(m_km1 - m_km2)
            d_km2, d_km1, d_k, d_kp1 = _akima_slope_adjoint_weighted(
                dy_kv, m_km1, m_k, w1, w2, wsum, s1, s2
            )
            db = dy_bar[n - 1]
            _akima_scatter_secant_adjoint!(f_bar, d_km2 * db, n - 3, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_km1 * db, n - 2, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_k * db, n - 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_kp1 * db, n, x, n)
        end
    end

    # ── Point k=n: uses m[n-2], m[n-1], m[n], m[n+1] ────────────────
    @inbounds begin
        w1 = abs(m_np2 - m_np1)
        w2 = abs(m_k - m_km1)
        wsum = w1 + w2
        if wsum == zero(wsum)
            db = dy_bar[n]
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, n - 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, Tv(1 // 2) * db, n, x, n)
        else
            dy_kv = (w1 * m_k + w2 * m_np1) / wsum
            s1 = sign(m_np2 - m_np1)
            s2 = sign(m_k - m_km1)
            d_km2, d_km1, d_k, d_kp1 = _akima_slope_adjoint_weighted(
                dy_kv, m_k, m_np1, w1, w2, wsum, s1, s2
            )
            db = dy_bar[n]
            _akima_scatter_secant_adjoint!(f_bar, d_km2 * db, n - 2, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_km1 * db, n - 1, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_k * db, n, x, n)
            _akima_scatter_secant_adjoint!(f_bar, d_kp1 * db, n + 1, x, n)
        end
    end

    return nothing
end

# ========================================
# Core Apply Function
# ========================================

@inline _adjoint_1d_apply!(f_bar, adj::AkimaAdjoint1D, y_bar, deriv) =
    _akima_adjoint_apply!(f_bar, adj, y_bar, deriv)

@with_pool pool function _akima_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::AkimaAdjoint1D{Tg},
        y_bar,
        deriv::DerivOp = EvalValue()
    ) where {Tv, Tg}
    n = adj.grid_size
    dy_bar = zeros!(pool, Tv, n)

    # Step 1: Hermite scatter -> (f_bar, dy_bar)
    _scatter_hermite_adjoint!(f_bar, dy_bar, adj.anchors, y_bar, deriv)

    # Step 2: Akima slope J^T * dy_bar -> f_bar update
    _akima_slope_adjoint!(f_bar, dy_bar, adj.grid, adj.data)

    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    akima_adjoint(x, y, x_query; extrap=NoExtrap()) -> AkimaAdjoint1D

Create an Akima adjoint operator (query-baked, linearized at data `y`).

Computes `f_bar = W^T * y_bar` where `W` is the forward Akima interpolation
weight matrix, including the slope-from-data dependence.

Because Akima slopes involve data-dependent branching (absolute-value weights),
the adjoint is linearized at the given `y`. The dot-product identity holds
exactly at this operating point:

    dot(akima_interp(x, y).(xq), y_bar) == dot(y, akima_adjoint(x, y, xq)(y_bar))

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `y::AbstractVector`: Data values (determines slope weight conditions)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
y = randn(50)
xq = sort(rand(30))
y_bar = randn(30)

itp = akima_interp(x, y)
adj = akima_adjoint(x, y, xq)
f_bar = adj(y_bar)

# Dot-product identity: <W*y, y_bar> = <y, W^T*y_bar>
@assert dot(itp.(xq), y_bar) ≈ dot(y, adj(y_bar))
```
"""
function akima_adjoint(
        x::AbstractVector,
        y::AbstractVector,
        x_query::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    Tg = _promote_grid_float(eltype(x), eltype(x_query))
    x_p = _to_float(x, Tg)
    xq_p = _to_float(x_query, Tg)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # NoExtrap: validate all queries in-domain
    if extrap isa NoExtrap
        x_lo, x_hi = first(x_p), last(x_p)
        @inbounds for i in eachindex(xq_p)
            xq_i = xq_p[i]
            (x_lo <= xq_i <= x_hi) || throw(
                DomainError(xq_i, "query point outside domain [$x_lo, $x_hi]")
            )
        end
    end

    # Build spacing and anchors
    spacing = _create_spacing(x_p)
    anchors = _bake_hermite_adjoint_anchors(x_p, spacing, xq_p, extrap)

    Tv = eltype(y)
    return AkimaAdjoint1D{Tg, Tv, typeof(extrap)}(
        anchors, collect(x_p), collect(Tv, y), length(x_p), extrap
    )
end

# Scalar query convenience
function akima_adjoint(
        x::AbstractVector,
        y::AbstractVector,
        x_query::Real;
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    return akima_adjoint(x, y, [x_query]; extrap = extrap)
end
