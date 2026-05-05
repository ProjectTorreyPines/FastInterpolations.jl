# ========================================
# PchipAdjoint1D: Adjoint (Transpose) Operator
# ========================================
#
# Computes f_bar = W^T * y_bar where W is the implicit forward PCHIP
# interpolation matrix.
#
# PCHIP forward pipeline:
#   1. Slope computation:  dy[k] = pchip_slopes(y, x, k)  (Fritsch-Carlson)
#   2. Hermite evaluation: P(t) = h00*yL + h10*h*dyL + h01*yR + h11*h*dyR
#
# Adjoint pipeline reverses both stages:
#   1. Hermite scatter -> (f_bar_direct, dy_bar)  [shared core from hermite_adjoint.jl]
#   2. PCHIP slope J^T * dy_bar -> f_bar_slope    [PCHIP-specific, this file]
#   3. Final: f_bar_total = f_bar_direct + f_bar_slope
#
# Unlike Cardinal (where slopes are linear in y everywhere), PCHIP has
# branch conditions (sign checks) that depend on the data y. The adjoint
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
# PchipAdjoint1D Struct
# ========================================

"""
    PchipAdjoint1D{Tg, Tv, EP}

Adjoint (transpose) operator for PCHIP (Fritsch-Carlson) interpolation.
Computes `f_bar = W^T * y_bar` where `W` is the forward PCHIP interpolation
weight matrix, including the slope-from-data dependence.

Because PCHIP slopes involve data-dependent branching (sign checks for
monotonicity clamping), this adjoint is constructed at a specific `y`
operating point. The dot-product identity holds exactly at that `y`:

    dot(pchip_interp(x, y).(xq), y_bar) == dot(y, adj(y_bar))

# Type Parameters
- `Tg`: Grid type — normally Float32/Float64, unconstrained for duck-typed grids (e.g. ForwardDiff.Dual)
- `Tv`: Value type (must support sign, abs, zero, division)
- `EP`: Extrapolation policy type

# Usage
```julia
adj = pchip_adjoint(x, y, xq)

f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # derivative adjoint
adj(f_bar, y_bar)                       # in-place
```
"""
struct PchipAdjoint1D{Tg, Tv <: Real, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_HermiteAdjointAnchor1D{Tg}}
    grid::Vector{Tg}       # Grid points (needed for slope adjoint stencil widths)
    data::Vector{Tv}       # y values (needed for slope clamp conditions)
    grid_size::Int
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================

@inline _n_queries(adj::PchipAdjoint1D) = length(adj.anchors)
@inline _adjoint_output_length(adj::PchipAdjoint1D) = adj.grid_size

# ========================================
# PCHIP Slope Adjoint: J^T * dy_bar -> f_bar update
# ========================================
#
# PCHIP slopes depend on data y through secants δ[k] = (y[k+1]-y[k])/h[k].
# The slope formula has branches:
#   - Interior: clamped (dy=0) when sign(δ[k-1]) != sign(δ[k])
#   - Interior: active — weighted harmonic mean otherwise
#   - Endpoints: unclamped / zero-clamped / sat-clamped (3 branches)
#
# For each point k, the transpose scatters dy_bar[k] to f_bar[j] for j in stencil.

@inline function _pchip_slope_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        x::AbstractVector{Tg}, y::AbstractVector{Tv}
    ) where {Tg, Tv}
    n = length(x)

    # Special case: n=2 → dy[1] = dy[2] = δ = (y[2]-y[1])/(x[2]-x[1])
    # ∂dy[1]/∂y[1] = -1/h,  ∂dy[1]/∂y[2] = +1/h  (same for dy[2])
    if n == 2
        @inbounds begin
            inv_h = one(Tg) / (x[2] - x[1])
            c1 = inv_h * dy_bar[1]
            c2 = inv_h * dy_bar[2]
            f_bar[1] -= c1 + c2
            f_bar[2] += c1 + c2
        end
        return nothing
    end

    # Recompute secants for all intervals (O(n), no allocation needed beyond stack)
    # We'll do a single forward pass, maintaining running secant values.

    @inbounds h_prev = x[2] - x[1]
    @inbounds δ_prev = (y[2] - y[1]) / h_prev
    @inbounds h_curr = x[3] - x[2]
    @inbounds δ_curr = (y[3] - y[2]) / h_curr

    # ── Left endpoint (k=1) ──────────────────────────────────────────────
    # d = ((2h1+h2)*δ1 - h1*δ2) / (h1+h2)
    # Then branch: zero-clamped / sat-clamped / unclamped
    @inbounds begin
        h1 = h_prev
        h2 = h_curr
        δ1 = δ_prev
        δ2 = δ_curr
        d_left = ((2 * h1 + h2) * δ1 - h1 * δ2) / (h1 + h2)

        if sign(d_left) != sign(δ1)
            # Zero-clamped: dy[1] = 0, all derivatives zero → skip
        elseif sign(δ1) != sign(δ2) && abs(d_left) > abs(3 * δ1)
            # Sat-clamped: dy[1] = 3*δ1
            # ∂dy[1]/∂δ1 = 3, ∂dy[1]/∂δ2 = 0
            # δ1 = (y[2]-y[1])/h1 → ∂δ1/∂y[1] = -1/h1, ∂δ1/∂y[2] = +1/h1
            c = (3 / h1) * dy_bar[1]
            f_bar[1] -= c
            f_bar[2] += c
        else
            # Unclamped: dy[1] = ((2h1+h2)*δ1 - h1*δ2) / (h1+h2)
            # ∂dy[1]/∂δ1 = (2h1+h2)/(h1+h2)
            # ∂dy[1]/∂δ2 = -h1/(h1+h2)
            ddy_dδ1 = (2 * h1 + h2) / (h1 + h2)
            ddy_dδ2 = -h1 / (h1 + h2)
            db = dy_bar[1]
            # δ1: y[1] → -1/h1, y[2] → +1/h1
            c1 = (ddy_dδ1 / h1) * db
            f_bar[1] -= c1
            f_bar[2] += c1
            # δ2: y[2] → -1/h2, y[3] → +1/h2
            c2 = (ddy_dδ2 / h2) * db
            f_bar[2] -= c2
            f_bar[3] += c2
        end
    end

    # ── Interior slopes (k=2..n-1) ──────────────────────────────────────
    # Reset running secants
    @inbounds h_prev = x[2] - x[1]
    @inbounds δ_prev = (y[2] - y[1]) / h_prev
    @inbounds h_curr = x[3] - x[2]
    @inbounds δ_curr = (y[3] - y[2]) / h_curr

    @inbounds for k in 2:(n - 1)
        if sign(δ_prev) != sign(δ_curr)
            # Clamped: dy[k] = 0 → all derivatives zero, skip
        else
            # Active: weighted harmonic mean
            # dy[k] = S / D where S = w1+w2, D = w1/δ_prev + w2/δ_curr
            w1 = 2 * h_curr + h_prev
            w2 = h_curr + 2 * h_prev
            S = w1 + w2
            D = w1 / δ_prev + w2 / δ_curr
            D2 = D * D

            # ∂dy/∂δ_prev = S * w1 / (D² * δ_prev²)
            # ∂dy/∂δ_curr = S * w2 / (D² * δ_curr²)
            ddy_dδ_prev = S * w1 / (D2 * δ_prev * δ_prev)
            ddy_dδ_curr = S * w2 / (D2 * δ_curr * δ_curr)

            db = dy_bar[k]
            # δ_prev is secant of interval [k-1, k]: ∂/∂y[k-1] = -1/h_prev, ∂/∂y[k] = +1/h_prev
            c_prev = (ddy_dδ_prev / h_prev) * db
            f_bar[k - 1] -= c_prev
            f_bar[k] += c_prev

            # δ_curr is secant of interval [k, k+1]: ∂/∂y[k] = -1/h_curr, ∂/∂y[k+1] = +1/h_curr
            c_curr = (ddy_dδ_curr / h_curr) * db
            f_bar[k] -= c_curr
            f_bar[k + 1] += c_curr
        end

        # Advance to next interval
        if k < n - 1
            h_prev = h_curr
            δ_prev = δ_curr
            h_curr = x[k + 2] - x[k + 1]
            δ_curr = (y[k + 2] - y[k + 1]) / h_curr
        end
    end

    # ── Right endpoint (k=n) ─────────────────────────────────────────────
    # After the loop: h_prev = h[n-2], h_curr = h[n-1], δ_prev = δ[n-2], δ_curr = δ[n-1]
    # Right endpoint mirrors left: _pchip_endpoint_slope(h_curr, h_prev, δ_curr, δ_prev)
    # d = ((2*h_curr + h_prev)*δ_curr - h_curr*δ_prev) / (h_curr + h_prev)
    @inbounds begin
        h1_r = h_curr   # boundary interval
        h2_r = h_prev   # next-to-boundary interval
        δ1_r = δ_curr   # boundary secant
        δ2_r = δ_prev   # next-to-boundary secant
        d_right = ((2 * h1_r + h2_r) * δ1_r - h1_r * δ2_r) / (h1_r + h2_r)

        if sign(d_right) != sign(δ1_r)
            # Zero-clamped → skip
        elseif sign(δ1_r) != sign(δ2_r) && abs(d_right) > abs(3 * δ1_r)
            # Sat-clamped: dy[n] = 3*δ1_r = 3*δ[n-1]
            # δ[n-1] = (y[n]-y[n-1])/h[n-1] → ∂/∂y[n-1] = -1/h[n-1], ∂/∂y[n] = +1/h[n-1]
            c = (3 / h1_r) * dy_bar[n]
            f_bar[n - 1] -= c
            f_bar[n] += c
        else
            # Unclamped: dy[n] = ((2h1_r+h2_r)*δ1_r - h1_r*δ2_r) / (h1_r+h2_r)
            # ∂dy[n]/∂δ1_r = (2h1_r+h2_r)/(h1_r+h2_r)
            # ∂dy[n]/∂δ2_r = -h1_r/(h1_r+h2_r)
            ddy_dδ1 = (2 * h1_r + h2_r) / (h1_r + h2_r)
            ddy_dδ2 = -h1_r / (h1_r + h2_r)
            db = dy_bar[n]
            # δ1_r = δ[n-1]: interval [n-1, n]
            # ∂/∂y[n-1] = -1/h[n-1], ∂/∂y[n] = +1/h[n-1]
            c1 = (ddy_dδ1 / h1_r) * db
            f_bar[n - 1] -= c1
            f_bar[n] += c1
            # δ2_r = δ[n-2]: interval [n-2, n-1]
            # ∂/∂y[n-2] = -1/h[n-2], ∂/∂y[n-1] = +1/h[n-2]
            c2 = (ddy_dδ2 / h2_r) * db
            f_bar[n - 2] -= c2
            f_bar[n - 1] += c2
        end
    end

    return nothing
end

# ========================================
# Core Apply Function
# ========================================

@inline _adjoint_1d_apply!(f_bar, adj::PchipAdjoint1D, y_bar, deriv) =
    _pchip_adjoint_apply!(f_bar, adj, y_bar, deriv)

@with_pool pool function _pchip_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::PchipAdjoint1D{Tg},
        y_bar,
        deriv::DerivOp = EvalValue()
    ) where {Tv, Tg}
    n = adj.grid_size
    dy_bar = zeros!(pool, Tv, n)

    # Step 1: Hermite scatter -> (f_bar, dy_bar)
    _scatter_hermite_adjoint!(f_bar, dy_bar, adj.anchors, y_bar, deriv)

    # Step 2: PCHIP slope J^T * dy_bar -> f_bar update
    _pchip_slope_adjoint!(f_bar, dy_bar, adj.grid, adj.data)

    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    pchip_adjoint(x, y, x_query; extrap=NoExtrap()) -> PchipAdjoint1D

Create a PCHIP adjoint operator (query-baked, linearized at data `y`).

Computes `f_bar = W^T * y_bar` where `W` is the forward PCHIP interpolation
weight matrix, including the slope-from-data dependence.

Because PCHIP slopes involve data-dependent branching (monotonicity clamping),
the adjoint is linearized at the given `y`. The dot-product identity holds
exactly at this operating point:

    dot(pchip_interp(x, y).(xq), y_bar) == dot(y, pchip_adjoint(x, y, xq)(y_bar))

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `y::AbstractVector`: Data values (determines slope branch conditions)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
y = randn(50)
xq = sort(rand(30))
y_bar = randn(30)

itp = pchip_interp(x, y)
adj = pchip_adjoint(x, y, xq)
f_bar = adj(y_bar)

# Dot-product identity: <W*y, y_bar> = <y, W^T*y_bar>
@assert dot(itp.(xq), y_bar) ≈ dot(y, adj(y_bar))
```
"""
function pchip_adjoint(
        x::AbstractVector,
        y::AbstractVector,
        x_query::AbstractVector;
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    x_p, xq_p, Tg = _promote_adjoint_inputs(x, x_query)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # NoExtrap: validate all queries in-domain
    if extrap isa NoExtrap
        x_lo, x_hi = first(x_p), last(x_p)
        @inbounds for i in eachindex(xq_p)
            xq_i = xq_p[i]
            (_extract_primal(x_lo) <= xq_i <= _extract_primal(x_hi)) || throw(
                DomainError(xq_i, "query point outside domain [$(_extract_primal(x_lo)), $(_extract_primal(x_hi))]")
            )
        end
    end

    # Wrap axis (axis-as-truth) and bake anchors
    x_axis = _cache_axis(x_p, NoBC())
    anchors = _bake_hermite_adjoint_anchors(x_axis, xq_p, extrap)

    # Promote y to float: slope adjoint computes fractional derivatives (Int division loses precision)
    _, y_p = _promote_itp_inputs(x, y)
    Tv = eltype(y_p)
    return PchipAdjoint1D{Tg, Tv, typeof(extrap)}(
        anchors, collect(x_p), collect(Tv, y_p), length(x_p), extrap
    )
end

# Scalar query convenience
function pchip_adjoint(
        x::AbstractVector,
        y::AbstractVector,
        x_query::Real;
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    return pchip_adjoint(x, y, [x_query]; extrap = extrap)
end
