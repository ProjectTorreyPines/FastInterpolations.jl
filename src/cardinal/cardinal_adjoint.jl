# ========================================
# CardinalAdjoint1D: Adjoint (Transpose) Operator
# ========================================
#
# Computes f_bar = W^T * y_bar where W is the implicit forward cardinal
# interpolation matrix.
#
# Cardinal forward pipeline:
#   1. Slope computation:  dy[k] = (1-tension) * finite_difference(y, x, k)
#   2. Hermite evaluation: P(t) = h00*yL + h10*h*dyL + h01*yR + h11*h*dyR
#
# Adjoint pipeline reverses both stages:
#   1. Hermite scatter -> (f_bar_direct, dy_bar)  [shared core from hermite_adjoint.jl]
#   2. Slope J^T * dy_bar -> f_bar_slope          [cardinal-specific]
#   3. Final: f_bar_total = f_bar_direct + f_bar_slope
#
# The entire forward is LINEAR in y (slopes are linear in y), so the
# dot-product identity holds exactly: dot(itp.(xq), y_bar) = dot(y, adj(y_bar)).
#
# Dependencies (already included before this file):
# - _HermiteAdjointAnchor1D, _bake_hermite_adjoint_anchors (hermite_adjoint.jl)
# - _scatter_hermite_adjoint! (hermite_adjoint.jl)
# - AbstractAdjoint1D, _throw_adjoint_grid_too_small (adjoint_protocol.jl)
# - _promote_grid_float, _to_float (promotion helpers)
# - _create_spacing (grid_spacing.jl)

# ========================================
# CardinalAdjoint1D Struct
# ========================================

"""
    CardinalAdjoint1D{Tg, EP}

Adjoint (transpose) operator for cardinal spline interpolation.
Computes `f_bar = W^T * y_bar` where `W` is the forward cardinal interpolation
weight matrix, including the slope-from-data dependence.

Because cardinal slopes are linear in `y`, the full forward operation is linear
and the dot-product identity holds exactly:

    dot(cardinal_interp(x, y; tension=t).(xq), y_bar) == dot(y, adj(y_bar))

# Type Parameters
- `Tg`: Grid float type (Float32 or Float64)
- `EP`: Extrapolation policy type

# Usage
```julia
adj = cardinal_adjoint(x, xq; tension=0.5)

f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # derivative adjoint
adj(f_bar, y_bar)                       # in-place
```
"""
struct CardinalAdjoint1D{Tg, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_HermiteAdjointAnchor1D{Tg}}
    grid::Vector{Tg}       # Grid points (needed for slope adjoint stencil widths)
    grid_size::Int
    tension::Tg
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================

@inline _n_queries(adj::CardinalAdjoint1D) = length(adj.anchors)
@inline _adjoint_output_length(adj::CardinalAdjoint1D) = adj.grid_size

# ========================================
# Slope Adjoint: J^T * dy_bar -> f_bar update
# ========================================
#
# Cardinal slopes:
#   Left endpoint:  dy[1] = scale * (y[2] - y[1]) / (x[2] - x[1])
#   Interior k:     dy[k] = scale * (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])
#   Right endpoint: dy[n] = scale * (y[n] - y[n-1]) / (x[n] - x[n-1])
#
# Transpose: for each k, dy_bar[k] scatters to 2 f_bar entries.

@inline function _cardinal_slope_adjoint!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        x::AbstractVector{Tg}, tension::Tg
    ) where {Tg}
    n = length(x)
    scale = one(Tg) - tension

    # Left endpoint: dy[1] = scale*(y[2]-y[1])/(x[2]-x[1])
    @inbounds begin
        c1 = scale * dy_bar[1] / (x[2] - x[1])
        f_bar[1] -= c1
        f_bar[2] += c1
    end

    # Interior k=2..n-1: dy[k] = scale*(y[k+1]-y[k-1])/(x[k+1]-x[k-1])
    @inbounds for k in 2:(n - 1)
        ck = scale * dy_bar[k] / (x[k + 1] - x[k - 1])
        f_bar[k - 1] -= ck
        f_bar[k + 1] += ck
    end

    # Right endpoint: dy[n] = scale*(y[n]-y[n-1])/(x[n]-x[n-1])
    @inbounds begin
        cn = scale * dy_bar[n] / (x[n] - x[n - 1])
        f_bar[n - 1] -= cn
        f_bar[n] += cn
    end

    return nothing
end

# ========================================
# Core Apply Function
# ========================================

@inline _adjoint_1d_apply!(f_bar, adj::CardinalAdjoint1D, y_bar, deriv) =
    _cardinal_adjoint_apply!(f_bar, adj, y_bar, deriv)

@with_pool pool function _cardinal_adjoint_apply!(
        f_bar::AbstractVector{Tv},
        adj::CardinalAdjoint1D{Tg},
        y_bar,
        deriv::DerivOp = EvalValue()
    ) where {Tv, Tg}
    n = adj.grid_size
    dy_bar = zeros!(pool, Tv, n)

    # Step 1: Hermite scatter -> (f_bar, dy_bar)
    _scatter_hermite_adjoint!(f_bar, dy_bar, adj.anchors, y_bar, deriv)

    # Step 2: Slope J^T * dy_bar -> f_bar update
    _cardinal_slope_adjoint!(f_bar, dy_bar, adj.grid, adj.tension)

    return nothing
end

# ========================================
# Constructor
# ========================================

"""
    cardinal_adjoint(x, x_query; tension=0.0, extrap=NoExtrap()) -> CardinalAdjoint1D

Create a cardinal spline adjoint operator (query-baked, data-free).

Computes `f_bar = W^T * y_bar` where `W` is the forward cardinal interpolation
weight matrix, including the slope-from-data dependence.

# Arguments
- `x::AbstractVector`: Grid points (must be sorted)
- `x_query::AbstractVector`: Query points (baked into the operator)
- `tension::Real`: Tension parameter (0 = CatmullRom, 1 = zero slopes)
- `extrap::AbstractExtrap`: Extrapolation mode (default: `NoExtrap()`)

# Example
```julia
using LinearAlgebra
x = collect(range(0, 1, 50))
xq = sort(rand(30))
y = randn(50)
y_bar = randn(30)

itp = cardinal_interp(x, y; tension=0.5)
adj = cardinal_adjoint(x, xq; tension=0.5)
f_bar = adj(y_bar)

# Dot-product identity: <W*y, y_bar> = <y, W^T*y_bar>
@assert dot(itp.(xq), y_bar) ≈ dot(y, adj(y_bar))
```
"""
function cardinal_adjoint(
        x::AbstractVector,
        x_query::AbstractVector;
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    Tg = _promote_grid_float(eltype(x), eltype(x_query))
    x_p = _to_float(x, Tg)
    Tq_float = Tg <: AbstractFloat ? Tg : float(eltype(x_query))
    xq_p = _to_float(x_query, Tq_float)

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

    # Build spacing and anchors
    spacing = _create_spacing(x_p)
    anchors = _bake_hermite_adjoint_anchors(x_p, spacing, xq_p, extrap)

    return CardinalAdjoint1D{Tg, typeof(extrap)}(
        anchors, collect(x_p), length(x_p), Tg(tension), extrap
    )
end

# Scalar query convenience
function cardinal_adjoint(
        x::AbstractVector,
        x_query::Real;
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    return cardinal_adjoint(x, [x_query]; tension = tension, extrap = extrap)
end
