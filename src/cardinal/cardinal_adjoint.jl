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
- `Tg`: Grid type — normally Float32/Float64, unconstrained for duck-typed grids (e.g. ForwardDiff.Dual)
- `EP`: Extrapolation policy type

# Usage
```julia
adj = cardinal_adjoint(x, xq; tension=0.5)

f_bar = adj(y_bar)                      # value adjoint
f_bar = adj(y_bar; deriv=DerivOp(1))    # derivative adjoint
adj(f_bar, y_bar)                       # in-place
```
"""
struct CardinalAdjoint1D{Tg, BC <: AbstractBC, EP <: AbstractExtrap} <: AbstractAdjoint1D{Tg}
    anchors::Vector{_HermiteAdjointAnchor1D{Tg}}
    grid::Vector{Tg}       # Grid points (extended to length n+1 for `:exclusive`)
    grid_size::Int         # Internal length: n+1 for `:exclusive`, n otherwise
    tension::Tg
    bc::BC
    extrap::EP
end

# ========================================
# 1D Adjoint Protocol Accessors
# ========================================
# Callables, Base.size, Base.Matrix, exclusive-periodic in-place seam fold
# inherited from AbstractAdjoint1D via src/core/adjoint_protocol.jl.

@inline _n_queries(adj::CardinalAdjoint1D) = length(adj.anchors)

@inline _adjoint_output_length(adj::CardinalAdjoint1D) =
    adj.bc isa PeriodicBC{:exclusive} ? adj.grid_size - 1 : adj.grid_size

@inline _adjoint_internal_length(adj::CardinalAdjoint1D) = adj.grid_size

@inline _adjoint_1d_has_exclusive_periodic(adj::CardinalAdjoint1D) =
    adj.bc isa PeriodicBC{:exclusive}

function _adjoint_1d_finalize(f_bar::AbstractVector, adj::CardinalAdjoint1D)
    if adj.bc isa PeriodicBC{:exclusive}
        n_internal = adj.grid_size
        @inbounds f_bar[1] += f_bar[n_internal]
        return f_bar[1:(n_internal - 1)]
    end
    return f_bar
end

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

# Closed-cycle Cardinal slope adjoint (after `_periodic_extend_1d`-style extension).
# Forward:  dy[k] = scale * (m_prev*h_prev + m_curr*h_curr) / (h_prev + h_curr)
#   where  m_prev = (y[j_prev+1] - y[j_prev]) / h_prev,
#          m_curr = (y[j_curr+1] - y[j_curr]) / h_curr,
#          j_prev = mod1(k-1, m_cyc),  j_curr = mod1(k, m_cyc),  m_cyc = n-1.
#
# Adjoint scatters dy_bar[k] to FOUR f_bar entries (j_prev, j_prev+1, j_curr,
# j_curr+1). For interior k (where j_prev+1 == j_curr), two of those writes
# fall on the same index with opposite signs and cancel — equivalent to the
# legacy 2-write central FD adjoint. For boundary k ∈ {1, n} in closed cycle,
# the 4 indices are DISTINCT (the join wraps so j_prev+1 ≠ j_curr), and all
# four writes contribute. Unified across both interior and boundary.
@inline function _cardinal_slope_adjoint_periodic!(
        f_bar::AbstractVector, dy_bar::AbstractVector,
        x::AbstractVector{Tg}, tension::Tg
    ) where {Tg}
    n = length(x)
    n >= 2 || return nothing
    m_cyc = n - 1
    scale = one(Tg) - tension

    @inbounds for k in 1:n
        j_prev = mod1(k - 1, m_cyc)
        j_curr = mod1(k, m_cyc)
        h_prev = x[j_prev + 1] - x[j_prev]
        h_curr = x[j_curr + 1] - x[j_curr]
        c = scale * dy_bar[k] / (h_prev + h_curr)
        # Chain rule for both secants (`m_prev` and `m_curr`) — at interior
        # the +c at j_prev+1 and -c at j_curr cancel naturally.
        f_bar[j_prev]     -= c
        f_bar[j_prev + 1] += c
        f_bar[j_curr]     -= c
        f_bar[j_curr + 1] += c
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

    # Step 2: Slope J^T * dy_bar -> f_bar update.
    # PeriodicBC (closed-cycle internal grid) → wrap-aware path.
    if adj.bc isa PeriodicBC
        _cardinal_slope_adjoint_periodic!(f_bar, dy_bar, adj.grid, adj.tension)
    else
        _cardinal_slope_adjoint!(f_bar, dy_bar, adj.grid, adj.tension)
    end

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
        bc::AbstractBC = NoBC(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    x_p, xq_p, Tg = _promote_adjoint_inputs(x, x_query)

    length(x_p) >= 2 || _throw_adjoint_grid_too_small(length(x_p))

    # Closed-cycle extension for periodic BCs — mirrors the forward
    # `cardinal_interpolant` persistent path. Cardinal adjoint is data-free
    # (slopes linear in y), so we extend the x grid only via the x-only
    # sibling `_prepare_periodic_grid`. `:exclusive` becomes length-(n+1)
    # with the seam endpoint as a real grid point; `:inclusive` is already
    # closed at n. After this, the slope adjoint runs over the extended
    # grid via the unified periodic 4-corner formula
    # (`_cardinal_slope_adjoint_periodic!`).
    x_ext = _prepare_periodic_grid(x_p, bc)
    extrap_eff = _resolve_extrap(extrap, bc, x_ext)
    bc_eff = _bc_after_extend(bc)

    # NoExtrap: validate queries against extended domain.
    if extrap_eff isa NoExtrap
        x_lo, x_hi = _extract_primal(first(x_ext)), _extract_primal(last(x_ext))
        @inbounds for i in eachindex(xq_p)
            xq_i = xq_p[i]
            (x_lo <= xq_i <= x_hi) || throw(
                DomainError(xq_i, "query point outside domain [$x_lo, $x_hi]")
            )
        end
    end

    # Wrap the extended grid for cached `h`/`inv_h` (matches forward
    # Cardinal persistent's storage type). After `bc_eff` normalization
    # any periodic input becomes `:inclusive` over the closed-cycle grid,
    # so `_cache_axis` here returns `_CachedRange` / `_CachedVector` of
    # length n+1 (NOT `_ExclusivePeriodicAxis` — the seam is now a real
    # grid point in `x_ext`).
    x_axis = _cache_axis(x_ext, bc_eff, Tg)
    anchors = _bake_hermite_adjoint_anchors(x_axis, xq_p, extrap_eff)

    return CardinalAdjoint1D{Tg, typeof(bc), typeof(extrap_eff)}(
        anchors, collect(Tg, x_ext), length(x_ext), Tg(tension), bc, extrap_eff
    )
end

# Scalar query convenience
function cardinal_adjoint(
        x::AbstractVector,
        x_query::Real;
        bc::AbstractBC = NoBC(),
        tension::Real = 0.0,
        extrap::AbstractExtrap = NoExtrap(),
        _extra...
    )
    return cardinal_adjoint(x, [x_query]; bc = bc, tension = tension, extrap = extrap)
end
