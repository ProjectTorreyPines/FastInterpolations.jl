# ========================================
# CellPoly: Lightweight Polynomial on a Spline Cell
# ========================================
#
# Coefficients in ascending power order for `evalpoly` compatibility:
#   S(u) = p[1] + p[2]·u + p[3]·u² + ... + p[N]·uⁿ⁻¹
#   where u = x - xL  (shifted local coordinate)
#
# Cubic:     CellPoly{4} — p = (d, c, b, a)
# Quadratic: CellPoly{3} — p = (y₀, slope, a)
# Linear:    CellPoly{2} — p = (y₀, slope)
# Constant:  CellPoly{1} — p = (y₀,)

"""
    CellPoly{N, Tv, Tg}

Lightweight immutable polynomial on a spline cell.
Coefficients in ascending power order for `evalpoly` compatibility.

# Type Parameters
- `N`: Number of coefficients (= polynomial degree + 1)
- `Tv`: Value type (unconstrained)
- `Tg`: Grid coordinate type (Float64, Float32)

# Fields
- `p::NTuple{N, Tv}`: Polynomial coefficients in ascending power order
- `xL::Tg`: Left endpoint of cell
- `xR::Tg`: Right endpoint of cell

# Polynomial form

    S(x) = p₁ + p₂·u + p₃·u² + ... + pₙ·uⁿ⁻¹    where u = x - xL

# Usage
```julia
cell = coeffs(itp, 1.5)
cell(1.5)                   # callable: evaluates polynomial at global x
evalpoly(1.5, cell)          # overloaded: auto-computes u = x - xL
cell.p                       # raw coefficients for custom operations
cell.xL, cell.xR             # cell boundaries
```
"""
struct CellPoly{N, Tv, Tg <: AbstractFloat}
    p::NTuple{N, Tv}
    xL::Tg
    xR::Tg
end

# Callable: evaluate polynomial at global x
@inline (c::CellPoly)(x::Real) = evalpoly(x - c.xL, c.p)

# evalpoly overload: user writes evalpoly(x, cell) naturally
@inline Base.evalpoly(x::Real, c::CellPoly) = evalpoly(x - c.xL, c.p)

function Base.show(io::IO, c::CellPoly{N, Tv, Tg}) where {N, Tv, Tg}
    deg = N - 1
    return print(io, "CellPoly{deg=$deg, $Tv}(x ∈ [$(c.xL), $(c.xR)])")
end

# ========================================
# coeffs: Extract local polynomial from interpolant
# ========================================

"""
    coeffs(itp, x; search=..., hint=...) → CellPoly

Extract the local polynomial coefficients of `itp` at query point `x`.
Returns a [`CellPoly`](@ref) with ascending-power coefficients for `evalpoly`.

The returned `CellPoly` is callable: `cell(x)` evaluates the polynomial at `x`.
Alternatively, use `evalpoly(x, cell)` for the same result.

# Cubic
For `CubicInterpolant`, returns `CellPoly{4}` with coefficients `(d, c, b, a)`
representing `S(u) = d + c·u + b·u² + a·u³` where `u = x - xL`.

# Quadratic
For `QuadraticInterpolant`, returns `CellPoly{3}` with coefficients `(y₀, d, a)`
representing `S(u) = y₀ + d·u + a·u²`.

# Linear
For `LinearInterpolant`, returns `CellPoly{2}` with coefficients `(y₀, slope)`
representing `S(u) = y₀ + slope·u`.

# Constant
For `ConstantInterpolant`, returns `CellPoly{1}` with coefficient `(y₀,)`.

# Example
```julia
itp = cubic_interp([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0])
cell = coeffs(itp, 1.5)
cell(1.5)            # same as itp(1.5)
evalpoly(1.5, cell)  # same result
cell.p               # (d, c, b, a) in ascending power order
```
"""
function coeffs end

# ── CubicInterpolant ──

@inline function coeffs(
        itp::CubicInterpolant{Tg, Tv}, xq::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv}
    x = itp.cache.x
    searcher = _resolve_search(x, xq, search, hint)
    i, i_R, xL, xR = search_interval(searcher, x, xq)
    h = _get_h(x, i)
    inv_h = _get_inv_h(x, i)
    inv6 = inv(Tg(6))    # const-folded
    inv_6h = inv_h * inv6 # 1/(6h), shared factor         fmul
    h_inv6 = h * inv6     # h/6                           fmul
    @inbounds begin
        zL, zR = itp.z[i], itp.z[i_R]
        yL, yR = itp.y[i], itp.y[i_R]
    end
    z_sum = muladd(Tg(2), zL, zR)                       # 2zL + zR       fmadd
    a = (zR - zL) * inv_6h                               # (zR-zL)/(6h)   fsub, fmul
    b = zL * inv(Tg(2))                                  # zL/2           fmul
    c = muladd(-h_inv6, z_sum, (yR - yL) * inv_h)        # (yR-yL)/h - h(2zL+zR)/6  fsub, fmul, fnmsub
    d = yL
    return CellPoly{4, Tv, Tg}((d, c, b, a), xL, xR)
end

# ── QuadraticInterpolant ──

@inline function coeffs(
        itp::QuadraticInterpolant{Tg, Tv}, xq::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv}
    searcher = _resolve_search(itp.x, xq, search, hint)
    i, _, xL, xR = search_interval(searcher, itp.x, xq)
    @inbounds return CellPoly{3, Tv, Tg}((itp.y[i], itp.d[i], itp.a[i]), xL, xR)
end

# ── LinearInterpolant ──

@inline function coeffs(
        itp::LinearInterpolant{Tg, Tv}, xq::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv}
    searcher = _resolve_search(itp.x, xq, search, hint)
    i, i_R, xL, xR = search_interval(searcher, itp.x, xq)
    @inbounds begin
        # itp.x is `_CachedRange`, `_CachedVector`, or `_ExclusivePeriodicAxis`.
        # 2-arg `_get_inv_h(x, i)` dispatches to cached scalar / vector lookup, or
        # (for the axis wrapper at seam) computes from period. itp.y is
        # `Vector{Tv}` for non-periodic / Range-extended paths and
        # `_ExclusivePeriodicData` for Vector + `:exclusive` — the data wrapper
        # auto-cycles `i_R = n+1` back to inner[1], so plain `itp.y[i_R]` works
        # uniformly without `_resolve_idx`.
        slope = (itp.y[i_R] - itp.y[i]) * _get_inv_h(itp.x, i)
        return CellPoly{2, Tv, Tg}((itp.y[i], slope), xL, xR)
    end
end

# ── ConstantInterpolant ──

@inline function coeffs(
        itp::ConstantInterpolant{Tg, Tv}, xq::Real;
        search::AbstractSearchPolicy = itp.search_policy,
        hint::Union{Nothing, Base.RefValue{Int}} = nothing
    ) where {Tg, Tv}
    searcher = _resolve_search(itp.x, xq, search, hint)
    i, i_R, xL, xR = search_interval(searcher, itp.x, xq)
    # itp.x: `_CachedRange` / `_CachedVector` / `_ExclusivePeriodicAxis`.
    # itp.y: plain `Vector{Tv}` for non-periodic / `_ExclusivePeriodicData`
    # for Vector + `:exclusive` — the data wrapper auto-cycles `i_R = n+1`
    # back to inner[1], so plain `itp.y[i_R]` works uniformly.
    @inbounds y0 = _constant_kernel(EvalValue(), itp.y[i], itp.y[i_R], _get_h(itp.x, i), xq - xL, itp.side)
    return CellPoly{1, Tv, Tg}((y0,), xL, xR)
end
