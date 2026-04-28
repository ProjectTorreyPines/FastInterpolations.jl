# ========================================
# Slope Method Tag Types
# ========================================
# Lightweight dispatch tags for local (per-index) slope computation.
# Stored in the `dy` field of Hermite interpolant types when using
# OnTheFly coefficient strategy (CS = OnTheFly).
#
# For PreCompute (CS = PreCompute), the `dy` field stores a Vector{Tv}
# of precomputed slopes as before. The DY type parameter drives dispatch
# at the kernel level (_hermite_eval_at_point).
#
# Each slope type carries a `bc::BC` field (default `NoBC()`) so that
# boundary-index slope computation can dispatch on the boundary
# condition. `BC` is a type parameter, so non-periodic call sites pay
# zero runtime cost — boundary branches dead-code-eliminate at compile
# time when `BC === NoBC`.

"""
    AbstractSlopeMethod

Abstract supertype for on-the-fly slope computation methods.
Stored in the `dy` field of Hermite interpolants when `coeffs=OnTheFly()`.

Subtypes define how slopes are computed locally at each grid index:
- [`PchipSlopes`](@ref): Fritsch-Carlson monotone-preserving (3-point stencil)
- [`CardinalSlopes`](@ref): Central finite difference with tension (3-point stencil)
- [`AkimaSlopes`](@ref): Weighted-average outlier-robust (5-point stencil)
"""
abstract type AbstractSlopeMethod end

"""
    PchipSlopes{BC} <: AbstractSlopeMethod

On-the-fly slope computation using the Fritsch-Carlson (1980) monotone-preserving algorithm.
Each slope depends on at most 3 neighboring points (the cell's own interval and one neighbor).

The `bc` field selects the endpoint slope formula:
- `NoBC()`: one-sided 3-point finite difference with monotonicity clamping.
- `PeriodicBC()`: closed-cycle wrapped formula via `_periodic_secant` /
  `_periodic_cell_width` (oneshot path stays zero-copy on the user's grid;
  the persistent path performs a one-time `_periodic_extend_1d` at
  construction).

Compile-time dispatch via type parameter — zero runtime cost when `BC === NoBC`.

Used as: `pchip_interp(x, y; coeffs=OnTheFly())`
"""
struct PchipSlopes{BC <: AbstractBC} <: AbstractSlopeMethod
    bc::BC
end
PchipSlopes() = PchipSlopes(NoBC())

"""
    CardinalSlopes{Tg, BC} <: AbstractSlopeMethod

On-the-fly slope computation using cardinal spline formula with tension parameter.
`tension=0` gives Catmull-Rom (simple central finite difference).

Each slope depends on at most 3 neighboring points.

Used as: `cardinal_interp(x, y; tension=0.5, coeffs=OnTheFly())`
"""
struct CardinalSlopes{Tg, BC <: AbstractBC} <: AbstractSlopeMethod
    tension::Tg
    bc::BC
end
CardinalSlopes(tension::Tg) where {Tg} = CardinalSlopes(tension, NoBC())

"""
    AkimaSlopes{BC} <: AbstractSlopeMethod

On-the-fly slope computation using the Akima (1970) weighted-average algorithm.
Outlier-robust: gives less weight to deviant secants.

Each slope depends on at most 5 neighboring points (4 secants).

Used as: `akima_interp(x, y; coeffs=OnTheFly())`
"""
struct AkimaSlopes{BC <: AbstractBC} <: AbstractSlopeMethod
    bc::BC
end
AkimaSlopes() = AkimaSlopes(NoBC())
