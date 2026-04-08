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
    PchipSlopes <: AbstractSlopeMethod

On-the-fly slope computation using the Fritsch-Carlson (1980) monotone-preserving algorithm.
Each slope depends on at most 3 neighboring points (the cell's own interval and one neighbor).

Used as: `pchip_interp(x, y; coeffs=OnTheFly())`
"""
struct PchipSlopes <: AbstractSlopeMethod end

"""
    CardinalSlopes{Tg} <: AbstractSlopeMethod

On-the-fly slope computation using cardinal spline formula with tension parameter.
`tension=0` gives Catmull-Rom (simple central finite difference).

Each slope depends on at most 3 neighboring points.

Used as: `cardinal_interp(x, y; tension=0.5, coeffs=OnTheFly())`
"""
struct CardinalSlopes{Tg} <: AbstractSlopeMethod
    tension::Tg
end

"""
    AkimaSlopes <: AbstractSlopeMethod

On-the-fly slope computation using the Akima (1970) weighted-average algorithm.
Outlier-robust: gives less weight to deviant secants.

Each slope depends on at most 5 neighboring points (4 secants).

Used as: `akima_interp(x, y; coeffs=OnTheFly())`
"""
struct AkimaSlopes <: AbstractSlopeMethod end
