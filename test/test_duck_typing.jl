# Tests for duck-typing support: custom value types (non-Real, non-Complex)
# Verifies that arbitrary types work with ONLY the documented minimum operations.
#
# Key design: custom types are NOT <: Number. This is intentional:
# Julia's muladd(::Number, ::Number, ::Number) tries promotion → fails without promote_rule.
# By staying outside Number, the generic muladd(x,y,z) = x*y+z fires automatically.
#
# Four custom types tested:
# 1. BareMinFloat (bare struct, wraps Float64) — TRUE minimum: 7 core ops only
# 2. ScaledFloat (bare struct, wraps Float64) — 7 core + convert + isapprox
# 3. MeasuredFloat (bare struct, val ± err) — 7 core + convert + isapprox
# 4. SVector (from StaticArrays.jl) — real-world non-Number type

using Test
using FastInterpolations
using StaticArrays

# ================================================================
# Custom Type 1: ScaledFloat (NOT <: Number)
# ================================================================
# Minimal ring type wrapping Float64 — defines ONLY the documented operations.
# NOT <: Number: Julia's muladd(::Number, ::Number, ::Number) tries promotion,
# which fails without promote_rule. Outside Number, the generic muladd(x,y,z) = x*y+z
# fallback fires automatically. No muladd, no ordering, no Tv×Tv needed.

struct ScaledFloat
    val::Float64
end

# Constant: zero(::Type{Tv})
Base.zero(::Type{ScaledFloat}) = ScaledFloat(0.0)

# Linear: +(Tv,Tv), -(Tv,Tv), *(Tg,Tv)
Base.:+(a::ScaledFloat, b::ScaledFloat) = ScaledFloat(a.val + b.val)
Base.:-(a::ScaledFloat, b::ScaledFloat) = ScaledFloat(a.val - b.val)
Base.:*(a::Float64, b::ScaledFloat) = ScaledFloat(a * b.val)

# Quadratic/Cubic + ND: *(Tv,Tg), *(Int,Tv), /(Tv,Tg)
Base.:*(a::ScaledFloat, b::Float64) = ScaledFloat(a.val * b)
Base.:*(a::Integer, b::ScaledFloat) = ScaledFloat(a * b.val)
Base.:/(a::ScaledFloat, b::Float64) = ScaledFloat(a.val / b)

# Extra (not in core 7): BC convenience for Deriv(Real_literal), test ≈ assertions
Base.convert(::Type{ScaledFloat}, x::Real) = ScaledFloat(Float64(x))
Base.isapprox(a::ScaledFloat, b::ScaledFloat; kwargs...) = isapprox(a.val, b.val; kwargs...)

# ================================================================
# Custom Type 2: MeasuredFloat (NOT <: Number)
# ================================================================
# Value with uncertainty — simplified error propagation (mimics Measurements.jl).
# Two fields (val, err) — same minimal operations as ScaledFloat.

struct MeasuredFloat
    val::Float64
    err::Float64
end

# Constant: zero(::Type{Tv})
Base.zero(::Type{MeasuredFloat}) = MeasuredFloat(0.0, 0.0)

# Linear: +(Tv,Tv), -(Tv,Tv), *(Tg,Tv)
Base.:+(a::MeasuredFloat, b::MeasuredFloat) = MeasuredFloat(a.val + b.val, hypot(a.err, b.err))
Base.:-(a::MeasuredFloat, b::MeasuredFloat) = MeasuredFloat(a.val - b.val, hypot(a.err, b.err))
Base.:*(a::Float64, b::MeasuredFloat) = MeasuredFloat(a * b.val, abs(a) * b.err)

# Quadratic/Cubic + ND: *(Tv,Tg), *(Int,Tv), /(Tv,Tg)
Base.:*(a::MeasuredFloat, b::Float64) = MeasuredFloat(a.val * b, a.err * abs(b))
Base.:*(a::Integer, b::MeasuredFloat) = MeasuredFloat(a * b.val, abs(a) * b.err)
Base.:/(a::MeasuredFloat, b::Float64) = MeasuredFloat(a.val / b, a.err / abs(b))

# Extra (not in core 7): BC convenience for Deriv(Real_literal), test ≈ assertions
Base.convert(::Type{MeasuredFloat}, x::Real) = MeasuredFloat(Float64(x), 0.0)
Base.isapprox(a::MeasuredFloat, b::MeasuredFloat; kwargs...) = isapprox(a.val, b.val; kwargs...)

# ================================================================
# Core Minimum Type: BareMinFloat (NOT <: Number)
# ================================================================
# TRUE minimum: ONLY 7 arithmetic operations — no convert, no isapprox,
# no /(Tv,Int), no ==. Proves the core operations are sufficient for
# ALL interpolation methods with default boundary conditions.

struct BareMinFloat
    val::Float64
end

# Core 7 operations:
Base.zero(::Type{BareMinFloat}) = BareMinFloat(0.0)
Base.:+(a::BareMinFloat, b::BareMinFloat) = BareMinFloat(a.val + b.val)
Base.:-(a::BareMinFloat, b::BareMinFloat) = BareMinFloat(a.val - b.val)
Base.:*(a::Float64, b::BareMinFloat) = BareMinFloat(a * b.val)
Base.:*(a::BareMinFloat, b::Float64) = BareMinFloat(a.val * b)
Base.:*(a::Integer, b::BareMinFloat) = BareMinFloat(a * b.val)
Base.:/(a::BareMinFloat, b::Float64) = BareMinFloat(a.val / b)
# That's it. Nothing else. This is the documented minimum.


@testset "Duck Typing" begin

    # ================================================================
    # 1D TEST DATA
    # ================================================================
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_scaled = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
    y_measured = [MeasuredFloat(v, 0.1) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
    y_svec = [SVector(v, 2v, 3v) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
    xq = 1.5

    # ================================================================
    # CORE MINIMUM: 7 operations only (BareMinFloat)
    # Proves sufficiency for ALL methods with default BCs.
    # Uses .val for assertions since isapprox is not defined.
    # ================================================================
    @testset "Core Minimum (7 ops)" begin
        y_bare = BareMinFloat.([1.0, 4.0, 2.0, 5.0, 3.0])

        @testset "1D" begin
            @testset "Constant" begin
                itp = constant_interp(x, y_bare)
                @test itp(xq) isa BareMinFloat
            end
            @testset "Linear" begin
                itp = linear_interp(x, y_bare)
                result = itp(xq)
                @test result isa BareMinFloat
                @test result.val ≈ 3.0
            end
            @testset "Quadratic (default BC=QuadraticFit)" begin
                itp = quadratic_interp(x, y_bare)
                @test itp(xq) isa BareMinFloat
            end
            @testset "Cubic (default BC=CubicFit)" begin
                itp = cubic_interp(x, y_bare)
                @test itp(xq) isa BareMinFloat
            end
        end

        # Range grids exercise _compute_deriv1 stencils (uniform-grid fast path).
        # Vector grids go through _weighted_sum (precomputed Tg coefficients).
        # Both paths must work with core 7 ops only — no unary negation on Tv.
        @testset "1D Range grid" begin
            xr = range(0.0, 4.0, 5)
            yr = BareMinFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
            xrq = 1.5
            @testset "Constant" begin
                itp = constant_interp(xr, yr)
                @test itp(xrq) isa BareMinFloat
            end
            @testset "Linear" begin
                itp = linear_interp(xr, yr)
                result = itp(xrq)
                @test result isa BareMinFloat
                @test result.val ≈ 3.0
            end
            @testset "Quadratic (default BC=QuadraticFit)" begin
                itp = quadratic_interp(xr, yr)
                @test itp(xrq) isa BareMinFloat
            end
            @testset "Cubic (default BC=CubicFit)" begin
                itp = cubic_interp(xr, yr)
                @test itp(xrq) isa BareMinFloat
            end
            @testset "Quadratic explicit BCs" begin
                itp = quadratic_interp(xr, yr; bc=Left(Deriv1(BareMinFloat(0.0))))
                @test itp(xrq) isa BareMinFloat
            end
            @testset "Cubic explicit BCs" begin
                itp = cubic_interp(xr, yr; bc=Deriv1(BareMinFloat(0.0)))
                @test itp(xrq) isa BareMinFloat
                itp2 = cubic_interp(xr, yr; bc=ZeroCurvBC())
                @test itp2(xrq) isa BareMinFloat
            end
        end

        @testset "2D ND" begin
            xg = [0.0, 1.0, 2.0, 3.0]
            yg = [0.0, 1.0, 2.0, 3.0]
            data = [BareMinFloat(xi + 2yj) for xi in xg, yj in yg]
            @testset "Constant" begin
                itp = constant_interp((xg, yg), data)
                @test itp((0.5, 1.5)) isa BareMinFloat
            end
            @testset "Linear" begin
                itp = linear_interp((xg, yg), data)
                result = itp((0.5, 1.5))
                @test result isa BareMinFloat
                @test result.val ≈ 0.5 + 2 * 1.5
            end
            @testset "Quadratic" begin
                itp = quadratic_interp((xg, yg), data)
                @test itp((0.5, 1.5)) isa BareMinFloat
            end
            @testset "Cubic" begin
                itp = cubic_interp((xg, yg), data)
                @test itp((0.5, 1.5)) isa BareMinFloat
            end
        end

        @testset "2D ND Range grid" begin
            xgr = range(0.0, 3.0, 4)
            ygr = range(0.0, 3.0, 4)
            data_r = [BareMinFloat(xi + 2yj) for xi in xgr, yj in ygr]
            @testset "Constant" begin
                itp = constant_interp((xgr, ygr), data_r)
                @test itp((0.5, 1.5)) isa BareMinFloat
            end
            @testset "Linear" begin
                itp = linear_interp((xgr, ygr), data_r)
                result = itp((0.5, 1.5))
                @test result isa BareMinFloat
                @test result.val ≈ 0.5 + 2 * 1.5
            end
            @testset "Quadratic" begin
                itp = quadratic_interp((xgr, ygr), data_r)
                @test itp((0.5, 1.5)) isa BareMinFloat
            end
            @testset "Cubic" begin
                itp = cubic_interp((xgr, ygr), data_r)
                @test itp((0.5, 1.5)) isa BareMinFloat
            end
        end

        @testset "Series" begin
            y1 = BareMinFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
            y2 = BareMinFloat.([2.0, 1.0, 5.0, 3.0, 4.0])
            for (name, fn) in [("Constant", constant_interp), ("Linear", linear_interp),
                               ("Quadratic", quadratic_interp), ("Cubic", cubic_interp)]
                @testset "$name" begin
                    sitp = fn(x, Series(y1, y2))
                    result = sitp(xq)
                    @test length(result) == 2
                    @test eltype(result) === BareMinFloat
                end
            end
        end

        @testset "Explicit Deriv BCs with Tv values" begin
            # Deriv(Tv_value): convert(Tv, Tv) is identity (Julia built-in) → works.
            # Both quadratic and cubic promote BC to Tv, so identity convert applies.
            @testset "quadratic Deriv1(Tv_value)" begin
                itp = quadratic_interp(x, y_bare; bc=Left(Deriv1(BareMinFloat(0.0))))
                @test itp(xq) isa BareMinFloat
            end

            # Cubic: ZeroCurvBC/ZeroSlopeBC use zero(Tv) in normalize, cache uses zero(Tg)
            @testset "cubic ZeroCurvBC" begin
                itp = cubic_interp(x, y_bare; bc=ZeroCurvBC())
                @test itp(xq) isa BareMinFloat
            end
            @testset "cubic ZeroSlopeBC" begin
                itp = cubic_interp(x, y_bare; bc=ZeroSlopeBC())
                @test itp(xq) isa BareMinFloat
            end

            # Cubic: explicit Deriv(Tv_value) via BCPair
            @testset "cubic BCPair(Deriv1(Tv_value))" begin
                bc = BCPair(Deriv1(BareMinFloat(0.0)), Deriv1(BareMinFloat(0.0)))
                itp = cubic_interp(x, y_bare; bc=bc)
                @test itp(xq) isa BareMinFloat
            end

            # Cubic: Deriv1(Tv_value) (single → symmetric BCPair)
            @testset "cubic Deriv1(Tv_value)" begin
                itp = cubic_interp(x, y_bare; bc=Deriv1(BareMinFloat(0.0)))
                @test itp(xq) isa BareMinFloat
            end

            # PolyFit BCs (CubicFit, QuadraticFit) carry no values → always work
            @test quadratic_interp(x, y_bare; bc=Left(QuadraticFit())).y[1] isa BareMinFloat
            @test cubic_interp(x, y_bare).y[1] isa BareMinFloat  # default = CubicFit()
        end
    end

    # ================================================================
    # CONDITIONAL REQUIREMENTS
    # Each test proves exactly when an extra operation is needed
    # beyond the core 7.
    # ================================================================
    @testset "Conditional Requirements" begin

        @testset "convert(Tv, Real) — only for Deriv BCs with Real literals" begin
            y_bare = BareMinFloat.([1.0, 4.0, 2.0, 5.0, 3.0])

            # Deriv1(0.0) → _promote_pointbc does convert(Tv, 0.0) → MethodError
            # BareMinFloat lacks convert(BareMinFloat, Real)
            @testset "quadratic Deriv1(0.0) fails without convert" begin
                @test_throws MethodError quadratic_interp(x, y_bare; bc=Left(Deriv1(0.0)))
            end
            @testset "cubic Deriv1(0.0) fails without convert" begin
                @test_throws MethodError cubic_interp(x, y_bare; bc=Deriv1(0.0))
            end

            # ScaledFloat defines convert(ScaledFloat, Real) → Deriv1(0.0) works
            @testset "quadratic Deriv1(0.0) works with convert" begin
                y_sf = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
                itp = quadratic_interp(x, y_sf; bc=Left(Deriv1(0.0)))
                @test itp(xq) isa ScaledFloat
            end
            @testset "cubic Deriv1(0.0) works with convert" begin
                y_sf = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
                itp = cubic_interp(x, y_sf; bc=Deriv1(0.0))
                @test itp(xq) isa ScaledFloat
            end
        end

        @testset "quadratic Deriv2 BC works with core 7" begin
            y_bare = BareMinFloat.([1.0, 4.0, 2.0, 5.0, 3.0])

            # Deriv2 BC now uses κ*(h/2) instead of (κ/2)*h → only *(Tv,Tg), no /(Tv,Int)
            @testset "Deriv2(Tv_value)" begin
                bc = Left(Deriv2(BareMinFloat(0.0)))
                itp = quadratic_interp(x, y_bare; bc=bc)
                @test itp(xq) isa BareMinFloat
            end
        end

        @testset "isapprox — only for PeriodicBC(endpoint=:inclusive)" begin
            # endpoint=:exclusive skips endpoint validation → no isapprox needed
            # Use Range grid (required for :exclusive without explicit period kwarg)
            @testset "PeriodicBC(:exclusive) works without isapprox" begin
                x_per = range(0.0, 3.0, 4)
                y_per = BareMinFloat.([1.0, 2.0, 3.0, 2.0])
                itp = cubic_interp(x_per, y_per; bc=PeriodicBC(endpoint=:exclusive))
                @test itp(0.5) isa BareMinFloat
            end

            # endpoint=:inclusive with exact match → == (bitwise for immutable structs) passes
            @testset "PeriodicBC(:inclusive) exact match works" begin
                x_per = range(0.0, 4.0, 5)
                y_per = BareMinFloat.([1.0, 3.0, 2.0, 3.0, 1.0])
                itp = cubic_interp(x_per, y_per; bc=PeriodicBC(endpoint=:inclusive))
                @test itp(0.5) isa BareMinFloat
            end

            # endpoint=:inclusive with approximate match → == fails, isapprox → MethodError
            @testset "PeriodicBC(:inclusive) approx match fails without isapprox" begin
                x_per = range(0.0, 4.0, 5)
                y_approx = BareMinFloat.([1.0, 3.0, 2.0, 3.0, 1.0 + 1e-14])
                @test_throws MethodError cubic_interp(
                    x_per, y_approx; bc=PeriodicBC(endpoint=:inclusive))
            end
        end
    end

    # ================================================================
    # ScaledFloat — 1D (all 4 types)
    # ================================================================
    @testset "ScaledFloat 1D" begin
        @testset "Constant" begin
            itp = constant_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            @test itp(xq) isa ScaledFloat
        end

        @testset "Linear" begin
            itp = linear_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            result = itp(xq)
            @test result isa ScaledFloat
            @test result ≈ ScaledFloat(3.0)
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            @test itp(xq) isa ScaledFloat
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, y_scaled)
            @test eltype(itp.y) === ScaledFloat
            @test itp(xq) isa ScaledFloat
        end

        @testset "Integer grid" begin
            itp = linear_interp([0, 1, 2, 3, 4], y_scaled)
            @test eltype(itp.x) <: AbstractFloat
            @test eltype(itp.y) === ScaledFloat
            @test itp(1.5) ≈ ScaledFloat(3.0)
        end

        @testset "Deriv BC" begin
            bc = Deriv1(ScaledFloat(0.0))
            @test bc isa Deriv1{ScaledFloat}
            bc2 = Deriv2(ScaledFloat(0.0))
            @test bc2 isa Deriv2{ScaledFloat}
        end
    end

    # ================================================================
    # MeasuredFloat — 1D (all 4 types)
    # ================================================================
    @testset "MeasuredFloat 1D" begin
        @testset "Constant" begin
            itp = constant_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result.err > 0  # uncertainty preserved
        end

        @testset "Linear" begin
            itp = linear_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result ≈ MeasuredFloat(3.0, 0.0)  # value correct
            @test result.err > 0  # uncertainty propagated through interpolation
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result.err > 0
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, y_measured)
            @test eltype(itp.y) === MeasuredFloat
            result = itp(xq)
            @test result isa MeasuredFloat
            @test result.err > 0
        end

        @testset "Error propagation sanity" begin
            # Larger input errors → larger output errors
            y_small_err = [MeasuredFloat(v, 0.01) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
            y_large_err = [MeasuredFloat(v, 1.0) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
            itp_small = linear_interp(x, y_small_err)
            itp_large = linear_interp(x, y_large_err)
            @test itp_small(xq).err < itp_large(xq).err
            @test itp_small(xq).val ≈ itp_large(xq).val  # same central value
        end
    end

    # ================================================================
    # SVector — 1D (all 4 types)
    # ================================================================
    @testset "SVector 1D" begin
        @testset "Constant" begin
            itp = constant_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            @test itp(xq) isa SVector{3, Float64}
        end

        @testset "Linear" begin
            itp = linear_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            result = itp(xq)
            @test result isa SVector{3, Float64}
            # At xq=1.5: lerp between [4,8,12] and [2,4,6] → [3,6,9]
            @test result ≈ SVector(3.0, 6.0, 9.0)
        end

        @testset "Quadratic" begin
            itp = quadratic_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            result = itp(xq)
            @test result isa SVector{3, Float64}
            # Each component should be proportional: result[2] ≈ 2*result[1]
            @test result[2] ≈ 2 * result[1]
            @test result[3] ≈ 3 * result[1]
        end

        @testset "Cubic" begin
            itp = cubic_interp(x, y_svec)
            @test eltype(itp.y) === SVector{3, Float64}
            result = itp(xq)
            @test result isa SVector{3, Float64}
            @test result[2] ≈ 2 * result[1]
            @test result[3] ≈ 3 * result[1]
        end

        @testset "Grid point exactness" begin
            itp = linear_interp(x, y_svec)
            for i in eachindex(x)
                @test itp(x[i]) ≈ y_svec[i]
            end
        end
    end

    # ================================================================
    # ScaledFloat — 2D ND
    # ================================================================
    @testset "ScaledFloat 2D" begin
        xg = [0.0, 1.0, 2.0, 3.0]
        yg = [0.0, 1.0, 2.0, 3.0]
        # f(x,y) = x + 2y  (linear function → exact for linear interp)
        data_scaled = [ScaledFloat(xi + 2*yj) for xi in xg, yj in yg]

        @testset "Linear ND" begin
            itp = linear_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
            @test result ≈ ScaledFloat(0.5 + 2*1.5)
        end

        @testset "Constant ND" begin
            itp = constant_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
        end

        @testset "Quadratic ND" begin
            itp = quadratic_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
        end

        @testset "Cubic ND" begin
            # Cubic ND solver uses Tv(scalar) conversions and muladd(Tv, Tv, Tv)
            itp = cubic_interp((xg, yg), data_scaled)
            result = itp((0.5, 1.5))
            @test result isa ScaledFloat
        end
    end

    # ================================================================
    # SVector — 2D ND (vector-valued surface)
    # ================================================================
    @testset "SVector 2D" begin
        xg = [0.0, 1.0, 2.0, 3.0]
        yg = [0.0, 1.0, 2.0, 3.0]
        # Vector-valued: f(x,y) = [x+y, x-y] (linear)
        data_sv = [SVector(xi + yj, xi - yj) for xi in xg, yj in yg]

        @testset "Linear ND" begin
            itp = linear_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
            @test result ≈ SVector(0.5 + 1.5, 0.5 - 1.5)
        end

        @testset "Constant ND" begin
            itp = constant_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
        end

        @testset "Quadratic ND" begin
            itp = quadratic_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
        end

        @testset "Cubic ND" begin
            itp = cubic_interp((xg, yg), data_sv)
            result = itp((0.5, 1.5))
            @test result isa SVector{2, Float64}
        end

        @testset "Grid point exactness (linear)" begin
            itp = linear_interp((xg, yg), data_sv)
            for (i, xi) in enumerate(xg), (j, yj) in enumerate(yg)
                @test itp((xi, yj)) ≈ data_sv[i, j]
            end
        end
    end

    # ================================================================
    # MeasuredFloat — 2D ND
    # ================================================================
    @testset "MeasuredFloat 2D" begin
        xg = [0.0, 1.0, 2.0, 3.0]
        yg = [0.0, 1.0, 2.0, 3.0]
        data_mf = [MeasuredFloat(xi + yj, 0.05) for xi in xg, yj in yg]

        @testset "Linear ND" begin
            itp = linear_interp((xg, yg), data_mf)
            result = itp((0.5, 1.5))
            @test result isa MeasuredFloat
            @test result ≈ MeasuredFloat(2.0, 0.0)
            @test result.err > 0  # uncertainty propagated
        end

        @testset "Cubic ND" begin
            # MeasuredFloat is a scalar <: Number → Tv(scalar) works
            itp = cubic_interp((xg, yg), data_mf)
            result = itp((0.5, 1.5))
            @test result isa MeasuredFloat
            @test result.err > 0
        end
    end

    # ================================================================
    # Series — ScaledFloat (tests BC promotion to Tv_out)
    # ================================================================
    @testset "ScaledFloat Series" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y1_sf = ScaledFloat.([1.0, 4.0, 2.0, 5.0, 3.0])
        y2_sf = ScaledFloat.([2.0, 1.0, 5.0, 3.0, 4.0])

        @testset "Quadratic Series (default BC)" begin
            sitp = quadratic_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end

        @testset "Quadratic Series (Deriv1 BC)" begin
            bc = Left(Deriv1(ScaledFloat(0.0)))
            sitp = quadratic_interp(x, Series(y1_sf, y2_sf); bc=bc)
            result = sitp(1.5)
            @test eltype(result) === ScaledFloat
        end

        @testset "Cubic Series (default BC)" begin
            sitp = cubic_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end

        @testset "Cubic Series (ZeroCurvBC)" begin
            sitp = cubic_interp(x, Series(y1_sf, y2_sf); bc=ZeroCurvBC())
            result = sitp(1.5)
            @test eltype(result) === ScaledFloat
        end

        @testset "Linear Series" begin
            sitp = linear_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end

        @testset "Constant Series" begin
            sitp = constant_interp(x, Series(y1_sf, y2_sf))
            result = sitp(1.5)
            @test length(result) == 2
            @test eltype(result) === ScaledFloat
        end
    end

    # ================================================================
    # Series — MeasuredFloat (tests uncertainty propagation through series)
    # ================================================================
    @testset "MeasuredFloat Series" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y1_mf = [MeasuredFloat(v, 0.1) for v in [1.0, 4.0, 2.0, 5.0, 3.0]]
        y2_mf = [MeasuredFloat(v, 0.2) for v in [2.0, 1.0, 5.0, 3.0, 4.0]]

        @testset "Linear Series" begin
            sitp = linear_interp(x, Series(y1_mf, y2_mf))
            result = sitp(1.5)
            @test length(result) == 2
            @test all(r -> r isa MeasuredFloat, result)
            @test all(r -> r.err > 0, result)
        end

        @testset "Quadratic Series" begin
            sitp = quadratic_interp(x, Series(y1_mf, y2_mf))
            result = sitp(1.5)
            @test all(r -> r isa MeasuredFloat, result)
            @test all(r -> r.err > 0, result)
        end

        @testset "Cubic Series" begin
            sitp = cubic_interp(x, Series(y1_mf, y2_mf))
            result = sitp(1.5)
            @test all(r -> r isa MeasuredFloat, result)
            @test all(r -> r.err > 0, result)
        end
    end

    # ================================================================
    # Standard promotion (regression tests)
    # ================================================================
    @testset "Standard promotion unchanged" begin
        @test eltype(linear_interp([0.0, 1.0, 2.0], [10, 20, 30]).y) === Float64
        @test eltype(linear_interp([0.0, 1.0, 2.0], Float32[1, 2, 3]).y) === Float64
        @test eltype(linear_interp([0.0, 1.0, 2.0], ComplexF64[1, 2, 3]).y) === ComplexF64
        @test eltype(linear_interp([0, 1, 2], [1.0, 2.0, 3.0]).x) === Float64
    end

    @testset "Mixed precision widening" begin
        itp = linear_interp(Float32[0, 1, 2, 3, 4], [1.0, 4.0, 2.0, 5.0, 3.0])
        @test eltype(itp.x) === Float64
        @test eltype(itp.y) === Float64
    end

    # ================================================================
    # Internal API
    # ================================================================
    @testset "_promote_itp_inputs" begin
        @testset "preserves custom Tv" begin
            x_in = [0.0, 1.0, 2.0]
            y_in = ScaledFloat.([1.0, 2.0, 3.0])
            x_out, y_out = FastInterpolations._promote_itp_inputs(x_in, y_in)
            @test x_out === x_in
            @test y_out === y_in
            @test eltype(y_out) === ScaledFloat
        end

        @testset "preserves SVector Tv" begin
            x_in = [0.0, 1.0, 2.0]
            y_in = [SVector(1.0, 2.0), SVector(3.0, 4.0), SVector(5.0, 6.0)]
            x_out, y_out = FastInterpolations._promote_itp_inputs(x_in, y_in)
            @test x_out === x_in
            @test y_out === y_in
            @test eltype(y_out) === SVector{2, Float64}
        end

        @testset "promotes standard types" begin
            x_out, y_out = FastInterpolations._promote_itp_inputs([0.0, 1.0, 2.0], [1, 2, 3])
            @test eltype(y_out) === Float64
        end
    end

end
