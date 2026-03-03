# Comprehensive duck typing tests: exhaustive method × kwarg × deriv × integrate coverage.
# Uses a 5-op-only type (DuckFloat5) — the documented minimum from custom_value_types.md.
#
# Tests ALL public API paths that should work with just:
#   +(Tv,Tv), -(Tv,Tv), *(Tg,Tv), *(Tv,Tg), *(Int,Tv)
#
# NO: zero(::Type), /(Tv,Tg), convert, isapprox, ==, <, >, muladd(Tv,Tv,Tv)

using Test
using FastInterpolations

# ================================================================
# DuckFloat5 — strict 5-op type for comprehensive testing
# ================================================================
struct DuckFloat5
    v::Float64
end

Base.:+(a::DuckFloat5, b::DuckFloat5) = DuckFloat5(a.v + b.v)
Base.:-(a::DuckFloat5, b::DuckFloat5) = DuckFloat5(a.v - b.v)
Base.:*(a::Float64, b::DuckFloat5) = DuckFloat5(a * b.v)
Base.:*(a::DuckFloat5, b::Float64) = DuckFloat5(a.v * b)
Base.:*(a::Integer, b::DuckFloat5) = DuckFloat5(a * b.v)

# Helper: extract raw value for assertions (since isapprox is NOT defined)
_val(d::DuckFloat5) = d.v

@testset "Duck Typing — Comprehensive" begin

    # ================================================================
    # SHARED TEST DATA
    # ================================================================
    # 1D grids
    x_vec = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_rng = range(0.0, 6.0, 7)
    xq = 2.7  # interior query point
    xq_vec = [1.5, 2.7, 4.3]

    # Polynomial data for exact-result testing:
    #   f(x) = 2x + 1  (linear — exact for linear and higher)
    y_linear = DuckFloat5.(2 .* collect(x_vec) .+ 1)

    #   f(x) = x² - x + 1  (quadratic — exact for quadratic and higher)
    y_quad = DuckFloat5.(collect(x_vec) .^ 2 .- collect(x_vec) .+ 1)

    #   f(x) = x³/6 - x/2 + 1  (cubic — exact for cubic)
    y_cubic = DuckFloat5.(collect(x_vec) .^ 3 ./ 6 .- collect(x_vec) ./ 2 .+ 1)

    # Generic non-polynomial data
    y_generic = DuckFloat5.([1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5])

    # 2D grids
    xg = [0.0, 1.0, 2.0, 3.0]
    yg = [0.0, 1.0, 2.0, 3.0]
    xg_r = range(0.0, 3.0, 4)
    yg_r = range(0.0, 3.0, 4)
    data_2d = [DuckFloat5(xi + 2yj) for xi in xg, yj in yg]  # linear in x,y
    q2d = (1.5, 1.5)

    # ================================================================
    # SECTION 1: INTERPOLANT CONSTRUCTION — all methods × grid types
    # ================================================================
    @testset "1. Construction — 1D" begin
        for (name, x) in [("Vector", x_vec), ("Range", x_rng)]
            @testset "$name grid" begin
                @testset "constant" begin
                    itp = constant_interp(x, y_generic)
                    @test itp(xq) isa DuckFloat5
                end
                @testset "linear" begin
                    itp = linear_interp(x, y_linear)
                    r = itp(xq)
                    @test r isa DuckFloat5
                    @test _val(r) ≈ 2 * xq + 1  # exact for linear data
                end
                @testset "quadratic (default BC)" begin
                    itp = quadratic_interp(x, y_generic)
                    @test itp(xq) isa DuckFloat5
                end
                @testset "cubic (default BC)" begin
                    itp = cubic_interp(x, y_generic)
                    @test itp(xq) isa DuckFloat5
                end
            end
        end
    end

    @testset "2. Construction — 2D ND" begin
        for (name, grids) in [("Vector", (xg, yg)), ("Range", (xg_r, yg_r))]
            @testset "$name grid" begin
                @testset "constant" begin
                    itp = constant_interp(grids, data_2d)
                    @test itp(q2d) isa DuckFloat5
                end
                @testset "linear" begin
                    itp = linear_interp(grids, data_2d)
                    r = itp(q2d)
                    @test r isa DuckFloat5
                    @test _val(r) ≈ 1.5 + 2 * 1.5
                end
                @testset "quadratic" begin
                    itp = quadratic_interp(grids, data_2d)
                    @test itp(q2d) isa DuckFloat5
                end
                @testset "cubic" begin
                    itp = cubic_interp(grids, data_2d)
                    @test itp(q2d) isa DuckFloat5
                end
            end
        end
    end

    # ================================================================
    # SECTION 2: BOUNDARY CONDITIONS — all BCs × {quadratic, cubic}
    # ================================================================
    @testset "3. Quadratic BCs" begin
        @testset "Left(QuadraticFit())" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Left(QuadraticFit()))
            @test itp(xq) isa DuckFloat5
        end
        @testset "Left(Deriv1(Tv))" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Left(Deriv1(DuckFloat5(0.0))))
            @test itp(xq) isa DuckFloat5
        end
        @testset "Left(Deriv2(Tv))" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Left(Deriv2(DuckFloat5(0.0))))
            @test itp(xq) isa DuckFloat5
        end
        @testset "Right(QuadraticFit())" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Right(QuadraticFit()))
            @test itp(xq) isa DuckFloat5
        end
        @testset "Right(Deriv1(Tv))" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Right(Deriv1(DuckFloat5(0.0))))
            @test itp(xq) isa DuckFloat5
        end
        @testset "Left(LinearFit())" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Left(LinearFit()))
            @test itp(xq) isa DuckFloat5
        end
    end

    @testset "4. Cubic BCs" begin
        @testset "CubicFit() (default)" begin
            itp = cubic_interp(x_vec, y_generic)
            @test itp(xq) isa DuckFloat5
        end
        @testset "Deriv1(Tv) symmetric" begin
            itp = cubic_interp(x_vec, y_generic; bc=Deriv1(DuckFloat5(0.0)))
            @test itp(xq) isa DuckFloat5
        end
        @testset "Deriv2(Tv) symmetric" begin
            itp = cubic_interp(x_vec, y_generic; bc=Deriv2(DuckFloat5(0.0)))
            @test itp(xq) isa DuckFloat5
        end
        @testset "ZeroCurvBC()" begin
            itp = cubic_interp(x_vec, y_generic; bc=ZeroCurvBC())
            @test itp(xq) isa DuckFloat5
        end
        @testset "ZeroSlopeBC()" begin
            itp = cubic_interp(x_vec, y_generic; bc=ZeroSlopeBC())
            @test itp(xq) isa DuckFloat5
        end
        @testset "BCPair(Deriv1, Deriv2)" begin
            bc = BCPair(Deriv1(DuckFloat5(0.0)), Deriv2(DuckFloat5(0.0)))
            itp = cubic_interp(x_vec, y_generic; bc=bc)
            @test itp(xq) isa DuckFloat5
        end
        @testset "BCPair(CubicFit, Deriv1)" begin
            bc = BCPair(CubicFit(), Deriv1(DuckFloat5(0.0)))
            itp = cubic_interp(x_vec, y_generic; bc=bc)
            @test itp(xq) isa DuckFloat5
        end
        @testset "PeriodicBC(:exclusive)" begin
            xp = range(0.0, 6.0, 7)
            yp = DuckFloat5.([1.0, 3.0, 2.0, 4.0, 2.0, 3.0, 1.5])
            itp = cubic_interp(xp, yp; bc=PeriodicBC(endpoint=:exclusive))
            @test itp(1.5) isa DuckFloat5
        end
        @testset "PeriodicBC(:inclusive) exact match" begin
            xp = range(0.0, 6.0, 7)
            yp = DuckFloat5.([1.0, 3.0, 2.0, 4.0, 2.0, 3.0, 1.0])  # y[1]==y[end]
            itp = cubic_interp(xp, yp; bc=PeriodicBC(endpoint=:inclusive))
            @test itp(1.5) isa DuckFloat5
        end
    end

    @testset "5. ND BCs" begin
        @testset "2D per-axis BCs" begin
            bc_pair = (ZeroCurvBC(), ZeroCurvBC())
            itp = cubic_interp((xg, yg), data_2d; bc=bc_pair)
            @test itp(q2d) isa DuckFloat5
        end
        @testset "2D mixed BC (Periodic + ZeroCurv)" begin
            xp = collect(range(0.0, 3.0, 4))
            # Periodic in x: y[1,:] == y[end,:] exactly
            data_p = [DuckFloat5(2yj + 1.0) for xi in xp, yj in yg]
            itp = cubic_interp((xp, yg), data_p; bc=(PeriodicBC(), ZeroCurvBC()))
            @test itp((0.5, 1.5)) isa DuckFloat5
        end
        @testset "2D quadratic ZeroCurvBC" begin
            itp = quadratic_interp((xg, yg), data_2d; bc=ZeroCurvBC())
            @test itp(q2d) isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 3: DERIVATIVES — all orders × {linear, quadratic, cubic}
    # ================================================================
    @testset "6. Derivatives — 1D" begin
        @testset "linear deriv1" begin
            itp = linear_interp(x_vec, y_linear)
            r = itp(xq; deriv=DerivOp(1))
            @test r isa DuckFloat5
            @test _val(r) ≈ 2.0  # d/dx(2x+1) = 2
        end

        @testset "quadratic deriv1" begin
            itp = quadratic_interp(x_vec, y_quad)
            r = itp(xq; deriv=DerivOp(1))
            @test r isa DuckFloat5
            @test _val(r) ≈ 2 * xq - 1 atol = 0.1  # d/dx(x²-x+1) = 2x-1
        end
        @testset "quadratic deriv2" begin
            itp = quadratic_interp(x_vec, y_quad)
            r = itp(xq; deriv=DerivOp(2))
            @test r isa DuckFloat5
        end

        @testset "cubic deriv1" begin
            itp = cubic_interp(x_vec, y_cubic)
            r = itp(xq; deriv=DerivOp(1))
            @test r isa DuckFloat5
        end
        @testset "cubic deriv2" begin
            itp = cubic_interp(x_vec, y_cubic)
            r = itp(xq; deriv=DerivOp(2))
            @test r isa DuckFloat5
        end
        @testset "cubic deriv3" begin
            itp = cubic_interp(x_vec, y_cubic)
            r = itp(xq; deriv=DerivOp(3))
            @test r isa DuckFloat5
        end
    end

    @testset "7. DerivativeView — 1D" begin
        @testset "linear" begin
            itp = linear_interp(x_vec, y_linear)
            d1 = deriv1(itp)
            r = d1(xq)
            @test r isa DuckFloat5
            @test _val(r) ≈ 2.0
        end

        @testset "cubic" begin
            itp = cubic_interp(x_vec, y_generic)
            d1 = deriv1(itp)
            @test d1(xq) isa DuckFloat5
            d2 = deriv2(itp)
            @test d2(xq) isa DuckFloat5
            d3 = deriv3(itp)
            @test d3(xq) isa DuckFloat5
        end

        @testset "quadratic" begin
            itp = quadratic_interp(x_vec, y_generic)
            d1 = deriv1(itp)
            @test d1(xq) isa DuckFloat5
            d2 = deriv2(itp)
            @test d2(xq) isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 4: INTEGRATION — 1D all methods
    # ================================================================
    @testset "8. Integration — 1D" begin
        @testset "linear" begin
            itp = linear_interp(x_vec, y_linear)
            r = integrate(itp, 1.0, 4.0)
            @test r isa DuckFloat5
            # ∫₁⁴ (2x+1)dx = [x²+x]₁⁴ = 20-2 = 18
            @test _val(r) ≈ 18.0
        end

        @testset "quadratic" begin
            itp = quadratic_interp(x_vec, y_generic)
            r = integrate(itp, 1.0, 4.0)
            @test r isa DuckFloat5
        end

        @testset "cubic" begin
            itp = cubic_interp(x_vec, y_generic)
            r = integrate(itp, 1.0, 4.0)
            @test r isa DuckFloat5
        end

        @testset "constant (left side)" begin
            itp = constant_interp(x_vec, y_generic; side=LeftSide())
            r = integrate(itp, 1.0, 4.0)
            @test r isa DuckFloat5
        end

        @testset "constant (right side)" begin
            itp = constant_interp(x_vec, y_generic; side=RightSide())
            r = integrate(itp, 1.0, 4.0)
            @test r isa DuckFloat5
        end

        @testset "constant (nearest side)" begin
            itp = constant_interp(x_vec, y_generic)
            r = integrate(itp, 1.0, 4.0)
            @test r isa DuckFloat5
        end

        @testset "partial cell (non-grid-aligned bounds)" begin
            itp = cubic_interp(x_vec, y_generic)
            r = integrate(itp, 0.3, 5.7)
            @test r isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 5: ONE-SHOT API (with targets)
    # ================================================================
    @testset "9. One-shot API — scalar target" begin
        @testset "constant" begin
            r = constant_interp(x_vec, y_generic, xq)
            @test r isa DuckFloat5
        end
        @testset "linear" begin
            r = linear_interp(x_vec, y_linear, xq)
            @test r isa DuckFloat5
            @test _val(r) ≈ 2 * xq + 1
        end
        @testset "quadratic" begin
            r = quadratic_interp(x_vec, y_generic, xq)
            @test r isa DuckFloat5
        end
        @testset "cubic" begin
            r = cubic_interp(x_vec, y_generic, xq)
            @test r isa DuckFloat5
        end
    end

    @testset "10. One-shot API — vector target" begin
        @testset "constant" begin
            r = constant_interp(x_vec, y_generic, xq_vec)
            @test eltype(r) === DuckFloat5
            @test length(r) == 3
        end
        @testset "linear" begin
            r = linear_interp(x_vec, y_linear, xq_vec)
            @test eltype(r) === DuckFloat5
            for (i, q) in enumerate(xq_vec)
                @test _val(r[i]) ≈ 2q + 1
            end
        end
        @testset "quadratic" begin
            r = quadratic_interp(x_vec, y_generic, xq_vec)
            @test eltype(r) === DuckFloat5
        end
        @testset "cubic" begin
            r = cubic_interp(x_vec, y_generic, xq_vec)
            @test eltype(r) === DuckFloat5
        end
    end

    @testset "11. One-shot API — with deriv kwarg" begin
        @testset "linear deriv1" begin
            r = linear_interp(x_vec, y_linear, xq; deriv=DerivOp(1))
            @test r isa DuckFloat5
            @test _val(r) ≈ 2.0
        end
        @testset "cubic deriv1" begin
            r = cubic_interp(x_vec, y_generic, xq; deriv=DerivOp(1))
            @test r isa DuckFloat5
        end
        @testset "cubic deriv2" begin
            r = cubic_interp(x_vec, y_generic, xq; deriv=DerivOp(2))
            @test r isa DuckFloat5
        end
        @testset "quadratic deriv1" begin
            r = quadratic_interp(x_vec, y_generic, xq; deriv=DerivOp(1))
            @test r isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 6: EXTRAPOLATION MODES
    # ================================================================
    @testset "12. Extrapolation modes" begin
        xq_left = -0.5
        xq_right = 6.5

        @testset "NoExtrap throws" begin
            itp = linear_interp(x_vec, y_generic; extrap=NoExtrap())
            @test_throws DomainError itp(xq_left)
            @test_throws DomainError itp(xq_right)
        end

        @testset "ConstExtrap" begin
            itp = linear_interp(x_vec, y_generic; extrap=ConstExtrap())
            @test itp(xq_left) isa DuckFloat5
            @test itp(xq_right) isa DuckFloat5
            @test _val(itp(xq_left)) == _val(y_generic[1])
            @test _val(itp(xq_right)) == _val(y_generic[end])
        end

        @testset "ExtendExtrap" begin
            itp = linear_interp(x_vec, y_generic; extrap=ExtendExtrap())
            @test itp(xq_left) isa DuckFloat5
            @test itp(xq_right) isa DuckFloat5
        end

        @testset "ExtendExtrap cubic" begin
            itp = cubic_interp(x_vec, y_generic; extrap=ExtendExtrap())
            @test itp(xq_left) isa DuckFloat5
            @test itp(xq_right) isa DuckFloat5
        end

        @testset "ConstExtrap quadratic" begin
            itp = quadratic_interp(x_vec, y_generic; extrap=ConstExtrap())
            @test itp(xq_left) isa DuckFloat5
            @test itp(xq_right) isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 7: SERIES
    # ================================================================
    @testset "13. Series — all methods" begin
        y1 = DuckFloat5.([1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5])
        y2 = DuckFloat5.([2.0, 1.0, 5.0, 3.0, 4.0, 1.0, 3.5])
        y3 = DuckFloat5.([3.0, 2.0, 1.0, 4.0, 5.0, 2.0, 4.5])
        s = Series(y1, y2, y3)

        for (name, fn) in [("constant", constant_interp), ("linear", linear_interp),
                           ("quadratic", quadratic_interp), ("cubic", cubic_interp)]
            @testset "$name" begin
                sitp = fn(x_vec, s)
                result = sitp(xq)
                @test length(result) == 3
                @test eltype(result) === DuckFloat5
            end
        end

        @testset "cubic Series with ZeroCurvBC" begin
            sitp = cubic_interp(x_vec, s; bc=ZeroCurvBC())
            result = sitp(xq)
            @test length(result) == 3
            @test eltype(result) === DuckFloat5
        end

        @testset "quadratic Series with Deriv1(Tv)" begin
            sitp = quadratic_interp(x_vec, s; bc=Left(Deriv1(DuckFloat5(0.0))))
            result = sitp(xq)
            @test eltype(result) === DuckFloat5
        end
    end

    # ================================================================
    # SECTION 8: SEARCH POLICIES
    # ================================================================
    @testset "14. Search policies" begin
        for sp in [BinarySearch(), LinearBinarySearch(), AutoSearch()]
            @testset "$(typeof(sp))" begin
                itp = linear_interp(x_vec, y_linear; search=sp)
                @test itp(xq) isa DuckFloat5
            end
        end
    end

    # ================================================================
    # SECTION 9: GRID POINT EXACTNESS
    # ================================================================
    @testset "15. Grid point exactness" begin
        @testset "linear" begin
            itp = linear_interp(x_vec, y_linear)
            for i in eachindex(x_vec)
                @test _val(itp(x_vec[i])) ≈ _val(y_linear[i])
            end
        end
        @testset "quadratic" begin
            itp = quadratic_interp(x_vec, y_generic)
            for i in eachindex(x_vec)
                @test _val(itp(x_vec[i])) ≈ _val(y_generic[i])
            end
        end
        @testset "cubic" begin
            itp = cubic_interp(x_vec, y_generic)
            for i in eachindex(x_vec)
                @test _val(itp(x_vec[i])) ≈ _val(y_generic[i])
            end
        end
    end

    # ================================================================
    # SECTION 10: ND DERIVATIVES
    # ================================================================
    @testset "16. ND derivatives" begin
        @testset "2D linear deriv" begin
            itp = linear_interp((xg, yg), data_2d)
            # ∂f/∂x at (1.5,1.5): f = x + 2y → ∂f/∂x = 1
            r = itp(q2d; deriv=DerivOp(1, 0))
            @test r isa DuckFloat5
            @test _val(r) ≈ 1.0
            # ∂f/∂y: f = x + 2y → ∂f/∂y = 2
            r = itp(q2d; deriv=DerivOp(0, 1))
            @test r isa DuckFloat5
            @test _val(r) ≈ 2.0
        end

        @testset "2D cubic deriv" begin
            itp = cubic_interp((xg, yg), data_2d)
            r = itp(q2d; deriv=DerivOp(1, 0))
            @test r isa DuckFloat5
            r = itp(q2d; deriv=DerivOp(0, 1))
            @test r isa DuckFloat5
        end

        @testset "2D quadratic deriv" begin
            itp = quadratic_interp((xg, yg), data_2d)
            r = itp(q2d; deriv=DerivOp(1, 0))
            @test r isa DuckFloat5
            r = itp(q2d; deriv=DerivOp(0, 1))
            @test r isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 11: ND INTEGRATION
    # ================================================================
    @testset "17. ND Integration" begin
        @testset "2D linear" begin
            itp = linear_interp((xg, yg), data_2d)
            r = integrate(itp, (0.5, 0.5), (2.5, 2.5))
            @test r isa DuckFloat5
        end

        @testset "2D cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            r = integrate(itp, (0.5, 0.5), (2.5, 2.5))
            @test r isa DuckFloat5
        end

        @testset "2D quadratic" begin
            itp = quadratic_interp((xg, yg), data_2d)
            r = integrate(itp, (0.5, 0.5), (2.5, 2.5))
            @test r isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 12: SIDE SELECTION (constant only)
    # ================================================================
    @testset "18. Constant side selection" begin
        for side in [LeftSide(), RightSide(), NearestSide()]
            @testset "$(typeof(side))" begin
                itp = constant_interp(x_vec, y_generic; side=side)
                @test itp(xq) isa DuckFloat5
            end
        end
    end

    # ================================================================
    # SECTION 13: IN-PLACE ONE-SHOT API
    # ================================================================
    @testset "19. In-place one-shot" begin
        out = Vector{DuckFloat5}(undef, length(xq_vec))

        @testset "linear!" begin
            linear_interp!(out, x_vec, y_linear, xq_vec)
            @test eltype(out) === DuckFloat5
            for (i, q) in enumerate(xq_vec)
                @test _val(out[i]) ≈ 2q + 1
            end
        end

        @testset "cubic!" begin
            cubic_interp!(out, x_vec, y_generic, xq_vec)
            @test eltype(out) === DuckFloat5
        end

        @testset "quadratic!" begin
            quadratic_interp!(out, x_vec, y_generic, xq_vec)
            @test eltype(out) === DuckFloat5
        end

        @testset "constant!" begin
            constant_interp!(out, x_vec, y_generic, xq_vec)
            @test eltype(out) === DuckFloat5
        end
    end

    # ================================================================
    # SECTION 14: DERIV + EXTRAP COMBINATIONS
    # ================================================================
    @testset "20. Deriv + Extrap combos" begin
        @testset "cubic deriv1 + ExtendExtrap" begin
            itp = cubic_interp(x_vec, y_generic; extrap=ExtendExtrap())
            r = itp(-0.5; deriv=DerivOp(1))
            @test r isa DuckFloat5
            r = itp(6.5; deriv=DerivOp(1))
            @test r isa DuckFloat5
        end

        @testset "cubic deriv2 + ConstExtrap" begin
            itp = cubic_interp(x_vec, y_generic; extrap=ConstExtrap())
            # ConstExtrap with deriv → 0 (derivative of constant is 0)
            r = itp(-0.5; deriv=DerivOp(1))
            @test r isa DuckFloat5
        end

        @testset "linear deriv1 + ExtendExtrap" begin
            itp = linear_interp(x_vec, y_linear; extrap=ExtendExtrap())
            r = itp(-0.5; deriv=DerivOp(1))
            @test r isa DuckFloat5
            @test _val(r) ≈ 2.0
        end
    end

    # ================================================================
    # SECTION 15: DERIV + BC COMBINATIONS
    # ================================================================
    @testset "21. Deriv + BC combos" begin
        @testset "cubic ZeroCurvBC + deriv1" begin
            itp = cubic_interp(x_vec, y_generic; bc=ZeroCurvBC())
            @test itp(xq; deriv=DerivOp(1)) isa DuckFloat5
        end
        @testset "cubic ZeroSlopeBC + deriv1" begin
            itp = cubic_interp(x_vec, y_generic; bc=ZeroSlopeBC())
            @test itp(xq; deriv=DerivOp(1)) isa DuckFloat5
        end
        @testset "cubic Deriv1(Tv) + deriv2" begin
            itp = cubic_interp(x_vec, y_generic; bc=Deriv1(DuckFloat5(0.0)))
            @test itp(xq; deriv=DerivOp(2)) isa DuckFloat5
        end
        @testset "quadratic Deriv1(Tv) + deriv1" begin
            itp = quadratic_interp(x_vec, y_generic; bc=Left(Deriv1(DuckFloat5(0.0))))
            @test itp(xq; deriv=DerivOp(1)) isa DuckFloat5
        end
    end

    # ================================================================
    # SECTION 16: EDGE CASES
    # ================================================================
    @testset "22. Edge cases" begin
        @testset "query at left boundary" begin
            itp = cubic_interp(x_vec, y_generic)
            @test _val(itp(x_vec[1])) ≈ _val(y_generic[1])
        end
        @testset "query at right boundary" begin
            itp = cubic_interp(x_vec, y_generic)
            @test _val(itp(x_vec[end])) ≈ _val(y_generic[end])
        end
        @testset "minimum grid (3 points) — quadratic" begin
            x3 = [0.0, 1.0, 2.0]
            y3 = DuckFloat5.([1.0, 3.0, 2.0])
            itp = quadratic_interp(x3, y3)
            @test itp(0.5) isa DuckFloat5
        end
        @testset "minimum grid (4 points) — cubic" begin
            x4 = [0.0, 1.0, 2.0, 3.0]
            y4 = DuckFloat5.([1.0, 3.0, 2.0, 4.0])
            itp = cubic_interp(x4, y4)
            @test itp(1.5) isa DuckFloat5
        end
        @testset "minimum grid (2 points) — linear" begin
            x2 = [0.0, 1.0]
            y2 = DuckFloat5.([1.0, 3.0])
            itp = linear_interp(x2, y2)
            r = itp(0.5)
            @test r isa DuckFloat5
            @test _val(r) ≈ 2.0
        end
    end

end
