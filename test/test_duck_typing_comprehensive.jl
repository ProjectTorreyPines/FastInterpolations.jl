# Comprehensive duck typing tests: exhaustive method × kwarg × deriv × integrate coverage.
# Uses a 5-op-only type (DuckFloat5) — the documented minimum from custom_value_types.md.
#
# Tests ALL public API paths that should work with just:
#   +(Tv,Tv), -(Tv,Tv), *(Tg,Tv), *(Tv,Tg), *(Int,Tv)
#
# NO: zero(::Type), /(Tv,Tg), convert, isapprox, ==, <, >, muladd(Tv,Tv,Tv)
#
# CORRECTNESS: Each test builds a reference Float64 interpolant with identical
# parameters and verifies numerical equality: _val(duck_result) ≈ ref_result.
# Note: LLVM may apply FMA contraction for native Float64 but not for the
# opaque DuckFloat5 wrapper, causing expected 1-3 ULP differences. We use ≈
# (rtol=sqrt(eps)) which catches real algorithmic bugs while allowing FMA diffs.

@testitem "Duck Typing: 1D Construction & BCs & Derivatives & Integration" setup = [DuckTypeSetup] begin
    # ================================================================
    # SECTION 1: INTERPOLANT CONSTRUCTION — all methods × grid types
    # ================================================================
    @testset "1. Construction — 1D" begin
        for (name, x) in [("Vector", x_vec), ("Range", x_rng)]
            @testset "$name grid" begin
                @testset "constant" begin
                    itp = constant_interp(x, y_generic)
                    itp_ref = constant_interp(x, y_generic_flat)
                    @test (@inferred itp(xq)) isa MyDuck
                    @test _val(itp(xq)) ≈ itp_ref(xq)
                end
                @testset "linear" begin
                    itp = linear_interp(x, y_linear)
                    itp_ref = linear_interp(x, y_linear_flat)
                    r = itp(xq)
                    @test r isa MyDuck
                    @test _val(r) ≈ 2 * xq + 1  # exact for linear data
                    @test _val(r) ≈ itp_ref(xq)
                end
                @testset "quadratic (default BC)" begin
                    itp = quadratic_interp(x, y_generic)
                    itp_ref = quadratic_interp(x, y_generic_flat)
                    @test (@inferred itp(xq)) isa MyDuck
                    @test _val(itp(xq)) ≈ itp_ref(xq)
                end
                @testset "cubic (default BC)" begin
                    itp = cubic_interp(x, y_generic)
                    itp_ref = cubic_interp(x, y_generic_flat)
                    @test (@inferred itp(xq)) isa MyDuck
                    @test _val(itp(xq)) ≈ itp_ref(xq)
                end
            end
        end
    end

    @testset "2. Construction — 2D ND" begin
        for (name, grids) in [("Vector", (xg, yg)), ("Range", (xg_r, yg_r))]
            @testset "$name grid" begin
                @testset "constant" begin
                    itp = constant_interp(grids, data_2d)
                    itp_ref = constant_interp(grids, data_2d_flat)
                    @test (@inferred itp(q2d)) isa MyDuck
                    @test _val(itp(q2d)) ≈ itp_ref(q2d)
                end
                @testset "linear" begin
                    itp = linear_interp(grids, data_2d)
                    itp_ref = linear_interp(grids, data_2d_flat)
                    r = itp(q2d)
                    @test r isa MyDuck
                    @test _val(r) ≈ 1.5 + 2 * 1.5
                    @test _val(r) ≈ itp_ref(q2d)
                end
                @testset "quadratic" begin
                    itp = quadratic_interp(grids, data_2d)
                    itp_ref = quadratic_interp(grids, data_2d_flat)
                    @test (@inferred itp(q2d)) isa MyDuck
                    @test _val(itp(q2d)) ≈ itp_ref(q2d)
                end
                @testset "cubic" begin
                    itp = cubic_interp(grids, data_2d)
                    itp_ref = cubic_interp(grids, data_2d_flat)
                    @test (@inferred itp(q2d)) isa MyDuck
                    @test _val(itp(q2d)) ≈ itp_ref(q2d)
                end
            end
        end
    end

    # ================================================================
    # SECTION 2: BOUNDARY CONDITIONS — all BCs × {quadratic, cubic}
    # ================================================================
    @testset "3. Quadratic BCs" begin
        @testset "Left(QuadraticFit())" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Left(QuadraticFit()))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Left(QuadraticFit()))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Left(Deriv1(Tv))" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Left(Deriv1(MyDuck(0.0))))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Left(Deriv1(0.0)))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Left(Deriv2(Tv))" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Left(Deriv2(MyDuck(0.0))))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Left(Deriv2(0.0)))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Right(QuadraticFit())" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Right(QuadraticFit()))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Right(QuadraticFit()))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Right(Deriv1(Tv))" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Right(Deriv1(MyDuck(0.0))))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Right(Deriv1(0.0)))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Left(LinearFit())" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Left(LinearFit()))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Left(LinearFit()))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
    end

    @testset "4. Cubic BCs" begin
        @testset "CubicFit() (default)" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Deriv1(Tv) symmetric" begin
            itp = cubic_interp(x_vec, y_generic; bc = Deriv1(MyDuck(0.0)))
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = Deriv1(0.0))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Deriv2(Tv) symmetric" begin
            itp = cubic_interp(x_vec, y_generic; bc = Deriv2(MyDuck(0.0)))
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = Deriv2(0.0))
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "ZeroCurvBC()" begin
            itp = cubic_interp(x_vec, y_generic; bc = ZeroCurvBC())
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = ZeroCurvBC())
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "ZeroSlopeBC()" begin
            itp = cubic_interp(x_vec, y_generic; bc = ZeroSlopeBC())
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = ZeroSlopeBC())
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "BCPair(Deriv1, Deriv2)" begin
            bc = BCPair(Deriv1(MyDuck(0.0)), Deriv2(MyDuck(0.0)))
            bc_ref = BCPair(Deriv1(0.0), Deriv2(0.0))
            itp = cubic_interp(x_vec, y_generic; bc = bc)
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = bc_ref)
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "BCPair(CubicFit, Deriv1)" begin
            bc = BCPair(CubicFit(), Deriv1(MyDuck(0.0)))
            bc_ref = BCPair(CubicFit(), Deriv1(0.0))
            itp = cubic_interp(x_vec, y_generic; bc = bc)
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = bc_ref)
            @test (@inferred itp(xq)) isa MyDuck
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "PeriodicBC(:exclusive)" begin
            xp = range(0.0, 6.0, 7)
            yp = MyDuck.([1.0, 3.0, 2.0, 4.0, 2.0, 3.0, 1.5])
            yp_flat = _val.(yp)
            itp = cubic_interp(xp, yp; bc = PeriodicBC(endpoint = :exclusive))
            itp_ref = cubic_interp(xp, yp_flat; bc = PeriodicBC(endpoint = :exclusive))
            @test (@inferred itp(1.5)) isa MyDuck
            @test _val(itp(1.5)) ≈ itp_ref(1.5)
        end
        @testset "PeriodicBC(:inclusive) exact match" begin
            xp = range(0.0, 6.0, 7)
            yp = MyDuck.([1.0, 3.0, 2.0, 4.0, 2.0, 3.0, 1.0])  # y[1]==y[end]
            yp_flat = _val.(yp)
            itp = cubic_interp(xp, yp; bc = PeriodicBC(endpoint = :inclusive))
            itp_ref = cubic_interp(xp, yp_flat; bc = PeriodicBC(endpoint = :inclusive))
            @test (@inferred itp(1.5)) isa MyDuck
            @test _val(itp(1.5)) ≈ itp_ref(1.5)
        end
    end

    @testset "5. ND BCs" begin
        @testset "2D per-axis BCs" begin
            bc_pair = (ZeroCurvBC(), ZeroCurvBC())
            itp = cubic_interp((xg, yg), data_2d; bc = bc_pair)
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = bc_pair)
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "2D mixed BC (Periodic + ZeroCurv)" begin
            xp = collect(range(0.0, 3.0, 4))
            # Periodic in x: y[1,:] == y[end,:] exactly
            data_p = [MyDuck(2yj + 1.0) for xi in xp, yj in yg]
            data_p_flat = _val.(data_p)
            itp = cubic_interp((xp, yg), data_p; bc = (PeriodicBC(), ZeroCurvBC()))
            itp_ref = cubic_interp((xp, yg), data_p_flat; bc = (PeriodicBC(), ZeroCurvBC()))
            @test (@inferred itp((0.5, 1.5))) isa MyDuck
            @test _val(itp((0.5, 1.5))) ≈ itp_ref((0.5, 1.5))
        end
        @testset "2D quadratic ZeroCurvBC" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = ZeroCurvBC())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = ZeroCurvBC())
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
    end

    # ================================================================
    # SECTION 3: DERIVATIVES — all orders × {linear, quadratic, cubic}
    # ================================================================
    @testset "6. Derivatives — 1D" begin
        @testset "linear deriv1" begin
            itp = linear_interp(x_vec, y_linear)
            itp_ref = linear_interp(x_vec, y_linear_flat)
            r = itp(xq; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ 2.0  # d/dx(2x+1) = 2
            @test _val(r) ≈ itp_ref(xq; deriv = DerivOp(1))
        end

        @testset "quadratic deriv1" begin
            itp = quadratic_interp(x_vec, y_quad)
            itp_ref = quadratic_interp(x_vec, y_quad_flat)
            r = itp(xq; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ 2 * xq - 1 atol = 0.1  # d/dx(x²-x+1) = 2x-1
            @test _val(r) ≈ itp_ref(xq; deriv = DerivOp(1))
        end
        @testset "quadratic deriv2" begin
            itp = quadratic_interp(x_vec, y_quad)
            itp_ref = quadratic_interp(x_vec, y_quad_flat)
            r = itp(xq; deriv = DerivOp(2))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(xq; deriv = DerivOp(2))
        end

        @testset "cubic deriv1" begin
            itp = cubic_interp(x_vec, y_cubic)
            itp_ref = cubic_interp(x_vec, y_cubic_flat)
            r = itp(xq; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(xq; deriv = DerivOp(1))
        end
        @testset "cubic deriv2" begin
            itp = cubic_interp(x_vec, y_cubic)
            itp_ref = cubic_interp(x_vec, y_cubic_flat)
            r = itp(xq; deriv = DerivOp(2))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(xq; deriv = DerivOp(2))
        end
        @testset "cubic deriv3" begin
            itp = cubic_interp(x_vec, y_cubic)
            itp_ref = cubic_interp(x_vec, y_cubic_flat)
            r = itp(xq; deriv = DerivOp(3))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(xq; deriv = DerivOp(3))
        end
    end

    @testset "7. DerivativeView — 1D" begin
        @testset "linear" begin
            itp = linear_interp(x_vec, y_linear)
            itp_ref = linear_interp(x_vec, y_linear_flat)
            d1 = deriv1(itp)
            d1_ref = deriv1(itp_ref)
            r = d1(xq)
            @test r isa MyDuck
            @test _val(r) ≈ 2.0
            @test _val(r) ≈ d1_ref(xq)
        end

        @testset "cubic" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            d1 = deriv1(itp)
            d1_ref = deriv1(itp_ref)
            @test (@inferred d1(xq)) isa MyDuck
            @test _val(d1(xq)) ≈ d1_ref(xq)
            d2 = deriv2(itp)
            d2_ref = deriv2(itp_ref)
            @test (@inferred d2(xq)) isa MyDuck
            @test _val(d2(xq)) ≈ d2_ref(xq)
            d3 = deriv3(itp)
            d3_ref = deriv3(itp_ref)
            @test (@inferred d3(xq)) isa MyDuck
            @test _val(d3(xq)) ≈ d3_ref(xq)
        end

        @testset "quadratic" begin
            itp = quadratic_interp(x_vec, y_generic)
            itp_ref = quadratic_interp(x_vec, y_generic_flat)
            d1 = deriv1(itp)
            d1_ref = deriv1(itp_ref)
            @test (@inferred d1(xq)) isa MyDuck
            @test _val(d1(xq)) ≈ d1_ref(xq)
            d2 = deriv2(itp)
            d2_ref = deriv2(itp_ref)
            @test (@inferred d2(xq)) isa MyDuck
            @test _val(d2(xq)) ≈ d2_ref(xq)
        end
    end

    # ================================================================
    # SECTION 4: INTEGRATION — 1D all methods
    # ================================================================
    @testset "8. Integration — 1D" begin
        @testset "linear" begin
            itp = linear_interp(x_vec, y_linear)
            itp_ref = linear_interp(x_vec, y_linear_flat)
            r = integrate(itp, 1.0, 4.0)
            @test r isa MyDuck
            # ∫₁⁴ (2x+1)dx = [x²+x]₁⁴ = 20-2 = 18
            @test _val(r) ≈ 18.0
            @test _val(r) ≈ integrate(itp_ref, 1.0, 4.0)
        end

        @testset "quadratic" begin
            itp = quadratic_interp(x_vec, y_generic)
            itp_ref = quadratic_interp(x_vec, y_generic_flat)
            r = integrate(itp, 1.0, 4.0)
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, 1.0, 4.0)
        end

        @testset "cubic" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            r = integrate(itp, 1.0, 4.0)
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, 1.0, 4.0)
        end

        @testset "constant (left side)" begin
            itp = constant_interp(x_vec, y_generic; side = LeftSide())
            itp_ref = constant_interp(x_vec, y_generic_flat; side = LeftSide())
            r = integrate(itp, 1.0, 4.0)
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, 1.0, 4.0)
        end

        @testset "constant (right side)" begin
            itp = constant_interp(x_vec, y_generic; side = RightSide())
            itp_ref = constant_interp(x_vec, y_generic_flat; side = RightSide())
            r = integrate(itp, 1.0, 4.0)
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, 1.0, 4.0)
        end

        @testset "constant (nearest side)" begin
            itp = constant_interp(x_vec, y_generic)
            itp_ref = constant_interp(x_vec, y_generic_flat)
            r = integrate(itp, 1.0, 4.0)
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, 1.0, 4.0)
        end

        @testset "partial cell (non-grid-aligned bounds)" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            r = integrate(itp, 0.3, 5.7)
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, 0.3, 5.7)
        end
    end

end

# ================================================================
# SECTION 5: ONE-SHOT API (with targets)
# ================================================================
@testitem "Duck Typing: 1D APIs & Series & Edge cases" setup = [DuckTypeSetup] begin
    @testset "9. One-shot API — scalar target" begin
        @testset "constant" begin
            r = constant_interp(x_vec, y_generic, xq)
            r_ref = constant_interp(x_vec, y_generic_flat, xq)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "linear" begin
            r = linear_interp(x_vec, y_linear, xq)
            r_ref = linear_interp(x_vec, y_linear_flat, xq)
            @test r isa MyDuck
            @test _val(r) ≈ 2 * xq + 1
            @test _val(r) ≈ r_ref
        end
        @testset "quadratic" begin
            r = quadratic_interp(x_vec, y_generic, xq)
            r_ref = quadratic_interp(x_vec, y_generic_flat, xq)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "cubic" begin
            r = cubic_interp(x_vec, y_generic, xq)
            r_ref = cubic_interp(x_vec, y_generic_flat, xq)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
    end

    @testset "10. One-shot API — vector target" begin
        @testset "constant" begin
            r = constant_interp(x_vec, y_generic, xq_vec)
            r_ref = constant_interp(x_vec, y_generic_flat, xq_vec)
            @test eltype(r) === MyDuck
            @test length(r) == 3
            @test _val.(r) ≈ r_ref
        end
        @testset "linear" begin
            r = linear_interp(x_vec, y_linear, xq_vec)
            r_ref = linear_interp(x_vec, y_linear_flat, xq_vec)
            @test eltype(r) === MyDuck
            for (i, q) in enumerate(xq_vec)
                @test _val(r[i]) ≈ 2q + 1
            end
            @test _val.(r) ≈ r_ref
        end
        @testset "quadratic" begin
            r = quadratic_interp(x_vec, y_generic, xq_vec)
            r_ref = quadratic_interp(x_vec, y_generic_flat, xq_vec)
            @test eltype(r) === MyDuck
            @test _val.(r) ≈ r_ref
        end
        @testset "cubic" begin
            r = cubic_interp(x_vec, y_generic, xq_vec)
            r_ref = cubic_interp(x_vec, y_generic_flat, xq_vec)
            @test eltype(r) === MyDuck
            @test _val.(r) ≈ r_ref
        end
    end

    @testset "11. One-shot API — with deriv kwarg" begin
        @testset "linear deriv1" begin
            r = linear_interp(x_vec, y_linear, xq; deriv = DerivOp(1))
            r_ref = linear_interp(x_vec, y_linear_flat, xq; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ 2.0
            @test _val(r) ≈ r_ref
        end
        @testset "cubic deriv1" begin
            r = cubic_interp(x_vec, y_generic, xq; deriv = DerivOp(1))
            r_ref = cubic_interp(x_vec, y_generic_flat, xq; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "cubic deriv2" begin
            r = cubic_interp(x_vec, y_generic, xq; deriv = DerivOp(2))
            r_ref = cubic_interp(x_vec, y_generic_flat, xq; deriv = DerivOp(2))
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "quadratic deriv1" begin
            r = quadratic_interp(x_vec, y_generic, xq; deriv = DerivOp(1))
            r_ref = quadratic_interp(x_vec, y_generic_flat, xq; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
    end

    # ================================================================
    # SECTION 6: EXTRAPOLATION MODES
    # ================================================================
    @testset "12. Extrapolation modes" begin
        xq_left = -0.5
        xq_right = 6.5

        @testset "NoExtrap throws" begin
            itp = linear_interp(x_vec, y_generic; extrap = NoExtrap())
            @test_throws DomainError itp(xq_left)
            @test_throws DomainError itp(xq_right)
        end

        @testset "ClampExtrap" begin
            itp = linear_interp(x_vec, y_generic; extrap = ClampExtrap())
            itp_ref = linear_interp(x_vec, y_generic_flat; extrap = ClampExtrap())
            @test (@inferred itp(xq_left)) isa MyDuck
            @test (@inferred itp(xq_right)) isa MyDuck
            @test _val(itp(xq_left)) == _val(y_generic[1])
            @test _val(itp(xq_right)) == _val(y_generic[end])
            @test _val(itp(xq_left)) ≈ itp_ref(xq_left)
            @test _val(itp(xq_right)) ≈ itp_ref(xq_right)
        end

        @testset "ExtendExtrap" begin
            itp = linear_interp(x_vec, y_generic; extrap = ExtendExtrap())
            itp_ref = linear_interp(x_vec, y_generic_flat; extrap = ExtendExtrap())
            @test (@inferred itp(xq_left)) isa MyDuck
            @test (@inferred itp(xq_right)) isa MyDuck
            @test _val(itp(xq_left)) ≈ itp_ref(xq_left)
            @test _val(itp(xq_right)) ≈ itp_ref(xq_right)
        end

        @testset "ExtendExtrap cubic" begin
            itp = cubic_interp(x_vec, y_generic; extrap = ExtendExtrap())
            itp_ref = cubic_interp(x_vec, y_generic_flat; extrap = ExtendExtrap())
            @test (@inferred itp(xq_left)) isa MyDuck
            @test (@inferred itp(xq_right)) isa MyDuck
            @test _val(itp(xq_left)) ≈ itp_ref(xq_left)
            @test _val(itp(xq_right)) ≈ itp_ref(xq_right)
        end

        @testset "ClampExtrap quadratic" begin
            itp = quadratic_interp(x_vec, y_generic; extrap = ClampExtrap())
            itp_ref = quadratic_interp(x_vec, y_generic_flat; extrap = ClampExtrap())
            @test (@inferred itp(xq_left)) isa MyDuck
            @test (@inferred itp(xq_right)) isa MyDuck
            @test _val(itp(xq_left)) ≈ itp_ref(xq_left)
            @test _val(itp(xq_right)) ≈ itp_ref(xq_right)
        end
    end

    # ================================================================
    # SECTION 7: SERIES
    # ================================================================
    @testset "13. Series — all methods" begin
        y1 = MyDuck.([1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5])
        y2 = MyDuck.([2.0, 1.0, 5.0, 3.0, 4.0, 1.0, 3.5])
        y3 = MyDuck.([3.0, 2.0, 1.0, 4.0, 5.0, 2.0, 4.5])
        s = Series(y1, y2, y3)
        y1_flat = _val.(y1)
        y2_flat = _val.(y2)
        y3_flat = _val.(y3)
        s_flat = Series(y1_flat, y2_flat, y3_flat)

        for (name, fn) in [
                ("constant", constant_interp), ("linear", linear_interp),
                ("quadratic", quadratic_interp), ("cubic", cubic_interp),
            ]
            @testset "$name" begin
                sitp = fn(x_vec, s)
                sitp_ref = fn(x_vec, s_flat)
                result = sitp(xq)
                result_ref = sitp_ref(xq)
                @test length(result) == 3
                @test eltype(result) === MyDuck
                @test _val.(result) ≈ result_ref
            end
        end

        @testset "cubic Series with ZeroCurvBC" begin
            sitp = cubic_interp(x_vec, s; bc = ZeroCurvBC())
            sitp_ref = cubic_interp(x_vec, s_flat; bc = ZeroCurvBC())
            result = sitp(xq)
            result_ref = sitp_ref(xq)
            @test length(result) == 3
            @test eltype(result) === MyDuck
            @test _val.(result) ≈ result_ref
        end

        @testset "quadratic Series with Deriv1(Tv)" begin
            sitp = quadratic_interp(x_vec, s; bc = Left(Deriv1(MyDuck(0.0))))
            sitp_ref = quadratic_interp(x_vec, s_flat; bc = Left(Deriv1(0.0)))
            result = sitp(xq)
            result_ref = sitp_ref(xq)
            @test eltype(result) === MyDuck
            @test _val.(result) ≈ result_ref
        end
    end

    # ================================================================
    # SECTION 8: SEARCH POLICIES
    # ================================================================
    @testset "14. Search policies" begin
        for sp in [BinarySearch(), LinearBinarySearch(), AutoSearch()]
            @testset "$(typeof(sp))" begin
                itp = linear_interp(x_vec, y_linear; search = sp)
                itp_ref = linear_interp(x_vec, y_linear_flat; search = sp)
                @test (@inferred itp(xq)) isa MyDuck
                @test _val(itp(xq)) ≈ itp_ref(xq)
            end
        end
    end

    # ================================================================
    # SECTION 9: GRID POINT EXACTNESS
    # ================================================================
    @testset "15. Grid point exactness" begin
        @testset "linear" begin
            itp = linear_interp(x_vec, y_linear)
            itp_ref = linear_interp(x_vec, y_linear_flat)
            for i in eachindex(x_vec)
                @test _val(itp(x_vec[i])) ≈ _val(y_linear[i])
                @test _val(itp(x_vec[i])) ≈ itp_ref(x_vec[i])
            end
        end
        @testset "quadratic" begin
            itp = quadratic_interp(x_vec, y_generic)
            itp_ref = quadratic_interp(x_vec, y_generic_flat)
            for i in eachindex(x_vec)
                @test _val(itp(x_vec[i])) ≈ _val(y_generic[i])
                @test _val(itp(x_vec[i])) ≈ itp_ref(x_vec[i])
            end
        end
        @testset "cubic" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            for i in eachindex(x_vec)
                @test _val(itp(x_vec[i])) ≈ _val(y_generic[i])
                @test _val(itp(x_vec[i])) ≈ itp_ref(x_vec[i])
            end
        end
    end

    # ================================================================
    # SECTION 10: ND DERIVATIVES
    # ================================================================
    @testset "16. ND derivatives" begin
        @testset "2D linear deriv" begin
            itp = linear_interp((xg, yg), data_2d)
            itp_ref = linear_interp((xg, yg), data_2d_flat)
            # ∂f/∂x at (1.5,1.5): f = x + 2y → ∂f/∂x = 1
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ 1.0
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            # ∂f/∂y: f = x + 2y → ∂f/∂y = 2
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ 2.0
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end

        @testset "2D cubic deriv" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end

        @testset "2D quadratic deriv" begin
            itp = quadratic_interp((xg, yg), data_2d)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat)
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end
    end

    # ================================================================
    # SECTION 11: ND INTEGRATION
    # ================================================================
    @testset "17. ND Integration" begin
        @testset "2D linear" begin
            itp = linear_interp((xg, yg), data_2d)
            itp_ref = linear_interp((xg, yg), data_2d_flat)
            r = integrate(itp, (0.5, 0.5), (2.5, 2.5))
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, (0.5, 0.5), (2.5, 2.5))
        end

        @testset "2D cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            r = integrate(itp, (0.5, 0.5), (2.5, 2.5))
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, (0.5, 0.5), (2.5, 2.5))
        end

        @testset "2D quadratic" begin
            itp = quadratic_interp((xg, yg), data_2d)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat)
            r = integrate(itp, (0.5, 0.5), (2.5, 2.5))
            @test r isa MyDuck
            @test _val(r) ≈ integrate(itp_ref, (0.5, 0.5), (2.5, 2.5))
        end
    end

    # ================================================================
    # SECTION 12: SIDE SELECTION (constant only)
    # ================================================================
    @testset "18. Constant side selection" begin
        for side in [LeftSide(), RightSide(), NearestSide()]
            @testset "$(typeof(side))" begin
                itp = constant_interp(x_vec, y_generic; side = side)
                itp_ref = constant_interp(x_vec, y_generic_flat; side = side)
                @test (@inferred itp(xq)) isa MyDuck
                @test _val(itp(xq)) ≈ itp_ref(xq)
            end
        end
    end

    # ================================================================
    # SECTION 13: IN-PLACE ONE-SHOT API
    # ================================================================
    @testset "19. In-place one-shot" begin
        out = Vector{MyDuck}(undef, length(xq_vec))
        out_ref = Vector{Float64}(undef, length(xq_vec))

        @testset "linear!" begin
            linear_interp!(out, x_vec, y_linear, xq_vec)
            linear_interp!(out_ref, x_vec, y_linear_flat, xq_vec)
            @test eltype(out) === MyDuck
            for (i, q) in enumerate(xq_vec)
                @test _val(out[i]) ≈ 2q + 1
            end
            @test _val.(out) ≈ out_ref
        end

        @testset "cubic!" begin
            cubic_interp!(out, x_vec, y_generic, xq_vec)
            cubic_interp!(out_ref, x_vec, y_generic_flat, xq_vec)
            @test eltype(out) === MyDuck
            @test _val.(out) ≈ out_ref
        end

        @testset "quadratic!" begin
            quadratic_interp!(out, x_vec, y_generic, xq_vec)
            quadratic_interp!(out_ref, x_vec, y_generic_flat, xq_vec)
            @test eltype(out) === MyDuck
            @test _val.(out) ≈ out_ref
        end

        @testset "constant!" begin
            constant_interp!(out, x_vec, y_generic, xq_vec)
            constant_interp!(out_ref, x_vec, y_generic_flat, xq_vec)
            @test eltype(out) === MyDuck
            @test _val.(out) ≈ out_ref
        end
    end

    # ================================================================
    # SECTION 14: DERIV + EXTRAP COMBINATIONS
    # ================================================================
    @testset "20. Deriv + Extrap combos" begin
        @testset "cubic deriv1 + ExtendExtrap" begin
            itp = cubic_interp(x_vec, y_generic; extrap = ExtendExtrap())
            itp_ref = cubic_interp(x_vec, y_generic_flat; extrap = ExtendExtrap())
            r = itp(-0.5; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(-0.5; deriv = DerivOp(1))
            r = itp(6.5; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(6.5; deriv = DerivOp(1))
        end

        @testset "cubic deriv2 + ClampExtrap" begin
            itp = cubic_interp(x_vec, y_generic; extrap = ClampExtrap())
            itp_ref = cubic_interp(x_vec, y_generic_flat; extrap = ClampExtrap())
            # ClampExtrap with deriv → 0 (derivative of constant is 0)
            r = itp(-0.5; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(-0.5; deriv = DerivOp(1))
        end

        @testset "linear deriv1 + ExtendExtrap" begin
            itp = linear_interp(x_vec, y_linear; extrap = ExtendExtrap())
            itp_ref = linear_interp(x_vec, y_linear_flat; extrap = ExtendExtrap())
            r = itp(-0.5; deriv = DerivOp(1))
            @test r isa MyDuck
            @test _val(r) ≈ 2.0
            @test _val(r) ≈ itp_ref(-0.5; deriv = DerivOp(1))
        end
    end

    # ================================================================
    # SECTION 15: DERIV + BC COMBINATIONS
    # ================================================================
    @testset "21. Deriv + BC combos" begin
        @testset "cubic ZeroCurvBC + deriv1" begin
            itp = cubic_interp(x_vec, y_generic; bc = ZeroCurvBC())
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = ZeroCurvBC())
            @test (@inferred itp(xq; deriv = DerivOp(1))) isa MyDuck
            @test _val(itp(xq; deriv = DerivOp(1))) ≈ itp_ref(xq; deriv = DerivOp(1))
        end
        @testset "cubic ZeroSlopeBC + deriv1" begin
            itp = cubic_interp(x_vec, y_generic; bc = ZeroSlopeBC())
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = ZeroSlopeBC())
            @test (@inferred itp(xq; deriv = DerivOp(1))) isa MyDuck
            @test _val(itp(xq; deriv = DerivOp(1))) ≈ itp_ref(xq; deriv = DerivOp(1))
        end
        @testset "cubic Deriv1(Tv) + deriv2" begin
            itp = cubic_interp(x_vec, y_generic; bc = Deriv1(MyDuck(0.0)))
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = Deriv1(0.0))
            @test (@inferred itp(xq; deriv = DerivOp(2))) isa MyDuck
            @test _val(itp(xq; deriv = DerivOp(2))) ≈ itp_ref(xq; deriv = DerivOp(2))
        end
        @testset "quadratic Deriv1(Tv) + deriv1" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Left(Deriv1(MyDuck(0.0))))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Left(Deriv1(0.0)))
            @test (@inferred itp(xq; deriv = DerivOp(1))) isa MyDuck
            @test _val(itp(xq; deriv = DerivOp(1))) ≈ itp_ref(xq; deriv = DerivOp(1))
        end
    end

    # ================================================================
    # SECTION 16: EDGE CASES
    # ================================================================
    @testset "22. Edge cases" begin
        @testset "query at left boundary" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            @test _val(itp(x_vec[1])) ≈ _val(y_generic[1])
            @test _val(itp(x_vec[1])) ≈ itp_ref(x_vec[1])
        end
        @testset "query at right boundary" begin
            itp = cubic_interp(x_vec, y_generic)
            itp_ref = cubic_interp(x_vec, y_generic_flat)
            @test _val(itp(x_vec[end])) ≈ _val(y_generic[end])
            @test _val(itp(x_vec[end])) ≈ itp_ref(x_vec[end])
        end
        @testset "minimum grid (3 points) — quadratic" begin
            x3 = [0.0, 1.0, 2.0]
            y3 = MyDuck.([1.0, 3.0, 2.0])
            y3_flat = _val.(y3)
            itp = quadratic_interp(x3, y3)
            itp_ref = quadratic_interp(x3, y3_flat)
            @test (@inferred itp(0.5)) isa MyDuck
            @test _val(itp(0.5)) ≈ itp_ref(0.5)
        end
        @testset "minimum grid (4 points) — cubic" begin
            x4 = [0.0, 1.0, 2.0, 3.0]
            y4 = MyDuck.([1.0, 3.0, 2.0, 4.0])
            y4_flat = _val.(y4)
            itp = cubic_interp(x4, y4)
            itp_ref = cubic_interp(x4, y4_flat)
            @test (@inferred itp(1.5)) isa MyDuck
            @test _val(itp(1.5)) ≈ itp_ref(1.5)
        end
        @testset "minimum grid (2 points) — linear" begin
            x2 = [0.0, 1.0]
            y2 = MyDuck.([1.0, 3.0])
            y2_flat = _val.(y2)
            itp = linear_interp(x2, y2)
            itp_ref = linear_interp(x2, y2_flat)
            r = itp(0.5)
            @test r isa MyDuck
            @test _val(r) ≈ 2.0
            @test _val(r) ≈ itp_ref(0.5)
        end
    end

end

# ================================================================
# SECTION 17: ND ONE-SHOT API — scalar, SoA batch, AoS batch
# ================================================================
@testitem "Duck Typing: ND Construction & BCs" setup = [DuckTypeSetup] begin
    @testset "23. ND One-shot API" begin
        qx_nd = [0.5, 1.5, 2.5]
        qy_nd = [0.5, 1.5, 2.5]
        queries_aos = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]

        @testset "scalar query" begin
            for (name, fn) in [
                    ("constant", constant_interp), ("linear", linear_interp),
                    ("quadratic", quadratic_interp), ("cubic", cubic_interp),
                ]
                @testset "$name" begin
                    r = fn((xg, yg), data_2d, q2d)
                    r_ref = fn((xg, yg), data_2d_flat, q2d)
                    @test r isa MyDuck
                    @test _val(r) ≈ r_ref
                end
            end
        end

        @testset "scalar with BCs" begin
            @testset "cubic ZeroCurvBC" begin
                r = cubic_interp((xg, yg), data_2d, q2d; bc = ZeroCurvBC())
                r_ref = cubic_interp((xg, yg), data_2d_flat, q2d; bc = ZeroCurvBC())
                @test r isa MyDuck
                @test _val(r) ≈ r_ref
            end
            @testset "cubic ZeroSlopeBC" begin
                r = cubic_interp((xg, yg), data_2d, q2d; bc = ZeroSlopeBC())
                r_ref = cubic_interp((xg, yg), data_2d_flat, q2d; bc = ZeroSlopeBC())
                @test r isa MyDuck
                @test _val(r) ≈ r_ref
            end
            @testset "quadratic ZeroCurvBC" begin
                r = quadratic_interp((xg, yg), data_2d, q2d; bc = ZeroCurvBC())
                r_ref = quadratic_interp((xg, yg), data_2d_flat, q2d; bc = ZeroCurvBC())
                @test r isa MyDuck
                @test _val(r) ≈ r_ref
            end
        end

        @testset "SoA batch" begin
            for (name, fn) in [
                    ("constant", constant_interp), ("linear", linear_interp),
                    ("quadratic", quadratic_interp), ("cubic", cubic_interp),
                ]
                @testset "$name" begin
                    r = fn((xg, yg), data_2d, (qx_nd, qy_nd))
                    r_ref = fn((xg, yg), data_2d_flat, (qx_nd, qy_nd))
                    @test eltype(r) === MyDuck
                    @test length(r) == 3
                    @test _val.(r) ≈ r_ref
                end
            end
        end

        @testset "AoS batch" begin
            for (name, fn) in [
                    ("constant", constant_interp), ("linear", linear_interp),
                    ("quadratic", quadratic_interp), ("cubic", cubic_interp),
                ]
                @testset "$name" begin
                    r = fn((xg, yg), data_2d, queries_aos)
                    r_ref = fn((xg, yg), data_2d_flat, queries_aos)
                    @test eltype(r) === MyDuck
                    @test length(r) == 3
                    @test _val.(r) ≈ r_ref
                end
            end
        end
    end

    # ================================================================
    # SECTION 18: ND IN-PLACE ONE-SHOT
    # ================================================================
    @testset "24. ND In-place one-shot" begin
        qx_nd = [0.5, 1.5, 2.5]
        qy_nd = [0.5, 1.5, 2.5]
        queries_aos = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]
        out_nd = Vector{MyDuck}(undef, 3)
        out_nd_ref = Vector{Float64}(undef, 3)

        @testset "SoA batch" begin
            for (name, fn!) in [
                    ("linear!", linear_interp!), ("cubic!", cubic_interp!),
                    ("quadratic!", quadratic_interp!), ("constant!", constant_interp!),
                ]
                @testset "$name" begin
                    fn!(out_nd, (xg, yg), data_2d, (qx_nd, qy_nd))
                    fn!(out_nd_ref, (xg, yg), data_2d_flat, (qx_nd, qy_nd))
                    @test eltype(out_nd) === MyDuck
                    @test _val.(out_nd) ≈ out_nd_ref
                end
            end
        end

        @testset "AoS batch" begin
            for (name, fn!) in [
                    ("linear!", linear_interp!), ("cubic!", cubic_interp!),
                    ("quadratic!", quadratic_interp!), ("constant!", constant_interp!),
                ]
                @testset "$name" begin
                    fn!(out_nd, (xg, yg), data_2d, queries_aos)
                    fn!(out_nd_ref, (xg, yg), data_2d_flat, queries_aos)
                    @test eltype(out_nd) === MyDuck
                    @test _val.(out_nd) ≈ out_nd_ref
                end
            end
        end
    end

    # ================================================================
    # SECTION 19: ND BCs EXPANDED — cubic
    # ================================================================
    @testset "25. ND BCs — Cubic expanded" begin
        @testset "ZeroSlopeBC symmetric" begin
            itp = cubic_interp((xg, yg), data_2d; bc = ZeroSlopeBC())
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = ZeroSlopeBC())
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "ZeroSlopeBC + ZeroCurvBC mixed per-axis" begin
            itp = cubic_interp((xg, yg), data_2d; bc = (ZeroSlopeBC(), ZeroCurvBC()))
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = (ZeroSlopeBC(), ZeroCurvBC()))
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "CubicFit per-axis symmetric" begin
            itp = cubic_interp((xg, yg), data_2d; bc = CubicFit())
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = CubicFit())
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "CubicFit + ZeroSlopeBC mixed" begin
            itp = cubic_interp((xg, yg), data_2d; bc = (CubicFit(), ZeroSlopeBC()))
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = (CubicFit(), ZeroSlopeBC()))
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "PeriodicBC(:exclusive) in dim 1" begin
            xp = range(0.0, 3.0, 4)   # must be Range (not collected) for PeriodicBC auto-period
            data_pe = [MyDuck(2yj + 1.0) for xi in xp, yj in yg_r]
            data_pe_flat = _val.(data_pe)
            itp = cubic_interp((xp, yg_r), data_pe; bc = (PeriodicBC(endpoint = :exclusive), CubicFit()))
            itp_ref = cubic_interp((xp, yg_r), data_pe_flat; bc = (PeriodicBC(endpoint = :exclusive), CubicFit()))
            @test (@inferred itp((0.5, 1.5))) isa MyDuck
            @test _val(itp((0.5, 1.5))) ≈ itp_ref((0.5, 1.5))
        end
        @testset "PeriodicBC(:exclusive) in both dims" begin
            xp = range(0.0, 3.0, 4)
            yp = range(0.0, 3.0, 4)
            data_pp = [MyDuck(1.0 + 0.5 * (xi + yj)) for xi in xp, yj in yp]
            data_pp_flat = _val.(data_pp)
            itp = cubic_interp((xp, yp), data_pp; bc = (PeriodicBC(endpoint = :exclusive), PeriodicBC(endpoint = :exclusive)))
            itp_ref = cubic_interp((xp, yp), data_pp_flat; bc = (PeriodicBC(endpoint = :exclusive), PeriodicBC(endpoint = :exclusive)))
            @test (@inferred itp((0.5, 0.5))) isa MyDuck
            @test _val(itp((0.5, 0.5))) ≈ itp_ref((0.5, 0.5))
        end
        @testset "ZeroSlopeBC + deriv" begin
            itp = cubic_interp((xg, yg), data_2d; bc = ZeroSlopeBC())
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = ZeroSlopeBC())
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end
    end

    # ================================================================
    # SECTION 20: ND BCs EXPANDED — quadratic
    # ================================================================
    @testset "26. ND BCs — Quadratic expanded" begin
        @testset "ZeroSlopeBC symmetric" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = ZeroSlopeBC())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = ZeroSlopeBC())
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "MinCurvFit symmetric" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = MinCurvFit())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = MinCurvFit())
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "ZeroCurvBC symmetric" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = ZeroCurvBC())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = ZeroCurvBC())
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "Left(QuadraticFit()) per-axis" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = (Left(QuadraticFit()), Left(QuadraticFit())))
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = (Left(QuadraticFit()), Left(QuadraticFit())))
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "Right(QuadraticFit()) per-axis" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = (Right(QuadraticFit()), Right(QuadraticFit())))
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = (Right(QuadraticFit()), Right(QuadraticFit())))
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "Left(Deriv1(Tv)) per-axis" begin
            bc = (Left(Deriv1(MyDuck(0.0))), Left(Deriv1(MyDuck(0.0))))
            bc_ref = (Left(Deriv1(0.0)), Left(Deriv1(0.0)))
            itp = quadratic_interp((xg, yg), data_2d; bc = bc)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = bc_ref)
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "mixed: Left(Deriv1) + ZeroCurvBC" begin
            bc = (Left(Deriv1(MyDuck(0.0))), ZeroCurvBC())
            bc_ref = (Left(Deriv1(0.0)), ZeroCurvBC())
            itp = quadratic_interp((xg, yg), data_2d; bc = bc)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = bc_ref)
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "ZeroCurvBC + deriv" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = ZeroCurvBC())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = ZeroCurvBC())
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end
        @testset "ZeroCurvBC + Left(QuadraticFit()) mixed + deriv" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = (ZeroCurvBC(), Left(QuadraticFit())))
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = (ZeroCurvBC(), Left(QuadraticFit())))
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
        end
        @testset "ZeroSlopeBC + deriv" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = ZeroSlopeBC())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = ZeroSlopeBC())
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end
        @testset "MinCurvFit + deriv" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = MinCurvFit())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = MinCurvFit())
            r = itp(q2d; deriv = DerivOp(1, 0))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(1, 0))
            r = itp(q2d; deriv = DerivOp(0, 1))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref(q2d; deriv = DerivOp(0, 1))
        end
        @testset "ZeroSlopeBC + MinCurvFit mixed" begin
            itp = quadratic_interp((xg, yg), data_2d; bc = (ZeroSlopeBC(), MinCurvFit()))
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; bc = (ZeroSlopeBC(), MinCurvFit()))
            @test (@inferred itp(q2d)) isa MyDuck
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
    end

end

# ================================================================
# SECTION 21: VECTOR CALCULUS — gradient, hessian, laplacian
# ================================================================
@testitem "Duck Typing: ND Advanced & Type Stability" setup = [DuckTypeSetup] begin
    @testset "27. Vector Calculus" begin
        @testset "gradient — cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            g = gradient(itp, q2d)
            g_ref = gradient(itp_ref, q2d)
            @test length(g) == 2
            @test g[1] isa MyDuck
            @test g[2] isa MyDuck
            @test _val(g[1]) ≈ g_ref[1]
            @test _val(g[2]) ≈ g_ref[2]
        end
        @testset "gradient — quadratic" begin
            itp = quadratic_interp((xg, yg), data_2d)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat)
            g = gradient(itp, q2d)
            g_ref = gradient(itp_ref, q2d)
            @test g[1] isa MyDuck
            @test _val(g[1]) ≈ g_ref[1]
            @test _val(g[2]) ≈ g_ref[2]
        end
        @testset "gradient — linear" begin
            itp = linear_interp((xg, yg), data_2d)
            itp_ref = linear_interp((xg, yg), data_2d_flat)
            g = gradient(itp, q2d)
            g_ref = gradient(itp_ref, q2d)
            @test g[1] isa MyDuck
            # f = x + 2y → ∂f/∂x = 1, ∂f/∂y = 2
            @test _val(g[1]) ≈ 1.0
            @test _val(g[2]) ≈ 2.0
            @test _val(g[1]) ≈ g_ref[1]
            @test _val(g[2]) ≈ g_ref[2]
        end
        @testset "gradient! — cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            G = Vector{MyDuck}(undef, 2)
            G_ref = zeros(2)
            gradient!(G, itp, q2d)
            gradient!(G_ref, itp_ref, q2d)
            @test G[1] isa MyDuck
            @test _val(G[1]) ≈ G_ref[1]
            @test _val(G[2]) ≈ G_ref[2]
        end
        @testset "hessian — cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            H = hessian(itp, q2d)
            H_ref = hessian(itp_ref, q2d)
            @test size(H) == (2, 2)
            @test H[1, 1] isa MyDuck
            @test _val(H[1, 1]) ≈ H_ref[1, 1]
            # Off-diagonal: data is linear (xi+2yj) so ∂²f/∂x∂y ≈ 0.
            # FMA contraction produces different near-zero rounding artifacts; use atol.
            @test isapprox(_val(H[1, 2]), H_ref[1, 2]; atol = 1.0e-14)
            @test _val(H[2, 2]) ≈ H_ref[2, 2]
        end
        @testset "hessian! — cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            H = Matrix{MyDuck}(undef, 2, 2)
            H_ref = zeros(2, 2)
            hessian!(H, itp, q2d)
            hessian!(H_ref, itp_ref, q2d)
            @test H[1, 1] isa MyDuck
            @test _val(H[1, 1]) ≈ H_ref[1, 1]
            @test isapprox(_val(H[2, 1]), H_ref[2, 1]; atol = 1.0e-14)
        end
        @testset "laplacian — cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            lap = laplacian(itp, q2d)
            lap_ref = laplacian(itp_ref, q2d)
            @test lap isa MyDuck
            @test _val(lap) ≈ lap_ref
        end
        @testset "laplacian — quadratic" begin
            itp = quadratic_interp((xg, yg), data_2d)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat)
            lap = laplacian(itp, q2d)
            lap_ref = laplacian(itp_ref, q2d)
            @test lap isa MyDuck
            @test _val(lap) ≈ lap_ref
        end
        @testset "laplacian — linear" begin
            itp = linear_interp((xg, yg), data_2d)
            itp_ref = linear_interp((xg, yg), data_2d_flat)
            lap = laplacian(itp, q2d)
            lap_ref = laplacian(itp_ref, q2d)
            @test lap isa MyDuck
            @test _val(lap) ≈ lap_ref
        end
        @testset "gradient with ZeroSlopeBC" begin
            itp = cubic_interp((xg, yg), data_2d; bc = ZeroSlopeBC())
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = ZeroSlopeBC())
            g = gradient(itp, q2d)
            g_ref = gradient(itp_ref, q2d)
            @test g[1] isa MyDuck
            @test _val(g[1]) ≈ g_ref[1]
            @test _val(g[2]) ≈ g_ref[2]
        end
    end

    # ================================================================
    # SECTION 22: FULL-DOMAIN INTEGRATION — ND
    # ================================================================
    @testset "28. Full-domain integration — ND" begin
        @testset "2D linear" begin
            itp = linear_interp((xg, yg), data_2d)
            itp_ref = linear_interp((xg, yg), data_2d_flat)
            r = integrate(itp)
            r_ref = integrate(itp_ref)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "2D cubic" begin
            itp = cubic_interp((xg, yg), data_2d)
            itp_ref = cubic_interp((xg, yg), data_2d_flat)
            r = integrate(itp)
            r_ref = integrate(itp_ref)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "2D quadratic" begin
            itp = quadratic_interp((xg, yg), data_2d)
            itp_ref = quadratic_interp((xg, yg), data_2d_flat)
            r = integrate(itp)
            r_ref = integrate(itp_ref)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
        @testset "2D constant" begin
            itp = constant_interp((xg, yg), data_2d)
            itp_ref = constant_interp((xg, yg), data_2d_flat)
            r = integrate(itp)
            r_ref = integrate(itp_ref)
            @test r isa MyDuck
            @test _val(r) ≈ r_ref
        end
    end

    # ================================================================
    # SECTION 23: ND EXTRAPOLATION — per-axis modes
    # ================================================================
    @testset "29. ND Extrapolation" begin
        @testset "2D cubic ClampExtrap" begin
            itp = cubic_interp((xg, yg), data_2d; extrap = ClampExtrap())
            itp_ref = cubic_interp((xg, yg), data_2d_flat; extrap = ClampExtrap())
            r = itp((-0.5, 1.5))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref((-0.5, 1.5))
            r = itp((1.5, 3.5))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref((1.5, 3.5))
        end
        @testset "2D linear ClampExtrap" begin
            itp = linear_interp((xg, yg), data_2d; extrap = ClampExtrap())
            itp_ref = linear_interp((xg, yg), data_2d_flat; extrap = ClampExtrap())
            r = itp((-0.5, 1.5))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref((-0.5, 1.5))
        end
        @testset "2D cubic ExtendExtrap" begin
            itp = cubic_interp((xg, yg), data_2d; extrap = ExtendExtrap())
            itp_ref = cubic_interp((xg, yg), data_2d_flat; extrap = ExtendExtrap())
            r = itp((-0.5, 1.5))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref((-0.5, 1.5))
        end
        @testset "2D quadratic ClampExtrap" begin
            itp = quadratic_interp((xg, yg), data_2d; extrap = ClampExtrap())
            itp_ref = quadratic_interp((xg, yg), data_2d_flat; extrap = ClampExtrap())
            r = itp((-0.5, 1.5))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref((-0.5, 1.5))
        end
        @testset "2D per-axis extrap (ClampExtrap, ExtendExtrap)" begin
            itp = cubic_interp((xg, yg), data_2d; extrap = (ClampExtrap(), ExtendExtrap()))
            itp_ref = cubic_interp((xg, yg), data_2d_flat; extrap = (ClampExtrap(), ExtendExtrap()))
            r = itp((-0.5, 3.5))
            @test r isa MyDuck
            @test _val(r) ≈ itp_ref((-0.5, 3.5))
        end
    end

    # ================================================================
    # SECTION 24: TYPE STABILITY — @inferred checks
    # ================================================================
    # @inferred requires direct function calls (not loop variables) because
    # Julia cannot infer return types through runtime-dispatched callables.
    @testset "30. Type Stability (@inferred)" begin
        @testset "1D scalar one-shot" begin
            @test @inferred(constant_interp(x_vec, y_generic, xq)) isa MyDuck
            @test @inferred(linear_interp(x_vec, y_linear, xq)) isa MyDuck
            @test @inferred(quadratic_interp(x_vec, y_generic, xq)) isa MyDuck
            @test @inferred(cubic_interp(x_vec, y_generic, xq)) isa MyDuck
        end

        @testset "1D vector one-shot" begin
            @test eltype(@inferred(constant_interp(x_vec, y_generic, xq_vec))) === MyDuck
            @test eltype(@inferred(linear_interp(x_vec, y_linear, xq_vec))) === MyDuck
            @test eltype(@inferred(quadratic_interp(x_vec, y_generic, xq_vec))) === MyDuck
            @test eltype(@inferred(cubic_interp(x_vec, y_generic, xq_vec))) === MyDuck
        end

        @testset "ND scalar one-shot" begin
            @test @inferred(constant_interp((xg, yg), data_2d, q2d)) isa MyDuck
            @test @inferred(linear_interp((xg, yg), data_2d, q2d)) isa MyDuck
            @test @inferred(quadratic_interp((xg, yg), data_2d, q2d)) isa MyDuck
            @test @inferred(cubic_interp((xg, yg), data_2d, q2d)) isa MyDuck
        end

        @testset "ND SoA batch" begin
            qx_ts = [0.5, 1.5, 2.5]
            qy_ts = [0.5, 1.5, 2.5]
            @test eltype(@inferred(constant_interp((xg, yg), data_2d, (qx_ts, qy_ts)))) === MyDuck
            @test eltype(@inferred(linear_interp((xg, yg), data_2d, (qx_ts, qy_ts)))) === MyDuck
            @test eltype(@inferred(quadratic_interp((xg, yg), data_2d, (qx_ts, qy_ts)))) === MyDuck
            @test eltype(@inferred(cubic_interp((xg, yg), data_2d, (qx_ts, qy_ts)))) === MyDuck
        end

        @testset "ND AoS batch" begin
            q_aos_ts = [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)]
            @test eltype(@inferred(constant_interp((xg, yg), data_2d, q_aos_ts))) === MyDuck
            @test eltype(@inferred(linear_interp((xg, yg), data_2d, q_aos_ts))) === MyDuck
            @test eltype(@inferred(quadratic_interp((xg, yg), data_2d, q_aos_ts))) === MyDuck
            @test eltype(@inferred(cubic_interp((xg, yg), data_2d, q_aos_ts))) === MyDuck
        end

        @testset "scalar with BCs" begin
            @test @inferred(cubic_interp((xg, yg), data_2d, q2d; bc = ZeroCurvBC())) isa MyDuck
            @test @inferred(cubic_interp((xg, yg), data_2d, q2d; bc = ZeroSlopeBC())) isa MyDuck
            @test @inferred(quadratic_interp((xg, yg), data_2d, q2d; bc = ZeroCurvBC())) isa MyDuck
            @test @inferred(quadratic_interp((xg, yg), data_2d, q2d; bc = ZeroSlopeBC())) isa MyDuck
            @test @inferred(quadratic_interp((xg, yg), data_2d, q2d; bc = MinCurvFit())) isa MyDuck
        end

        @testset "scalar with deriv" begin
            @test @inferred(linear_interp(x_vec, y_linear, xq; deriv = DerivOp(1))) isa MyDuck
            @test @inferred(cubic_interp(x_vec, y_generic, xq; deriv = DerivOp(1))) isa MyDuck
            @test @inferred(quadratic_interp(x_vec, y_generic, xq; deriv = DerivOp(1))) isa MyDuck
        end
    end

    # ================================================================
    # SECTION 25: EXPLICIT DERIV BCs — expanded combos
    # Tests Deriv3, nonzero BC values, BCPair exotic mixes, ND per-axis
    # explicit Deriv, and 3D. These exercise the same-type Deriv BC path
    # (Deriv1(Tv(val)) where val has the same type as data values).
    # ================================================================
    @testset "31. Deriv BC — expanded combos" begin
        # --- Cubic 1D: new Deriv types + nonzero ---
        @testset "Deriv3(Tv) 1D" begin
            itp = cubic_interp(x_vec, y_generic; bc = Deriv3(MyDuck(0.0)))
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = Deriv3(0.0))
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Deriv1(Tv(1.5)) nonzero" begin
            itp = cubic_interp(x_vec, y_generic; bc = Deriv1(MyDuck(1.5)))
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = Deriv1(1.5))
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "Deriv2(Tv(-2.0)) nonzero" begin
            itp = cubic_interp(x_vec, y_generic; bc = Deriv2(MyDuck(-2.0)))
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = Deriv2(-2.0))
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end

        # --- BCPair exotic combos ---
        @testset "BCPair(Deriv1, Deriv3)" begin
            bc = BCPair(Deriv1(MyDuck(0.0)), Deriv3(MyDuck(0.0)))
            bc_ref = BCPair(Deriv1(0.0), Deriv3(0.0))
            itp = cubic_interp(x_vec, y_generic; bc = bc)
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = bc_ref)
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "BCPair(Deriv3(0.5), Deriv1(-0.5)) nonzero" begin
            bc = BCPair(Deriv3(MyDuck(0.5)), Deriv1(MyDuck(-0.5)))
            bc_ref = BCPair(Deriv3(0.5), Deriv1(-0.5))
            itp = cubic_interp(x_vec, y_generic; bc = bc)
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = bc_ref)
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end
        @testset "BCPair(Deriv2, CubicFit)" begin
            bc = BCPair(Deriv2(MyDuck(0.0)), CubicFit())
            bc_ref = BCPair(Deriv2(0.0), CubicFit())
            itp = cubic_interp(x_vec, y_generic; bc = bc)
            itp_ref = cubic_interp(x_vec, y_generic_flat; bc = bc_ref)
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end

        # --- Quadratic 1D: missing direction+type ---
        @testset "Right(Deriv2(Tv(0.5))) nonzero" begin
            itp = quadratic_interp(x_vec, y_generic; bc = Right(Deriv2(MyDuck(0.5))))
            itp_ref = quadratic_interp(x_vec, y_generic_flat; bc = Right(Deriv2(0.5)))
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end

        # --- Cubic ND: explicit Deriv per-axis (was skipped before fix) ---
        @testset "ND broadcast Deriv1(Tv)" begin
            itp = cubic_interp((xg, yg), data_2d; bc = Deriv1(MyDuck(0.0)))
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = Deriv1(0.0))
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "ND per-axis (Deriv1, Deriv2)" begin
            itp = cubic_interp(
                (xg, yg), data_2d;
                bc = (Deriv1(MyDuck(0.0)), Deriv2(MyDuck(0.0)))
            )
            itp_ref = cubic_interp(
                (xg, yg), data_2d_flat;
                bc = (Deriv1(0.0), Deriv2(0.0))
            )
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "ND mixed (Deriv1, ZeroCurvBC)" begin
            itp = cubic_interp(
                (xg, yg), data_2d;
                bc = (Deriv1(MyDuck(0.0)), ZeroCurvBC())
            )
            itp_ref = cubic_interp(
                (xg, yg), data_2d_flat;
                bc = (Deriv1(0.0), ZeroCurvBC())
            )
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end
        @testset "ND per-axis BCPair + ZeroCurvBC" begin
            bc_x = BCPair(Deriv1(MyDuck(0.0)), Deriv2(MyDuck(0.0)))
            bc_ref_x = BCPair(Deriv1(0.0), Deriv2(0.0))
            itp = cubic_interp((xg, yg), data_2d; bc = (bc_x, ZeroCurvBC()))
            itp_ref = cubic_interp((xg, yg), data_2d_flat; bc = (bc_ref_x, ZeroCurvBC()))
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end

        # --- 3D: all-different Deriv BCs ---
        @testset "3D (Deriv1, Deriv2, Deriv3)" begin
            zg = [0.0, 1.0, 2.0, 3.0]
            data_3d = [MyDuck(xi + yj + zk) for xi in xg, yj in yg, zk in zg]
            data_3d_flat = _val.(data_3d)
            bc = (Deriv1(MyDuck(0.0)), Deriv2(MyDuck(0.0)), Deriv3(MyDuck(0.0)))
            bc_ref = (Deriv1(0.0), Deriv2(0.0), Deriv3(0.0))
            itp = cubic_interp((xg, yg, zg), data_3d; bc = bc)
            itp_ref = cubic_interp((xg, yg, zg), data_3d_flat; bc = bc_ref)
            @test _val(itp((1.5, 1.5, 1.5))) ≈ itp_ref((1.5, 1.5, 1.5))
        end

        # --- Quadratic ND: mixed direction ---
        @testset "ND quadratic (Left(Deriv1), Right(Deriv2))" begin
            itp = quadratic_interp(
                (xg, yg), data_2d;
                bc = (Left(Deriv1(MyDuck(0.0))), Right(Deriv2(MyDuck(0.0))))
            )
            itp_ref = quadratic_interp(
                (xg, yg), data_2d_flat;
                bc = (Left(Deriv1(0.0)), Right(Deriv2(0.0)))
            )
            @test _val(itp(q2d)) ≈ itp_ref(q2d)
        end

        # --- PeriodicBC + Deriv mixed ND ---
        @testset "ND mixed (Periodic + Deriv1)" begin
            xp_r = range(0.0, 3.0, 4)
            data_m = [MyDuck(sin(2π * xi / 3) + yj) for xi in xp_r, yj in yg]
            data_m_flat = _val.(data_m)
            itp = cubic_interp(
                (xp_r, yg), data_m;
                bc = (PeriodicBC(endpoint = :exclusive), Deriv1(MyDuck(0.0)))
            )
            itp_ref = cubic_interp(
                (xp_r, yg), data_m_flat;
                bc = (PeriodicBC(endpoint = :exclusive), Deriv1(0.0))
            )
            @test _val(itp((0.5, 1.5))) ≈ itp_ref((0.5, 1.5))
        end
    end

    # ================================================================
    # SECTION 26: CONSTRUCTION TYPE STABILITY (@inferred)
    # Tests that interpolant construction is type-stable when BCs
    # carry duck-typed values through normalization/cache/solver.
    # ================================================================
    @testset "32. Construction @inferred" begin
        @testset "1D cubic Deriv1(Tv)" begin
            @inferred cubic_interp(x_vec, y_generic; bc = Deriv1(MyDuck(0.0)))
        end
        @testset "1D cubic Deriv3(Tv)" begin
            @inferred cubic_interp(x_vec, y_generic; bc = Deriv3(MyDuck(0.0)))
        end
        @testset "1D cubic BCPair(Deriv1,Deriv2)" begin
            @inferred cubic_interp(
                x_vec, y_generic;
                bc = BCPair(Deriv1(MyDuck(0.0)), Deriv2(MyDuck(0.0)))
            )
        end
        @testset "1D quadratic Left(Deriv1(Tv))" begin
            @inferred quadratic_interp(
                x_vec, y_generic;
                bc = Left(Deriv1(MyDuck(0.0)))
            )
        end
        @testset "ND cubic Deriv1(Tv) broadcast" begin
            @inferred cubic_interp(
                (xg, yg), data_2d;
                bc = Deriv1(MyDuck(0.0))
            )
        end
        @testset "ND cubic per-axis (Deriv1, Deriv2)" begin
            @inferred cubic_interp(
                (xg, yg), data_2d;
                bc = (Deriv1(MyDuck(0.0)), Deriv2(MyDuck(0.0)))
            )
        end
        @testset "ND quadratic Left(Deriv1(Tv))" begin
            @inferred quadratic_interp(
                (xg, yg), data_2d;
                bc = Left(Deriv1(MyDuck(0.0)))
            )
        end
    end

    # ================================================================
    # SECTION 27: CONTRACT BOUNDARIES — negative tests & promotion
    # Proves that cross-type Deriv BCs correctly fail, and that
    # standard numeric promotion is unaffected by duck-typing paths.
    # ================================================================
    @testset "33. Contract boundaries" begin
        # DuckFloat5 has no convert(DuckFloat5, Float64) → cross-type Deriv must fail
        @testset "cubic Deriv1(0.0) fails without convert" begin
            @test_throws MethodError cubic_interp(x_vec, y_generic; bc = Deriv1(0.0))
        end
        @testset "quadratic Left(Deriv1(0.0)) fails without convert" begin
            @test_throws MethodError quadratic_interp(
                x_vec, y_generic;
                bc = Left(Deriv1(0.0))
            )
        end

        # PeriodicBC :inclusive with approximate (but not exact) match
        @testset "PeriodicBC(:inclusive) approx mismatch → ArgumentError" begin
            xp = range(0.0, 6.0, 7)
            yp = MyDuck.([1.0, 3.0, 2.0, 4.0, 2.0, 3.0, 1.0 + 1.0e-14])
            @test_throws ArgumentError cubic_interp(
                xp, yp;
                bc = PeriodicBC(endpoint = :inclusive)
            )
        end

        # Standard promotion unchanged by duck-typing infrastructure
        @testset "Int → Float64 promotion" begin
            @test eltype(linear_interp([0.0, 1.0, 2.0], [10, 20, 30]).y) === Float64
        end
        @testset "Float32 → Float64 widening" begin
            @test eltype(linear_interp([0.0, 1.0, 2.0], Float32[1, 2, 3]).y) === Float64
        end
        @testset "ComplexF64 preserved" begin
            @test eltype(linear_interp([0.0, 1.0, 2.0], ComplexF64[1, 2, 3]).y) === ComplexF64
        end
    end

    # ================================================================
    # SECTION 28: FILLEXTRAP FILL VALUE — duck type support
    # Verifies that FillExtrap(Tv(val)) works with the bare-minimum
    # 5-op duck type: construction, _promote_extrap, all 4 interp
    # types (scalar + series), derivatives, ND guard, and integration.
    # ================================================================
    @testset "34. FillExtrap fill value — duck type" begin
        using FastInterpolations: _promote_extrap

        fill_val = MyDuck(999.0)
        xq_left = -0.5
        xq_right = 6.5

        # --- Construction ---
        @testset "construction with duck type" begin
            e = FillExtrap(fill_val)
            @test e isa FillExtrap{MyDuck}
            @test e.fill_value === fill_val
        end

        @testset "_promote_extrap identity for matching Tv" begin
            e = FillExtrap(fill_val)
            ep = _promote_extrap(e, MyDuck)
            @test ep isa FillExtrap{MyDuck}
            @test ep.fill_value === fill_val
        end

        @testset "_promote_extrap with Float64 fill → MethodError for duck Tv" begin
            # FillExtrap(0.0) has Float64 fill; promoting to MyDuck requires
            # convert(MyDuck, 0.0) which isn't defined → must fail at construction
            @test_throws MethodError _promote_extrap(FillExtrap(0.0), MyDuck)
        end

        # --- 1D scalar eval: all 4 interp types ---
        @testset "cubic fill value" begin
            itp = cubic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            itp_ref = cubic_interp(x_vec, y_generic_flat; extrap = FillExtrap(999.0))
            # Out-of-domain → fill value
            @test (@inferred itp(xq_left)) isa MyDuck
            @test _val(itp(xq_left)) == 999.0
            @test _val(itp(xq_right)) == 999.0
            # In-domain → normal interpolation (unchanged by fill)
            @test _val(itp(xq)) ≈ itp_ref(xq)
        end

        @testset "linear fill value" begin
            itp = linear_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @test (@inferred itp(xq_left)) isa MyDuck
            @test _val(itp(xq_left)) == 999.0
            @test _val(itp(xq_right)) == 999.0
        end

        @testset "quadratic fill value" begin
            itp = quadratic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @test (@inferred itp(xq_left)) isa MyDuck
            @test _val(itp(xq_left)) == 999.0
            @test _val(itp(xq_right)) == 999.0
        end

        @testset "constant fill value" begin
            itp = constant_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @test (@inferred itp(xq_left)) isa MyDuck
            @test _val(itp(xq_left)) == 999.0
            @test _val(itp(xq_right)) == 999.0
        end

        # --- Derivatives with fill → 0 * y_bnd (uses Int*Tv path) ---
        @testset "derivatives return zero (not fill)" begin
            itp = cubic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            r1 = itp(xq_left; deriv = DerivOp(1))
            @test r1 isa MyDuck
            @test _val(r1) == 0.0   # 0 * y_bnd, not 0 * fill_val
            r2 = itp(xq_left; deriv = DerivOp(2))
            @test r2 isa MyDuck
            @test _val(r2) == 0.0

            # Linear deriv
            itp_l = linear_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            rl = itp_l(xq_right; deriv = DerivOp(1))
            @test rl isa MyDuck
            @test _val(rl) == 0.0

            # Quadratic deriv
            itp_q = quadratic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            rq = itp_q(xq_right; deriv = DerivOp(1))
            @test rq isa MyDuck
            @test _val(rq) == 0.0
        end

        # --- Series with fill value ---
        @testset "series fill value" begin
            y1 = MyDuck.([1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5])
            y2 = MyDuck.([2.0, 1.0, 5.0, 3.0, 4.0, 1.0, 3.5])
            s = Series(y1, y2)

            for (name, fn) in [
                    ("cubic", cubic_interp), ("linear", linear_interp),
                    ("quadratic", quadratic_interp), ("constant", constant_interp),
                ]
                @testset "$name series" begin
                    sitp = fn(x_vec, s; extrap = FillExtrap(fill_val))
                    result = sitp(xq_left)
                    @test length(result) == 2
                    @test eltype(result) === MyDuck
                    @test all(r -> _val(r) == 999.0, result)

                    # In-domain: unchanged
                    result_in = sitp(xq)
                    @test eltype(result_in) === MyDuck
                    @test all(r -> _val(r) != 999.0, result_in)
                end
            end
        end

        # --- ND fill value works with duck types ---
        @testset "ND fill value with duck type" begin
            itp_c = cubic_interp(
                (xg, yg), data_2d; extrap = FillExtrap(fill_val)
            )
            @test itp_c((-0.1, 0.5)) === fill_val  # OOB → fill
            @test (@inferred itp_c((0.5, 0.5))) isa MyDuck      # in-domain → normal

            itp_l = linear_interp(
                (xg, yg), data_2d; extrap = FillExtrap(fill_val)
            )
            @test itp_l((-0.1, 0.5)) === fill_val
            @test (@inferred itp_l((0.5, 0.5))) isa MyDuck
        end

        # --- Type stability ---
        @testset "construction @inferred" begin
            @inferred cubic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @inferred linear_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @inferred quadratic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @inferred constant_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
        end

        @testset "eval @inferred" begin
            itp = cubic_interp(x_vec, y_generic; extrap = FillExtrap(fill_val))
            @test @inferred(itp(xq_left)) isa MyDuck
            @test @inferred(itp(xq)) isa MyDuck
        end

        # --- Boundary clamp (ClampExtrap()) still works with duck types ---
        @testset "boundary clamp unchanged" begin
            itp = cubic_interp(x_vec, y_generic; extrap = ClampExtrap())
            itp_ref = cubic_interp(x_vec, y_generic_flat; extrap = ClampExtrap())
            @test (@inferred itp(xq_left)) isa MyDuck
            @test _val(itp(xq_left)) ≈ itp_ref(xq_left)    # y[1]
            @test _val(itp(xq_right)) ≈ itp_ref(xq_right)  # y[end]
        end
    end

    # ================================================================
    # SECTION: ONE-SHOT SERIES — duck-typed values
    # ================================================================
    @testset "One-shot Series — duck typed" begin
        y1 = MyDuck.([1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 2.5])
        y2 = MyDuck.([2.0, 1.0, 5.0, 3.0, 4.0, 1.0, 3.5])
        y1_flat = _val.(y1)
        y2_flat = _val.(y2)
        s = Series(y1, y2)
        s_flat = Series(y1_flat, y2_flat)

        for (name, fn, fn!) in [
                ("constant", constant_interp, constant_interp!),
                ("linear", linear_interp, linear_interp!),
                ("quadratic", quadratic_interp, quadratic_interp!),
                ("cubic", cubic_interp, cubic_interp!),
            ]
            @testset "$name" begin
                # Scalar allocating — container must be typed, not Vector{Any}
                @testset "scalar" begin
                    result = fn(x_vec, s, xq)
                    ref = fn(x_vec, s_flat, xq)
                    @test length(result) == 2
                    @test eltype(result) === MyDuck
                    @test _val.(result) ≈ ref
                end

                # Scalar in-place — typed output preserved
                @testset "scalar in-place" begin
                    out = Vector{MyDuck}(undef, 2)
                    ret = fn!(out, x_vec, s, xq)
                    ref = fn(x_vec, s_flat, xq)
                    @test ret === out
                    @test eltype(out) === MyDuck
                    @test _val.(out) ≈ ref
                end

                # Vector allocating — container must be typed, not Vector{Any}
                @testset "vector" begin
                    result = fn(x_vec, s, xq_vec)
                    ref = fn(x_vec, s_flat, xq_vec)
                    @test length(result) == 2
                    for k in 1:2
                        @test eltype(result[k]) === MyDuck
                        @test _val.(result[k]) ≈ ref[k]
                    end
                end

                # Vector in-place — typed output preserved
                @testset "vector in-place" begin
                    outputs = [Vector{MyDuck}(undef, length(xq_vec)) for _ in 1:2]
                    ret = fn!(outputs, x_vec, s, xq_vec)
                    ref = fn(x_vec, s_flat, xq_vec)
                    @test ret === outputs
                    for k in 1:2
                        @test eltype(outputs[k]) === MyDuck
                        @test _val.(outputs[k]) ≈ ref[k]
                    end
                end

                # DerivOp(1) — derivative should also return typed container
                @testset "deriv=1" begin
                    result = fn(x_vec, s, xq; deriv = DerivOp(1))
                    ref = fn(x_vec, s_flat, xq; deriv = DerivOp(1))
                    @test eltype(result) === MyDuck
                    @test _val.(result) ≈ ref
                end
            end
        end

        # Matrix input form
        @testset "Matrix Series" begin
            Y = hcat(y1, y2)
            Y_flat = hcat(y1_flat, y2_flat)
            result = linear_interp(x_vec, Series(Y), xq)
            ref = linear_interp(x_vec, Series(Y_flat), xq)
            @test eltype(result) === MyDuck
            @test _val.(result) ≈ ref
        end

        # @inferred — type stability (complements eltype checks: @inferred
        # passes even for Vector{Any} since it's concrete, but catches
        # return-type instability that eltype checks miss)
        @testset "@inferred" begin
            @testset "scalar" begin
                @inferred linear_interp(x_vec, s, xq)
                @inferred constant_interp(x_vec, s, xq)
                @inferred quadratic_interp(x_vec, s, xq)
                @inferred cubic_interp(x_vec, s, xq)
            end
            @testset "scalar in-place" begin
                out = Vector{MyDuck}(undef, 2)
                @inferred linear_interp!(out, x_vec, s, xq)
                @inferred constant_interp!(out, x_vec, s, xq)
                @inferred quadratic_interp!(out, x_vec, s, xq)
                @inferred cubic_interp!(out, x_vec, s, xq)
            end
        end
    end

end

# Single-arg full-domain integrate duck contract: values see only `*`/`+` (see
# `_integrate_op`) plus the accumulator seed `zero(Tout)`. Deliberately NO `/` —
# only the Real grid step h may be divided. Local duck (not the snippet's MyDuck):
# MyDuck omits `zero` to keep the eval-path pins minimal.
@testitem "Duck Typing: 1D full-domain integrate (Vector + Range grids)" begin
    struct IntDuck
        v::Float64
    end
    Base.:+(a::IntDuck, b::IntDuck) = IntDuck(a.v + b.v)
    Base.:*(a::Real, b::IntDuck) = IntDuck(a * b.v)
    Base.:*(a::IntDuck, b::Real) = IntDuck(a.v * b)
    Base.zero(::Type{IntDuck}) = IntDuck(0.0)

    x_vec = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_rng = range(0.0, 6.0, 7)
    yf = 2 .* x_vec .+ 1
    yd = IntDuck.(yf)
    for (name, x) in [("Vector", x_vec), ("Range", x_rng)]
        @testset "linear ($name grid)" begin
            itp = linear_interp(x, yd)
            itp_ref = linear_interp(x, yf)
            r = integrate(itp)
            @test r isa IntDuck
            @test r.v ≈ integrate(itp_ref)
        end
    end
end

# ND duck integrate exercises the separable engine's non-Number zero init
# (`_nd_int_zero` → `0 * sample`) — the path the retired generic per-cell engine
# used to serve. Rank-1 families (Linear/Constant) build without differentiating
# the duck payload, so they're the clean pin for the ND value contract.
@testitem "duck ND integrate — separable engine (full + bounded)" begin
    struct NdDuck
        v::Float64
    end
    Base.:+(a::NdDuck, b::NdDuck) = NdDuck(a.v + b.v)
    Base.:-(a::NdDuck, b::NdDuck) = NdDuck(a.v - b.v)
    Base.:*(a::Real, b::NdDuck) = NdDuck(a * b.v)
    Base.:*(a::NdDuck, b::Real) = NdDuck(a.v * b)
    Base.zero(::Type{NdDuck}) = NdDuck(0.0)
    _v(d::NdDuck) = d.v

    x = [0.0, 0.5, 1.3, 2.0, 3.0]           # Vector × Range → also mixed-grid path
    y = range(0.0, 2.0, length = 6)
    dat = [NdDuck(sin(xi) + 2yj) for xi in x, yj in y]
    ref = [d.v for d in dat]
    lo = (0.3, 0.4);  hi = (2.6, 1.7)

    @testset "linear ND (rank-1 payload)" begin
        itp = linear_interp((x, y), dat);  rf = linear_interp((x, y), ref)
        @test integrate(itp) isa NdDuck
        @test _v(integrate(itp)) ≈ integrate(rf) rtol = 1.0e-12
        @test _v(integrate(itp, lo, hi)) ≈ integrate(rf, lo, hi) rtol = 1.0e-12
    end

    @testset "constant ND (rank-1, side weights)" begin
        itp = constant_interp((x, y), dat);  rf = constant_interp((x, y), ref)
        @test _v(integrate(itp)) ≈ integrate(rf) rtol = 1.0e-12
        @test _v(integrate(itp, lo, hi)) ≈ integrate(rf, lo, hi) rtol = 1.0e-12
    end
end

# Multi-channel vector values (SVector): unlike the scalar-wrapper ducks above,
# SVector{N} defines `zero`/`+`/`*` AND survives the rank-2 (cubic) differentiated
# payload, so it pins channel-wise integration end to end — the result is a fresh
# SVector with each channel integrated independently. Reference per channel: the
# integral of that channel's own scalar interpolant.
@testitem "SVector-valued integrate — channel-wise (1D + ND, all ranks)" begin
    using StaticArrays

    # integrate channel `c` via its own scalar interpolant (`map` keeps 1D/ND shape)
    chan(mk, grids, data, c, args...) = integrate(mk(grids, map(d -> d[c], data)), args...)

    @testset "1D — full + bounded + cumulative" begin
        x = range(0.0, 2.0, length = 7)
        data = [SVector(sin(xi), xi^2, 1.0) for xi in x]
        for mk in (linear_interp, cubic_interp, constant_interp)
            itp = mk(x, data)
            I = integrate(itp)
            @test I isa SVector{3, Float64}
            @test I ≈ SVector(ntuple(c -> chan(mk, x, data, c), 3)) rtol = 1.0e-12
            @test integrate(itp, 0.3, 1.6) ≈
                SVector(ntuple(c -> chan(mk, x, data, c, 0.3, 1.6), 3)) rtol = 1.0e-12
            C = cumulative_integrate(itp)
            @test C[1] == zero(SVector{3, Float64})
            @test C[end] ≈ I rtol = 1.0e-12
        end
    end

    @testset "ND — full + bounded (rank-1 linear/constant + rank-2 cubic)" begin
        x = range(0.0, 2.0, length = 6);  y = range(0.0, 3.0, length = 5)
        data = [SVector(sin(xi) + yj, xi * yj, 2.0) for xi in x, yj in y]
        lo = (0.3, 0.4);  hi = (1.7, 2.5)
        for mk in (linear_interp, cubic_interp, constant_interp)
            itp = mk((x, y), data)
            I = integrate(itp)
            @test I isa SVector{3, Float64}
            @test I ≈ SVector(ntuple(c -> chan(mk, (x, y), data, c), 3)) rtol = 1.0e-12
            @test integrate(itp, lo, hi) ≈
                SVector(ntuple(c -> chan(mk, (x, y), data, c, lo, hi), 3)) rtol = 1.0e-12
        end
    end
end
