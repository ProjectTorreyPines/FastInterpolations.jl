using Test
using FastInterpolations

@testset "TensorProductInterpolantND — PreCompute" begin
    # ========================================
    # Test Setup
    # ========================================
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    z = range(0.0, 1.0, 20)
    f(xi, yj) = sin(xi) * cos(yj)
    data_2d = [f(xi, yj) for xi in x, yj in y]

    qx, qy, qz = 1.7, 0.8, 0.45

    # ========================================
    # 1. All-cubic PreCompute == CubicInterpolantND
    # ========================================
    @testset "All-cubic PreCompute matches CubicInterpolantND" begin
        itp_pre = interp_nd(
            (x, y), data_2d;
            methods = (CubicInterp(), CubicInterp()), coeffs = PreCompute()
        )
        itp_ref = cubic_interp((x, y), data_2d)

        @test itp_pre((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-14
    end

    # ========================================
    # 2. All-quadratic PreCompute == QuadraticInterpolantND
    # ========================================
    @testset "All-quadratic PreCompute matches QuadraticInterpolantND" begin
        itp_pre = interp_nd(
            (x, y), data_2d;
            methods = (QuadraticInterp(), QuadraticInterp()), coeffs = PreCompute()
        )
        itp_ref = quadratic_interp((x, y), data_2d)

        @test itp_pre((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-14
    end

    # ========================================
    # 3. Cubic×Linear PreCompute == OnTheFly
    # ========================================
    @testset "Cubic×Linear: PreCompute matches OnTheFly" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp_pre = interp_nd((x, y), data_2d; methods = methods_cl, coeffs = PreCompute())
        itp_otf = interp_nd((x, y), data_2d; methods = methods_cl)

        for qxi in range(0.2, 6.0, 20), qyj in range(0.1, 3.0, 15)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end
    end

    # ========================================
    # 4. Cubic×Quadratic PreCompute == OnTheFly
    # ========================================
    @testset "Cubic×Quadratic: PreCompute matches OnTheFly" begin
        methods_cq = (CubicInterp(), QuadraticInterp())
        itp_pre = interp_nd((x, y), data_2d; methods = methods_cq, coeffs = PreCompute())
        itp_otf = interp_nd((x, y), data_2d; methods = methods_cq)

        for qxi in range(0.2, 6.0, 10), qyj in range(0.1, 3.0, 10)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end
    end

    # ========================================
    # 5. Linear×Cubic PreCompute == OnTheFly
    # ========================================
    @testset "Linear×Cubic: PreCompute matches OnTheFly" begin
        methods_lc = (LinearInterp(), CubicInterp())
        itp_pre = interp_nd((x, y), data_2d; methods = methods_lc, coeffs = PreCompute())
        itp_otf = interp_nd((x, y), data_2d; methods = methods_lc)

        for qxi in range(0.2, 6.0, 10), qyj in range(0.1, 3.0, 10)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end
    end

    # ========================================
    # 6. 3D Cubic×Linear×Quadratic PreCompute == OnTheFly
    # ========================================
    @testset "3D Cubic×Linear×Quadratic: PreCompute matches OnTheFly" begin
        data_3d = [sin(xi) * cos(yj) * (zk^2 + 1) for xi in x, yj in y, zk in z]
        methods_clq = (CubicInterp(), LinearInterp(), QuadraticInterp())
        itp_pre = interp_nd((x, y, z), data_3d; methods = methods_clq, coeffs = PreCompute())
        itp_otf = interp_nd((x, y, z), data_3d; methods = methods_clq)

        for qxi in range(0.5, 5.5, 5), qyj in range(0.2, 2.8, 5), qzk in range(0.1, 0.9, 5)
            @test itp_pre((qxi, qyj, qzk)) ≈ itp_otf((qxi, qyj, qzk)) rtol = 1.0e-10
        end
    end

    # ========================================
    # 7. Analytic exactness: Cubic×Linear on p3(x)×p1(y)
    # ========================================
    @testset "Exactness: Cubic×Linear on polynomial data" begin
        p3(xi) = 2xi^3 - 3xi^2 + xi - 1
        p1(yj) = 4yj + 7
        xg = range(-1.0, 3.0, 20)
        yg = range(0.0, 5.0, 15)
        data_poly = [p3(xi) * p1(yj) for xi in xg, yj in yg]
        itp_pre = interp_nd(
            (xg, yg), data_poly;
            methods = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )

        for qxi in range(-0.9, 2.9, 15), qyj in range(0.1, 4.9, 15)
            @test itp_pre((qxi, qyj)) ≈ p3(qxi) * p1(qyj) atol = 1.0e-10
        end
    end

    # ========================================
    # 8. Analytic exactness: Cubic×Quadratic on p3(x)×p2(y)
    # ========================================
    @testset "Exactness: Cubic×Quadratic on polynomial data" begin
        p3(xi) = xi^3 - xi
        p2(yj) = yj^2 - yj
        xg = range(-1.0, 3.0, 20)
        yg = range(0.0, 5.0, 15)
        data_poly = [p3(xi) * p2(yj) for xi in xg, yj in yg]
        itp_pre = interp_nd(
            (xg, yg), data_poly;
            methods = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute()
        )

        for qxi in range(-0.9, 2.9, 10), qyj in range(0.1, 4.9, 10)
            @test itp_pre((qxi, qyj)) ≈ p3(qxi) * p2(qyj) atol = 1.0e-8
        end
    end

    # ========================================
    # 9. Derivatives: PreCompute matches OnTheFly
    # ========================================
    @testset "Derivatives: PreCompute df/dx matches OnTheFly" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp_pre = interp_nd((x, y), data_2d; methods = methods_cl, coeffs = PreCompute())
        itp_otf = interp_nd((x, y), data_2d; methods = methods_cl)

        for qxi in range(0.5, 5.5, 10), qyj in range(0.2, 2.8, 10)
            @test itp_pre((qxi, qyj); deriv = (DerivOp(1), DerivOp(0))) ≈
                itp_otf((qxi, qyj); deriv = (DerivOp(1), DerivOp(0))) rtol = 1.0e-10
            @test itp_pre((qxi, qyj); deriv = (DerivOp(0), DerivOp(1))) ≈
                itp_otf((qxi, qyj); deriv = (DerivOp(0), DerivOp(1))) rtol = 1.0e-10
        end
    end

    # ========================================
    # 10. gradient() works with PreCompute
    # ========================================
    @testset "gradient() with PreCompute" begin
        itp_pre = interp_nd(
            (x, y), data_2d;
            methods = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )
        grad = gradient(itp_pre, (qx, qy))

        dfdx = itp_pre((qx, qy); deriv = (DerivOp(1), DerivOp(0)))
        dfdy = itp_pre((qx, qy); deriv = (DerivOp(0), DerivOp(1)))

        @test grad[1] ≈ dfdx rtol = 1.0e-12
        @test grad[2] ≈ dfdy rtol = 1.0e-12
    end

    # ========================================
    # 11. Zero-allocation eval (PreCompute)
    # ========================================
    # Function barrier pattern: setup + warmup + @allocated inside ONE function
    # to avoid @testset try/catch type instability artifacts.

    @testset "Zero-allocation: PreCompute Cubic×Cubic" begin
        function _test_alloc_cubic_cubic()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp_nd(
                (xg, yg), d;
                methods = (CubicInterp(), CubicInterp()), coeffs = PreCompute()
            )
            itp((1.0, 0.5))
            itp((1.0, 0.5))
            return @allocated itp((1.0, 0.5))
        end
        @test _test_alloc_cubic_cubic() == 0
    end

    @testset "Zero-allocation: PreCompute Cubic×Linear" begin
        function _test_alloc_cubic_linear()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp_nd(
                (xg, yg), d;
                methods = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
            )
            itp((1.0, 0.5))
            itp((1.0, 0.5))
            return @allocated itp((1.0, 0.5))
        end
        @test _test_alloc_cubic_linear() == 0
    end

    @testset "Zero-allocation: PreCompute Linear×Linear" begin
        function _test_alloc_linear_linear()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp_nd(
                (xg, yg), d;
                methods = (LinearInterp(), LinearInterp()), coeffs = PreCompute()
            )
            itp((1.0, 0.5))
            itp((1.0, 0.5))
            return @allocated itp((1.0, 0.5))
        end
        @test _test_alloc_linear_linear() == 0
    end

    @testset "Zero-allocation: PreCompute Cubic×Quadratic" begin
        function _test_alloc_cubic_quadratic()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp_nd(
                (xg, yg), d;
                methods = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute()
            )
            itp((1.0, 0.5))
            itp((1.0, 0.5))
            return @allocated itp((1.0, 0.5))
        end
        @test _test_alloc_cubic_quadratic() == 0
    end

    @testset "Zero-allocation: PreCompute gradient" begin
        function _test_alloc_gradient()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp_nd(
                (xg, yg), d;
                methods = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
            )
            gradient(itp, (1.0, 0.5))
            gradient(itp, (1.0, 0.5))
            return @allocated gradient(itp, (1.0, 0.5))
        end
        @test _test_alloc_gradient() == 0
    end

    # ========================================
    # 12. Compact storage memory verification
    # ========================================
    @testset "Compact storage: memory savings" begin
        ng = 50
        xg = range(0.0, 2π, ng)
        yg = range(0.0, π, ng)
        zg = range(0.0, 1.0, ng)
        data2 = [sin(xi) * cos(yj) for xi in xg, yj in yg]
        data3 = [sin(xi) * cos(yj) * zk for xi in xg, yj in yg, zk in zg]

        # Only heterogeneous combos produce TensorProductInterpolantND with HeteroPartials.
        # Homogeneous (Cubic×Cubic, Linear×Linear) auto-dispatch to existing ND types.

        # 2D Cubic×Linear: prod(sizes) = 2×1 = 2 (vs 2^2 = 4, 2× savings)
        itp_cl = interp_nd(
            (xg, yg), data2;
            methods = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )
        @test size(itp_cl.data.partials, 1) == 2   # compact!

        # 3D Cubic×Linear×Linear: prod(sizes) = 2×1×1 = 2 (vs 2^3 = 8, 4× savings)
        itp_cll = interp_nd(
            (xg, yg, zg), data3;
            methods = (CubicInterp(), LinearInterp(), LinearInterp()), coeffs = PreCompute()
        )
        @test size(itp_cll.data.partials, 1) == 2   # 4× savings

        # 3D Cubic×Linear×Quadratic: prod(sizes) = 2×1×2 = 4 (vs 2^3 = 8, 2× savings)
        itp_clq = interp_nd(
            (xg, yg, zg), data3;
            methods = (CubicInterp(), LinearInterp(), QuadraticInterp()), coeffs = PreCompute()
        )
        @test size(itp_clq.data.partials, 1) == 4   # 2× savings
    end

    # ========================================
    # 13. BUG-1: ConstantInterp derivative must return zero (PreCompute)
    # ========================================
    # The @generated kernel for ConstantInterp must respect the `op` parameter.
    # Constant interpolation has zero derivative at all orders.
    # OnTheFly handles this correctly; PreCompute must match.

    @testset "BUG-1: ConstantInterp derivative returns zero (PreCompute)" begin
        xg = range(0.0, 2π, 30)
        yg = range(0.0, π, 25)
        data_cc = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        itp_pre = interp_nd(
            (xg, yg), data_cc;
            methods = (CubicInterp(), ConstantInterp()), coeffs = PreCompute()
        )
        itp_otf = interp_nd(
            (xg, yg), data_cc;
            methods = (CubicInterp(), ConstantInterp()), coeffs = OnTheFly()
        )

        qxi, qyj = 1.7, 0.8

        # ∂f/∂y on Constant axis must be zero
        deriv_y_pre = itp_pre((qxi, qyj); deriv = (DerivOp(0), DerivOp(1)))
        deriv_y_otf = itp_otf((qxi, qyj); deriv = (DerivOp(0), DerivOp(1)))
        @test deriv_y_otf == 0.0                    # OnTheFly reference (known correct)
        @test deriv_y_pre == 0.0                    # PreCompute must match

        # ∂²f/∂y² on Constant axis must be zero
        deriv2_y_pre = itp_pre((qxi, qyj); deriv = (DerivOp(0), DerivOp(2)))
        @test deriv2_y_pre == 0.0

        # ∂f/∂x on Cubic axis must be non-zero (sanity check: not all zeros)
        deriv_x_pre = itp_pre((qxi, qyj); deriv = (DerivOp(1), DerivOp(0)))
        @test deriv_x_pre != 0.0

        # gradient: 2nd component (Constant axis) must be zero
        grad = gradient(itp_pre, (qxi, qyj))
        @test grad[2] == 0.0
        @test grad[1] != 0.0
    end

    # ========================================
    # 14. BUG-2: ConstantInterp `side` field must be respected (PreCompute)
    # ========================================
    # ConstantInterp(side=LeftSide()) must always return the left neighbor,
    # ConstantInterp(side=RightSide()) always the right, and NearestSide the nearest.
    # The @generated kernel must dispatch on the side type, not hard-code NearestSide.

    @testset "BUG-2: ConstantInterp side parameter respected (PreCompute)" begin
        # Use a simple step function where left ≠ right in each cell
        xg = range(0.0, 4.0, 5)   # [0, 1, 2, 3, 4]
        yg = range(0.0, 3.0, 4)   # [0, 1, 2, 3]
        # Monotonically increasing in y so left ≠ right in every cell
        data_step = [Float64(xi + 10yj) for xi in xg, yj in yg]

        itp_left = interp_nd(
            (xg, yg), data_step;
            methods = (LinearInterp(), ConstantInterp(side = LeftSide())),
            coeffs = PreCompute()
        )
        itp_right = interp_nd(
            (xg, yg), data_step;
            methods = (LinearInterp(), ConstantInterp(side = RightSide())),
            coeffs = PreCompute()
        )

        # Query at y = 0.7 — between yg[1]=0 and yg[2]=1
        # LeftSide → use y=0 data, RightSide → use y=1 data
        qxi = 2.0
        qyj = 0.7  # interior of first y-cell

        val_left = itp_left((qxi, qyj))
        val_right = itp_right((qxi, qyj))

        # Left: interp x at data[:, 1] (y=0 row), Right: interp x at data[:, 2] (y=1 row)
        # data[:, 1] = [0, 1, 2, 3, 4], data[:, 2] = [10, 11, 12, 13, 14]
        # linear_interp at x=2.0 → 2.0 (left) vs 12.0 (right)
        @test val_left ≈ 2.0 atol = 1.0e-12
        @test val_right ≈ 12.0 atol = 1.0e-12
        @test val_left != val_right   # Side must make a difference
    end
end
