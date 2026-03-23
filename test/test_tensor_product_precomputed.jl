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
        itp_pre = interp(
            (x, y), data_2d;
            method = (CubicInterp(), CubicInterp()), coeffs = PreCompute()
        )
        itp_ref = cubic_interp((x, y), data_2d)

        @test itp_pre((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-14
    end

    # ========================================
    # 2. All-quadratic PreCompute == QuadraticInterpolantND
    # ========================================
    @testset "All-quadratic PreCompute matches QuadraticInterpolantND" begin
        itp_pre = interp(
            (x, y), data_2d;
            method = (QuadraticInterp(), QuadraticInterp()), coeffs = PreCompute()
        )
        itp_ref = quadratic_interp((x, y), data_2d)

        @test itp_pre((qx, qy)) ≈ itp_ref((qx, qy)) rtol = 1.0e-14
    end

    # ========================================
    # 3. Cubic×Linear PreCompute == OnTheFly
    # ========================================
    @testset "Cubic×Linear: PreCompute matches OnTheFly" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp_pre = interp((x, y), data_2d; method = methods_cl, coeffs = PreCompute())
        itp_otf = interp((x, y), data_2d; method = methods_cl)

        for qxi in range(0.2, 6.0, 20), qyj in range(0.1, 3.0, 15)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end
    end

    # ========================================
    # 4. Cubic×Quadratic PreCompute == OnTheFly
    # ========================================
    @testset "Cubic×Quadratic: PreCompute matches OnTheFly" begin
        methods_cq = (CubicInterp(), QuadraticInterp())
        itp_pre = interp((x, y), data_2d; method = methods_cq, coeffs = PreCompute())
        itp_otf = interp((x, y), data_2d; method = methods_cq)

        for qxi in range(0.2, 6.0, 10), qyj in range(0.1, 3.0, 10)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end
    end

    # ========================================
    # 5. Linear×Cubic PreCompute == OnTheFly
    # ========================================
    @testset "Linear×Cubic: PreCompute matches OnTheFly" begin
        methods_lc = (LinearInterp(), CubicInterp())
        itp_pre = interp((x, y), data_2d; method = methods_lc, coeffs = PreCompute())
        itp_otf = interp((x, y), data_2d; method = methods_lc)

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
        itp_pre = interp((x, y, z), data_3d; method = methods_clq, coeffs = PreCompute())
        itp_otf = interp((x, y, z), data_3d; method = methods_clq)

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
        itp_pre = interp(
            (xg, yg), data_poly;
            method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
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
        itp_pre = interp(
            (xg, yg), data_poly;
            method = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute()
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
        itp_pre = interp((x, y), data_2d; method = methods_cl, coeffs = PreCompute())
        itp_otf = interp((x, y), data_2d; method = methods_cl)

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
        itp_pre = interp(
            (x, y), data_2d;
            method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
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
            itp = interp(
                (xg, yg), d;
                method = (CubicInterp(), CubicInterp()), coeffs = PreCompute()
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
            itp = interp(
                (xg, yg), d;
                method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
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
            itp = interp(
                (xg, yg), d;
                method = (LinearInterp(), LinearInterp()), coeffs = PreCompute()
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
            itp = interp(
                (xg, yg), d;
                method = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute()
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
            itp = interp(
                (xg, yg), d;
                method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
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
        itp_cl = interp(
            (xg, yg), data2;
            method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )
        @test size(itp_cl.data.partials, 1) == 2   # compact!

        # 3D Cubic×Linear×Linear: prod(sizes) = 2×1×1 = 2 (vs 2^3 = 8, 4× savings)
        itp_cll = interp(
            (xg, yg, zg), data3;
            method = (CubicInterp(), LinearInterp(), LinearInterp()), coeffs = PreCompute()
        )
        @test size(itp_cll.data.partials, 1) == 2   # 4× savings

        # 3D Cubic×Linear×Quadratic: prod(sizes) = 2×1×2 = 4 (vs 2^3 = 8, 2× savings)
        itp_clq = interp(
            (xg, yg, zg), data3;
            method = (CubicInterp(), LinearInterp(), QuadraticInterp()), coeffs = PreCompute()
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

        itp_pre = interp(
            (xg, yg), data_cc;
            method = (CubicInterp(), ConstantInterp()), coeffs = PreCompute()
        )
        itp_otf = interp(
            (xg, yg), data_cc;
            method = (CubicInterp(), ConstantInterp()), coeffs = OnTheFly()
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

        itp_left = interp(
            (xg, yg), data_step;
            method = (LinearInterp(), ConstantInterp(side = LeftSide())),
            coeffs = PreCompute()
        )
        itp_right = interp(
            (xg, yg), data_step;
            method = (LinearInterp(), ConstantInterp(side = RightSide())),
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

    # ========================================
    # 15. Custom BC: ZeroCurvBC on cubic axis in hetero combo
    # ========================================
    @testset "Custom BC: ZeroCurvBC Cubic × Linear" begin
        xg = range(0.0, 2π, 30)
        yg = range(0.0, π, 25)
        data_bc = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        itp_pre = interp(
            (xg, yg), data_bc;
            method = (CubicInterp(bc = ZeroCurvBC()), LinearInterp()), coeffs = PreCompute()
        )
        itp_otf = interp(
            (xg, yg), data_bc;
            method = (CubicInterp(bc = ZeroCurvBC()), LinearInterp()), coeffs = OnTheFly()
        )

        # PreCompute must match OnTheFly for all interior points
        for qxi in range(0.5, 5.5, 8), qyj in range(0.2, 2.8, 8)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end

        # ZeroCurvBC must produce different results from CubicFit near boundaries
        itp_fit = interp(
            (xg, yg), data_bc;
            method = (CubicInterp(bc = CubicFit()), LinearInterp()), coeffs = PreCompute()
        )
        @test itp_pre((0.05, 0.8)) != itp_fit((0.05, 0.8))
    end

    # ========================================
    # 16. Custom BC: PeriodicBC (exclusive) on cubic axis in hetero combo
    # ========================================
    @testset "Custom BC: PeriodicBC(exclusive) Cubic × Linear" begin
        xp = range(0.0, step = 2π / 30, length = 30)  # [0, 2π) exclusive endpoint
        yp = range(0.0, 5.0, 20)
        data_per = [sin(xi) * (2yj + 1) for xi in xp, yj in yp]

        itp_pre = interp(
            (xp, yp), data_per;
            method = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), LinearInterp()),
            extrap = (WrapExtrap(), NoExtrap()), coeffs = PreCompute()
        )
        itp_otf = interp(
            (xp, yp), data_per;
            method = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), LinearInterp()),
            extrap = (WrapExtrap(), NoExtrap()), coeffs = OnTheFly()
        )

        # Separable reference: cubic_periodic(x, sin) * linear(y, 2y+1)
        g_vals = [sin(xi) for xi in xp]
        h_vals = [(2yj + 1) for yj in yp]
        itp_g = cubic_interp(xp, g_vals; bc = PeriodicBC(endpoint = :exclusive), extrap = WrapExtrap())
        itp_h = linear_interp(yp, h_vals)

        for qxi in range(0.3, 5.8, 10), qyj in range(0.3, 4.5, 8)
            ref = itp_g(qxi) * itp_h(qyj)
            @test itp_pre((qxi, qyj)) ≈ ref rtol = 1.0e-12
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end

        # WrapExtrap: f(x + 2π) == f(x)
        @test itp_pre((1.5 + 2π, 2.3)) ≈ itp_pre((1.5, 2.3)) rtol = 1.0e-12

        # Gradient on periodic axis
        grad = gradient(itp_pre, (1.5, 2.3))
        dfdx_ref = itp_g(1.5; deriv = DerivOp(1)) * itp_h(2.3)
        @test grad[1] ≈ dfdx_ref rtol = 1.0e-10
    end

    # ========================================
    # 17. Custom BC: PeriodicBC (inclusive) on cubic axis in hetero combo
    # ========================================
    @testset "Custom BC: PeriodicBC(inclusive) Cubic × Linear" begin
        xp = range(0.0, 2π, 31)   # 31 points, inclusive endpoint
        yp = range(0.0, 5.0, 20)
        data_per = [sin(xi) * (2yj + 1) for xi in xp, yj in yp]
        data_per[end, :] .= data_per[1, :]   # enforce exact periodicity

        itp_pre = interp(
            (xp, yp), data_per;
            method = (CubicInterp(bc = PeriodicBC()), LinearInterp()),
            extrap = (WrapExtrap(), NoExtrap()), coeffs = PreCompute()
        )
        itp_otf = interp(
            (xp, yp), data_per;
            method = (CubicInterp(bc = PeriodicBC()), LinearInterp()),
            extrap = (WrapExtrap(), NoExtrap()), coeffs = OnTheFly()
        )

        for qxi in range(0.3, 5.8, 8), qyj in range(0.3, 4.5, 6)
            @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
        end
    end

    # ========================================
    # 18. hessian() / laplacian() on PreCompute
    # ========================================
    @testset "hessian/laplacian on PreCompute Cubic × Linear" begin
        xg = range(0.0, 2π, 30)
        yg = range(0.0, π, 25)
        data_hl = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        itp_pre = interp(
            (xg, yg), data_hl;
            method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )

        qxi, qyj = 1.7, 0.8
        H = hessian(itp_pre, (qxi, qyj))
        @test size(H) == (2, 2)
        @test H[1, 2] ≈ H[2, 1] rtol = 1.0e-12    # symmetry

        # H[2,2] = ∂²f/∂y² on linear axis → must be 0
        @test H[2, 2] == 0.0

        # H[1,1] = ∂²f/∂x² on cubic axis → non-zero
        @test H[1, 1] != 0.0

        # laplacian = tr(H) = H[1,1] + H[2,2]
        lap = laplacian(itp_pre, (qxi, qyj))
        @test lap ≈ H[1, 1] + H[2, 2] rtol = 1.0e-12

        # PreCompute hessian matches OnTheFly
        itp_otf = interp(
            (xg, yg), data_hl;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        H_otf = hessian(itp_otf, (qxi, qyj))
        for i in 1:2, j in 1:2
            @test H[i, j] ≈ H_otf[i, j] rtol = 1.0e-10
        end
    end

    # ========================================
    # 19. Float32 data and grids (PreCompute)
    # ========================================
    @testset "Float32 PreCompute" begin
        x32 = range(0.0f0, 2.0f0 * Float32(π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = [sin(xi) * cos(yj) for xi in x32, yj in y32]

        itp32 = interp(
            (x32, y32), data32;
            method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )
        val = itp32((1.0f0, 0.5f0))
        @test val isa Float32
        @test val ≈ sin(1.0f0) * cos(0.5f0) atol = 0.01f0

        # Gradient should also be Float32
        grad = gradient(itp32, (1.0f0, 0.5f0))
        @test eltype(grad) == Float32
    end

    # ========================================
    # 20. Mixed grid types: Range × Vector (PreCompute)
    # ========================================
    @testset "Mixed grids: Range × Vector PreCompute" begin
        x_range = range(0.0, 2π, 30)
        y_vec = collect(range(0.0, π, 25))
        data_mixed = [sin(xi) * cos(yj) for xi in x_range, yj in y_vec]

        itp_pre = interp(
            (x_range, y_vec), data_mixed;
            method = (CubicInterp(), LinearInterp()), coeffs = PreCompute()
        )
        itp_otf = interp(
            (x_range, y_vec), data_mixed;
            method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
        )

        qxi, qyj = 1.7, 0.8
        @test itp_pre((qxi, qyj)) ≈ itp_otf((qxi, qyj)) rtol = 1.0e-12
    end
end
