using Test
using FastInterpolations

@testset "interp_nd One-Shot API" begin
    # ========================================
    # Test Setup
    # ========================================
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    data_2d = [sin(xi) * cos(yj) for xi in x, yj in y]
    qx, qy = 1.7, 0.8

    # ========================================
    # 1. Homogeneous scalar: all-cubic
    # ========================================
    @testset "Homo scalar: all-cubic" begin
        ref = cubic_interp((x, y), data_2d, (qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = CubicInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 2. Homogeneous scalar: all-linear
    # ========================================
    @testset "Homo scalar: all-linear" begin
        ref = linear_interp((x, y), data_2d, (qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = LinearInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 3. Homogeneous scalar: all-quadratic
    # ========================================
    @testset "Homo scalar: all-quadratic" begin
        ref = quadratic_interp((x, y), data_2d, (qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = QuadraticInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 4. Homogeneous scalar: all-constant
    # ========================================
    @testset "Homo scalar: all-constant" begin
        ref = constant_interp((x, y), data_2d, (qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = ConstantInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 5. Homogeneous batch: all-cubic
    # ========================================
    @testset "Homo batch: all-cubic" begin
        queries = ([1.0, 1.5, 2.0], [0.5, 0.8, 1.0])
        ref = cubic_interp((x, y), data_2d, queries)
        vals = interp_nd((x, y), data_2d, queries; methods = CubicInterp())
        @test vals ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 6. Heterogeneous scalar: Cubic × Linear
    # ========================================
    @testset "Hetero scalar: Cubic × Linear" begin
        itp = interp_nd((x, y), data_2d; methods = (CubicInterp(), LinearInterp()), coeffs = PreCompute())
        ref = itp((qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = (CubicInterp(), LinearInterp()))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 7. Heterogeneous scalar: Linear × Cubic
    # ========================================
    @testset "Hetero scalar: Linear × Cubic" begin
        itp = interp_nd((x, y), data_2d; methods = (LinearInterp(), CubicInterp()), coeffs = PreCompute())
        ref = itp((qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = (LinearInterp(), CubicInterp()))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 8. Heterogeneous scalar: Cubic × Quadratic
    # ========================================
    @testset "Hetero scalar: Cubic × Quadratic" begin
        itp = interp_nd(
            (x, y), data_2d;
            methods = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute()
        )
        ref = itp((qx, qy))
        val = interp_nd((x, y), data_2d, (qx, qy); methods = (CubicInterp(), QuadraticInterp()))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 9. Heterogeneous batch + in-place
    # ========================================
    @testset "Hetero batch + in-place" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp = interp_nd((x, y), data_2d; methods = methods_cl, coeffs = PreCompute())

        queries = ([1.0, 1.5, 2.0, 2.5, 3.0], [0.5, 0.8, 1.0, 1.2, 1.5])

        # Allocating batch
        vals = interp_nd((x, y), data_2d, queries; methods = methods_cl)
        for k in 1:5
            @test vals[k] ≈ itp((queries[1][k], queries[2][k])) rtol = 1.0e-14
        end

        # In-place batch
        output = zeros(5)
        interp_nd!(output, (x, y), data_2d, queries; methods = methods_cl)
        @test output ≈ vals rtol = 1.0e-14
    end

    # ========================================
    # 10. Derivatives
    # ========================================
    @testset "Derivatives: ∂f/∂x on Cubic × Linear" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp = interp_nd((x, y), data_2d; methods = methods_cl, coeffs = PreCompute())

        d10 = (DerivOp(1), DerivOp(0))
        d01 = (DerivOp(0), DerivOp(1))

        ref_dx = itp((qx, qy); deriv = d10)
        ref_dy = itp((qx, qy); deriv = d01)

        val_dx = interp_nd((x, y), data_2d, (qx, qy); methods = methods_cl, deriv = d10)
        val_dy = interp_nd((x, y), data_2d, (qx, qy); methods = methods_cl, deriv = d01)

        @test val_dx ≈ ref_dx rtol = 1.0e-12
        @test val_dy ≈ ref_dy rtol = 1.0e-12
    end

    # ========================================
    # 11. Zero-allocation (hetero scalar)
    # ========================================
    @testset "Zero-allocation: hetero scalar" begin
        function _test_alloc_oneshot()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            m = (CubicInterp(), LinearInterp())
            q = (1.0, 0.5)
            interp_nd((xg, yg), d, q; methods = m)
            interp_nd((xg, yg), d, q; methods = m)
            return @allocated interp_nd((xg, yg), d, q; methods = m)
        end
        @test _test_alloc_oneshot() == 0
    end

    # ========================================
    # 12. PeriodicBC (exclusive) in hetero one-shot
    # ========================================
    @testset "PeriodicBC exclusive one-shot" begin
        xp = range(0.0, step = 2π / 30, length = 30)
        yp = range(0.0, 5.0, 20)
        data_per = [sin(xi) * (2yj + 1) for xi in xp, yj in yp]

        methods_pl = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), LinearInterp())
        ext = (WrapExtrap(), NoExtrap())

        itp = interp_nd(
            (xp, yp), data_per;
            methods = methods_pl, extrap = ext, coeffs = PreCompute()
        )
        ref = itp((1.5, 2.3))
        val = interp_nd((xp, yp), data_per, (1.5, 2.3); methods = methods_pl, extrap = ext)
        @test val ≈ ref rtol = 1.0e-12
    end

    # ========================================
    # 13. 3D: Cubic × Linear × Quadratic
    # ========================================
    @testset "3D: Cubic × Linear × Quadratic" begin
        z = range(0.0, 1.0, 20)
        data_3d = [sin(xi) * cos(yj) * (zk^2 + 1) for xi in x, yj in y, zk in z]

        methods_clq = (CubicInterp(), LinearInterp(), QuadraticInterp())
        itp = interp_nd(
            (x, y, z), data_3d;
            methods = methods_clq, coeffs = PreCompute()
        )

        q3 = (1.7, 0.8, 0.45)
        ref = itp(q3)
        val = interp_nd((x, y, z), data_3d, q3; methods = methods_clq)
        @test val ≈ ref rtol = 1.0e-12
    end
end
