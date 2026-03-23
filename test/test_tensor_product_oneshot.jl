using Test
using FastInterpolations

@testset "interp One-Shot API" begin
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
        val = interp((x, y), data_2d, (qx, qy); method = CubicInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 2. Homogeneous scalar: all-linear
    # ========================================
    @testset "Homo scalar: all-linear" begin
        ref = linear_interp((x, y), data_2d, (qx, qy))
        val = interp((x, y), data_2d, (qx, qy); method = LinearInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 3. Homogeneous scalar: all-quadratic
    # ========================================
    @testset "Homo scalar: all-quadratic" begin
        ref = quadratic_interp((x, y), data_2d, (qx, qy))
        val = interp((x, y), data_2d, (qx, qy); method = QuadraticInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 4. Homogeneous scalar: all-constant
    # ========================================
    @testset "Homo scalar: all-constant" begin
        ref = constant_interp((x, y), data_2d, (qx, qy))
        val = interp((x, y), data_2d, (qx, qy); method = ConstantInterp())
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 5. Homogeneous batch: all-cubic
    # ========================================
    @testset "Homo batch: all-cubic" begin
        queries = ([1.0, 1.5, 2.0], [0.5, 0.8, 1.0])
        ref = cubic_interp((x, y), data_2d, queries)
        vals = interp((x, y), data_2d, queries; method = CubicInterp())
        @test vals ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 6. Heterogeneous scalar: Cubic × Linear
    # ========================================
    @testset "Hetero scalar: Cubic × Linear" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()), coeffs = PreCompute())
        ref = itp((qx, qy))
        val = interp((x, y), data_2d, (qx, qy); method = (CubicInterp(), LinearInterp()))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 7. Heterogeneous scalar: Linear × Cubic
    # ========================================
    @testset "Hetero scalar: Linear × Cubic" begin
        itp = interp((x, y), data_2d; method = (LinearInterp(), CubicInterp()), coeffs = PreCompute())
        ref = itp((qx, qy))
        val = interp((x, y), data_2d, (qx, qy); method = (LinearInterp(), CubicInterp()))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 8. Heterogeneous scalar: Cubic × Quadratic
    # ========================================
    @testset "Hetero scalar: Cubic × Quadratic" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute()
        )
        ref = itp((qx, qy))
        val = interp((x, y), data_2d, (qx, qy); method = (CubicInterp(), QuadraticInterp()))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 9. Heterogeneous batch + in-place
    # ========================================
    @testset "Hetero batch + in-place" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp = interp((x, y), data_2d; method = methods_cl, coeffs = PreCompute())

        queries = ([1.0, 1.5, 2.0, 2.5, 3.0], [0.5, 0.8, 1.0, 1.2, 1.5])

        # Allocating batch
        vals = interp((x, y), data_2d, queries; method = methods_cl)
        for k in 1:5
            @test vals[k] ≈ itp((queries[1][k], queries[2][k])) rtol = 1.0e-14
        end

        # In-place batch
        output = zeros(5)
        interp!(output, (x, y), data_2d, queries; method = methods_cl)
        @test output ≈ vals rtol = 1.0e-14
    end

    # ========================================
    # 10. Derivatives
    # ========================================
    @testset "Derivatives: ∂f/∂x on Cubic × Linear" begin
        methods_cl = (CubicInterp(), LinearInterp())
        itp = interp((x, y), data_2d; method = methods_cl, coeffs = PreCompute())

        d10 = (DerivOp(1), DerivOp(0))
        d01 = (DerivOp(0), DerivOp(1))

        ref_dx = itp((qx, qy); deriv = d10)
        ref_dy = itp((qx, qy); deriv = d01)

        val_dx = interp((x, y), data_2d, (qx, qy); method = methods_cl, deriv = d10)
        val_dy = interp((x, y), data_2d, (qx, qy); method = methods_cl, deriv = d01)

        @test val_dx ≈ ref_dx rtol = 1.0e-12
        @test val_dy ≈ ref_dy rtol = 1.0e-12
    end

    # ========================================
    # 11. Zero-allocation: hetero scalar
    # ========================================
    @testset "Zero-allocation: hetero scalar" begin
        function _test_alloc_oneshot_hetero()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            m = (CubicInterp(), LinearInterp())
            q = (1.0, 0.5)
            interp((xg, yg), d, q; method = m)
            interp((xg, yg), d, q; method = m)
            return @allocated interp((xg, yg), d, q; method = m)
        end
        @test _test_alloc_oneshot_hetero() <= ND_ALLOC_THRESHOLD
    end

    # ========================================
    # 11b. Zero-allocation: homo scalar (auto-dispatch)
    # ========================================
    @testset "Zero-allocation: homo scalar (cubic)" begin
        function _test_alloc_oneshot_homo()
            xg = range(0.0, 2π, 30)
            yg = range(0.0, π, 25)
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            q = (1.0, 0.5)
            interp((xg, yg), d, q; method = CubicInterp())
            interp((xg, yg), d, q; method = CubicInterp())
            return @allocated interp((xg, yg), d, q; method = CubicInterp())
        end
        @test _test_alloc_oneshot_homo() <= ND_ALLOC_THRESHOLD
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

        itp = interp(
            (xp, yp), data_per;
            method = methods_pl, extrap = ext, coeffs = PreCompute()
        )
        ref = itp((1.5, 2.3))
        val = interp((xp, yp), data_per, (1.5, 2.3); method = methods_pl, extrap = ext)
        @test val ≈ ref rtol = 1.0e-12
    end

    # ========================================
    # 13. 3D: Cubic × Linear × Quadratic
    # ========================================
    @testset "3D: Cubic × Linear × Quadratic" begin
        z = range(0.0, 1.0, 20)
        data_3d = [sin(xi) * cos(yj) * (zk^2 + 1) for xi in x, yj in y, zk in z]

        methods_clq = (CubicInterp(), LinearInterp(), QuadraticInterp())
        itp = interp(
            (x, y, z), data_3d;
            method = methods_clq, coeffs = PreCompute()
        )

        q3 = (1.7, 0.8, 0.45)
        ref = itp(q3)
        val = interp((x, y, z), data_3d, q3; method = methods_clq)
        @test val ≈ ref rtol = 1.0e-12
    end

    # ========================================
    # 14. Float32 one-shot
    # ========================================
    @testset "Float32 one-shot" begin
        x32 = range(0.0f0, 2.0f0 * Float32(π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = [sin(xi) * cos(yj) for xi in x32, yj in y32]

        # Hetero scalar
        val = interp((x32, y32), data32, (1.0f0, 0.5f0); method = (CubicInterp(), LinearInterp()))
        @test val isa Float32
        @test val ≈ sin(1.0f0) * cos(0.5f0) atol = 0.01f0

        # Homo scalar
        val_c = interp((x32, y32), data32, (1.0f0, 0.5f0); method = CubicInterp())
        @test val_c isa Float32
    end

    # ========================================
    # 15. Duck-typing: custom value type
    # ========================================
    @testset "Duck-typing: MyDuck value type" begin
        # Minimal duck type: +, -, Real*Tv, Tv*Real, Int*Tv
        struct _OneshotDuck
            v::Float64
        end
        Base.:+(a::_OneshotDuck, b::_OneshotDuck) = _OneshotDuck(a.v + b.v)
        Base.:-(a::_OneshotDuck, b::_OneshotDuck) = _OneshotDuck(a.v - b.v)
        Base.:*(a::Real, b::_OneshotDuck) = _OneshotDuck(a * b.v)
        Base.:*(a::_OneshotDuck, b::Real) = _OneshotDuck(a.v * b)

        xg = range(0.0, 4.0, 20)
        yg = range(0.0, 3.0, 15)
        data_duck = [_OneshotDuck(xi + 2yj) for xi in xg, yj in yg]

        # Linear × Linear (duck types work with linear kernel)
        val = interp((xg, yg), data_duck, (2.0, 1.5); method = (LinearInterp(), LinearInterp()))
        @test val isa _OneshotDuck
        @test val.v ≈ 2.0 + 2 * 1.5 atol = 1.0e-12
    end

    # ========================================
    # 16. Homo batch in-place
    # ========================================
    @testset "Homo batch in-place" begin
        queries = ([1.0, 1.5, 2.0], [0.5, 0.8, 1.0])

        # cubic
        ref = cubic_interp((x, y), data_2d, queries)
        output = zeros(3)
        interp!(output, (x, y), data_2d, queries; method = CubicInterp())
        @test output ≈ ref rtol = 1.0e-14

        # linear
        ref_l = linear_interp((x, y), data_2d, queries)
        output_l = zeros(3)
        interp!(output_l, (x, y), data_2d, queries; method = LinearInterp())
        @test output_l ≈ ref_l rtol = 1.0e-14
    end

    # ========================================
    # 17. Extrapolation modes
    # ========================================
    @testset "Extrapolation: ClampExtrap one-shot" begin
        methods_cl = (CubicInterp(), LinearInterp())

        # OOB on axis 1 → clamped (no error)
        val_clamp = interp(
            (x, y), data_2d, (-1.0, qy);
            method = methods_cl, extrap = ClampExtrap()
        )
        @test val_clamp isa Float64

        # OOB on axis 2 → NoExtrap throws
        @test_throws Exception interp(
            (x, y), data_2d, (qx, -1.0);
            method = methods_cl, extrap = NoExtrap()
        )
    end
end
