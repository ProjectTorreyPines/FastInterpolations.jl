using Test
using FastInterpolations

@testset "NoInterp + GridIdx" begin
    # ========================================
    # Test Setup
    # ========================================
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    z = range(0.0, 1.0, 20)
    f2(xi, yj) = sin(xi) * cos(yj)
    f3(xi, yj, zk) = sin(xi) * cos(yj) * exp(-zk)
    data_2d = [f2(xi, yj) for xi in x, yj in y]
    data_3d = [f3(xi, yj, zk) for xi in x, yj in y, zk in z]

    qx, qy, qz = 1.7, 0.8, 0.45

    # ========================================
    # 1. GridIdx Type
    # ========================================
    @testset "GridIdx basics" begin
        g = GridIdx(5)
        @test g.idx == 5
        @test sprint(show, g) == "GridIdx(5)"
        @test_throws ArgumentError GridIdx(0)
        @test_throws ArgumentError GridIdx(-1)
        @test !(GridIdx <: Real)
    end

    # ========================================
    # 2. NoInterp Type
    # ========================================
    @testset "NoInterp basics" begin
        @test NoInterp() isa AbstractInterpMethod
        @test sprint(show, NoInterp()) == "NoInterp()"
    end

    # ========================================
    # 3. One-Shot with GridIdx
    # ========================================
    @testset "One-shot: Cubic×NoInterp 2D" begin
        for k in [1, 5, 10, 25]
            val = interp((x, y), data_2d, (qx, GridIdx(k)); method = (CubicInterp(), NoInterp()))
            ref = cubic_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "One-shot: NoInterp×Cubic 2D" begin
        for k in [1, 10, 30]
            val = interp((x, y), data_2d, (GridIdx(k), qy); method = (NoInterp(), CubicInterp()))
            ref = cubic_interp(y, data_2d[k, :], qy)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "One-shot: GridIdx on non-NoInterp method (pre-slice)" begin
        # GridIdx on LinearInterp axis → still works in one-shot (pre-slices)
        for k in [1, 5, 25]
            val = interp((x, y), data_2d, (qx, GridIdx(k)); method = (CubicInterp(), LinearInterp()))
            ref = cubic_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "One-shot: 3D with NoInterp in middle" begin
        for k in [1, 10, 25]
            val = interp(
                (x, y, z), data_3d, (qx, GridIdx(k), qz);
                method = (CubicInterp(), NoInterp(), LinearInterp())
            )
            ref = interp(
                (x, z), data_3d[:, k, :], (qx, qz);
                method = (CubicInterp(), LinearInterp())
            )
            @test val ≈ ref rtol = 1.0e-13
        end
    end

    @testset "One-shot: All-GridIdx (pure table lookup)" begin
        val = interp(
            (x, y), data_2d, (GridIdx(3), GridIdx(7));
            method = (NoInterp(), NoInterp())
        )
        @test val == data_2d[3, 7]
    end

    @testset "One-shot: 3D all-GridIdx" begin
        val = interp(
            (x, y, z), data_3d, (GridIdx(5), GridIdx(10), GridIdx(15));
            method = (NoInterp(), NoInterp(), NoInterp())
        )
        @test val == data_3d[5, 10, 15]
    end

    # ========================================
    # 4. Interpolant with NoInterp (PreCompute)
    # ========================================
    @testset "Interpolant PreCompute: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        for k in [1, 5, 10, 25]
            val = itp((qx, GridIdx(k)))
            ref = cubic_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Interpolant PreCompute: NoInterp×Linear 2D" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), LinearInterp()))
        for k in [1, 15, 30]
            val = itp((GridIdx(k), qy))
            ref = linear_interp(y, data_2d[k, :], qy)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Interpolant PreCompute: 3D Cubic×NoInterp×Linear" begin
        itp = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        for k in [1, 10, 25]
            val = itp((qx, GridIdx(k), qz))
            ref = interp(
                (x, z), data_3d[:, k, :], (qx, qz);
                method = (CubicInterp(), LinearInterp())
            )
            @test val ≈ ref rtol = 1.0e-13
        end
    end

    @testset "Interpolant PreCompute: Multiple NoInterp axes" begin
        # (NoInterp, Cubic, NoInterp) → reduces to 1D cubic
        itp = interp(
            (x, y, z), data_3d;
            method = (NoInterp(), CubicInterp(), NoInterp())
        )
        for ki in [1, 15], kj in [1, 10]
            val = itp((GridIdx(ki), qy, GridIdx(kj)))
            ref = cubic_interp(y, data_3d[ki, :, kj], qy)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Interpolant PreCompute: All-NoInterp (pure lookup)" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()))
        @test itp((GridIdx(3), GridIdx(7))) == data_2d[3, 7]
        @test itp((GridIdx(1), GridIdx(1))) == data_2d[1, 1]
    end

    # ========================================
    # 5. Interpolant with NoInterp (OnTheFly)
    # ========================================
    @testset "Interpolant OnTheFly: Cubic×NoInterp 2D" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly()
        )
        for k in [1, 5, 25]
            val = itp((qx, GridIdx(k)))
            ref = cubic_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Interpolant OnTheFly: 3D Cubic×NoInterp×Linear" begin
        itp = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp()), coeffs = OnTheFly()
        )
        for k in [1, 10]
            val = itp((qx, GridIdx(k), qz))
            ref = interp(
                (x, z), data_3d[:, k, :], (qx, qz);
                method = (CubicInterp(), LinearInterp())
            )
            @test val ≈ ref rtol = 1.0e-13
        end
    end

    # ========================================
    # 6. Error Tests
    # ========================================
    @testset "Error: GridIdx out of bounds" begin
        # GridIdx(0) → ArgumentError from GridIdx constructor (i >= 1 check)
        @test_throws ArgumentError GridIdx(0)
        # GridIdx(26) on a 25-point grid → ArgumentError from bounds check
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, GridIdx(26));
            method = (CubicInterp(), NoInterp())
        )
    end

    @testset "Error: GridIdx on non-NoInterp in interpolant" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        @test_throws ArgumentError itp((GridIdx(3), GridIdx(5)))
    end

    @testset "Error: NoInterp axis missing GridIdx in interpolant" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        # The all-Real callable doesn't hit our validation, but it will fail
        # at InBounds extrap handler — any error is acceptable
        @test_throws Exception itp((qx, qy))
    end

    # ========================================
    # 7. Derivative Tests
    # ========================================
    @testset "Derivative: NoInterp axis deriv=0 (value)" begin
        # d/dx on cubic axis, EvalValue on NoInterp axis
        val = interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), NoInterp()),
            deriv = (DerivOp(1), DerivOp(0))
        )
        ref = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test val ≈ ref rtol = 1.0e-12
    end

    @testset "Derivative: via interpolant" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        val = itp((qx, GridIdx(5)); deriv = (DerivOp(1), DerivOp(0)))
        ref = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test val ≈ ref rtol = 1.0e-12
    end

    # ========================================
    # 8. Type Inference Tests
    # ========================================
    @testset "Type inference: PreCompute interpolant" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        @test @inferred(itp((qx, GridIdx(5)))) isa Float64
    end

    @testset "Type inference: OnTheFly interpolant" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly()
        )
        @test @inferred(itp((qx, GridIdx(5)))) isa Float64
    end

    @testset "Type inference: one-shot" begin
        @test @inferred(
            interp(
                (x, y), data_2d, (qx, GridIdx(5));
                method = (CubicInterp(), NoInterp())
            )
        ) isa Float64
    end

    # ========================================
    # 9. OOB GridIdx on Interpolant Path
    # ========================================
    @testset "Error: OOB GridIdx on PreCompute interpolant" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        @test_throws ArgumentError itp((qx, GridIdx(26)))  # y has 25 points
        @test_throws ArgumentError itp((qx, GridIdx(100)))
    end

    @testset "Error: OOB GridIdx on OnTheFly interpolant" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly()
        )
        @test_throws ArgumentError itp((qx, GridIdx(26)))
    end

    @testset "Error: OOB GridIdx on All-NoInterp interpolant" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()))
        @test_throws ArgumentError itp((GridIdx(31), GridIdx(5)))  # x has 30 points
        @test_throws ArgumentError itp((GridIdx(5), GridIdx(26)))  # y has 25 points
    end

    # ========================================
    # 10. Float32 Support
    # ========================================
    @testset "Float32: PreCompute interpolant" begin
        x32 = range(0.0f0, Float32(2π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = [sin(xi) * cos(yj) for xi in x32, yj in y32]
        itp32 = interp((x32, y32), data32; method = (CubicInterp(), NoInterp()))
        val = itp32((1.7f0, GridIdx(5)))
        ref = cubic_interp(x32, data32[:, 5], 1.7f0)
        @test val isa Float32
        @test val ≈ ref rtol = 1.0f-6
    end

    @testset "Float32: one-shot" begin
        x32 = range(0.0f0, Float32(2π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = [sin(xi) * cos(yj) for xi in x32, yj in y32]
        val = interp((x32, y32), data32, (1.7f0, GridIdx(5)); method = (CubicInterp(), NoInterp()))
        ref = cubic_interp(x32, data32[:, 5], 1.7f0)
        @test val isa Float32
        @test val ≈ ref rtol = 1.0f-6
    end

    # ========================================
    # 11. Zero-Allocation Tests
    # ========================================
    @testset "Zero-allocation: PreCompute interpolant eval" begin
        function _test_alloc_precompute()
            itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
            itp((1.7, GridIdx(5)))  # warmup
            return @allocated itp((1.7, GridIdx(5)))
        end
        @test _test_alloc_precompute() == 0
    end

    @testset "Zero-allocation: PreCompute interpolant eval with deriv" begin
        function _test_alloc_deriv()
            itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
            itp((1.7, GridIdx(5)); deriv = (DerivOp(1), DerivOp(0)))  # warmup
            return @allocated itp((1.7, GridIdx(5)); deriv = (DerivOp(1), DerivOp(0)))
        end
        @test _test_alloc_deriv() == 0
    end

    @testset "Zero-allocation: OnTheFly interpolant eval" begin
        function _test_alloc_onthefly()
            itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()), coeffs = OnTheFly())
            itp((1.7, GridIdx(5)))  # warmup
            return @allocated itp((1.7, GridIdx(5)))
        end
        @test _test_alloc_onthefly() == 0
    end

    # ========================================
    # 12. Vararg Callable
    # ========================================
    @testset "Vararg callable: itp(0.5, GridIdx(k))" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        val_vararg = itp(qx, GridIdx(5))
        val_tuple = itp((qx, GridIdx(5)))
        @test val_vararg == val_tuple
    end

    @testset "Vararg callable: 3D" begin
        itp3 = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        val_vararg = itp3(qx, GridIdx(5), qz)
        val_tuple = itp3((qx, GridIdx(5), qz))
        @test val_vararg == val_tuple
    end

    # ========================================
    # 13. gradient / hessian / laplacian
    # ========================================
    @testset "gradient: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        g = gradient(itp, (qx, GridIdx(5)))
        ref = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test g[1] ≈ ref rtol = 1.0e-12
        @test g[2] == 0.0
        @test length(g) == 2
    end

    @testset "gradient: 3D Cubic×NoInterp×Linear" begin
        itp = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        g = gradient(itp, (qx, GridIdx(5), qz))
        @test g[2] == 0.0  # NoInterp axis
        @test length(g) == 3
        # g[1] and g[3] should be non-zero (interp axes)
        @test g[1] != 0.0
        @test g[3] != 0.0
    end

    @testset "hessian: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        H = hessian(itp, (qx, GridIdx(5)))
        ref_d2 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(2))
        @test size(H) == (2, 2)
        @test H[1, 1] ≈ ref_d2 rtol = 1.0e-10
        @test H[1, 2] == 0.0  # NoInterp column
        @test H[2, 1] == 0.0  # NoInterp row
        @test H[2, 2] == 0.0  # NoInterp diagonal
    end

    @testset "laplacian: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        L = laplacian(itp, (qx, GridIdx(5)))
        ref_d2 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(2))
        @test L ≈ ref_d2 rtol = 1.0e-10
    end

    # ========================================
    # 14. Batch interp_batch_grididx!
    # ========================================
    @testset "Batch: interp_batch_grididx! 2D" begin
        xq_batch = collect(range(0.5, 5.0, 50))
        output = zeros(50)
        interp_batch_grididx!(
            output, (x, y), data_2d, (xq_batch, GridIdx(5));
            method = (CubicInterp(), NoInterp())
        )
        ref = [cubic_interp(x, data_2d[:, 5], xqi) for xqi in xq_batch]
        @test output ≈ ref rtol = 1.0e-14
    end

    @testset "Batch: interp_batch_grididx! 3D" begin
        xq_batch = collect(range(0.5, 5.0, 30))
        zq_batch = collect(range(0.1, 0.9, 30))
        output = zeros(30)
        interp_batch_grididx!(
            output, (x, y, z), data_3d, (xq_batch, GridIdx(10), zq_batch);
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        ref = [
            interp(
                    (x, z), data_3d[:, 10, :], (xq_batch[i], zq_batch[i]);
                    method = (CubicInterp(), LinearInterp())
                )
                for i in 1:30
        ]
        @test output ≈ ref rtol = 1.0e-13
    end

    # ========================================
    # 15. Regression: DerivOp on NoInterp axis returns 0
    # ========================================
    @testset "Regression: deriv on NoInterp axis returns 0 (interpolant)" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        @test itp((qx, GridIdx(5)); deriv = (DerivOp(0), DerivOp(1))) == 0.0
        @test itp((qx, GridIdx(5)); deriv = (DerivOp(0), DerivOp(2))) == 0.0
    end

    @testset "Regression: deriv on NoInterp axis returns 0 (one-shot)" begin
        @test interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), NoInterp()), deriv = (DerivOp(0), DerivOp(1))
        ) == 0.0
    end

    @testset "Regression: all-NoInterp deriv returns 0" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()))
        @test itp((GridIdx(3), GridIdx(5)); deriv = (DerivOp(1), DerivOp(0))) == 0.0
    end

    # ========================================
    # 16. Regression: OnTheFly FillExtrap OOB
    # ========================================
    @testset "Regression: OnTheFly FillExtrap returns fill value on OOB" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly(),
            extrap = (FillExtrap(-99.0), NoExtrap())
        )
        @test itp((99.0, GridIdx(5))) == -99.0
        @test itp((-99.0, GridIdx(5))) == -99.0
    end

    # ========================================
    # 17. Regression: search=BinarySearch() scalar
    # ========================================
    @testset "Regression: scalar search kwarg works with GridIdx" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        ref = itp((qx, GridIdx(5)))
        @test itp((qx, GridIdx(5)); search = BinarySearch()) == ref
        @test itp((qx, GridIdx(5)); search = AutoSearch()) == ref
    end

    # ========================================
    # 18. Regression: Singleton NoInterp grid
    # ========================================
    @testset "Regression: singleton grid for NoInterp axis" begin
        data_single = rand(30, 1)
        itp = interp((x, [0.0]), data_single; method = (LinearInterp(), NoInterp()))
        val = itp((qx, GridIdx(1)))
        ref = linear_interp(x, data_single[:, 1], qx)
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 19. Edge cases: domain/OOB before deriv zero, batch deriv, etc.
    # ========================================
    @testset "Edge: OOB Real axis + NoInterp deriv → DomainError (not zero)" begin
        itp = interp((x, y), data_2d; method = (LinearInterp(), NoInterp()))
        @test_throws DomainError itp((99.0, GridIdx(2)); deriv = (DerivOp(0), DerivOp(1)))
    end

    @testset "Edge: batch deriv on GridIdx axis → zeros" begin
        xq_b = collect(range(0.5, 5.0, 10))
        out = zeros(10)
        interp_batch_grididx!(
            out, (x, y), data_2d, (xq_b, GridIdx(5));
            method = (CubicInterp(), NoInterp()), deriv = (DerivOp(0), DerivOp(1))
        )
        @test all(out .== 0.0)
    end

    @testset "Edge: all-NoInterp laplacian OOB → error" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()))
        @test_throws ArgumentError laplacian(itp, (GridIdx(99), GridIdx(1)))
    end

    @testset "Edge: empty batch" begin
        out = Float64[]
        interp_batch_grididx!(
            out, (x, y), data_2d, (Float64[], GridIdx(5));
            method = (CubicInterp(), NoInterp())
        )
        @test isempty(out)
    end

    @testset "Edge: Float32 type promotion on zero deriv" begin
        x32 = range(0.0f0, Float32(2π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = Float32[sin(xi) * cos(yj) for xi in x32, yj in y32]
        itp32 = interp((x32, y32), data32; method = (NoInterp(), CubicInterp()))
        val_norm = itp32((GridIdx(2), Float64(2.3)))
        val_deriv = itp32((GridIdx(2), Float64(2.3)); deriv = (DerivOp(1), DerivOp(0)))
        @test typeof(val_norm) == typeof(val_deriv)  # both Float64
        g = gradient(itp32, (GridIdx(2), Float64(2.3)))
        @test typeof(g[1]) == typeof(g[2])  # both Float64
    end
end
