@testitem "NoInterp + GridIdx: Basics & One-shot & Interpolant" setup = [AllocConstants] begin
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
        # `occursin` (not `==`): show() may prefix the module qualifier
        # depending on the active IOContext (VSCode test runner vs CLI vs
        # Pkg.test all differ). The unqualified name is always present.
        @test occursin("GridIdx(5)", sprint(show, g))
        @test_throws ArgumentError GridIdx(0)
        @test_throws ArgumentError GridIdx(-1)
        @test GridIdx <: Real
    end

    # ========================================
    # 2. NoInterp Type
    # ========================================
    @testset "NoInterp basics" begin
        @test NoInterp() isa AbstractInterpMethod
        @test occursin("NoInterp()", sprint(show, NoInterp()))
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

    @testset "GridIdx on non-NoInterp axis: converts to grids[d][k]" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        # GridIdx(3) on CubicInterp axis → converts to x[3], GridIdx(5) on NoInterp stays
        val = itp((GridIdx(3), GridIdx(5)))
        ref = itp((x[3], GridIdx(5)))
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "Error: NoInterp axis missing GridIdx in interpolant" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        # Clear error: tells user to use GridIdx(k) for NoInterp axes
        @test_throws ArgumentError itp((qx, qy))
        @test_throws ArgumentError itp(qx, qy)
        err = try
            itp((qx, qy))
        catch e
            e
        end
        @test occursin("NoInterp on axis 2", err.msg)
        @test occursin("x1::Real, GridIdx(k2::Int)", err.msg)
        # Regression: GridIdx on non-NoInterp + Real on NoInterp → ArgumentError (not MethodError)
        # GridIdx(2) converts to x[2], but axis 2 (NoInterp) still has Real → error
        @test_throws ArgumentError itp((GridIdx(2), qy))
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
        @test _test_alloc_precompute() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: PreCompute interpolant eval with deriv" begin
        function _test_alloc_deriv()
            itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
            itp((1.7, GridIdx(5)); deriv = (DerivOp(1), DerivOp(0)))  # warmup
            return @allocated itp((1.7, GridIdx(5)); deriv = (DerivOp(1), DerivOp(0)))
        end
        @test _test_alloc_deriv() <= ND_ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: OnTheFly interpolant eval" begin
        function _test_alloc_onthefly()
            itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()), coeffs = OnTheFly())
            itp((1.7, GridIdx(5)))  # warmup
            return @allocated itp((1.7, GridIdx(5)))
        end
        @test _test_alloc_onthefly() <= ND_ALLOC_THRESHOLD
    end

end

# ========================================
# 12. Vararg Callable
# ========================================
@testitem "NoInterp + GridIdx: Vararg & Calculus & Batch & Regression" setup = [AllocConstants] begin
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    z = range(0.0, 1.0, 20)
    f2(xi, yj) = sin(xi) * cos(yj)
    f3(xi, yj, zk) = sin(xi) * cos(yj) * exp(-zk)
    data_2d = [f2(xi, yj) for xi in x, yj in y]
    data_3d = [f3(xi, yj, zk) for xi in x, yj in y, zk in z]
    qx, qy, qz = 1.7, 0.8, 0.45

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

    @testset "Batch: GridIdx on an INTERPOLATING axis matches the scalar query" begin
        # Existing batch coverage puts GridIdx on a NoInterp axis, whose kernel
        # never reads the coordinate — so the unresolved `val = NaN` sentinel
        # stayed invisible. On an interpolating axis the batch loops must resolve
        # per point, exactly as the scalar entries do.
        for (nm, itp) in (
                ("linear", interp((x, y), data_2d; method = LinearInterp())),
                ("constant", interp((x, y), data_2d; method = ConstantInterp())),
                ("cubic", cubic_interp((x, y), data_2d)),
                ("quadratic", quadratic_interp((x, y), data_2d)),
            )
            @testset "$nm" begin
                # AoS (`Vector{<:Tuple}`): every point carries its own GridIdx.
                # (SoA with a scalar axis is a separate, pre-sliced entry — see
                # the `interp!` NoInterp testsets below.)
                @test itp([(GridIdx(2), GridIdx(3))])[1] === itp((GridIdx(2), GridIdx(3)))
                @test itp([(qx, GridIdx(3)), (GridIdx(2), qy)]) ==
                    [itp((qx, GridIdx(3))), itp((GridIdx(2), qy))]
            end
        end

        @testset "one-shot batch" begin
            @test cubic_interp((x, y), data_2d, [(GridIdx(2), GridIdx(3))])[1] ≈
                cubic_interp((x, y), data_2d)((GridIdx(2), GridIdx(3))) rtol = 1.0e-14
            @test interp((x, y), data_2d, [(GridIdx(2), GridIdx(3))]; method = LinearInterp())[1] ===
                interp((x, y), data_2d; method = LinearInterp())((GridIdx(2), GridIdx(3)))
        end

        @testset "3D + derivative" begin
            itp3 = interp((x, y, z), data_3d; method = LinearInterp())
            @test itp3([(qx, GridIdx(3), qz)])[1] === itp3((qx, GridIdx(3), qz))
            d = (DerivOp(1), DerivOp(0), DerivOp(0))
            @test itp3([(qx, GridIdx(3), qz)]; deriv = d)[1] === itp3((qx, GridIdx(3), qz); deriv = d)
        end
    end

    # ========================================
    # 14. Batch interp!
    # ========================================
    @testset "Batch: interp! 2D" begin
        xq_batch = collect(range(0.5, 5.0, 50))
        output = zeros(50)
        interp!(
            output, (x, y), data_2d, (xq_batch, GridIdx(5));
            method = (CubicInterp(), NoInterp())
        )
        ref = [cubic_interp(x, data_2d[:, 5], xqi) for xqi in xq_batch]
        @test output ≈ ref rtol = 1.0e-14
    end

    @testset "Batch: interp! 3D" begin
        xq_batch = collect(range(0.5, 5.0, 30))
        zq_batch = collect(range(0.1, 0.9, 30))
        output = zeros(30)
        interp!(
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
        interp!(
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
        interp!(
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
        # Laplacian type promotion (BUG 2 regression)
        L = laplacian(itp32, (GridIdx(2), Float64(2.3)))
        @test typeof(L) == Float64
    end

    # ========================================
    # 20. Regression: batch OOB + deriv zero (BUG 1)
    # ========================================
    @testset "Edge: batch OOB Real axis + NoInterp deriv → DomainError (not zeros)" begin
        out = zeros(1)
        @test_throws DomainError interp!(
            out, (x, y), data_2d, ([99.0], GridIdx(5));
            method = (CubicInterp(), NoInterp()), deriv = (DerivOp(0), DerivOp(1))
        )
    end

    # ========================================
    # 21. All-NoInterp gradient and hessian
    # ========================================
    @testset "gradient: all-NoInterp returns all zeros" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()))
        g = gradient(itp, (GridIdx(3), GridIdx(5)))
        @test all(g .== 0.0)
        @test length(g) == 2
    end

    @testset "hessian: all-NoInterp returns zero matrix" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()))
        H = hessian(itp, (GridIdx(3), GridIdx(5)))
        @test all(H .== 0.0)
        @test size(H) == (2, 2)
    end

    # ========================================
    # 22. Type inference for gradient/hessian/laplacian
    # ========================================
    @testset "Type inference: gradient with NoInterp" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        @test @inferred(gradient(itp, (qx, GridIdx(5)))) isa NTuple{2, Float64}
    end

    @testset "Type inference: laplacian with NoInterp" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        @test @inferred(laplacian(itp, (qx, GridIdx(5)))) isa Float64
    end

    # ========================================
    # 23. Batch with FillExtrap
    # ========================================
    @testset "Batch: FillExtrap with GridIdx" begin
        xq_b = collect(range(-1.0, 10.0, 20))  # some OOB
        out = zeros(20)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(5));
            method = (CubicInterp(), NoInterp()), extrap = (FillExtrap(-99.0), NoExtrap())
        )
        # OOB queries should get fill value
        @test out[1] == -99.0   # -1.0 is OOB (x starts at 0.0)
        @test out[end] == -99.0 # 10.0 is OOB (x ends at 2π ≈ 6.28)
        # In-range queries should be valid interpolation
        in_range = findall(xi -> 0.0 <= xi <= 2π, xq_b)
        @test all(out[in_range] .!= -99.0)
    end

    # ========================================
    # 24. QuadraticInterp × NoInterp
    # ========================================
    @testset "One-shot: Quadratic×NoInterp 2D" begin
        for k in [1, 10, 25]
            val = interp((x, y), data_2d, (qx, GridIdx(k)); method = (QuadraticInterp(), NoInterp()))
            ref = quadratic_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-13
        end
    end

    @testset "Interpolant PreCompute: Quadratic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (QuadraticInterp(), NoInterp()))
        for k in [1, 10, 25]
            val = itp((qx, GridIdx(k)))
            ref = quadratic_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-13
        end
        # Derivative on quadratic axis
        val_d = itp((qx, GridIdx(5)); deriv = (DerivOp(1), DerivOp(0)))
        ref_d = quadratic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test val_d ≈ ref_d rtol = 1.0e-10
    end

    # ========================================
    # 25. ConstantInterp × NoInterp
    # ========================================
    @testset "One-shot: Constant×NoInterp 2D" begin
        for k in [1, 10, 25]
            val = interp((x, y), data_2d, (qx, GridIdx(k)); method = (ConstantInterp(), NoInterp()))
            ref = constant_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Interpolant PreCompute: Constant×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (ConstantInterp(), NoInterp()))
        for k in [1, 10, 25]
            val = itp((qx, GridIdx(k)))
            ref = constant_interp(x, data_2d[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    # ========================================
    # 26. OnTheFly gradient / hessian / laplacian
    # ========================================
    @testset "OnTheFly gradient: Cubic×NoInterp 2D" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly()
        )
        g = gradient(itp, (qx, GridIdx(5)))
        ref = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test g[1] ≈ ref rtol = 1.0e-12
        @test g[2] == 0.0
    end

    @testset "OnTheFly hessian: Cubic×NoInterp 2D" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly()
        )
        H = hessian(itp, (qx, GridIdx(5)))
        ref_d2 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(2))
        @test H[1, 1] ≈ ref_d2 rtol = 1.0e-10
        @test H[1, 2] == 0.0
        @test H[2, 1] == 0.0
        @test H[2, 2] == 0.0
    end

    @testset "OnTheFly laplacian: Cubic×NoInterp 2D" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly()
        )
        L = laplacian(itp, (qx, GridIdx(5)))
        ref_d2 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(2))
        @test L ≈ ref_d2 rtol = 1.0e-10
    end

    # ========================================
    # 27. 3D hessian with NoInterp (mixed-partial correctness)
    # ========================================
    @testset "hessian: 3D Cubic×NoInterp×Linear" begin
        itp3 = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        H = hessian(itp3, (qx, GridIdx(5), qz))
        @test size(H) == (3, 3)
        # NoInterp row/column (axis 2) must be all zeros
        @test H[2, 1] == 0.0
        @test H[1, 2] == 0.0
        @test H[2, 3] == 0.0
        @test H[3, 2] == 0.0
        @test H[2, 2] == 0.0
        # Diagonal on interpolated axes should be non-trivial
        # H[1,1] = ∂²f/∂x² on the k=5 slice
        ref_d2x = interp(
            (x, z), data_3d[:, 5, :], (qx, qz);
            method = (CubicInterp(), LinearInterp()), deriv = (DerivOp(2), DerivOp(0))
        )
        @test H[1, 1] ≈ ref_d2x rtol = 1.0e-10
        # Mixed partial H[1,3] = ∂²f/∂x∂z on the k=5 slice
        ref_dxdz = interp(
            (x, z), data_3d[:, 5, :], (qx, qz);
            method = (CubicInterp(), LinearInterp()), deriv = (DerivOp(1), DerivOp(1))
        )
        @test H[1, 3] ≈ ref_dxdz rtol = 1.0e-10
        @test H[3, 1] ≈ ref_dxdz rtol = 1.0e-10  # symmetry
    end

end

# ========================================
# 28. ClampExtrap × NoInterp
# ========================================
@testitem "NoInterp + GridIdx: ClampExtrap & Periodic & Hints & Cross-method" setup = [AllocConstants] begin
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    z = range(0.0, 1.0, 20)
    f2(xi, yj) = sin(xi) * cos(yj)
    f3(xi, yj, zk) = sin(xi) * cos(yj) * exp(-zk)
    data_2d = [f2(xi, yj) for xi in x, yj in y]
    data_3d = [f3(xi, yj, zk) for xi in x, yj in y, zk in z]
    qx, qy, qz = 1.7, 0.8, 0.45

    @testset "ClampExtrap: interpolant with NoInterp" begin
        itp = interp(
            (x, y), data_2d;
            method = (CubicInterp(), NoInterp()), extrap = (ClampExtrap(), NoExtrap())
        )
        # Query mid-grid interior reference for high/low OOB → clamped to boundary
        # Use mid-grid k to avoid near-zero sin boundary values
        val_hi = itp((99.0, GridIdx(12)))
        ref_hi = cubic_interp(x, data_2d[:, 12], last(x))  # clamped to x[end]
        @test val_hi ≈ ref_hi atol = 1.0e-14
        val_lo = itp((-1.0, GridIdx(12)))
        ref_lo = cubic_interp(x, data_2d[:, 12], first(x))  # clamped to x[1]
        @test val_lo ≈ ref_lo atol = 1.0e-14
    end

    @testset "ClampExtrap: one-shot with NoInterp" begin
        val = interp(
            (x, y), data_2d, (99.0, GridIdx(12));
            method = (CubicInterp(), NoInterp()), extrap = (ClampExtrap(), NoExtrap())
        )
        ref = cubic_interp(x, data_2d[:, 12], last(x))
        @test val ≈ ref atol = 1.0e-14
    end

    # ========================================
    # 29. show() for NoInterp interpolant
    # ========================================
    @testset "show: interpolant with NoInterp" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        s = sprint(show, itp)
        @test occursin("NoInterp", s)
        @test occursin("Cubic", s)
        @test occursin("30×25", s)
    end

    # ========================================
    # 30. Batch with multiple GridIdx axes (3D, 2 NoInterp)
    # ========================================
    @testset "Batch: 3D with 2 GridIdx axes" begin
        # (NoInterp, NoInterp, Cubic) → batch over z axis only
        zq_batch = collect(range(0.1, 0.9, 20))
        out = zeros(20)
        interp!(
            out, (x, y, z), data_3d, (GridIdx(5), GridIdx(10), zq_batch);
            method = (NoInterp(), NoInterp(), CubicInterp())
        )
        ref = [cubic_interp(z, data_3d[5, 10, :], zqi) for zqi in zq_batch]
        @test out ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 31. One-shot deriv + FillExtrap combined edge case
    # ========================================
    @testset "One-shot: FillExtrap OOB + deriv on Real axis" begin
        # OOB query with deriv on the Real axis → zero (derivative of constant fill = 0)
        val = interp(
            (x, y), data_2d, (99.0, GridIdx(5));
            method = (CubicInterp(), NoInterp()),
            deriv = (DerivOp(1), DerivOp(0)),
            extrap = (FillExtrap(-99.0), NoExtrap())
        )
        @test val == 0.0
        # EvalValue on OOB → fill value (baseline check)
        val0 = interp(
            (x, y), data_2d, (99.0, GridIdx(5));
            method = (CubicInterp(), NoInterp()),
            extrap = (FillExtrap(-99.0), NoExtrap())
        )
        @test val0 == -99.0
    end

    @testset "One-shot: FillExtrap OOB + deriv on NoInterp axis → DomainError" begin
        # OOB on Real axis + deriv on NoInterp axis → DomainError (domain check first)
        @test_throws DomainError interp(
            (x, y), data_2d, (99.0, GridIdx(5));
            method = (CubicInterp(), NoInterp()),
            deriv = (DerivOp(0), DerivOp(1))
            # NoExtrap() default → DomainError on OOB Real axis
        )
    end

    # ========================================
    # 32. Coverage gap tests
    # ========================================

    @testset "Coverage: GridIdx callable on CubicInterpolantND" begin
        itp_c = cubic_interp((x, y), data_2d)
        # Tuple form
        val = itp_c((qx, GridIdx(10)))
        ref = itp_c((qx, y[10]))
        @test val ≈ ref rtol = 1.0e-14
        # Vararg form
        @test itp_c(qx, GridIdx(10)) ≈ ref rtol = 1.0e-14
    end

    @testset "Coverage: GridIdx callable on LinearInterpolantND" begin
        itp_l = linear_interp((x, y), data_2d)
        val = itp_l((qx, GridIdx(10)))
        ref = itp_l((qx, y[10]))
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "Coverage: All-NoInterp OnTheFly pure lookup" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), NoInterp()), coeffs = OnTheFly())
        @test itp((GridIdx(3), GridIdx(7))) == data_2d[3, 7]
    end

    @testset "Coverage: All-GridIdx one-shot with deriv → zero" begin
        val = interp(
            (x, y), data_2d, (GridIdx(3), GridIdx(7));
            method = (NoInterp(), NoInterp()), deriv = (DerivOp(1), DerivOp(0))
        )
        @test val == 0.0
    end

    @testset "Coverage: All-GridIdx batch early return" begin
        out = [0.0]
        interp!(
            out, (x, y), data_2d, (GridIdx(3), GridIdx(7));
            method = (NoInterp(), NoInterp())
        )
        @test out == [0.0]  # returned unchanged (no Real axes to batch)
    end

    @testset "Coverage: Invalid query element type in one-shot" begin
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, "not_a_number");
            method = (CubicInterp(), NoInterp())
        )
    end

    @testset "Coverage: One-shot NoInterp guard (all-Real query + NoInterp method)" begin
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, qy);
            method = (CubicInterp(), NoInterp())
        )
    end

    @testset "Coverage: show NoInterp on non-last axis" begin
        itp = interp((x, y), data_2d; method = (NoInterp(), CubicInterp()))
        s = sprint(show, MIME("text/plain"), itp)
        @test occursin("Axis 1: NoInterp", s)
        @test occursin("Axis 2: Cubic", s)
        @test occursin("BC: CubicFit", s)
    end

    @testset "Coverage: show ConstantInterp side in HeteroInterpolantND" begin
        itp = interp((x, y), data_2d; method = (ConstantInterp(), NoInterp()))
        s = sprint(show, MIME("text/plain"), itp)
        @test occursin("Side:", s)
        @test occursin("NearestSide", s)
    end

    @testset "Coverage: NoInterp grid/data dimension mismatch" begin
        bad_data = rand(29, 25)  # x has 30 points, data has 29
        @test_throws DimensionMismatch interp(
            (x, y), bad_data; method = (CubicInterp(), NoInterp())
        )
    end

    @testset "Coverage: Non-NoInterp axis with 1-point grid (NoInterp sibling)" begin
        tiny_grid = [0.0]
        big_data = rand(1, 25)
        # Axis 1 has 1 point but is NOT NoInterp → should error
        @test_throws ArgumentError interp(
            (tiny_grid, y), big_data; method = (CubicInterp(), NoInterp())
        )
    end

    # ========================================
    # 33. PeriodicBC + NoInterp (Gap 3)
    # ========================================
    @testset "PeriodicBC(exclusive) × NoInterp: interpolant" begin
        # Periodic on axis 1 (exclusive endpoint), NoInterp on axis 2
        xp = range(0.0, step = 2π / 30, length = 30)  # exclusive: last ≠ first
        data_p = [sin(xi) * (k / 25.0) for xi in xp, k in 1:25]
        itp = interp(
            (xp, y), data_p;
            method = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), NoInterp()),
            extrap = (WrapExtrap(), NoExtrap())
        )
        for k in [1, 10, 25]
            val = itp((1.5, GridIdx(k)))
            ref = cubic_interp(
                xp, data_p[:, k], 1.5;
                bc = PeriodicBC(endpoint = :exclusive), extrap = WrapExtrap()
            )
            @test val ≈ ref rtol = 1.0e-12
        end
        # Periodicity: f(x + 2π) == f(x) with NoInterp axis
        @test itp((1.5 + 2π, GridIdx(5))) ≈ itp((1.5, GridIdx(5))) rtol = 1.0e-12
    end

    @testset "PeriodicBC(exclusive) × NoInterp: one-shot" begin
        xp = range(0.0, step = 2π / 30, length = 30)
        data_p = [sin(xi) * cos(yj) for xi in xp, yj in y]
        val = interp(
            (xp, y), data_p, (1.5, GridIdx(10));
            method = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), NoInterp()),
            extrap = (WrapExtrap(), NoExtrap())
        )
        ref = cubic_interp(
            xp, data_p[:, 10], 1.5;
            bc = PeriodicBC(endpoint = :exclusive), extrap = WrapExtrap()
        )
        @test val ≈ ref rtol = 1.0e-12
    end

    @testset "PeriodicBC(inclusive) × NoInterp: interpolant" begin
        # Inclusive endpoint: first == last, with 31 points spanning [0, 2π]
        xp_inc = range(0.0, 2π, 31)
        data_inc = [sin(xi) * (k / 25.0) for xi in xp_inc, k in 1:25]
        # Enforce exact periodicity: data[end,:] = data[1,:]
        data_inc[end, :] .= data_inc[1, :]
        itp = interp(
            (xp_inc, y), data_inc;
            method = (CubicInterp(bc = PeriodicBC(endpoint = :inclusive)), NoInterp()),
            extrap = (WrapExtrap(), NoExtrap())
        )
        for k in [1, 12, 25]
            val = itp((2.0, GridIdx(k)))
            ref = cubic_interp(
                xp_inc, data_inc[:, k], 2.0;
                bc = PeriodicBC(endpoint = :inclusive), extrap = WrapExtrap()
            )
            @test val ≈ ref rtol = 1.0e-12
        end
    end

    @testset "PeriodicBC × NoInterp: gradient" begin
        xp = range(0.0, step = 2π / 30, length = 30)
        data_p = [sin(xi) * (k / 25.0) for xi in xp, k in 1:25]
        itp = interp(
            (xp, y), data_p;
            method = (CubicInterp(bc = PeriodicBC(endpoint = :exclusive)), NoInterp()),
            extrap = (WrapExtrap(), NoExtrap())
        )
        g = gradient(itp, (1.5, GridIdx(10)))
        ref_d1 = cubic_interp(
            xp, data_p[:, 10], 1.5;
            bc = PeriodicBC(endpoint = :exclusive), extrap = WrapExtrap(),
            deriv = DerivOp(1)
        )
        @test g[1] ≈ ref_d1 rtol = 1.0e-10
        @test g[2] == 0.0  # NoInterp axis
    end

    # ========================================
    # 34. Vector grid on NoInterp axis (Gap 4)
    # ========================================
    @testset "Vector grid on NoInterp axis: interpolant" begin
        y_vec = collect(range(0.0, π, 25))  # Vector, not Range
        data_vy = [sin(xi) * cos(yj) for xi in x, yj in y_vec]
        itp = interp((x, y_vec), data_vy; method = (CubicInterp(), NoInterp()))
        for k in [1, 5, 15, 25]
            val = itp((qx, GridIdx(k)))
            ref = cubic_interp(x, data_vy[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Vector grid on NoInterp axis: one-shot" begin
        y_vec = collect(range(0.0, π, 25))
        data_vy = [sin(xi) * cos(yj) for xi in x, yj in y_vec]
        for k in [1, 12, 25]
            val = interp(
                (x, y_vec), data_vy, (qx, GridIdx(k));
                method = (CubicInterp(), NoInterp())
            )
            ref = cubic_interp(x, data_vy[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Vector grid on NoInterp axis: gradient" begin
        y_vec = collect(range(0.0, π, 25))
        data_vy = [sin(xi) * cos(yj) for xi in x, yj in y_vec]
        itp = interp((x, y_vec), data_vy; method = (CubicInterp(), NoInterp()))
        g = gradient(itp, (qx, GridIdx(5)))
        ref = cubic_interp(x, data_vy[:, 5], qx; deriv = DerivOp(1))
        @test g[1] ≈ ref rtol = 1.0e-12
        @test g[2] == 0.0
    end

    @testset "Vector grid on interp axis + NoInterp: interpolant" begin
        x_vec = collect(range(0.0, 2π, 30))  # Vector grid on interpolated axis
        data_vx = [sin(xi) * cos(yj) for xi in x_vec, yj in y]
        itp = interp((x_vec, y), data_vx; method = (CubicInterp(), NoInterp()))
        for k in [1, 10, 25]
            val = itp((qx, GridIdx(k)))
            ref = cubic_interp(x_vec, data_vx[:, k], qx)
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    # ========================================
    # 35. Hints with NoInterp (Gap 5)
    # ========================================
    # Hint = NTuple{N, Ref{Int}}, N-element (full dims).
    # @generated _eval_nointerp filters by real_dims: hint_r = (hint[d] for d in real_dims).
    # After search, hint Ref is updated to the found interval index.
    # NoInterp axis hints must NOT be touched.

    @testset "Hints: all axes updated (real=interval, NoInterp=GridIdx)" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        hint = (Ref(1), Ref(1))
        val = itp((qx, GridIdx(5)); hint = hint)
        ref = cubic_interp(x, data_2d[:, 5], qx)
        @test val ≈ ref rtol = 1.0e-14
        # Real axis: updated to correct interval index
        @test hint[1][] > 1
        @test 1 <= hint[1][] <= length(x) - 1
        @test x[hint[1][]] <= qx < x[hint[1][] + 1]
        # NoInterp axis: updated to GridIdx value
        @test hint[2][] == 5
    end

    @testset "Hints: NoInterp hint tracks GridIdx across queries" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        hint = (Ref(1), Ref(1))
        itp((qx, GridIdx(3)); hint = hint)
        @test hint[2][] == 3
        itp((qx, GridIdx(20)); hint = hint)
        @test hint[2][] == 20
        itp((qx, GridIdx(1)); hint = hint)
        @test hint[2][] == 1
    end

    @testset "Hints: sequential queries update progressively" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        hint = (Ref(1), Ref(1))
        prev_idx = 0
        for qi in range(0.5, 5.5, 10)
            val = itp((qi, GridIdx(5)); hint = hint)
            ref = cubic_interp(x, data_2d[:, 5], qi)
            @test val ≈ ref rtol = 1.0e-14
            @test hint[1][] >= prev_idx
            @test x[hint[1][]] <= qi < x[hint[1][] + 1]
            prev_idx = hint[1][]
        end
        @test hint[2][] == 5  # NoInterp axis reflects last GridIdx
    end

    @testset "Hints: one-shot with GridIdx" begin
        hint = (Ref(1), Ref(1))
        for qi in range(0.5, 5.0, 5)
            val = interp(
                (x, y), data_2d, (qi, GridIdx(12));
                method = (CubicInterp(), NoInterp()), hint = hint
            )
            ref = cubic_interp(x, data_2d[:, 12], qi)
            @test val ≈ ref rtol = 1.0e-14
        end
        @test hint[1][] > 1
        @test hint[2][] == 12
    end

    @testset "Hints: 3D Cubic×NoInterp×Linear, all axes updated" begin
        itp3 = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        hint = (Ref(1), Ref(1), Ref(1))
        val = itp3((qx, GridIdx(10), qz); hint = hint)
        ref = interp(
            (x, z), data_3d[:, 10, :], (qx, qz);
            method = (CubicInterp(), LinearInterp())
        )
        @test val ≈ ref rtol = 1.0e-13
        # Axis 1 (Cubic): interval index
        @test hint[1][] > 1
        @test x[hint[1][]] <= qx < x[hint[1][] + 1]
        # Axis 2 (NoInterp): GridIdx value
        @test hint[2][] == 10
        # Axis 3 (Linear): interval index
        @test hint[3][] >= 1
        @test z[hint[3][]] <= qz < z[hint[3][] + 1]
    end

    # ========================================
    # 36. Batch deriv on real axis + NoInterp (Gap 7)
    # ========================================
    @testset "Batch: deriv on real axis (not NoInterp axis)" begin
        xq_b = collect(range(0.5, 5.0, 20))
        out = zeros(20)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(5));
            method = (CubicInterp(), NoInterp()), deriv = (DerivOp(1), DerivOp(0))
        )
        ref = [cubic_interp(x, data_2d[:, 5], xqi; deriv = DerivOp(1)) for xqi in xq_b]
        @test out ≈ ref rtol = 1.0e-12
    end

    @testset "Batch: 2nd deriv on real axis + NoInterp" begin
        xq_b = collect(range(0.5, 5.0, 20))
        out = zeros(20)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(5));
            method = (CubicInterp(), NoInterp()), deriv = (DerivOp(2), DerivOp(0))
        )
        ref = [cubic_interp(x, data_2d[:, 5], xqi; deriv = DerivOp(2)) for xqi in xq_b]
        @test out ≈ ref rtol = 1.0e-10
    end

    # ========================================
    # 37. value_gradient with GridIdx (Gap 1)
    # ========================================
    @testset "value_gradient: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        val, g = value_gradient(itp, (qx, GridIdx(5)))
        ref_val = cubic_interp(x, data_2d[:, 5], qx)
        ref_d1 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test val ≈ ref_val rtol = 1.0e-14
        @test g[1] ≈ ref_d1 rtol = 1.0e-12
        @test g[2] == 0.0  # NoInterp axis
        @test length(g) == 2
    end

    @testset "value_gradient: non-NoInterp with GridIdx (generic path)" begin
        # CubicInterpolantND (not HeteroInterpolantND) → generic conversion path
        itp_c = cubic_interp((x, y), data_2d)
        val, g = value_gradient(itp_c, (qx, GridIdx(10)))
        ref_val, ref_g = value_gradient(itp_c, (qx, y[10]))
        @test val ≈ ref_val rtol = 1.0e-14
        @test g[1] ≈ ref_g[1] rtol = 1.0e-14
        @test g[2] ≈ ref_g[2] rtol = 1.0e-14
    end

    # ========================================
    # 38. gradient! with GridIdx (Gap 2)
    # ========================================
    @testset "gradient!: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        G = zeros(2)
        gradient!(G, itp, (qx, GridIdx(5)))
        ref_d1 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(1))
        @test G[1] ≈ ref_d1 rtol = 1.0e-12
        @test G[2] == 0.0
    end

    @testset "gradient!: non-NoInterp with GridIdx (generic path)" begin
        itp_c = cubic_interp((x, y), data_2d)
        G1 = zeros(2)
        G2 = zeros(2)
        gradient!(G1, itp_c, (qx, GridIdx(10)))
        gradient!(G2, itp_c, (qx, y[10]))
        @test G1 ≈ G2 rtol = 1.0e-14
    end

    # ========================================
    # 39. hessian! with GridIdx (Gap 2)
    # ========================================
    @testset "hessian!: Cubic×NoInterp 2D" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        H = zeros(2, 2)
        hessian!(H, itp, (qx, GridIdx(5)))
        ref_d2 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(2))
        @test H[1, 1] ≈ ref_d2 rtol = 1.0e-10
        @test H[1, 2] == 0.0
        @test H[2, 1] == 0.0
        @test H[2, 2] == 0.0
    end

    @testset "hessian!: non-NoInterp with GridIdx (generic path)" begin
        itp_c = cubic_interp((x, y), data_2d)
        H1 = zeros(2, 2)
        H2 = zeros(2, 2)
        hessian!(H1, itp_c, (qx, GridIdx(10)))
        hessian!(H2, itp_c, (qx, y[10]))
        @test H1 ≈ H2 rtol = 1.0e-14
    end

    # ========================================
    # 40. GridIdx on non-NoInterp axes (universal query protocol)
    # ========================================
    @testset "GridIdx universal: hetero-interpolant callable" begin
        # HeteroInterpolantND with no NoInterp — GridIdx should convert to grids[d][k]
        itp_h = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))
        val = itp_h((qx, GridIdx(10)))
        ref = itp_h((qx, y[10]))
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "GridIdx universal: hetero 3D, multiple GridIdx" begin
        itp_h3 = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), LinearInterp(), CubicInterp())
        )
        val = itp_h3((qx, GridIdx(10), GridIdx(5)))
        ref = itp_h3((qx, y[10], z[5]))
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "GridIdx universal: hetero with deriv" begin
        itp_h = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))
        val = itp_h((qx, GridIdx(10)); deriv = (DerivOp(1), DerivOp(0)))
        ref = itp_h((qx, y[10]); deriv = (DerivOp(1), DerivOp(0)))
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "GridIdx universal: mixed NoInterp + non-NoInterp GridIdx" begin
        # 3D: axis 1 Cubic (GridIdx converts to Real), axis 2 NoInterp, axis 3 Linear (GridIdx converts)
        itp_mix = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        # GridIdx on all 3 axes: axis 1,3 convert to Real, axis 2 stays GridIdx
        val = itp_mix((GridIdx(15), GridIdx(10), GridIdx(5)))
        ref = itp_mix((x[15], GridIdx(10), z[5]))
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 41. Tuple-deriv auto-promotion (Gap 1)
    # ========================================
    # _all_eval_value must accept both scalar EvalValue() and tuple of EvalValue().
    # Both forms should trigger GridIdx → NoInterp auto-promotion in one-shot path.
    @testset "Auto-promotion: tuple deriv (EvalValue(), EvalValue())" begin
        # One-shot with explicit tuple of all-EvalValue — should match scalar default
        val_scalar = interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), CubicInterp()), deriv = EvalValue()
        )
        val_tuple = interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), CubicInterp()), deriv = (EvalValue(), EvalValue())
        )
        @test val_scalar ≈ val_tuple rtol = 1.0e-14
        # Reference: GridIdx(5) → y[5], both should match full 2D eval
        ref = interp(
            (x, y), data_2d, (qx, y[5]);
            method = (CubicInterp(), CubicInterp())
        )
        @test val_tuple ≈ ref rtol = 1.0e-14
    end

    @testset "Auto-promotion: tuple deriv with non-EvalValue skips promotion" begin
        # deriv=(DerivOp(1), EvalValue()) — NOT all EvalValue → no auto-promotion
        # GridIdx on axis 2 with CubicInterp should still work (search short-circuit)
        val = interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(1), EvalValue())
        )
        ref = interp(
            (x, y), data_2d, (qx, y[5]);
            method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(1), EvalValue())
        )
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "Auto-promotion: scalar DerivOp(1) skips promotion" begin
        # Scalar deriv (not tuple) — _all_eval_value(::DerivOp) = false
        # GridIdx on non-NoInterp axis → search short-circuit, no auto-promotion
        val = interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), CubicInterp()), deriv = DerivOp(1)
        )
        ref = interp(
            (x, y), data_2d, (qx, y[5]);
            method = (CubicInterp(), CubicInterp()), deriv = DerivOp(1)
        )
        @test val ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 42. Float32 grids + GridIdx resolution (Gap 2)
    # ========================================
    @testset "Float32: GridIdx resolves to GridIdx{Float32}" begin
        x32 = range(0.0f0, Float32(2π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = Float32[sin(xi) * cos(yj) for xi in x32, yj in y32]
        # Bare GridIdx has Float64 NaN sentinel; after resolution, val should be Float32
        resolved = FastInterpolations._resolve_grididx(GridIdx(5), y32)
        @test resolved isa FastInterpolations.GridIdx{Float32}
        @test resolved.val == y32[5]
        @test resolved.val isa Float32
        # Full pipeline: Float32 grid + GridIdx → Float32 result
        itp32 = interp((x32, y32), data32; method = (CubicInterp(), NoInterp()))
        val = itp32((1.7f0, GridIdx(10)))
        @test val isa Float32
        ref = cubic_interp(x32, data32[:, 10], 1.7f0)
        @test val ≈ ref rtol = 1.0f-5
    end

    @testset "Float32: one-shot GridIdx on non-NoInterp axis" begin
        x32 = range(0.0f0, Float32(2π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        data32 = Float32[sin(xi) * cos(yj) for xi in x32, yj in y32]
        val = interp(
            (x32, y32), data32, (1.7f0, GridIdx(10));
            method = (CubicInterp(), CubicInterp())
        )
        ref = interp(
            (x32, y32), data32, (1.7f0, y32[10]);
            method = (CubicInterp(), CubicInterp())
        )
        @test val ≈ ref rtol = 1.0f-5
        @test val isa Float32
    end

    # ========================================
    # 43. Unresolved GridIdx arithmetic (Gap 3)
    # ========================================
    @testset "Unresolved GridIdx: NaN sentinel" begin
        g = GridIdx(5)
        # Before resolution, val is NaN (poison sentinel)
        @test isnan(convert(Float64, g))
        @test isnan(float(g))
        @test isnan(Float64(g))
        # After resolution, val is valid
        resolved = FastInterpolations._resolve_grididx(g, x)
        @test !isnan(convert(Float64, resolved))
        @test convert(Float64, resolved) ≈ x[5]
    end

    # ========================================
    # 44. ForwardDiff AD + GridIdx (Gap 4)
    # ========================================
    @testset "ForwardDiff: Dual + GridIdx in same query (interpolant)" begin
        ForwardDiff = Base.require(
            Base.PkgId(
                Base.UUID("f6369f11-7733-5829-9624-2563aa707210"), "ForwardDiff"
            )
        )
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        # Differentiate w.r.t. Real axis (axis 1) while NoInterp axis uses GridIdx
        df = ForwardDiff.derivative(t -> itp((t, GridIdx(10))), qx)
        ref = cubic_interp(x, data_2d[:, 10], qx; deriv = DerivOp(1))
        @test df ≈ ref rtol = 1.0e-10
    end

    @testset "ForwardDiff: Dual + GridIdx on non-NoInterp axis" begin
        ForwardDiff = Base.require(
            Base.PkgId(
                Base.UUID("f6369f11-7733-5829-9624-2563aa707210"), "ForwardDiff"
            )
        )
        # HeteroInterpolantND with Cubic×Linear, GridIdx on axis 2 (not NoInterp)
        itp_h = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()))
        df_grididx = ForwardDiff.derivative(t -> itp_h((t, GridIdx(10))), qx)
        df_real = ForwardDiff.derivative(t -> itp_h((t, y[10])), qx)
        @test df_grididx ≈ df_real rtol = 1.0e-12
    end

    # ========================================
    # 45. Batch interp! with GridIdx correctness (Gap 5)
    # ========================================
    @testset "Batch interp!: GridIdx on non-NoInterp axis" begin
        # interp! with GridIdx in batch queries (no NoInterp method)
        xq_b = collect(range(0.5, 5.0, 20))
        out = zeros(20)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(5));
            method = (CubicInterp(), NoInterp())
        )
        ref = [cubic_interp(x, data_2d[:, 5], xqi) for xqi in xq_b]
        @test out ≈ ref rtol = 1.0e-14
    end

    @testset "Batch interp!: GridIdx with CubicInterp (no NoInterp)" begin
        # Both axes are CubicInterp, GridIdx on axis 2 — batch path correctness
        xq_b = collect(range(0.5, 5.0, 15))
        out = zeros(15)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(10));
            method = (CubicInterp(), CubicInterp())
        )
        ref = [interp((x, y), data_2d, (xqi, y[10]); method = (CubicInterp(), CubicInterp())) for xqi in xq_b]
        @test out ≈ ref rtol = 1.0e-14
    end

    # ========================================
    # 46. Laplacian with GridIdx (Gap 6)
    # ========================================
    @testset "laplacian: Cubic×NoInterp 2D correctness" begin
        itp = interp((x, y), data_2d; method = (CubicInterp(), NoInterp()))
        L = laplacian(itp, (qx, GridIdx(5)))
        # Laplacian = sum of 2nd derivs; NoInterp axis contributes 0
        ref_d2 = cubic_interp(x, data_2d[:, 5], qx; deriv = DerivOp(2))
        @test L ≈ ref_d2 rtol = 1.0e-10
    end

    @testset "laplacian: 3D Cubic×NoInterp×Linear" begin
        itp3 = interp(
            (x, y, z), data_3d;
            method = (CubicInterp(), NoInterp(), LinearInterp())
        )
        L = laplacian(itp3, (qx, GridIdx(10), qz))
        # Laplacian = ∂²f/∂x² + 0 (NoInterp) + ∂²f/∂z²
        # Linear 2nd deriv is zero, so laplacian = just ∂²f/∂x²
        ref = itp3((qx, GridIdx(10), qz); deriv = (DerivOp(2), DerivOp(0), DerivOp(0)))
        @test L ≈ ref rtol = 1.0e-10
    end

    @testset "laplacian: non-NoInterp with GridIdx (generic path)" begin
        itp_c = cubic_interp((x, y), data_2d)
        L_grididx = laplacian(itp_c, (qx, GridIdx(10)))
        L_real = laplacian(itp_c, (qx, y[10]))
        @test L_grididx ≈ L_real rtol = 1.0e-14
    end

    # ========================================
    # 47. GridIdx on non-NoInterp: full correctness (one-shot + batch + deriv)
    # ========================================
    # GridIdx on a non-NoInterp axis must produce the same result as using
    # the grid coordinate directly. This applies to value, all derivatives,
    # and both one-shot and batch paths.

    @testset "Non-NoInterp GridIdx: one-shot value matches Real" begin
        for k in [1, 10, 25]
            val = interp(
                (x, y), data_2d, (qx, GridIdx(k));
                method = (CubicInterp(), CubicInterp())
            )
            ref = interp(
                (x, y), data_2d, (qx, y[k]);
                method = (CubicInterp(), CubicInterp())
            )
            @test val ≈ ref rtol = 1.0e-14
        end
    end

    @testset "Non-NoInterp GridIdx: one-shot deriv on GridIdx axis" begin
        # ∂f/∂y at y[k] — GridIdx should NOT zero this (it's not NoInterp)
        val = interp(
            (x, y), data_2d, (qx, GridIdx(10));
            method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(0), DerivOp(1))
        )
        ref = interp(
            (x, y), data_2d, (qx, y[10]);
            method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(0), DerivOp(1))
        )
        @test val ≈ ref rtol = 1.0e-14
        @test val != 0.0  # must NOT be zero-filled
    end

    @testset "Non-NoInterp GridIdx: one-shot scalar deriv" begin
        val = interp(
            (x, y), data_2d, (qx, GridIdx(10));
            method = (CubicInterp(), CubicInterp()), deriv = DerivOp(1)
        )
        ref = interp(
            (x, y), data_2d, (qx, y[10]);
            method = (CubicInterp(), CubicInterp()), deriv = DerivOp(1)
        )
        @test val ≈ ref rtol = 1.0e-14
    end

    @testset "Non-NoInterp GridIdx: batch deriv on GridIdx axis" begin
        xq_b = collect(range(0.5, 5.0, 15))
        out = zeros(15)
        # ∂f/∂y at y[10] via batch — must NOT zero-fill
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(10));
            method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(0), DerivOp(1))
        )
        ref = [
            interp(
                    (x, y), data_2d, (xqi, y[10]);
                    method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(0), DerivOp(1))
                ) for xqi in xq_b
        ]
        @test out ≈ ref rtol = 1.0e-14
        @test any(!iszero, out)  # must NOT be all zeros
    end

    @testset "Non-NoInterp GridIdx: batch value (no deriv)" begin
        xq_b = collect(range(0.5, 5.0, 15))
        out = zeros(15)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(10));
            method = (CubicInterp(), CubicInterp())
        )
        ref = [
            interp(
                    (x, y), data_2d, (xqi, y[10]);
                    method = (CubicInterp(), CubicInterp())
                ) for xqi in xq_b
        ]
        @test out ≈ ref rtol = 1.0e-14
    end

    @testset "Non-NoInterp GridIdx: batch deriv on Real axis" begin
        xq_b = collect(range(0.5, 5.0, 15))
        out = zeros(15)
        # ∂f/∂x with GridIdx on axis 2 (non-NoInterp)
        interp!(
            out, (x, y), data_2d, (xq_b, GridIdx(10));
            method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(1), DerivOp(0))
        )
        ref = [
            interp(
                    (x, y), data_2d, (xqi, y[10]);
                    method = (CubicInterp(), CubicInterp()), deriv = (DerivOp(1), DerivOp(0))
                ) for xqi in xq_b
        ]
        @test out ≈ ref rtol = 1.0e-14
    end

    @testset "Mixed NoInterp + non-NoInterp GridIdx: batch deriv" begin
        # 3D: axis 1 Cubic (Real batch), axis 2 NoInterp (GridIdx), axis 3 Linear (GridIdx)
        # deriv on axis 3 (non-NoInterp GridIdx) — must NOT zero-fill
        xq_b = collect(range(0.5, 5.0, 10))
        out = zeros(10)
        interp!(
            out, (x, y, z), data_3d, (xq_b, GridIdx(10), GridIdx(5));
            method = (CubicInterp(), NoInterp(), LinearInterp()),
            deriv = (DerivOp(0), DerivOp(0), DerivOp(1))
        )
        ref = [
            interp(
                    (x, y, z), data_3d, (xqi, GridIdx(10), z[5]);
                    method = (CubicInterp(), NoInterp(), LinearInterp()),
                    deriv = (DerivOp(0), DerivOp(0), DerivOp(1))
                ) for xqi in xq_b
        ]
        @test out ≈ ref rtol = 1.0e-12
    end

    # ========================================
    # Coeffs kwarg interaction with GridIdx auto-promotion
    #
    # When GridIdx(k) is supplied on a Hermite (PchipInterp/CardinalInterp/
    # AkimaInterp) axis with `deriv=EvalValue()`, the auto-promotion replaces
    # that axis's method with NoInterp() and pre-slices the data. The reduced
    # problem no longer contains a Hermite axis, so an explicit
    # `coeffs=PreCompute()` should be honored on the surviving axes rather
    # than rejected by the un-promoted method tuple.
    # ========================================

    @testset "GridIdx auto-promotion: scalar interp + coeffs=PreCompute" begin
        # Pchip axis is GridIdx'd → after promotion, only the cubic axis remains.
        # PreCompute() must succeed and match the manually-sliced 1D cubic call.
        for k in (1, 5, 13, 25)
            val = interp(
                (x, y), data_2d, (qx, GridIdx(k));
                method = (CubicInterp(), PchipInterp()), coeffs = PreCompute(),
            )
            @test val ≈ cubic_interp(x, data_2d[:, k], qx) rtol = 1.0e-12
        end
        # Non-EvalValue deriv on the Pchip axis disables promotion → must reject.
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, GridIdx(5));
            method = (CubicInterp(), PchipInterp()),
            deriv = (EvalValue(), DerivOp(1)),
            coeffs = PreCompute(),
        )
    end

    @testset "GridIdx auto-promotion: batch interp! forwards coeffs" begin
        # Mirrors the scalar auto-promotion test above (line 1544), but for
        # `interp!` batch. The batch GridIdx helper previously expanded non-
        # NoInterp GridIdx axes to constant Real vectors and recursed with the
        # UNPROMOTED method tuple, which (incorrectly) rejected `coeffs=
        # PreCompute()` for any local-Hermite axis — even though the reduced
        # 1-D problem after slicing was pure Cubic and fully PreCompute-capable.
        # Fix: promote GridIdx → NoInterp when `_all_eval_value(deriv)` is true,
        # mirroring the scalar `interp` GridIdx auto-promotion implemented by
        # `_promote_grididx_to_nointerp`.

        # Case (a): explicit NoInterp axis — `coeffs` must flow to the reduced
        # sub-problem (pre-existing; verifies `coeffs` isn't silently dropped
        # on the way down).
        qx_b = collect(range(0.5, 2.5, 7))
        out_pre = zeros(7)
        out_otf = zeros(7)
        interp!(
            out_pre, (x, y), data_2d, (qx_b, GridIdx(8));
            method = (CubicInterp(), NoInterp()), coeffs = PreCompute(),
        )
        interp!(
            out_otf, (x, y), data_2d, (qx_b, GridIdx(8));
            method = (CubicInterp(), NoInterp()), coeffs = OnTheFly(),
        )
        ref = [cubic_interp(x, data_2d[:, 8], q) for q in qx_b]
        @test out_pre ≈ ref rtol = 1.0e-12
        @test out_otf ≈ ref rtol = 1.0e-12

        # Case (b): Hermite axis is GridIdx'd + EvalValue deriv + PreCompute.
        # Auto-promotion converts Pchip → NoInterp at the GridIdx position,
        # leaving a pure Cubic 1-D problem that PreCompute supports. Result
        # must match the manually-sliced 1-D cubic call element-wise.
        # Regression guard for the concrete example Codex flagged:
        #   interp!(out, (x,y), data, (qx_batch, GridIdx(8));
        #           method=(CubicInterp(), PchipInterp()), coeffs=PreCompute())
        # used to throw "PreCompute not yet supported for PchipInterp in ND".
        out_promoted_pre = zeros(7)
        out_promoted_otf = zeros(7)
        interp!(
            out_promoted_pre, (x, y), data_2d, (qx_b, GridIdx(8));
            method = (CubicInterp(), PchipInterp()), coeffs = PreCompute(),
        )
        interp!(
            out_promoted_otf, (x, y), data_2d, (qx_b, GridIdx(8));
            method = (CubicInterp(), PchipInterp()), coeffs = OnTheFly(),
        )
        @test out_promoted_pre ≈ ref rtol = 1.0e-12
        @test out_promoted_otf ≈ ref rtol = 1.0e-12
        # NOTE: The symmetric swap `(GridIdx(k), qy_b)` + `(CubicInterp,
        # PchipInterp)` + PreCompute would reduce to a 1-D Pchip problem, which
        # the dedicated `pchip_interp!` supports but the generic `interp!` ND
        # validator rejects even at N=1. That's an orthogonal limitation in
        # `_validate_nd_coeffs`, not part of the batch auto-promotion fix, and
        # the scalar test at line 1547 avoids it for the same reason. See
        # follow-up ticket if/when generic N=1 Pchip+PreCompute gets plumbed
        # through the dedicated 1-D path.

        # Case (c): differentiator — nonzero deriv disables the
        # `_all_eval_value(deriv)` gate, so method stays `(CubicInterp,
        # PchipInterp)`, falls through to expand-and-recurse, and PreCompute +
        # Pchip ND must still reject (mirrors the scalar reject check at
        # line 1554-1560).
        out_reject = zeros(7)
        @test_throws ArgumentError interp!(
            out_reject, (x, y), data_2d, (qx_b, GridIdx(8));
            method = (CubicInterp(), PchipInterp()),
            deriv = (EvalValue(), DerivOp(1)),
            coeffs = PreCompute(),
        )
    end

    # ========================================
    # Phase 5b: NoInterp + Hermite OnTheFly with cell-local windowing
    # ========================================
    # The Phase 5b path is exercised when `_eval_nointerp` is called on an interpolant
    # whose method tuple contains BOTH NoInterp axes AND at least one local-Hermite
    # axis (PCHIP/Cardinal/Akima). The expected behavior:
    #   1. NoInterp axes are pre-sliced via GridIdx (existing behavior).
    #   2. The remaining real axes are searched for the cell once, windowed to the
    #      cell-local stencil, sliced again, then evaluated by `_collapse_dims`.
    #   3. Result must equal the equivalent call where the NoInterp axis is replaced
    #      by a real axis at the corresponding grid point.
    @testset "Phase 5b: NoInterp × Hermite OnTheFly windowed" begin
        x = collect(range(0.0, 2π, 30))
        y = collect(range(-1.0, 1.0, 25))
        z = collect(range(0.0, 1.0, 12))
        data_3d = [sin(2xi) * exp(-yj^2) * (1 + zk) for xi in x, yj in y, zk in z]

        # 3D interpolant with NoInterp on the middle axis + Hermite on the others.
        for methods in (
                (CardinalInterp(), NoInterp(), CardinalInterp()),
                (PchipInterp(), NoInterp(), AkimaInterp()),
                (CardinalInterp(), NoInterp(), CubicInterp()),
                (CubicInterp(), NoInterp(), PchipInterp()),
            )
            itp = interp((x, y, z), data_3d; method = methods, coeffs = OnTheFly())
            for j in (1, 7, 13, 25)            # iterate over y indices, including boundaries
                for (qx, qz) in ((1.0, 0.4), (3.5, 0.7), (5.5, 0.1))
                    # GridIdx route — exercises _eval_nointerp Phase 5b path
                    val_idx = itp((qx, GridIdx(j), qz))

                    # Reference: build a 2D itp on the (x, z) plane sliced at y[j],
                    # using only the real-axis methods. This avoids any NoInterp logic.
                    itp_ref = interp(
                        (x, z), data_3d[:, j, :];
                        method = (methods[1], methods[3]),
                        coeffs = OnTheFly(),
                    )
                    val_ref = itp_ref((qx, qz))
                    @test val_idx ≈ val_ref atol = 1.0e-12 rtol = 1.0e-12
                end
            end
        end

        # Hint persistence across Phase 5b path: pre-search must update real-axis hints.
        let methods = (CardinalInterp(), NoInterp(), CardinalInterp())
            itp = interp((x, y, z), data_3d; method = methods, coeffs = OnTheFly())
            hint = (Ref(0), Ref(0), Ref(0))
            itp((1.5, GridIdx(10), 0.4); hint = hint)
            # Real axes should have been updated to absolute interval indices.
            @test 1 <= hint[1][] <= length(x) - 1
            @test 1 <= hint[3][] <= length(z) - 1
            # NoInterp axis hint is set to the GridIdx index (existing behavior).
            @test hint[2][] == 10
        end
    end
end
