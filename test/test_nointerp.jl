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
        # GridIdx(26) on a 25-point grid → BoundsError from _slice_grididx
        @test_throws BoundsError interp(
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
end
