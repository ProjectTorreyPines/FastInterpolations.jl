# ========================================
# Deriv3 Tests for FastInterpolations.jl
# ========================================
# Phase 1: Core Types and Dispatch

using Test
using FastInterpolations
using FastInterpolations: EvalDeriv3, AbstractEvalOp

@testset "Deriv3 - Phase 1: Core Types" begin

    @testset "EvalDeriv3 type exists and is subtype of AbstractEvalOp" begin
        @test isdefined(FastInterpolations, :EvalDeriv3)
        @test EvalDeriv3 <: AbstractEvalOp
        @test EvalDeriv3() isa AbstractEvalOp
    end

    @testset "@_dispatch_deriv handles deriv=3" begin
        result = FastInterpolations.@_dispatch_deriv 3 => op begin
            op
        end
        @test result isa EvalDeriv3
    end

    @testset "@_dispatch_deriv throws for deriv=4" begin
        @test_throws ArgumentError begin
            FastInterpolations.@_dispatch_deriv 4 => op begin
                op
            end
        end
    end

    @testset "@_dispatch_deriv throws for deriv=-1" begin
        @test_throws ArgumentError begin
            FastInterpolations.@_dispatch_deriv -1 => op begin
                op
            end
        end
    end
end

@testset "Deriv3 - Phase 2: Kernel Implementations" begin

    @testset "Cubic kernel - S'''(x) = (zR - zL) / h" begin
        zL, zR = 1.0, 3.0
        yL, yR = 0.0, 1.0
        h = 0.5
        inv_h = 1.0 / h
        dL, dR = 0.2, 0.3

        result = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, yL, yR, h, inv_h, dL, dR
        )

        expected = (zR - zL) / h
        @test result ≈ expected
    end

    @testset "Cubic kernel - result is constant within interval" begin
        zL, zR = 2.0, 5.0
        h, inv_h = 0.1, 10.0

        result1 = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0, 0.0, h, inv_h, 0.02, 0.08
        )
        result2 = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0, 0.0, h, inv_h, 0.05, 0.05
        )
        result3 = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0, 0.0, h, inv_h, 0.09, 0.01
        )

        @test result1 ≈ result2 ≈ result3
    end

    @testset "Cubic kernel - Float32 support" begin
        zL, zR = 1.0f0, 2.0f0
        h, inv_h = 0.5f0, 2.0f0

        result = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0f0, 0.0f0, h, inv_h, 0.1f0, 0.4f0
        )

        @test result isa Float32
        @test result ≈ 2.0f0
    end

    @testset "Linear kernel - returns zero" begin
        yL, yR = 1.0, 5.0
        h, dL = 0.5, 0.2

        result = FastInterpolations._linear_kernel(EvalDeriv3(), yL, yR, h, dL)
        @test result === zero(Float64)
    end

    @testset "Linear kernel - Float32 returns Float32 zero" begin
        result = FastInterpolations._linear_kernel(
            EvalDeriv3(), 1.0f0, 2.0f0, 0.5f0, 0.1f0
        )
        @test result === zero(Float32)
    end

    @testset "Quadratic kernel - returns zero" begin
        a, d, y = 1.0, 2.0, 3.0
        dL = 0.5

        result = FastInterpolations._quadratic_kernel(EvalDeriv3(), a, d, y, dL)
        @test result === zero(Float64)
    end

    @testset "Quadratic kernel - Float32 returns Float32 zero" begin
        result = FastInterpolations._quadratic_kernel(
            EvalDeriv3(), 1.0f0, 2.0f0, 3.0f0, 0.5f0
        )
        @test result === zero(Float32)
    end

    @testset "Constant kernel - returns zero" begin
        yL, yR = 5.0, 5.0
        h, dL = 0.5, 0.2

        result = FastInterpolations._constant_kernel(
            EvalDeriv3(), yL, yR, h, dL, Val(:left)
        )
        @test result === zero(Float64)
    end

    @testset "Constant kernel - Float32 returns Float32 zero" begin
        result = FastInterpolations._constant_kernel(
            EvalDeriv3(), 5.0f0, 5.0f0, 0.5f0, 0.2f0, Val(:left)
        )
        @test result === zero(Float32)
    end
end
