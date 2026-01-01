# Tests for quadratic (C1 piecewise quadratic) spline interpolation
#
# Phase 1: BC Tags (Left/Right types)
# This file follows TDD: tests are written BEFORE implementation.

# ============================================================================
# Group 1: BC Type Tests (Phase 1)
# ============================================================================
@testset "Quadratic Interpolation - BC Types" begin

    @testset "Left BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Left(Deriv1(0.5))
            @test bc1 isa Left{Float64, Deriv1{Float64}}

            bc2 = Left(Deriv2(1.0))
            @test bc2 isa Left{Float64, Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Left(Deriv2(1.0f0))
            @test bc isa Left{Float32, Deriv2{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Left(Deriv1(1))
            @test bc isa Left{Float64, Deriv1{Float64}}
        end

        @testset "accessor (inner BC value)" begin
            bc = Left(Deriv1(0.5))
            @test bc.bc.val == 0.5

            bc2 = Left(Deriv2(2.0))
            @test bc2.bc.val == 2.0
        end

        @testset "type stability" begin
            @test @inferred(Left(Deriv1(0.0))) isa Left
            @test @inferred(Left(Deriv2(1.0))) isa Left
        end

        @testset "subtype relationship" begin
            @test Left{Float64, Deriv1{Float64}} <: AbstractBC{Float64}
            @test Left{Float64, Deriv2{Float64}} <: AbstractBC{Float64}
            @test Left{Float32, Deriv1{Float32}} <: AbstractBC{Float32}
        end
    end

    @testset "Right BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Right(Deriv1(-0.5))
            @test bc1 isa Right{Float64, Deriv1{Float64}}

            bc2 = Right(Deriv2(0.0))
            @test bc2 isa Right{Float64, Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Right(Deriv1(1.0f0))
            @test bc isa Right{Float32, Deriv1{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Right(Deriv2(0))
            @test bc isa Right{Float64, Deriv2{Float64}}
        end

        @testset "accessor (inner BC value)" begin
            bc = Right(Deriv1(-0.5))
            @test bc.bc.val == -0.5

            bc2 = Right(Deriv2(1.0))
            @test bc2.bc.val == 1.0
        end

        @testset "type stability" begin
            @test @inferred(Right(Deriv1(0.0))) isa Right
            @test @inferred(Right(Deriv2(1.0))) isa Right
        end

        @testset "subtype relationship" begin
            @test Right{Float64, Deriv1{Float64}} <: AbstractBC{Float64}
            @test Right{Float64, Deriv2{Float64}} <: AbstractBC{Float64}
            @test Right{Float32, Deriv2{Float32}} <: AbstractBC{Float32}
        end
    end

    @testset "Left/Right distinctness" begin
        # Left and Right should be distinct types
        @test Left(Deriv1(0.0)) isa Left
        @test Right(Deriv1(0.0)) isa Right
        @test !(Left(Deriv1(0.0)) isa Right)
        @test !(Right(Deriv1(0.0)) isa Left)
    end

end
