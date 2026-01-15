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
