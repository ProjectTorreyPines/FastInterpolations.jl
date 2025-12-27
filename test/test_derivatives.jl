# ========================================
# Derivative Tests for FastInterpolations.jl
# ========================================
# Phase 1: Foundation tests for EvalOp types and @_dispatch_order macro
# Phase 2+: Kernel functions, cubic/linear derivative evaluation

using Test
using FastInterpolations

# Import internal types/macros for testing
using FastInterpolations: @_dispatch_order

@testset "Derivatives" begin

    # ========================================
    # Phase 1: EvalOp Types
    # ========================================
    @testset "EvalOp types" begin
        @test EvalValue() isa AbstractEvalOp
        @test EvalDeriv1() isa AbstractEvalOp
        @test EvalDeriv2() isa AbstractEvalOp

        # Ensure they are distinct types
        @test typeof(EvalValue()) !== typeof(EvalDeriv1())
        @test typeof(EvalDeriv1()) !== typeof(EvalDeriv2())
        @test typeof(EvalValue()) !== typeof(EvalDeriv2())

        # Singleton check (no fields)
        @test fieldcount(EvalValue) == 0
        @test fieldcount(EvalDeriv1) == 0
        @test fieldcount(EvalDeriv2) == 0
    end

    # ========================================
    # Phase 1: @_dispatch_order Macro
    # ========================================
    @testset "@_dispatch_order macro" begin
        # order=0 → EvalValue
        result0 = @_dispatch_order 0 op begin
            typeof(op)
        end
        @test result0 === EvalValue

        # order=1 → EvalDeriv1
        result1 = @_dispatch_order 1 op begin
            typeof(op)
        end
        @test result1 === EvalDeriv1

        # order=2 → EvalDeriv2
        result2 = @_dispatch_order 2 op begin
            typeof(op)
        end
        @test result2 === EvalDeriv2

        # Invalid order throws ArgumentError
        @test_throws ArgumentError @_dispatch_order 3 op begin
            nothing
        end
        @test_throws ArgumentError @_dispatch_order -1 op begin
            nothing
        end
    end

    @testset "@_dispatch_order with runtime variable" begin
        # Test that macro works with runtime-determined order
        for order in 0:2
            result = @_dispatch_order order op begin
                op
            end
            if order == 0
                @test result isa EvalValue
            elseif order == 1
                @test result isa EvalDeriv1
            else
                @test result isa EvalDeriv2
            end
        end
    end

    @testset "@_dispatch_order type stability" begin
        # The dispatched function should maintain type stability
        function test_dispatch(order::Int)
            @_dispatch_order order op begin
                # Return something that depends on op type
                op isa EvalValue ? 1.0 :
                op isa EvalDeriv1 ? 2.0 : 3.0
            end
        end

        @test test_dispatch(0) === 1.0
        @test test_dispatch(1) === 2.0
        @test test_dispatch(2) === 3.0

        # Type inference should work
        @test @inferred(test_dispatch(0)) === 1.0
    end

end # @testset "Derivatives"
