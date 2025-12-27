# ========================================
# Derivative Tests for FastInterpolations.jl
# ========================================
# Phase 1: Foundation tests for EvalOp types and @_dispatch_order macro
# Phase 2+: Kernel functions, cubic/linear derivative evaluation

using Test
using FastInterpolations

# Import internal types/macros for testing
using FastInterpolations: @_dispatch_order, _linear_kernel, _cubic_kernel

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

    # ========================================
    # Phase 2: Linear Kernels
    # ========================================
    @testset "Linear kernels" begin
        # Test case: L(x) = 1 + 2x on [0, 1]
        # y0 = L(0) = 1, y1 = L(1) = 3
        # L(0.5) = 2, L'(x) = 2, L''(x) = 0
        h, y0, y1 = 1.0, 1.0, 3.0
        dt1 = 0.5  # x = 0.5

        @testset "EvalValue" begin
            @test _linear_kernel(EvalValue(), y0, y1, h, dt1) ≈ 2.0
            # Edge cases
            @test _linear_kernel(EvalValue(), y0, y1, h, 0.0) ≈ y0  # left boundary
            @test _linear_kernel(EvalValue(), y0, y1, h, h) ≈ y1    # right boundary
        end

        @testset "EvalDeriv1" begin
            @test _linear_kernel(EvalDeriv1(), y0, y1, h, dt1) ≈ 2.0
            # Slope is constant everywhere
            @test _linear_kernel(EvalDeriv1(), y0, y1, h, 0.0) ≈ 2.0
            @test _linear_kernel(EvalDeriv1(), y0, y1, h, 0.9) ≈ 2.0
        end

        @testset "EvalDeriv2" begin
            @test _linear_kernel(EvalDeriv2(), y0, y1, h, dt1) ≈ 0.0
            # Second derivative is always zero
            @test _linear_kernel(EvalDeriv2(), y0, y1, h, 0.0) === 0.0
            @test _linear_kernel(EvalDeriv2(), y0, y1, h, h) === 0.0
        end

        @testset "Type stability" begin
            @test @inferred(_linear_kernel(EvalValue(), y0, y1, h, dt1)) isa Float64
            @test @inferred(_linear_kernel(EvalDeriv1(), y0, y1, h, dt1)) isa Float64
            @test @inferred(_linear_kernel(EvalDeriv2(), y0, y1, h, dt1)) isa Float64

            # Float32 preservation
            y0_f32, y1_f32, h_f32, dt1_f32 = 1.0f0, 3.0f0, 1.0f0, 0.5f0
            @test @inferred(_linear_kernel(EvalValue(), y0_f32, y1_f32, h_f32, dt1_f32)) isa Float32
        end

        @testset "Different slopes" begin
            # Negative slope: L(x) = 5 - 3x on [0, 2]
            @test _linear_kernel(EvalDeriv1(), 5.0, -1.0, 2.0, 1.0) ≈ -3.0
            # Zero slope: constant function
            @test _linear_kernel(EvalDeriv1(), 4.0, 4.0, 2.0, 1.0) ≈ 0.0
        end
    end

    # ========================================
    # Phase 2: Cubic Kernels
    # ========================================
    @testset "Cubic kernels" begin
        # Test with known quadratic: f(x) = x² on [0, 1]
        # f(0) = 0, f(1) = 1, f'(x) = 2x, f''(x) = 2
        # For natural spline on x² with enough points, z values approximate f''
        h_i = 1.0
        y_i, y_ip1 = 0.0, 1.0
        z_i, z_ip1 = 2.0, 2.0  # f''(x) = 2 (constant for quadratic)

        @testset "EvalValue - quadratic exactness" begin
            # At x = 0.5: f(0.5) = 0.25
            dt1, dt2 = 0.5, 0.5
            result = _cubic_kernel(EvalValue(), z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
            @test result ≈ 0.25 atol=1e-10

            # At boundaries
            @test _cubic_kernel(EvalValue(), z_i, z_ip1, y_i, y_ip1, h_i, 0.0, 1.0) ≈ y_i atol=1e-10
            @test _cubic_kernel(EvalValue(), z_i, z_ip1, y_i, y_ip1, h_i, 1.0, 0.0) ≈ y_ip1 atol=1e-10
        end

        @testset "EvalDeriv1 - derivative of quadratic" begin
            # f'(x) = 2x, so f'(0.5) = 1.0
            dt1, dt2 = 0.5, 0.5
            result = _cubic_kernel(EvalDeriv1(), z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)
            @test result ≈ 1.0 atol=1e-10

            # f'(0) = 0
            @test _cubic_kernel(EvalDeriv1(), z_i, z_ip1, y_i, y_ip1, h_i, 0.0, 1.0) ≈ 0.0 atol=1e-10
            # f'(1) = 2
            @test _cubic_kernel(EvalDeriv1(), z_i, z_ip1, y_i, y_ip1, h_i, 1.0, 0.0) ≈ 2.0 atol=1e-10
        end

        @testset "EvalDeriv2 - second derivative of quadratic" begin
            # f''(x) = 2 everywhere
            @test _cubic_kernel(EvalDeriv2(), z_i, z_ip1, y_i, y_ip1, h_i, 0.5, 0.5) ≈ 2.0 atol=1e-10
            @test _cubic_kernel(EvalDeriv2(), z_i, z_ip1, y_i, y_ip1, h_i, 0.0, 1.0) ≈ 2.0 atol=1e-10
            @test _cubic_kernel(EvalDeriv2(), z_i, z_ip1, y_i, y_ip1, h_i, 1.0, 0.0) ≈ 2.0 atol=1e-10
        end

        @testset "Type stability" begin
            dt1, dt2 = 0.5, 0.5
            @test @inferred(_cubic_kernel(EvalValue(), z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)) isa Float64
            @test @inferred(_cubic_kernel(EvalDeriv1(), z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)) isa Float64
            @test @inferred(_cubic_kernel(EvalDeriv2(), z_i, z_ip1, y_i, y_ip1, h_i, dt1, dt2)) isa Float64

            # Float32 preservation
            args_f32 = (2.0f0, 2.0f0, 0.0f0, 1.0f0, 1.0f0, 0.5f0, 0.5f0)
            @test @inferred(_cubic_kernel(EvalValue(), args_f32...)) isa Float32
        end

        @testset "Varying z values (non-constant curvature)" begin
            # Test with z_i ≠ z_ip1 (linear interpolation of z)
            z_left, z_right = 0.0, 4.0
            dt1, dt2 = 0.5, 0.5

            # f''(0.5) should be average of z values
            result = _cubic_kernel(EvalDeriv2(), z_left, z_right, y_i, y_ip1, h_i, dt1, dt2)
            @test result ≈ 2.0 atol=1e-10  # (0*0.5 + 4*0.5) / 1 = 2
        end

        @testset "Cubic polynomial exactness" begin
            # For a true cubic f(x) = x³ on [0, 1]:
            # f(0) = 0, f(1) = 1
            # f''(x) = 6x, so z_i = f''(0) = 0, z_ip1 = f''(1) = 6
            y0_cubic, y1_cubic = 0.0, 1.0
            z0_cubic, z1_cubic = 0.0, 6.0
            h = 1.0

            # At x = 0.5: f(0.5) = 0.125, f'(0.5) = 0.75, f''(0.5) = 3
            dt1, dt2 = 0.5, 0.5
            @test _cubic_kernel(EvalValue(), z0_cubic, z1_cubic, y0_cubic, y1_cubic, h, dt1, dt2) ≈ 0.125 atol=1e-10
            @test _cubic_kernel(EvalDeriv1(), z0_cubic, z1_cubic, y0_cubic, y1_cubic, h, dt1, dt2) ≈ 0.75 atol=1e-10
            @test _cubic_kernel(EvalDeriv2(), z0_cubic, z1_cubic, y0_cubic, y1_cubic, h, dt1, dt2) ≈ 3.0 atol=1e-10
        end
    end

end # @testset "Derivatives"
