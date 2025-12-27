# ========================================
# Derivative Tests for FastInterpolations.jl
# ========================================
# Phase 1: Foundation tests for EvalOp types and @_dispatch_order macro
# Phase 2+: Kernel functions, cubic/linear derivative evaluation

using Test
using FastInterpolations

# Import internal types/macros for testing
using FastInterpolations: @_dispatch_order, _linear_kernel, _cubic_kernel
using FastInterpolations: _eval_cubic_at_point, _eval_cubic_with_extrap, _get_cubic_cache, _solve_system!

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

    # ========================================
    # Phase 3: Cubic Internal Wrappers
    # ========================================
    @testset "Cubic internal functions with op" begin
        # Test with quadratic f(x) = x² on [0, 2] with step 0.5
        x = collect(0.0:0.5:2.0)
        y = x .^ 2  # [0, 0.25, 1, 2.25, 4]

        # Use D2 BC with f''(x) = 2 for exact quadratic representation
        cache = _get_cubic_cache(x, BCPair(D2(2.0), D2(2.0)))
        z = _solve_system!(cache, y, cache.bc_data)

        @testset "_eval_cubic_at_point with op" begin
            # Value at midpoint x=1.0: f(1) = 1.0
            val = _eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalValue())
            @test val ≈ 1.0 atol=1e-10

            # First derivative at x=1.0: f'(x) = 2x, so f'(1) = 2.0
            deriv1 = _eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalDeriv1())
            @test deriv1 ≈ 2.0 atol=0.1  # Spline approximation

            # Second derivative: f''(x) = 2.0
            deriv2 = _eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalDeriv2())
            @test deriv2 ≈ 2.0 atol=0.1
        end

        @testset "_eval_cubic_at_point backward compatibility" begin
            # Without op parameter should still work (returns value)
            val_old = _eval_cubic_at_point(x, y, cache.h, z, 1.0)
            val_new = _eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalValue())
            @test val_old ≈ val_new atol=1e-14
        end

        @testset "_eval_cubic_with_extrap with op" begin
            # Test constant extrapolation with derivatives
            # Outside left boundary: should return 0 for derivatives
            left_val = _eval_cubic_with_extrap(x, y, cache.h, z, -0.5, Val(:constant), EvalValue())
            @test left_val ≈ y[1]  # y[1] = 0.0

            left_deriv1 = _eval_cubic_with_extrap(x, y, cache.h, z, -0.5, Val(:constant), EvalDeriv1())
            @test left_deriv1 === 0.0  # Constant extrap → derivative = 0

            left_deriv2 = _eval_cubic_with_extrap(x, y, cache.h, z, -0.5, Val(:constant), EvalDeriv2())
            @test left_deriv2 === 0.0

            # Inside domain: should use normal evaluation
            mid_deriv1 = _eval_cubic_with_extrap(x, y, cache.h, z, 1.0, Val(:constant), EvalDeriv1())
            @test mid_deriv1 ≈ 2.0 atol=0.1

            # Extension extrapolation: use boundary polynomial
            ext_deriv1 = _eval_cubic_with_extrap(x, y, cache.h, z, -0.5, Val(:extension), EvalDeriv1())
            @test ext_deriv1 isa Float64  # Should not throw
        end

        @testset "Type stability with op" begin
            @test @inferred(_eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalValue())) isa Float64
            @test @inferred(_eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalDeriv1())) isa Float64
            @test @inferred(_eval_cubic_at_point(x, y, cache.h, z, 1.0, EvalDeriv2())) isa Float64
        end

        @testset "Derivative at different points" begin
            # f'(0) = 0, f'(0.5) = 1, f'(1) = 2, f'(1.5) = 3, f'(2) = 4
            for (xi, expected_deriv) in [(0.0, 0.0), (0.5, 1.0), (1.5, 3.0), (2.0, 4.0)]
                deriv = _eval_cubic_at_point(x, y, cache.h, z, xi, EvalDeriv1())
                @test deriv ≈ expected_deriv atol=0.15
            end
        end
    end

end # @testset "Derivatives"
