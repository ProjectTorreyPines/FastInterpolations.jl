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

# Julia version-aware threshold (1.12+ has improved allocation tracking)
const DERIV_ALLOC_THRESHOLD = VERSION >= v"1.12.0-DEV" ? 0 : 64

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

    # ========================================
    # Phase 4: Cubic Public API with order parameter
    # ========================================
    @testset "Cubic public API with order" begin

        @testset "Polynomial exactness" begin
            # Quadratic f(x) = x² with exact D2 BC
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            bc = BCPair(D2(2.0), D2(2.0))  # f''(x) = 2
            xi = 0.5

            # Value: f(0.5) = 0.25
            @test cubic_interp(x, y, xi; bc=bc, order=0) ≈ 0.25 atol=1e-10

            # First derivative: f'(0.5) = 2*0.5 = 1.0
            @test cubic_interp(x, y, xi; bc=bc, order=1) ≈ 1.0 atol=1e-10

            # Second derivative: f''(x) = 2.0
            @test cubic_interp(x, y, xi; bc=bc, order=2) ≈ 2.0 atol=1e-10
        end

        @testset "Cubic polynomial exactness" begin
            # Cubic f(x) = x³ with exact D2 BC
            x = collect(0.0:0.1:1.0)
            y = x .^ 3
            bc = BCPair(D2(0.0), D2(6.0))  # f''(0)=0, f''(1)=6
            xi = 0.5

            # Value: f(0.5) = 0.125
            @test cubic_interp(x, y, xi; bc=bc, order=0) ≈ 0.125 atol=1e-10

            # First derivative: f'(0.5) = 3*(0.5)² = 0.75
            @test cubic_interp(x, y, xi; bc=bc, order=1) ≈ 0.75 atol=1e-10

            # Second derivative: f''(0.5) = 6*0.5 = 3.0
            @test cubic_interp(x, y, xi; bc=bc, order=2) ≈ 3.0 atol=1e-10
        end

        @testset "Backward compatibility (no order arg)" begin
            x = collect(0.0:0.2:1.0)
            y = sin.(x)
            xi = 0.5

            # Without order parameter should work as before
            val_old = cubic_interp(x, y, xi)
            val_new = cubic_interp(x, y, xi; order=0)
            @test val_old ≈ val_new atol=1e-14
        end

        @testset "Vector query with order" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            bc = BCPair(D2(2.0), D2(2.0))
            x_query = [0.25, 0.5, 0.75]

            # Values
            vals = cubic_interp(x, y, x_query; bc=bc, order=0)
            @test vals ≈ x_query .^ 2 atol=1e-10

            # First derivatives: f'(x) = 2x
            derivs = cubic_interp(x, y, x_query; bc=bc, order=1)
            @test derivs ≈ 2.0 .* x_query atol=1e-10

            # Second derivatives: f''(x) = 2
            derivs2 = cubic_interp(x, y, x_query; bc=bc, order=2)
            @test all(d ≈ 2.0 for d in derivs2)
        end

        @testset "Cache-based with order" begin
            x = collect(0.0:0.1:1.0)
            cache = CubicSplineCache(x; bc=BCPair(D2(2.0), D2(2.0)))
            y = x .^ 2
            xi = 0.5

            @test cubic_interp(cache, y, xi; order=0) ≈ 0.25 atol=1e-10
            @test cubic_interp(cache, y, xi; order=1) ≈ 1.0 atol=1e-10
            @test cubic_interp(cache, y, xi; order=2) ≈ 2.0 atol=1e-10
        end

        @testset "Type stability with order" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            bc = BCPair(D2(2.0), D2(2.0))
            xi = 0.5

            @test @inferred(cubic_interp(x, y, xi; bc=bc, order=0)) isa Float64
            @test @inferred(cubic_interp(x, y, xi; bc=bc, order=1)) isa Float64
            @test @inferred(cubic_interp(x, y, xi; bc=bc, order=2)) isa Float64
        end
    end

    @testset "Cubic allocation with order" begin
        x = collect(0.0:0.1:1.0)
        cache = CubicSplineCache(x; bc=BCPair(D2(2.0), D2(2.0)))
        y = x .^ 2
        xi = 0.5

        # Warm-up
        cubic_interp(cache, y, xi; order=0)
        cubic_interp(cache, y, xi; order=1)
        cubic_interp(cache, y, xi; order=2)

        # Check allocations (scalar query should be zero-allocation)
        alloc0 = @allocated cubic_interp(cache, y, xi; order=0)
        alloc1 = @allocated cubic_interp(cache, y, xi; order=1)
        alloc2 = @allocated cubic_interp(cache, y, xi; order=2)

        @test alloc0 == 0
        @test alloc1 == 0
        @test alloc2 == 0
    end

    @testset "Cubic extrapolation with order" begin
        x = collect(0.0:0.25:1.0)
        y = x .^ 2
        bc = BCPair(D2(2.0), D2(2.0))

        @testset "Constant extrapolation" begin
            # Left boundary constant extrap: returns y[1] for value, 0 for derivatives
            @test cubic_interp(x, y, -0.5; bc=bc, extrap=:constant, order=0) ≈ 0.0
            @test cubic_interp(x, y, -0.5; bc=bc, extrap=:constant, order=1) ≈ 0.0
            @test cubic_interp(x, y, -0.5; bc=bc, extrap=:constant, order=2) ≈ 0.0

            # Right boundary
            @test cubic_interp(x, y, 1.5; bc=bc, extrap=:constant, order=0) ≈ 1.0
            @test cubic_interp(x, y, 1.5; bc=bc, extrap=:constant, order=1) ≈ 0.0
            @test cubic_interp(x, y, 1.5; bc=bc, extrap=:constant, order=2) ≈ 0.0
        end

        @testset "Extension extrapolation" begin
            # Extension: continue boundary polynomial
            # For x², extension should give approximately correct derivatives
            val = cubic_interp(x, y, 1.5; bc=bc, extrap=:extension, order=0)
            @test val ≈ 2.25 atol=0.1  # (1.5)² ≈ 2.25

            deriv1 = cubic_interp(x, y, 1.5; bc=bc, extrap=:extension, order=1)
            @test deriv1 ≈ 3.0 atol=0.2  # 2*1.5 ≈ 3.0

            deriv2 = cubic_interp(x, y, 1.5; bc=bc, extrap=:extension, order=2)
            @test deriv2 ≈ 2.0 atol=0.1  # f''(x) = 2
        end
    end

    @testset "CubicInterpolant derivative methods" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2
        bc = BCPair(D2(2.0), D2(2.0))
        itp = cubic_interp(x, y; bc=bc)

        @testset "derivative scalar" begin
            @test derivative(itp, 0.5) ≈ 1.0 atol=1e-10
            @test derivative(itp, 0.0) ≈ 0.0 atol=1e-10
            @test derivative(itp, 1.0) ≈ 2.0 atol=1e-10
        end

        @testset "derivative2 scalar" begin
            @test derivative2(itp, 0.5) ≈ 2.0 atol=1e-10
            @test derivative2(itp, 0.0) ≈ 2.0 atol=1e-10
            @test derivative2(itp, 1.0) ≈ 2.0 atol=1e-10
        end

        @testset "derivative vector" begin
            x_query = [0.25, 0.5, 0.75]
            derivs = derivative(itp, x_query)
            @test derivs ≈ 2.0 .* x_query atol=1e-10
        end

        @testset "derivative2 vector" begin
            x_query = [0.25, 0.5, 0.75]
            derivs2 = derivative2(itp, x_query)
            @test all(d ≈ 2.0 for d in derivs2)
        end

        @testset "Derivative allocation" begin
            # Warm-up
            derivative(itp, 0.5)
            derivative2(itp, 0.5)

            alloc1 = @allocated derivative(itp, 0.5)
            alloc2 = @allocated derivative2(itp, 0.5)

            @test alloc1 == 0
            @test alloc2 == 0
        end
    end

    # ========================================
    # Phase 4+: Rigorous Allocation Tests
    # ========================================
    # Following test_allocation.jl patterns for maximum rigor

    @testset "Rigorous derivative allocation tests" begin
        # Function-wrapped tests for type stability
        function test_derivative_alloc(itp, xi::T) where {T}
            derivative(itp, xi)
        end

        function test_derivative2_alloc(itp, xi::T) where {T}
            derivative2(itp, xi)
        end

        function test_order_alloc(cache, y, xi, order::Int)
            cubic_interp(cache, y, xi; order=order)
        end

        @testset "Function-wrapped CubicInterpolant derivatives" begin
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Multiple warmup iterations
            for _ in 1:5
                test_derivative_alloc(itp, 0.5)
                test_derivative2_alloc(itp, 0.5)
            end

            # Test at multiple query points
            for xi in [0.1, 0.25, 0.5, 0.75, 0.9]
                alloc1 = @allocated test_derivative_alloc(itp, xi)
                alloc2 = @allocated test_derivative2_alloc(itp, xi)

                @test alloc1 <= DERIV_ALLOC_THRESHOLD
                @test alloc2 <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "Function-wrapped cubic_interp with order" begin
            x = collect(range(0.0, 1.0, 51))
            cache = CubicSplineCache(x)
            y = x .^ 3

            # Multiple warmup
            for _ in 1:5
                test_order_alloc(cache, y, 0.5, 0)
                test_order_alloc(cache, y, 0.5, 1)
                test_order_alloc(cache, y, 0.5, 2)
            end

            # All orders should be zero-allocation
            for order in 0:2
                for xi in [0.25, 0.5, 0.75]
                    allocs = @allocated test_order_alloc(cache, y, xi, order)
                    @test allocs <= DERIV_ALLOC_THRESHOLD
                end
            end
        end

        @testset "Derivative stress test - repeated calls" begin
            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            itp = cubic_interp(x, y)

            # Warmup
            for _ in 1:10
                derivative(itp, 0.5)
                derivative2(itp, 0.5)
            end

            # 100 repeated calls should all be zero-allocation
            total_alloc1 = 0
            total_alloc2 = 0
            for _ in 1:100
                total_alloc1 += @allocated derivative(itp, 0.5)
                total_alloc2 += @allocated derivative2(itp, 0.5)
            end

            @test total_alloc1 <= DERIV_ALLOC_THRESHOLD * 100
            @test total_alloc2 <= DERIV_ALLOC_THRESHOLD * 100
        end

        @testset "Derivative with different query points" begin
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Query at many different points - should all be zero-allocation
            query_points = range(0.01, 0.99, 20)

            # Warmup at all points
            for xi in query_points
                derivative(itp, xi)
                derivative2(itp, xi)
            end

            # All should be zero-allocation
            for xi in query_points
                alloc1 = @allocated derivative(itp, xi)
                alloc2 = @allocated derivative2(itp, xi)
                @test alloc1 <= DERIV_ALLOC_THRESHOLD
                @test alloc2 <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "Float32 derivative allocation" begin
            x = Float32.(collect(range(0.0f0, 1.0f0, 51)))
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Warmup
            for _ in 1:5
                derivative(itp, 0.5f0)
                derivative2(itp, 0.5f0)
            end

            alloc1 = @allocated derivative(itp, 0.5f0)
            alloc2 = @allocated derivative2(itp, 0.5f0)

            @test alloc1 <= DERIV_ALLOC_THRESHOLD
            @test alloc2 <= DERIV_ALLOC_THRESHOLD
        end

        @testset "Derivative with different BCs allocation" begin
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2

            bc_types = [
                NaturalBC(),
                ClampedBC(),
                BCPair(D1(1.0), D1(1.0)),
                BCPair(D2(2.0), D2(0.0)),
            ]

            for bc in bc_types
                itp = cubic_interp(x, y; bc=bc)

                # Warmup
                derivative(itp, 0.5)
                derivative2(itp, 0.5)
                derivative(itp, 0.5)
                derivative2(itp, 0.5)

                alloc1 = @allocated derivative(itp, 0.5)
                alloc2 = @allocated derivative2(itp, 0.5)

                @test alloc1 <= DERIV_ALLOC_THRESHOLD
                @test alloc2 <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "Derivative with extrapolation modes allocation" begin
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2

            for extrap in [:none, :constant, :extension]
                itp = cubic_interp(x, y; extrap=extrap)

                # Warmup
                derivative(itp, 0.5)
                derivative2(itp, 0.5)
                derivative(itp, 0.5)
                derivative2(itp, 0.5)

                alloc1 = @allocated derivative(itp, 0.5)
                alloc2 = @allocated derivative2(itp, 0.5)

                @test alloc1 <= DERIV_ALLOC_THRESHOLD
                @test alloc2 <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "Periodic BC derivative allocation" begin
            x = collect(range(0.0, 2π, 101))
            y = sin.(x)
            y[end] = y[1]  # Ensure periodic
            itp = cubic_interp(x, y; bc=PeriodicBC())

            # Warmup
            for _ in 1:5
                derivative(itp, 1.0)
                derivative2(itp, 1.0)
            end

            alloc1 = @allocated derivative(itp, 1.0)
            alloc2 = @allocated derivative2(itp, 1.0)

            @test alloc1 <= DERIV_ALLOC_THRESHOLD
            @test alloc2 <= DERIV_ALLOC_THRESHOLD

            # Query outside domain (wraps)
            alloc1_wrap = @allocated derivative(itp, 7.0)
            alloc2_wrap = @allocated derivative2(itp, 7.0)

            @test alloc1_wrap <= DERIV_ALLOC_THRESHOLD
            @test alloc2_wrap <= DERIV_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Phase 5: Linear API with order parameter
    # ========================================
    @testset "Linear public API with order" begin

        @testset "Constant slope segments" begin
            # Two segments with different slopes
            x = [0.0, 1.0, 3.0]
            y = [0.0, 2.0, 4.0]  # slopes: 2.0 (first), 1.0 (second)

            # Values
            @test linear_interp(x, y, 0.5; order=0) ≈ 1.0  # midpoint first segment
            @test linear_interp(x, y, 2.0; order=0) ≈ 3.0  # midpoint second segment

            # First derivatives (constant within segment)
            @test linear_interp(x, y, 0.5; order=1) ≈ 2.0  # first segment slope
            @test linear_interp(x, y, 2.0; order=1) ≈ 1.0  # second segment slope
            @test linear_interp(x, y, 0.0; order=1) ≈ 2.0  # at left boundary
            @test linear_interp(x, y, 1.0; order=1) ≈ 1.0  # at knot (use right segment)

            # Second derivatives (always zero for linear)
            @test linear_interp(x, y, 0.5; order=2) ≈ 0.0
            @test linear_interp(x, y, 2.0; order=2) ≈ 0.0
        end

        @testset "Backward compatibility (no order arg)" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 1.0, 4.0]
            xi = 0.5

            val_old = linear_interp(x, y, xi)
            val_new = linear_interp(x, y, xi; order=0)
            @test val_old ≈ val_new atol=1e-14
        end

        @testset "Vector query with order" begin
            x = [0.0, 1.0, 3.0]
            y = [0.0, 2.0, 4.0]  # slopes: 2.0, 1.0
            x_query = [0.25, 0.75, 1.5, 2.5]

            # Values
            vals = linear_interp(x, y, x_query; order=0)
            @test vals[1] ≈ 0.5   # 0 + 2.0*0.25
            @test vals[2] ≈ 1.5   # 0 + 2.0*0.75
            @test vals[3] ≈ 2.5   # 2 + 1.0*0.5
            @test vals[4] ≈ 3.5   # 2 + 1.0*1.5

            # First derivatives
            derivs = linear_interp(x, y, x_query; order=1)
            @test derivs[1] ≈ 2.0  # first segment
            @test derivs[2] ≈ 2.0  # first segment
            @test derivs[3] ≈ 1.0  # second segment
            @test derivs[4] ≈ 1.0  # second segment

            # Second derivatives (all zero)
            derivs2 = linear_interp(x, y, x_query; order=2)
            @test all(d ≈ 0.0 for d in derivs2)
        end

        @testset "In-place with order" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 2.0, 6.0]  # slopes: 2.0, 4.0
            x_query = [0.5, 1.5]
            output = zeros(2)

            # Value
            linear_interp!(output, x, y, x_query; order=0)
            @test output[1] ≈ 1.0
            @test output[2] ≈ 4.0

            # First derivative
            linear_interp!(output, x, y, x_query; order=1)
            @test output[1] ≈ 2.0
            @test output[2] ≈ 4.0

            # Second derivative
            linear_interp!(output, x, y, x_query; order=2)
            @test all(o ≈ 0.0 for o in output)
        end

        @testset "Type stability with order" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 1.0, 4.0]
            xi = 0.5

            @test @inferred(linear_interp(x, y, xi; order=0)) isa Float64
            @test @inferred(linear_interp(x, y, xi; order=1)) isa Float64
            @test @inferred(linear_interp(x, y, xi; order=2)) isa Float64
        end
    end

    @testset "Linear extrapolation with order" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 2.0, 6.0]  # slopes: 2.0, 4.0

        @testset "Constant extrapolation" begin
            # Left boundary: returns y[1], derivatives = 0
            @test linear_interp(x, y, -0.5; extrap=:constant, order=0) ≈ 0.0
            @test linear_interp(x, y, -0.5; extrap=:constant, order=1) ≈ 0.0
            @test linear_interp(x, y, -0.5; extrap=:constant, order=2) ≈ 0.0

            # Right boundary: returns y[end], derivatives = 0
            @test linear_interp(x, y, 2.5; extrap=:constant, order=0) ≈ 6.0
            @test linear_interp(x, y, 2.5; extrap=:constant, order=1) ≈ 0.0
            @test linear_interp(x, y, 2.5; extrap=:constant, order=2) ≈ 0.0
        end

        @testset "Extension extrapolation" begin
            # Left: extends first segment (slope 2.0)
            @test linear_interp(x, y, -0.5; extrap=:extension, order=0) ≈ -1.0
            @test linear_interp(x, y, -0.5; extrap=:extension, order=1) ≈ 2.0
            @test linear_interp(x, y, -0.5; extrap=:extension, order=2) ≈ 0.0

            # Right: extends last segment (slope 4.0)
            @test linear_interp(x, y, 2.5; extrap=:extension, order=0) ≈ 8.0
            @test linear_interp(x, y, 2.5; extrap=:extension, order=1) ≈ 4.0
            @test linear_interp(x, y, 2.5; extrap=:extension, order=2) ≈ 0.0
        end

        @testset "Wrap extrapolation" begin
            # Domain [0, 2), wrap 2.5 -> 0.5 (first segment)
            @test linear_interp(x, y, 2.5; extrap=:wrap, order=0) ≈ 1.0  # same as 0.5
            @test linear_interp(x, y, 2.5; extrap=:wrap, order=1) ≈ 2.0  # first segment slope
            @test linear_interp(x, y, 2.5; extrap=:wrap, order=2) ≈ 0.0
        end
    end

    @testset "Linear allocation with order" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        xi = 0.5

        # Warm-up
        linear_interp(x, y, xi; order=0)
        linear_interp(x, y, xi; order=1)
        linear_interp(x, y, xi; order=2)

        # Check allocations (scalar query should be zero-allocation)
        alloc0 = @allocated linear_interp(x, y, xi; order=0)
        alloc1 = @allocated linear_interp(x, y, xi; order=1)
        alloc2 = @allocated linear_interp(x, y, xi; order=2)

        @test alloc0 <= DERIV_ALLOC_THRESHOLD
        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD
    end

    @testset "LinearInterpolant derivative methods" begin
        x = [0.0, 1.0, 3.0]
        y = [0.0, 2.0, 4.0]  # slopes: 2.0, 1.0
        itp = linear_interp(x, y)

        @testset "derivative scalar" begin
            @test derivative(itp, 0.5) ≈ 2.0   # first segment
            @test derivative(itp, 2.0) ≈ 1.0   # second segment
            @test derivative(itp, 0.0) ≈ 2.0   # left boundary
        end

        @testset "derivative2 scalar" begin
            # Always zero for linear
            @test derivative2(itp, 0.5) ≈ 0.0
            @test derivative2(itp, 2.0) ≈ 0.0
        end

        @testset "derivative vector" begin
            x_query = [0.25, 0.75, 1.5, 2.5]
            derivs = derivative(itp, x_query)
            @test derivs[1] ≈ 2.0
            @test derivs[2] ≈ 2.0
            @test derivs[3] ≈ 1.0
            @test derivs[4] ≈ 1.0
        end

        @testset "derivative2 vector" begin
            x_query = [0.25, 0.75, 1.5, 2.5]
            derivs2 = derivative2(itp, x_query)
            @test all(d ≈ 0.0 for d in derivs2)
        end

        @testset "Derivative allocation" begin
            # Warm-up
            derivative(itp, 0.5)
            derivative2(itp, 0.5)

            alloc1 = @allocated derivative(itp, 0.5)
            alloc2 = @allocated derivative2(itp, 0.5)

            @test alloc1 == 0
            @test alloc2 == 0
        end
    end

    @testset "Linear Range optimization with order" begin
        # Range should use O(1) path
        x = 0.0:0.1:1.0
        y = collect(x) .^ 2
        xi = 0.55

        @test linear_interp(x, y, xi; order=0) ≈ linear_interp(collect(x), y, xi; order=0)
        @test linear_interp(x, y, xi; order=1) ≈ linear_interp(collect(x), y, xi; order=1)
        @test linear_interp(x, y, xi; order=2) ≈ 0.0
    end

    # ========================================
    # Phase 6: Comprehensive Testing & Polish
    # ========================================

    @testset "Periodic BC derivative continuity" begin
        # Test that derivatives are continuous at the wrap point
        x = collect(range(0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        itp = cubic_interp(x, y; bc=PeriodicBC())

        ε = 1e-6

        @testset "First derivative continuity at boundaries" begin
            # Derivative at left boundary should match derivative at right boundary
            d_left = derivative(itp, ε)
            d_right = derivative(itp, 2π - ε)

            # For sin(x), d/dx = cos(x), so cos(0) ≈ cos(2π) ≈ 1
            @test d_left ≈ d_right atol=1e-4

            # Also test using order parameter
            d_left_order = cubic_interp(x, y, ε; bc=PeriodicBC(), order=1)
            d_right_order = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), order=1)
            @test d_left_order ≈ d_right_order atol=1e-4
        end

        @testset "Second derivative continuity at boundaries" begin
            d2_left = derivative2(itp, ε)
            d2_right = derivative2(itp, 2π - ε)

            # For sin(x), d²/dx² = -sin(x), so -sin(0) ≈ -sin(2π) ≈ 0
            @test d2_left ≈ d2_right atol=1e-4

            # Also test using order parameter
            d2_left_order = cubic_interp(x, y, ε; bc=PeriodicBC(), order=2)
            d2_right_order = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), order=2)
            @test d2_left_order ≈ d2_right_order atol=1e-4
        end

        @testset "Derivative at wrap point" begin
            # Test querying exactly at 0 and 2π (they should be equivalent)
            d_at_zero = derivative(itp, 0.0)
            # Query outside domain wraps to inside
            d_at_2pi_plus = derivative(itp, 2π + ε)

            @test d_at_zero ≈ d_at_2pi_plus atol=1e-4
        end

        @testset "Cosine function derivatives" begin
            # cos(x) has d/dx = -sin(x), d²/dx² = -cos(x)
            y_cos = cos.(x)
            y_cos[end] = y_cos[1]
            itp_cos = cubic_interp(x, y_cos; bc=PeriodicBC())

            # At x = π/2: cos(π/2) = 0, cos'(π/2) = -sin(π/2) = -1, cos''(π/2) = -cos(π/2) = 0
            @test itp_cos(π/2) ≈ 0.0 atol=1e-3
            @test derivative(itp_cos, π/2) ≈ -1.0 atol=1e-2
            @test derivative2(itp_cos, π/2) ≈ 0.0 atol=1e-2

            # At x = π: cos(π) = -1, cos'(π) = 0, cos''(π) = 1
            @test itp_cos(π) ≈ -1.0 atol=1e-3
            @test derivative(itp_cos, π) ≈ 0.0 atol=1e-2
            @test derivative2(itp_cos, π) ≈ 1.0 atol=1e-2
        end
    end

    @testset "Boundary point behavior" begin
        # Test right-continuous behavior at knots

        @testset "Cubic at knot points" begin
            x = collect(0.0:0.25:1.0)
            y = x .^ 2
            bc = BCPair(D2(2.0), D2(2.0))
            itp = cubic_interp(x, y; bc=bc)

            # At interior knot (x=0.5), derivative should be well-defined
            @test derivative(itp, 0.5) ≈ 1.0 atol=1e-10  # f'(0.5) = 2*0.5 = 1

            # At boundaries
            @test derivative(itp, 0.0) ≈ 0.0 atol=1e-10  # f'(0) = 0
            @test derivative(itp, 1.0) ≈ 2.0 atol=1e-10  # f'(1) = 2

            # Second derivative should be constant (=2) for quadratic
            @test derivative2(itp, 0.0) ≈ 2.0 atol=1e-10
            @test derivative2(itp, 0.5) ≈ 2.0 atol=1e-10
            @test derivative2(itp, 1.0) ≈ 2.0 atol=1e-10
        end

        @testset "Linear at knot points" begin
            x = [0.0, 1.0, 2.0, 3.0]
            y = [0.0, 1.0, 4.0, 9.0]  # slopes: 1, 3, 5
            itp = linear_interp(x, y)

            # At interior knots, derivative uses the right segment
            @test derivative(itp, 1.0) ≈ 3.0  # slope of [1,2] segment
            @test derivative(itp, 2.0) ≈ 5.0  # slope of [2,3] segment

            # At boundaries
            @test derivative(itp, 0.0) ≈ 1.0  # slope of first segment
            @test derivative(itp, 3.0) ≈ 5.0  # slope of last segment (at right boundary)
        end

        @testset "Derivative consistency across knots" begin
            # Test that querying just before and after a knot gives expected values
            x = collect(0.0:0.5:2.0)
            y = x .^ 3
            bc = BCPair(D2(0.0), D2(12.0))  # f''(0)=0, f''(2)=12 for x³
            itp = cubic_interp(x, y; bc=bc)

            ε = 1e-8
            # At x=1: f'(1) = 3*1² = 3
            d_before = derivative(itp, 1.0 - ε)
            d_after = derivative(itp, 1.0 + ε)
            d_at = derivative(itp, 1.0)

            # All should be approximately equal (C1 continuity)
            @test d_before ≈ d_at atol=1e-4
            @test d_after ≈ d_at atol=1e-4
        end
    end

    @testset "Comprehensive type stability" begin
        @testset "Cubic derivative type inference" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Scalar queries
            @test @inferred(derivative(itp, 0.5)) isa Float64
            @test @inferred(derivative2(itp, 0.5)) isa Float64

            # With different input type (converts)
            @test @inferred(derivative(itp, 0.5f0)) isa Float64
            @test @inferred(derivative2(itp, 0.5f0)) isa Float64

            # Vector queries
            x_query = [0.25, 0.5, 0.75]
            @test @inferred(derivative(itp, x_query)) isa Vector{Float64}
            @test @inferred(derivative2(itp, x_query)) isa Vector{Float64}
        end

        @testset "Linear derivative type inference" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 1.0, 4.0]
            itp = linear_interp(x, y)

            # Scalar queries
            @test @inferred(derivative(itp, 0.5)) isa Float64
            @test @inferred(derivative2(itp, 0.5)) isa Float64

            # With different input type
            @test @inferred(derivative(itp, 0.5f0)) isa Float64
            @test @inferred(derivative2(itp, 0.5f0)) isa Float64

            # Vector queries
            x_query = [0.25, 0.5, 1.5]
            @test @inferred(derivative(itp, x_query)) isa Vector{Float64}
            @test @inferred(derivative2(itp, x_query)) isa Vector{Float64}
        end

        @testset "Float32 type preservation" begin
            x = Float32.(collect(0.0f0:0.1f0:1.0f0))
            y = x .^ 2

            # Cubic
            itp_cubic = cubic_interp(x, y)
            @test @inferred(derivative(itp_cubic, 0.5f0)) isa Float32
            @test @inferred(derivative2(itp_cubic, 0.5f0)) isa Float32

            # Linear
            itp_linear = linear_interp(x, y)
            @test @inferred(derivative(itp_linear, 0.5f0)) isa Float32
            @test @inferred(derivative2(itp_linear, 0.5f0)) isa Float32
        end

        @testset "Order parameter type inference" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2

            # Cubic with order
            @test @inferred(cubic_interp(x, y, 0.5; order=0)) isa Float64
            @test @inferred(cubic_interp(x, y, 0.5; order=1)) isa Float64
            @test @inferred(cubic_interp(x, y, 0.5; order=2)) isa Float64

            # Linear with order
            @test @inferred(linear_interp(x, y, 0.5; order=0)) isa Float64
            @test @inferred(linear_interp(x, y, 0.5; order=1)) isa Float64
            @test @inferred(linear_interp(x, y, 0.5; order=2)) isa Float64
        end

        @testset "Cache-based type inference" begin
            x = collect(0.0:0.1:1.0)
            cache = CubicSplineCache(x)
            y = x .^ 2

            @test @inferred(cubic_interp(cache, y, 0.5; order=0)) isa Float64
            @test @inferred(cubic_interp(cache, y, 0.5; order=1)) isa Float64
            @test @inferred(cubic_interp(cache, y, 0.5; order=2)) isa Float64
        end
    end

    @testset "Comprehensive allocation tests" begin
        @testset "All cubic paths zero-allocation" begin
            # Test all combinations of BC, extrap, and order
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2

            bc_types = [
                (NaturalBC(), "Natural"),
                (ClampedBC(), "Clamped"),
                (BCPair(D1(0.5), D1(1.5)), "D1-D1"),
                (BCPair(D2(2.0), D2(2.0)), "D2-D2"),
                (BCPair(D1(0.5), D2(2.0)), "D1-D2"),
            ]

            extrap_modes = [:none, :constant, :extension]

            for (bc, bc_name) in bc_types
                for extrap in extrap_modes
                    for order in 0:2
                        itp = cubic_interp(x, y; bc=bc, extrap=extrap)

                        # Warmup
                        for _ in 1:3
                            if order == 0
                                itp(0.5)
                            elseif order == 1
                                derivative(itp, 0.5)
                            else
                                derivative2(itp, 0.5)
                            end
                        end

                        # Measure
                        if order == 0
                            alloc = @allocated itp(0.5)
                        elseif order == 1
                            alloc = @allocated derivative(itp, 0.5)
                        else
                            alloc = @allocated derivative2(itp, 0.5)
                        end

                        @test alloc <= DERIV_ALLOC_THRESHOLD
                    end
                end
            end
        end

        @testset "Periodic BC all orders zero-allocation" begin
            x = collect(range(0.0, 2π, 101))
            y = sin.(x)
            y[end] = y[1]
            itp = cubic_interp(x, y; bc=PeriodicBC())

            for order in 0:2
                # Warmup
                for _ in 1:3
                    if order == 0
                        itp(1.0)
                    elseif order == 1
                        derivative(itp, 1.0)
                    else
                        derivative2(itp, 1.0)
                    end
                end

                # Measure
                if order == 0
                    alloc = @allocated itp(1.0)
                elseif order == 1
                    alloc = @allocated derivative(itp, 1.0)
                else
                    alloc = @allocated derivative2(itp, 1.0)
                end

                @test alloc <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "All linear paths zero-allocation" begin
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2

            extrap_modes = [:none, :constant, :extension, :wrap]

            for extrap in extrap_modes
                itp = linear_interp(x, y; extrap=extrap)

                for order in 0:2
                    # Warmup
                    for _ in 1:3
                        if order == 0
                            itp(0.5)
                        elseif order == 1
                            derivative(itp, 0.5)
                        else
                            derivative2(itp, 0.5)
                        end
                    end

                    # Measure
                    if order == 0
                        alloc = @allocated itp(0.5)
                    elseif order == 1
                        alloc = @allocated derivative(itp, 0.5)
                    else
                        alloc = @allocated derivative2(itp, 0.5)
                    end

                    @test alloc <= DERIV_ALLOC_THRESHOLD
                end
            end
        end

        @testset "Linear Range path zero-allocation" begin
            x = 0.0:0.02:1.0
            y = collect(x) .^ 2

            for extrap in [:none, :extension]
                itp = linear_interp(x, y; extrap=extrap)

                # Warmup
                for _ in 1:3
                    itp(0.5)
                    derivative(itp, 0.5)
                    derivative2(itp, 0.5)
                end

                alloc0 = @allocated itp(0.5)
                alloc1 = @allocated derivative(itp, 0.5)
                alloc2 = @allocated derivative2(itp, 0.5)

                @test alloc0 <= DERIV_ALLOC_THRESHOLD
                @test alloc1 <= DERIV_ALLOC_THRESHOLD
                @test alloc2 <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "Cache-based cubic all orders" begin
            x = collect(range(0.0, 1.0, 51))

            bc_types = [
                CubicSplineCache(x),
                CubicSplineCache(x; bc=ClampedBC()),
                CubicSplineCache(x; bc=BCPair(D2(2.0), D2(2.0))),
                CubicSplineCache(x; bc=PeriodicBC()),
            ]

            y = x .^ 2

            for cache in bc_types
                for order in 0:2
                    # Warmup
                    for _ in 1:3
                        cubic_interp(cache, y, 0.5; order=order)
                    end

                    alloc = @allocated cubic_interp(cache, y, 0.5; order=order)
                    @test alloc <= DERIV_ALLOC_THRESHOLD
                end
            end
        end
    end

    @testset "Edge cases and robustness" begin
        @testset "Very small grid (minimum size)" begin
            # Minimum for cubic: 3 points (can we still get derivatives?)
            x = [0.0, 0.5, 1.0]
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Should work without errors
            @test derivative(itp, 0.25) isa Float64
            @test derivative2(itp, 0.25) isa Float64

            # Linear minimum: 2 points
            x_lin = [0.0, 1.0]
            y_lin = [0.0, 2.0]
            itp_lin = linear_interp(x_lin, y_lin)

            @test derivative(itp_lin, 0.5) ≈ 2.0  # slope
            @test derivative2(itp_lin, 0.5) ≈ 0.0  # always zero
        end

        @testset "Query at domain boundaries" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Exactly at left boundary
            @test derivative(itp, 0.0) isa Float64
            @test derivative2(itp, 0.0) isa Float64

            # Exactly at right boundary
            @test derivative(itp, 1.0) isa Float64
            @test derivative2(itp, 1.0) isa Float64
        end

        @testset "Constant function" begin
            x = collect(0.0:0.1:1.0)
            y = ones(length(x)) * 5.0  # f(x) = 5

            itp_cubic = cubic_interp(x, y)
            @test derivative(itp_cubic, 0.5) ≈ 0.0 atol=1e-10
            @test derivative2(itp_cubic, 0.5) ≈ 0.0 atol=1e-10

            itp_linear = linear_interp(x, y)
            @test derivative(itp_linear, 0.5) ≈ 0.0 atol=1e-10
            @test derivative2(itp_linear, 0.5) ≈ 0.0 atol=1e-10
        end

        @testset "Linear function" begin
            x = collect(0.0:0.1:1.0)
            y = 2.0 .* x .+ 3.0  # f(x) = 2x + 3

            # Cubic should reproduce linear exactly
            itp_cubic = cubic_interp(x, y)
            @test derivative(itp_cubic, 0.5) ≈ 2.0 atol=1e-10
            @test derivative2(itp_cubic, 0.5) ≈ 0.0 atol=1e-10

            # Linear should be exact
            itp_linear = linear_interp(x, y)
            @test derivative(itp_linear, 0.5) ≈ 2.0 atol=1e-10
            @test derivative2(itp_linear, 0.5) ≈ 0.0 atol=1e-10
        end

        @testset "Non-uniform grid" begin
            # Non-uniform spacing
            x = [0.0, 0.1, 0.15, 0.5, 0.9, 1.0]
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Should still work reasonably
            @test derivative(itp, 0.5) ≈ 1.0 atol=0.1  # f'(0.5) = 2*0.5 = 1
            @test derivative2(itp, 0.5) ≈ 2.0 atol=0.2  # f''(x) = 2
        end

        @testset "Large grid" begin
            x = collect(range(0.0, 10.0, 1001))
            y = sin.(x)
            itp = cubic_interp(x, y)

            # Should handle large grids efficiently
            @test derivative(itp, 5.0) ≈ cos(5.0) atol=1e-3
            @test derivative2(itp, 5.0) ≈ -sin(5.0) atol=1e-3

            # Allocation should still be zero
            derivative(itp, 5.0)
            alloc = @allocated derivative(itp, 5.0)
            @test alloc <= DERIV_ALLOC_THRESHOLD
        end
    end

end # @testset "Derivatives"
