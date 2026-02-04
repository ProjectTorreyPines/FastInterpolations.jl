# ========================================
# Derivative Tests for FastInterpolations.jl
# ========================================
# Phase 1: Foundation tests for EvalOp types and @_dispatch_deriv macro
# Phase 2+: Kernel functions, cubic/linear derivative evaluation

using Test
using FastInterpolations

# Import internal types/macros for testing
using FastInterpolations: @_dispatch_deriv, _linear_kernel, _cubic_kernel
using FastInterpolations: _eval_cubic_at_point, _eval_cubic_with_extrap, _get_cubic_cache, _solve_system!
using FastInterpolations: AbstractEvalOp, EvalValue, EvalDeriv1, EvalDeriv2
using FastInterpolations: _to_searcher

# Julia version-aware threshold (1.12+ has improved allocation tracking)
# Note: Tg/Tv type separation (for Complex support) adds ~80-300 bytes overhead on LTS
const DERIV_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 600

# ========================================
# Group 1: Core Types and Dispatch
# ========================================
@testset "Derivative Core" begin

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

    @testset "@_dispatch_deriv macro" begin
        # deriv=0 → EvalValue
        result0 = @_dispatch_deriv 0 => op begin
            typeof(op)
        end
        @test result0 === EvalValue

        # deriv=1 → EvalDeriv1
        result1 = @_dispatch_deriv 1 => op begin
            typeof(op)
        end
        @test result1 === EvalDeriv1

        # deriv=2 → EvalDeriv2
        result2 = @_dispatch_deriv 2 => op begin
            typeof(op)
        end
        @test result2 === EvalDeriv2

        # Invalid deriv throws ArgumentError (macro-level)
        @test_throws ArgumentError @_dispatch_deriv 4 => op begin
            typeof(op)
        end
        @test_throws ArgumentError @_dispatch_deriv -1 => op begin
            typeof(op)
        end

        # Invalid deriv throws ArgumentError (public API - for coverage)
        x = [0.0, 0.5, 1.0]
        y = [0.0, 0.25, 1.0]
        @test_throws ArgumentError cubic_interp(x, y, 0.5; deriv=4)
        @test_throws ArgumentError cubic_interp(x, y, 0.5; deriv=-1)
        @test_throws ArgumentError linear_interp(x, y, 0.5; deriv=4)
        @test_throws ArgumentError linear_interp(x, y, 0.5; deriv=-1)
    end

    @testset "@_dispatch_deriv with runtime variable" begin
        # Test that macro works with runtime-determined deriv
        for deriv in 0:2
            result = @_dispatch_deriv deriv => op begin
                op
            end
            if deriv == 0
                @test result isa EvalValue
            elseif deriv == 1
                @test result isa EvalDeriv1
            else
                @test result isa EvalDeriv2
            end
        end
    end

    @testset "@_dispatch_deriv type stability" begin
        # The dispatched function should maintain type stability
        function test_dispatch(deriv::Int)
            @_dispatch_deriv deriv => op begin
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

end # Derivative Core

# ========================================
# Group 2: Kernel Functions
# ========================================
@testset "Derivative Kernels" begin

    @testset "Linear kernels" begin
        # Test case: L(x) = 1 + 2x on [0, 1]
        # yL = L(0) = 1, yR = L(1) = 3
        # L(0.5) = 2, L'(x) = 2, L''(x) = 0
        h, yL, yR = 1.0, 1.0, 3.0
        dL = 0.5  # x = 0.5

        @testset "EvalValue" begin
            @test _linear_kernel(EvalValue(), yL, yR, h, dL) ≈ 2.0
            # Edge cases
            @test _linear_kernel(EvalValue(), yL, yR, h, 0.0) ≈ yL  # left boundary
            @test _linear_kernel(EvalValue(), yL, yR, h, h) ≈ yR    # right boundary
        end

        @testset "EvalDeriv1" begin
            @test _linear_kernel(EvalDeriv1(), yL, yR, h, dL) ≈ 2.0
            # Slope is constant everywhere
            @test _linear_kernel(EvalDeriv1(), yL, yR, h, 0.0) ≈ 2.0
            @test _linear_kernel(EvalDeriv1(), yL, yR, h, 0.9) ≈ 2.0
        end

        @testset "EvalDeriv2" begin
            @test _linear_kernel(EvalDeriv2(), yL, yR, h, dL) ≈ 0.0
            # Second derivative is always zero
            @test _linear_kernel(EvalDeriv2(), yL, yR, h, 0.0) === 0.0
            @test _linear_kernel(EvalDeriv2(), yL, yR, h, h) === 0.0
        end

        @testset "Type stability" begin
            @test @inferred(_linear_kernel(EvalValue(), yL, yR, h, dL)) isa Float64
            @test @inferred(_linear_kernel(EvalDeriv1(), yL, yR, h, dL)) isa Float64
            @test @inferred(_linear_kernel(EvalDeriv2(), yL, yR, h, dL)) isa Float64

            # Float32 preservation
            yL_f32, yR_f32, h_f32, dL_f32 = 1.0f0, 3.0f0, 1.0f0, 0.5f0
            @test @inferred(_linear_kernel(EvalValue(), yL_f32, yR_f32, h_f32, dL_f32)) isa Float32
        end

        @testset "Different slopes" begin
            # Negative slope: L(x) = 5 - 3x on [0, 2]
            @test _linear_kernel(EvalDeriv1(), 5.0, -1.0, 2.0, 1.0) ≈ -3.0
            # Zero slope: constant function
            @test _linear_kernel(EvalDeriv1(), 4.0, 4.0, 2.0, 1.0) ≈ 0.0
        end
    end

    @testset "Cubic kernels" begin
        # Test with known quadratic: f(x) = x² on [0, 1]
        # f(0) = 0, f(1) = 1, f'(x) = 2x, f''(x) = 2
        # For natural spline on x² with enough points, z values approximate f''
        h = 1.0
        inv_h = inv(h)  # Precomputed reciprocal for kernel
        yL, yR = 0.0, 1.0
        zL, zR = 2.0, 2.0  # f''(x) = 2 (constant for quadratic)

        @testset "EvalValue - quadratic exactness" begin
            # At x = 0.5: f(0.5) = 0.25
            dL, dR = 0.5, 0.5
            result = _cubic_kernel(EvalValue(), zL, zR, yL, yR, h, inv_h, dL, dR)
            @test result ≈ 0.25 atol=1e-10

            # At boundaries
            @test _cubic_kernel(EvalValue(), zL, zR, yL, yR, h, inv_h, 0.0, 1.0) ≈ yL atol=1e-10
            @test _cubic_kernel(EvalValue(), zL, zR, yL, yR, h, inv_h, 1.0, 0.0) ≈ yR atol=1e-10
        end

        @testset "EvalDeriv1 - derivative of quadratic" begin
            # f'(x) = 2x, so f'(0.5) = 1.0
            dL, dR = 0.5, 0.5
            result = _cubic_kernel(EvalDeriv1(), zL, zR, yL, yR, h, inv_h, dL, dR)
            @test result ≈ 1.0 atol=1e-10

            # f'(0) = 0
            @test _cubic_kernel(EvalDeriv1(), zL, zR, yL, yR, h, inv_h, 0.0, 1.0) ≈ 0.0 atol=1e-10
            # f'(1) = 2
            @test _cubic_kernel(EvalDeriv1(), zL, zR, yL, yR, h, inv_h, 1.0, 0.0) ≈ 2.0 atol=1e-10
        end

        @testset "EvalDeriv2 - second derivative of quadratic" begin
            # f''(x) = 2 everywhere
            @test _cubic_kernel(EvalDeriv2(), zL, zR, yL, yR, h, inv_h, 0.5, 0.5) ≈ 2.0 atol=1e-10
            @test _cubic_kernel(EvalDeriv2(), zL, zR, yL, yR, h, inv_h, 0.0, 1.0) ≈ 2.0 atol=1e-10
            @test _cubic_kernel(EvalDeriv2(), zL, zR, yL, yR, h, inv_h, 1.0, 0.0) ≈ 2.0 atol=1e-10
        end

        @testset "Type stability" begin
            dL, dR = 0.5, 0.5
            @test @inferred(_cubic_kernel(EvalValue(), zL, zR, yL, yR, h, inv_h, dL, dR)) isa Float64
            @test @inferred(_cubic_kernel(EvalDeriv1(), zL, zR, yL, yR, h, inv_h, dL, dR)) isa Float64
            @test @inferred(_cubic_kernel(EvalDeriv2(), zL, zR, yL, yR, h, inv_h, dL, dR)) isa Float64

            # Float32 preservation
            args_f32 = (2.0f0, 2.0f0, 0.0f0, 1.0f0, 1.0f0, inv(1.0f0), 0.5f0, 0.5f0)
            @test @inferred(_cubic_kernel(EvalValue(), args_f32...)) isa Float32
        end

        @testset "Varying z values (non-constant curvature)" begin
            # Test with zL ≠ zR (linear interpolation of z)
            zL, zR = 0.0, 4.0
            dL, dR = 0.5, 0.5

            # f''(0.5) should be average of z values
            result = _cubic_kernel(EvalDeriv2(), zL, zR, yL, yR, h, inv_h, dL, dR)
            @test result ≈ 2.0 atol=1e-10  # (0*0.5 + 4*0.5) / 1 = 2
        end

        @testset "Cubic polynomial exactness" begin
            # For a true cubic f(x) = x³ on [0, 1]:
            # f(0) = 0, f(1) = 1
            # f''(x) = 6x, so zL = f''(0) = 0, zR = f''(1) = 6
            yL_cubic, yR_cubic = 0.0, 1.0
            zL_cubic, zR_cubic = 0.0, 6.0
            h = 1.0
            inv_h = inv(h)

            # At x = 0.5: f(0.5) = 0.125, f'(0.5) = 0.75, f''(0.5) = 3
            dL, dR = 0.5, 0.5
            @test _cubic_kernel(EvalValue(), zL_cubic, zR_cubic, yL_cubic, yR_cubic, h, inv_h, dL, dR) ≈ 0.125 atol=1e-10
            @test _cubic_kernel(EvalDeriv1(), zL_cubic, zR_cubic, yL_cubic, yR_cubic, h, inv_h, dL, dR) ≈ 0.75 atol=1e-10
            @test _cubic_kernel(EvalDeriv2(), zL_cubic, zR_cubic, yL_cubic, yR_cubic, h, inv_h, dL, dR) ≈ 3.0 atol=1e-10
        end
    end

end # Derivative Kernels

# ========================================
# Group 3: Cubic Derivative API
# ========================================
@testset "Cubic Derivatives" begin

    @testset "Cubic internal functions with op" begin
        # Test with quadratic f(x) = x² on [0, 2] with step 0.5
        x = collect(0.0:0.5:2.0)
        y = x .^ 2  # [0, 0.25, 1, 2.25, 4]

        # Use Deriv2 BC with f''(x) = 2 for exact quadratic representation
        cache = _get_cubic_cache(x, BCPair(Deriv2(2.0), Deriv2(2.0)))
        z = similar(y)
        _solve_system!(z, cache, y, cache.bc_config)

        @testset "_eval_cubic_at_point with op" begin
            # Value at midpoint x=1.0: f(1) = 1.0
            val = _eval_cubic_at_point(x, y, cache.spacing, z, 1.0, EvalValue())
            @test val ≈ 1.0 atol=1e-10

            # First derivative at x=1.0: f'(x) = 2x, so f'(1) = 2.0
            deriv1 = _eval_cubic_at_point(x, y, cache.spacing, z, 1.0, EvalDeriv1())
            @test deriv1 ≈ 2.0 atol=0.1  # Spline approximation

            # Second derivative: f''(x) = 2.0
            deriv2 = _eval_cubic_at_point(x, y, cache.spacing, z, 1.0, EvalDeriv2())
            @test deriv2 ≈ 2.0 atol=0.1
        end

        @testset "_eval_cubic_with_extrap with op" begin
            # Test constant extrapolation with derivatives

            searcher = _to_searcher(Binary())

            # Outside left boundary: should return 0 for derivatives
            left_val = _eval_cubic_with_extrap(x, y, cache.spacing, z, -0.5, Val(:constant), EvalValue(), searcher)
            @test left_val ≈ y[1]  # y[1] = 0.0

            left_deriv1 = _eval_cubic_with_extrap(x, y, cache.spacing, z, -0.5, Val(:constant), EvalDeriv1(), searcher)
            @test left_deriv1 === 0.0  # Constant extrap → derivative = 0

            left_deriv2 = _eval_cubic_with_extrap(x, y, cache.spacing, z, -0.5, Val(:constant), EvalDeriv2(), searcher  )
            @test left_deriv2 === 0.0

            # Inside domain: should use normal evaluation
            mid_deriv1 = _eval_cubic_with_extrap(x, y, cache.spacing, z, 1.0, Val(:constant), EvalDeriv1(), searcher)
            @test mid_deriv1 ≈ 2.0 atol=0.1

            # Extension extrapolation: use boundary polynomial
            ext_deriv1 = _eval_cubic_with_extrap(x, y, cache.spacing, z, -0.5, Val(:extension), EvalDeriv1(), searcher)
            @test ext_deriv1 isa Float64  # Should not throw
        end

        @testset "Type stability with op" begin
            @test @inferred(_eval_cubic_at_point(x, y, cache.spacing, z, 1.0, EvalValue())) isa Float64
            @test @inferred(_eval_cubic_at_point(x, y, cache.spacing, z, 1.0, EvalDeriv1())) isa Float64
            @test @inferred(_eval_cubic_at_point(x, y, cache.spacing, z, 1.0, EvalDeriv2())) isa Float64
        end

        @testset "Derivative at different points" begin
            # f'(0) = 0, f'(0.5) = 1, f'(1) = 2, f'(1.5) = 3, f'(2) = 4
            for (xi, expected_deriv) in [(0.0, 0.0), (0.5, 1.0), (1.5, 3.0), (2.0, 4.0)]
                deriv = _eval_cubic_at_point(x, y, cache.spacing, z, xi, EvalDeriv1())
                @test deriv ≈ expected_deriv atol=0.15
            end
        end
    end

    @testset "Cubic public API with deriv" begin

        @testset "Polynomial exactness" begin
            # Quadratic f(x) = x² with exact Deriv2 BC
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            bc = BCPair(Deriv2(2.0), Deriv2(2.0))  # f''(x) = 2
            xi = 0.5

            # Value: f(0.5) = 0.25
            @test cubic_interp(x, y, xi; bc=bc, deriv=0) ≈ 0.25 atol=1e-10

            # First derivative: f'(0.5) = 2*0.5 = 1.0
            @test cubic_interp(x, y, xi; bc=bc, deriv=1) ≈ 1.0 atol=1e-10

            # Second derivative: f''(x) = 2.0
            @test cubic_interp(x, y, xi; bc=bc, deriv=2) ≈ 2.0 atol=1e-10
        end

        @testset "Cubic polynomial exactness" begin
            # Cubic f(x) = x³ with exact Deriv2 BC
            x = collect(0.0:0.1:1.0)
            y = x .^ 3
            bc = BCPair(Deriv2(0.0), Deriv2(6.0))  # f''(0)=0, f''(1)=6
            xi = 0.5

            # Value: f(0.5) = 0.125
            @test cubic_interp(x, y, xi; bc=bc, deriv=0) ≈ 0.125 atol=1e-10

            # First derivative: f'(0.5) = 3*(0.5)² = 0.75
            @test cubic_interp(x, y, xi; bc=bc, deriv=1) ≈ 0.75 atol=1e-10

            # Second derivative: f''(0.5) = 6*0.5 = 3.0
            @test cubic_interp(x, y, xi; bc=bc, deriv=2) ≈ 3.0 atol=1e-10
        end

        @testset "Backward compatibility (no deriv arg)" begin
            x = collect(0.0:0.2:1.0)
            y = sin.(x)
            xi = 0.5

            # Without deriv parameter should work as before
            val_old = cubic_interp(x, y, xi)
            val_new = cubic_interp(x, y, xi; deriv=0)
            @test val_old ≈ val_new atol=1e-14
        end

        @testset "Vector query with deriv" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            bc = BCPair(Deriv2(2.0), Deriv2(2.0))
            x_query = [0.25, 0.5, 0.75]

            # Values
            vals = cubic_interp(x, y, x_query; bc=bc, deriv=0)
            @test vals ≈ x_query .^ 2 atol=1e-10

            # First derivatives: f'(x) = 2x
            derivs = cubic_interp(x, y, x_query; bc=bc, deriv=1)
            @test derivs ≈ 2.0 .* x_query atol=1e-10

            # Second derivatives: f''(x) = 2
            derivs2 = cubic_interp(x, y, x_query; bc=bc, deriv=2)
            @test all(d ≈ 2.0 for d in derivs2)
        end

        @testset "Cache-based with deriv" begin
            x = collect(0.0:0.1:1.0)
            cache = CubicSplineCache(x; bc=BCPair(Deriv2(2.0), Deriv2(2.0)))
            y = x .^ 2
            xi = 0.5

            @test cubic_interp(cache, y, xi; deriv=0) ≈ 0.25 atol=1e-10
            @test cubic_interp(cache, y, xi; deriv=1) ≈ 1.0 atol=1e-10
            @test cubic_interp(cache, y, xi; deriv=2) ≈ 2.0 atol=1e-10
        end

        @testset "Type stability with deriv" begin
            x = collect(0.0:0.1:1.0)
            y = x .^ 2
            bc = BCPair(Deriv2(2.0), Deriv2(2.0))
            xi = 0.5

            @test @inferred(cubic_interp(x, y, xi; bc=bc, deriv=0)) isa Float64
            @test @inferred(cubic_interp(x, y, xi; bc=bc, deriv=1)) isa Float64
            @test @inferred(cubic_interp(x, y, xi; bc=bc, deriv=2)) isa Float64
        end
    end

    @testset "Cubic extrapolation with deriv" begin
        x = collect(0.0:0.25:1.0)
        y = x .^ 2
        bc = BCPair(Deriv2(2.0), Deriv2(2.0))

        @testset "Constant extrapolation" begin
            # Left boundary constant extrap: returns y[1] for value, 0 for derivatives
            @test cubic_interp(x, y, -0.5; bc=bc, extrap=:constant, deriv=0) ≈ 0.0
            @test cubic_interp(x, y, -0.5; bc=bc, extrap=:constant, deriv=1) ≈ 0.0
            @test cubic_interp(x, y, -0.5; bc=bc, extrap=:constant, deriv=2) ≈ 0.0

            # Right boundary
            @test cubic_interp(x, y, 1.5; bc=bc, extrap=:constant, deriv=0) ≈ 1.0
            @test cubic_interp(x, y, 1.5; bc=bc, extrap=:constant, deriv=1) ≈ 0.0
            @test cubic_interp(x, y, 1.5; bc=bc, extrap=:constant, deriv=2) ≈ 0.0
        end

        @testset "Extension extrapolation" begin
            # Extension: continue boundary polynomial
            # For x², extension should give approximately correct derivatives
            val = cubic_interp(x, y, 1.5; bc=bc, extrap=:extension, deriv=0)
            @test val ≈ 2.25 atol=0.1  # (1.5)² ≈ 2.25

            deriv1 = cubic_interp(x, y, 1.5; bc=bc, extrap=:extension, deriv=1)
            @test deriv1 ≈ 3.0 atol=0.2  # 2*1.5 ≈ 3.0

            deriv2 = cubic_interp(x, y, 1.5; bc=bc, extrap=:extension, deriv=2)
            @test deriv2 ≈ 2.0 atol=0.1  # f''(x) = 2
        end
    end

    @testset "CubicInterpolant itp(xi; deriv=N) API" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2
        bc = BCPair(Deriv2(2.0), Deriv2(2.0))
        itp = cubic_interp(x, y; bc=bc)

        @testset "deriv=1 scalar" begin
            @test itp(0.5; deriv=1) ≈ 1.0 atol=1e-10
            @test itp(0.0; deriv=1) ≈ 0.0 atol=1e-10
            @test itp(1.0; deriv=1) ≈ 2.0 atol=1e-10
        end

        @testset "deriv=2 scalar" begin
            @test itp(0.5; deriv=2) ≈ 2.0 atol=1e-10
            @test itp(0.0; deriv=2) ≈ 2.0 atol=1e-10
            @test itp(1.0; deriv=2) ≈ 2.0 atol=1e-10
        end

        @testset "deriv=1 vector" begin
            x_query = [0.25, 0.5, 0.75]
            derivs = itp(x_query; deriv=1)
            @test derivs ≈ 2.0 .* x_query atol=1e-10
        end

        @testset "deriv=2 vector" begin
            x_query = [0.25, 0.5, 0.75]
            derivs2 = itp(x_query; deriv=2)
            @test all(d ≈ 2.0 for d in derivs2)
        end

        @testset "in-place deriv=1" begin
            x_query = [0.25, 0.5, 0.75]
            output = zeros(3)
            itp(output, x_query; deriv=1)
            @test output ≈ 2.0 .* x_query atol=1e-10
        end

        @testset "in-place deriv=2" begin
            x_query = [0.25, 0.5, 0.75]
            output = zeros(3)
            itp(output, x_query; deriv=2)
            @test all(d ≈ 2.0 for d in output)
        end
    end

    @testset "CubicInterpolant deriv keyword" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2
        bc = BCPair(Deriv2(2.0), Deriv2(2.0))
        itp = cubic_interp(x, y; bc=bc)

        @testset "deriv=0 matches default call" begin
            @test itp(0.5) == itp(0.5; deriv=0)
            @test itp(0.25) == itp(0.25; deriv=0)
            @test itp(0.75) == itp(0.75; deriv=0)
        end

        @testset "deriv=1 returns first derivative" begin
            # f(x) = x², f'(x) = 2x
            @test itp(0.5; deriv=1) ≈ 1.0 atol=1e-10
            @test itp(0.0; deriv=1) ≈ 0.0 atol=1e-10
            @test itp(1.0; deriv=1) ≈ 2.0 atol=1e-10
        end

        @testset "deriv=2 returns second derivative" begin
            # f(x) = x², f''(x) = 2
            @test itp(0.5; deriv=2) ≈ 2.0 atol=1e-10
            @test itp(0.0; deriv=2) ≈ 2.0 atol=1e-10
            @test itp(1.0; deriv=2) ≈ 2.0 atol=1e-10
        end

        @testset "Real input works with deriv keyword" begin
            # Integer input should work (exact equality - same code path after conversion)
            @test itp(1; deriv=0) == itp(1.0; deriv=0)
            @test itp(1; deriv=1) == itp(1.0; deriv=1)
            @test itp(1; deriv=2) == itp(1.0; deriv=2)

            # Float32 input should work (exact equality - 0.5f0 converts exactly to 0.5)
            @test itp(0.5f0; deriv=1) == itp(0.5; deriv=1)
        end

        @testset "Type stability with deriv keyword" begin
            @test @inferred(itp(0.5; deriv=0)) isa Float64
            @test @inferred(itp(0.5; deriv=1)) isa Float64
            @test @inferred(itp(0.5; deriv=2)) isa Float64
        end

        @testset "Polynomial exactness" begin
            # f(x) = x², f'(x) = 2x, f''(x) = 2
            @test itp(0.5; deriv=0) ≈ 0.25 atol=1e-10
            @test itp(0.5; deriv=1) ≈ 1.0 atol=1e-10
            @test itp(0.5; deriv=2) ≈ 2.0 atol=1e-10
        end
    end

    @testset "CubicInterpolant deriv keyword - different BCs" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(x)

        bc_types = [
            NaturalBC(),
            ClampedBC(),
            BCPair(Deriv1(1.0), Deriv1(cos(1.0))),
        ]

        for bc in bc_types
            itp = cubic_interp(x, y; bc=bc)

            # Should work without errors
            @test itp(0.5; deriv=0) isa Float64
            @test itp(0.5; deriv=1) isa Float64
            @test itp(0.5; deriv=2) isa Float64

            # deriv=1 gives first derivative (approximately cos(0.5) for sin)
            @test itp(0.5; deriv=1) ≈ cos(0.5) atol=0.01
        end
    end

    @testset "CubicInterpolant deriv keyword - Periodic BC" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]
        itp = cubic_interp(x, y; bc=PeriodicBC())

        # Should work with periodic BC
        @test itp(1.0; deriv=0) isa Float64
        @test itp(1.0; deriv=1) isa Float64
        @test itp(1.0; deriv=2) isa Float64

        # deriv=1 gives first derivative (cos(1.0) for sin)
        @test itp(1.0; deriv=1) ≈ cos(1.0) atol=0.01

        # Wrap around domain works
        @test itp(7.0; deriv=1) ≈ cos(7.0 - 2π) atol=0.01
    end

end # Cubic Derivatives

# ========================================
# Group 4: Linear Derivative API
# ========================================
@testset "Linear Derivatives" begin

    @testset "Linear public API with deriv" begin

        @testset "Constant slope segments" begin
            # Two segments with different slopes
            x = [0.0, 1.0, 3.0]
            y = [0.0, 2.0, 4.0]  # slopes: 2.0 (first), 1.0 (second)

            # Values
            @test linear_interp(x, y, 0.5; deriv=0) ≈ 1.0  # midpoint first segment
            @test linear_interp(x, y, 2.0; deriv=0) ≈ 3.0  # midpoint second segment

            # First derivatives (constant within segment)
            @test linear_interp(x, y, 0.5; deriv=1) ≈ 2.0  # first segment slope
            @test linear_interp(x, y, 2.0; deriv=1) ≈ 1.0  # second segment slope
            @test linear_interp(x, y, 0.0; deriv=1) ≈ 2.0  # at left boundary
            @test linear_interp(x, y, 1.0; deriv=1) ≈ 1.0  # at knot (use right segment)

            # Second derivatives (always zero for linear)
            @test linear_interp(x, y, 0.5; deriv=2) ≈ 0.0
            @test linear_interp(x, y, 2.0; deriv=2) ≈ 0.0
        end

        @testset "Backward compatibility (no deriv arg)" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 1.0, 4.0]
            xi = 0.5

            val_old = linear_interp(x, y, xi)
            val_new = linear_interp(x, y, xi; deriv=0)
            @test val_old ≈ val_new atol=1e-14
        end

        @testset "Vector query with deriv" begin
            x = [0.0, 1.0, 3.0]
            y = [0.0, 2.0, 4.0]  # slopes: 2.0, 1.0
            x_query = [0.25, 0.75, 1.5, 2.5]

            # Values
            vals = linear_interp(x, y, x_query; deriv=0)
            @test vals[1] ≈ 0.5   # 0 + 2.0*0.25
            @test vals[2] ≈ 1.5   # 0 + 2.0*0.75
            @test vals[3] ≈ 2.5   # 2 + 1.0*0.5
            @test vals[4] ≈ 3.5   # 2 + 1.0*1.5

            # First derivatives
            derivs = linear_interp(x, y, x_query; deriv=1)
            @test derivs[1] ≈ 2.0  # first segment
            @test derivs[2] ≈ 2.0  # first segment
            @test derivs[3] ≈ 1.0  # second segment
            @test derivs[4] ≈ 1.0  # second segment

            # Second derivatives (all zero)
            derivs2 = linear_interp(x, y, x_query; deriv=2)
            @test all(d ≈ 0.0 for d in derivs2)
        end

        @testset "In-place with deriv" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 2.0, 6.0]  # slopes: 2.0, 4.0
            x_query = [0.5, 1.5]
            output = zeros(2)

            # Value
            linear_interp!(output, x, y, x_query; deriv=0)
            @test output[1] ≈ 1.0
            @test output[2] ≈ 4.0

            # First derivative
            linear_interp!(output, x, y, x_query; deriv=1)
            @test output[1] ≈ 2.0
            @test output[2] ≈ 4.0

            # Second derivative
            linear_interp!(output, x, y, x_query; deriv=2)
            @test all(o ≈ 0.0 for o in output)
        end

        @testset "Type stability with deriv" begin
            x = [0.0, 1.0, 2.0]
            y = [0.0, 1.0, 4.0]
            xi = 0.5

            @test @inferred(linear_interp(x, y, xi; deriv=0)) isa Float64
            @test @inferred(linear_interp(x, y, xi; deriv=1)) isa Float64
            @test @inferred(linear_interp(x, y, xi; deriv=2)) isa Float64
        end
    end

    @testset "Linear extrapolation with deriv" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 2.0, 6.0]  # slopes: 2.0, 4.0

        @testset "Constant extrapolation" begin
            # Left boundary: returns y[1], derivatives = 0
            @test linear_interp(x, y, -0.5; extrap=:constant, deriv=0) ≈ 0.0
            @test linear_interp(x, y, -0.5; extrap=:constant, deriv=1) ≈ 0.0
            @test linear_interp(x, y, -0.5; extrap=:constant, deriv=2) ≈ 0.0

            # Right boundary: returns y[end], derivatives = 0
            @test linear_interp(x, y, 2.5; extrap=:constant, deriv=0) ≈ 6.0
            @test linear_interp(x, y, 2.5; extrap=:constant, deriv=1) ≈ 0.0
            @test linear_interp(x, y, 2.5; extrap=:constant, deriv=2) ≈ 0.0
        end

        @testset "Extension extrapolation" begin
            # Left: extends first segment (slope 2.0)
            @test linear_interp(x, y, -0.5; extrap=:extension, deriv=0) ≈ -1.0
            @test linear_interp(x, y, -0.5; extrap=:extension, deriv=1) ≈ 2.0
            @test linear_interp(x, y, -0.5; extrap=:extension, deriv=2) ≈ 0.0

            # Right: extends last segment (slope 4.0)
            @test linear_interp(x, y, 2.5; extrap=:extension, deriv=0) ≈ 8.0
            @test linear_interp(x, y, 2.5; extrap=:extension, deriv=1) ≈ 4.0
            @test linear_interp(x, y, 2.5; extrap=:extension, deriv=2) ≈ 0.0
        end

        @testset "Wrap extrapolation" begin
            # Domain [0, 2), wrap 2.5 -> 0.5 (first segment)
            @test linear_interp(x, y, 2.5; extrap=:wrap, deriv=0) ≈ 1.0  # same as 0.5
            @test linear_interp(x, y, 2.5; extrap=:wrap, deriv=1) ≈ 2.0  # first segment slope
            @test linear_interp(x, y, 2.5; extrap=:wrap, deriv=2) ≈ 0.0
        end
    end

    @testset "LinearInterpolant itp(xi; deriv=N) API" begin
        x = [0.0, 1.0, 3.0]
        y = [0.0, 2.0, 4.0]  # slopes: 2.0, 1.0
        itp = linear_interp(x, y)

        @testset "deriv=1 scalar" begin
            @test itp(0.5; deriv=1) ≈ 2.0   # first segment
            @test itp(2.0; deriv=1) ≈ 1.0   # second segment
            @test itp(0.0; deriv=1) ≈ 2.0   # left boundary
        end

        @testset "deriv=2 scalar" begin
            # Always zero for linear
            @test itp(0.5; deriv=2) ≈ 0.0
            @test itp(2.0; deriv=2) ≈ 0.0
        end

        @testset "deriv=1 vector" begin
            x_query = [0.25, 0.75, 1.5, 2.5]
            derivs = itp(x_query; deriv=1)
            @test derivs[1] ≈ 2.0
            @test derivs[2] ≈ 2.0
            @test derivs[3] ≈ 1.0
            @test derivs[4] ≈ 1.0
        end

        @testset "deriv=2 vector" begin
            x_query = [0.25, 0.75, 1.5, 2.5]
            derivs2 = itp(x_query; deriv=2)
            @test all(d ≈ 0.0 for d in derivs2)
        end

        @testset "in-place deriv=1" begin
            x_query = [0.25, 0.75, 1.5, 2.5]
            output = zeros(4)
            itp(output, x_query; deriv=1)
            @test output[1] ≈ 2.0
            @test output[2] ≈ 2.0
            @test output[3] ≈ 1.0
            @test output[4] ≈ 1.0
        end

        @testset "in-place deriv=2" begin
            x_query = [0.25, 0.75, 1.5, 2.5]
            output = zeros(4)
            itp(output, x_query; deriv=2)
            @test all(d ≈ 0.0 for d in output)
        end
    end

    @testset "Linear Range optimization with deriv" begin
        # Range should use O(1) path
        x = 0.0:0.1:1.0
        y = collect(x) .^ 2
        xi = 0.55

        @test linear_interp(x, y, xi; deriv=0) ≈ linear_interp(collect(x), y, xi; deriv=0)
        @test linear_interp(x, y, xi; deriv=1) ≈ linear_interp(collect(x), y, xi; deriv=1)
        @test linear_interp(x, y, xi; deriv=2) ≈ 0.0
    end

    # ========================================
    # LinearInterpolant Order Keyword Tests (Phase 2)
    # ========================================

    @testset "LinearInterpolant deriv keyword" begin
        # Linear function: y = 2x on [0,1] and y = 4x - 2 on [1,2]
        # Slopes: 2.0 on [0,1], 4.0 on [1,2]
        x = [0.0, 1.0, 2.0]
        y = [0.0, 2.0, 6.0]
        litp = linear_interp(x, y)

        @testset "deriv=0 matches default call" begin
            @test litp(0.5) == litp(0.5; deriv=0)
            @test litp(0.25) == litp(0.25; deriv=0)
            @test litp(1.5) == litp(1.5; deriv=0)
        end

        @testset "deriv=1 returns correct slopes" begin
            # Interval [0,1]: slope = (2-0)/(1-0) = 2.0
            @test litp(0.5; deriv=1) ≈ 2.0
            @test litp(0.25; deriv=1) ≈ 2.0
            # Interval [1,2]: slope = (6-2)/(2-1) = 4.0
            @test litp(1.5; deriv=1) ≈ 4.0
        end

        @testset "deriv=2 returns zero" begin
            # Linear interpolation has no curvature
            @test litp(0.5; deriv=2) === 0.0
            @test litp(1.0; deriv=2) === 0.0
            @test litp(1.5; deriv=2) === 0.0
        end

        @testset "deriv=1 returns correct slopes" begin
            # Interval [0,1]: slope = (2-0)/(1-0) = 2.0
            @test litp(0.5; deriv=1) ≈ 2.0
            @test litp(0.0; deriv=1) ≈ 2.0
            # Interval [1,2]: slope = (6-2)/(2-1) = 4.0
            @test litp(1.5; deriv=1) ≈ 4.0
            @test litp(2.0; deriv=1) ≈ 4.0
        end

        @testset "deriv=2 returns zero" begin
            # Linear interpolation has no curvature
            @test litp(0.5; deriv=2) === 0.0
            @test litp(1.0; deriv=2) === 0.0
            @test litp(1.5; deriv=2) === 0.0
        end

        @testset "Real input works with deriv keyword" begin
            # Integer input should work (exact equality - same code path after conversion)
            @test litp(1; deriv=0) == litp(1.0; deriv=0)
            @test litp(1; deriv=1) == litp(1.0; deriv=1)
            @test litp(1; deriv=2) == litp(1.0; deriv=2)

            # Float32 input should work (exact equality - 0.5f0 converts exactly to 0.5)
            @test litp(0.5f0; deriv=1) == litp(0.5; deriv=1)
        end

        @testset "Type stability with deriv keyword" begin
            @test @inferred(litp(0.5; deriv=0)) isa Float64
            @test @inferred(litp(0.5; deriv=1)) isa Float64
            @test @inferred(litp(0.5; deriv=2)) isa Float64
        end
    end

    @testset "LinearInterpolant deriv keyword - different extrap modes" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 2.0, 6.0]  # slopes: 2.0, 4.0

        extrap_modes = [:none, :constant, :extension, :wrap]

        for mode in extrap_modes
            mode == :none && continue  # Skip :none for out-of-domain test

            litp = linear_interp(x, y; extrap=mode)
            @testset "extrap=$mode" begin
                # In-domain tests work for all modes
                @test litp(0.5; deriv=0) ≈ 1.0
                @test litp(0.5; deriv=1) ≈ 2.0
                @test litp(0.5; deriv=2) === 0.0
            end
        end
    end

end # Linear Derivatives

# ========================================
# Group 5: Periodic and Boundary Behavior
# ========================================
@testset "Derivative Boundary Behavior" begin

    @testset "Periodic BC derivative continuity" begin
        # Test that derivatives are continuous at the wrap point
        x = collect(range(0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        itp = cubic_interp(x, y; bc=PeriodicBC())

        ε = 1e-6

        @testset "First derivative continuity at boundaries" begin
            # Derivative at left boundary should match derivative at right boundary
            d_left = itp(ε; deriv=1)
            d_right = itp(2π - ε; deriv=1)

            # For sin(x), d/dx = cos(x), so cos(0) ≈ cos(2π) ≈ 1
            @test d_left ≈ d_right atol=1e-4

            # Also test using 4-arg API
            d_left_4arg = cubic_interp(x, y, ε; bc=PeriodicBC(), deriv=1)
            d_right_4arg = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), deriv=1)
            @test d_left_4arg ≈ d_right_4arg atol=1e-4
        end

        @testset "Second derivative continuity at boundaries" begin
            d2_left = itp(ε; deriv=2)
            d2_right = itp(2π - ε; deriv=2)

            # For sin(x), d²/dx² = -sin(x), so -sin(0) ≈ -sin(2π) ≈ 0
            @test d2_left ≈ d2_right atol=1e-4

            # Also test using 4-arg API
            d2_left_4arg = cubic_interp(x, y, ε; bc=PeriodicBC(), deriv=2)
            d2_right_4arg = cubic_interp(x, y, 2π - ε; bc=PeriodicBC(), deriv=2)
            @test d2_left_4arg ≈ d2_right_4arg atol=1e-4
        end

        @testset "Derivative at wrap point" begin
            # Test querying exactly at 0 and 2π (they should be equivalent)
            d_at_zero = itp(0.0; deriv=1)
            # Query outside domain wraps to inside
            d_at_2pi_plus = itp(2π + ε; deriv=1)

            @test d_at_zero ≈ d_at_2pi_plus atol=1e-4
        end

        @testset "Cosine function derivatives" begin
            # cos(x) has d/dx = -sin(x), d²/dx² = -cos(x)
            y_cos = cos.(x)
            y_cos[end] = y_cos[1]
            itp_cos = cubic_interp(x, y_cos; bc=PeriodicBC())

            # At x = π/2: cos(π/2) = 0, cos'(π/2) = -sin(π/2) = -1, cos''(π/2) = -cos(π/2) = 0
            @test itp_cos(π/2) ≈ 0.0 atol=1e-3
            @test itp_cos(π/2; deriv=1) ≈ -1.0 atol=1e-2
            @test itp_cos(π/2; deriv=2) ≈ 0.0 atol=1e-2

            # At x = π: cos(π) = -1, cos'(π) = 0, cos''(π) = 1
            @test itp_cos(π) ≈ -1.0 atol=1e-3
            @test itp_cos(π; deriv=1) ≈ 0.0 atol=1e-2
            @test itp_cos(π; deriv=2) ≈ 1.0 atol=1e-2
        end
    end

    @testset "Boundary point behavior" begin
        # Test right-continuous behavior at knots

        @testset "Cubic at knot points" begin
            x = collect(0.0:0.25:1.0)
            y = x .^ 2
            bc = BCPair(Deriv2(2.0), Deriv2(2.0))
            itp = cubic_interp(x, y; bc=bc)

            # At interior knot (x=0.5), derivative should be well-defined
            @test itp(0.5; deriv=1) ≈ 1.0 atol=1e-10  # f'(0.5) = 2*0.5 = 1

            # At boundaries
            @test itp(0.0; deriv=1) ≈ 0.0 atol=1e-10  # f'(0) = 0
            @test itp(1.0; deriv=1) ≈ 2.0 atol=1e-10  # f'(1) = 2

            # Second derivative should be constant (=2) for quadratic
            @test itp(0.0; deriv=2) ≈ 2.0 atol=1e-10
            @test itp(0.5; deriv=2) ≈ 2.0 atol=1e-10
            @test itp(1.0; deriv=2) ≈ 2.0 atol=1e-10
        end

        @testset "Linear at knot points" begin
            x = [0.0, 1.0, 2.0, 3.0]
            y = [0.0, 1.0, 4.0, 9.0]  # slopes: 1, 3, 5
            itp = linear_interp(x, y)

            # At interior knots, derivative uses the right segment
            @test itp(1.0; deriv=1) ≈ 3.0  # slope of [1,2] segment
            @test itp(2.0; deriv=1) ≈ 5.0  # slope of [2,3] segment

            # At boundaries
            @test itp(0.0; deriv=1) ≈ 1.0  # slope of first segment
            @test itp(3.0; deriv=1) ≈ 5.0  # slope of last segment (at right boundary)
        end

        @testset "Derivative consistency across knots" begin
            # Test that querying just before and after a knot gives expected values
            x = collect(0.0:0.5:2.0)
            y = x .^ 3
            bc = BCPair(Deriv2(0.0), Deriv2(12.0))  # f''(0)=0, f''(2)=12 for x³
            itp = cubic_interp(x, y; bc=bc)

            ε = 1e-8
            # At x=1: f'(1) = 3*1² = 3
            d_before = itp(1.0 - ε; deriv=1)
            d_after = itp(1.0 + ε; deriv=1)
            d_at = itp(1.0; deriv=1)

            # All should be approximately equal (C1 continuity)
            @test d_before ≈ d_at atol=1e-4
            @test d_after ≈ d_at atol=1e-4
        end
    end

end # Derivative Boundary Behavior

# ========================================
# Group 6: Type Stability
# ========================================
@testset "Derivative Type Stability" begin

    @testset "Cubic derivative type inference" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Scalar queries
        @test @inferred(itp(0.5; deriv=1)) isa Float64
        @test @inferred(itp(0.5; deriv=2)) isa Float64

        # With different input type (converts)
        @test @inferred(itp(0.5f0; deriv=1)) isa Float64
        @test @inferred(itp(0.5f0; deriv=2)) isa Float64

        # Vector queries
        x_query = [0.25, 0.5, 0.75]
        @test @inferred(itp(x_query; deriv=1)) isa Vector{Float64}
        @test @inferred(itp(x_query; deriv=2)) isa Vector{Float64}
    end

    @testset "Linear derivative type inference" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 4.0]
        itp = linear_interp(x, y)

        # Scalar queries
        @test @inferred(itp(0.5; deriv=1)) isa Float64
        @test @inferred(itp(0.5; deriv=2)) isa Float64

        # With different input type
        @test @inferred(itp(0.5f0; deriv=1)) isa Float64
        @test @inferred(itp(0.5f0; deriv=2)) isa Float64

        # Vector queries
        x_query = [0.25, 0.5, 1.5]
        @test @inferred(itp(x_query; deriv=1)) isa Vector{Float64}
        @test @inferred(itp(x_query; deriv=2)) isa Vector{Float64}
    end

    @testset "Float32 type preservation" begin
        x = Float32.(collect(0.0f0:0.1f0:1.0f0))
        y = x .^ 2

        # Cubic
        itp_cubic = cubic_interp(x, y)
        @test @inferred(itp_cubic(0.5f0; deriv=1)) isa Float32
        @test @inferred(itp_cubic(0.5f0; deriv=2)) isa Float32

        # Linear
        itp_linear = linear_interp(x, y)
        @test @inferred(itp_linear(0.5f0; deriv=1)) isa Float32
        @test @inferred(itp_linear(0.5f0; deriv=2)) isa Float32
    end

    @testset "Order parameter type inference" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2

        # Cubic with deriv
        @test @inferred(cubic_interp(x, y, 0.5; deriv=0)) isa Float64
        @test @inferred(cubic_interp(x, y, 0.5; deriv=1)) isa Float64
        @test @inferred(cubic_interp(x, y, 0.5; deriv=2)) isa Float64

        # Linear with deriv
        @test @inferred(linear_interp(x, y, 0.5; deriv=0)) isa Float64
        @test @inferred(linear_interp(x, y, 0.5; deriv=1)) isa Float64
        @test @inferred(linear_interp(x, y, 0.5; deriv=2)) isa Float64
    end

    @testset "Cache-based type inference" begin
        x = collect(0.0:0.1:1.0)
        cache = CubicSplineCache(x)
        y = x .^ 2

        @test @inferred(cubic_interp(cache, y, 0.5; deriv=0)) isa Float64
        @test @inferred(cubic_interp(cache, y, 0.5; deriv=1)) isa Float64
        @test @inferred(cubic_interp(cache, y, 0.5; deriv=2)) isa Float64
    end

end # Derivative Type Stability

# ========================================
# Group 7: Edge Cases
# ========================================
@testset "Derivative Edge Cases" begin

    @testset "Very small grid (minimum size)" begin
        # Minimum for cubic: 3 points (can we still get derivatives?)
        x = [0.0, 0.5, 1.0]
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Should work without errors
        @test itp(0.25; deriv=1) isa Float64
        @test itp(0.25; deriv=2) isa Float64

        # Linear minimum: 2 points
        x_lin = [0.0, 1.0]
        y_lin = [0.0, 2.0]
        itp_lin = linear_interp(x_lin, y_lin)

        @test itp_lin(0.5; deriv=1) ≈ 2.0  # slope
        @test itp_lin(0.5; deriv=2) ≈ 0.0  # always zero
    end

    @testset "Query at domain boundaries" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Exactly at left boundary
        @test itp(0.0; deriv=1) isa Float64
        @test itp(0.0; deriv=2) isa Float64

        # Exactly at right boundary
        @test itp(1.0; deriv=1) isa Float64
        @test itp(1.0; deriv=2) isa Float64
    end

    @testset "Constant function" begin
        x = collect(0.0:0.1:1.0)
        y = ones(length(x)) * 5.0  # f(x) = 5

        itp_cubic = cubic_interp(x, y)
        @test itp_cubic(0.5; deriv=1) ≈ 0.0 atol=1e-10
        @test itp_cubic(0.5; deriv=2) ≈ 0.0 atol=1e-10

        itp_linear = linear_interp(x, y)
        @test itp_linear(0.5; deriv=1) ≈ 0.0 atol=1e-10
        @test itp_linear(0.5; deriv=2) ≈ 0.0 atol=1e-10
    end

    @testset "Linear function" begin
        x = collect(0.0:0.1:1.0)
        y = 2.0 .* x .+ 3.0  # f(x) = 2x + 3

        # Cubic should reproduce linear exactly
        itp_cubic = cubic_interp(x, y)
        @test itp_cubic(0.5; deriv=1) ≈ 2.0 atol=1e-10
        @test itp_cubic(0.5; deriv=2) ≈ 0.0 atol=1e-10

        # Linear should be exact
        itp_linear = linear_interp(x, y)
        @test itp_linear(0.5; deriv=1) ≈ 2.0 atol=1e-10
        @test itp_linear(0.5; deriv=2) ≈ 0.0 atol=1e-10
    end

    @testset "Non-uniform grid" begin
        # Non-uniform spacing
        x = [0.0, 0.1, 0.15, 0.5, 0.9, 1.0]
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Should still work reasonably
        @test itp(0.5; deriv=1) ≈ 1.0 atol=0.1  # f'(0.5) = 2*0.5 = 1
        @test itp(0.5; deriv=2) ≈ 2.0 atol=0.2  # f''(x) = 2
    end

    @testset "Large grid" begin
        x = collect(range(0.0, 10.0, 1001))
        y = sin.(x)
        itp = cubic_interp(x, y)

        # Should handle large grids efficiently
        @test itp(5.0; deriv=1) ≈ cos(5.0) atol=1e-3
        @test itp(5.0; deriv=2) ≈ -sin(5.0) atol=1e-3

        # Note: Allocation tests are in the dedicated "Derivative Allocations" section
    end

end # Derivative Edge Cases

# ========================================
# Group 8: Allocation Tests
# ========================================
@testset "Derivative Allocations" begin

    @testset "Cubic allocation with deriv" begin
        x = collect(0.0:0.1:1.0)
        cache = CubicSplineCache(x; bc=BCPair(Deriv2(2.0), Deriv2(2.0)))
        y = x .^ 2
        xi = 0.5

        # Warm-up
        cubic_interp(cache, y, xi; deriv=0)
        cubic_interp(cache, y, xi; deriv=1)
        cubic_interp(cache, y, xi; deriv=2)

        # Check allocations (scalar query should be zero-allocation)
        alloc0 = @allocated cubic_interp(cache, y, xi; deriv=0)
        alloc1 = @allocated cubic_interp(cache, y, xi; deriv=1)
        alloc2 = @allocated cubic_interp(cache, y, xi; deriv=2)

        @test alloc0 <= DERIV_ALLOC_THRESHOLD
        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD
    end

    @testset "CubicInterpolant deriv keyword allocation" begin
        x = collect(0.0:0.1:1.0)
        y = x .^ 2
        bc = BCPair(Deriv2(2.0), Deriv2(2.0))
        itp = cubic_interp(x, y; bc=bc)

        # Warm-up
        itp(0.5; deriv=1)
        itp(0.5; deriv=2)

        alloc1 = @allocated itp(0.5; deriv=1)
        alloc2 = @allocated itp(0.5; deriv=2)

        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD
    end

    @testset "CubicInterpolant deriv keyword allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Warmup
        for _ in 1:5
            itp(0.5; deriv=0)
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)
        end

        @test @allocated(itp(0.5; deriv=0)) <= DERIV_ALLOC_THRESHOLD
        @test @allocated(itp(0.5; deriv=1)) <= DERIV_ALLOC_THRESHOLD
        @test @allocated(itp(0.5; deriv=2)) <= DERIV_ALLOC_THRESHOLD
    end

    @testset "Linear allocation with deriv" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        xi = 0.5

        # Warm-up
        linear_interp(x, y, xi; deriv=0)
        linear_interp(x, y, xi; deriv=1)
        linear_interp(x, y, xi; deriv=2)

        # Check allocations (scalar query should be zero-allocation)
        alloc0 = @allocated linear_interp(x, y, xi; deriv=0)
        alloc1 = @allocated linear_interp(x, y, xi; deriv=1)
        alloc2 = @allocated linear_interp(x, y, xi; deriv=2)

        @test alloc0 <= DERIV_ALLOC_THRESHOLD
        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD
    end

    @testset "LinearInterpolant deriv keyword allocation" begin
        x = [0.0, 1.0, 3.0]
        y = [0.0, 2.0, 4.0]
        itp = linear_interp(x, y)

        # Warm-up
        itp(0.5; deriv=1)
        itp(0.5; deriv=2)

        alloc1 = @allocated itp(0.5; deriv=1)
        alloc2 = @allocated itp(0.5; deriv=2)

        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD
    end

    @testset "LinearInterpolant deriv keyword allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        litp = linear_interp(x, y)

        # Warmup
        for _ in 1:5
            litp(0.5; deriv=0)
            litp(0.5; deriv=1)
            litp(0.5; deriv=2)
        end

        @test @allocated(litp(0.5; deriv=0)) <= DERIV_ALLOC_THRESHOLD
        @test @allocated(litp(0.5; deriv=1)) <= DERIV_ALLOC_THRESHOLD
        @test @allocated(litp(0.5; deriv=2)) <= DERIV_ALLOC_THRESHOLD
    end

    @testset "Function-wrapped allocation tests" begin
        # Function-wrapped tests for type stability
        function test_deriv1_alloc(itp, xi::T) where {T}
            itp(xi; deriv=1)
        end

        function test_deriv2_alloc(itp, xi::T) where {T}
            itp(xi; deriv=2)
        end

        function test_deriv_alloc(cache, y, xi, deriv::Int)
            cubic_interp(cache, y, xi; deriv=deriv)
        end

        @testset "Function-wrapped CubicInterpolant derivatives" begin
            x = collect(range(0.0, 1.0, 51))
            y = x .^ 2
            itp = cubic_interp(x, y)

            # Multiple warmup iterations
            for _ in 1:5
                test_deriv1_alloc(itp, 0.5)
                test_deriv2_alloc(itp, 0.5)
            end

            # Test at multiple query points
            for xi in [0.1, 0.25, 0.5, 0.75, 0.9]
                alloc1 = @allocated test_deriv1_alloc(itp, xi)
                alloc2 = @allocated test_deriv2_alloc(itp, xi)

                @test alloc1 <= DERIV_ALLOC_THRESHOLD
                @test alloc2 <= DERIV_ALLOC_THRESHOLD
            end
        end

        @testset "Function-wrapped cubic_interp with deriv" begin
            x = collect(range(0.0, 1.0, 51))
            cache = CubicSplineCache(x)
            y = x .^ 3

            # Multiple warmup
            for _ in 1:5
                test_deriv_alloc(cache, y, 0.5, 0)
                test_deriv_alloc(cache, y, 0.5, 1)
                test_deriv_alloc(cache, y, 0.5, 2)
            end

            # All deriv values should be zero-allocation
            for deriv in 0:2
                for xi in [0.25, 0.5, 0.75]
                    allocs = @allocated test_deriv_alloc(cache, y, xi, deriv)
                    @test allocs <= DERIV_ALLOC_THRESHOLD
                end
            end
        end
    end

    @testset "Derivative stress test - repeated calls" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        # Warmup
        for _ in 1:10
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)
        end

        # 100 repeated calls should all be zero-allocation
        total_alloc1 = 0
        total_alloc2 = 0
        for _ in 1:100
            total_alloc1 += @allocated itp(0.5; deriv=1)
            total_alloc2 += @allocated itp(0.5; deriv=2)
        end

        @test total_alloc1 <= DERIV_ALLOC_THRESHOLD * 100
        @test total_alloc2 <= DERIV_ALLOC_THRESHOLD * 100
    end

    @testset "Float32 derivative allocation" begin
        x = Float32.(collect(range(0.0f0, 1.0f0, 51)))
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Warmup
        for _ in 1:5
            itp(0.5f0; deriv=1)
            itp(0.5f0; deriv=2)
        end

        alloc1 = @allocated itp(0.5f0; deriv=1)
        alloc2 = @allocated itp(0.5f0; deriv=2)

        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD
    end

    @testset "Derivative with different BCs allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2

        bc_types = [
            NaturalBC(),
            ClampedBC(),
            BCPair(Deriv1(1.0), Deriv1(1.0)),
            BCPair(Deriv2(2.0), Deriv2(0.0)),
        ]

        for bc in bc_types
            itp = cubic_interp(x, y; bc=bc)

            # Warmup
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)

            alloc1 = @allocated itp(0.5; deriv=1)
            alloc2 = @allocated itp(0.5; deriv=2)

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
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)

            alloc1 = @allocated itp(0.5; deriv=1)
            alloc2 = @allocated itp(0.5; deriv=2)

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
            itp(1.0; deriv=1)
            itp(1.0; deriv=2)
        end

        alloc1 = @allocated itp(1.0; deriv=1)
        alloc2 = @allocated itp(1.0; deriv=2)

        @test alloc1 <= DERIV_ALLOC_THRESHOLD
        @test alloc2 <= DERIV_ALLOC_THRESHOLD

        # Query outside domain (wraps)
        alloc1_wrap = @allocated itp(7.0; deriv=1)
        alloc2_wrap = @allocated itp(7.0; deriv=2)

        @test alloc1_wrap <= DERIV_ALLOC_THRESHOLD
        @test alloc2_wrap <= DERIV_ALLOC_THRESHOLD
    end

end # Derivative Allocations

# ========================================
# Group 9: Comprehensive Path Coverage
# ========================================
@testset "Derivative Comprehensive Coverage" begin

    @testset "All cubic paths zero-allocation" begin
        # Test all combinations of BC, extrap, and deriv
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2

        bc_types = [
            (NaturalBC(), "Natural"),
            (ClampedBC(), "Clamped"),
            (BCPair(Deriv1(0.5), Deriv1(1.5)), "Deriv1-Deriv1"),
            (BCPair(Deriv2(2.0), Deriv2(2.0)), "Deriv2-Deriv2"),
            (BCPair(Deriv1(0.5), Deriv2(2.0)), "Deriv1-Deriv2"),
        ]

        extrap_modes = [:none, :constant, :extension]

        for (bc, bc_name) in bc_types
            for extrap in extrap_modes
                for deriv in 0:2
                    itp = cubic_interp(x, y; bc=bc, extrap=extrap)

                    # Warmup
                    for _ in 1:3
                        itp(0.5; deriv=deriv)
                    end

                    # Measure
                    alloc = @allocated itp(0.5; deriv=deriv)

                    @test alloc <= DERIV_ALLOC_THRESHOLD
                end
            end
        end
    end

    @testset "Periodic BC all deriv zero-allocation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]
        itp = cubic_interp(x, y; bc=PeriodicBC())

        for deriv in 0:2
            # Warmup
            for _ in 1:3
                itp(1.0; deriv=deriv)
            end

            # Measure
            alloc = @allocated itp(1.0; deriv=deriv)

            @test alloc <= DERIV_ALLOC_THRESHOLD
        end
    end

    @testset "All linear paths zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2

        extrap_modes = [:none, :constant, :extension, :wrap]

        for extrap in extrap_modes
            itp = linear_interp(x, y; extrap=extrap)

            for deriv in 0:2
                # Warmup
                for _ in 1:3
                    itp(0.5; deriv=deriv)
                end

                # Measure
                alloc = @allocated itp(0.5; deriv=deriv)

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
                itp(0.5; deriv=0)
                itp(0.5; deriv=1)
                itp(0.5; deriv=2)
            end

            alloc0 = @allocated itp(0.5; deriv=0)
            alloc1 = @allocated itp(0.5; deriv=1)
            alloc2 = @allocated itp(0.5; deriv=2)

            @test alloc0 <= DERIV_ALLOC_THRESHOLD
            @test alloc1 <= DERIV_ALLOC_THRESHOLD
            @test alloc2 <= DERIV_ALLOC_THRESHOLD
        end
    end

    @testset "Cache-based cubic all deriv" begin
        x = collect(range(0.0, 1.0, 51))

        bc_types = [
            CubicSplineCache(x),
            CubicSplineCache(x; bc=ClampedBC()),
            CubicSplineCache(x; bc=BCPair(Deriv2(2.0), Deriv2(2.0))),
            CubicSplineCache(x; bc=PeriodicBC()),
        ]

        y = x .^ 2

        for cache in bc_types
            for deriv in 0:2
                # Warmup
                for _ in 1:3
                    cubic_interp(cache, y, 0.5; deriv=deriv)
                end

                alloc = @allocated cubic_interp(cache, y, 0.5; deriv=deriv)
                @test alloc <= DERIV_ALLOC_THRESHOLD
            end
        end
    end

end # Derivative Comprehensive Coverage

# ========================================
# Group 10: DerivativeView Wrapper (Phase 3)
# ========================================
@testset "DerivativeView Wrapper" begin

    @testset "DerivativeView wrapper - CubicInterpolant" begin
        x = collect(0.0:0.2:2.0)
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Factory functions return DerivativeView
        d1 = deriv1(itp)
        d2 = deriv2(itp)

        @test d1 isa FastInterpolations.DerivativeView
        @test d2 isa FastInterpolations.DerivativeView

        # Callable equivalence
        @test d1(0.5) ≈ itp(0.5; deriv=1)
        @test d2(0.5) ≈ itp(0.5; deriv=2)

        # Multiple points
        @test d1(0.25) ≈ itp(0.25; deriv=1)
        @test d1(1.5) ≈ itp(1.5; deriv=1)
        @test d2(1.0) ≈ itp(1.0; deriv=2)

        # Real input works (Integer → Float64)
        @test d1(1) ≈ d1(1.0)
        @test d2(1) ≈ d2(1.0)

        # Type stability
        @test @inferred(d1(0.5)) isa Float64
        @test @inferred(d2(0.5)) isa Float64
    end

    @testset "DerivativeView wrapper - LinearInterpolant" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 2.0, 6.0]  # slopes: 2.0, 4.0
        litp = linear_interp(x, y)

        d1 = deriv1(litp)
        d2 = deriv2(litp)

        @test d1 isa FastInterpolations.DerivativeView
        @test d2 isa FastInterpolations.DerivativeView

        # Callable equivalence
        @test d1(0.5) ≈ litp(0.5; deriv=1)
        @test d2(0.5) === litp(0.5; deriv=2)

        # Correct slope values
        @test d1(0.5) ≈ 2.0
        @test d1(1.5) ≈ 4.0
        @test d2(0.5) === 0.0

        # Type stability
        @test @inferred(d1(0.5)) isa Float64
        @test @inferred(d2(0.5)) isa Float64
    end

    @testset "Broadcast idiom" begin
        x = collect(0.0:0.1:2.0)
        y = x .^ 2
        itp = cubic_interp(x, y)
        d1 = deriv1(itp)

        # Broadcast works
        xs = [0.25, 0.5, 0.75, 1.0]
        @test d1.(xs) ≈ [itp(xi; deriv=1) for xi in xs]

        # Fused broadcast works
        @test (@. 2.0 * d1(xs)) ≈ 2.0 .* d1.(xs)
    end

    @testset "DerivativeView wrapper - ND mixed partials" begin
        x = collect(range(0.0, 1.0, 21))
        y = collect(range(0.0, 1.0, 17))
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        dx = deriv_view(itp, (1, 0))
        dxy = deriv_view(itp, (1, 1))

        @test dx isa FastInterpolations.DerivativeView
        @test dxy isa FastInterpolations.DerivativeView

        q = (0.35, 0.6)
        @test dx(q) ≈ itp(q; deriv=(1, 0))
        @test dxy(q) ≈ itp(q; deriv=(1, 1))

        pts = [(0.1, 0.2), (0.3, 0.4), (0.7, 0.9)]
        @test dx.(pts) ≈ [itp(p; deriv=(1, 0)) for p in pts]
        @test dxy.(pts) ≈ [itp(p; deriv=(1, 1)) for p in pts]

        @test_throws ArgumentError dx(q; deriv=(0, 0))
    end

    @testset "DerivativeView wrapper - ND int order" begin
        x = collect(range(0.0, 1.0, 21))
        y = collect(range(0.0, 1.0, 17))
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        d1 = deriv_view(itp, 1)
        d2 = deriv_view(itp, 2)

        @test d1 isa FastInterpolations.DerivativeView
        @test d2 isa FastInterpolations.DerivativeView

        q = (0.25, 0.6)
        @test d1(q) ≈ itp(q; deriv=1)
        @test d1(q) ≈ itp(q; deriv=(1, 1))
        @test d2(q) ≈ itp(q; deriv=2)
        @test d2(q) ≈ itp(q; deriv=(2, 2))
    end

    @testset "DerivativeView wrapper - ND deriv1/2/3 errors" begin
        x = collect(range(0.0, 1.0, 21))
        y = collect(range(0.0, 1.0, 17))
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data)

        @test_throws ArgumentError deriv1(itp)
        @test_throws ArgumentError deriv2(itp)
        @test_throws ArgumentError deriv3(itp)
    end

    @testset "DerivativeView wrapper - deriv_view on 1D" begin
        x = collect(0.0:0.2:2.0)
        y = x .^ 2
        itp = cubic_interp(x, y)

        d1 = deriv_view(itp, 1)
        d2 = deriv_view(itp, 2)

        @test d1 isa FastInterpolations.DerivativeView
        @test d2 isa FastInterpolations.DerivativeView

        @test d1(0.5) ≈ itp(0.5; deriv=1)
        @test d2(0.5) ≈ itp(0.5; deriv=2)
    end

    @testset "DerivativeView allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        itp = cubic_interp(x, y)

        # Warmup
        d1 = deriv1(itp)
        d1(0.5)

        # Wrapper creation should be cheap (struct with single field)
        @test @allocated(deriv1(itp)) <= DERIV_ALLOC_THRESHOLD

        # Scalar call should be zero-allocation
        @test @allocated(d1(0.5)) <= DERIV_ALLOC_THRESHOLD
    end

    @testset "DerivativeView with different BCs" begin
        x = collect(range(0.0, 1.0, 21))
        y = sin.(x)

        # Non-periodic BCs work with regular sin data
        for bc in [NaturalBC(), ClampedBC()]
            itp = cubic_interp(x, y; bc=bc)
            d1 = deriv1(itp)
            d2 = deriv2(itp)

            @testset "BC=$(typeof(bc).name.name)" begin
                # Should be callable
                @test d1(0.5) isa Float64
                @test d2(0.5) isa Float64

                # Should match deriv keyword
                @test d1(0.5) == itp(0.5; deriv=1)
                @test d2(0.5) == itp(0.5; deriv=2)
            end
        end

        # PeriodicBC requires y[1] ≈ y[end], use full period of sin
        @testset "BC=PeriodicBC" begin
            x_periodic = collect(range(0.0, 2π, 51))
            y_periodic = sin.(x_periodic)  # sin(0) = sin(2π) = 0

            itp = cubic_interp(x_periodic, y_periodic; bc=PeriodicBC())
            d1 = deriv1(itp)
            d2 = deriv2(itp)

            # Should be callable
            @test d1(π/4) isa Float64
            @test d2(π/4) isa Float64

            # Should match deriv keyword
            @test d1(π/4) == itp(π/4; deriv=1)
            @test d2(π/4) == itp(π/4; deriv=2)
        end
    end
end # DerivativeView Wrapper

# ========================================
# Group 11: Deriv=3 Extensions
# ========================================
@testset "Deriv=3 Extensions" begin

    @testset "EvalDeriv3 core type" begin
        @test isdefined(FastInterpolations, :EvalDeriv3)
        @test FastInterpolations.EvalDeriv3 <: FastInterpolations.AbstractEvalOp
        @test FastInterpolations.EvalDeriv3() isa FastInterpolations.AbstractEvalOp
    end

    @testset "@_dispatch_deriv handles deriv=3" begin
        result = FastInterpolations.@_dispatch_deriv 3 => op begin
            op
        end
        @test result isa FastInterpolations.EvalDeriv3

        # Invalid deriv values should throw
        @test_throws ArgumentError begin
            FastInterpolations.@_dispatch_deriv 4 => op begin
                op
            end
        end
    end

    @testset "Cubic kernel - S'''(x) = (zR - zL) / h" begin
        zL, zR = 1.0, 3.0
        yL, yR = 0.0, 1.0
        h = 0.5
        inv_h = 1.0 / h
        dL, dR = 0.2, 0.3

        result = FastInterpolations._cubic_kernel(
            FastInterpolations.EvalDeriv3(), zL, zR, yL, yR, h, inv_h, dL, dR
        )

        expected = (zR - zL) / h
        @test result ≈ expected

        # Result should be constant within interval
        result1 = FastInterpolations._cubic_kernel(
            FastInterpolations.EvalDeriv3(), 2.0, 5.0, 0.0, 0.0, 0.1, 10.0, 0.02, 0.08
        )
        result2 = FastInterpolations._cubic_kernel(
            FastInterpolations.EvalDeriv3(), 2.0, 5.0, 0.0, 0.0, 0.1, 10.0, 0.05, 0.05
        )
        @test result1 ≈ result2
    end

    @testset "Lower-order kernels return zero" begin
        # Linear kernel (h=0.5, dL=0.2)
        @test FastInterpolations._linear_kernel(
            FastInterpolations.EvalDeriv3(), 1.0, 5.0, 0.5, 0.2
        ) === zero(Float64)

        # Quadratic kernel
        @test FastInterpolations._quadratic_kernel(
            FastInterpolations.EvalDeriv3(), 1.0, 2.0, 3.0, 0.5
        ) === zero(Float64)

        # Constant kernel
        @test FastInterpolations._constant_kernel(
            FastInterpolations.EvalDeriv3(), 5.0, 5.0, 0.5, 0.2, Val(:left)
        ) === zero(Float64)
    end

    @testset "Anchor weight computation for deriv=3" begin
        FI = FastInterpolations
        h, inv_h = 0.1, 10.0
        dL, dR = 0.03, 0.07

        w3 = FI._compute_anchor_weights(FI.EvalDeriv3(), h, inv_h, dL, dR)

        # Optimized: w3 is now (wzL, wzR) - removed zero y-weights
        @test w3 isa NTuple{2, Float64}
        @test w3[1] === -10.0  # wzL
        @test w3[2] === 10.0   # wzR
    end

    @testset "Numerical validation - f(x) = x³, f'''(x) = 6" begin
        x = collect(range(0.0, 1.0, 101))
        y = x.^3
        bc = BCPair(Deriv2(0.0), Deriv2(6.0))
        itp = cubic_interp(x, y; bc=bc)

        for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
            @test itp(xq; deriv=3) ≈ 6.0 atol=1e-8
        end
    end

    @testset "Numerical validation - finite difference" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        xq = 0.5
        h = 1e-5

        d2_plus = itp(xq + h; deriv=2)
        d2_minus = itp(xq - h; deriv=2)
        fd_approx = (d2_plus - d2_minus) / (2h)

        analytical = itp(xq; deriv=3)

        @test analytical ≈ fd_approx rtol=1e-4
    end

    @testset "Constant within interval property" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        # Third derivative should be constant within each interval
        for i in 1:length(x)-1
            mid1 = x[i] + 0.25 * (x[i+1] - x[i])
            mid2 = x[i] + 0.50 * (x[i+1] - x[i])
            mid3 = x[i] + 0.75 * (x[i+1] - x[i])

            val1 = itp(mid1; deriv=3)
            val2 = itp(mid2; deriv=3)
            val3 = itp(mid3; deriv=3)

            @test val1 ≈ val2 ≈ val3
        end
    end

    @testset "deriv3() factory function" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        d3 = deriv3(itp)

        @test d3 isa FastInterpolations.DerivativeView{3}
        @test d3.parent === itp
        @test d3(0.5) == itp(0.5; deriv=3)
    end

    @testset "Lower-order interpolants return zero for deriv=3" begin
        x = collect(range(0.0, 1.0, 11))

        # Linear
        litp = linear_interp(x, 2.0 .* x)
        @test litp(0.5; deriv=3) === 0.0
        @test deriv3(litp)(0.5) === 0.0

        # Quadratic
        qitp = quadratic_interp(x, x.^2)
        @test qitp(0.5; deriv=3) === 0.0
        @test deriv3(qitp)(0.5) === 0.0

        # Constant
        citp = constant_interp(x, fill(5.0, length(x)))
        @test citp(0.5; deriv=3) === 0.0
        @test deriv3(citp)(0.5) === 0.0
    end

    @testset "Extrapolation modes with deriv=3" begin
        x = collect(range(0.0, 1.0, 11))
        y = x.^3

        # Constant extrapolation
        itp_const = cubic_interp(x, y; extrap=:constant)
        @test itp_const(-0.5; deriv=3) === 0.0
        @test itp_const(1.5; deriv=3) === 0.0

        # Extension extrapolation
        itp_ext = cubic_interp(x, y; extrap=:extension)
        val_below = itp_ext(-0.5; deriv=3)
        val_first = itp_ext(0.05; deriv=3)
        @test val_below ≈ val_first

        # None throws
        itp_none = cubic_interp(x, y; extrap=:none)
        @test_throws DomainError itp_none(-0.5; deriv=3)
    end

    @testset "Linear/Constant oneshot deriv=3 with constant extrapolation" begin
        # This tests the _linear_eval_constant_extrap and _constant_eval_extrap
        # dispatch for EvalDeriv3, which returns zero outside domain
        x = collect(range(0.0, 1.0, 11))
        y_linear = 2.0 .* x
        y_const = fill(5.0, length(x))

        # Linear interpolation: deriv=3 with :constant extrap outside domain
        @test linear_interp(x, y_linear, -0.5; extrap=:constant, deriv=3) === 0.0
        @test linear_interp(x, y_linear, 1.5; extrap=:constant, deriv=3) === 0.0

        # Constant interpolation: deriv=3 with :constant extrap outside domain
        @test constant_interp(x, y_const, -0.5; extrap=:constant, deriv=3) === 0.0
        @test constant_interp(x, y_const, 1.5; extrap=:constant, deriv=3) === 0.0
    end

    @testset "Type stability for deriv=3" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        @test @inferred(itp(0.5; deriv=3)) isa Float64
        @test @inferred(deriv3(itp)) isa FastInterpolations.DerivativeView{3}
        @test @inferred(deriv3(itp)(0.5)) isa Float64
    end

    @testset "Zero-allocation for deriv=3" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        # Warmup
        itp(0.5; deriv=3)
        d3 = deriv3(itp)
        d3(0.5)

        # Measure
        alloc_direct = @allocated itp(0.5; deriv=3)
        alloc_view = @allocated d3(0.5)

        @test alloc_direct <= DERIV_ALLOC_THRESHOLD
        @test alloc_view <= DERIV_ALLOC_THRESHOLD
    end

end # Deriv=3 Extensions

# ========================================
# Group 12: DerivativeView Vector Queries
# ========================================
@testset "DerivativeView Vector Queries" begin

    @testset "DerivativeView accepts vector queries - Cubic" begin
        x = collect(range(0.0, 1.0, 101))
        y = x.^2
        bc = BCPair(Deriv2(2.0), Deriv2(2.0))
        itp = cubic_interp(x, y; bc=bc)

        d1 = deriv1(itp)
        d2 = deriv2(itp)
        d3 = deriv3(itp)

        x_query = [0.1, 0.5, 0.9]

        # Direct vector query (not broadcast)
        vals1 = d1(x_query)
        vals2 = d2(x_query)
        vals3 = d3(x_query)

        @test vals1 isa Vector{Float64}
        @test vals2 isa Vector{Float64}
        @test vals3 isa Vector{Float64}

        # Should match individual queries
        @test vals1 ≈ [d1(xq) for xq in x_query]
        @test vals2 ≈ [d2(xq) for xq in x_query]
        @test vals3 ≈ [d3(xq) for xq in x_query]

        # Should match itp(x_query; deriv=N)
        @test vals1 ≈ itp(x_query; deriv=1)
        @test vals2 ≈ itp(x_query; deriv=2)
        @test vals3 ≈ itp(x_query; deriv=3)
    end

    @testset "DerivativeView accepts vector queries - Linear" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 2.0, 6.0]
        itp = linear_interp(x, y)

        d1 = deriv1(itp)
        d2 = deriv2(itp)
        d3 = deriv3(itp)

        x_query = [0.5, 1.5]

        vals1 = d1(x_query)
        vals2 = d2(x_query)
        vals3 = d3(x_query)

        @test vals1 ≈ itp(x_query; deriv=1)
        @test vals2 ≈ itp(x_query; deriv=2)
        @test vals3 ≈ itp(x_query; deriv=3)
    end

    @testset "DerivativeView vector query type stability" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2
        itp = cubic_interp(x, y)

        d1 = deriv1(itp)
        x_query = [0.25, 0.5, 0.75]

        @test @inferred(d1(x_query)) isa Vector{Float64}
    end

end # DerivativeView Vector Queries

# ========================================
# Group 13: SeriesInterpolant Derivatives
# ========================================
@testset "SeriesInterpolant Derivatives" begin

    @testset "CubicSeriesInterpolant with deriv keyword - scalar" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        # Scalar queries with deriv=0,1,2,3
        vals0 = sitp(0.5; deriv=0)
        vals1 = sitp(0.5; deriv=1)
        vals2 = sitp(0.5; deriv=2)
        vals3 = sitp(0.5; deriv=3)

        @test length(vals0) == 2
        @test length(vals1) == 2
        @test length(vals2) == 2
        @test length(vals3) == 2

        @test vals0 isa Vector{Float64}
        @test vals1 isa Vector{Float64}
        @test vals2 isa Vector{Float64}
        @test vals3 isa Vector{Float64}
    end

    @testset "CubicSeriesInterpolant with deriv keyword - vector" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        x_query = [0.1, 0.5, 0.9]

        # Vector queries with deriv=0,1,2,3
        results0 = sitp(x_query; deriv=0)
        results1 = sitp(x_query; deriv=1)
        results2 = sitp(x_query; deriv=2)
        results3 = sitp(x_query; deriv=3)

        @test length(results0) == 2
        @test length(results1) == 2
        @test length(results2) == 2
        @test length(results3) == 2

        @test length(results0[1]) == 3
        @test length(results1[1]) == 3
        @test length(results2[1]) == 3
        @test length(results3[1]) == 3
    end

    @testset "CubicSeriesInterpolant in-place with deriv keyword" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        # Scalar in-place
        out_scalar = similar([0.0, 0.0])
        sitp(out_scalar, 0.5; deriv=1)
        @test out_scalar ≈ sitp(0.5; deriv=1)

        # Vector in-place
        x_query = [0.1, 0.5, 0.9]
        outputs = [similar(x_query) for _ in 1:2]
        sitp(outputs, x_query; deriv=1)

        results = sitp(x_query; deriv=1)
        @test outputs[1] ≈ results[1]
        @test outputs[2] ≈ results[2]
    end

    @testset "LinearSeriesInterpolant with deriv keyword" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = 2.0 .* x
        y2 = 3.0 .* x
        sitp = linear_interp(x, [y1, y2])

        # deriv=0: values
        vals0 = sitp(0.5; deriv=0)
        @test vals0[1] ≈ 1.0
        @test vals0[2] ≈ 1.5

        # deriv=1: slopes
        vals1 = sitp(0.5; deriv=1)
        @test vals1[1] ≈ 2.0
        @test vals1[2] ≈ 3.0

        # deriv=2,3: zero
        vals2 = sitp(0.5; deriv=2)
        vals3 = sitp(0.5; deriv=3)
        @test all(v === 0.0 for v in vals2)
        @test all(v === 0.0 for v in vals3)
    end

    @testset "SeriesInterpolant extrapolation with deriv keyword" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = x.^3
        y2 = x.^2
        sitp = cubic_interp(x, [y1, y2]; extrap=:constant)

        # Outside domain with deriv=3
        vals_below = sitp(-0.5; deriv=3)
        vals_above = sitp(1.5; deriv=3)

        @test vals_below[1] === 0.0
        @test vals_below[2] === 0.0
        @test vals_above[1] === 0.0
        @test vals_above[2] === 0.0
    end

    @testset "SeriesInterpolant + deriv1/deriv2/deriv3 factories" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        # Factory functions should work
        d1 = deriv1(sitp)
        d2 = deriv2(sitp)
        d3 = deriv3(sitp)

        @test d1 isa FastInterpolations.DerivativeView{1}
        @test d2 isa FastInterpolations.DerivativeView{2}
        @test d3 isa FastInterpolations.DerivativeView{3}

        # Scalar evaluation
        vals1 = d1(0.5)
        vals2 = d2(0.5)
        vals3 = d3(0.5)

        @test vals1 ≈ sitp(0.5; deriv=1)
        @test vals2 ≈ sitp(0.5; deriv=2)
        @test vals3 ≈ sitp(0.5; deriv=3)
    end

    @testset "SeriesInterpolant DerivativeView vector queries" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = x.^2
        y2 = x.^3
        sitp = cubic_interp(x, [y1, y2])

        d1 = deriv1(sitp)
        x_query = [0.25, 0.5, 0.75]

        # Direct vector query
        results = d1(x_query)

        @test length(results) == 2
        @test length(results[1]) == 3
        @test length(results[2]) == 3

        # Should match itp(x_query; deriv=1)
        expected = sitp(x_query; deriv=1)
        @test results[1] ≈ expected[1]
        @test results[2] ≈ expected[2]
    end

    @testset "SeriesInterpolant DerivativeView broadcasting" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        d1 = deriv1(sitp)
        x_query = [0.25, 0.5, 0.75]

        # Broadcasting should work (but returns array of vectors, not vector of arrays)
        results_bc = d1.(x_query)

        @test length(results_bc) == 3
        @test all(r -> length(r) == 2, results_bc)
    end

    @testset "SeriesInterpolant DerivativeView in-place scalar query" begin
        # Covers: (d::DerivativeView)(out::AbstractArray{<:Real}, xq::Real)
        x = collect(range(0.0, 1.0, 101))
        y1 = x.^2
        y2 = x.^3
        sitp = cubic_interp(x, [y1, y2])

        d1 = deriv1(sitp)
        d2 = deriv2(sitp)

        # In-place scalar query: d(out, xq)
        out1 = zeros(2)
        d1(out1, 0.5)
        @test out1 ≈ sitp(0.5; deriv=1)

        out2 = zeros(2)
        d2(out2, 0.5)
        @test out2 ≈ sitp(0.5; deriv=2)
    end

    @testset "SeriesInterpolant DerivativeView in-place vector query" begin
        # Covers: (d::DerivativeView)(out::AbstractArray{<:AbstractArray{<:Real}}, xq::AbstractArray{<:Real})
        x = collect(range(0.0, 1.0, 101))
        y1 = x.^2
        y2 = x.^3
        sitp = cubic_interp(x, [y1, y2])

        d1 = deriv1(sitp)
        x_query = [0.25, 0.5, 0.75]

        # In-place vector query: d(out, xq_vec)
        out = [zeros(3), zeros(3)]  # Vector of vectors
        d1(out, x_query)

        expected = sitp(x_query; deriv=1)
        @test out[1] ≈ expected[1]
        @test out[2] ≈ expected[2]
    end

    @testset "SeriesInterpolant deriv keyword type stability" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        @test @inferred(sitp(0.5; deriv=1)) isa Vector{Float64}
        @test @inferred(sitp([0.1, 0.5]; deriv=1)) isa Vector{Vector{Float64}}
    end

    @testset "SeriesInterpolant zero-allocation for scalar queries" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        # Warmup
        out = sitp(0.5; deriv=1)
        output = similar(out)
        sitp(output, 0.5; deriv=1)

        # In-place should be zero-allocation
        alloc = @allocated sitp(output, 0.5; deriv=1)
        @test alloc <= DERIV_ALLOC_THRESHOLD
    end

end # SeriesInterpolant Derivatives

# ========================================
# DerivativeView search/hint keywords and in-place vector
# ========================================

@testset "DerivativeView search/hint keywords" begin
    # Shared cubic test data
    x_cubic = collect(range(0.0, 1.0, 51))
    y_cubic = x_cubic.^2
    xq = [0.25, 0.5, 0.75]

    @testset "Cubic - search keyword passthrough" begin
        itp = cubic_interp(x_cubic, y_cubic; search=Binary())
        d1 = deriv1(itp)
        d2 = deriv2(itp)

        # Default uses parent's search_policy
        @test d1(0.5) ≈ itp(0.5; deriv=1)
        @test d2(0.5) ≈ itp(0.5; deriv=2)

        # Explicit search override
        @test d1(0.5; search=Linear()) ≈ itp(0.5; deriv=1, search=Linear())
        @test d1(0.5; search=LinearBinary()) ≈ itp(0.5; deriv=1, search=LinearBinary())
        @test d2(0.5; search=Linear()) ≈ itp(0.5; deriv=2, search=Linear())
    end

    @testset "Cubic - hint keyword passthrough" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        hint = Ref(1)
        result = d1(0.5; hint=hint)
        @test result ≈ itp(0.5; deriv=1)

        # Hint should be updated after call
        @test hint[] >= 1
    end

    @testset "Linear - search and hint keywords" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 2.0, 6.0, 12.0, 20.0]
        itp = linear_interp(x, y)
        d1 = deriv1(itp)

        @test d1(0.5; search=Binary()) ≈ itp(0.5; deriv=1, search=Binary())
        @test d1(1.5; search=Linear()) ≈ itp(1.5; deriv=1, search=Linear())

        hint = Ref(1)
        @test d1(2.5; hint=hint) ≈ itp(2.5; deriv=1)
    end

    @testset "Vector query with search and hint" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        hint = Ref(1)
        result = d1(xq; search=LinearBinary(), hint=hint)
        expected = itp(xq; deriv=1, search=LinearBinary())
        @test result ≈ expected
    end

    @testset "Error on deriv keyword override" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        @test_throws ArgumentError d1(0.5; deriv=2)
        @test_throws ArgumentError d1(xq; deriv=3)
    end
end # DerivativeView search/hint keywords

@testset "DerivativeView single-series in-place vector" begin
    # Shared test data
    x_base = collect(range(0.0, 1.0, 51))
    xq = [0.25, 0.5, 0.75]

    @testset "CubicInterpolant in-place vector" begin
        itp = cubic_interp(x_base, x_base.^2)
        d1 = deriv1(itp)
        d2 = deriv2(itp)

        output1 = zeros(3)
        output2 = zeros(3)

        # In-place vector call
        d1(output1, xq)
        d2(output2, xq)

        @test output1 ≈ itp(xq; deriv=1)
        @test output2 ≈ itp(xq; deriv=2)
    end

    @testset "LinearInterpolant in-place vector" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 2.0, 6.0, 12.0, 20.0]
        itp = linear_interp(x, y)
        d1 = deriv1(itp)

        xq_lin = [0.5, 1.5, 2.5]
        output = zeros(3)
        d1(output, xq_lin)

        @test output ≈ itp(xq_lin; deriv=1)
    end

    @testset "QuadraticInterpolant in-place vector" begin
        itp = quadratic_interp(x_base, x_base.^3)
        d1 = deriv1(itp)

        output = zeros(3)
        d1(output, xq)

        @test output ≈ itp(xq; deriv=1)
    end

    @testset "In-place vector with search and hint" begin
        itp = cubic_interp(x_base, x_base.^2)
        d1 = deriv1(itp)

        output = zeros(3)
        hint = Ref(1)

        d1(output, xq; search=LinearBinary(), hint=hint)
        @test output ≈ itp(xq; deriv=1, search=LinearBinary())
    end

    @testset "In-place vector zero allocation" begin
        itp = cubic_interp(x_base, x_base.^2)
        d1 = deriv1(itp)

        xq_alloc = collect(range(0.1, 0.9, 10))
        output = zeros(10)

        # Warmup
        d1(output, xq_alloc)

        # Should be zero-allocation
        alloc = @allocated d1(output, xq_alloc)
        @test alloc <= DERIV_ALLOC_THRESHOLD
    end

end # DerivativeView single-series in-place vector

@testset "DerivativeView SeriesInterpolant search/hint keywords" begin
    # Shared test data for series interpolant
    x = collect(range(0.0, 1.0, 51))
    sitp = cubic_interp(x, [x.^2, x.^3])
    xq = [0.25, 0.5, 0.75]

    @testset "SeriesInterpolant scalar with search and hint" begin
        d1 = deriv1(sitp)

        # Scalar with search and hint
        @test d1(0.5; search=Binary()) ≈ sitp(0.5; deriv=1, search=Binary())
        @test d1(0.5; search=Linear()) ≈ sitp(0.5; deriv=1, search=Linear())

        hint = Ref(1)
        result = d1(0.5; hint=hint)
        @test result ≈ sitp(0.5; deriv=1)
    end

    @testset "SeriesInterpolant in-place scalar with keywords" begin
        d1 = deriv1(sitp)

        out = zeros(2)
        d1(out, 0.5; search=Linear(), hint=Ref(1))
        @test out ≈ sitp(0.5; deriv=1)
    end

    @testset "SeriesInterpolant in-place vector with keywords" begin
        d1 = deriv1(sitp)

        outputs = [zeros(3), zeros(3)]
        d1(outputs, xq; search=LinearBinary(), hint=Ref(1))

        expected = sitp(xq; deriv=1)
        @test outputs[1] ≈ expected[1]
        @test outputs[2] ≈ expected[2]
    end

end # DerivativeView SeriesInterpolant search/hint keywords

@testset "DerivativeView type stability and performance" begin
    # Shared cubic test data
    x_cubic = collect(range(0.0, 1.0, 51))
    y_cubic = x_cubic.^2
    xq = [0.25, 0.5, 0.75]

    @testset "Type stability - scalar evaluation" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)
        d2 = deriv2(itp)

        # Basic scalar - should infer Float64
        @test @inferred(d1(0.5)) isa Float64
        @test @inferred(d2(0.5)) isa Float64

        # With search keyword
        @test @inferred(d1(0.5; search=Binary())) isa Float64
        @test @inferred(d1(0.5; search=Linear())) isa Float64
        @test @inferred(d1(0.5; search=LinearBinary())) isa Float64

        # With hint keyword
        hint = Ref(1)
        @test @inferred(d1(0.5; hint=hint)) isa Float64

        # With both keywords
        @test @inferred(d1(0.5; search=LinearBinary(), hint=Ref(1))) isa Float64
    end

    @testset "Type stability - vector evaluation" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        # Basic vector
        @test @inferred(d1(xq)) isa Vector{Float64}

        # With keywords
        @test @inferred(d1(xq; search=Binary())) isa Vector{Float64}
        @test @inferred(d1(xq; search=LinearBinary(), hint=Ref(1))) isa Vector{Float64}
    end

    @testset "Type stability - in-place evaluation" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        output = zeros(3)

        # In-place should return the output array
        result = @inferred d1(output, xq)
        @test result === output

        # With keywords
        result2 = @inferred d1(output, xq; search=LinearBinary())
        @test result2 === output
    end

    @testset "Type stability - Linear interpolant" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 2.0, 6.0, 12.0, 20.0]
        itp = linear_interp(x, y)
        d1 = deriv1(itp)

        @test @inferred(d1(0.5)) isa Float64
        @test @inferred(d1(0.5; search=Binary())) isa Float64
        @test @inferred(d1([0.5, 1.5])) isa Vector{Float64}
    end

    @testset "Type stability - SeriesInterpolant" begin
        sitp = cubic_interp(x_cubic, [x_cubic.^2, x_cubic.^3])
        d1 = deriv1(sitp)

        # Scalar returns Vector
        @test @inferred(d1(0.5)) isa Vector{Float64}
        @test @inferred(d1(0.5; search=Binary())) isa Vector{Float64}

        # Vector returns Vector of Vectors
        @test @inferred(d1(xq)) isa Vector{Vector{Float64}}
    end

    @testset "Zero allocation - scalar with kwargs" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        # Warmup
        d1(0.5)
        d1(0.5; search=Binary())
        hint = Ref(1)
        d1(0.5; hint=hint)

        # Test allocations
        alloc_basic = @allocated d1(0.5)
        alloc_search = @allocated d1(0.5; search=Binary())
        alloc_hint = @allocated d1(0.5; hint=hint)

        @test alloc_basic <= DERIV_ALLOC_THRESHOLD
        @test alloc_search <= DERIV_ALLOC_THRESHOLD
        @test alloc_hint <= DERIV_ALLOC_THRESHOLD
    end

    @testset "Zero allocation - in-place with kwargs" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        xq_alloc = collect(range(0.1, 0.9, 10))
        output = zeros(10)
        hint = Ref(1)

        # Warmup
        d1(output, xq_alloc)
        d1(output, xq_alloc; search=Binary())
        d1(output, xq_alloc; hint=hint)
        d1(output, xq_alloc; search=LinearBinary(), hint=hint)

        # Test allocations
        alloc_basic = @allocated d1(output, xq_alloc)
        alloc_search = @allocated d1(output, xq_alloc; search=Binary())
        alloc_hint = @allocated d1(output, xq_alloc; hint=hint)
        alloc_both = @allocated d1(output, xq_alloc; search=LinearBinary(), hint=hint)

        @test alloc_basic <= DERIV_ALLOC_THRESHOLD
        @test alloc_search <= DERIV_ALLOC_THRESHOLD
        @test alloc_hint <= DERIV_ALLOC_THRESHOLD
        @test alloc_both <= DERIV_ALLOC_THRESHOLD
    end

    @testset "kwargs forwarding preserves behavior" begin
        itp = cubic_interp(x_cubic, y_cubic)
        d1 = deriv1(itp)

        # DerivativeView with kwargs should match direct itp call with kwargs
        for search_policy in [Binary(), Linear(), LinearBinary(), HintedBinary()]
            hint = Ref(1)
            hint2 = Ref(1)

            # Scalar
            dv_result = d1(0.5; search=search_policy, hint=hint)
            itp_result = itp(0.5; deriv=1, search=search_policy, hint=hint2)
            @test dv_result ≈ itp_result

            # Vector
            hint = Ref(1)
            hint2 = Ref(1)
            dv_result = d1(xq; search=search_policy, hint=hint)
            itp_result = itp(xq; deriv=1, search=search_policy, hint=hint2)
            @test dv_result ≈ itp_result
        end
    end

end # DerivativeView type stability and performance

@testset "AbstractDerivativeView type hierarchy" begin
    # Create test interpolants
    x = collect(range(0.0, 1.0, 11))
    y = x.^2

    cubic_itp = cubic_interp(x, y)
    linear_itp = linear_interp(x, y)
    quad_itp = quadratic_interp(x, y)
    const_itp = constant_interp(x, y)

    @testset "DerivativeView subtype relationship" begin
        # DerivativeView is a subtype of AbstractDerivativeView
        @test DerivativeView <: AbstractDerivativeView

        # Concrete instances satisfy isa check
        d1_cubic = deriv1(cubic_itp)
        d2_cubic = deriv2(cubic_itp)
        d3_cubic = deriv3(cubic_itp)

        @test d1_cubic isa AbstractDerivativeView
        @test d2_cubic isa AbstractDerivativeView
        @test d3_cubic isa AbstractDerivativeView

        @test d1_cubic isa DerivativeView
        @test d2_cubic isa DerivativeView
        @test d3_cubic isa DerivativeView
    end

    @testset "Partial type parameter dispatch" begin
        # DerivativeView{N} matches any interpolant type
        d1_cubic = deriv1(cubic_itp)
        d1_linear = deriv1(linear_itp)
        d2_cubic = deriv2(cubic_itp)

        # Dispatch on order only (any interpolant)
        @test d1_cubic isa DerivativeView{1}
        @test d1_linear isa DerivativeView{1}
        @test d2_cubic isa DerivativeView{2}

        @test !(d1_cubic isa DerivativeView{2})
        @test !(d2_cubic isa DerivativeView{1})
    end

    @testset "Full type parameter dispatch" begin
        d1_cubic = deriv1(cubic_itp)
        d1_linear = deriv1(linear_itp)

        # Dispatch on order AND interpolant type
        @test d1_cubic isa DerivativeView{1, <:CubicInterpolant}
        @test d1_linear isa DerivativeView{1, <:LinearInterpolant}

        @test !(d1_cubic isa DerivativeView{1, <:LinearInterpolant})
        @test !(d1_linear isa DerivativeView{1, <:CubicInterpolant})
    end

    @testset "All interpolant types work with AbstractDerivativeView" begin
        # All interpolant types produce AbstractDerivativeView
        for (name, itp) in [
            ("Cubic", cubic_itp),
            ("Linear", linear_itp),
            ("Quadratic", quad_itp),
            ("Constant", const_itp)
        ]
            d1 = deriv1(itp)
            d2 = deriv2(itp)
            d3 = deriv3(itp)

            @test d1 isa AbstractDerivativeView
            @test d2 isa AbstractDerivativeView
            @test d3 isa AbstractDerivativeView

            @test d1 isa DerivativeView{1}
            @test d2 isa DerivativeView{2}
            @test d3 isa DerivativeView{3}
        end
    end

    @testset "Type dispatch function example" begin
        # Example: function that accepts any derivative view
        function accepts_any(d::AbstractDerivativeView)
            return :any
        end

        # Example: function specialized for first derivative
        function accepts_first(d::DerivativeView{1})
            return :first
        end

        d1 = deriv1(cubic_itp)
        d2 = deriv2(cubic_itp)

        @test accepts_any(d1) == :any
        @test accepts_any(d2) == :any
        @test accepts_first(d1) == :first
        @test_throws MethodError accepts_first(d2)  # d2 is not DerivativeView{1}
    end
end # AbstractDerivativeView type hierarchy
