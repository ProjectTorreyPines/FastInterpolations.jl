# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    AUTO-DIFFERENTIATION TESTS                              ║
# ║         Tests for ForwardDiff Dual type support in interpolants            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Tests verify that interpolants correctly propagate derivatives through
# ForwardDiff's Dual type for automatic differentiation.
#
# Key insight: For AD to work, the Dual type must flow through:
#   1. Query point input → 2. Interval arithmetic → 3. Output value
#
# The analytical derivative (deriv=DerivOp(1)) should match ForwardDiff.derivative().
#

using Test
using FastInterpolations
using ForwardDiff

const FI = FastInterpolations

@testset "AutoDiff Support" begin

    # ========================================
    # LinearInterpolant (Single Series)
    # ========================================

    @testset "LinearInterpolant with ForwardDiff" begin
        # Linear data: y = 2x + 1
        # Derivative should be exactly 2 everywhere in the interior
        x = collect(0.0:0.5:5.0)
        y = 2.0 .* x .+ 1.0

        itp = linear_interp(x, y; extrap=ExtendExtrap())

        @testset "derivative matches analytical" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                # ForwardDiff derivative
                fd_deriv = ForwardDiff.derivative(itp, xq)

                # Analytical derivative from interpolant
                analytical = itp(xq; deriv=DerivOp(1))

                @test fd_deriv ≈ analytical atol=1e-10
                @test fd_deriv ≈ 2.0 atol=1e-10  # Known slope
            end
        end

        @testset "second derivative works (nested Dual)" begin
            # y = 2x + 1 is linear, so d²/dx² is 0 away from knots.
            xq = 0.25
            fd_d2 = ForwardDiff.derivative(q -> ForwardDiff.derivative(itp, q), xq)
            @test fd_d2 ≈ 0.0 atol=1e-10

            # Also cover Float32-grid interpolants (previously hit Float32(::Dual) errors).
            itp32 = linear_interp(Float32.(x), Float32.(y); extrap=ExtendExtrap())
            fd_d2_32 = ForwardDiff.derivative(q -> ForwardDiff.derivative(itp32, q), xq)
            @test fd_d2_32 ≈ 0.0 atol=1e-6
        end

        @testset "value is preserved" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                fd_value = ForwardDiff.value(ForwardDiff.Dual(xq, 1.0) |> itp)
                direct_value = itp(xq)

                @test fd_value ≈ direct_value atol=1e-10
            end
        end

        @testset "works with nonlinear data" begin
            # Quadratic data: y = x²
            y_quad = x .^ 2
            itp_quad = linear_interp(x, y_quad; extrap=ExtendExtrap())

            # At x=2.0 (grid point), linear interp between x=1.5 and x=2.0
            # or between x=2.0 and x=2.5 depending on interval
            xq = 2.25  # Midpoint of [2.0, 2.5]

            # Linear slope between (2.0, 4.0) and (2.5, 6.25)
            expected_slope = (6.25 - 4.0) / (2.5 - 2.0)  # = 4.5

            fd_deriv = ForwardDiff.derivative(itp_quad, xq)
            analytical = itp_quad(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ expected_slope atol=1e-10
            @test fd_deriv ≈ analytical atol=1e-10
        end

        @testset "one-shot API AD" begin
            # One-shot API should also support ForwardDiff
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> linear_interp(x, y, q), xq)
            analytical = linear_interp(x, y, xq; deriv=DerivOp(1))
            @test fd_deriv ≈ analytical atol=1e-10
            @test fd_deriv ≈ 2.0 atol=1e-10  # Known slope for y = 2x + 1
        end
    end

    # ========================================
    # LinearSeriesInterpolant (Multi Series)
    # ========================================

    @testset "LinearSeriesInterpolant with ForwardDiff" begin
        # Two series: sin and cos
        x = collect(range(0.0, 2π, 21))
        y1 = sin.(x)
        y2 = cos.(x)
        sitp = linear_interp(x, Series(y1, y2); extrap=ExtendExtrap())

        @testset "derivative matches analytical (interior points)" begin
            # Test at interior points only (not at grid points)
            test_points = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

            for xq in test_points
                # ForwardDiff computes derivative of entire vector at once
                fd_derivs = ForwardDiff.derivative(sitp, xq)
                analytical = sitp(xq; deriv=DerivOp(1))

                @test fd_derivs ≈ analytical atol=1e-10
            end
        end

        @testset "value is preserved" begin
            test_points = [0.5, 1.5, 2.5]

            for xq in test_points
                dual_result = sitp(ForwardDiff.Dual(xq, 1.0))
                fd_values = ForwardDiff.value.(dual_result)

                @test fd_values ≈ sitp(xq) atol=1e-10
            end
        end

        @testset "type stability" begin
            xq = ForwardDiff.Dual(1.5, 1.0)
            result = sitp(xq)

            # Output should be vector of Dual
            @test eltype(result) <: ForwardDiff.Dual
        end

        @testset "linear data exact derivative" begin
            # Linear data: y1 = 2x, y2 = -x + 5
            x_lin = collect(0.0:0.5:5.0)
            y1_lin = 2.0 .* x_lin
            y2_lin = -x_lin .+ 5.0
            sitp_lin = linear_interp(x_lin, Series(y1_lin, y2_lin); extrap=ExtendExtrap())

            xq = 2.25
            fd_derivs = ForwardDiff.derivative(sitp_lin, xq)
            analytical = sitp_lin(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-10
            @test fd_derivs ≈ [2.0, -1.0] atol=1e-10  # d/dx(2x)=2, d/dx(-x+5)=-1
        end
    end

    # ========================================
    # Complex-Valued Interpolation
    # ========================================

    @testset "Complex values with ForwardDiff" begin
        x = collect(0.0:0.5:5.0)
        # Complex linear: z = (2 + i)x + (1 - i)
        # Real part: 2x + 1, derivative = 2
        # Imag part: x - 1, derivative = 1
        y_complex = (2.0 + 1.0im) .* x .+ (1.0 - 1.0im)

        itp = linear_interp(x, y_complex; extrap=ExtendExtrap())

        @testset "complex derivative" begin
            xq = 2.25

            # ForwardDiff on complex function
            fd_deriv = ForwardDiff.derivative(itp, xq)

            # Analytical
            analytical = itp(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ analytical atol=1e-10
            @test real(fd_deriv) ≈ 2.0 atol=1e-10  # d/dx(2x+1)
            @test imag(fd_deriv) ≈ 1.0 atol=1e-10  # d/dx(x-1)
        end

        @testset "one-shot API AD with complex" begin
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> linear_interp(x, y_complex, q), xq)
            analytical = linear_interp(x, y_complex, xq; deriv=DerivOp(1))
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    # ========================================
    # Type Stability Tests
    # ========================================

    @testset "type stability" begin
        x = collect(0.0:0.5:5.0)
        y = sin.(x)
        itp = linear_interp(x, y; extrap=ExtendExtrap())

        @testset "Dual output type for Dual input" begin
            xq = ForwardDiff.Dual(2.5, 1.0)
            result = itp(xq)
            @test result isa ForwardDiff.Dual
        end

        @testset "Float64 output for Float64 input" begin
            result = itp(2.5)
            @test result isa Float64
        end

        # NOTE: @inferred may fail due to dynamic dispatch on xq type.
        # The callable accepts `xq` without type constraint to support both
        # Float64 and Dual. This is a trade-off for AD flexibility.
        # The actual computation is still type-stable once dispatched.
    end

    # ========================================
    # Gradient Computation Tests
    # ========================================

    @testset "gradient computation" begin
        x = collect(0.0:0.5:5.0)
        y = x .^ 2  # y = x²

        itp = linear_interp(x, y; extrap=ExtendExtrap())

        @testset "ForwardDiff.gradient on function using interpolant" begin
            # Function that uses interpolation
            function loss(params)
                xq = params[1]
                return itp(xq)^2  # square the interpolated value
            end

            # At xq = 2.25: itp(2.25) ≈ linear interp of x²
            # d/d(xq)[itp(xq)²] = 2 * itp(xq) * itp'(xq)
            params = [2.25]
            grad = ForwardDiff.gradient(loss, params)

            # Manual calculation
            val = itp(2.25)
            deriv = itp(2.25; deriv=DerivOp(1))
            expected_grad = 2 * val * deriv

            @test grad[1] ≈ expected_grad atol=1e-10
        end
    end

    # ========================================
    # Float32 Tests
    # ========================================

    @testset "Float32 support" begin
        x32 = Float32.(collect(0.0:0.5:5.0))
        y32 = Float32.(2.0 .* x32 .+ 1.0)

        itp32 = linear_interp(x32, y32; extrap=ExtendExtrap())

        @testset "Float32 derivative" begin
            xq = 2.25f0
            fd_deriv = ForwardDiff.derivative(itp32, xq)
            analytical = itp32(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ analytical atol=1e-5
            @test fd_deriv isa Float32
        end
    end

    # ========================================
    # ConstantInterpolant with ForwardDiff
    # ========================================

    @testset "ConstantInterpolant with ForwardDiff" begin
        x = collect(0.0:1.0:5.0)
        y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        @testset "interpolant callable AD - derivative is zero" begin
            itp = constant_interp(x, y; side=LeftSide(), extrap=ExtendExtrap())

            # Constant interpolation derivative should be 0 (step function)
            for xq in [0.5, 1.5, 2.5, 3.5]
                fd_deriv = ForwardDiff.derivative(itp, xq)
                analytical = itp(xq; deriv=DerivOp(1))
                @test fd_deriv ≈ 0.0 atol=1e-10
                @test fd_deriv ≈ analytical atol=1e-10
            end
        end

        @testset "one-shot API AD" begin
            # Verify one-shot API works with ForwardDiff
            xq = 2.5
            fd_deriv = ForwardDiff.derivative(q -> constant_interp(x, y, q; side=LeftSide()), xq)
            @test fd_deriv ≈ 0.0 atol=1e-10
        end

        @testset "value is preserved" begin
            itp = constant_interp(x, y; side=LeftSide(), extrap=ExtendExtrap())
            test_points = [0.5, 1.5, 2.5, 3.5]

            for xq in test_points
                dual_val = ForwardDiff.value(ForwardDiff.Dual(xq, 1.0) |> itp)
                direct_val = itp(xq)
                @test dual_val ≈ direct_val atol=1e-10
            end
        end

        @testset "side modes" begin
            # Test all side modes work with AD
            for side_mode in [LeftSide(), RightSide(), NearestSide()]
                itp = constant_interp(x, y; side=side_mode, extrap=ExtendExtrap())
                fd_deriv = ForwardDiff.derivative(itp, 2.5)
                @test fd_deriv ≈ 0.0 atol=1e-10
            end
        end
    end

    # ========================================
    # ConstantSeriesInterpolant with ForwardDiff
    # ========================================

    @testset "ConstantSeriesInterpolant with ForwardDiff" begin
        x = collect(0.0:1.0:5.0)
        y1 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        y2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        sitp = constant_interp(x, Series(y1, y2); side=LeftSide(), extrap=ExtendExtrap())

        @testset "series derivative is zero" begin
            # ForwardDiff.derivative on series returns vector of derivatives
            for xq in [0.5, 1.5, 2.5]
                fd_derivs = ForwardDiff.derivative(sitp, xq)
                analytical = sitp(xq; deriv=DerivOp(1))

                @test all(fd_derivs .≈ 0.0)
                @test fd_derivs ≈ analytical atol=1e-10
            end
        end

        @testset "series value is preserved" begin
            xq = 2.5
            dual_result = sitp(ForwardDiff.Dual(xq, 1.0))
            fd_values = ForwardDiff.value.(dual_result)

            @test fd_values ≈ sitp(xq) atol=1e-10
        end
    end

    # ========================================
    # Constant + Complex values with ForwardDiff
    # ========================================

    @testset "Constant complex values with ForwardDiff" begin
        x = collect(0.0:1.0:5.0)
        y_complex = [10.0+1.0im, 20.0+2.0im, 30.0+3.0im, 40.0+4.0im, 50.0+5.0im, 60.0+6.0im]

        itp = constant_interp(x, y_complex; side=LeftSide(), extrap=ExtendExtrap())

        @testset "complex derivative" begin
            xq = 2.5

            fd_deriv = ForwardDiff.derivative(itp, xq)
            analytical = itp(xq; deriv=DerivOp(1))

            # Derivative should be zero for both real and imaginary parts
            @test real(fd_deriv) ≈ 0.0 atol=1e-10
            @test imag(fd_deriv) ≈ 0.0 atol=1e-10
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    # ========================================
    # Constant gradient composition
    # ========================================

    @testset "Constant gradient composition" begin
        x = collect(0.0:1.0:5.0)
        y = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        itp = constant_interp(x, y; side=LeftSide(), extrap=ExtendExtrap())

        @testset "gradient through loss function" begin
            function loss(params)
                xq = params[1]
                return itp(xq)^2
            end

            # d/d(xq)[f(xq)²] = 2 * f(xq) * f'(xq) = 2 * f(xq) * 0 = 0
            grad = ForwardDiff.gradient(loss, [2.5])
            @test grad[1] ≈ 0.0 atol=1e-10
        end
    end

    # ========================================
    # QuadraticInterpolant with ForwardDiff
    # ========================================

    @testset "QuadraticInterpolant with ForwardDiff" begin
        # Quadratic data: y = x²
        # For quadratic spline with quadratic data, derivatives should be exact
        x = collect(0.0:0.5:5.0)
        y = x .^ 2

        itp = quadratic_interp(x, y; extrap=ExtendExtrap())

        @testset "derivative matches analytical" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                # ForwardDiff derivative
                fd_deriv = ForwardDiff.derivative(itp, xq)

                # Analytical derivative from interpolant
                analytical = itp(xq; deriv=DerivOp(1))

                @test fd_deriv ≈ analytical atol=1e-10
            end
        end

        @testset "exact quadratic reproduction" begin
            # Quadratic spline should reproduce y=x² exactly (up to BC effects)
            test_points = [0.25, 1.25, 2.25, 3.25]

            for xq in test_points
                fd_deriv = ForwardDiff.derivative(itp, xq)
                # True derivative of x² is 2x
                expected = 2.0 * xq
                @test fd_deriv ≈ expected atol=1e-10
            end
        end

        @testset "value is preserved" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                fd_value = ForwardDiff.value(ForwardDiff.Dual(xq, 1.0) |> itp)
                direct_value = itp(xq)

                @test fd_value ≈ direct_value atol=1e-10
            end
        end

        @testset "works with cubic data" begin
            # Cubic data: y = x³ - piecewise quadratic won't be exact
            y_cubic = x .^ 3
            itp_cubic = quadratic_interp(x, y_cubic; extrap=ExtendExtrap())

            xq = 2.25

            fd_deriv = ForwardDiff.derivative(itp_cubic, xq)
            analytical = itp_cubic(xq; deriv=DerivOp(1))

            # ForwardDiff and analytical should still match
            @test fd_deriv ≈ analytical atol=1e-10
        end

        @testset "type stability" begin
            xq = ForwardDiff.Dual(2.5, 1.0)
            result = itp(xq)
            @test result isa ForwardDiff.Dual
        end

        @testset "Float32 support" begin
            x32 = Float32.(collect(0.0:0.5:5.0))
            y32 = x32 .^ 2
            itp32 = quadratic_interp(x32, y32; extrap=ExtendExtrap())

            xq = 2.25f0
            fd_deriv = ForwardDiff.derivative(itp32, xq)
            analytical = itp32(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ analytical atol=1e-5
            @test fd_deriv isa Float32
        end

        @testset "gradient computation" begin
            function loss(params)
                xq = params[1]
                return itp(xq)^2
            end

            params = [2.25]
            grad = ForwardDiff.gradient(loss, params)

            # d/d(xq)[itp(xq)²] = 2 * itp(xq) * itp'(xq)
            val = itp(2.25)
            deriv = itp(2.25; deriv=DerivOp(1))
            expected_grad = 2 * val * deriv

            @test grad[1] ≈ expected_grad atol=1e-10
        end

        @testset "one-shot API AD" begin
            # One-shot API should also support ForwardDiff
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> quadratic_interp(x, y, q), xq)
            analytical = quadratic_interp(x, y, xq; deriv=DerivOp(1))
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    @testset "Quadratic complex values with ForwardDiff" begin
        x = collect(0.0:0.5:5.0)
        # Complex quadratic: z = (1 + i)x²
        y_complex = (1.0 + 1.0im) .* x .^ 2

        itp = quadratic_interp(x, y_complex; extrap=ExtendExtrap())

        @testset "complex derivative" begin
            xq = 2.25

            fd_deriv = ForwardDiff.derivative(itp, xq)
            analytical = itp(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ analytical atol=1e-10
            # For (1+i)x², derivative is (1+i)*2x = (2+2i)*x
            # At x=2.25: (2+2i)*2.25 = 4.5 + 4.5i
            @test real(fd_deriv) ≈ 2.0 * 2.25 atol=1e-10
            @test imag(fd_deriv) ≈ 2.0 * 2.25 atol=1e-10
        end

        @testset "one-shot API AD with complex" begin
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> quadratic_interp(x, y_complex, q), xq)
            analytical = quadratic_interp(x, y_complex, xq; deriv=DerivOp(1))
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    # ========================================
    # CubicInterpolant with ForwardDiff
    # ========================================

    @testset "CubicInterpolant with ForwardDiff" begin
        # Cubic data: y = x³
        # For cubic spline with cubic data, derivatives should be close
        x = collect(0.0:0.5:5.0)
        y = x .^ 3

        itp = cubic_interp(x, y; extrap=ExtendExtrap())

        @testset "derivative matches analytical" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                # ForwardDiff derivative
                fd_deriv = ForwardDiff.derivative(itp, xq)

                # Analytical derivative from interpolant
                analytical = itp(xq; deriv=DerivOp(1))

                @test fd_deriv ≈ analytical atol=1e-10
            end
        end

        @testset "cubic data derivative accuracy" begin
            # Cubic spline should reproduce cubic polynomial derivatives well
            test_points = [1.25, 2.25, 3.25]

            for xq in test_points
                fd_deriv = ForwardDiff.derivative(itp, xq)
                # True derivative of x³ is 3x²
                expected = 3.0 * xq^2
                # Cubic spline approximation, not exact
                @test fd_deriv ≈ expected rtol=0.05
            end
        end

        @testset "value is preserved" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                fd_value = ForwardDiff.value(ForwardDiff.Dual(xq, 1.0) |> itp)
                direct_value = itp(xq)

                @test fd_value ≈ direct_value atol=1e-10
            end
        end

        @testset "works with sine data" begin
            # Sin data: more realistic use case
            y_sin = sin.(x)
            itp_sin = cubic_interp(x, y_sin; extrap=ExtendExtrap())

            xq = 1.5

            fd_deriv = ForwardDiff.derivative(itp_sin, xq)
            analytical = itp_sin(xq; deriv=DerivOp(1))

            # ForwardDiff and analytical should match
            @test fd_deriv ≈ analytical atol=1e-10
            # Should be close to cos(1.5)
            @test fd_deriv ≈ cos(1.5) rtol=0.01
        end

        @testset "type stability" begin
            xq = ForwardDiff.Dual(2.5, 1.0)
            result = itp(xq)
            @test result isa ForwardDiff.Dual
        end

        @testset "Float32 support" begin
            x32 = Float32.(collect(0.0:0.5:5.0))
            y32 = x32 .^ 3
            itp32 = cubic_interp(x32, y32; extrap=ExtendExtrap())

            xq = 2.25f0
            fd_deriv = ForwardDiff.derivative(itp32, xq)
            analytical = itp32(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ analytical atol=1e-4
            @test fd_deriv isa Float32
        end

        @testset "gradient computation" begin
            function loss(params)
                xq = params[1]
                return itp(xq)^2
            end

            params = [2.25]
            grad = ForwardDiff.gradient(loss, params)

            # d/d(xq)[itp(xq)²] = 2 * itp(xq) * itp'(xq)
            val = itp(2.25)
            deriv = itp(2.25; deriv=DerivOp(1))
            expected_grad = 2 * val * deriv

            @test grad[1] ≈ expected_grad atol=1e-10
        end

        @testset "different BC types" begin
            # Test different boundary conditions work with AD
            for bc in [ZeroCurvBC(), ZeroSlopeBC()]
                itp_bc = cubic_interp(x, y; bc=bc, extrap=ExtendExtrap())

                xq = 2.25
                fd_deriv = ForwardDiff.derivative(itp_bc, xq)
                analytical = itp_bc(xq; deriv=DerivOp(1))

                @test fd_deriv ≈ analytical atol=1e-10
            end
        end

        @testset "one-shot API AD" begin
            # One-shot API should also support ForwardDiff
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> cubic_interp(x, y, q), xq)
            analytical = cubic_interp(x, y, xq; deriv=DerivOp(1))
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    @testset "Cubic complex values with ForwardDiff" begin
        x = collect(0.0:0.5:5.0)
        # Complex cubic: z = (1 + i)x³
        y_complex = (1.0 + 1.0im) .* x .^ 3

        itp = cubic_interp(x, y_complex; extrap=ExtendExtrap())

        @testset "complex derivative" begin
            xq = 2.25

            fd_deriv = ForwardDiff.derivative(itp, xq)
            analytical = itp(xq; deriv=DerivOp(1))

            @test fd_deriv ≈ analytical atol=1e-10
            # For (1+i)x³, derivative is (1+i)*3x²
            # At x=2.25: (3+3i)*2.25² ≈ 15.1875 + 15.1875i
            expected_deriv = 3.0 * 2.25^2
            @test real(fd_deriv) ≈ expected_deriv rtol=0.05
            @test imag(fd_deriv) ≈ expected_deriv rtol=0.05
        end

        @testset "one-shot API AD with complex" begin
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> cubic_interp(x, y_complex, q), xq)
            analytical = cubic_interp(x, y_complex, xq; deriv=DerivOp(1))
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    # ========================================
    # QuadraticSeriesInterpolant with ForwardDiff
    # ========================================

    @testset "QuadraticSeriesInterpolant with ForwardDiff" begin
        # Two series: sin and cos
        x = collect(range(0.0, 2π, 21))
        y1 = sin.(x)
        y2 = cos.(x)
        sitp = quadratic_interp(x, Series(y1, y2); extrap=ExtendExtrap())

        @testset "derivative matches analytical (interior points)" begin
            # Test at interior points only (not at grid points)
            test_points = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

            for xq in test_points
                # ForwardDiff computes derivative of entire vector at once
                fd_derivs = ForwardDiff.derivative(sitp, xq)
                analytical = sitp(xq; deriv=DerivOp(1))

                @test fd_derivs ≈ analytical atol=1e-10
            end
        end

        @testset "value is preserved" begin
            test_points = [0.5, 1.5, 2.5]

            for xq in test_points
                dual_result = sitp(ForwardDiff.Dual(xq, 1.0))
                fd_values = ForwardDiff.value.(dual_result)

                @test fd_values ≈ sitp(xq) atol=1e-10
            end
        end

        @testset "type stability" begin
            xq = ForwardDiff.Dual(1.5, 1.0)
            result = sitp(xq)

            # Output should be vector of Dual
            @test eltype(result) <: ForwardDiff.Dual
        end

        @testset "quadratic data exact derivative" begin
            # Quadratic data: y1 = x², y2 = 2x²
            x_quad = collect(0.0:0.5:5.0)
            y1_quad = x_quad .^ 2
            y2_quad = 2.0 .* x_quad .^ 2
            sitp_quad = quadratic_interp(x_quad, Series(y1_quad, y2_quad); extrap=ExtendExtrap())

            xq = 2.25
            fd_derivs = ForwardDiff.derivative(sitp_quad, xq)
            analytical = sitp_quad(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-10
            # Derivative of x² is 2x, derivative of 2x² is 4x
            @test fd_derivs ≈ [2.0 * xq, 4.0 * xq] atol=1e-10
        end

        @testset "Float32 support" begin
            x32 = Float32.(collect(range(0.0, 2π, 21)))
            y1_32 = sin.(x32)
            y2_32 = cos.(x32)
            sitp32 = quadratic_interp(x32, Series(y1_32, y2_32); extrap=ExtendExtrap())

            xq = 1.5f0
            fd_derivs = ForwardDiff.derivative(sitp32, xq)
            analytical = sitp32(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-5
            @test eltype(fd_derivs) == Float32
        end

        @testset "gradient computation" begin
            # Gradient of scalar function using interpolant
            function loss(params)
                xq = params[1]
                vals = sitp(xq)
                return sum(vals .^ 2)  # scalar output
            end

            params = [2.25]
            grad = ForwardDiff.gradient(loss, params)

            # d/d(xq)[sum(f_i(xq)²)] = 2 * sum(f_i(xq) * f_i'(xq))
            vals = sitp(2.25)
            derivs = sitp(2.25; deriv=DerivOp(1))
            expected_grad = 2 * sum(vals .* derivs)

            @test grad[1] ≈ expected_grad atol=1e-10
        end
    end

    @testset "QuadraticSeriesInterpolant complex values with ForwardDiff" begin
        x = collect(range(0.0, 2π, 21))
        # Complex series
        y1_complex = (1.0 + 1.0im) .* sin.(x)
        y2_complex = (2.0 - 1.0im) .* cos.(x)
        sitp = quadratic_interp(x, Series(y1_complex, y2_complex); extrap=ExtendExtrap())

        @testset "complex derivative" begin
            xq = 1.5

            fd_derivs = ForwardDiff.derivative(sitp, xq)
            analytical = sitp(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-10
        end

        @testset "value preserved for complex" begin
            xq = 1.5
            dual_result = sitp(ForwardDiff.Dual(xq, 1.0))
            fd_values = ForwardDiff.value.(dual_result)

            @test fd_values ≈ sitp(xq) atol=1e-10
        end
    end

    # ========================================
    # CubicSeriesInterpolant with ForwardDiff
    # ========================================

    @testset "CubicSeriesInterpolant with ForwardDiff" begin
        # Two series: sin and cos
        x = collect(range(0.0, 2π, 21))
        y1 = sin.(x)
        y2 = cos.(x)
        sitp = cubic_interp(x, Series(y1, y2); extrap=ExtendExtrap())

        @testset "derivative matches analytical (interior points)" begin
            # Test at interior points only (not at grid points)
            test_points = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

            for xq in test_points
                # ForwardDiff computes derivative of entire vector at once
                fd_derivs = ForwardDiff.derivative(sitp, xq)
                analytical = sitp(xq; deriv=DerivOp(1))

                @test fd_derivs ≈ analytical atol=1e-10
            end
        end

        @testset "value is preserved" begin
            test_points = [0.5, 1.5, 2.5]

            for xq in test_points
                dual_result = sitp(ForwardDiff.Dual(xq, 1.0))
                fd_values = ForwardDiff.value.(dual_result)

                @test fd_values ≈ sitp(xq) atol=1e-10
            end
        end

        @testset "type stability" begin
            xq = ForwardDiff.Dual(1.5, 1.0)
            result = sitp(xq)

            # Output should be vector of Dual
            @test eltype(result) <: ForwardDiff.Dual
        end

        @testset "cubic data exact derivative" begin
            # Cubic data: y1 = x³, y2 = 2x³
            x_cub = collect(0.0:0.5:5.0)
            y1_cub = x_cub .^ 3
            y2_cub = 2.0 .* x_cub .^ 3
            sitp_cub = cubic_interp(x_cub, Series(y1_cub, y2_cub); extrap=ExtendExtrap(), bc=CubicFit())

            xq = 2.25
            fd_derivs = ForwardDiff.derivative(sitp_cub, xq)
            analytical = sitp_cub(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-10
            # Derivative of x³ is 3x², derivative of 2x³ is 6x²
            @test fd_derivs ≈ [3.0 * xq^2, 6.0 * xq^2] atol=1e-10
        end

        @testset "Float32 support" begin
            x32 = Float32.(collect(range(0.0, 2π, 21)))
            y1_32 = sin.(x32)
            y2_32 = cos.(x32)
            sitp32 = cubic_interp(x32, Series(y1_32, y2_32); extrap=ExtendExtrap())

            xq = 1.5f0
            fd_derivs = ForwardDiff.derivative(sitp32, xq)
            analytical = sitp32(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-5
            @test eltype(fd_derivs) == Float32
        end

        @testset "gradient computation" begin
            # Gradient of scalar function using interpolant
            function loss(params)
                xq = params[1]
                vals = sitp(xq)
                return sum(vals .^ 2)  # scalar output
            end

            params = [2.25]
            grad = ForwardDiff.gradient(loss, params)

            # d/d(xq)[sum(f_i(xq)²)] = 2 * sum(f_i(xq) * f_i'(xq))
            vals = sitp(2.25)
            derivs = sitp(2.25; deriv=DerivOp(1))
            expected_grad = 2 * sum(vals .* derivs)

            @test grad[1] ≈ expected_grad atol=1e-10
        end
    end

    @testset "CubicSeriesInterpolant complex values with ForwardDiff" begin
        x = collect(range(0.0, 2π, 21))
        # Complex series
        y1_complex = (1.0 + 1.0im) .* sin.(x)
        y2_complex = (2.0 - 1.0im) .* cos.(x)
        sitp = cubic_interp(x, Series(y1_complex, y2_complex); extrap=ExtendExtrap())

        @testset "complex derivative" begin
            xq = 1.5

            fd_derivs = ForwardDiff.derivative(sitp, xq)
            analytical = sitp(xq; deriv=DerivOp(1))

            @test fd_derivs ≈ analytical atol=1e-10
        end

        @testset "value preserved for complex" begin
            xq = 1.5
            dual_result = sitp(ForwardDiff.Dual(xq, 1.0))
            fd_values = ForwardDiff.value.(dual_result)

            @test fd_values ≈ sitp(xq) atol=1e-10
        end
    end

    # ========================================
    # Extrapolation Mode AD Tests
    # ========================================
    # Tests that ForwardDiff derivatives match analytical derivatives
    # for all extrapolation modes (extension, constant, wrap).
    #
    # The wrap mode is particularly important: _wrap_to_domain must
    # preserve ForwardDiff.Dual type through the mod() operation.

    @testset "Extrapolation modes with ForwardDiff" begin

        @testset "Linear extrap modes" begin
            x = collect(0.0:0.5:5.0)
            y_quad = x .^ 2  # y = x², slope at interval [2.0,2.5] is (6.25-4)/0.5 = 4.5

            @testset "extrap=ExtendExtrap()" begin
                itp = linear_interp(x, y_quad; extrap=ExtendExtrap())

                # In-domain
                fd = ForwardDiff.derivative(itp, 2.25)
                an = itp(2.25; deriv=DerivOp(1))
                @test fd ≈ an atol=1e-10

                # Outside domain (extends linearly)
                fd_out = ForwardDiff.derivative(itp, 6.5)
                an_out = itp(6.5; deriv=DerivOp(1))
                @test fd_out ≈ an_out atol=1e-10
            end

            @testset "extrap=ConstExtrap()" begin
                itp = linear_interp(x, y_quad; extrap=ConstExtrap())

                # In-domain
                fd = ForwardDiff.derivative(itp, 2.25)
                an = itp(2.25; deriv=DerivOp(1))
                @test fd ≈ an atol=1e-10

                # Outside domain right: derivative is 0 for constant extrap
                fd_right = ForwardDiff.derivative(itp, 6.5)
                an_right = itp(6.5; deriv=DerivOp(1))
                @test fd_right ≈ 0.0 atol=1e-10
                @test fd_right ≈ an_right atol=1e-10

                # Outside domain left: derivative is also 0
                fd_left = ForwardDiff.derivative(itp, -1.0)
                an_left = itp(-1.0; deriv=DerivOp(1))
                @test fd_left ≈ 0.0 atol=1e-10
                @test fd_left ≈ an_left atol=1e-10
            end

            @testset "extrap=WrapExtrap()" begin
                itp = linear_interp(x, y_quad; extrap=WrapExtrap())

                # In-domain: AD must preserve Dual type
                fd_in = ForwardDiff.derivative(itp, 2.25)
                an_in = itp(2.25; deriv=DerivOp(1))
                @test fd_in ≈ an_in atol=1e-10

                # Out-of-domain positive: wraps via mod(), AD must work
                fd_pos = ForwardDiff.derivative(itp, 6.5)
                an_pos = itp(6.5; deriv=DerivOp(1))
                @test fd_pos ≈ an_pos atol=1e-10
                # 6.5 mod 5.0 = 1.5 → in interval [1.0, 1.5]
                @test fd_pos ≈ 3.5 atol=1e-10  # slope at x=1.5

                # Out-of-domain negative: wraps correctly
                fd_neg = ForwardDiff.derivative(itp, -1.0)
                an_neg = itp(-1.0; deriv=DerivOp(1))
                @test fd_neg ≈ an_neg atol=1e-10
                # -1.0 mod 5.0 = 4.0 → in interval [4.0, 4.5]
                @test fd_neg ≈ 8.5 atol=1e-10  # slope at x=4.5
            end
        end

        @testset "Cubic extrap modes" begin
            x = collect(0.0:0.5:5.0)
            y_cubic = x .^ 3

            @testset "extrap=ExtendExtrap()" begin
                itp = cubic_interp(x, y_cubic; extrap=ExtendExtrap())

                fd = ForwardDiff.derivative(itp, 2.25)
                an = itp(2.25; deriv=DerivOp(1))
                @test fd ≈ an atol=1e-10

                fd_out = ForwardDiff.derivative(itp, 6.5)
                an_out = itp(6.5; deriv=DerivOp(1))
                @test fd_out ≈ an_out atol=1e-10
            end

            @testset "extrap=ConstExtrap()" begin
                itp = cubic_interp(x, y_cubic; extrap=ConstExtrap())

                fd = ForwardDiff.derivative(itp, 2.25)
                an = itp(2.25; deriv=DerivOp(1))
                @test fd ≈ an atol=1e-10

                # Out-of-domain: derivative is 0 for constant extrap
                fd_right = ForwardDiff.derivative(itp, 6.5)
                an_right = itp(6.5; deriv=DerivOp(1))
                @test fd_right ≈ 0.0 atol=1e-10
                @test fd_right ≈ an_right atol=1e-10

                fd_left = ForwardDiff.derivative(itp, -1.0)
                an_left = itp(-1.0; deriv=DerivOp(1))
                @test fd_left ≈ 0.0 atol=1e-10
                @test fd_left ≈ an_left atol=1e-10
            end

            @testset "extrap=WrapExtrap()" begin
                itp = cubic_interp(x, y_cubic; extrap=WrapExtrap())

                # In-domain
                fd_in = ForwardDiff.derivative(itp, 2.25)
                an_in = itp(2.25; deriv=DerivOp(1))
                @test fd_in ≈ an_in atol=1e-10

                # Out-of-domain: wraps and AD still works
                fd_out = ForwardDiff.derivative(itp, 6.5)
                an_out = itp(6.5; deriv=DerivOp(1))
                @test fd_out ≈ an_out atol=1e-10
            end
        end

        @testset "Quadratic extrap modes" begin
            x = collect(0.0:0.5:5.0)
            y_quad = x .^ 2

            @testset "extrap=ExtendExtrap()" begin
                itp = quadratic_interp(x, y_quad; extrap=ExtendExtrap())

                fd = ForwardDiff.derivative(itp, 2.25)
                an = itp(2.25; deriv=DerivOp(1))
                @test fd ≈ an atol=1e-10

                fd_out = ForwardDiff.derivative(itp, 6.5)
                an_out = itp(6.5; deriv=DerivOp(1))
                @test fd_out ≈ an_out atol=1e-10
            end

            @testset "extrap=ConstExtrap()" begin
                itp = quadratic_interp(x, y_quad; extrap=ConstExtrap())

                fd = ForwardDiff.derivative(itp, 2.25)
                an = itp(2.25; deriv=DerivOp(1))
                @test fd ≈ an atol=1e-10

                # Out-of-domain: derivative is 0 for constant extrap
                fd_right = ForwardDiff.derivative(itp, 6.5)
                an_right = itp(6.5; deriv=DerivOp(1))
                @test fd_right ≈ 0.0 atol=1e-10
                @test fd_right ≈ an_right atol=1e-10

                fd_left = ForwardDiff.derivative(itp, -1.0)
                an_left = itp(-1.0; deriv=DerivOp(1))
                @test fd_left ≈ 0.0 atol=1e-10
                @test fd_left ≈ an_left atol=1e-10
            end

            @testset "extrap=WrapExtrap()" begin
                itp = quadratic_interp(x, y_quad; extrap=WrapExtrap())

                # In-domain
                fd_in = ForwardDiff.derivative(itp, 2.25)
                an_in = itp(2.25; deriv=DerivOp(1))
                @test fd_in ≈ an_in atol=1e-10

                # Out-of-domain: wraps and AD still works
                fd_out = ForwardDiff.derivative(itp, 6.5)
                an_out = itp(6.5; deriv=DerivOp(1))
                @test fd_out ≈ an_out atol=1e-10

                # Value preservation after wrapping
                @test itp(6.5) ≈ itp(1.5) atol=1e-10  # 6.5 mod 5 = 1.5
            end
        end

        @testset "Value preservation across extrap modes" begin
            x = collect(0.0:1.0:10.0)
            y = 2.0 .* x .+ 1.0  # y = 2x + 1

            for ext in (ExtendExtrap(), ConstExtrap(), WrapExtrap())
                itp = linear_interp(x, y; extrap=ext)

                # In-domain value test
                xq = 5.5
                dual_val = itp(ForwardDiff.Dual(xq, 1.0))
                @test ForwardDiff.value(dual_val) ≈ itp(xq) atol=1e-10
            end
        end
    end

    # ========================================
    # ND Interpolant ForwardDiff Tests
    # ========================================
    # These test heterogeneous tuple queries: (Dual, Float64)
    # which arise when computing partial derivatives via ForwardDiff.

    @testset "LinearInterpolantND with ForwardDiff" begin
        x = collect(range(0.0, 2.0, 11))
        y = collect(range(0.0, 1.0, 6))
        data = [xi + 2yi for xi in x, yi in y]  # f(x,y) = x + 2y

        itp = linear_interp((x, y), data; extrap=ExtendExtrap())

        @testset "partial derivative via heterogeneous tuple" begin
            # (Dual, Float64) tuple — requires Tuple{Vararg{Real,N}} not NTuple{N,<:Real}
            df_dx = ForwardDiff.derivative(t -> itp((t, 0.5)), 1.0)
            @test df_dx ≈ 1.0 atol=1e-10  # ∂f/∂x = 1

            df_dy = ForwardDiff.derivative(t -> itp((1.0, t)), 0.5)
            @test df_dy ≈ 2.0 atol=1e-10  # ∂f/∂y = 2
        end

        @testset "gradient via vector query" begin
            grad = ForwardDiff.gradient(v -> itp(v), [1.0, 0.5])
            @test grad ≈ [1.0, 2.0] atol=1e-10
        end

        @testset "value preserved under Dual" begin
            result = itp((ForwardDiff.Dual(1.0, 1.0), 0.5))
            @test ForwardDiff.value(result) ≈ itp((1.0, 0.5)) atol=1e-10
        end
    end

    @testset "ConstantInterpolantND with ForwardDiff" begin
        x = collect(range(0.0, 2.0, 11))
        y = collect(range(0.0, 1.0, 6))
        data = [xi + 2yi for xi in x, yi in y]

        itp = constant_interp((x, y), data; extrap=ExtendExtrap())

        @testset "partial derivative via heterogeneous tuple" begin
            # Constant interpolation derivative is 0 (step function)
            df_dx = ForwardDiff.derivative(t -> itp((t, 0.5)), 1.0)
            @test df_dx ≈ 0.0 atol=1e-10

            df_dy = ForwardDiff.derivative(t -> itp((1.0, t)), 0.5)
            @test df_dy ≈ 0.0 atol=1e-10
        end

        @testset "gradient via vector query" begin
            grad = ForwardDiff.gradient(v -> itp(v), [1.0, 0.5])
            @test grad ≈ [0.0, 0.0] atol=1e-10
        end

        @testset "value preserved under Dual" begin
            result = itp((ForwardDiff.Dual(1.0, 1.0), 0.5))
            @test ForwardDiff.value(result) ≈ itp((1.0, 0.5)) atol=1e-10
        end
    end

end  # testset "AutoDiff Support"
