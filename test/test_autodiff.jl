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
# The analytical derivative (deriv=1) should match ForwardDiff.derivative().
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

        itp = linear_interp(x, y; extrap=:extension)

        @testset "derivative matches analytical" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                # ForwardDiff derivative
                fd_deriv = ForwardDiff.derivative(itp, xq)

                # Analytical derivative from interpolant
                analytical = itp(xq; deriv=1)

                @test fd_deriv ≈ analytical atol=1e-10
                @test fd_deriv ≈ 2.0 atol=1e-10  # Known slope
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

        @testset "works with nonlinear data" begin
            # Quadratic data: y = x²
            y_quad = x .^ 2
            itp_quad = linear_interp(x, y_quad; extrap=:extension)

            # At x=2.0 (grid point), linear interp between x=1.5 and x=2.0
            # or between x=2.0 and x=2.5 depending on interval
            xq = 2.25  # Midpoint of [2.0, 2.5]

            # Linear slope between (2.0, 4.0) and (2.5, 6.25)
            expected_slope = (6.25 - 4.0) / (2.5 - 2.0)  # = 4.5

            fd_deriv = ForwardDiff.derivative(itp_quad, xq)
            analytical = itp_quad(xq; deriv=1)

            @test fd_deriv ≈ expected_slope atol=1e-10
            @test fd_deriv ≈ analytical atol=1e-10
        end

        @testset "one-shot API AD" begin
            # One-shot API should also support ForwardDiff
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> linear_interp(x, y, q), xq)
            analytical = linear_interp(x, y, xq; deriv=1)
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
        sitp = linear_interp(x, [y1, y2]; extrap=:extension)

        @testset "derivative matches analytical (interior points)" begin
            # Test at interior points only (not at grid points)
            test_points = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

            for xq in test_points
                # ForwardDiff computes derivative of entire vector at once
                fd_derivs = ForwardDiff.derivative(sitp, xq)
                analytical = sitp(xq; deriv=1)

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
            sitp_lin = linear_interp(x_lin, [y1_lin, y2_lin]; extrap=:extension)

            xq = 2.25
            fd_derivs = ForwardDiff.derivative(sitp_lin, xq)
            analytical = sitp_lin(xq; deriv=1)

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

        itp = linear_interp(x, y_complex; extrap=:extension)

        @testset "complex derivative" begin
            xq = 2.25

            # ForwardDiff on complex function
            fd_deriv = ForwardDiff.derivative(itp, xq)

            # Analytical
            analytical = itp(xq; deriv=1)

            @test fd_deriv ≈ analytical atol=1e-10
            @test real(fd_deriv) ≈ 2.0 atol=1e-10  # d/dx(2x+1)
            @test imag(fd_deriv) ≈ 1.0 atol=1e-10  # d/dx(x-1)
        end

        @testset "one-shot API AD with complex" begin
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> linear_interp(x, y_complex, q), xq)
            analytical = linear_interp(x, y_complex, xq; deriv=1)
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    # ========================================
    # Type Stability Tests
    # ========================================

    @testset "type stability" begin
        x = collect(0.0:0.5:5.0)
        y = sin.(x)
        itp = linear_interp(x, y; extrap=:extension)

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

        itp = linear_interp(x, y; extrap=:extension)

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
            deriv = itp(2.25; deriv=1)
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

        itp32 = linear_interp(x32, y32; extrap=:extension)

        @testset "Float32 derivative" begin
            xq = 2.25f0
            fd_deriv = ForwardDiff.derivative(itp32, xq)
            analytical = itp32(xq; deriv=1)

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
            itp = constant_interp(x, y; side=:left, extrap=:extension)

            # Constant interpolation derivative should be 0 (step function)
            for xq in [0.5, 1.5, 2.5, 3.5]
                fd_deriv = ForwardDiff.derivative(itp, xq)
                analytical = itp(xq; deriv=1)
                @test fd_deriv ≈ 0.0 atol=1e-10
                @test fd_deriv ≈ analytical atol=1e-10
            end
        end

        @testset "one-shot API AD" begin
            # Verify one-shot API works with ForwardDiff
            xq = 2.5
            fd_deriv = ForwardDiff.derivative(q -> constant_interp(x, y, q; side=:left), xq)
            @test fd_deriv ≈ 0.0 atol=1e-10
        end

        @testset "value is preserved" begin
            itp = constant_interp(x, y; side=:left, extrap=:extension)
            test_points = [0.5, 1.5, 2.5, 3.5]

            for xq in test_points
                dual_val = ForwardDiff.value(ForwardDiff.Dual(xq, 1.0) |> itp)
                direct_val = itp(xq)
                @test dual_val ≈ direct_val atol=1e-10
            end
        end

        @testset "side modes" begin
            # Test all side modes work with AD
            for side_mode in [:left, :right, :nearest]
                itp = constant_interp(x, y; side=side_mode, extrap=:extension)
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
        sitp = constant_interp(x, [y1, y2]; side=:left, extrap=:extension)

        @testset "series derivative is zero" begin
            # ForwardDiff.derivative on series returns vector of derivatives
            for xq in [0.5, 1.5, 2.5]
                fd_derivs = ForwardDiff.derivative(sitp, xq)
                analytical = sitp(xq; deriv=1)

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

        itp = constant_interp(x, y_complex; side=:left, extrap=:extension)

        @testset "complex derivative" begin
            xq = 2.5

            fd_deriv = ForwardDiff.derivative(itp, xq)
            analytical = itp(xq; deriv=1)

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
        itp = constant_interp(x, y; side=:left, extrap=:extension)

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

        itp = quadratic_interp(x, y; extrap=:extension)

        @testset "derivative matches analytical" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                # ForwardDiff derivative
                fd_deriv = ForwardDiff.derivative(itp, xq)

                # Analytical derivative from interpolant
                analytical = itp(xq; deriv=1)

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
            itp_cubic = quadratic_interp(x, y_cubic; extrap=:extension)

            xq = 2.25

            fd_deriv = ForwardDiff.derivative(itp_cubic, xq)
            analytical = itp_cubic(xq; deriv=1)

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
            itp32 = quadratic_interp(x32, y32; extrap=:extension)

            xq = 2.25f0
            fd_deriv = ForwardDiff.derivative(itp32, xq)
            analytical = itp32(xq; deriv=1)

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
            deriv = itp(2.25; deriv=1)
            expected_grad = 2 * val * deriv

            @test grad[1] ≈ expected_grad atol=1e-10
        end

        @testset "one-shot API AD" begin
            # One-shot API should also support ForwardDiff
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> quadratic_interp(x, y, q), xq)
            analytical = quadratic_interp(x, y, xq; deriv=1)
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    @testset "Quadratic complex values with ForwardDiff" begin
        x = collect(0.0:0.5:5.0)
        # Complex quadratic: z = (1 + i)x²
        y_complex = (1.0 + 1.0im) .* x .^ 2

        itp = quadratic_interp(x, y_complex; extrap=:extension)

        @testset "complex derivative" begin
            xq = 2.25

            fd_deriv = ForwardDiff.derivative(itp, xq)
            analytical = itp(xq; deriv=1)

            @test fd_deriv ≈ analytical atol=1e-10
            # For (1+i)x², derivative is (1+i)*2x = (2+2i)*x
            # At x=2.25: (2+2i)*2.25 = 4.5 + 4.5i
            @test real(fd_deriv) ≈ 2.0 * 2.25 atol=1e-10
            @test imag(fd_deriv) ≈ 2.0 * 2.25 atol=1e-10
        end

        @testset "one-shot API AD with complex" begin
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> quadratic_interp(x, y_complex, q), xq)
            analytical = quadratic_interp(x, y_complex, xq; deriv=1)
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

        itp = cubic_interp(x, y; extrap=:extension)

        @testset "derivative matches analytical" begin
            test_points = [0.25, 1.0, 2.5, 3.75, 4.5]

            for xq in test_points
                # ForwardDiff derivative
                fd_deriv = ForwardDiff.derivative(itp, xq)

                # Analytical derivative from interpolant
                analytical = itp(xq; deriv=1)

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
            itp_sin = cubic_interp(x, y_sin; extrap=:extension)

            xq = 1.5

            fd_deriv = ForwardDiff.derivative(itp_sin, xq)
            analytical = itp_sin(xq; deriv=1)

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
            itp32 = cubic_interp(x32, y32; extrap=:extension)

            xq = 2.25f0
            fd_deriv = ForwardDiff.derivative(itp32, xq)
            analytical = itp32(xq; deriv=1)

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
            deriv = itp(2.25; deriv=1)
            expected_grad = 2 * val * deriv

            @test grad[1] ≈ expected_grad atol=1e-10
        end

        @testset "different BC types" begin
            # Test different boundary conditions work with AD
            for bc in [NaturalBC(), ClampedBC()]
                itp_bc = cubic_interp(x, y; bc=bc, extrap=:extension)

                xq = 2.25
                fd_deriv = ForwardDiff.derivative(itp_bc, xq)
                analytical = itp_bc(xq; deriv=1)

                @test fd_deriv ≈ analytical atol=1e-10
            end
        end

        @testset "one-shot API AD" begin
            # One-shot API should also support ForwardDiff
            xq = 2.25
            fd_deriv = ForwardDiff.derivative(q -> cubic_interp(x, y, q), xq)
            analytical = cubic_interp(x, y, xq; deriv=1)
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

    @testset "Cubic complex values with ForwardDiff" begin
        x = collect(0.0:0.5:5.0)
        # Complex cubic: z = (1 + i)x³
        y_complex = (1.0 + 1.0im) .* x .^ 3

        itp = cubic_interp(x, y_complex; extrap=:extension)

        @testset "complex derivative" begin
            xq = 2.25

            fd_deriv = ForwardDiff.derivative(itp, xq)
            analytical = itp(xq; deriv=1)

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
            analytical = cubic_interp(x, y_complex, xq; deriv=1)
            @test fd_deriv ≈ analytical atol=1e-10
        end
    end

end  # testset "AutoDiff Support"
