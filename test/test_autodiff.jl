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

end  # testset "AutoDiff Support"
