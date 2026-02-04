using Test
using FastInterpolations

@testset "ND Cubic Interpolation (2D)" begin

    @testset "Basic 2D Interpolation" begin
        # Create a simple test function: f(x,y) = sin(x) * cos(y)
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # Create interpolant
        itp = cubic_interp((x, y), data)

        # Test type
        @test itp isa CubicInterpolantND
        @test ndims(itp) == 2
        @test size(itp) == (21, 11)

        # Test grid point pass-through
        for i in 1:length(x)
            for j in 1:length(y)
                @test itp((x[i], y[j])) ≈ data[i, j] atol=1e-12
            end
        end

        # Test interpolation at interior points
        xq, yq = 1.5, 0.8
        expected = sin(xq) * cos(yq)
        @test itp((xq, yq)) ≈ expected atol=1e-3  # Cubic interpolation accuracy

        # Test tuple input (same as above)
        @test itp((xq, yq)) ≈ expected atol=1e-3
    end

    @testset "Complex Value Support" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)

        # Complex-valued function: f(x,y) = sin(x)*cos(y) + i*cos(x)*sin(y)
        data = [sin(xi) * cos(yj) + im * cos(xi) * sin(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        # Check type parameters
        @test grid_type(itp) == Float64
        @test value_type(itp) == ComplexF64

        # Test grid point pass-through
        @test itp((x[5], y[3])) ≈ data[5, 3] atol=1e-12

        # Test interpolation
        xq, yq = 1.5, 0.8
        expected = sin(xq) * cos(yq) + im * cos(xq) * sin(yq)
        result = itp((xq, yq))
        @test result isa ComplexF64
        @test result ≈ expected atol=1e-3
    end

    @testset "Oneshot API" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # Single point oneshot
        xq, yq = 1.5, 0.8
        val = cubic_interp((x, y), data, (xq, yq))
        expected = sin(xq) * cos(yq)
        @test val ≈ expected atol=1e-3

        # Multiple points oneshot
        xqs = [0.5, 1.0, 1.5, 2.0]
        yqs = [0.2, 0.4, 0.6, 0.8]
        vals = cubic_interp((x, y), data, (xqs, yqs))
        @test length(vals) == 4
        for k in 1:4
            expected_k = sin(xqs[k]) * cos(yqs[k])
            @test vals[k] ≈ expected_k atol=2e-3
        end
    end

    @testset "Vector Evaluation" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        # Evaluate at multiple points
        xqs = collect(range(0.5, 2.0, 10))
        yqs = collect(range(0.2, 0.8, 10))
        vals = itp((xqs, yqs))

        @test length(vals) == 10
        for k in 1:10
            expected = sin(xqs[k]) * cos(yqs[k])
            @test vals[k] ≈ expected atol=2e-3
        end
    end

    @testset "Boundary Conditions" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # NaturalBC (default)
        itp_natural = cubic_interp((x, y), data; bc=NaturalBC())
        @test itp_natural((1.0, 0.5)) isa Float64

        # ClampedBC
        itp_clamped = cubic_interp((x, y), data; bc=ClampedBC())
        @test itp_clamped((1.0, 0.5)) isa Float64

        # Per-axis BC
        itp_mixed = cubic_interp((x, y), data; bc=(NaturalBC(), ClampedBC()))
        @test itp_mixed((1.0, 0.5)) isa Float64

        # BCPair with specific derivatives
        itp_deriv = cubic_interp((x, y), data;
            bc=(BCPair(Deriv1(0.0), Deriv1(0.0)), NaturalBC()))
        @test itp_deriv((1.0, 0.5)) isa Float64
    end

    @testset "CubicFit BC (Polynomial Endpoint Estimation)" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data; bc=CubicFit())
        @test itp((1.0, 0.5)) isa Float64

        # Should give good accuracy at interior points
        xq, yq = 1.5, 0.8
        expected = sin(xq) * cos(yq)
        @test itp((xq, yq)) ≈ expected atol=1e-3
    end

    @testset "Extrapolation Modes" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi * yj for xi in x, yj in y]

        # :none (default) - should throw outside domain
        itp_none = cubic_interp((x, y), data; extrap=:none)
        @test_throws DomainError itp_none((-0.1, 0.5))
        @test_throws DomainError itp_none((0.5, -0.1))

        # :constant - clamp to boundary
        itp_const = cubic_interp((x, y), data; extrap=:constant)
        @test itp_const((-0.1, 0.5)) ≈ itp_const((0.0, 0.5))
        @test itp_const((2.1, 0.5)) ≈ itp_const((2.0, 0.5))
    end

    @testset "Derivative Evaluation" begin
        # f(x,y) = x^2 * y^2 → ∂f/∂x = 2xy², ∂f/∂y = 2x²y, ∂²f/∂x∂y = 4xy
        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        data = [xi^2 * yj^2 for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        xq, yq = 1.0, 0.5

        # Value
        @test itp((xq, yq); deriv=(0, 0)) ≈ xq^2 * yq^2 atol=1e-4

        # ∂f/∂x = 2xy²
        dfdx_expected = 2 * xq * yq^2
        @test itp((xq, yq); deriv=(1, 0)) ≈ dfdx_expected atol=1e-3

        # ∂f/∂y = 2x²y
        dfdy_expected = 2 * xq^2 * yq
        @test itp((xq, yq); deriv=(0, 1)) ≈ dfdy_expected atol=1e-3

        # ∂²f/∂x∂y = 4xy
        d2fdxdy_expected = 4 * xq * yq
        @test itp((xq, yq); deriv=(1, 1)) ≈ d2fdxdy_expected atol=1e-2
    end

    @testset "Non-uniform Grids" begin
        # Non-uniform grid spacing
        x = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
        y = [0.0, 0.2, 0.5, 1.0]
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        # Test grid point pass-through
        @test itp((x[3], y[2])) ≈ data[3, 2] atol=1e-12

        # Test interpolation
        xq, yq = 0.5, 0.3
        expected = sin(xq) * cos(yq)
        @test itp((xq, yq)) ≈ expected atol=1e-3
    end

    @testset "Float32 Support" begin
        x = range(0.0f0, 2.0f0, 11)
        y = range(0.0f0, 1.0f0, 6)
        data = Float32[sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        @test grid_type(itp) == Float32
        @test value_type(itp) == Float32

        result = itp((1.0f0, 0.5f0))
        @test result isa Float32
        @test result ≈ sin(1.0f0) * cos(0.5f0) atol=5e-4
    end

    @testset "Type Introspection" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = rand(11, 11)

        itp = cubic_interp((x, y), data)

        @test grid_type(itp) == Float64
        @test value_type(itp) == Float64
        @test ndims(itp) == 2
        @test size(itp) == (11, 11)
        @test length(axes(itp)) == 2
    end

    @testset "Strategy Types" begin
        # PreCompute is default
        @test PreCompute() isa AbstractCoeffStrategy
        @test OnTheFly() isa AbstractCoeffStrategy

        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = rand(11, 11)

        # Explicit PreCompute
        itp = cubic_interp((x, y), data; coeffs=PreCompute())
        @test itp isa CubicInterpolantND

        # OnTheFly should throw (not implemented)
        @test_throws ArgumentError cubic_interp((x, y), data; coeffs=OnTheFly())
    end

    @testset "Dimension Mismatch Errors" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 6)
        data = rand(10, 6)  # Wrong x dimension

        @test_throws DimensionMismatch cubic_interp((x, y), data)

        data2 = rand(11, 5)  # Wrong y dimension
        @test_throws DimensionMismatch cubic_interp((x, y), data2)
    end

end
