# ========================================
# Complex Cubic Interpolation Tests
# ========================================
# Tests for native Complex number support in CubicInterpolant.
# Validates the Tg/Tv type separation design.

using Test
using FastInterpolations

@testset "Complex Cubic Interpolation" begin

    # ========================================
    # Basic Complex Interpolation
    # ========================================
    @testset "ComplexF64 values" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        # Type checks
        @test itp isa CubicInterpolant{Float64, ComplexF64}
        @test grid_type(itp) == Float64
        @test value_type(itp) == ComplexF64
        @test eval_type(itp, Float64) == ComplexF64

        # Evaluation returns ComplexF64
        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # ComplexF32 Support
    # ========================================
    @testset "ComplexF32 values" begin
        x = Float32[0, 1, 2, 3, 4]
        y = ComplexF32[1 + 2im, 3 + 4im, 5 + 6im, 7 + 8im, 9 + 10im]

        itp = cubic_interp(x, y)

        @test itp isa CubicInterpolant{Float32, ComplexF32}
        @test grid_type(itp) == Float32
        @test value_type(itp) == ComplexF32
        @test eval_type(itp, Float32) == ComplexF32

        val = itp(0.5f0)
        @test val isa ComplexF32
    end

    # ========================================
    # Integer Grid with Complex Values
    # ========================================
    @testset "Integer grid + Complex values" begin
        x = 0:4  # Range{Int}
        y = Complex{Int}[1 + 2im, 3 + 4im, 5 + 6im, 7 + 8im, 9 + 10im]

        itp = cubic_interp(x, y)

        # x promoted to Float64, y promoted to ComplexF64
        @test itp isa CubicInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Mixed Types: Float32 grid + ComplexF64 values
    # ========================================
    @testset "Float32 grid + ComplexF64 values" begin
        x = Float32[0, 1, 2, 3, 4]
        y = ComplexF64[1 + 2im, 3 + 4im, 5 + 6im, 7 + 8im, 9 + 10im]

        itp = cubic_interp(x, y)

        # Grid promoted to Float64 to match Complex{Float64}
        @test itp isa CubicInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Polynomial Reproduction with Complex Coefficients
    # ========================================
    @testset "Polynomial reproduction" begin
        # Test that cubic spline exactly reproduces cubic polynomials
        # Complex polynomial: f(x) = (1+i)*x^3 + (2-i)*x^2 + (3+2i)*x + (4-i)
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        a = 1.0 + 1.0im
        b = 2.0 - 1.0im
        c = 3.0 + 2.0im
        d = 4.0 - 1.0im
        f(t) = a * t^3 + b * t^2 + c * t + d
        y = f.(x)

        # CubicFit BC should reproduce cubic polynomials
        itp = cubic_interp(x, y; bc = CubicFit())

        # Test interpolation at various points
        @test itp(0.5) ≈ f(0.5) atol = 1.0e-10
        @test itp(1.5) ≈ f(1.5) atol = 1.0e-10
        @test itp(2.5) ≈ f(2.5) atol = 1.0e-10
        @test itp(3.5) ≈ f(3.5) atol = 1.0e-10
    end

    # ========================================
    # Derivatives Return Complex
    # ========================================
    @testset "Derivatives return Complex" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        # All derivatives return ComplexF64
        @test itp(0.5; deriv = DerivOp(0)) isa ComplexF64
        @test itp(0.5; deriv = DerivOp(1)) isa ComplexF64
        @test itp(0.5; deriv = DerivOp(2)) isa ComplexF64
        @test itp(0.5; deriv = DerivOp(3)) isa ComplexF64
    end

    # ========================================
    # Derivative Accuracy
    # ========================================
    @testset "Derivative accuracy" begin
        # Use cubic polynomial for exact derivatives
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        a = 1.0 + 0.5im
        b = 2.0 - 1.0im
        c = 3.0 + 2.0im
        d = 1.0 - 0.5im
        f(t) = a * t^3 + b * t^2 + c * t + d
        f1(t) = 3 * a * t^2 + 2 * b * t + c  # First derivative
        f2(t) = 6 * a * t + 2 * b           # Second derivative
        f3(t) = 6 * a                    # Third derivative
        y = f.(x)

        itp = cubic_interp(x, y; bc = CubicFit())

        # Test at interior point
        t = 1.5
        @test itp(t; deriv = DerivOp(0)) ≈ f(t) atol = 1.0e-10
        @test itp(t; deriv = DerivOp(1)) ≈ f1(t) atol = 1.0e-10
        @test itp(t; deriv = DerivOp(2)) ≈ f2(t) atol = 1.0e-10
        @test itp(t; deriv = DerivOp(3)) ≈ f3(t) atol = 1.0e-10
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation with Complex" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        # ClampExtrap() extrapolation
        itp_const = cubic_interp(x, y; extrap = ClampExtrap())
        @test itp_const(-0.5) == y[1]
        @test itp_const(4.5) == y[end]

        # ExtendExtrap() extrapolation
        itp_ext = cubic_interp(x, y; extrap = ExtendExtrap())
        @test itp_ext(-0.5) isa ComplexF64
        @test itp_ext(4.5) isa ComplexF64
    end

    # ========================================
    # Vector Evaluation
    # ========================================
    @testset "Vector evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        # Allocating call
        x_query = [0.5, 1.5, 2.5]
        vals = itp(x_query)
        @test vals isa Vector{ComplexF64}
        @test length(vals) == 3
    end

    # ========================================
    # Broadcast Evaluation
    # ========================================
    @testset "Broadcast evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        # Broadcast
        x_query = [0.5, 1.5, 2.5]
        vals = itp.(x_query)
        @test vals isa Vector{ComplexF64}
        @test length(vals) == 3
    end

    # ========================================
    # Type Stability
    # ========================================
    @testset "Type stability" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        # Type-stable scalar evaluation
        @test (@inferred itp(0.5)) isa ComplexF64
        @test (@inferred itp(0.5; deriv = DerivOp(1))) isa ComplexF64
    end

    # ========================================
    # Grid Point Reproduction
    # ========================================
    @testset "Grid point reproduction" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        # Interpolant should reproduce exact values at grid points
        for i in eachindex(x)
            @test itp(x[i]) ≈ y[i] atol = 1.0e-14
        end
    end

    # ========================================
    # Backward Compatibility with Real Values
    # ========================================
    @testset "Backward compatibility with Real" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]  # y = (x+1)^2

        itp = cubic_interp(x, y)

        # Should still work with real values
        @test itp isa CubicInterpolant{Float64, Float64}
        @test itp(0.5) isa Float64
    end

    # ========================================
    # Different BC Modes
    # ========================================
    @testset "Boundary condition modes" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        # ZeroCurvBC
        itp_natural = cubic_interp(x, y; bc = ZeroCurvBC())
        @test value_type(itp_natural) == ComplexF64

        # ZeroSlopeBC
        itp_clamped = cubic_interp(x, y; bc = ZeroSlopeBC())
        @test value_type(itp_clamped) == ComplexF64

        # CubicFit
        itp_cubicfit = cubic_interp(x, y; bc = CubicFit())
        @test value_type(itp_cubicfit) == ComplexF64

        # QuadraticFit
        itp_quadfit = cubic_interp(x, y; bc = QuadraticFit())
        @test value_type(itp_quadfit) == ComplexF64
    end

    # ========================================
    # BC Type Promotion
    # ========================================
    @testset "BC type promotion" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_complex = [1.0 + 1.0im, 2.0 + 2.0im, 3.0 + 3.0im, 4.0 + 4.0im, 5.0 + 5.0im]

        # Real BC with Complex y - should promote BC values
        bc = BCPair(Deriv1(1.0), Deriv2(0.0))
        itp = cubic_interp(x, y_complex; bc = bc)
        @test value_type(itp) == ComplexF64
        @test itp(0.5) isa ComplexF64
    end

    # ========================================
    # Periodic BC with Complex
    # ========================================
    @testset "Periodic BC with Complex" begin
        # Create periodic data
        x = collect(range(0.0, 2π, 101))
        y = exp.(im .* x)  # Complex exponential
        y[end] = y[1]  # Ensure periodicity

        itp = cubic_interp(x, y; bc = PeriodicBC())

        @test value_type(itp) == ComplexF64
        @test itp(0.5) isa ComplexF64

        # Test periodicity
        @test itp(0.0) ≈ itp(2π) atol = 1.0e-12
    end

    # ========================================
    # Zero Allocation (Scalar)
    # ========================================
    @testset "Zero allocation (scalar)" begin
        x = collect(range(0.0, 10.0, 101))
        y = rand(ComplexF64, 101)
        itp = cubic_interp(x, y)

        # Warmup
        itp(0.5)
        itp(0.5)

        # Test allocation
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # In-Place Vector Evaluation
    # ========================================
    @testset "In-place vector evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im, 7.0 + 8.0im, 9.0 + 10.0im]

        itp = cubic_interp(x, y)

        x_query = [0.5, 1.5, 2.5]
        output = Vector{ComplexF64}(undef, 3)

        # In-place call
        result = itp(output, x_query)
        @test result === output
        @test all(isfinite, output)
    end

end
