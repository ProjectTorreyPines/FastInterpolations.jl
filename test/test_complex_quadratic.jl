# ========================================
# Complex Quadratic Interpolation Tests
# ========================================
# Tests for native Complex number support in QuadraticInterpolant.
# Validates the Tg/Tv type separation design.

using Test
using FastInterpolations

@testset "Complex Quadratic Interpolation" begin

    # ========================================
    # Basic Complex Interpolation
    # ========================================
    @testset "ComplexF64 values" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]

        itp = quadratic_interp(x, y)

        # Type checks
        @test itp isa QuadraticInterpolant{Float64, ComplexF64}
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
        x = Float32[0, 1, 2, 3]
        y = ComplexF32[1+2im, 3+4im, 5+6im, 7+8im]

        itp = quadratic_interp(x, y)

        @test itp isa QuadraticInterpolant{Float32, ComplexF32}
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
        x = 0:3  # Range{Int}
        y = Complex{Int}[1+2im, 3+4im, 5+6im, 7+8im]

        itp = quadratic_interp(x, y)

        # x promoted to Float64, y promoted to ComplexF64
        @test itp isa QuadraticInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Mixed Types: Float32 grid + ComplexF64 values
    # ========================================
    @testset "Float32 grid + ComplexF64 values" begin
        x = Float32[0, 1, 2, 3]
        y = ComplexF64[1+2im, 3+4im, 5+6im, 7+8im]

        itp = quadratic_interp(x, y)

        # Grid promoted to Float64 to match Complex{Float64}
        @test itp isa QuadraticInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Polynomial Reproduction with Complex Coefficients
    # ========================================
    @testset "Polynomial reproduction" begin
        # Test that quadratic spline exactly reproduces quadratic polynomials
        # Complex polynomial: f(x) = (1+i)*x^2 + (2-i)*x + (3+2i)
        x = [0.0, 1.0, 2.0, 3.0]
        a_coef = 1.0 + 1.0im
        b_coef = 2.0 - 1.0im
        c_coef = 3.0 + 2.0im
        f(t) = a_coef*t^2 + b_coef*t + c_coef
        y = f.(x)

        itp = quadratic_interp(x, y)

        # Test interpolation at various points
        @test itp(0.5) ≈ f(0.5) atol=1e-12
        @test itp(1.5) ≈ f(1.5) atol=1e-12
        @test itp(2.5) ≈ f(2.5) atol=1e-12
    end

    # ========================================
    # Derivatives with Complex Values
    # ========================================
    @testset "Derivatives return Complex" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]

        itp = quadratic_interp(x, y)

        # First derivative should return ComplexF64
        d1 = itp(0.5; deriv=DerivOp(1))
        @test d1 isa ComplexF64

        # Second derivative should return ComplexF64
        d2 = itp(0.5; deriv=DerivOp(2))
        @test d2 isa ComplexF64

        # Third derivative should be zero (quadratic)
        d3 = itp(0.5; deriv=DerivOp(3))
        @test d3 isa ComplexF64
        @test d3 == zero(ComplexF64)
    end

    # ========================================
    # Derivative Accuracy with Complex Polynomial
    # ========================================
    @testset "Derivative accuracy" begin
        # Complex quadratic: f(x) = (1+i)*x^2 + (2-i)*x + (3+2i)
        # f'(x) = 2*(1+i)*x + (2-i)
        # f''(x) = 2*(1+i)
        x = [0.0, 1.0, 2.0, 3.0]
        a_coef = 1.0 + 1.0im
        b_coef = 2.0 - 1.0im
        c_coef = 3.0 + 2.0im
        f(t) = a_coef*t^2 + b_coef*t + c_coef
        df(t) = 2*a_coef*t + b_coef
        d2f(t) = 2*a_coef
        y = f.(x)

        itp = quadratic_interp(x, y)

        # Check first derivative
        @test itp(0.5; deriv=DerivOp(1)) ≈ df(0.5) atol=1e-12
        @test itp(1.5; deriv=DerivOp(1)) ≈ df(1.5) atol=1e-12

        # Check second derivative
        @test itp(0.5; deriv=DerivOp(2)) ≈ d2f(0.5) atol=1e-12
        @test itp(1.5; deriv=DerivOp(2)) ≈ d2f(1.5) atol=1e-12
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation modes" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        # :constant mode
        itp_const = quadratic_interp(x, y; extrap=ClampExtrap())
        @test itp_const(-1.0) == 1.0+1.0im  # Clamped to first
        @test itp_const(5.0) == 4.0+4.0im   # Clamped to last

        # :extension mode
        itp_ext = quadratic_interp(x, y; extrap=ExtendExtrap())
        val_ext = itp_ext(-1.0)
        @test val_ext isa ComplexF64
    end

    # ========================================
    # Vector Evaluation
    # ========================================
    @testset "Vector evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = quadratic_interp(x, y)

        xq = [0.5, 1.5, 2.5]
        vals = itp(xq)

        @test vals isa Vector{ComplexF64}
        @test length(vals) == 3
    end

    # ========================================
    # Broadcast Evaluation
    # ========================================
    @testset "Broadcast evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = quadratic_interp(x, y)

        xq = [0.5, 1.5, 2.5]
        vals = itp.(xq)

        @test vals isa Vector{ComplexF64}
    end

    # ========================================
    # Type Stability
    # ========================================
    @testset "Type stability" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = quadratic_interp(x, y)

        # Scalar evaluation should be type-stable
        @test @inferred(itp(0.5)) isa ComplexF64

        # First derivative should be type-stable
        @test @inferred(itp(0.5; deriv=DerivOp(1))) isa ComplexF64
    end

    # ========================================
    # Zero Allocation (Scalar)
    # ========================================
    @testset "Zero allocation (scalar)" begin
        x = collect(range(0.0, 10.0, 101))
        y = rand(ComplexF64, 101)
        itp = quadratic_interp(x, y)

        # Warmup
        itp(0.5)
        itp(0.5)

        # Measure allocation
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # In-place Vector Evaluation
    # ========================================
    @testset "In-place vector evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = quadratic_interp(x, y)

        xq = [0.5, 1.5, 2.5]
        output = Vector{ComplexF64}(undef, 3)

        itp(output, xq)

        @test output isa Vector{ComplexF64}
        @test length(output) == 3
    end

    # ========================================
    # BC Modes: QuadraticFit
    # ========================================
    @testset "BC mode: QuadraticFit" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

        # Left QuadraticFit (default)
        itp_left = quadratic_interp(x, y; bc=Left(QuadraticFit()))
        val_left = itp_left(0.5)
        @test val_left isa ComplexF64

        # Right QuadraticFit
        itp_right = quadratic_interp(x, y; bc=Right(QuadraticFit()))
        val_right = itp_right(0.5)
        @test val_right isa ComplexF64
    end

    # ========================================
    # BC Modes: MinCurvFit
    # ========================================
    @testset "BC mode: MinCurvFit" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

        itp = quadratic_interp(x, y; bc=MinCurvFit())

        val = itp(0.5)
        @test val isa ComplexF64

        d1 = itp(0.5; deriv=DerivOp(1))
        @test d1 isa ComplexF64
    end

    # ========================================
    # Grid Point Behavior
    # ========================================
    @testset "Grid point behavior" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = quadratic_interp(x, y)

        # At exact grid points, should return y values
        @test itp(0.0) ≈ 1.0+1.0im atol=1e-14
        @test itp(1.0) ≈ 2.0+2.0im atol=1e-14
        @test itp(2.0) ≈ 3.0+3.0im atol=1e-14
        @test itp(3.0) ≈ 4.0+4.0im atol=1e-14
    end

    # ========================================
    # Real-valued Backward Compatibility
    # ========================================
    @testset "Real-valued backward compatibility" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]

        itp = quadratic_interp(x, y)

        # Type checks for backward compatibility
        @test itp isa QuadraticInterpolant{Float64, Float64}
        @test grid_type(itp) == Float64
        @test value_type(itp) == Float64
        @test eval_type(itp, Float64) == Float64

        val = itp(0.5)
        @test val isa Float64

        d1 = itp(0.5; deriv=DerivOp(1))
        @test d1 isa Float64
    end

    # ========================================
    # BC Type Promotion and Conversion Tests
    # ========================================

    @testset "BC Type Promotion Edge Cases" begin
        x = [0.0, 1.0, 2.0, 3.0]

        @testset "Complex y + Real BC (promotion)" begin
            # y is Complex, BC value is Real → should promote BC to Complex
            y_complex = [1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

            # Real BC value (Float64) with Complex y
            bc = Left(Deriv1(2.0))  # Deriv1{Float64}
            itp = quadratic_interp(x, y_complex; bc=bc)

            @test value_type(itp) == ComplexF64
            val = itp(0.5)
            @test val isa ComplexF64

            # Derivative should be promoted to Complex (with zero imaginary)
            d1 = itp(x[1]; deriv=DerivOp(1))
            @test d1 isa ComplexF64
            @test real(d1) ≈ 2.0
            @test imag(d1) ≈ 0.0 atol=1e-14
        end

        @testset "Real y + Complex BC (zero imaginary)" begin
            # y is Real, BC is Complex with zero imaginary → should extract real part
            y_real = [1.0, 2.0, 5.0, 10.0]

            # Complex BC with zero imaginary part
            bc = Left(Deriv1(2.0 + 0.0im))  # Deriv1{ComplexF64}
            itp = quadratic_interp(x, y_real; bc=bc)

            @test value_type(itp) == Float64  # y determines value type
            val = itp(0.5)
            @test val isa Float64

            # Derivative should be Real (Complex BC converted)
            d1 = itp(x[1]; deriv=DerivOp(1))
            @test d1 isa Float64
            @test d1 ≈ 2.0
        end

        @testset "Real y + Complex BC (non-zero imaginary) → ERROR" begin
            # y is Real, BC is Complex with non-zero imaginary → should throw
            y_real = [1.0, 2.0, 5.0, 10.0]

            # Complex BC with non-zero imaginary - cannot convert to Float64
            bc = Left(Deriv1(2.0 + 1.0im))  # Deriv1{ComplexF64}

            # Should throw InexactError during construction
            @test_throws InexactError quadratic_interp(x, y_real; bc=bc)
        end

        @testset "Explicit Complex BC with Complex y" begin
            # Both y and BC are Complex - exact type match path
            y_complex = [1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

            # Deriv1 with explicit Complex value
            bc = Left(Deriv1(2.0 + 1.0im))  # Deriv1{ComplexF64}
            itp = quadratic_interp(x, y_complex; bc=bc)

            @test value_type(itp) == ComplexF64
            d1 = itp(x[1]; deriv=DerivOp(1))
            @test d1 ≈ 2.0 + 1.0im

            # Deriv2 with explicit Complex value
            bc2 = Left(Deriv2(0.5 + 0.5im))
            itp2 = quadratic_interp(x, y_complex; bc=bc2)
            @test itp2(0.5) isa ComplexF64
        end

        @testset "Right endpoint BC type conversion" begin
            y_complex = [1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

            # Real BC on Complex y (Right endpoint)
            bc = Right(Deriv1(3.0))  # Float64
            itp = quadratic_interp(x, y_complex; bc=bc)

            d_end = itp(x[end]; deriv=DerivOp(1))
            @test d_end isa ComplexF64
            @test real(d_end) ≈ 3.0
            @test imag(d_end) ≈ 0.0 atol=1e-14
        end
    end

    @testset "BC Dispatch Path Verification" begin
        # Verify that exact type match vs generic dispatch both work correctly
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

        @testset "Exact type match path" begin
            # BC type matches y type exactly → _fill_slopes!(d, s, h, bc::Left{Deriv1{Tv}}, ...)
            bc = Left(Deriv1(2.0 + 1.0im))  # Deriv1{ComplexF64} matches ComplexF64 y
            itp = quadratic_interp(x, y; bc=bc)
            @test itp(0.5) isa ComplexF64
        end

        @testset "Generic dispatch with conversion" begin
            # BC type differs from y type → _fill_slopes!(d, s, h, bc::Left{<:Deriv1}, ...)
            bc = Left(Deriv1(2.0))  # Deriv1{Float64} needs convert to ComplexF64
            itp = quadratic_interp(x, y; bc=bc)
            @test itp(0.5) isa ComplexF64
        end

        @testset "Evaluation is zero-allocation" begin
            bc = Left(Deriv1(2.0 + 1.0im))
            itp = quadratic_interp(x, y; bc=bc)

            # Warm up
            _ = itp(0.5)

            # Evaluation should be zero-allocation
            allocs = @allocated itp(0.5)
            @test allocs == 0
        end
    end

    @testset "Float32 type preservation with BC" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        y32 = ComplexF32[1.0+1.0im, 2.0+2.0im, 5.0+5.0im, 10.0+10.0im]

        @testset "Float32 BC promoted to ComplexF32" begin
            bc = Left(Deriv1(2.0f0))  # Deriv1{Float32}
            itp = quadratic_interp(x32, y32; bc=bc)

            @test grid_type(itp) == Float32
            @test value_type(itp) == ComplexF32
            @test eval_type(itp, Float32) == ComplexF32
            @test itp(0.5f0) isa ComplexF32
        end

        @testset "Float64 BC with Float32 y" begin
            # Float64 BC should be demoted to Float32 to match y
            bc = Left(Deriv1(2.0))  # Deriv1{Float64}
            itp = quadratic_interp(x32, y32; bc=bc)

            @test value_type(itp) == ComplexF32
            d1 = itp(x32[1]; deriv=DerivOp(1))
            @test d1 isa ComplexF32
        end
    end

end
