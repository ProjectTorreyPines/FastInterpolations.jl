# ========================================
# Complex Cubic Series Interpolation Tests
# ========================================
# Tests for native Complex number support in CubicSeriesInterpolant.
# Validates the Tg/Tv type separation design for series interpolants.

using Test
using FastInterpolations

@testset "Complex Cubic Series Interpolation" begin

    # ========================================
    # Basic Complex Series Interpolation
    # ========================================
    @testset "ComplexF64 values" begin
        x = range(0.0, 1.0, 11)
        # Complex exponential and linear functions
        y1 = exp.(2im .* π .* x)
        y2 = (1.0 + 2.0im) .* collect(x)

        sitp = cubic_interp(x, [y1, y2])

        # Type checks
        @test sitp isa CubicSeriesInterpolant{Float64, ComplexF64}
        @test grid_type(sitp) == Float64
        @test value_type(sitp) == ComplexF64

        # Scalar evaluation returns Vector{ComplexF64}
        vals = sitp(0.5)
        @test vals isa Vector{ComplexF64}
        @test length(vals) == 2

        # Check approximate correctness
        # y1 at x=0.5: exp(2im*π*0.5) = exp(im*π) = -1 (spline may differ slightly)
        @test isapprox(vals[1], -1.0 + 0.0im, rtol=0.1)
        # y2 at x=0.5: (1+2im)*0.5 = 0.5+1.0im (linear function fits exactly)
        @test isapprox(vals[2], 0.5 + 1.0im, atol=1e-10)
    end

    # ========================================
    # ComplexF32 Support
    # ========================================
    @testset "ComplexF32 values" begin
        x = range(0.0f0, 1.0f0, 11)
        y1 = Complex{Float32}.(exp.(2im .* π .* Float32.(x)))
        y2 = Complex{Float32}.((1.0f0 + 2.0f0im) .* collect(x))

        sitp = cubic_interp(x, [y1, y2])

        @test sitp isa CubicSeriesInterpolant{Float32, ComplexF32}
        @test grid_type(sitp) == Float32
        @test value_type(sitp) == ComplexF32

        vals = sitp(0.5f0)
        @test vals isa Vector{ComplexF32}
    end

    # ========================================
    # Integer Grid with Complex Values
    # ========================================
    @testset "Integer grid + Complex values" begin
        x = 0:10  # Range{Int}
        y1 = Complex{Int}[i + 2im*i for i in 0:10]  # Linear: (1+2i)*x
        y2 = Complex{Int}[2i + 1im*i for i in 0:10]  # Linear: (2+i)*x

        sitp = cubic_interp(x, [y1, y2])

        # x promoted to Float64, y promoted to ComplexF64
        @test sitp isa CubicSeriesInterpolant{Float64, ComplexF64}

        vals = sitp(5.5)
        @test vals isa Vector{ComplexF64}

        # For linear function, cubic spline interpolation should be exact
        @test isapprox(vals[1], (1.0 + 2.0im) * 5.5, rtol=1e-10)
    end

    # ========================================
    # Mixed Types: Float32 grid + ComplexF64 values
    # ========================================
    @testset "Float32 grid + ComplexF64 values" begin
        x = range(0.0f0, 1.0f0, 11)
        y1 = ComplexF64.(exp.(2im .* π .* x))  # ComplexF64
        y2 = ComplexF64.((1.0 + 2.0im) .* x)

        sitp = cubic_interp(x, [y1, y2])

        # Grid stays Float32, values are ComplexF64
        @test sitp isa CubicSeriesInterpolant{Float32, ComplexF64}

        vals = sitp(0.5)
        @test vals isa Vector{ComplexF64}
    end

    # ========================================
    # Matrix Input
    # ========================================
    @testset "Matrix input with Complex values" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)  # Linear
        y2 = (2.0 - 1.0im) .* collect(x)  # Linear
        Y = hcat(y1, y2)  # 11×2 matrix

        sitp = cubic_interp(x, Y)

        @test sitp isa CubicSeriesInterpolant{Float64, ComplexF64}
        @test length(sitp(0.5)) == 2  # Two series

        vals = sitp(0.5)
        @test isapprox(vals[1], 0.5 + 1.0im, atol=1e-10)
        @test isapprox(vals[2], 1.0 - 0.5im, atol=1e-10)
    end

    # ========================================
    # Vector Evaluation
    # ========================================
    @testset "Vector evaluation" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)  # Linear complex function
        y2 = (2.0 - 1.0im) .* collect(x)

        sitp = cubic_interp(x, [y1, y2])

        # Vector query
        xq = [0.25, 0.5, 0.75]
        results = sitp(xq)

        @test length(results) == 2  # Two series
        @test all(r -> r isa Vector{ComplexF64}, results)
        @test all(r -> length(r) == 3, results)

        # Check values (linear function interpolated exactly by cubic)
        @test isapprox(results[1][1], (1.0 + 2.0im) * 0.25, rtol=1e-10)
        @test isapprox(results[1][2], (1.0 + 2.0im) * 0.5, rtol=1e-10)
        @test isapprox(results[2][3], (2.0 - 1.0im) * 0.75, rtol=1e-10)
    end

    # ========================================
    # In-place Evaluation
    # ========================================
    @testset "In-place scalar evaluation" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)
        y2 = (2.0 - 1.0im) .* collect(x)

        sitp = cubic_interp(x, [y1, y2])

        output = Vector{ComplexF64}(undef, 2)
        sitp(output, 0.5)

        @test output[1] isa ComplexF64
        @test isapprox(output[1], (1.0 + 2.0im) * 0.5, rtol=1e-10)
        @test isapprox(output[2], (2.0 - 1.0im) * 0.5, rtol=1e-10)
    end

    @testset "In-place vector evaluation" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)
        y2 = (2.0 - 1.0im) .* collect(x)

        sitp = cubic_interp(collect(x), [y1, y2])

        xq = collect(range(0.1, 0.9, 5))
        outputs = [Vector{ComplexF64}(undef, 5) for _ in 1:2]

        sitp(outputs, xq)

        @test outputs[1][3] isa ComplexF64
        @test isapprox(outputs[1][3], (1.0 + 2.0im) * 0.5, rtol=1e-10)
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation modes" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)
        y2 = (2.0 - 1.0im) .* collect(x)

        # Extension mode
        sitp_ext = cubic_interp(x, [y1, y2]; extrap=:extension)
        vals_ext = sitp_ext(1.5)  # Beyond domain
        @test vals_ext isa Vector{ComplexF64}
        # Linear function extended by cubic spline
        @test isapprox(vals_ext[1], (1.0 + 2.0im) * 1.5, rtol=1e-10)

        # Constant mode
        sitp_const = cubic_interp(x, [y1, y2]; extrap=:constant)
        vals_const = sitp_const(1.5)  # Beyond domain
        @test vals_const isa Vector{ComplexF64}
        @test isapprox(vals_const[1], y1[end], rtol=1e-10)
    end

    # ========================================
    # Type Stability
    # ========================================
    @testset "Type stability" begin
        x = range(0.0, 1.0, 11)
        y1 = rand(ComplexF64, 11)
        y2 = rand(ComplexF64, 11)

        sitp = cubic_interp(x, [y1, y2])

        # Scalar evaluation should be type-stable
        @test @inferred(sitp(0.5)) isa Vector{ComplexF64}
    end

    # ========================================
    # Zero Allocation (Scalar In-place)
    # ========================================
    @testset "Zero allocation (scalar in-place)" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = rand(ComplexF64, 101)
        y2 = rand(ComplexF64, 101)

        sitp = cubic_interp(x, [y1, y2])
        output = Vector{ComplexF64}(undef, 2)

        # Warmup
        sitp(output, 0.5)
        sitp(output, 0.5)

        # Measure allocation
        allocs = @allocated sitp(output, 0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Derivative Support
    # ========================================
    @testset "Derivative for Complex values" begin
        x = range(0.0, 1.0, 21)
        # Cubic complex function: y = (1+i)x^3 - (2+3i)x^2 + (1-i)x + (2+2i)
        a = 1.0 + 1.0im   # x^3 coefficient
        b = -2.0 - 3.0im  # x^2 coefficient
        c = 1.0 - 1.0im   # x coefficient
        d = 2.0 + 2.0im   # constant
        y = [a*xi^3 + b*xi^2 + c*xi + d for xi in x]

        sitp = cubic_interp(x, [y])

        # First derivative: 3ax^2 + 2bx + c
        xq = 0.5
        d1 = sitp(xq; deriv=1)
        expected_d1 = 3*a*xq^2 + 2*b*xq + c
        @test d1 isa Vector{ComplexF64}
        @test isapprox(d1[1], expected_d1, rtol=0.1)  # Spline approximation

        # Second derivative: 6ax + 2b
        d2 = sitp(xq; deriv=2)
        expected_d2 = 6*a*xq + 2*b
        @test d2 isa Vector{ComplexF64}
        @test isapprox(d2[1], expected_d2, rtol=0.2)  # Spline approximation
    end

    # ========================================
    # Boundary Condition Options
    # ========================================
    @testset "Boundary conditions with Complex values" begin
        x = range(0.0, 1.0, 11)
        y = exp.(2im .* π .* x)  # Complex exponential

        # Natural BC (default)
        sitp_natural = cubic_interp(x, [y]; bc=NaturalBC())
        @test sitp_natural isa CubicSeriesInterpolant{Float64, ComplexF64}

        # Clamped BC
        sitp_clamped = cubic_interp(x, [y]; bc=ClampedBC())
        @test sitp_clamped isa CubicSeriesInterpolant{Float64, ComplexF64}

        # Both should produce valid interpolations
        @test sitp_natural(0.5) isa Vector{ComplexF64}
        @test sitp_clamped(0.5) isa Vector{ComplexF64}
    end

    # ========================================
    # Real-valued Backward Compatibility
    # ========================================
    @testset "Real-valued backward compatibility" begin
        # Ensure the changes don't break real-valued interpolation
        x = range(0.0, 1.0, 11)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        sitp = cubic_interp(x, [y1, y2])

        # Type checks for backward compatibility
        @test sitp isa CubicSeriesInterpolant{Float64, Float64}
        @test grid_type(sitp) == Float64
        @test value_type(sitp) == Float64

        vals = sitp(0.5)
        @test vals isa Vector{Float64}
    end

    # ========================================
    # Tg Calculation Policy (Query Independence)
    # ========================================
    @testset "Tg from x/y only, not query" begin
        # Float32 data + Float64 query → Float32 output
        x32 = Float32.(0:0.1:1)
        y1 = sin.(x32)
        y2 = cos.(x32)

        sitp = cubic_interp(x32, [y1, y2])
        @test sitp isa CubicSeriesInterpolant{Float32, Float32}

        # Float64 query should return Float32 (Tg from x/y)
        result = sitp(0.5)  # 0.5 is Float64
        @test eltype(result) === Float32
    end

    # ========================================
    # Complex Cubic Function Accuracy
    # ========================================
    @testset "Complex cubic function accuracy" begin
        # Test that cubic spline closely fits a cubic function
        x = collect(0.0:0.05:1.0)  # Fine grid for better fit

        # Complex cubic: y = (1+i)x^3 - (2+3i)x^2 + (1-i)x + (2+2i)
        a, b, c, d = 1.0+1.0im, -2.0-3.0im, 1.0-1.0im, 2.0+2.0im
        y = [a*xi^3 + b*xi^2 + c*xi + d for xi in x]

        sitp = cubic_interp(x, [y])

        # Test at several points (not on grid)
        for xq in [0.15, 0.35, 0.65, 0.85]
            expected = a*xq^3 + b*xq^2 + c*xq + d
            result = sitp(xq)[1]
            @test isapprox(result, expected, rtol=1e-4)
        end
    end

    # ========================================
    # Periodic BC with Complex Values
    # ========================================
    @testset "Periodic BC with Complex values" begin
        # Periodic function: exp(2im*π*x) is periodic on [0, 1]
        x = range(0.0, 1.0, 11)
        y = exp.(2im .* π .* x)

        # Note: For periodic BC, first and last y values should match
        # exp(0) = 1, exp(2im*π) = 1 ✓
        @test isapprox(y[1], y[end], atol=1e-10)

        sitp = cubic_interp(x, [y]; bc=PeriodicBC())
        @test sitp isa CubicSeriesInterpolant{Float64, ComplexF64}

        # Values should be smooth across the boundary
        vals_near_0 = sitp(0.01)
        vals_near_1 = sitp(0.99)
        @test vals_near_0 isa Vector{ComplexF64}
        @test vals_near_1 isa Vector{ComplexF64}
    end

    # ========================================
    # Show Method
    # ========================================
    @testset "Show method for Complex series" begin
        x = range(0.0, 1.0, 11)
        y = exp.(2im .* π .* x)

        sitp = cubic_interp(x, [y])

        # Should display type info correctly
        str = sprint(show, sitp)
        @test occursin("CubicSeriesInterpolant", str)
        @test occursin("Float64", str)
        @test occursin("ComplexF64", str)
    end

end
