# ========================================
# Complex Linear Interpolation Tests
# ========================================
# Tests for native Complex number support in LinearInterpolant.
# Validates the Tg/Tv type separation design.

using Test
using FastInterpolations

@testset "Complex Linear Interpolation" begin

    # ========================================
    # Basic Complex Interpolation
    # ========================================
    @testset "ComplexF64 values" begin
        x = range(0.0, 1.0, 11)
        # Complex exponential: smooth function for interpolation
        y = exp.(2im .* π .* x)

        itp = linear_interp(x, y)

        # Type checks
        @test itp isa LinearInterpolant{Float64, ComplexF64}
        @test grid_type(itp) == Float64
        @test value_type(itp) == ComplexF64
        @test eval_type(itp, Float64) == ComplexF64

        # Evaluation returns ComplexF64
        val = itp(0.5)
        @test val isa ComplexF64

        # Check approximate correctness (linear interpolation of complex values)
        # At x=0.5, we expect interpolation between y[6] and y[7]
        @test isapprox(real(val), real(exp(2im * π * 0.5)), rtol=0.1)
        @test isapprox(imag(val), imag(exp(2im * π * 0.5)), rtol=0.1)

        # First derivative also returns ComplexF64
        d1 = itp(0.5; deriv=1)
        @test d1 isa ComplexF64

        # Second derivative returns zero (linear interpolation)
        d2 = itp(0.5; deriv=2)
        @test d2 isa ComplexF64
        @test d2 == zero(ComplexF64)
    end

    # ========================================
    # ComplexF32 Support
    # ========================================
    @testset "ComplexF32 values" begin
        x = range(0.0f0, 1.0f0, 11)
        y = Complex{Float32}.(exp.(2im .* π .* Float32.(x)))

        itp = linear_interp(x, y)

        @test itp isa LinearInterpolant{Float32, ComplexF32}
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
        x = 0:10  # Range{Int}
        y = Complex{Int}[i + 2im*i for i in 0:10]

        itp = linear_interp(x, y)

        # x promoted to Float64, y promoted to ComplexF64
        @test itp isa LinearInterpolant{Float64, ComplexF64}

        val = itp(5.5)
        @test val isa ComplexF64

        # Check interpolation: between y[6]=(5+10im) and y[7]=(6+12im)
        # At 5.5: (5+10im) + 0.5*(6+12im - 5-10im) = 5.5 + 11im
        @test isapprox(val, 5.5 + 11.0im, rtol=1e-10)
    end

    # ========================================
    # Mixed Types: Float32 grid + ComplexF64 values
    # ========================================
    @testset "Float32 grid + ComplexF64 values" begin
        x = range(0.0f0, 1.0f0, 11)
        y = ComplexF64.(exp.(2im .* π .* x))  # ComplexF64

        itp = linear_interp(x, y)

        # Grid promoted to Float64 to match Complex{Float64}
        @test itp isa LinearInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Vector Evaluation
    # ========================================
    @testset "Vector evaluation" begin
        x = range(0.0, 1.0, 11)
        y = (1.0 + 2.0im) .* x  # Linear complex function

        itp = linear_interp(x, y)

        # Vector query
        xq = [0.25, 0.5, 0.75]
        vals = itp(xq)

        @test vals isa Vector{ComplexF64}
        @test length(vals) == 3

        # Check values
        @test isapprox(vals[1], (1.0 + 2.0im) * 0.25, rtol=1e-10)
        @test isapprox(vals[2], (1.0 + 2.0im) * 0.5, rtol=1e-10)
        @test isapprox(vals[3], (1.0 + 2.0im) * 0.75, rtol=1e-10)
    end

    # ========================================
    # Broadcast Evaluation
    # ========================================
    @testset "Broadcast evaluation" begin
        x = range(0.0, 1.0, 11)
        y = (1.0 + 2.0im) .* x

        itp = linear_interp(x, y)

        # Broadcast query
        xq = [0.25, 0.5, 0.75]
        vals = itp.(xq)

        @test vals isa Vector{ComplexF64}
        @test length(vals) == 3
        @test isapprox(vals[2], (1.0 + 2.0im) * 0.5, rtol=1e-10)
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation modes" begin
        x = range(0.0, 1.0, 11)
        y = (1.0 + 2.0im) .* x

        # Extension mode
        itp_ext = linear_interp(x, y; extrap=ExtendExtrap())
        val_ext = itp_ext(1.5)  # Beyond domain
        @test val_ext isa ComplexF64
        @test isapprox(val_ext, (1.0 + 2.0im) * 1.5, rtol=1e-10)

        # Constant mode
        itp_const = linear_interp(x, y; extrap=ConstExtrap())
        val_const = itp_const(1.5)  # Beyond domain
        @test val_const isa ComplexF64
        @test isapprox(val_const, y[end], rtol=1e-10)

        # Wrap mode
        itp_wrap = linear_interp(x, y; extrap=WrapExtrap())
        val_wrap = itp_wrap(1.5)  # Should wrap to 0.5
        @test val_wrap isa ComplexF64
    end

    # ========================================
    # Type Stability
    # ========================================
    @testset "Type stability" begin
        x = range(0.0, 1.0, 11)
        y_complex = rand(ComplexF64, 11)
        itp = linear_interp(x, y_complex)

        # Scalar evaluation should be type-stable
        @test @inferred(itp(0.5)) isa ComplexF64

        # First derivative should be type-stable
        @test @inferred(itp(0.5; deriv=1)) isa ComplexF64
    end

    # ========================================
    # Zero Allocation (Scalar)
    # ========================================
    @testset "Zero allocation (scalar)" begin
        x = collect(range(0.0, 1.0, 101))
        y = rand(ComplexF64, 101)
        itp = linear_interp(x, y)

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
        x = range(0.0, 1.0, 11)
        y = (1.0 + 2.0im) .* collect(x)

        itp = linear_interp(collect(x), y)

        xq = collect(range(0.1, 0.9, 5))
        output = Vector{ComplexF64}(undef, 5)

        itp(output, xq)

        @test output[1] isa ComplexF64
        @test isapprox(output[3], (1.0 + 2.0im) * 0.5, rtol=1e-10)
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "Edge cases" begin
        x = range(0.0, 1.0, 3)
        y = [1.0 + 0.0im, 0.0 + 1.0im, -1.0 + 0.0im]

        itp = linear_interp(x, y)

        # Query at grid points
        @test isapprox(itp(0.0), y[1], rtol=1e-10)
        @test isapprox(itp(0.5), y[2], rtol=1e-10)
        @test isapprox(itp(1.0), y[3], rtol=1e-10)

        # Query at midpoints
        mid1 = itp(0.25)
        @test isapprox(mid1, 0.5 + 0.5im, rtol=1e-10)

        mid2 = itp(0.75)
        @test isapprox(mid2, -0.5 + 0.5im, rtol=1e-10)
    end

    # ========================================
    # Derivative for Complex Values
    # ========================================
    @testset "Derivative for Complex values" begin
        x = range(0.0, 1.0, 11)
        # Linear complex function: y = (2+3i)*x + (1+1i)
        slope = 2.0 + 3.0im
        intercept = 1.0 + 1.0im
        y = slope .* x .+ intercept

        itp = linear_interp(x, y)

        # First derivative should be the complex slope
        d1 = itp(0.5; deriv=1)
        @test isapprox(d1, slope, rtol=1e-10)

        # Second derivative should be zero
        d2 = itp(0.5; deriv=2)
        @test d2 == zero(ComplexF64)
    end

    # ========================================
    # Real-valued Backward Compatibility
    # ========================================
    @testset "Real-valued backward compatibility" begin
        # Ensure the changes don't break real-valued interpolation
        x = range(0.0, 1.0, 11)
        y = sin.(2π .* x)

        itp = linear_interp(x, y)

        # Type checks for backward compatibility
        @test itp isa LinearInterpolant{Float64, Float64}
        @test grid_type(itp) == Float64
        @test value_type(itp) == Float64
        @test eval_type(itp, Float64) == Float64

        val = itp(0.5)
        @test val isa Float64

        d1 = itp(0.5; deriv=1)
        @test d1 isa Float64
    end

    # ========================================
    # Tg Calculation Policy Tests
    # ========================================
    # POLICY: Tg is computed from x and y ONLY, NOT from query types.
    # This ensures predictable output types regardless of query precision.
    @testset "Tg calculation policy - query type independence" begin

        # -----------------------------------------
        # Test 1: Int data + Float32 query → Float64 output
        # -----------------------------------------
        # Tg = float(Int) = Float64, regardless of Float32 query
        @testset "Int data + Float32 query → Float64" begin
            x_int = 0:10
            y_int = x_int .^ 2

            # 2-arg constructor (interpolant)
            itp = linear_interp(x_int, y_int)
            @test itp isa LinearInterpolant{Float64, Float64}  # Tg = Float64 from Int

            # Scalar query with Float32 - output should still be Float64
            result = itp(Float32(5.5))
            @test result isa Float64  # NOT Float32
            @test isapprox(result, 5.5^2, rtol=0.1)  # Approximate quadratic

            # 3-arg oneshot scalar
            result_oneshot = linear_interp(x_int, y_int, Float32(5.5))
            @test result_oneshot isa Float64
        end

        # -----------------------------------------
        # Test 2: Float32 data + Float64 query → Float64 output (lossless promotion)
        # -----------------------------------------
        # promote_type(Float32, Float64) = Float64 (wider type wins)
        @testset "Float32 data + Float64 query → Float64 (lossless)" begin
            x32 = Float32.(0:0.1:1)
            y32 = sin.(x32)

            # 2-arg constructor
            itp = linear_interp(x32, y32)
            @test itp isa LinearInterpolant{Float32, Float32}

            # Float64 query promotes output to Float64 (lossless - wider type)
            result = itp(0.5)  # 0.5 is Float64
            @test result isa Float64  # wider type wins

            # 3-arg oneshot scalar - still uses internal Tv
            result_oneshot = linear_interp(x32, y32, 0.5)
            @test result_oneshot isa Float32  # oneshot uses internal Tv

            # Vector query with Float64 elements → Float64 output
            xq = [0.25, 0.5, 0.75]  # Vector{Float64}
            results = itp(xq)
            @test eltype(results) === Float64
        end

        # -----------------------------------------
        # Test 3: Int data + Complex query type invariance
        # -----------------------------------------
        # Complex values should also respect the x/y policy
        @testset "Int grid + Complex values → ComplexF64" begin
            x_int = 0:10
            y_complex = Complex{Int}[i + 2im*i for i in 0:10]

            itp = linear_interp(x_int, y_complex)
            @test itp isa LinearInterpolant{Float64, ComplexF64}

            # Float32 query still returns ComplexF64
            result = itp(Float32(5.5))
            @test result isa ComplexF64
        end

        # -----------------------------------------
        # Test 4: Oneshot in-place API respects policy
        # -----------------------------------------
        @testset "linear_interp! respects Tg policy" begin
            x_int = 0:10
            y_int = collect(x_int .^ 2)  # Vector for in-place

            # Output should be Float64 (from Int data)
            xq = Float32[2.5f0, 5.5f0, 7.5f0]
            output = Vector{Float64}(undef, 3)

            linear_interp!(output, x_int, y_int, xq)

            @test eltype(output) === Float64
            @test isapprox(output[2], 5.5^2, rtol=0.1)
        end

        # -----------------------------------------
        # Test 5: Mixed Float32/Float64 in y promotes to Float64
        # -----------------------------------------
        # When x is Float32 but y is Float64, Tg = Float64
        @testset "Float32 x + Float64 y → Float64" begin
            x32 = Float32.(0:0.1:1)
            y64 = Float64.(sin.(x32))  # Float64 values

            itp = linear_interp(x32, y64)
            @test itp isa LinearInterpolant{Float64, Float64}  # Tg promoted to Float64

            result = itp(0.5f0)  # Float32 query
            @test result isa Float64
        end
    end

end
