# ========================================
# Complex Constant Series Interpolation Tests
# ========================================
# Tests for native Complex number support in ConstantSeriesInterpolant.
# Validates the Tg/Tv type separation design for series interpolants.

using Test
using FastInterpolations

@testset "Complex Constant Series Interpolation" begin

    # ========================================
    # Basic Complex Series Interpolation
    # ========================================
    @testset "ComplexF64 values" begin
        x = range(0.0, 1.0, 11)
        # Complex exponential and linear functions
        y1 = exp.(2im .* π .* x)
        y2 = (1.0 + 2.0im) .* collect(x)

        sitp = constant_interp(x, Series(y1, y2))

        # Type checks
        @test sitp isa ConstantSeriesInterpolant{Float64, ComplexF64}
        @test grid_type(sitp) == Float64
        @test value_type(sitp) == ComplexF64
        @test eval_type(sitp, Float64) == ComplexF64

        # Scalar evaluation returns Vector{ComplexF64}
        vals = sitp(0.5)
        @test vals isa Vector{ComplexF64}
        @test length(vals) == 2
    end

    # ========================================
    # ComplexF32 Support
    # ========================================
    @testset "ComplexF32 values" begin
        x = range(0.0f0, 1.0f0, 11)
        y1 = Complex{Float32}.(exp.(2im .* π .* Float32.(x)))
        y2 = Complex{Float32}.((1.0f0 + 2.0f0im) .* collect(x))

        sitp = constant_interp(x, Series(y1, y2))

        @test sitp isa ConstantSeriesInterpolant{Float32, ComplexF32}
        @test grid_type(sitp) == Float32
        @test value_type(sitp) == ComplexF32
        @test eval_type(sitp, Float32) == ComplexF32

        vals = sitp(0.5f0)
        @test vals isa Vector{ComplexF32}
    end

    # ========================================
    # Integer Grid with Complex Values
    # ========================================
    @testset "Integer grid + Complex values" begin
        x = 0:10  # Range{Int}
        y1 = Complex{Int}[i + 2im * i for i in 0:10]
        y2 = Complex{Int}[2i + 1im * i for i in 0:10]

        sitp = constant_interp(x, Series(y1, y2))

        # x promoted to Float64, y promoted to ComplexF64
        @test sitp isa ConstantSeriesInterpolant{Float64, ComplexF64}

        vals = sitp(5.5)
        @test vals isa Vector{ComplexF64}

        # Constant interpolation returns step value (nearest by default)
        # At 5.5 with NearestSide(), rounds to index 6 (x=5) → value 5+10i
        @test isapprox(vals[1], 5.0 + 10.0im, rtol = 1.0e-10)
    end

    # ========================================
    # Mixed Types: Float32 grid + ComplexF64 values
    # ========================================
    @testset "Float32 grid + ComplexF64 values" begin
        x = range(0.0f0, 1.0f0, 11)
        y1 = ComplexF64.(exp.(2im .* π .* x))  # ComplexF64
        y2 = ComplexF64.((1.0 + 2.0im) .* x)

        sitp = constant_interp(x, Series(y1, y2))

        # Grid promoted to Float64 to match Complex{Float64}
        @test sitp isa ConstantSeriesInterpolant{Float64, ComplexF64}

        vals = sitp(0.5)
        @test vals isa Vector{ComplexF64}
    end

    # ========================================
    # Matrix Input
    # ========================================
    @testset "Matrix input with Complex values" begin
        x = range(0.0, 1.0, 11)
        y1 = exp.(2im .* π .* x)
        y2 = (1.0 + 2.0im) .* collect(x)
        Y = hcat(collect(y1), y2)  # 11×2 matrix

        sitp = constant_interp(x, Series(Y))

        @test sitp isa ConstantSeriesInterpolant{Float64, ComplexF64}
        @test length(sitp(0.5)) == 2  # Two series
    end

    # ========================================
    # Side Options (NearestSide, LeftSide, RightSide)
    # ========================================
    @testset "Side options with Complex values" begin
        x = collect(0.0:1.0:5.0)  # [0, 1, 2, 3, 4, 5]
        y1 = ComplexF64[1 + 1im, 2 + 2im, 3 + 3im, 4 + 4im, 5 + 5im, 6 + 6im]

        # Test LeftSide()
        sitp_left = constant_interp(x, Series(y1); side = LeftSide())
        @test sitp_left isa ConstantSeriesInterpolant{Float64, ComplexF64}
        vals_left = sitp_left(2.5)  # Between 2 and 3
        @test vals_left isa Vector{ComplexF64}
        @test isapprox(vals_left[1], 3.0 + 3.0im, rtol = 1.0e-10)  # Left value at x=2

        # Test RightSide()
        sitp_right = constant_interp(x, Series(y1); side = RightSide())
        vals_right = sitp_right(2.5)
        @test isapprox(vals_right[1], 4.0 + 4.0im, rtol = 1.0e-10)  # Right value at x=3
    end

    # ========================================
    # Vector Evaluation
    # ========================================
    @testset "Vector evaluation" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)  # Linear complex function
        y2 = (2.0 - 1.0im) .* collect(x)

        sitp = constant_interp(x, Series(y1, y2))

        # Vector query
        xq = [0.25, 0.5, 0.75]
        results = sitp(xq)

        @test length(results) == 2  # Two series
        @test all(r -> r isa Vector{ComplexF64}, results)
        @test all(r -> length(r) == 3, results)
    end

    # ========================================
    # In-place Evaluation
    # ========================================
    @testset "In-place scalar evaluation" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)
        y2 = (2.0 - 1.0im) .* collect(x)

        sitp = constant_interp(x, Series(y1, y2))

        output = Vector{ComplexF64}(undef, 2)
        sitp(output, 0.5)

        @test output[1] isa ComplexF64
    end

    @testset "In-place vector evaluation" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)
        y2 = (2.0 - 1.0im) .* collect(x)

        sitp = constant_interp(collect(x), Series(y1, y2))

        xq = collect(range(0.1, 0.9, 5))
        outputs = [Vector{ComplexF64}(undef, 5) for _ in 1:2]

        sitp(outputs, xq)

        @test outputs[1][3] isa ComplexF64
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation modes" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)
        y2 = (2.0 - 1.0im) .* collect(x)

        # Extension mode
        sitp_ext = constant_interp(x, Series(y1, y2); extrap = ExtendExtrap())
        vals_ext = sitp_ext(1.5)  # Beyond domain
        @test vals_ext isa Vector{ComplexF64}

        # Constant mode
        sitp_const = constant_interp(x, Series(y1, y2); extrap = ClampExtrap())
        vals_const = sitp_const(1.5)  # Beyond domain
        @test vals_const isa Vector{ComplexF64}
        @test isapprox(vals_const[1], y1[end], rtol = 1.0e-10)
    end

    # ========================================
    # Type Stability
    # ========================================
    @testset "Type stability" begin
        x = range(0.0, 1.0, 11)
        y1 = rand(ComplexF64, 11)
        y2 = rand(ComplexF64, 11)

        sitp = constant_interp(x, Series(y1, y2))

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

        sitp = constant_interp(x, Series(y1, y2))
        output = Vector{ComplexF64}(undef, 2)

        # Warmup
        sitp(output, 0.5)
        sitp(output, 0.5)

        # Measure allocation
        allocs = @allocated sitp(output, 0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Derivative Support (Always Zero for Step Function)
    # ========================================
    @testset "Derivative for Complex values (always zero)" begin
        x = range(0.0, 1.0, 11)
        y1 = (1.0 + 2.0im) .* collect(x)

        sitp = constant_interp(x, Series(y1))

        # First derivative should be zero (step function)
        d1 = sitp(0.5; deriv = DerivOp(1))
        @test d1 isa Vector{ComplexF64}
        @test d1[1] == zero(ComplexF64)

        # Second derivative should be zero
        d2 = sitp(0.5; deriv = DerivOp(2))
        @test d2 isa Vector{ComplexF64}
        @test d2[1] == zero(ComplexF64)
    end

    # ========================================
    # Real-valued Backward Compatibility
    # ========================================
    @testset "Real-valued backward compatibility" begin
        # Ensure the changes don't break real-valued interpolation
        x = range(0.0, 1.0, 11)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        sitp = constant_interp(x, Series(y1, y2))

        # Type checks for backward compatibility
        @test sitp isa ConstantSeriesInterpolant{Float64, Float64}
        @test grid_type(sitp) == Float64
        @test value_type(sitp) == Float64
        @test eval_type(sitp, Float64) == Float64

        vals = sitp(0.5)
        @test vals isa Vector{Float64}
    end

    # ========================================
    # Tg Calculation Policy (Query Independence)
    # ========================================
    @testset "Lossless type promotion" begin
        # Float32 data + Float64 query → Float64 output (wider type wins)
        x32 = Float32.(0:0.1:1)
        y1 = sin.(x32)
        y2 = cos.(x32)

        sitp = constant_interp(x32, Series(y1, y2))
        @test sitp isa ConstantSeriesInterpolant{Float32, Float32}

        # Float64 query promotes output to Float64 (lossless - wider type)
        result = sitp(0.5)  # 0.5 is Float64
        @test eltype(result) === Float64
    end

end
