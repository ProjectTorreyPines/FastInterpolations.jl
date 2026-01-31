# ========================================
# Complex Constant Interpolation Tests
# ========================================
# Tests for native Complex number support in ConstantInterpolant.
# Validates the Tg/Tv type separation design.

using Test
using FastInterpolations

@testset "Complex Constant Interpolation" begin

    # ========================================
    # Basic Complex Interpolation
    # ========================================
    @testset "ComplexF64 values" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]

        itp = constant_interp(x, y)

        # Type checks
        @test itp isa ConstantInterpolant{Float64, ComplexF64}
        @test grid_type(itp) == Float64
        @test value_type(itp) == ComplexF64
        @test eval_type(itp, Float64) == ComplexF64

        # Evaluation returns ComplexF64
        val = itp(0.5)
        @test val isa ComplexF64

        # Check value (nearest mode by default, 0.5 is exactly at midpoint → left)
        @test val == 1.0+2.0im
    end

    # ========================================
    # ComplexF32 Support
    # ========================================
    @testset "ComplexF32 values" begin
        x = Float32[0, 1, 2, 3]
        y = ComplexF32[1+2im, 3+4im, 5+6im, 7+8im]

        itp = constant_interp(x, y)

        @test itp isa ConstantInterpolant{Float32, ComplexF32}
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

        itp = constant_interp(x, y)

        # x promoted to Float64, y promoted to ComplexF64
        @test itp isa ConstantInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Mixed Types: Float32 grid + ComplexF64 values
    # ========================================
    @testset "Float32 grid + ComplexF64 values" begin
        x = Float32[0, 1, 2, 3]
        y = ComplexF64[1+2im, 3+4im, 5+6im, 7+8im]

        itp = constant_interp(x, y)

        # Grid promoted to Float64 to match Complex{Float64}
        @test itp isa ConstantInterpolant{Float64, ComplexF64}

        val = itp(0.5)
        @test val isa ComplexF64
    end

    # ========================================
    # Side Modes
    # ========================================
    @testset "Side modes" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        # :left mode
        itp_left = constant_interp(x, y; side=:left)
        @test itp_left(0.5) == 1.0+1.0im
        @test itp_left(1.5) == 2.0+2.0im

        # :right mode
        itp_right = constant_interp(x, y; side=:right)
        @test itp_right(0.5) == 2.0+2.0im
        @test itp_right(1.0) == 2.0+2.0im  # At grid point, still gets left value

        # :nearest mode
        itp_nearest = constant_interp(x, y; side=:nearest)
        @test itp_nearest(0.4) == 1.0+1.0im  # Closer to left
        @test itp_nearest(0.6) == 2.0+2.0im  # Closer to right
        @test itp_nearest(0.5) == 1.0+1.0im  # Midpoint → left (tie-breaking)
    end

    # ========================================
    # Derivatives (Always Zero)
    # ========================================
    @testset "Derivatives return Complex zero" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+2.0im, 3.0+4.0im, 5.0+6.0im, 7.0+8.0im]

        itp = constant_interp(x, y)

        # First derivative should be Complex zero
        d1 = itp(0.5; deriv=1)
        @test d1 isa ComplexF64
        @test d1 == zero(ComplexF64)

        # Second derivative should be Complex zero
        d2 = itp(0.5; deriv=2)
        @test d2 isa ComplexF64
        @test d2 == zero(ComplexF64)
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation modes" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        # :constant mode
        itp_const = constant_interp(x, y; extrap=:constant)
        @test itp_const(-1.0) == 1.0+1.0im  # Clamped to first
        @test itp_const(5.0) == 4.0+4.0im   # Clamped to last

        # :extension mode (same as constant for step functions)
        itp_ext = constant_interp(x, y; extrap=:extension)
        @test itp_ext(-1.0) == 1.0+1.0im
        @test itp_ext(5.0) == 4.0+4.0im

        # :wrap mode
        itp_wrap = constant_interp(x, y; extrap=:wrap)
        val_wrap = itp_wrap(4.5)
        @test val_wrap isa ComplexF64
    end

    # ========================================
    # Vector Evaluation
    # ========================================
    @testset "Vector evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = constant_interp(x, y; side=:left)

        xq = [0.5, 1.5, 2.5]
        vals = itp(xq)

        @test vals isa Vector{ComplexF64}
        @test length(vals) == 3
        @test vals[1] == 1.0+1.0im
        @test vals[2] == 2.0+2.0im
        @test vals[3] == 3.0+3.0im
    end

    # ========================================
    # Broadcast Evaluation
    # ========================================
    @testset "Broadcast evaluation" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = constant_interp(x, y; side=:left)

        xq = [0.5, 1.5, 2.5]
        vals = itp.(xq)

        @test vals isa Vector{ComplexF64}
        @test vals[2] == 2.0+2.0im
    end

    # ========================================
    # Type Stability
    # ========================================
    @testset "Type stability" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = constant_interp(x, y)

        # Scalar evaluation should be type-stable
        @test @inferred(itp(0.5)) isa ComplexF64

        # First derivative should be type-stable
        @test @inferred(itp(0.5; deriv=1)) isa ComplexF64
    end

    # ========================================
    # Zero Allocation (Scalar)
    # ========================================
    @testset "Zero allocation (scalar)" begin
        x = collect(range(0.0, 10.0, 101))
        y = rand(ComplexF64, 101)
        itp = constant_interp(x, y)

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

        itp = constant_interp(x, y; side=:left)

        xq = [0.5, 1.5, 2.5]
        output = Vector{ComplexF64}(undef, 3)

        itp(output, xq)

        @test output[1] == 1.0+1.0im
        @test output[2] == 2.0+2.0im
        @test output[3] == 3.0+3.0im
    end

    # ========================================
    # Grid Point Behavior
    # ========================================
    @testset "Grid point behavior" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0+1.0im, 2.0+2.0im, 3.0+3.0im, 4.0+4.0im]

        itp = constant_interp(x, y)

        # At exact grid points
        @test itp(0.0) == 1.0+1.0im
        @test itp(1.0) == 2.0+2.0im
        @test itp(2.0) == 3.0+3.0im
        @test itp(3.0) == 4.0+4.0im
    end

    # ========================================
    # Real-valued Backward Compatibility
    # ========================================
    @testset "Real-valued backward compatibility" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]

        itp = constant_interp(x, y)

        # Type checks for backward compatibility
        @test itp isa ConstantInterpolant{Float64, Float64}
        @test grid_type(itp) == Float64
        @test value_type(itp) == Float64
        @test eval_type(itp, Float64) == Float64

        val = itp(0.5)
        @test val isa Float64
        @test val == 10.0

        d1 = itp(0.5; deriv=1)
        @test d1 isa Float64
        @test d1 == 0.0
    end

end
