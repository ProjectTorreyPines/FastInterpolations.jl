# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     SERIES CALLABLE INTERFACE TESTS                       ║
# ║         Tests for generic callable implementations in series_callable.jl  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase C: Tests verify that default callable implementations work correctly
# for AbstractSeriesInterpolant types using trait-based dispatch.
#

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "Series Callable Interface" begin

    # ========================================
    # Test Setup: Create CubicSeriesInterpolant
    # ========================================
    # CubicSeriesInterpolant is used as the reference implementation
    # since it implements all required traits

    x = collect(0.0:0.1:1.0)
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    ys = [y1, y2]
    sitp = cubic_interp(x, ys)

    # ========================================
    # Signature 1: Scalar Out-of-Place
    # ========================================

    @testset "scalar out-of-place evaluation" begin
        # Test: Returns vector of correct length
        result = sitp(0.5)
        @test length(result) == 2
        @test result isa Vector{Float64}

        # Test: Values are approximately correct
        # sin(π) ≈ 0, cos(π) ≈ -1
        @test result[1] ≈ sin(π) atol=1e-2
        @test result[2] ≈ cos(π) atol=1e-2

        # Test: Derivative dispatch works
        deriv1 = sitp(0.5; deriv=1)
        @test length(deriv1) == 2
        @test deriv1 isa Vector{Float64}

        deriv2 = sitp(0.5; deriv=2)
        @test length(deriv2) == 2
    end

    # ========================================
    # Signature 2: Scalar In-Place
    # ========================================

    @testset "scalar in-place evaluation" begin
        output = zeros(2)

        # Test: Modifies output buffer
        result = sitp(output, 0.5)
        @test result === output  # Returns same buffer
        @test output[1] ≈ sin(π) atol=1e-2
        @test output[2] ≈ cos(π) atol=1e-2

        # Test: Zero allocation (after warmup)
        sitp(output, 0.5)  # Warmup
        allocs = @allocated sitp(output, 0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Test: Validates output size
        bad_output = zeros(3)  # Wrong size
        @test_throws DimensionMismatch sitp(bad_output, 0.5)

        # Test: Derivative in-place
        output_deriv = zeros(2)
        sitp(output_deriv, 0.5; deriv=1)
        @test output_deriv isa Vector{Float64}
    end

    # ========================================
    # Signature 3: Vector Out-of-Place
    # ========================================

    @testset "vector out-of-place evaluation" begin
        xq = [0.25, 0.5, 0.75]
        results = sitp(xq)

        # Test: Returns vector of vectors
        @test length(results) == FI.n_series(sitp)
        @test all(r -> length(r) == length(xq), results)

        # Test: Each result is a vector
        @test all(r -> r isa Vector{Float64}, results)

        # Test: Values are approximately correct
        # At x=0.5: sin(π) ≈ 0, cos(π) ≈ -1
        @test results[1][2] ≈ sin(π) atol=1e-2
        @test results[2][2] ≈ cos(π) atol=1e-2
    end

    # ========================================
    # Signature 4: Vector In-Place
    # ========================================

    @testset "vector in-place evaluation" begin
        xq = collect(0.0:0.05:1.0)
        n_query = length(xq)
        outputs = [zeros(n_query) for _ in 1:FI.n_series(sitp)]

        result = sitp(outputs, xq)
        @test result === outputs

        # Test: Values are populated
        @test all(r -> !all(iszero, r), outputs)

        # Test: Zero allocation (after warmup)
        sitp(outputs, xq)  # Warmup
        allocs = @allocated sitp(outputs, xq)
        @test allocs <= ALLOC_THRESHOLD

        # Test: Validates output dimensions - wrong number of series
        bad_outputs_count = [zeros(n_query) for _ in 1:3]
        @test_throws DimensionMismatch sitp(bad_outputs_count, xq)

        # Test: Validates output dimensions - wrong query length
        bad_outputs_size = [zeros(n_query - 1) for _ in 1:2]
        @test_throws DimensionMismatch sitp(bad_outputs_size, xq)
    end

    # ========================================
    # Signature 5: Pre-Anchored Evaluation
    # ========================================

    @testset "pre-anchored evaluation" begin
        xq = [0.25, 0.5, 0.75]
        n_query = length(xq)
        outputs = [zeros(n_query) for _ in 1:FI.n_series(sitp)]

        # Build anchors using the correct signature
        x_grid = FI._get_grid(sitp)
        aq_vec = FI._anchor_query(x_grid, xq)

        result = sitp(outputs, aq_vec)
        @test result === outputs

        # Compare with direct evaluation
        expected = sitp(xq)
        for k in 1:FI.n_series(sitp)
            @test outputs[k] ≈ expected[k]
        end

        # Test: Zero allocation with pre-built anchors
        sitp(outputs, aq_vec)  # Warmup
        allocs = @allocated sitp(outputs, aq_vec)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Extrapolation Handling
    # ========================================

    @testset "extrapolation modes" begin
        x_ext = collect(0.0:0.1:1.0)
        ys_ext = [sin.(2π .* x_ext)]

        @testset "extrap=:none throws on out-of-bounds" begin
            sitp_none = cubic_interp(x_ext, ys_ext; extrap=:none)
            @test_throws DomainError sitp_none(-0.1)
            @test_throws DomainError sitp_none(1.1)
        end

        @testset "extrap=:constant returns boundary value" begin
            sitp_const = cubic_interp(x_ext, ys_ext; extrap=:constant)
            # Below domain: returns y[1]
            @test sitp_const(-0.1)[1] ≈ sin(0.0) atol=1e-6
            # Above domain: returns y[end]
            @test sitp_const(1.1)[1] ≈ sin(2π) atol=1e-6
        end

        @testset "extrap=:extension extrapolates polynomial" begin
            sitp_ext = cubic_interp(x_ext, ys_ext; extrap=:extension)
            # Should not throw
            @test sitp_ext(-0.1) isa Vector{Float64}
            @test sitp_ext(1.1) isa Vector{Float64}
        end

        @testset "extrap=:wrap for periodic" begin
            ys_periodic = [sin.(2π .* x_ext)]  # sin(0) = sin(2π) = 0
            sitp_wrap = cubic_interp(x_ext, ys_periodic; bc=PeriodicBC(), extrap=:wrap)
            # Query at 1.1 should be approximately equal to query at 0.1
            @test sitp_wrap(1.1)[1] ≈ sitp_wrap(0.1)[1] atol=1e-6
        end
    end

    # ========================================
    # Type Stability
    # ========================================

    @testset "type stability" begin
        x32 = Float32.(collect(0.0:0.1:1.0))
        ys32 = [Float32.(sin.(2π .* Float64.(x32)))]

        x64 = collect(0.0:0.1:1.0)
        ys64 = [sin.(2π .* x64)]

        sitp32 = cubic_interp(x32, ys32)
        sitp64 = cubic_interp(x64, ys64)

        @test sitp32(0.5f0) isa Vector{Float32}
        @test sitp64(0.5) isa Vector{Float64}

        # Check inferred types (no Union types)
        @test_nowarn @inferred sitp64(0.5)
    end

    # ========================================
    # Trait Function Usage
    # ========================================

    @testset "callable uses trait functions" begin
        # Verify that the callable uses the trait functions
        @test FI.n_series(sitp) == 2
        @test FI._get_grid(sitp) ≈ x
        @test FI._get_extrap(sitp) isa FI.ExtrapVal
        @test FI._should_wrap(sitp) == false

        # Verify _make_anchor is available (indirectly tested via evaluation)
        # This test ensures the trait-based dispatch path works
        output = zeros(2)
        sitp(output, 0.5)
        @test !all(iszero, output)
    end

    # ========================================
    # Edge Cases
    # ========================================

    @testset "edge cases" begin
        @testset "query at grid points" begin
            # Evaluation at exact grid points should be accurate
            for i in eachindex(x)
                result = sitp(x[i])
                @test result[1] ≈ y1[i] atol=1e-10
                @test result[2] ≈ y2[i] atol=1e-10
            end
        end

        @testset "query at domain boundaries" begin
            # At x_min
            result_min = sitp(0.0)
            @test result_min[1] ≈ sin(0.0) atol=1e-10
            @test result_min[2] ≈ cos(0.0) atol=1e-10

            # At x_max
            result_max = sitp(1.0)
            @test result_max[1] ≈ sin(2π) atol=1e-10
            @test result_max[2] ≈ cos(2π) atol=1e-10
        end

        @testset "single series" begin
            sitp_single = cubic_interp(x, [y1])
            @test FI.n_series(sitp_single) == 1

            result = sitp_single(0.5)
            @test length(result) == 1
        end

        @testset "many series" begin
            ys_many = [sin.((2π * k) .* x) for k in 1:10]
            sitp_many = cubic_interp(x, ys_many)
            @test FI.n_series(sitp_many) == 10

            result = sitp_many(0.5)
            @test length(result) == 10
        end
    end

    # ========================================
    # Derivative Evaluation
    # ========================================

    @testset "derivative evaluation" begin
        # For sin(2πx): d/dx = 2π*cos(2πx), d²/dx² = -(2π)²*sin(2πx)
        # For cos(2πx): d/dx = -2π*sin(2πx), d²/dx² = -(2π)²*cos(2πx)

        # At x=0.25: sin(π/2) = 1, cos(π/2) = 0
        # Derivatives at x=0.25:
        # d/dx sin(2πx) = 2π*cos(π/2) = 0
        # d/dx cos(2πx) = -2π*sin(π/2) = -2π

        x_test = 0.25
        deriv1 = sitp(x_test; deriv=1)
        @test deriv1[1] ≈ 2π * cos(2π * x_test) atol=0.5  # Cubic approx
        @test deriv1[2] ≈ -2π * sin(2π * x_test) atol=0.5

        # In-place derivative evaluation
        output = zeros(2)
        sitp(output, x_test; deriv=1)
        @test output ≈ deriv1

        # Vector derivative evaluation
        xq = [0.25, 0.5, 0.75]
        derivs = sitp(xq; deriv=1)
        @test length(derivs) == 2
        @test all(d -> length(d) == 3, derivs)
    end

end  # testset "Series Callable Interface"
