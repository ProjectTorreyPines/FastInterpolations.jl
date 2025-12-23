# ========================================
# Generic Boundary Condition Tests
# ========================================
# Tests for D1, D2, and mixed BC types

using Test
using FastInterpolations

# Tolerance constants for numerical comparisons
# rtol: relative tolerance (scale-independent, good for most cases)
# atol: absolute tolerance (needed when values can be near zero)
const RTOL = 1e-14
const ATOL = 1e-14

@testset "Generic Boundary Conditions" begin

    # ========================================
    # Basic Functionality Tests
    # ========================================
    @testset "Basic BC Type Construction" begin
        # D1 construction
        @test D1(0.5) isa D1{Float64}
        @test D1(0.5f0) isa D1{Float32}
        @test D1(0).val == 0.0
        @test D1(1).val == 1.0

        # D2 construction
        @test D2(0.5) isa D2{Float64}
        @test D2(0.5f0) isa D2{Float32}
        @test D2(0).val == 0.0
        @test D2(1).val == 1.0
    end

    @testset "BC Symbol Equivalence" begin
        x = range(0.0, 1.0, 11)
        y = sin.(π .* x)
        xi = 0.5

        # :natural == D2(0), D2(0)
        r_natural = cubic_interp(x, y, xi; bc=:natural)
        r_d2_zero = cubic_interp(x, y, xi; bc=(D2(0.0), D2(0.0)))
        @test r_natural ≈ r_d2_zero rtol=RTOL atol=ATOL

        # :clamped == D1(0), D1(0)
        r_clamped = cubic_interp(x, y, xi; bc=:clamped)
        r_d1_zero = cubic_interp(x, y, xi; bc=(D1(0.0), D1(0.0)))
        @test r_clamped ≈ r_d1_zero rtol=RTOL atol=ATOL

        # Single D1/D2 should apply to both ends
        r_single_d1 = cubic_interp(x, y, xi; bc=D1(0.5))
        r_tuple_d1 = cubic_interp(x, y, xi; bc=(D1(0.5), D1(0.5)))
        @test r_single_d1 ≈ r_tuple_d1 rtol=RTOL atol=ATOL

        r_single_d2 = cubic_interp(x, y, xi; bc=D2(1.0))
        r_tuple_d2 = cubic_interp(x, y, xi; bc=(D2(1.0), D2(1.0)))
        @test r_single_d2 ≈ r_tuple_d2 rtol=RTOL atol=ATOL
    end

    @testset "Different BC Values Give Different Results" begin
        x = range(0.0, 1.0, 11)
        y = sin.(π .* x)
        # Use query point away from symmetry center for BC sensitivity
        xi = 0.15

        # D1 with different values
        r1 = cubic_interp(x, y, xi; bc=D1(0.0))
        r2 = cubic_interp(x, y, xi; bc=D1(5.0))
        @test r1 != r2

        # D2 with different values
        r3 = cubic_interp(x, y, xi; bc=D2(0.0))
        r4 = cubic_interp(x, y, xi; bc=D2(-20.0))
        @test r3 != r4

        # Mixed BC
        r5 = cubic_interp(x, y, xi; bc=(D1(0.0), D2(0.0)))
        r6 = cubic_interp(x, y, xi; bc=(D1(5.0), D2(0.0)))
        @test r5 != r6
    end

    # ========================================
    # Mathematical Correctness Tests
    # ========================================
    # Cubic splines should exactly reproduce polynomials up to degree 3

    @testset "Quadratic Polynomial: y = ax² + bx + c" begin
        # f(x) = 2x² - 3x + 1
        # f'(x) = 4x - 3
        # f''(x) = 4
        a, b, c = 2.0, -3.0, 1.0
        f(x) = a*x^2 + b*x + c
        f_prime(x) = 2a*x + b
        f_double_prime = 2a

        x = range(0.0, 2.0, 21)
        y = f.(x)

        # Query points (avoid grid points for stricter test)
        xi = [0.15, 0.5, 1.25, 1.8]

        @testset "D2, D2 (second derivative BC)" begin
            # Provide exact second derivatives at both ends
            bc = (D2(f_double_prime), D2(f_double_prime))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "D1, D1 (first derivative BC)" begin
            # Provide exact first derivatives at endpoints
            x0, xn = first(x), last(x)
            bc = (D1(f_prime(x0)), D1(f_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "D1, D2 (mixed BC)" begin
            x0, xn = first(x), last(x)
            bc = (D1(f_prime(x0)), D2(f_double_prime))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "D2, D1 (mixed BC reversed)" begin
            x0, xn = first(x), last(x)
            bc = (D2(f_double_prime), D1(f_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end
    end

    @testset "Cubic Polynomial: y = ax³ + bx² + cx + d" begin
        # f(x) = x³ - 2x² + x - 1
        # f'(x) = 3x² - 4x + 1
        # f''(x) = 6x - 4
        a, b, c, d = 1.0, -2.0, 1.0, -1.0
        f(x) = a*x^3 + b*x^2 + c*x + d
        f_prime(x) = 3a*x^2 + 2b*x + c
        f_double_prime(x) = 6a*x + 2b

        x = range(-1.0, 2.0, 31)
        y = f.(x)
        x0, xn = first(x), last(x)

        xi = [-0.7, 0.0, 0.5, 1.3, 1.9]

        @testset "D1, D1 (first derivative BC)" begin
            bc = (D1(f_prime(x0)), D1(f_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "D2, D2 (second derivative BC)" begin
            bc = (D2(f_double_prime(x0)), D2(f_double_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "D1, D2 (mixed BC)" begin
            bc = (D1(f_prime(x0)), D2(f_double_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end
    end

    @testset "Linear Function: y = mx + b" begin
        # Linear functions should be exact with any BC
        slope, intercept = 2.5, -1.0
        g(x) = slope*x + intercept

        x = range(0.0, 3.0, 11)
        y = g.(x)

        xi = [0.3, 1.5, 2.7]
        expected = g.(xi)

        @test cubic_interp(x, y, xi; bc=:natural) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=:clamped) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=(D1(slope), D1(slope))) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=(D2(0.0), D2(0.0))) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=(D1(slope), D2(0.0))) ≈ expected rtol=RTOL atol=ATOL
    end

    # ========================================
    # CubicSplineCache with Generic BC
    # ========================================
    @testset "CubicSplineCache with D1/D2" begin
        x = collect(range(0.0, 1.0, 21))
        y = sin.(π .* x)

        # Create cache with D1 BC
        cache_d1 = CubicSplineCache(x; bc=D1(0.5))
        @test cache_d1 isa CubicSplineCache
        result_d1 = cubic_interp(cache_d1, y, 0.5)
        @test isfinite(result_d1)

        # Create cache with tuple BC
        cache_mixed = CubicSplineCache(x; bc=(D1(1.0), D2(0.0)))
        @test cache_mixed isa CubicSplineCache
        result_mixed = cubic_interp(cache_mixed, y, 0.5)
        @test isfinite(result_mixed)

        # Cache reuse for multiple y vectors
        y2 = cos.(π .* x)
        result1 = cubic_interp(cache_d1, y, 0.5)
        result2 = cubic_interp(cache_d1, y2, 0.5)
        @test result1 != result2  # Different y should give different results
    end

    # ========================================
    # CubicInterpolant with Generic BC
    # ========================================
    @testset "CubicInterpolant (2-arg form) with D1/D2" begin
        x = range(0.0, 1.0, 21)
        y = sin.(π .* x)

        # Create interpolant with D1 BC
        itp_d1 = cubic_interp(x, y; bc=D1(0.0))
        @test itp_d1 isa CubicInterpolant
        @test isfinite(itp_d1(0.5))

        # Create interpolant with mixed BC
        itp_mixed = cubic_interp(x, y; bc=(D1(π), D2(0.0)))
        @test itp_mixed isa CubicInterpolant
        @test isfinite(itp_mixed(0.5))

        # Broadcast should work
        xi = [0.2, 0.4, 0.6, 0.8]
        results = itp_d1.(xi)
        @test length(results) == 4
        @test all(isfinite, results)
    end

    # ========================================
    # Type Stability and Float32 Support
    # ========================================
    @testset "Float32 Support" begin
        x = collect(range(0.0f0, 1.0f0, 21))
        y = sin.(π .* x)

        # D1/D2 with Float32
        result = cubic_interp(x, y, 0.5f0; bc=(D1(0.0f0), D2(0.0f0)))
        @test result isa Float32
        @test isfinite(result)

        # Cache with Float32
        cache = CubicSplineCache(x; bc=D1(0.5f0))
        @test eltype(cache.x) == Float32
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "Edge Cases" begin
        x = range(0.0, 1.0, 11)
        y = sin.(π .* x)

        # Query at grid points should be exact
        @test cubic_interp(x, y, 0.0; bc=D1(0.0)) ≈ y[1] rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, 1.0; bc=D1(0.0)) ≈ y[end] rtol=RTOL atol=ATOL

        # Vector query
        result = cubic_interp(x, y, [0.25, 0.5, 0.75]; bc=(D1(0.5), D2(-1.0)))
        @test length(result) == 3
        @test all(isfinite, result)

        # In-place version
        output = zeros(3)
        cubic_interp!(output, collect(x), collect(y), [0.25, 0.5, 0.75]; bc=D1(0.0))
        @test all(isfinite, output)
    end

end
