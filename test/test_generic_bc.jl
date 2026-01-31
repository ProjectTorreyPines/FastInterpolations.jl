# ========================================
# Generic Boundary Condition Tests
# ========================================
# Tests for Deriv1, Deriv2, and mixed BC types

using Test
using FastInterpolations

# Tolerance constants for numerical comparisons
# rtol: relative tolerance (scale-independent, good for most cases)
# atol: absolute tolerance (needed when values can be near zero)
const RTOL = 1e-14
const ATOL = 1e-14

@testset "Generic Boundary Conditions" begin

    # ========================================
    # Type Hierarchy Tests
    # ========================================
    @testset "BC Type Hierarchy" begin
        # Type-Free design: PointBC is abstract parent (no type parameter)
        # Concrete types like Deriv1{Float64} are subtypes of PointBC
        @test Deriv1{Float64} <: FastInterpolations.PointBC
        @test Deriv2{Float64} <: FastInterpolations.PointBC
        @test Deriv3{Float64} <: FastInterpolations.PointBC
        @test Deriv1{Float32} <: FastInterpolations.PointBC
        @test Deriv2{Float32} <: FastInterpolations.PointBC
        @test Deriv3{Float32} <: FastInterpolations.PointBC

        # Type-Free design: PointBC <: AbstractBC (no type parameters)
        @test FastInterpolations.PointBC <: AbstractBC

        # Type-Free design: BCPair{L, R} <: AbstractBC (no Tv parameter)
        @test BCPair{Deriv1{Float64}, Deriv2{Float64}} <: AbstractBC
        @test BCPair{Deriv3{Float64}, Deriv1{Float64}} <: AbstractBC

        # PeriodicBC <: AbstractBC (singleton)
        @test PeriodicBC <: AbstractBC
    end

    @testset "BCPair Construction" begin
        # Type-Free design: BCPair{L, R} (no Tv parameter)
        # Direct construction
        bc_pair = BCPair(Deriv1(0.5), Deriv2(1.0))
        @test bc_pair isa BCPair{Deriv1{Float64}, Deriv2{Float64}}
        @test bc_pair.left.val == 0.5
        @test bc_pair.right.val == 1.0

        # Tuple constructor
        bc_from_tuple = BCPair((Deriv1(0.5), Deriv2(1.0)))
        @test bc_from_tuple isa BCPair{Deriv1{Float64}, Deriv2{Float64}}
        @test bc_from_tuple.left.val == bc_pair.left.val
        @test bc_from_tuple.right.val == bc_pair.right.val

        # Float32
        bc_f32 = BCPair(Deriv1(0.5f0), Deriv2(1.0f0))
        @test bc_f32 isa BCPair{Deriv1{Float32}, Deriv2{Float32}}

        # BCPair with Deriv3
        bc_d3 = BCPair(Deriv3(6.0), Deriv3(2.0))
        @test bc_d3 isa BCPair{Deriv3{Float64}, Deriv3{Float64}}
        @test bc_d3.left.val == 6.0
        @test bc_d3.right.val == 2.0

        # Mixed BCPair with Deriv3
        bc_mixed = BCPair(Deriv3(0.0), Deriv1(1.0))
        @test bc_mixed isa BCPair{Deriv3{Float64}, Deriv1{Float64}}
    end

    @testset "PeriodicBC Construction" begin
        # Type-Free design: PeriodicBC is a singleton, no type parameter
        pbc = PeriodicBC()
        @test pbc isa PeriodicBC
        @test pbc isa AbstractBC

        # Singleton property - all instances are the same
        pbc1 = PeriodicBC()
        pbc2 = PeriodicBC()
        @test typeof(pbc1) === typeof(pbc2)
    end

    # ========================================
    # Type Conversion Constructors (Coverage)
    # ========================================
    @testset "BC Type Conversion Constructors" begin
        # Deriv1{T}(bc::Deriv1) - convert Deriv1{Float64} → Deriv1{Float32}
        d1_f64 = Deriv1(0.5)
        d1_f32 = Deriv1{Float32}(d1_f64)
        @test d1_f32 isa Deriv1{Float32}
        @test d1_f32.val == Float32(0.5)

        # Deriv2{T}(bc::Deriv2) - convert Deriv2{Float64} → Deriv2{Float32}
        d2_f64 = Deriv2(1.5)
        d2_f32 = Deriv2{Float32}(d2_f64)
        @test d2_f32 isa Deriv2{Float32}
        @test d2_f32.val == Float32(1.5)

        # Deriv3{T}(bc::Deriv3) - convert Deriv3{Float64} → Deriv3{Float32}
        d3_f64 = Deriv3(6.0)
        d3_f32 = Deriv3{Float32}(d3_f64)
        @test d3_f32 isa Deriv3{Float32}
        @test d3_f32.val == Float32(6.0)

        # PeriodicBC, NaturalBC, ClampedBC are now non-parametric singletons
        # No type conversion needed - they're type-agnostic
        @test PeriodicBC() isa PeriodicBC
        @test NaturalBC() isa NaturalBC
        @test ClampedBC() isa ClampedBC
    end

    # ========================================
    # BC Normalization (Coverage)
    # ========================================
    @testset "_normalize_bc Coverage" begin
        # Note: PeriodicBC is handled via _is_periodic_bc() before _normalize_bc is called.
        # No _normalize_bc(::PeriodicBC, T) method exists (dead code was removed).

        # BCPair with type promotion (Float64 BC → Float32 target)
        # Type-Free design: BCPair{L, R} without Tv parameter
        bc_f64 = BCPair(Deriv1(0.5), Deriv2(1.0))  # Float64
        bc_promoted = FastInterpolations._normalize_bc(bc_f64, Float32)
        @test bc_promoted isa BCPair{Deriv1{Float32}, Deriv2{Float32}}
        @test bc_promoted.left.val == Float32(0.5)
        @test bc_promoted.right.val == Float32(1.0)

        # PointBC with type promotion
        d1_f64 = Deriv1(0.25)  # Float64
        bc_from_d1 = FastInterpolations._normalize_bc(d1_f64, Float32)
        @test bc_from_d1 isa BCPair{Deriv1{Float32}, Deriv1{Float32}}
        @test bc_from_d1.left.val == Float32(0.25)
        @test bc_from_d1.right.val == Float32(0.25)

        d2_f64 = Deriv2(0.75)  # Float64
        bc_from_d2 = FastInterpolations._normalize_bc(d2_f64, Float32)
        @test bc_from_d2 isa BCPair{Deriv2{Float32}, Deriv2{Float32}}
    end

    # ========================================
    # _is_periodic_bc Predicate (Coverage)
    # ========================================
    @testset "_is_periodic_bc Predicate" begin
        @test FastInterpolations._is_periodic_bc(PeriodicBC()) == true
        @test FastInterpolations._is_periodic_bc(NaturalBC()) == false
        @test FastInterpolations._is_periodic_bc(ClampedBC()) == false
        @test FastInterpolations._is_periodic_bc(BCPair(Deriv1(0.0), Deriv2(0.0))) == false
        @test FastInterpolations._is_periodic_bc(Deriv1(0.0)) == false
        @test FastInterpolations._is_periodic_bc(Deriv2(0.0)) == false
        @test FastInterpolations._is_periodic_bc(Deriv3(0.0)) == false
        @test FastInterpolations._is_periodic_bc(BCPair(Deriv3(0.0), Deriv1(0.0))) == false
    end

    # ========================================
    # _promote_pointbc Helper (Coverage)
    # ========================================
    @testset "_promote_pointbc Helper" begin
        # Deriv1 promotion
        d1_promoted = FastInterpolations._promote_pointbc(Deriv1(0.5), Float32)
        @test d1_promoted isa Deriv1{Float32}
        @test d1_promoted.val == Float32(0.5)

        # Deriv2 promotion
        d2_promoted = FastInterpolations._promote_pointbc(Deriv2(1.5), Float32)
        @test d2_promoted isa Deriv2{Float32}
        @test d2_promoted.val == Float32(1.5)

        # Deriv3 promotion
        d3_promoted = FastInterpolations._promote_pointbc(Deriv3(6.0), Float32)
        @test d3_promoted isa Deriv3{Float32}
        @test d3_promoted.val == Float32(6.0)
    end

    # ========================================
    # Basic Functionality Tests
    # ========================================
    @testset "Basic BC Type Construction" begin
        # Deriv1 construction
        @test Deriv1(0.5) isa Deriv1{Float64}
        @test Deriv1(0.5f0) isa Deriv1{Float32}
        @test Deriv1(0).val == 0.0
        @test Deriv1(1).val == 1.0

        # Deriv2 construction
        @test Deriv2(0.5) isa Deriv2{Float64}
        @test Deriv2(0.5f0) isa Deriv2{Float32}
        @test Deriv2(0).val == 0.0
        @test Deriv2(1).val == 1.0

        # Deriv3 construction
        @test Deriv3(6.0) isa Deriv3{Float64}
        @test Deriv3(6.0f0) isa Deriv3{Float32}
        @test Deriv3(0).val == 0.0
        @test Deriv3(6).val == 6.0
    end

    @testset "BC Type Equivalence" begin
        x = range(0.0, 1.0, 11)
        y = sin.(π .* x)
        xi = 0.5

        # NaturalBC() == Deriv2(0), Deriv2(0)
        r_natural = cubic_interp(x, y, xi; bc=NaturalBC())
        r_d2_zero = cubic_interp(x, y, xi; bc=BCPair(Deriv2(0.0), Deriv2(0.0)))
        @test r_natural ≈ r_d2_zero rtol=RTOL atol=ATOL

        # ClampedBC() == Deriv1(0), Deriv1(0)
        r_clamped = cubic_interp(x, y, xi; bc=ClampedBC())
        r_d1_zero = cubic_interp(x, y, xi; bc=BCPair(Deriv1(0.0), Deriv1(0.0)))
        @test r_clamped ≈ r_d1_zero rtol=RTOL atol=ATOL

        # Single Deriv1/Deriv2 should apply to both ends
        r_single_d1 = cubic_interp(x, y, xi; bc=Deriv1(0.5))
        r_bcpair_d1 = cubic_interp(x, y, xi; bc=BCPair(Deriv1(0.5), Deriv1(0.5)))
        @test r_single_d1 ≈ r_bcpair_d1 rtol=RTOL atol=ATOL

        r_single_d2 = cubic_interp(x, y, xi; bc=Deriv2(1.0))
        r_bcpair_d2 = cubic_interp(x, y, xi; bc=BCPair(Deriv2(1.0), Deriv2(1.0)))
        @test r_single_d2 ≈ r_bcpair_d2 rtol=RTOL atol=ATOL
    end

    @testset "Different BC Values Give Different Results" begin
        x = range(0.0, 1.0, 11)
        y = sin.(π .* x)
        # Use query point away from symmetry center for BC sensitivity
        xi = 0.15

        # Deriv1 with different values
        r1 = cubic_interp(x, y, xi; bc=Deriv1(0.0))
        r2 = cubic_interp(x, y, xi; bc=Deriv1(5.0))
        @test r1 != r2

        # Deriv2 with different values
        r3 = cubic_interp(x, y, xi; bc=Deriv2(0.0))
        r4 = cubic_interp(x, y, xi; bc=Deriv2(-20.0))
        @test r3 != r4

        # Mixed BC
        r5 = cubic_interp(x, y, xi; bc=BCPair(Deriv1(0.0), Deriv2(0.0)))
        r6 = cubic_interp(x, y, xi; bc=BCPair(Deriv1(5.0), Deriv2(0.0)))
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

        @testset "Deriv2, Deriv2 (second derivative BC)" begin
            # Provide exact second derivatives at both ends
            bc = BCPair(Deriv2(f_double_prime), Deriv2(f_double_prime))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv1, Deriv1 (first derivative BC)" begin
            # Provide exact first derivatives at endpoints
            x0, xn = first(x), last(x)
            bc = BCPair(Deriv1(f_prime(x0)), Deriv1(f_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv1, Deriv2 (mixed BC)" begin
            x0, xn = first(x), last(x)
            bc = BCPair(Deriv1(f_prime(x0)), Deriv2(f_double_prime))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv2, Deriv1 (mixed BC reversed)" begin
            x0, xn = first(x), last(x)
            bc = BCPair(Deriv2(f_double_prime), Deriv1(f_prime(xn)))
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

        @testset "Deriv1, Deriv1 (first derivative BC)" begin
            bc = BCPair(Deriv1(f_prime(x0)), Deriv1(f_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv2, Deriv2 (second derivative BC)" begin
            bc = BCPair(Deriv2(f_double_prime(x0)), Deriv2(f_double_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv1, Deriv2 (mixed BC)" begin
            bc = BCPair(Deriv1(f_prime(x0)), Deriv2(f_double_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        # Deriv3 tests: f'''(x) = 6a (constant for cubic polynomial)
        f_triple_prime = 6a

        @testset "Deriv3, Deriv3 (third derivative BC)" begin
            # Third derivative is constant for cubic polynomial
            bc = BCPair(Deriv3(f_triple_prime), Deriv3(f_triple_prime))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv3, Deriv1 (mixed BC)" begin
            bc = BCPair(Deriv3(f_triple_prime), Deriv1(f_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv1, Deriv3 (mixed BC)" begin
            bc = BCPair(Deriv1(f_prime(x0)), Deriv3(f_triple_prime))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv3, Deriv2 (mixed BC)" begin
            bc = BCPair(Deriv3(f_triple_prime), Deriv2(f_double_prime(xn)))
            result = cubic_interp(x, y, xi; bc=bc)
            expected = f.(xi)
            @test result ≈ expected rtol=RTOL atol=ATOL
        end

        @testset "Deriv2, Deriv3 (mixed BC)" begin
            bc = BCPair(Deriv2(f_double_prime(x0)), Deriv3(f_triple_prime))
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

        @test cubic_interp(x, y, xi; bc=NaturalBC()) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=ClampedBC()) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=BCPair(Deriv1(slope), Deriv1(slope))) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=BCPair(Deriv2(0.0), Deriv2(0.0))) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=BCPair(Deriv1(slope), Deriv2(0.0))) ≈ expected rtol=RTOL atol=ATOL
        # Deriv3(0) for linear: f'''(x) = 0
        @test cubic_interp(x, y, xi; bc=Deriv3(0.0)) ≈ expected rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, xi; bc=BCPair(Deriv3(0.0), Deriv1(slope))) ≈ expected rtol=RTOL atol=ATOL
    end

    # ========================================
    # CubicSplineCache with Generic BC
    # ========================================
    @testset "CubicSplineCache with Deriv1/Deriv2/Deriv3" begin
        x = collect(range(0.0, 1.0, 21))
        y = sin.(π .* x)

        # Create cache with Deriv1 BC
        cache_d1 = CubicSplineCache(x; bc=Deriv1(0.5))
        @test cache_d1 isa CubicSplineCache
        result_d1 = cubic_interp(cache_d1, y, 0.5)
        @test isfinite(result_d1)

        # Create cache with BCPair
        cache_mixed = CubicSplineCache(x; bc=BCPair(Deriv1(1.0), Deriv2(0.0)))
        @test cache_mixed isa CubicSplineCache
        result_mixed = cubic_interp(cache_mixed, y, 0.5)
        @test isfinite(result_mixed)

        # Cache reuse for multiple y vectors
        y2 = cos.(π .* x)
        result1 = cubic_interp(cache_d1, y, 0.5)
        result2 = cubic_interp(cache_d1, y2, 0.5)
        @test result1 != result2  # Different y should give different results

        # Create cache with Deriv3 BC
        cache_d3 = CubicSplineCache(x; bc=Deriv3(0.0))
        @test cache_d3 isa CubicSplineCache
        result_d3 = cubic_interp(cache_d3, y, 0.5)
        @test isfinite(result_d3)

        # Create cache with mixed Deriv3 BC
        cache_d3_mixed = CubicSplineCache(x; bc=BCPair(Deriv3(1.0), Deriv2(0.0)))
        @test cache_d3_mixed isa CubicSplineCache

        # In-place API with Deriv3 cache
        x_query = [0.1, 0.5, 0.9]
        output = zeros(3)
        cubic_interp!(output, cache_d3, y, x_query)
        @test all(isfinite, output)
    end

    # ========================================
    # CubicInterpolant with Generic BC
    # ========================================
    @testset "CubicInterpolant (2-arg form) with Deriv1/Deriv2" begin
        x = range(0.0, 1.0, 21)
        y = sin.(π .* x)

        # Create interpolant with Deriv1 BC
        itp_d1 = cubic_interp(x, y; bc=Deriv1(0.0))
        @test itp_d1 isa CubicInterpolant
        @test isfinite(itp_d1(0.5))

        # Create interpolant with mixed BC
        itp_mixed = cubic_interp(x, y; bc=BCPair(Deriv1(π), Deriv2(0.0)))
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

        # Deriv1/Deriv2 with Float32
        result = cubic_interp(x, y, 0.5f0; bc=BCPair(Deriv1(0.0f0), Deriv2(0.0f0)))
        @test result isa Float32
        @test isfinite(result)

        # Cache with Float32
        cache = CubicSplineCache(x; bc=Deriv1(0.5f0))
        @test eltype(cache.x) == Float32
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "Edge Cases" begin
        x = range(0.0, 1.0, 11)
        y = sin.(π .* x)

        # Query at grid points should be exact
        @test cubic_interp(x, y, 0.0; bc=Deriv1(0.0)) ≈ y[1] rtol=RTOL atol=ATOL
        @test cubic_interp(x, y, 1.0; bc=Deriv1(0.0)) ≈ y[end] rtol=RTOL atol=ATOL

        # Vector query
        result = cubic_interp(x, y, [0.25, 0.5, 0.75]; bc=BCPair(Deriv1(0.5), Deriv2(-1.0)))
        @test length(result) == 3
        @test all(isfinite, result)

        # In-place version
        output = zeros(3)
        cubic_interp!(output, collect(x), collect(y), [0.25, 0.5, 0.75]; bc=Deriv1(0.0))
        @test all(isfinite, output)
    end

    # ========================================
    # CubicSeriesInterpolant with Generic BC
    # ========================================
    @testset "CubicSeriesInterpolant with Deriv3" begin
        x = 0.0:0.25:1.0
        Y = [sin.(π .* x) cos.(π .* x) x.^2]

        # Series interpolant with Deriv3 BC
        itp = cubic_interp(x, Y; bc=Deriv3(0.0))
        @test itp isa CubicSeriesInterpolant

        # Should work for all series
        @test size(itp(0.5)) == (3,)
        @test all(isfinite, itp(0.5))

        # With mixed BC
        itp_mixed = cubic_interp(x, Y; bc=BCPair(Deriv3(0.0), Deriv1(0.0)))
        @test itp_mixed isa CubicSeriesInterpolant
        @test all(isfinite, itp_mixed(0.5))
    end

end
