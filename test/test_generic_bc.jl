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
        # PointBC is abstract parent of Deriv1, Deriv2
        @test Deriv1{Float64} <: FastInterpolations.PointBC{Float64}
        @test Deriv2{Float64} <: FastInterpolations.PointBC{Float64}
        @test Deriv1{Float32} <: FastInterpolations.PointBC{Float32}
        @test Deriv2{Float32} <: FastInterpolations.PointBC{Float32}

        # PointBC <: AbstractBC
        @test FastInterpolations.PointBC{Float64} <: AbstractBC{Float64}

        # BCPair <: AbstractBC
        @test BCPair{Float64, Deriv1{Float64}, Deriv2{Float64}} <: AbstractBC{Float64}

        # PeriodicBC <: AbstractBC
        @test PeriodicBC{Float64} <: AbstractBC{Float64}
        @test PeriodicBC{Float32} <: AbstractBC{Float32}
    end

    @testset "BCPair Construction" begin
        # Direct construction
        bc_pair = BCPair(Deriv1(0.5), Deriv2(1.0))
        @test bc_pair isa BCPair{Float64, Deriv1{Float64}, Deriv2{Float64}}
        @test bc_pair.left.val == 0.5
        @test bc_pair.right.val == 1.0

        # Tuple constructor
        bc_from_tuple = BCPair((Deriv1(0.5), Deriv2(1.0)))
        @test bc_from_tuple isa BCPair{Float64, Deriv1{Float64}, Deriv2{Float64}}
        @test bc_from_tuple.left.val == bc_pair.left.val
        @test bc_from_tuple.right.val == bc_pair.right.val

        # Float32
        bc_f32 = BCPair(Deriv1(0.5f0), Deriv2(1.0f0))
        @test bc_f32 isa BCPair{Float32, Deriv1{Float32}, Deriv2{Float32}}
    end

    @testset "PeriodicBC Construction" begin
        # Default (Float64)
        pbc = PeriodicBC()
        @test pbc isa PeriodicBC{Float64}

        # Explicit Float64
        pbc64 = PeriodicBC{Float64}()
        @test pbc64 isa PeriodicBC{Float64}

        # Float32
        pbc32 = PeriodicBC{Float32}()
        @test pbc32 isa PeriodicBC{Float32}
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

        # PeriodicBC{T}(::PeriodicBC)
        pbc_f64 = PeriodicBC()
        pbc_f32 = PeriodicBC{Float32}(pbc_f64)
        @test pbc_f32 isa PeriodicBC{Float32}

        # NaturalBC{T}(::NaturalBC)
        nat_f64 = NaturalBC()
        nat_f32 = NaturalBC{Float32}(nat_f64)
        @test nat_f32 isa NaturalBC{Float32}

        # ClampedBC{T}(::ClampedBC)
        clamp_f64 = ClampedBC()
        clamp_f32 = ClampedBC{Float32}(clamp_f64)
        @test clamp_f32 isa ClampedBC{Float32}
    end

    # ========================================
    # BC Normalization (Coverage)
    # ========================================
    @testset "_normalize_bc Coverage" begin
        # Note: PeriodicBC is handled via _is_periodic_bc() before _normalize_bc is called.
        # No _normalize_bc(::PeriodicBC, T) method exists (dead code was removed).

        # BCPair with type promotion (Float64 BC → Float32 target)
        bc_f64 = BCPair(Deriv1(0.5), Deriv2(1.0))  # Float64
        bc_promoted = FastInterpolations._normalize_bc(bc_f64, Float32)
        @test bc_promoted isa BCPair{Float32, Deriv1{Float32}, Deriv2{Float32}}
        @test bc_promoted.left.val == Float32(0.5)
        @test bc_promoted.right.val == Float32(1.0)

        # PointBC with type promotion
        d1_f64 = Deriv1(0.25)  # Float64
        bc_from_d1 = FastInterpolations._normalize_bc(d1_f64, Float32)
        @test bc_from_d1 isa BCPair{Float32, Deriv1{Float32}, Deriv1{Float32}}
        @test bc_from_d1.left.val == Float32(0.25)
        @test bc_from_d1.right.val == Float32(0.25)

        d2_f64 = Deriv2(0.75)  # Float64
        bc_from_d2 = FastInterpolations._normalize_bc(d2_f64, Float32)
        @test bc_from_d2 isa BCPair{Float32, Deriv2{Float32}, Deriv2{Float32}}
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
    end

    # ========================================
    # CubicSplineCache with Generic BC
    # ========================================
    @testset "CubicSplineCache with Deriv1/Deriv2" begin
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

end
