# Tests for quadratic (C1 piecewise quadratic) spline interpolation
#
# This file follows TDD: tests are written BEFORE implementation.
# Phase 1: BC Tags (Left/Right types)
# Phase 2: QuadraticSplineCache + Autocache

# ============================================================================
# Group 1: BC Type Tests (Phase 1)
# ============================================================================
@testset "Quadratic Interpolation - BC Types" begin

    @testset "Left BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Left(Deriv1(0.5))
            @test bc1 isa Left{Float64, Deriv1{Float64}}

            bc2 = Left(Deriv2(1.0))
            @test bc2 isa Left{Float64, Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Left(Deriv2(1.0f0))
            @test bc isa Left{Float32, Deriv2{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Left(Deriv1(1))
            @test bc isa Left{Float64, Deriv1{Float64}}
        end

        @testset "accessor (inner BC value)" begin
            bc = Left(Deriv1(0.5))
            @test bc.bc.val == 0.5

            bc2 = Left(Deriv2(2.0))
            @test bc2.bc.val == 2.0
        end

        @testset "type stability" begin
            @test @inferred(Left(Deriv1(0.0))) isa Left
            @test @inferred(Left(Deriv2(1.0))) isa Left
        end

        @testset "subtype relationship" begin
            @test Left{Float64, Deriv1{Float64}} <: AbstractBC{Float64}
            @test Left{Float64, Deriv2{Float64}} <: AbstractBC{Float64}
            @test Left{Float32, Deriv1{Float32}} <: AbstractBC{Float32}
        end
    end

    @testset "Right BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Right(Deriv1(-0.5))
            @test bc1 isa Right{Float64, Deriv1{Float64}}

            bc2 = Right(Deriv2(0.0))
            @test bc2 isa Right{Float64, Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Right(Deriv1(1.0f0))
            @test bc isa Right{Float32, Deriv1{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Right(Deriv2(0))
            @test bc isa Right{Float64, Deriv2{Float64}}
        end

        @testset "accessor (inner BC value)" begin
            bc = Right(Deriv1(-0.5))
            @test bc.bc.val == -0.5

            bc2 = Right(Deriv2(1.0))
            @test bc2.bc.val == 1.0
        end

        @testset "type stability" begin
            @test @inferred(Right(Deriv1(0.0))) isa Right
            @test @inferred(Right(Deriv2(1.0))) isa Right
        end

        @testset "subtype relationship" begin
            @test Right{Float64, Deriv1{Float64}} <: AbstractBC{Float64}
            @test Right{Float64, Deriv2{Float64}} <: AbstractBC{Float64}
            @test Right{Float32, Deriv2{Float32}} <: AbstractBC{Float32}
        end
    end

    @testset "Left/Right distinctness" begin
        # Left and Right should be distinct types
        @test Left(Deriv1(0.0)) isa Left
        @test Right(Deriv1(0.0)) isa Right
        @test !(Left(Deriv1(0.0)) isa Right)
        @test !(Right(Deriv1(0.0)) isa Left)
    end

end

# ============================================================================
# Group 2: Cache Tests (Phase 2)
# ============================================================================
@testset "Quadratic Interpolation - Cache" begin

    @testset "QuadraticSplineCache construction" begin
        x = collect(range(0.0, 1.0, 11))

        cache = QuadraticSplineCache(x)
        @test cache isa QuadraticSplineCache{Float64}
        @test length(cache.h) == 10
        @test length(cache.inv_h) == 10
        @test cache.h[1] ≈ 0.1
        @test cache.inv_h[1] ≈ 10.0
    end

    @testset "QuadraticSplineCache Float32" begin
        x32 = Float32.(collect(range(0.0, 1.0, 11)))
        cache32 = QuadraticSplineCache(x32)
        @test cache32 isa QuadraticSplineCache{Float32}
        @test eltype(cache32.h) === Float32
    end

    @testset "QuadraticSplineCache edge cases" begin
        # Too few points
        @test_throws ArgumentError QuadraticSplineCache([1.0])

        # Not strictly increasing
        @test_throws ArgumentError QuadraticSplineCache([1.0, 1.0, 2.0])
        @test_throws ArgumentError QuadraticSplineCache([1.0, 0.5, 2.0])

        # Minimum valid (n=2)
        cache_min = QuadraticSplineCache([0.0, 1.0])
        @test length(cache_min.h) == 1
    end

    @testset "QuadraticSplineCache non-uniform grid" begin
        x_nu = [0.0, 0.1, 0.3, 0.6, 1.0]
        cache = QuadraticSplineCache(x_nu)
        @test cache.h[1] ≈ 0.1
        @test cache.h[2] ≈ 0.2
        @test cache.h[3] ≈ 0.3
        @test cache.h[4] ≈ 0.4
    end

end

@testset "Quadratic Interpolation - Autocache" begin
    using FastInterpolations: _get_quadratic_cache

    @testset "autocache basic" begin
        x = collect(range(0.0, 1.0, 11))
        clear_quadratic_cache!()

        # First call creates cache
        cache1 = _get_quadratic_cache(x)
        @test cache1 isa QuadraticSplineCache

        # Second call returns same cache (RCU hit)
        cache2 = _get_quadratic_cache(x)
        @test cache1 === cache2  # identity check
    end

    @testset "autocache different grids" begin
        clear_quadratic_cache!()

        x1 = collect(range(0.0, 1.0, 11))
        x2 = collect(range(0.0, 2.0, 11))

        cache1 = _get_quadratic_cache(x1)
        cache2 = _get_quadratic_cache(x2)

        @test cache1 !== cache2
    end

    @testset "autocache disabled" begin
        x = collect(range(0.0, 1.0, 11))
        clear_quadratic_cache!()

        # With autocache disabled, should create new cache each time
        cache1 = _get_quadratic_cache(x; autocache=false)
        cache2 = _get_quadratic_cache(x; autocache=false)
        @test cache1 !== cache2
    end

    @testset "autocache zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        clear_quadratic_cache!()
        _get_quadratic_cache(x)  # prime

        allocs = @allocated _get_quadratic_cache(x)
        @test allocs == 0
    end

    @testset "get/set cache size" begin
        @test get_quadratic_cache_size() > 0
        @test get_quadratic_cache_size() isa Int
    end

end
