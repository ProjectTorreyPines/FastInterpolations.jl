# Import internal function for testing
import FastInterpolations: _get_cubic_cache

@testset "Cubic Spline Auto-Cache" begin
    # Clear cache before tests
    clear_cubic_cache!()

    @testset "Basic auto-cache reuse" begin
        x = collect(range(0.0, 1.0, 51))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-x)
        x_query = [0.25, 0.5, 0.75]

        clear_cubic_cache!()

        # First call - cache miss
        result1 = cubic_interp(x, y1, x_query)

        # Second call with same x - cache hit
        result2 = cubic_interp(x, y2, x_query)

        # Third call with same x - another cache hit
        result3 = cubic_interp(x, y3, x_query)

        # Results should be different (different y values)
        @test result1 != result2
        @test result2 != result3
        @test result1 != result3

        # But all should be finite and reasonable
        @test all(isfinite, result1)
        @test all(isfinite, result2)
        @test all(isfinite, result3)
    end

    @testset "Multiple x-grids" begin
        clear_cubic_cache!()

        # Create 3 different x-grids
        x1 = collect(range(0.0, 1.0, 51))
        x2 = collect(range(0.0, 2.0, 51))
        x3 = collect(range(0.0, 3.0, 51))

        y = sin.(2π .* x1)
        x_query = [0.25, 0.5, 0.75]

        # Each grid creates a new cache entry
        r1 = cubic_interp(x1, y, x_query)
        r2 = cubic_interp(x2, y, x_query)
        r3 = cubic_interp(x3, y, x_query)

        # Reuse first grid - should hit cache and give same result
        r1_again = cubic_interp(x1, y, x_query)
        @test r1 ≈ r1_again
    end

    @testset "Cache size limit" begin
        clear_cubic_cache!()
        set_cubic_cache_size!(3)  # Small cache for testing

        # Create 4 different x-grids (exceeds cache size)
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:4]
        y = ones(51)
        x_query = [0.25]

        # Fill cache to limit (3 grids)
        for i in 1:3
            cubic_interp(grids[i], y, x_query)
        end

        # Add 4th grid - should evict oldest
        result4 = cubic_interp(grids[4], y, x_query)
        @test isfinite(result4[1])

        # Reset cache size to default
        set_cubic_cache_size!(16)
    end

    @testset "autocache parameter control" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5]

        # With autocache=true (default)
        result1 = cubic_interp(x, y, x_query)

        # Another call with autocache=true - should hit cache
        result2 = cubic_interp(x, y, x_query; autocache=true)

        # With autocache=false - creates fresh cache each time
        result3 = cubic_interp(x, y, x_query; autocache=false)

        # Results should be identical
        @test result1 ≈ result2
        @test result2 ≈ result3
    end

    @testset "Manual cache control" begin
        clear_cubic_cache!()

        # Test get/set cache size
        @test get_cubic_cache_size() == 16  # Default
        set_cubic_cache_size!(32)
        @test get_cubic_cache_size() == 32
        set_cubic_cache_size!(16)  # Reset

        # Test clear works without error
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        cubic_interp(x, y, [0.5])

        clear_cubic_cache!()  # Should complete without error
    end

    @testset "Hash collision handling (stress test)" begin
        clear_cubic_cache!()

        # Create many similar grids to increase collision probability
        n_grids = 50
        grids = [collect(range(0.0, 1.0, 51)) .+ (i * 1e-10) for i in 1:n_grids]
        y = ones(51)
        x_query = [0.5]

        # All should cache successfully and produce valid results
        for grid in grids
            result = cubic_interp(grid, y, x_query)
            @test isfinite(result[1])
        end
    end

    @testset "Scalar query point with autocache" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Scalar query with autocache
        result1 = cubic_interp(x, y, 0.5)
        result2 = cubic_interp(x, y, 0.75)

        # Disable autocache for scalar query
        result3 = cubic_interp(x, y, 0.25; autocache=false)

        @test isa(result1, Float64)
        @test isa(result2, Float64)
        @test isa(result3, Float64)
        @test isfinite(result1)
        @test isfinite(result2)
        @test isfinite(result3)
    end

    @testset "Correctness under heavy reuse" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        x_query = [0.2, 0.4, 0.6, 0.8]

        # Generate 20 different y functions
        for k in 1:20
            y = @. sin(k * 2π * x) + 0.5 * cos(k * 4π * x)

            result = cubic_interp(x, y, x_query)

            # Verify all results are finite
            @test all(isfinite, result)
        end
    end

    @testset "Auto-cache with Integer inputs" begin
        clear_cubic_cache!()

        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]

        # First call with integer inputs (should create cache)
        result1 = cubic_interp(x_int, y_int, x_query_float)

        # Second call with same integer x (should reuse cache)
        y_int2 = [cos(2π * i / 10) for i in x_int]
        result2 = cubic_interp(x_int, y_int2, x_query_float)

        # Verify results are correct
        @test all(isfinite, result1)
        @test all(isfinite, result2)
        @test result1 != result2
    end

    @testset "Autocache=false with Integer inputs" begin
        clear_cubic_cache!()

        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]

        result = cubic_interp(x_int, y_int, x_query_float; autocache=false)
        @test result isa Vector{Float64}
        @test all(isfinite, result)
    end

    @testset "AbstractRange fallback paths" begin
        clear_cubic_cache!()

        # Test with UnitRange (not StepRangeLen)
        x_unitrange = 1:51
        y = sin.(2π .* collect(x_unitrange) ./ 51)

        # This should trigger the AbstractRange{Float64} fallback
        result = cubic_interp(Float64.(x_unitrange), Float64.(y), 25.0)
        @test !isnan(result)
    end

    @testset "Float32 cache operations" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(2)  # Small cache for testing

        y = Float32.(ones(51))

        # Fill cache with Float32 Range grids
        x1_range = range(Float32(0), Float32(1), 51)
        x2_range = range(Float32(0), Float32(2), 51)
        x3_range = range(Float32(0), Float32(3), 51)

        r1 = cubic_interp(x1_range, y, Float32(0.5))
        r2 = cubic_interp(x2_range, y, Float32(0.5))
        r3 = cubic_interp(x3_range, y, Float32(0.5))  # Should trigger eviction

        @test isfinite(r1)
        @test isfinite(r2)
        @test isfinite(r3)

        # Fill cache with Float32 Vector grids
        x1_vec = Float32.(collect(range(0.0, 1.0, 51)))
        x2_vec = Float32.(collect(range(0.0, 2.0, 51)))
        x3_vec = Float32.(collect(range(0.0, 3.0, 51)))

        r4 = cubic_interp(x1_vec, y, Float32(0.5))
        r5 = cubic_interp(x2_vec, y, Float32(0.5))
        r6 = cubic_interp(x3_vec, y, Float32(0.5))  # Should trigger eviction

        @test isfinite(r4)
        @test isfinite(r5)
        @test isfinite(r6)

        set_cubic_cache_size!(old_size)
    end

    @testset "Float32 self-healing path" begin
        clear_cubic_cache!()

        x1 = Float32.(collect(range(0.0, 1.0, 51)))
        y = Float32.(sin.(2π .* x1))

        # Prime cache
        result1 = cubic_interp(x1, y, Float32(0.5))

        # Create equal but different object
        x2 = Float32.(collect(range(0.0, 1.0, 51)))
        @test x1 == x2
        @test objectid(x1) != objectid(x2)

        # This should trigger Pass 2 (equality check) for Float32
        result2 = cubic_interp(x2, y, Float32(0.5))

        # Results should be the same
        @test result1 ≈ result2
    end

    @testset "Range eviction paths" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(2)

        y = ones(51)

        # Test Float64 Range eviction
        grids = [range(0.0, Float64(i), 51) for i in 1:4]
        for grid in grids
            result = cubic_interp(grid, y, 0.5)
            @test isfinite(result)
        end

        set_cubic_cache_size!(old_size)
    end

    @testset "Range cache hit" begin
        clear_cubic_cache!()

        x = range(0.0, 1.0, 51)
        y = sin.(2π .* collect(x))

        # Prime cache
        result1 = cubic_interp(x, y, 0.5)

        # Second call should hit cache (same objectid for Range)
        result2 = cubic_interp(x, y, 0.5)

        @test result1 ≈ result2
    end

    # =========================================================================
    # AbstractVector API Compatibility Tests
    # =========================================================================

    @testset "_get_cubic_cache accepts AbstractVector (views, SubArrays)" begin
        clear_cubic_cache!()

        x_full = collect(range(0.0, 1.0, 101))
        y_full = sin.(2π .* x_full)

        # SubArray (view) should work
        x_view = @view x_full[1:51]
        y_view = @view y_full[1:51]

        cache = _get_cubic_cache(x_view, NaturalBC())
        @test cache isa CubicSplineCache

        # Cache should store collected Vector, not the view
        result = cubic_interp(collect(x_view), collect(y_view), 0.5)
        @test isfinite(result)
    end

    @testset "_get_cubic_cache accepts Float32 views" begin
        clear_cubic_cache!()

        x_full = Float32.(collect(range(0.0, 1.0, 101)))
        x_view = @view x_full[1:51]

        cache = _get_cubic_cache(x_view, NaturalBC())
        @test cache isa CubicSplineCache{Float32}
    end

    @testset "_get_cubic_cache fallback for other Real types" begin
        clear_cubic_cache!()

        # Integer range → should convert to Float64
        x_int = 0:10
        cache_int = _get_cubic_cache(x_int, NaturalBC())
        @test cache_int isa CubicSplineCache{Float64}

        # Float16 vector → kept as Float16 (native AbstractFloat)
        x_f16 = Float16.(collect(range(0.0, 1.0, 11)))
        cache_f16 = _get_cubic_cache(x_f16, NaturalBC())
        @test cache_f16 isa CubicSplineCache{Float16}
    end

    @testset "_get_cubic_cache keyword API" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Keyword API should work
        cache1 = _get_cubic_cache(x)  # default bc=NaturalBC()
        cache2 = _get_cubic_cache(x; bc=NaturalBC())
        cache3 = _get_cubic_cache(x; bc=PeriodicBC())

        @test cache1 isa CubicSplineCache
        @test cache2 isa CubicSplineCache
        @test cache3 isa CubicSplineCache

        # Natural and periodic caches should be different types
        @test typeof(cache1) == typeof(cache2)
        @test typeof(cache1) != typeof(cache3)
    end

    @testset "_get_cubic_cache typed BC API (type-stable path)" begin
        clear_cubic_cache!()

        x64 = collect(range(0.0, 1.0, 51))
        x32 = Float32.(x64)
        x_range = range(0.0, 1.0, 51)

        # All input types should work with Val API
        c1 = _get_cubic_cache(x64, NaturalBC())
        c2 = _get_cubic_cache(x32, NaturalBC())
        c3 = _get_cubic_cache(x_range, NaturalBC())
        c4 = _get_cubic_cache(x64, PeriodicBC())

        @test c1 isa CubicSplineCache{Float64}
        @test c2 isa CubicSplineCache{Float32}
        @test c3 isa CubicSplineCache{Float64}  # Range normalizes to Float64
        @test c4 isa CubicSplineCache{Float64}

        # Periodic cache should have PeriodicData BC
        @test c4.bc_data !== nothing
    end

    # =========================================================================
    # Coverage Tests for Uncovered Paths
    # =========================================================================

    @testset "_get_cubic_cache with ClampedBC (typed API)" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # ClampedBC typed API - previously uncovered
        cache = _get_cubic_cache(x, ClampedBC())
        @test cache isa CubicSplineCache{Float64}

        # Cache should be created with BCPair(Deriv1(0), Deriv1(0))
        @test cache.bc_data isa BCPair{Float64, Deriv1{Float64}, Deriv1{Float64}}

        # Should work for Float32 as well
        x32 = Float32.(x)
        cache32 = _get_cubic_cache(x32, ClampedBC())
        @test cache32 isa CubicSplineCache{Float32}
        @test cache32.bc_data isa BCPair{Float32, Deriv1{Float32}, Deriv1{Float32}}

        # Range input
        x_range = range(0.0, 1.0, 51)
        cache_range = _get_cubic_cache(x_range, ClampedBC())
        @test cache_range isa CubicSplineCache{Float64}
    end

    @testset "_get_cubic_cache with PointBC (convenience API)" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Deriv1 PointBC - applies symmetrically to both ends
        # Note: LU factorization depends only on BC TYPE, not values.
        # Cache stores values from first caller, but values are applied at solve time.
        cache_d1 = _get_cubic_cache(x, Deriv1(0.5))
        @test cache_d1 isa CubicSplineCache{Float64}
        @test cache_d1.bc_data isa BCPair{Float64, Deriv1{Float64}, Deriv1{Float64}}
        # On cache miss, actual BC values from request are stored
        @test cache_d1.bc_data.left.val == 0.5
        @test cache_d1.bc_data.right.val == 0.5

        # Deriv2 PointBC - applies symmetrically to both ends
        cache_d2 = _get_cubic_cache(x, Deriv2(1.0))
        @test cache_d2 isa CubicSplineCache{Float64}
        @test cache_d2.bc_data isa BCPair{Float64, Deriv2{Float64}, Deriv2{Float64}}
        # On cache miss, actual BC values from request are stored
        @test cache_d2.bc_data.left.val == 1.0
        @test cache_d2.bc_data.right.val == 1.0

        # Float32 with PointBC
        x32 = Float32.(x)
        cache_d1_32 = _get_cubic_cache(x32, Deriv1(Float32(0.5)))
        @test cache_d1_32 isa CubicSplineCache{Float32}
    end

    @testset "Int Vector fallback paths" begin
        clear_cubic_cache!()

        # Integer Vector with BCPair - should convert to Float64
        x_int = collect(0:10)
        cache = _get_cubic_cache(x_int, BCPair(Deriv1(0.0), Deriv2(0.0)))
        @test cache isa CubicSplineCache{Float64}

        # Integer Vector with periodic BC - should convert to Float64
        clear_cubic_cache!()
        cache_periodic = _get_cubic_cache(x_int, PeriodicBC())
        @test cache_periodic isa CubicSplineCache{Float64}

        # Integer Range with periodic BC - should convert to Float64
        clear_cubic_cache!()
        x_int_range = 0:10
        cache_periodic_range = _get_cubic_cache(x_int_range, PeriodicBC())
        @test cache_periodic_range isa CubicSplineCache{Float64}
    end

    @testset "Periodic cache self-healing path" begin
        clear_cubic_cache!()

        # Create first grid and cache
        x1 = collect(range(0.0, 2π, 51))
        y = sin.(x1)
        result1 = cubic_interp(x1, y, 0.5; bc=PeriodicBC())

        # Create equal but different object (different objectid)
        x2 = collect(range(0.0, 2π, 51))
        @test x1 == x2
        @test objectid(x1) != objectid(x2)

        # This should trigger Pass 2 (equality check) and self-healing
        result2 = cubic_interp(x2, y, 0.5; bc=PeriodicBC())

        # Results should be the same
        @test result1 ≈ result2

        # Now x2 should trigger Pass 1 (identity check) due to self-healing
        result3 = cubic_interp(x2, y, 0.5; bc=PeriodicBC())
        @test result2 ≈ result3
    end
end
