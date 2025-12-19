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
        stats1 = cubic_cache_stats()
        @test stats1.misses == 1
        @test stats1.hits == 0
        @test stats1.size == 1

        # Second call with same x - cache hit
        result2 = cubic_interp(x, y2, x_query)
        stats2 = cubic_cache_stats()
        @test stats2.misses == 1
        @test stats2.hits == 1
        @test stats2.size == 1

        # Third call with same x - another cache hit
        result3 = cubic_interp(x, y3, x_query)
        stats3 = cubic_cache_stats()
        @test stats3.misses == 1
        @test stats3.hits == 2
        @test stats3.size == 1
        @test stats3.efficiency == 66.7  # 2 hits / 3 total = 66.7%

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

        # First grid - miss
        cubic_interp(x1, y, x_query)
        stats1 = cubic_cache_stats()
        @test stats1.size == 1
        @test stats1.misses == 1

        # Second grid - miss
        cubic_interp(x2, y, x_query)
        stats2 = cubic_cache_stats()
        @test stats2.size == 2
        @test stats2.misses == 2

        # Third grid - miss
        cubic_interp(x3, y, x_query)
        stats3 = cubic_cache_stats()
        @test stats3.size == 3
        @test stats3.misses == 3

        # Reuse first grid - hit
        cubic_interp(x1, y, x_query)
        stats4 = cubic_cache_stats()
        @test stats4.size == 3
        @test stats4.hits == 1
        @test stats4.misses == 3
    end

    @testset "Cache size limit and LRU eviction" begin
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
        stats = cubic_cache_stats()
        @test stats.size == 3
        @test stats.misses == 3

        # Add 4th grid - should evict oldest
        cubic_interp(grids[4], y, x_query)
        stats = cubic_cache_stats()
        @test stats.size == 3  # Still at limit
        @test stats.evictions == 1

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
        stats1 = cubic_cache_stats()
        @test stats1.size == 1
        @test stats1.misses == 1

        # Another call with autocache=true - should hit
        result2 = cubic_interp(x, y, x_query; autocache=true)
        stats2 = cubic_cache_stats()
        @test stats2.hits == 1
        @test stats2.size == 1

        # With autocache=false - should not affect cache
        result3 = cubic_interp(x, y, x_query; autocache=false)
        stats3 = cubic_cache_stats()
        @test stats3.hits == 1  # No new hits
        @test stats3.misses == 1  # No new misses
        @test stats3.size == 1  # Cache size unchanged

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

        # Test clear
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        cubic_interp(x, y, [0.5])
        stats_before = cubic_cache_stats()
        @test stats_before.size > 0

        clear_cubic_cache!()
        stats_after = cubic_cache_stats()
        @test stats_after.size == 0
        @test stats_after.hits == 0
        @test stats_after.misses == 0
        @test stats_after.evictions == 0
    end

    @testset "Hash collision handling (stress test)" begin
        clear_cubic_cache!()

        # Create many similar grids to increase collision probability
        n_grids = 50
        grids = [collect(range(0.0, 1.0, 51)) .+ (i * 1e-10) for i in 1:n_grids]
        y = ones(51)
        x_query = [0.5]

        # All should cache successfully
        for grid in grids
            cubic_interp(grid, y, x_query)
        end

        stats = cubic_cache_stats()
        # With default cache size of 16, we should have evictions
        @test stats.size <= 16
        @test stats.misses >= n_grids
        @test stats.evictions >= (n_grids - 16)
    end

    @testset "Scalar query point with autocache" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Scalar query with autocache
        result1 = cubic_interp(x, y, 0.5)
        stats1 = cubic_cache_stats()
        @test stats1.size == 1
        @test stats1.misses == 1

        # Another scalar query - cache hit
        result2 = cubic_interp(x, y, 0.75)
        stats2 = cubic_cache_stats()
        @test stats2.hits == 1

        # Disable autocache for scalar query
        result3 = cubic_interp(x, y, 0.25; autocache=false)
        stats3 = cubic_cache_stats()
        @test stats3.hits == 1  # No new hits

        @test isa(result1, Float64)
        @test isa(result2, Float64)
        @test isa(result3, Float64)
    end

    @testset "Statistics accuracy" begin
        clear_cubic_cache!()

        # Create multiple grids and track stats
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:5]
        x_query = [0.25, 0.5, 0.75]

        # First pass - all misses
        for (i, x) in enumerate(grids)
            y = sin.(2π .* x)
            result = cubic_interp(x, y, x_query)
            stats = cubic_cache_stats()
            @test stats.misses == i
            @test stats.hits == 0
        end

        # Second pass - all hits
        for (i, x) in enumerate(grids)
            y = cos.(π .* x)
            result = cubic_interp(x, y, x_query)
            stats = cubic_cache_stats()
            @test stats.hits == i
            @test stats.misses == 5  # Should stay at 5
        end

        final_stats = cubic_cache_stats()
        @test final_stats.efficiency > 40.0  # 5 hits / 10 total = 50%
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

        # Verify cache was used (19 hits after first miss)
        stats = cubic_cache_stats()
        @test stats.misses == 1
        @test stats.hits == 19
    end

    @testset "Auto-cache with Integer inputs" begin
        clear_cubic_cache!()

        x_int = 0:10
        y_int = [sin(2π * i / 10) for i in x_int]
        x_query_float = [2.5, 5.5, 7.3]

        # First call with integer inputs (should create cache)
        result1 = cubic_interp(x_int, y_int, x_query_float)
        stats1 = cubic_cache_stats()
        @test stats1.misses == 1

        # Second call with same integer x (should reuse cache)
        y_int2 = [cos(2π * i / 10) for i in x_int]
        result2 = cubic_interp(x_int, y_int2, x_query_float)
        stats2 = cubic_cache_stats()
        @test stats2.hits == 1

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

        # Cache should not be populated
        stats = cubic_cache_stats()
        @test stats.misses == 0
        @test stats.hits == 0
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

        cubic_interp(x1_range, y, Float32(0.5))
        cubic_interp(x2_range, y, Float32(0.5))
        cubic_interp(x3_range, y, Float32(0.5))  # Should trigger eviction

        # Fill cache with Float32 Vector grids
        x1_vec = Float32.(collect(range(0.0, 1.0, 51)))
        x2_vec = Float32.(collect(range(0.0, 2.0, 51)))
        x3_vec = Float32.(collect(range(0.0, 3.0, 51)))

        cubic_interp(x1_vec, y, Float32(0.5))
        cubic_interp(x2_vec, y, Float32(0.5))
        cubic_interp(x3_vec, y, Float32(0.5))  # Should trigger eviction

        set_cubic_cache_size!(old_size)
    end

    @testset "Float32 self-healing path" begin
        clear_cubic_cache!()

        x1 = Float32.(collect(range(0.0, 1.0, 51)))
        y = Float32.(sin.(2π .* x1))

        # Prime cache
        cubic_interp(x1, y, Float32(0.5))

        # Create equal but different object
        x2 = Float32.(collect(range(0.0, 1.0, 51)))
        @test x1 == x2
        @test objectid(x1) != objectid(x2)

        # This should trigger Pass 2 (equality check) for Float32
        cubic_interp(x2, y, Float32(0.5))

        stats = cubic_cache_stats()
        @test stats.misses == 1  # Only first call is a miss
    end

    @testset "Range eviction paths" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(2)

        y = ones(51)

        # Test Float64 Range eviction
        grids = [range(0.0, Float64(i), 51) for i in 1:4]
        for grid in grids
            cubic_interp(grid, y, 0.5)
        end

        stats = cubic_cache_stats()
        @test stats.evictions >= 2  # At least 2 evictions for 4 grids in size-2 cache

        set_cubic_cache_size!(old_size)
    end

    @testset "Range cache hit" begin
        clear_cubic_cache!()

        x = range(0.0, 1.0, 51)
        y = sin.(2π .* collect(x))

        # Prime cache
        cubic_interp(x, y, 0.5)

        # Second call should hit cache (same objectid for Range)
        cubic_interp(x, y, 0.5)

        stats = cubic_cache_stats()
        # Should have 1 miss (first call) and 1 hit (second call)
        @test stats.misses == 1
        @test stats.hits >= 1
    end

    # =========================================================================
    # AbstractVector API Compatibility Tests
    # =========================================================================

    @testset "get_cubic_cache accepts AbstractVector (views, SubArrays)" begin
        clear_cubic_cache!()

        x_full = collect(range(0.0, 1.0, 101))
        y_full = sin.(2π .* x_full)

        # SubArray (view) should work
        x_view = @view x_full[1:51]
        y_view = @view y_full[1:51]

        cache = get_cubic_cache(x_view, Val(:natural))
        @test cache isa CubicSplineCache

        # Cache should store collected Vector, not the view
        result = cubic_interp(collect(x_view), collect(y_view), 0.5)
        @test isfinite(result)

        stats = cubic_cache_stats()
        @test stats.misses == 1
    end

    @testset "get_cubic_cache accepts Float32 views" begin
        clear_cubic_cache!()

        x_full = Float32.(collect(range(0.0, 1.0, 101)))
        x_view = @view x_full[1:51]

        cache = get_cubic_cache(x_view, Val(:natural))
        @test cache isa CubicSplineCache{Float32}
    end

    @testset "get_cubic_cache fallback for other Real types" begin
        clear_cubic_cache!()

        # Integer range → should convert to Float64
        x_int = 0:10
        cache_int = get_cubic_cache(x_int, Val(:natural))
        @test cache_int isa CubicSplineCache{Float64}

        # Float16 vector → should convert to Float64
        x_f16 = Float16.(collect(range(0.0, 1.0, 11)))
        cache_f16 = get_cubic_cache(x_f16, Val(:natural))
        @test cache_f16 isa CubicSplineCache{Float64}
    end

    @testset "get_cubic_cache keyword API" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Keyword API should work
        cache1 = get_cubic_cache(x)  # default bc=:natural
        cache2 = get_cubic_cache(x; bc=:natural)
        cache3 = get_cubic_cache(x; bc=:periodic)

        @test cache1 isa CubicSplineCache
        @test cache2 isa CubicSplineCache
        @test cache3 isa CubicSplineCache

        # Natural and periodic caches should be different types
        @test typeof(cache1) == typeof(cache2)
        @test typeof(cache1) != typeof(cache3)
    end

    @testset "get_cubic_cache Val API (type-stable path)" begin
        clear_cubic_cache!()

        x64 = collect(range(0.0, 1.0, 51))
        x32 = Float32.(x64)
        x_range = range(0.0, 1.0, 51)

        # All input types should work with Val API
        c1 = get_cubic_cache(x64, Val(:natural))
        c2 = get_cubic_cache(x32, Val(:natural))
        c3 = get_cubic_cache(x_range, Val(:natural))
        c4 = get_cubic_cache(x64, Val(:periodic))

        @test c1 isa CubicSplineCache{Float64}
        @test c2 isa CubicSplineCache{Float32}
        @test c3 isa CubicSplineCache{Float64}  # Range normalizes to Float64
        @test c4 isa CubicSplineCache{Float64}

        # Periodic cache should have PeriodicData BC
        @test c4.bc_data !== nothing
    end
end
