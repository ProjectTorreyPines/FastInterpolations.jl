"""
    test_allocation.jl

Comprehensive allocation tests for cubic spline interpolation.
Verifies zero-allocation guarantees for cache hits and optimized paths.

These tests validate the 2-Pass lookup with self-healing optimization:
- Pass 1: objectid() comparison (O(1), zero-allocation)
- Pass 2: isequal() comparison + self-healing (O(N), zero-allocation)
- Ring Buffer: O(1) eviction without memory movement

Note: Julia 1.12+ has improved escape analysis that eliminates small allocations
from mutable struct field access. Older versions may show ~16-64 bytes allocation.
"""

# ALLOC_THRESHOLD is defined in runtests.jl

@testset "Allocation Tests" begin

    # =========================================================================
    # Core Zero-Allocation Tests (Critical Path)
    # =========================================================================

    @testset "Zero-allocation: Cache hit (same object - Pass 1)" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()

        # Prime cache (creates entry)
        cubic_interp(x, y, 0.5)

        # Warmup (ensures JIT compilation complete)
        cubic_interp(x, y, 0.5)
        cubic_interp(x, y, 0.5)

        # Pass 1 hit (same objectid) - zero allocation on 1.12+
        allocs = @allocated cubic_interp(x, y, 0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Different query point - still zero allocation
        allocs = @allocated cubic_interp(x, y, 0.75)
        @test allocs <= ALLOC_THRESHOLD

        # Vector query allocates output array only
        x_query = [0.25, 0.5, 0.75]
        cubic_interp(x, y, x_query)  # Warmup vector path
        cubic_interp(x, y, x_query)
        allocs = @allocated cubic_interp(x, y, x_query)
        expected_output_allocs = sizeof(Float64) * length(x_query) + 40  # Array header
        @test allocs <= expected_output_allocs * 2  # Allow some overhead
    end

    @testset "Zero-allocation: Self-healing cache hit (Pass 2 → Pass 1)" begin
        clear_cubic_cache!()

        x1 = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x1)

        # Prime cache with x1
        cubic_interp(x1, y, 0.5)

        # Create NEW object with SAME content
        x2 = collect(range(0.0, 1.0, 51))
        @test x1 == x2                        # Equal content
        @test objectid(x1) != objectid(x2)    # Different memory addresses

        # First call with x2: Pass 2 hit + self-healing
        # (This call updates entry.id = objectid(x2))
        cubic_interp(x2, y, 0.5)
        cubic_interp(x2, y, 0.5)  # Warmup

        # Second call with x2: NOW Pass 1 hit (id was healed)
        allocs = @allocated cubic_interp(x2, y, 0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: In-place with explicit cache" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        # Create explicit cache (bypasses autocache)
        cache = CubicSplineCache(x)

        # Warmup
        cubic_interp!(output, cache, y, x_query)
        cubic_interp!(output, cache, y, x_query)

        # In-place with cache - MUST be zero allocation
        allocs = @allocated cubic_interp!(output, cache, y, x_query)
        @test allocs == 0
    end

    @testset "Zero-allocation: Callable scalar call" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Create callable interpolant (pre-computes z coefficients)
        itp = cubic_interp(x, y; autocache=false)

        # Warmup
        itp(0.5)
        itp(0.5)

        # Scalar call - zero allocation on 1.12+ (uses pre-computed z)
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Different query points - still zero allocation
        allocs = @allocated itp(0.25)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated itp(0.99)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: Callable with autocache" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()

        # Create callable with autocache (uses shared cache)
        itp = cubic_interp(x, y)  # autocache=true by default

        # Warmup
        itp(0.5)
        itp(0.5)

        # Scalar call on callable - zero allocation on 1.12+
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    # =========================================================================
    # Float32 Allocation Tests
    # =========================================================================

    @testset "Zero-allocation: Float32 cache hit" begin
        x = Float32.(collect(range(0.0, 1.0, 51)))
        y = Float32.(sin.(2π .* x))

        clear_cubic_cache!()

        # Prime cache
        cubic_interp(x, y, Float32(0.5))

        # Warmup
        cubic_interp(x, y, Float32(0.5))
        cubic_interp(x, y, Float32(0.5))

        # Cache hit - zero allocation on 1.12+
        allocs = @allocated cubic_interp(x, y, Float32(0.5))
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: Float32 in-place with cache" begin
        x = Float32.(collect(range(0.0, 1.0, 51)))
        y = Float32.(sin.(2π .* x))
        x_query = Float32[0.25, 0.5, 0.75]
        output = similar(x_query)

        cache = CubicSplineCache(x)

        # Warmup
        cubic_interp!(output, cache, y, x_query)
        cubic_interp!(output, cache, y, x_query)

        # In-place - zero allocation
        allocs = @allocated cubic_interp!(output, cache, y, x_query)
        @test allocs == 0
    end

    # =========================================================================
    # Ring Buffer Eviction Tests
    # =========================================================================

    @testset "Ring buffer eviction maintains zero-allocation" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(4)  # Small cache for testing

        y = ones(51)

        # Fill cache with 4 different x-grids
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:4]
        for grid in grids
            cubic_interp(grid, y, 0.5)
        end

        stats = cubic_cache_stats()
        @test stats.size == 4
        @test stats.evictions == 0

        # Add 5th grid - triggers ring buffer eviction
        x5 = collect(range(0.0, 5.0, 51))
        cubic_interp(x5, y, 0.5)

        stats = cubic_cache_stats()
        @test stats.evictions == 1

        # Warmup with x5
        cubic_interp(x5, y, 0.5)

        # Cache hit on x5 should still be zero allocation after eviction
        allocs = @allocated cubic_interp(x5, y, 0.5)
        @test allocs <= ALLOC_THRESHOLD

        set_cubic_cache_size!(old_size)
    end

    @testset "Ring buffer circular eviction" begin
        clear_cubic_cache!()
        old_size = get_cubic_cache_size()
        set_cubic_cache_size!(3)

        y = ones(51)

        # Fill cache (indices 1, 2, 3)
        grids = [collect(range(0.0, Float64(i), 51)) for i in 1:3]
        for grid in grids
            cubic_interp(grid, y, 0.5)
        end

        # Add 4th → evicts index 1 (ring pointer was at 1)
        x4 = collect(range(0.0, 4.0, 51))
        cubic_interp(x4, y, 0.5)
        @test cubic_cache_stats().evictions == 1

        # Add 5th → evicts index 2
        x5 = collect(range(0.0, 5.0, 51))
        cubic_interp(x5, y, 0.5)
        @test cubic_cache_stats().evictions == 2

        # Add 6th → evicts index 3
        x6 = collect(range(0.0, 6.0, 51))
        cubic_interp(x6, y, 0.5)
        @test cubic_cache_stats().evictions == 3

        # Add 7th → evicts index 1 again (wrapped around)
        x7 = collect(range(0.0, 7.0, 51))
        cubic_interp(x7, y, 0.5)
        @test cubic_cache_stats().evictions == 4

        # Size stays at limit
        @test cubic_cache_stats().size == 3

        set_cubic_cache_size!(old_size)
    end

    # =========================================================================
    # Allocation Budget Tests
    # =========================================================================

    @testset "Cache miss allocation budget" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Warmup JIT with different grid
        x_warmup = collect(range(0.0, 2.0, 51))
        cubic_interp(x_warmup, y, 0.5)
        clear_cubic_cache!()

        # Cache miss creates new cache entry
        allocs = @allocated cubic_interp(x, y, 0.5)

        # Cache miss should allocate ~6-8 KB (LU factorization, workspaces, etc.)
        # Allow reasonable budget
        @test allocs < 10_000  # 10 KB budget
        @test allocs > 1_000   # Should allocate something for cache creation
    end

    @testset "Callable creation allocation budget" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()

        # Callable creation (with autocache=false creates new cache)
        # Warmup
        _ = cubic_interp(x, y; autocache=false)

        # Measure
        allocs = @allocated cubic_interp(x, y; autocache=false)

        # Should include: cache creation + z copy
        @test allocs < 15_000  # 15 KB budget
    end

    @testset "Callable creation with autocache reuses cache" begin
        x = collect(range(0.0, 1.0, 51))

        clear_cubic_cache!()

        # Prime autocache
        y1 = sin.(2π .* x)
        _ = cubic_interp(x, y1)

        # Create callable with autocache (should reuse existing cache)
        y2 = cos.(2π .* x)

        # Warmup
        _ = cubic_interp(x, y2)

        # Callable creation with autocache - only allocates z vector
        allocs = @allocated cubic_interp(x, y2)

        # Should only allocate z copy (~400 bytes for 51 elements) + callable struct
        @test allocs < 1_000  # 1 KB budget (z vector + minor overhead)
    end

    # =========================================================================
    # Broadcast Fusion Tests
    # =========================================================================

    @testset "Broadcast fusion allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Create callable
        itp = cubic_interp(x, y; autocache=false)

        # Single element broadcast
        query_points = [0.25, 0.5, 0.75]

        # Warmup (multiple times for JIT stabilization)
        for _ in 1:5
            _ = itp.(query_points)
        end

        # Broadcast allocates output array + broadcast machinery overhead
        allocs = @allocated itp.(query_points)
        # Allow reasonable overhead: output array + broadcast generator
        @test allocs < 500  # 500 bytes budget for 3-element output
    end

    @testset "Fused broadcast with operations" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        itp = cubic_interp(x, y; autocache=false)

        query_points = [0.25, 0.5, 0.75]
        coef = 2.5

        # Warmup (multiple times)
        for _ in 1:5
            _ = @. coef * itp(query_points)
        end

        # Fused broadcast - output allocation + broadcast overhead
        allocs = @allocated @. coef * itp(query_points)
        @test allocs < 500  # 500 bytes budget
    end

    # =========================================================================
    # Stress Tests
    # =========================================================================

    @testset "Repeated cache hits maintain zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()

        # Prime cache
        cubic_interp(x, y, 0.5)

        # Warmup
        for _ in 1:10
            cubic_interp(x, y, 0.5)
        end

        # Many repeated calls should all be zero-allocation
        total_allocs = 0
        for i in 1:100
            allocs = @allocated cubic_interp(x, y, 0.5)
            total_allocs += allocs
        end

        @test total_allocs <= ALLOC_THRESHOLD * 100
    end

    @testset "Multiple y vectors with same x maintain zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))

        clear_cubic_cache!()

        # Prime cache with first y
        y1 = sin.(2π .* x)
        cubic_interp(x, y1, 0.5)

        # Multiple y vectors with same x
        y_vectors = [sin.(k .* 2π .* x) for k in 1:10]

        # Warmup
        for y in y_vectors
            cubic_interp(x, y, 0.5)
        end

        # All calls with same x should be zero-allocation
        for y in y_vectors
            allocs = @allocated cubic_interp(x, y, 0.5)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    @testset "Self-healing across multiple objects" begin
        clear_cubic_cache!()

        y = sin.(2π .* collect(range(0.0, 1.0, 51)))

        # Create multiple objects with same content
        objects = [collect(range(0.0, 1.0, 51)) for _ in 1:5]

        # All should be equal in content
        for obj in objects
            @test obj == objects[1]
        end

        # All should have different objectids
        ids = [objectid(obj) for obj in objects]
        @test length(unique(ids)) == 5

        # Prime with first object
        cubic_interp(objects[1], y, 0.5)

        # Use each object - self-healing should make subsequent calls fast
        for obj in objects
            # First call triggers self-healing
            cubic_interp(obj, y, 0.5)
            cubic_interp(obj, y, 0.5)  # Warmup

            # After self-healing, should be zero-allocation on 1.12+
            allocs = @allocated cubic_interp(obj, y, 0.5)
            @test allocs <= ALLOC_THRESHOLD
        end

        # Verify cache was reused (only 1 miss)
        stats = cubic_cache_stats()
        @test stats.misses == 1
        @test stats.hits >= 10  # Multiple hits from warmup + measurement
    end

    # =========================================================================
    # Edge Cases
    # =========================================================================

    @testset "Zero-allocation at grid boundaries" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()
        cubic_interp(x, y, 0.5)

        # Warmup at boundaries
        cubic_interp(x, y, 0.0)
        cubic_interp(x, y, 1.0)

        # At exact grid points
        allocs = @allocated cubic_interp(x, y, 0.0)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_interp(x, y, 1.0)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation with extrapolation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()
        cubic_interp(x, y, 0.5)

        # Warmup with extrapolation (explicit :extension mode)
        cubic_interp(x, y, -0.1; extrapolation=:extension)
        cubic_interp(x, y, 1.1; extrapolation=:extension)

        # Extrapolation should still be zero-allocation on 1.12+
        allocs = @allocated cubic_interp(x, y, -0.1; extrapolation=:extension)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_interp(x, y, 1.1; extrapolation=:extension)
        @test allocs <= ALLOC_THRESHOLD
    end

end
