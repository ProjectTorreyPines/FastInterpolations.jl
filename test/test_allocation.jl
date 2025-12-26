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
        @test allocs <= expected_output_allocs * 2 + ALLOC_THRESHOLD # Allow some overhead
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

    @testset "Zero-allocation: In-place with autocache" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        clear_cubic_cache!()

        # Prime autocache
        cubic_interp!(output, x, y, x_query)

        # Warmup
        cubic_interp!(output, x, y, x_query)
        cubic_interp!(output, x, y, x_query)

        # In-place with autocache - MUST be zero allocation
        allocs = @allocated cubic_interp!(output, x, y, x_query)
        @test allocs <= ALLOC_THRESHOLD
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
        @test stats.total_entries == 4
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
        @test cubic_cache_stats().total_entries == 3

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

        # Should only allocate y + z copies (~800 bytes for 51 elements each) + callable struct
        @test allocs < 1_500  # 2 KB budget (y + z vectors + minor overhead)
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
        cubic_interp(x, y, -0.1; extrap=:extension)
        cubic_interp(x, y, 1.1; extrap=:extension)

        # Extrapolation should still be zero-allocation on 1.12+
        allocs = @allocated cubic_interp(x, y, -0.1; extrap=:extension)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_interp(x, y, 1.1; extrap=:extension)
        @test allocs <= ALLOC_THRESHOLD
    end

    # =========================================================================
    # Linear Interpolation Allocation Tests
    # =========================================================================
    # Linear interpolation has no cache (O(1) or O(log n) lookup per point)

    @testset "Zero-allocation: linear_interp! in-place" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        # Warmup
        linear_interp!(output, x, y, x_query)
        linear_interp!(output, x, y, x_query)

        # In-place linear interpolation - MUST be zero allocation
        allocs = @allocated linear_interp!(output, x, y, x_query)
        @test allocs == 0
    end

    @testset "Zero-allocation: LinearInterpolant callable" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Create callable
        itp = LinearInterpolant(x, y)

        # Warmup
        itp(0.5)
        itp(0.5)

        # Scalar call - zero allocation on 1.12+
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated itp(0.25)
        @test allocs <= ALLOC_THRESHOLD
    end

    # =========================================================================
    # Periodic BC Allocation Tests
    # =========================================================================
    # Periodic BC now uses autocache for zero-allocation repeated interpolation.

    @testset "Periodic BC: CubicInterpolant callable is zero-allocation" begin
        # Periodic data (y[1] ≈ y[end])
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)

        # Create callable via cubic_interp (computes z coefficients)
        itp = cubic_interp(x, y; bc=PeriodicBC(), autocache=false)

        # Warmup
        itp(1.0)
        itp(1.0)

        # Scalar call on pre-constructed callable - zero allocation on 1.12+
        allocs = @allocated itp(1.0)
        @test allocs <= ALLOC_THRESHOLD

        # Different query points
        allocs = @allocated itp(3.0)
        @test allocs <= ALLOC_THRESHOLD

        # Query outside domain (wraps to domain)
        allocs = @allocated itp(7.0)  # 7.0 > 2π
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Periodic BC: cubic_interp with explicit cache is zero-allocation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        x_query = [0.5, 1.0, 1.5]
        output = similar(x_query)

        # Create explicit periodic cache
        cache = CubicSplineCache(x; bc=PeriodicBC())

        # Warmup
        cubic_interp!(output, cache, y, x_query)
        cubic_interp!(output, cache, y, x_query)

        # In-place with explicit cache - MUST be zero allocation
        allocs = @allocated cubic_interp!(output, cache, y, x_query)
        @test allocs == 0
    end

    @testset "Periodic BC: autocache achieves zero-allocation" begin
        # Periodic BC now uses autocache for zero-allocation repeated interpolation.

        # Use in-place version for accurate measurement (excludes output allocation)
        x_periodic = collect(range(0.0, 2π, 101))
        y_periodic = sin.(x_periodic)
        x_query = [1.0]
        output = similar(x_query)

        clear_cubic_cache!()

        # Prime periodic autocache (using typed BC API)
        cubic_interp!(output, x_periodic, y_periodic, x_query; bc=PeriodicBC())
        cubic_interp!(output, x_periodic, y_periodic, x_query; bc=PeriodicBC())

        # Natural BC with autocache for comparison (in-place)
        x_natural = collect(range(0.0, 1.0, 101))
        y_natural = sin.(2π .* x_natural)
        x_query_nat = [0.5]
        output_nat = similar(x_query_nat)
        cubic_interp!(output_nat, x_natural, y_natural, x_query_nat; bc=NaturalBC())  # Prime autocache
        cubic_interp!(output_nat, x_natural, y_natural, x_query_nat; bc=NaturalBC())  # Warmup
        natural_allocs = @allocated cubic_interp!(output_nat, x_natural, y_natural, x_query_nat; bc=NaturalBC())

        # Periodic BC with autocache (cache hit - zero allocation)
        periodic_allocs = @allocated cubic_interp!(output, x_periodic, y_periodic, x_query; bc=PeriodicBC())

        # Both natural and periodic BC should be zero-allocation with autocache
        @test natural_allocs <= ALLOC_THRESHOLD
        @test periodic_allocs <= ALLOC_THRESHOLD
    end

    @testset "Wrap extrap: LinearInterpolant callable is zero-allocation" begin
        # Wrap data
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)

        # Create callable
        itp = LinearInterpolant(x, y; extrap=:wrap)

        # Warmup
        itp(1.0)
        itp(1.0)

        # Scalar call - zero allocation on 1.12+
        allocs = @allocated itp(1.0)
        @test allocs <= ALLOC_THRESHOLD

        # Query outside domain (wraps)
        allocs = @allocated itp(7.0)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Wrap extrap: linear_interp functional API is zero-allocation" begin
        # Linear wrap doesn't need cache, so it should be zero-alloc
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)

        # Warmup
        linear_interp(x, y, 1.0; extrap=:wrap)
        linear_interp(x, y, 1.0; extrap=:wrap)

        # Linear interpolation is simple - should be zero-allocation
        allocs = @allocated linear_interp(x, y, 1.0; extrap=:wrap)
        @test allocs <= ALLOC_THRESHOLD

        # Outside domain
        allocs = @allocated linear_interp(x, y, 7.0; extrap=:wrap)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Wrap extrap: linear_interp! in-place is zero-allocation" begin
        # Linear wrap in-place - most accurate zero-alloc test
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        x_query = [1.0, 3.0, 7.0]  # includes out-of-domain (7.0 > 2π)
        output = similar(x_query)

        # Warmup
        linear_interp!(output, x, y, x_query; extrap=:wrap)
        linear_interp!(output, x, y, x_query; extrap=:wrap)

        # In-place linear wrap - MUST be zero allocation
        allocs = @allocated linear_interp!(output, x, y, x_query; extrap=:wrap)
        @test allocs <= ALLOC_THRESHOLD
    end

    # =========================================================================
    # Runtime Symbol Keyword API Tests
    # =========================================================================
    # These tests verify that passing symbols as runtime variables (not literals)
    # doesn't cause extra allocations. This validates the Val pattern refactoring
    # from Val(symbol_var) to direct branching with Val literals.

    @testset "Runtime symbol: linear_interp scalar" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        # Simulate user code passing runtime symbol
        function with_runtime_extrapolation(mode::Symbol)
            linear_interp(x, y, 0.5; extrap=mode)
        end

        # Warmup
        with_runtime_extrapolation(:extension)
        with_runtime_extrapolation(:extension)
        with_runtime_extrapolation(:wrap)
        with_runtime_extrapolation(:wrap)

        # Runtime extrapolation symbol - zero allocation on 1.12+
        allocs = @allocated with_runtime_extrapolation(:extension)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated with_runtime_extrapolation(:constant)
        @test allocs <= ALLOC_THRESHOLD

        # Runtime wrap extrap - zero allocation on 1.12+
        allocs = @allocated with_runtime_extrapolation(:wrap)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Runtime symbol: linear_interp! in-place" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        function inplace_runtime_extrapolation!(out, mode::Symbol)
            linear_interp!(out, x, y, x_query; extrap=mode)
        end

        # Warmup
        inplace_runtime_extrapolation!(output, :extension)
        inplace_runtime_extrapolation!(output, :extension)
        inplace_runtime_extrapolation!(output, :wrap)
        inplace_runtime_extrapolation!(output, :wrap)

        # Runtime extrapolation - MUST be zero allocation
        allocs = @allocated inplace_runtime_extrapolation!(output, :extension)
        @test allocs == 0

        allocs = @allocated inplace_runtime_extrapolation!(output, :constant)
        @test allocs == 0

        # Runtime wrap - MUST be zero allocation
        allocs = @allocated inplace_runtime_extrapolation!(output, :wrap)
        @test allocs == 0
    end

    @testset "Runtime symbol: cubic_interp scalar" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()

        # Prime cache
        cubic_interp(x, y, 0.5)

        function cubic_runtime_extrapolation(mode::Symbol)
            cubic_interp(x, y, 0.5; extrap=mode)
        end

        # Warmup ALL extrap modes (each mode = separate JIT path due to @_dispatch_extrap)
        for mode in (:none, :constant, :extension, :wrap)
            cubic_runtime_extrapolation(mode)
        end

        # Runtime extrapolation - zero allocation on 1.12+
        allocs = @allocated cubic_runtime_extrapolation(:extension)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_runtime_extrapolation(:constant)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Runtime symbol: cubic_interp! in-place" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        clear_cubic_cache!()

        function cubic_inplace_runtime_extrapolation!(out, mode::Symbol)
            cubic_interp!(out, x, y, x_query; extrap=mode)
        end

        # Warmup ALL extrap modes (each mode = separate JIT path due to @_dispatch_extrap)
        for mode in (:none, :constant, :extension, :wrap)
            cubic_inplace_runtime_extrapolation!(output, mode)
        end

        # Runtime extrapolation - MUST be zero allocation
        allocs = @allocated cubic_inplace_runtime_extrapolation!(output, :extension)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_inplace_runtime_extrapolation!(output, :constant)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_inplace_runtime_extrapolation!(output, :wrap)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Runtime symbol: get_cubic_cache" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Prime cache for both BC types (using typed BC API)
        get_cubic_cache(x, NaturalBC())
        get_cubic_cache(x, PeriodicBC())

        # Runtime BC type version (simulating user code passing runtime variable)
        function cache_runtime_bc_natural()
            get_cubic_cache(x, NaturalBC())
        end
        function cache_runtime_bc_periodic()
            get_cubic_cache(x, PeriodicBC())
        end

        # Extended warmup for JIT stabilization (important for inner functions)
        for _ in 1:10
            cache_runtime_bc_natural()
            cache_runtime_bc_periodic()
        end

        # Measure typed BC version
        allocs_natural = @allocated cache_runtime_bc_natural()
        allocs_periodic = @allocated cache_runtime_bc_periodic()

        # The 64/96 bytes comes from cache lookup (lock + objectid check).
        # Key test: repeated calls should be consistent (no growing allocation).
        allocs_natural2 = @allocated cache_runtime_bc_natural()
        allocs_periodic2 = @allocated cache_runtime_bc_periodic()

        # Same allocation on repeated calls = no leak
        @test allocs_natural == allocs_natural2
        @test allocs_periodic == allocs_periodic2

        # Reasonable budget (cache lookup overhead)
        @test allocs_natural <= 128    # Natural BC cache hit
        @test allocs_periodic <= 128   # Periodic BC cache hit
    end

    @testset "Runtime symbol: LinearInterpolant construction" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        function itp_runtime_extrapolation(mode::Symbol)
            LinearInterpolant(x, y; extrap=mode)
        end

        # Warmup ALL extrap modes (each mode = separate JIT path due to @_dispatch_extrap)
        for mode in (:none, :constant, :extension, :wrap)
            itp_runtime_extrapolation(mode)
            itp_runtime_extrapolation(mode)
        end

        # Construction allocates the struct itself, but no extra from Val pattern
        # Just verify it's reasonably small (struct + references only)
        allocs_ext = @allocated itp_runtime_extrapolation(:extension)
        allocs_const = @allocated itp_runtime_extrapolation(:constant)
        allocs_wrap = @allocated itp_runtime_extrapolation(:wrap)

        # All modes should have same allocation (no extra from runtime symbol)
        @test allocs_ext == allocs_const
        @test allocs_ext == allocs_wrap
        @test allocs_ext <= ALLOC_THRESHOLD + 64  # Struct (~48 bytes) + dispatch overhead
    end

    # =========================================================================
    # Dynamic BCPair Values in Loop Tests
    # =========================================================================
    # These tests verify that when BCPair derivative VALUES change dynamically
    # in a loop, the cache is still hit (same BC TYPE combination) and
    # zero-allocation is maintained.

    @testset "Dynamic BCPair values: scalar query zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.5

        clear_cubic_cache!()

        # Function simulating dynamic BC values in a loop
        function interp_with_dynamic_bc(left_curv::Float64, right_slope::Float64)
            cubic_interp(x, y, xi; bc=BCPair(D2(left_curv), D1(right_slope)))
        end

        # Prime cache (first call creates cache entry for D2-D1 type combination)
        interp_with_dynamic_bc(0.0, 0.0)

        # Warmup with varying values
        for i in 1:10
            interp_with_dynamic_bc(Float64(i) * 0.1, Float64(i) * 0.05)
        end

        # Test: different BC values but same BC TYPE should hit cache
        allocs = @allocated interp_with_dynamic_bc(0.5, 0.25)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated interp_with_dynamic_bc(1.0, -0.5)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated interp_with_dynamic_bc(-0.3, 0.8)
        @test allocs <= ALLOC_THRESHOLD

        # Verify cache was reused (only 1 miss for the initial prime)
        stats = cubic_cache_stats()
        @test stats.misses == 1
    end

    @testset "Dynamic BCPair values: in-place vector zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        clear_cubic_cache!()

        # Function simulating dynamic BC values in a loop (in-place version)
        function interp_inplace_dynamic_bc!(out, left_curv::Float64, right_slope::Float64)
            cubic_interp!(out, x, y, x_query; bc=BCPair(D2(left_curv), D1(right_slope)))
        end

        # Prime cache
        interp_inplace_dynamic_bc!(output, 0.0, 0.0)

        # Warmup with varying values
        for i in 1:10
            interp_inplace_dynamic_bc!(output, Float64(i) * 0.1, Float64(i) * 0.05)
        end

        # In-place with dynamic BC values - MUST be zero allocation
        allocs = @allocated interp_inplace_dynamic_bc!(output, 0.5, 0.25)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated interp_inplace_dynamic_bc!(output, 1.0, -0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Dynamic BCPair values: loop accumulation zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.5

        clear_cubic_cache!()

        # Realistic use case: accumulating results in a loop with varying BC
        function accumulate_with_varying_bc(n::Int)
            result = 0.0
            for i in 1:n
                left_curv = sin(Float64(i) * 0.1)
                right_slope = cos(Float64(i) * 0.1)
                result += cubic_interp(x, y, xi; bc=BCPair(D2(left_curv), D1(right_slope)))
            end
            return result
        end

        # Warmup
        accumulate_with_varying_bc(10)
        accumulate_with_varying_bc(10)

        # Loop with 100 iterations - should be ~zero allocation per iteration
        allocs = @allocated accumulate_with_varying_bc(100)
        # Allow small overhead per iteration (should be ≤ threshold * iterations in worst case)
        # But ideally much less due to cache hits
        @test allocs <= ALLOC_THRESHOLD * 100
    end

    @testset "Dynamic BCPair values: symmetric D2-D2 type" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.5

        clear_cubic_cache!()

        # Symmetric BC: D2 at both ends with different values
        function interp_symmetric_d2(left_curv::Float64, right_curv::Float64)
            cubic_interp(x, y, xi; bc=BCPair(D2(left_curv), D2(right_curv)))
        end

        # Prime cache
        interp_symmetric_d2(0.0, 0.0)

        # Warmup
        for i in 1:10
            interp_symmetric_d2(Float64(i) * 0.1, Float64(i) * 0.2)
        end

        # Different values, same type combination
        allocs = @allocated interp_symmetric_d2(0.5, 1.0)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated interp_symmetric_d2(-0.3, 0.8)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Dynamic BCPair values: symmetric D1-D1 type (ClampedBC equivalent)" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.5

        clear_cubic_cache!()

        # Symmetric BC: D1 at both ends with different values
        function interp_symmetric_d1(left_slope::Float64, right_slope::Float64)
            cubic_interp(x, y, xi; bc=BCPair(D1(left_slope), D1(right_slope)))
        end

        # Prime cache
        interp_symmetric_d1(0.0, 0.0)

        # Warmup
        for i in 1:10
            interp_symmetric_d1(Float64(i) * 0.1, Float64(i) * -0.1)
        end

        # Different values, same type combination
        allocs = @allocated interp_symmetric_d1(0.5, -0.5)
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated interp_symmetric_d1(1.0, 0.0)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Dynamic BCPair values: Float32 type" begin
        x = Float32.(collect(range(0.0, 1.0, 51)))
        y = Float32.(sin.(2π .* x))
        xi = Float32(0.5)

        clear_cubic_cache!()

        function interp_f32_dynamic_bc(left_curv::Float32, right_slope::Float32)
            cubic_interp(x, y, xi; bc=BCPair(D2(left_curv), D1(right_slope)))
        end

        # Prime cache
        interp_f32_dynamic_bc(Float32(0.0), Float32(0.0))

        # Warmup
        for i in 1:10
            interp_f32_dynamic_bc(Float32(i) * 0.1f0, Float32(i) * 0.05f0)
        end

        # Float32 with dynamic BC values
        allocs = @allocated interp_f32_dynamic_bc(Float32(0.5), Float32(0.25))
        @test allocs <= ALLOC_THRESHOLD
    end

end
