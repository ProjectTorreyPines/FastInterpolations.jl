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

# Import internal function for testing
import FastInterpolations: _get_cubic_cache

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
        @test allocs <= ALLOC_THRESHOLD
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
        @test allocs <= ALLOC_THRESHOLD
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

        # Add 5th grid - triggers ring buffer eviction
        x5 = collect(range(0.0, 5.0, 51))
        cubic_interp(x5, y, 0.5)

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

        # Add more grids - triggers evictions in ring buffer order
        x4 = collect(range(0.0, 4.0, 51))
        cubic_interp(x4, y, 0.5)

        x5 = collect(range(0.0, 5.0, 51))
        cubic_interp(x5, y, 0.5)

        x6 = collect(range(0.0, 6.0, 51))
        cubic_interp(x6, y, 0.5)

        x7 = collect(range(0.0, 7.0, 51))
        result = cubic_interp(x7, y, 0.5)
        @test isfinite(result)

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
        # Note: Tg/Tv type separation (for Complex support) adds ~300 bytes overhead
        @test allocs < 2_000  # 2 KB budget (y + z vectors + type parameter overhead)
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

        # Warmup with extrapolation (typed AbstractExtrap)
        cubic_interp(x, y, -0.1; extrap=ExtendExtrap())
        cubic_interp(x, y, 1.1; extrap=ExtendExtrap())

        # Extrapolation should still be zero-allocation on 1.12+
        allocs = @allocated cubic_interp(x, y, -0.1; extrap=ExtendExtrap())
        @test allocs <= ALLOC_THRESHOLD

        allocs = @allocated cubic_interp(x, y, 1.1; extrap=ExtendExtrap())
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
        @test allocs <= ALLOC_THRESHOLD
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
        @test allocs <= ALLOC_THRESHOLD
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

        # Zero-Curvature BC with autocache for comparison (in-place)
        x_natural = collect(range(0.0, 1.0, 101))
        y_natural = sin.(2π .* x_natural)
        x_query_nat = [0.5]
        output_nat = similar(x_query_nat)
        cubic_interp!(output_nat, x_natural, y_natural, x_query_nat; bc=ZeroCurvBC())  # Prime autocache
        cubic_interp!(output_nat, x_natural, y_natural, x_query_nat; bc=ZeroCurvBC())  # Warmup
        natural_allocs = @allocated cubic_interp!(output_nat, x_natural, y_natural, x_query_nat; bc=ZeroCurvBC())

        # Periodic BC with autocache (cache hit - zero allocation)
        periodic_allocs = @allocated cubic_interp!(output, x_periodic, y_periodic, x_query; bc=PeriodicBC())

        # Both ZeroCurv and periodic BC should be zero-allocation with autocache
        @test natural_allocs <= ALLOC_THRESHOLD
        @test periodic_allocs <= ALLOC_THRESHOLD
    end

    @testset "Wrap extrap: LinearInterpolant callable is zero-allocation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)

        itp = LinearInterpolant(x, y; extrap=WrapExtrap())

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
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)

        # Warmup
        linear_interp(x, y, 1.0; extrap=WrapExtrap())
        linear_interp(x, y, 1.0; extrap=WrapExtrap())

        # Linear interpolation is simple - should be zero-allocation
        allocs = @allocated linear_interp(x, y, 1.0; extrap=WrapExtrap())
        @test allocs <= ALLOC_THRESHOLD

        # Outside domain
        allocs = @allocated linear_interp(x, y, 7.0; extrap=WrapExtrap())
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Wrap extrap: linear_interp! in-place is zero-allocation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        x_query = [1.0, 3.0, 7.0]  # includes out-of-domain (7.0 > 2π)
        output = similar(x_query)

        # Warmup
        linear_interp!(output, x, y, x_query; extrap=WrapExtrap())
        linear_interp!(output, x, y, x_query; extrap=WrapExtrap())

        # In-place linear wrap - MUST be zero allocation
        allocs = @allocated linear_interp!(output, x, y, x_query; extrap=WrapExtrap())
        @test allocs <= ALLOC_THRESHOLD
    end

    # =========================================================================
    # Typed Extrapolation Mode Tests (AbstractExtrap)
    # =========================================================================
    # These tests verify zero-allocation for all 4 extrapolation modes
    # when passed as typed AbstractExtrap singletons (the recommended API).

    @testset "Typed extrap: linear_interp scalar" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        function linear_typed_extrap(mode::AbstractExtrap)
            linear_interp(x, y, 0.5; extrap=mode)
        end

        # Warmup all modes
        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            linear_typed_extrap(mode)
            linear_typed_extrap(mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            allocs = @allocated linear_typed_extrap(mode)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    @testset "Typed extrap: linear_interp! in-place" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        function linear_typed_extrap!(out, mode::AbstractExtrap)
            linear_interp!(out, x, y, x_query; extrap=mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            linear_typed_extrap!(output, mode)
            linear_typed_extrap!(output, mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            allocs = @allocated linear_typed_extrap!(output, mode)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    @testset "Typed extrap: cubic_interp scalar" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        clear_cubic_cache!()
        cubic_interp(x, y, 0.5)

        function cubic_typed_extrap(mode::AbstractExtrap)
            cubic_interp(x, y, 0.5; extrap=mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            cubic_typed_extrap(mode)
            cubic_typed_extrap(mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            allocs = @allocated cubic_typed_extrap(mode)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    @testset "Typed extrap: cubic_interp! in-place" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        clear_cubic_cache!()

        function cubic_typed_extrap!(out, mode::AbstractExtrap)
            cubic_interp!(out, x, y, x_query; extrap=mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            cubic_typed_extrap!(output, mode)
            cubic_typed_extrap!(output, mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            allocs = @allocated cubic_typed_extrap!(output, mode)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    @testset "Runtime symbol: _get_cubic_cache" begin
        clear_cubic_cache!()

        x = collect(range(0.0, 1.0, 51))

        # Prime cache for both BC types (using typed BC API)
        _get_cubic_cache(x, ZeroCurvBC())
        _get_cubic_cache(x, PeriodicBC())

        # Runtime BC type version (simulating user code passing runtime variable)
        function cache_runtime_bc_natural()
            _get_cubic_cache(x, ZeroCurvBC())
        end
        function cache_runtime_bc_periodic()
            _get_cubic_cache(x, PeriodicBC())
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
        @test allocs_natural <= 128    # Zero-Curvature BC cache hit
        @test allocs_periodic <= 128   # Periodic BC cache hit
    end

    @testset "Typed extrap: LinearInterpolant construction" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)

        function itp_typed_extrap(mode::AbstractExtrap)
            LinearInterpolant(x, y; extrap=mode)
        end

        for mode in (NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap())
            itp_typed_extrap(mode)
            itp_typed_extrap(mode)
        end

        # Construction allocates the struct itself, but no extra from mode dispatch
        allocs_ext = @allocated itp_typed_extrap(ExtendExtrap())
        allocs_const = @allocated itp_typed_extrap(ConstExtrap())
        allocs_wrap = @allocated itp_typed_extrap(WrapExtrap())

        # All modes should have same allocation
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
            cubic_interp(x, y, xi; bc=BCPair(Deriv2(left_curv), Deriv1(right_slope)))
        end

        # Prime cache (first call creates cache entry for Deriv2-Deriv1 type combination)
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
    end

    @testset "Dynamic BCPair values: in-place vector zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        x_query = [0.25, 0.5, 0.75]
        output = similar(x_query)

        clear_cubic_cache!()

        # Function simulating dynamic BC values in a loop (in-place version)
        function interp_inplace_dynamic_bc!(out, left_curv::Float64, right_slope::Float64)
            cubic_interp!(out, x, y, x_query; bc=BCPair(Deriv2(left_curv), Deriv1(right_slope)))
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
                result += cubic_interp(x, y, xi; bc=BCPair(Deriv2(left_curv), Deriv1(right_slope)))
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

    @testset "Dynamic BCPair values: symmetric Deriv2-Deriv2 type" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.5

        clear_cubic_cache!()

        # Symmetric BC: Deriv2 at both ends with different values
        function interp_symmetric_d2(left_curv::Float64, right_curv::Float64)
            cubic_interp(x, y, xi; bc=BCPair(Deriv2(left_curv), Deriv2(right_curv)))
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

    @testset "Dynamic BCPair values: symmetric Deriv1-Deriv1 type (ZeroSlopeBC equivalent)" begin
        x = collect(range(0.0, 1.0, 51))
        y = sin.(2π .* x)
        xi = 0.5

        clear_cubic_cache!()

        # Symmetric BC: Deriv1 at both ends with different values
        function interp_symmetric_d1(left_slope::Float64, right_slope::Float64)
            cubic_interp(x, y, xi; bc=BCPair(Deriv1(left_slope), Deriv1(right_slope)))
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
            cubic_interp(x, y, xi; bc=BCPair(Deriv2(left_curv), Deriv1(right_slope)))
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

    # =========================================================================
    # Range Cache Entry Efficiency Tests (Phase 1 - TDD Baseline)
    # =========================================================================
    # These tests verify that AbstractRange inputs remain memory-efficient.
    # Ranges are immutable and should NOT be materialized to Vector when cached.
    # Expected: PASS (establishes baseline for Range efficiency)

    @testset "Range cache entry does not allocate O(N) Vector" begin
        # Test 1.3: Verify Range cache operations are efficient

        @testset "Range cache hit is zero-allocation" begin
            clear_cubic_cache!()

            x = range(0.0, 1.0, 51)  # StepRangeLen
            y = sin.(2π .* collect(x))

            # Prime cache
            cubic_interp(x, y, 0.5)

            # Warmup
            cubic_interp(x, y, 0.5)
            cubic_interp(x, y, 0.5)

            # Range cache hit should be zero-allocation (same as Vector cache hit)
            allocs = @allocated cubic_interp(x, y, 0.5)
            @test allocs <= ALLOC_THRESHOLD
        end

        @testset "Range cache miss has bounded allocation" begin
            clear_cubic_cache!()

            x = range(0.0, 1.0, 101)
            y = sin.(2π .* collect(x))

            # Warmup JIT with different grid
            x_warmup = range(0.0, 2.0, 101)
            cubic_interp(x_warmup, sin.(2π .* collect(x_warmup)), 0.5)
            clear_cubic_cache!()

            # Cache miss for Range should NOT allocate O(N) bytes for the x snapshot
            # It should only allocate for LU factorization workspaces + cache entry overhead
            allocs = @allocated cubic_interp(x, y, 0.5)

            # Vector of 101 Float64 would be ~808 bytes
            # Range snapshot should be O(1) - just 3 numbers (first, last, length)
            # Total allocation should be dominated by LU factorization (~6-8 KB), not x snapshot
            @test allocs < 12_000  # Reasonable budget for cache creation
        end

        @testset "Float32 Range cache hit is zero-allocation" begin
            clear_cubic_cache!()

            x = range(Float32(0), Float32(1), 51)
            y = Float32.(sin.(2π .* collect(x)))

            # Prime cache
            cubic_interp(x, y, Float32(0.5))

            # Warmup
            cubic_interp(x, y, Float32(0.5))
            cubic_interp(x, y, Float32(0.5))

            # Float32 Range cache hit
            allocs = @allocated cubic_interp(x, y, Float32(0.5))
            @test allocs <= ALLOC_THRESHOLD
        end

        @testset "LinRange cache efficiency" begin
            clear_cubic_cache!()

            x = LinRange(0.0, 1.0, 51)
            y = cos.(collect(x))

            # Prime cache (LinRange gets normalized to StepRangeLen)
            cubic_interp(x, y, 0.5)

            # Warmup
            cubic_interp(x, y, 0.5)
            cubic_interp(x, y, 0.5)

            # LinRange cache hit - has small overhead from LinRange→StepRangeLen normalization
            # on each call (~112 bytes for creating StepRangeLen), but NOT O(N) allocation
            allocs = @allocated cubic_interp(x, y, 0.5)
            # Allow up to 200 bytes for range normalization overhead
            @test allocs <= ALLOC_THRESHOLD + 200
        end

        @testset "Periodic BC Range cache efficiency" begin
            clear_cubic_cache!()

            x = range(0.0, 2π, 51)
            y = sin.(collect(x))

            # Prime periodic cache
            cubic_interp(x, y, 1.0; bc=PeriodicBC())

            # Warmup
            cubic_interp(x, y, 1.0; bc=PeriodicBC())
            cubic_interp(x, y, 1.0; bc=PeriodicBC())

            # Periodic BC Range cache hit
            allocs = @allocated cubic_interp(x, y, 1.0; bc=PeriodicBC())
            @test allocs <= ALLOC_THRESHOLD
        end

        @testset "Range vs Vector cache miss allocation comparison" begin
            # This test verifies that Range cache miss doesn't allocate significantly more
            # than Vector cache miss (it should actually allocate LESS for the x storage)

            clear_cubic_cache!()
            n = 101

            # Measure Vector cache miss
            x_vec = collect(range(0.0, 1.0, n))
            y = sin.(2π .* x_vec)

            # Warmup JIT
            cubic_interp(x_vec, y, 0.5)
            clear_cubic_cache!()

            vec_allocs = @allocated cubic_interp(x_vec, y, 0.5)

            clear_cubic_cache!()

            # Measure Range cache miss
            x_range = range(0.0, 1.0, n)
            y_range = sin.(2π .* collect(x_range))

            range_allocs = @allocated cubic_interp(x_range, y_range, 0.5)

            # Range should allocate less or similar to Vector
            # (no need to copy O(N) x values for Range - it's stored compactly)
            @test range_allocs <= vec_allocs + 200  # Allow small overhead for normalization
        end
    end

end
