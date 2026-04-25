# ALLOC_THRESHOLD is defined in runtests.jl

@testitem "Random Grid (Non-uniform Spacing)" setup = [AllocConstants] begin
    using Random
    # Random grids require binary search O(log n) - tests the non-uniform path

    @testset "Linear Interpolation - Random Grid" begin
        # Generate sorted random x-grid
        Random.seed!(42)
        n = 50
        x_random = sort(rand(n)) .* 10.0  # Random points in [0, 10]
        y = sin.(x_random)

        @testset "Interior point interpolation" begin
            # Query points within the random grid bounds
            x_min, x_max = extrema(x_random)
            x_query = [x_min + 0.1, (x_min + x_max) / 2, x_max - 0.1]

            result = linear_interp(x_random, y, x_query)

            @test result isa Vector{Float64}
            @test length(result) == 3
            @test all(isfinite, result)

            # Verify by manually computing for one point
            xi = x_query[2]
            idx = searchsortedlast(x_random, xi)
            idx = clamp(idx, 1, n - 1)
            α = (xi - x_random[idx]) / (x_random[idx + 1] - x_random[idx])
            expected = y[idx] * (1 - α) + y[idx + 1] * α
            @test result[2] ≈ expected
        end

        @testset "Knot passage" begin
            # Interpolation should pass through data points
            itp = linear_interp(x_random, y)
            for (xi, yi) in zip(x_random, y)
                @test itp(xi) ≈ yi
            end
        end

        @testset "Extrapolation" begin
            x_min, x_max = extrema(x_random)
            x_extrap = [x_min - 1.0, x_max + 1.0]

            # Extension extrapolation (explicit mode)
            result_ext = linear_interp(x_random, y, x_extrap; extrap = ExtendExtrap())
            @test all(isfinite, result_ext)

            # Constant extrapolation
            result_const = linear_interp(x_random, y, x_extrap; extrap = ClampExtrap())
            @test result_const[1] ≈ y[1]
            @test result_const[2] ≈ y[end]
        end

        @testset "Callable interface" begin
            itp = linear_interp(x_random, y)

            @test itp isa LinearInterpolant
            @test itp.x isa Vector{Float64}  # Random grid stored as Vector

            x_min, x_max = extrema(x_random)
            xi = (x_min + x_max) / 2

            # Scalar call
            val = itp(xi)
            @test val isa Float64
            @test isfinite(val)

            # Broadcast
            x_query = [x_min + 0.5, (x_min + x_max) / 2, x_max - 0.5]
            result = itp.(x_query)
            @test result == linear_interp(x_random, y, x_query)
        end

        @testset "Zero-allocation (scalar)" begin
            itp = linear_interp(x_random, y)
            x_min, x_max = extrema(x_random)
            xi = (x_min + x_max) / 2

            itp(xi)  # warmup
            allocs = @allocated itp(xi)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    @testset "Cubic Interpolation - Random Grid" begin
        Random.seed!(123)
        n = 50
        x_random = sort(rand(n)) .* 10.0
        y = sin.(x_random) .+ 0.3 .* cos.(2 .* x_random)

        @testset "Basic correctness" begin
            x_min, x_max = extrema(x_random)
            x_query = [x_min + 0.2, (x_min + x_max) / 2, x_max - 0.2]

            result = cubic_interp(x_random, y, x_query)

            @test result isa Vector{Float64}
            @test length(result) == 3
            @test all(isfinite, result)

            # Results should be in reasonable range
            y_min, y_max = extrema(y)
            @test all(y_min - 1.0 .<= result .<= y_max + 1.0)
        end

        @testset "Knot passage" begin
            cache = CubicSplineCache(x_random)
            for (xi, yi) in zip(x_random, y)
                @test cubic_interp(cache, y, xi) ≈ yi
            end
        end

        @testset "Callable interface" begin
            itp = cubic_interp(x_random, y; autocache = false)

            @test itp isa CubicInterpolant
            @test itp.cache.x isa Vector{Float64}  # Random grid stored as Vector

            x_min, x_max = extrema(x_random)
            xi = (x_min + x_max) / 2

            val = itp(xi)
            @test val isa Float64
            @test isfinite(val)
        end

        @testset "Zero-allocation (scalar)" begin
            itp = cubic_interp(x_random, y; autocache = false)
            x_min, x_max = extrema(x_random)
            xi = (x_min + x_max) / 2

            itp(xi)  # warmup
            allocs = @allocated itp(xi)
            @test allocs <= ALLOC_THRESHOLD
        end

        @testset "Cache reuse with random grid" begin
            cache = CubicSplineCache(x_random)

            # Multiple y vectors on same random x-grid
            y_vectors = [sin.(k .* x_random) for k in 1:5]
            x_min, x_max = extrema(x_random)
            x_query = collect(range(x_min + 0.1, x_max - 0.1, 10))

            for y_i in y_vectors
                result = cubic_interp(cache, y_i, x_query)
                @test all(isfinite, result)
            end
        end
    end

    @testset "Highly Non-uniform Grid (Clustered Points)" begin
        # Points clustered near boundaries - tests binary search edge cases
        n = 30
        x_clustered = vcat(
            range(0.0, 0.5, 10),      # Dense near start
            range(1.0, 9.0, 10),      # Sparse in middle
            range(9.5, 10.0, 10)      # Dense near end
        ) |> collect |> sort |> unique
        y = sin.(x_clustered)

        @testset "Linear interpolation" begin
            # Query in sparse and dense regions
            x_query = [0.25, 5.0, 9.75]
            result = linear_interp(x_clustered, y, x_query)

            @test all(isfinite, result)

            # Verify knot passage
            itp = linear_interp(x_clustered, y)
            for (xi, yi) in zip(x_clustered, y)
                @test itp(xi) ≈ yi
            end
        end

        @testset "Cubic interpolation" begin
            x_query = [0.25, 5.0, 9.75]
            result = cubic_interp(x_clustered, y, x_query)

            @test all(isfinite, result)

            # Verify knot passage (CubicFit → tiny boundary offset)
            cache = CubicSplineCache(x_clustered)
            for (xi, yi) in zip(x_clustered, y)
                @test cubic_interp(cache, y, xi) ≈ yi atol = 1.0e-14
            end
        end
    end

    @testset "Random Grid vs Uniform Grid Comparison" begin
        # Same function values but different x-grids should give similar results
        # when querying at the same interior points

        n = 51
        Random.seed!(999)

        # Create uniform and random grids with same bounds
        x_uniform = collect(range(0.0, 10.0, n))
        x_random = sort(vcat(0.0, rand(n - 2) .* 10.0, 10.0))  # Ensure same bounds

        # Same y values at grid points for uniform grid
        y_uniform = sin.(x_uniform)

        # For random grid, use same function
        y_random = sin.(x_random)

        # Query at midpoint (should be similar for smooth function)
        xi = 5.0
        result_uniform = linear_interp(x_uniform, y_uniform, xi)
        result_random = linear_interp(x_random, y_random, xi)

        # Both should approximate sin(5.0)
        @test result_uniform ≈ sin(5.0) atol = 0.1
        @test result_random ≈ sin(5.0) atol = 0.1
    end

    @testset "Edge Cases - Very Small Grid" begin
        # Minimum viable random grid
        x_small = [0.0, 0.7, 1.0]  # 3 points, non-uniform
        y_small = [1.0, 2.5, 1.5]

        @testset "Linear interpolation" begin
            result = linear_interp(x_small, y_small, 0.35)
            @test isfinite(result)

            # Verify knot passage
            for (xi, yi) in zip(x_small, y_small)
                @test linear_interp(x_small, y_small, xi) ≈ yi
            end
        end

        @testset "Cubic interpolation" begin
            result = cubic_interp(x_small, y_small, 0.35; bc = ZeroCurvBC())
            @test isfinite(result)

            # Verify knot passage (3-point grid needs ZeroCurvBC; CubicFit requires 4+)
            cache = CubicSplineCache(x_small; bc = ZeroCurvBC())
            for (xi, yi) in zip(x_small, y_small)
                @test cubic_interp(cache, y_small, xi) ≈ yi
            end
        end
    end

    @testset "Stress Test - Large Random Grid" begin
        Random.seed!(2024)
        n = 1000
        x_large = sort(rand(n)) .* 100.0
        y_large = sin.(x_large) .+ 0.1 .* x_large

        x_min, x_max = extrema(x_large)
        x_query = collect(range(x_min + 1.0, x_max - 1.0, 100))

        @testset "Linear interpolation" begin
            result = linear_interp(x_large, y_large, x_query)
            @test length(result) == 100
            @test all(isfinite, result)
        end

        @testset "Cubic interpolation" begin
            result = cubic_interp(x_large, y_large, x_query)
            @test length(result) == 100
            @test all(isfinite, result)
        end
    end
end
