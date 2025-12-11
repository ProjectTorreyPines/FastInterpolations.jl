using Test
using FastInterpolations

@testset "FastInterpolations.jl" begin

    @testset "Linear Interpolation" begin
        @testset "Uniform grid (Range) - basic" begin
            x = 0.0:0.1:1.0
            y = sin.(x)
            x_targets = [0.25, 0.5, 0.75]

            result = linear_interp(x, y, x_targets)
            @test result isa Vector{Float64}
            @test length(result) == 3
            @test all(isfinite, result)
        end

        @testset "Non-uniform grid (Vector) - basic" begin
            x = [0.0, 0.1, 0.3, 0.6, 1.0]
            y = sin.(x)
            x_targets = [0.05, 0.2, 0.45, 0.8]

            result = linear_interp(x, y, x_targets)
            @test result isa Vector{Float64}
            @test length(result) == 4
            @test all(isfinite, result)
        end

        @testset "In-place version" begin
            x = 0.0:0.1:1.0
            y = sin.(x)
            x_targets = [0.25, 0.5, 0.75]
            output = similar(x_targets)

            linear_interp!(output, x, y, x_targets)
            @test output == linear_interp(x, y, x_targets)
        end

        @testset "Zero-allocation (in-place)" begin
            x = 0.0:0.01:1.0
            y = sin.(x)
            x_targets = [0.25, 0.5, 0.75]
            output = similar(x_targets)

            linear_interp!(output, x, y, x_targets)  # Warmup
            allocs = @allocated linear_interp!(output, x, y, x_targets)
            @test allocs == 0
        end

        @testset "Extrapolation :extension" begin
            x = 0.0:0.1:1.0
            y = 2.0 .* collect(x) .+ 1.0
            x_targets = [-0.2, 1.2]

            result = linear_interp(x, y, x_targets; extrapolation=:extension)
            @test result[1] ≈ 2.0 * (-0.2) + 1.0
            @test result[2] ≈ 2.0 * 1.2 + 1.0
        end

        @testset "Extrapolation :constant" begin
            x = 0.0:0.1:1.0
            y = sin.(x)
            x_targets = [-0.2, 1.2]

            result = linear_interp(x, y, x_targets; extrapolation=:constant)
            @test result[1] == y[1]
            @test result[2] == y[end]
        end

        @testset "Callable interpolator" begin
            x = 0.0:0.1:1.0
            y = sin.(x)

            itp = linear_interp(x, y)
            @test itp isa LinearInterpCallable

            # Scalar call
            val = itp(0.5)
            @test val isa Float64
            @test val == linear_interp(x, y, 0.5)

            # Broadcast
            x_targets = [0.25, 0.5, 0.75]
            result = itp.(x_targets)
            @test result == linear_interp(x, y, x_targets)
        end

        @testset "Integer input auto-promotion" begin
            x_int = 0:10
            y_int = [i^2 for i in x_int]

            result = linear_interp(x_int, y_int, [5.5])
            @test result isa Vector{Float64}
            @test result[1] ≈ 25.0 + 0.5 * (36.0 - 25.0)
        end
    end

    @testset "Cubic Interpolation" begin
        @testset "Basic correctness" begin
            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            x_query = [0.25, 0.5, 0.75]

            result = cubic_interp(x, y, x_query)
            @test result isa Vector{Float64}
            @test length(result) == 3
            @test all(isfinite, result)
        end

        @testset "Cache construction" begin
            x = collect(range(0.0, 1.0, 51))
            cache = CubicSplineCache(x)

            @test cache isa CubicSplineCache{Float64}
            @test length(cache.x) == 51
        end

        @testset "Cache reuse" begin
            x = collect(range(0.0, 1.0, 51))
            cache = CubicSplineCache(x)
            x_query = [0.25, 0.5, 0.75]

            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            result1 = cubic_interp(cache, y1, x_query)
            result2 = cubic_interp(cache, y2, x_query)

            @test result1 != result2  # Different y should give different results
            @test all(isfinite, result1)
            @test all(isfinite, result2)
        end

        @testset "In-place version" begin
            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            x_query = [0.25, 0.5, 0.75]
            output = Vector{Float64}(undef, 3)

            cache = CubicSplineCache(x)
            cubic_interp!(output, cache, y, x_query)
            @test output == cubic_interp(cache, y, x_query)
        end

        @testset "Auto-cache functionality" begin
            clear_cubic_cache!()

            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)
            x_query = [0.25, 0.5, 0.75]

            # First call - cache miss
            result1 = cubic_interp(x, y, x_query)
            stats1 = cubic_cache_stats()
            @test stats1.misses == 1
            @test stats1.hits == 0

            # Second call - cache hit
            result2 = cubic_interp(x, cos.(2π .* x), x_query)
            stats2 = cubic_cache_stats()
            @test stats2.hits == 1
        end

        @testset "Callable interpolator" begin
            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)

            itp = cubic_interp(x, y)
            @test itp isa CubicInterpCallable

            # Scalar call
            val = itp(0.5)
            @test val isa Float64
            @test isfinite(val)

            # Broadcast
            x_targets = [0.25, 0.5, 0.75]
            result = itp.(x_targets)
            @test all(isfinite, result)
        end

        @testset "Integer input auto-promotion" begin
            x_int = 0:10
            y_int = [sin(2π * i / 10) for i in x_int]

            result = cubic_interp(x_int, y_int, [2.5, 5.5])
            @test result isa Vector{Float64}
            @test all(isfinite, result)
        end

        @testset "Float32 support" begin
            x = Float32.(range(0.0, 1.0, 21))
            y = sin.(Float32(2π) .* x)
            x_query = Float32[0.25, 0.75]

            result = cubic_interp(x, y, x_query)
            @test result isa Vector{Float32}
            @test all(isfinite, result)
        end
    end

end
