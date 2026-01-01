# Tests for exact grid-value return (bit-exactness at grid points)
#
# Goal: When evaluating interpolator at query points that land exactly on grid
# coordinates x[i], return the stored data value y[i] bit-exactly.
#
# Note: Range grids may not achieve bit-exactness due to floating-point
# accumulation in range arithmetic. Vector grids should be bit-exact.

using Test
using FastInterpolations

@testset "Exact Grid-Value Return" begin

    @testset "Linear Interpolation - Vector Grid" begin
        # Non-uniform grid to ensure Vector dispatch path
        x = [0.0, 0.1, 0.35, 0.6, 0.85, 1.0]
        y = [1.5, -2.3, 0.7, 4.1, -1.2, 3.8]

        @testset "Scalar queries at grid points" begin
            for i in eachindex(x)
                result = linear_interp(x, y, x[i])
                @test result === y[i]
            end
        end

        @testset "Boundary points" begin
            @test linear_interp(x, y, first(x)) === first(y)
            @test linear_interp(x, y, last(x)) === last(y)
        end

        @testset "Vector query at all grid points" begin
            result = linear_interp(x, y, x)
            for i in eachindex(x)
                @test result[i] === y[i]
            end
        end

        @testset "LinearInterpolant callable" begin
            itp = linear_interp(x, y)
            for i in eachindex(x)
                @test itp(x[i]) === y[i]
            end
        end

        @testset "Special floating-point values" begin
            # Test with values that could cause floating-point issues
            x_special = [0.0, 0.1, 0.2, 0.3]
            y_special = [1e-15, 1e15, -1e-15, -1e15]  # Very small and large values

            for i in eachindex(x_special)
                result = linear_interp(x_special, y_special, x_special[i])
                @test result === y_special[i]
            end
        end
    end

    @testset "Linear Interpolation - Range Grid" begin
        # Uniform grid uses Range dispatch path
        x = 0.0:0.1:1.0
        y = collect(sin.(x))

        @testset "Boundary points" begin
            # Boundaries should still work due to first(x)/last(x) matching
            @test linear_interp(x, y, first(x)) === first(y)
            @test linear_interp(x, y, last(x)) === last(y)
        end

        @testset "Interior grid points (approximate)" begin
            # Interior points may not be bit-exact for Range due to FP accumulation
            # Use approximate comparison as the baseline
            for i in 2:length(x)-1
                result = linear_interp(x, y, x[i])
                @test result ≈ y[i]
            end
        end
    end

    @testset "Cubic Interpolation - Vector Grid" begin
        # Non-uniform grid to ensure Vector dispatch path
        x = [0.0, 0.15, 0.4, 0.65, 0.9, 1.0]
        y = [2.1, -1.5, 3.2, 0.8, -2.4, 1.7]

        @testset "Scalar queries at grid points" begin
            for i in eachindex(x)
                result = cubic_interp(x, y, x[i])
                # Currently uses ≈ because cubic formula accumulates FP errors
                # After optimization, this should be ===
                @test result ≈ y[i]
            end
        end

        @testset "Scalar queries at grid points - bit-exact" begin
            # This is the target behavior after implementing exact grid-value return
            for i in eachindex(x)
                result = cubic_interp(x, y, x[i])
                @test_skip result === y[i]  # Enable after implementing optimization
            end
        end

        @testset "Boundary points" begin
            @test cubic_interp(x, y, first(x)) ≈ first(y)
            @test cubic_interp(x, y, last(x)) ≈ last(y)
        end

        @testset "Vector query at all grid points" begin
            result = cubic_interp(x, y, x)
            for i in eachindex(x)
                @test result[i] ≈ y[i]
            end
        end

        @testset "CubicInterpolant callable" begin
            itp = cubic_interp(x, y)
            for i in eachindex(x)
                @test itp(x[i]) ≈ y[i]
            end
        end

        @testset "With CubicSplineCache" begin
            cache = CubicSplineCache(x)
            for i in eachindex(x)
                result = cubic_interp(cache, y, x[i])
                @test result ≈ y[i]
            end
        end
    end

    @testset "Cubic Interpolation - Range Grid" begin
        # Uniform grid uses Range dispatch path
        x = 0.0:0.1:1.0
        y = collect(cos.(x))

        @testset "Boundary points" begin
            @test cubic_interp(x, y, first(x)) ≈ first(y)
            @test cubic_interp(x, y, last(x)) ≈ last(y)
        end

        @testset "All grid points (approximate)" begin
            result = cubic_interp(x, y, collect(x))
            @test result ≈ y
        end
    end

    @testset "Cubic Interpolation - Periodic BC" begin
        x = collect(range(0.0, 2π, 17))
        y = sin.(x)
        y[end] = y[1]  # Force exact periodicity

        @testset "Grid points with periodic BC" begin
            for i in eachindex(x)
                result = cubic_interp(x, y, x[i]; bc=PeriodicBC())
                @test result ≈ y[i]
            end
        end
    end

    @testset "Edge Cases" begin
        @testset "Two-point grid" begin
            x = [0.0, 1.0]
            y = [3.5, -2.1]

            # Linear should be exact at endpoints
            @test linear_interp(x, y, x[1]) === y[1]
            @test linear_interp(x, y, x[2]) === y[2]
        end

        @testset "Three-point grid (minimum for cubic)" begin
            x = [0.0, 0.5, 1.0]
            y = [1.0, 2.0, 0.5]

            # Linear
            for i in eachindex(x)
                @test linear_interp(x, y, x[i]) === y[i]
            end

            # Cubic (approximate for now)
            for i in eachindex(x)
                @test cubic_interp(x, y, x[i]) ≈ y[i]
            end
        end

        @testset "Negative coordinates" begin
            x = [-1.0, -0.5, 0.0, 0.5, 1.0]
            y = [2.0, -1.0, 0.5, 1.5, -0.5]

            for i in eachindex(x)
                @test linear_interp(x, y, x[i]) === y[i]
                @test cubic_interp(x, y, x[i]) ≈ y[i]
            end
        end

        @testset "Large coordinate values" begin
            x = [1e6, 1e6 + 0.1, 1e6 + 0.5, 1e6 + 1.0]
            y = [1.0, 2.0, 1.5, 3.0]

            for i in eachindex(x)
                @test linear_interp(x, y, x[i]) === y[i]
                @test cubic_interp(x, y, x[i]) ≈ y[i]
            end
        end
    end

    @testset "Float32 Support" begin
        x = Float32[0.0, 0.25, 0.5, 0.75, 1.0]
        y = Float32[1.0, 2.5, -0.5, 3.2, 1.8]

        @testset "Linear - Float32" begin
            for i in eachindex(x)
                result = linear_interp(x, y, x[i])
                @test result === y[i]
                @test result isa Float32
            end
        end

        @testset "Cubic - Float32 (approximate)" begin
            for i in eachindex(x)
                result = cubic_interp(x, y, x[i])
                @test result ≈ y[i]
                @test result isa Float32
            end
        end
    end

end
