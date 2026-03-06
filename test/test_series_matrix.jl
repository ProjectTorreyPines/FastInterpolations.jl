# test/test_series_matrix.jl
# Phase B: Tests for shared matrix series infrastructure
# These tests verify the extracted lazy transpose and SIMD evaluation patterns

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "series_matrix - Lazy Transpose Infrastructure" begin

    @testset "LazyTranspose - single matrix" begin
        # Create a test matrix (n_points × n_series) = series-contiguous
        y = Float64[
            1 4 7;
            2 5 8;
            3 6 9
        ]  # 3 points × 3 series

        @testset "creates transpose on first call" begin
            lt = FI.LazyTranspose{Float64}()
            @test FI._get_snapshot(lt) === nothing

            y_point = FI._ensure_transpose!(lt, y)

            @test y_point !== nothing
            @test size(y_point) == (3, 3)  # (n_series × n_points)
            # Verify transpose correctness
            @test y_point[1, 1] == 1  # series 1, point 1
            @test y_point[2, 1] == 4  # series 2, point 1
            @test y_point[1, 3] == 3  # series 1, point 3
        end

        @testset "returns cached transpose on subsequent calls" begin
            lt = FI.LazyTranspose{Float64}()
            y_point1 = FI._ensure_transpose!(lt, y)
            y_point2 = FI._ensure_transpose!(lt, y)

            @test y_point1 === y_point2  # Same object (cached)
        end

        @testset "column access is contiguous for SIMD" begin
            lt = FI.LazyTranspose{Float64}()
            y_point = FI._ensure_transpose!(lt, y)

            # Column y_point[:, i] should give all series at point i
            @test y_point[:, 1] == [1, 4, 7]  # All series at point 1
            @test y_point[:, 2] == [2, 5, 8]  # All series at point 2
        end
    end

    @testset "LazyTransposePair - dual matrices (for Cubic)" begin
        y = Float64[1 4; 2 5; 3 6]  # 3 points × 2 series
        z = Float64[10 40; 20 50; 30 60]

        @testset "creates both transposes atomically" begin
            ltp = FI.LazyTransposePair{Float64}()
            @test FI._get_snapshot(ltp) === nothing

            y_point, z_point = FI._ensure_transpose_pair!(ltp, y, z)

            @test size(y_point) == (2, 3)
            @test size(z_point) == (2, 3)
            @test y_point[1, 1] == 1
            @test z_point[1, 1] == 10
        end

        @testset "returns cached pair on subsequent calls" begin
            ltp = FI.LazyTransposePair{Float64}()
            pair1 = FI._ensure_transpose_pair!(ltp, y, z)
            pair2 = FI._ensure_transpose_pair!(ltp, y, z)

            @test pair1[1] === pair2[1]  # Same y_point
            @test pair1[2] === pair2[2]  # Same z_point
        end
    end
end

@testset "series_matrix - Integration with CubicSeriesInterpolant" begin
    # Verify Cubic still works after refactoring to use shared infrastructure
    x = collect(0.0:0.1:1.0)  # 11 points
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-x)
    ys = [y1, y2, y3]

    sitp = cubic_interp(x, Series(ys))

    @testset "scalar evaluation unchanged" begin
        result = sitp(0.5)
        @test length(result) == 3
        @test result[1] ≈ sin(π) atol = 1.0e-10
        @test result[2] ≈ cos(π) atol = 1.0e-10
    end

    @testset "scalar in-place unchanged" begin
        output = zeros(3)
        sitp(output, 0.5)
        @test output[1] ≈ sin(π) atol = 1.0e-10
    end

    @testset "vector evaluation unchanged" begin
        xq = [0.25, 0.5, 0.75]
        results = sitp(xq)
        @test length(results) == 3
        @test all(r -> length(r) == 3, results)
    end

    @testset "derivatives unchanged" begin
        # First derivative at x=0.5
        deriv1 = sitp(0.5; deriv = DerivOp(1))
        @test length(deriv1) == 3
        @test deriv1[1] ≈ 2π * cos(π) atol = 0.1  # d/dx sin(2πx) = 2π cos(2πx)
    end

    @testset "zero allocation preserved" begin
        output = zeros(3)
        sitp(output, 0.5)  # Warmup

        allocs = @allocated sitp(output, 0.5)
        @test allocs <= ALLOC_THRESHOLD
    end
end
