# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  SERIES INTERPOLANT RANGE GRID TESTS                      ║
# ║         Tests for Range vs Vector grid support in series interpolants     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Verifies that series interpolants correctly handle both Range (StepRangeLen)
# and Vector grid types, producing consistent results.

@testitem "Series Interpolant Range Grid Support" begin
    FI = FastInterpolations

    # ========================================
    # Test Data Setup
    # ========================================

    n_pts = 101
    x_range = range(0.0, 1.0, n_pts)
    x_vector = collect(x_range)

    # Test functions
    y1 = sin.(2π .* x_vector)
    y2 = cos.(2π .* x_vector)
    ys = [y1, y2]

    # Query points
    xq_single = 0.35
    xq_vector = [0.1, 0.25, 0.5, 0.75, 0.9]

    # ========================================
    # LinearSeriesInterpolant
    # ========================================

    @testset "LinearSeriesInterpolant" begin
        sitp_range = linear_interp(x_range, Series(ys))
        sitp_vector = linear_interp(x_vector, Series(ys))

        @testset "grid type preservation" begin
            @test sitp_range.x isa FI._CachedRange
            # Vector inputs are wrapped in `_CachedVector` (caches h/inv_h);
            # `_CachedVector <: AbstractVector` so search/eval kernels see no change.
            @test sitp_vector.x isa FI._CachedVector{Float64, Float64}
        end

        @testset "single query evaluation" begin
            result_range = sitp_range(xq_single)
            result_vector = sitp_vector(xq_single)

            @test result_range isa Vector{Float64}
            @test length(result_range) == 2
            @test result_range ≈ result_vector atol = 1.0e-14
        end

        @testset "vector query evaluation" begin
            result_range = sitp_range(xq_vector)
            result_vector = sitp_vector(xq_vector)

            # Output structure: [series1_all_queries, series2_all_queries]
            @test result_range isa Vector{Vector{Float64}}
            @test length(result_range) == 2  # n_series
            @test all(length.(result_range) .== length(xq_vector))  # n_queries each
            @test result_range ≈ result_vector atol = 1.0e-14
        end

        @testset "output correctness" begin
            # At x=0.5, sin(π) ≈ 0, cos(π) ≈ -1
            result = sitp_range(0.5)
            @test abs(result[1]) < 0.1  # sin(π) ≈ 0
            @test result[2] ≈ -1.0 atol = 0.1  # cos(π) ≈ -1
        end
    end

    # ========================================
    # ConstantSeriesInterpolant
    # ========================================

    @testset "ConstantSeriesInterpolant" begin
        sitp_range = constant_interp(x_range, Series(ys))
        sitp_vector = constant_interp(x_vector, Series(ys))

        @testset "grid type preservation" begin
            @test sitp_range.x isa FI._CachedRange
            @test sitp_vector.x isa FI._CachedVector{Float64, Float64}
        end

        @testset "single query evaluation" begin
            result_range = sitp_range(xq_single)
            result_vector = sitp_vector(xq_single)

            @test result_range isa Vector{Float64}
            @test length(result_range) == 2
            @test result_range ≈ result_vector atol = 1.0e-14
        end

        @testset "vector query evaluation" begin
            result_range = sitp_range(xq_vector)
            result_vector = sitp_vector(xq_vector)

            # Output structure: [series1_all_queries, series2_all_queries]
            @test result_range isa Vector{Vector{Float64}}
            @test length(result_range) == 2  # n_series
            @test all(length.(result_range) .== length(xq_vector))  # n_queries each
            @test result_range ≈ result_vector atol = 1.0e-14
        end
    end

    # ========================================
    # QuadraticSeriesInterpolant
    # ========================================

    @testset "QuadraticSeriesInterpolant" begin
        sitp_range = quadratic_interp(x_range, Series(ys))
        sitp_vector = quadratic_interp(x_vector, Series(ys))

        @testset "grid type preservation" begin
            @test sitp_range.x isa FI._CachedRange
            @test sitp_vector.x isa Vector
        end

        @testset "single query evaluation" begin
            result_range = sitp_range(xq_single)
            result_vector = sitp_vector(xq_single)

            @test result_range isa Vector{Float64}
            @test length(result_range) == 2
            @test result_range ≈ result_vector atol = 1.0e-14
        end

        @testset "vector query evaluation" begin
            result_range = sitp_range(xq_vector)
            result_vector = sitp_vector(xq_vector)

            # Output structure: [series1_all_queries, series2_all_queries]
            @test result_range isa Vector{Vector{Float64}}
            @test length(result_range) == 2  # n_series
            @test all(length.(result_range) .== length(xq_vector))  # n_queries each
            @test result_range ≈ result_vector atol = 1.0e-14
        end

        @testset "output correctness (higher accuracy)" begin
            # Quadratic should be more accurate than linear
            result = sitp_range(0.5)
            @test abs(result[1]) < 0.01  # sin(π) ≈ 0 with better accuracy
            @test result[2] ≈ -1.0 atol = 0.01  # cos(π) ≈ -1
        end
    end

    # ========================================
    # Extrapolation Modes with Range Grid
    # ========================================

    @testset "extrapolation modes" begin
        for extrap in [ClampExtrap(), ExtendExtrap()]
            @testset "LinearSeriesInterpolant extrap=$extrap" begin
                sitp = linear_interp(x_range, Series(ys); extrap = extrap)

                # Query outside domain
                @test_nowarn sitp(-0.1)
                @test_nowarn sitp(1.1)
                @test_nowarn sitp([-0.1, 0.5, 1.1])
            end
        end
    end

end
