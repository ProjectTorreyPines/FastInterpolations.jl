# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  LINEAR SERIES INTERPOLANT TESTS                          ║
# ║         Tests for LinearSeriesInterpolant with direct matrix storage      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Phase E.1: Tests verify LinearSeriesInterpolant works correctly using the
# shared series infrastructure from Phases A-D.
#

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "LinearSeriesInterpolant" begin

    # ========================================
    # Constructor Tests
    # ========================================

    @testset "construction" begin
        x = collect(0.0:0.1:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        ys = [y1, y2]

        @testset "from vector of vectors" begin
            sitp = linear_interp(x, ys)
            @test sitp isa FI.LinearSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "from matrix" begin
            Y = hcat(y1, y2)
            sitp = linear_interp(x, Y)
            @test sitp isa FI.LinearSeriesInterpolant{Float64}
            @test FI.n_series(sitp) == 2
        end

        @testset "validates input dimensions" begin
            bad_y = [y1[1:end-1], y2]  # Wrong length
            @test_throws Exception linear_interp(x, bad_y)
        end

        @testset "single series" begin
            sitp = linear_interp(x, [y1])
            @test FI.n_series(sitp) == 1
        end
    end

    # ========================================
    # Trait Implementation Tests
    # ========================================

    @testset "trait implementations" begin
        x = collect(0.0:0.1:1.0)
        sitp = linear_interp(x, [sin.(2π .* x), cos.(2π .* x)])

        @test FI.n_series(sitp) == 2
        @test FI._get_grid(sitp) ≈ x
        @test FI._get_extrap(sitp) isa FI.ExtrapVal
        @test FI._should_wrap(sitp) == false
        @test FI._method_kind(typeof(sitp)) === Val(:linear)
    end

    # ========================================
    # Evaluation Accuracy Tests
    # ========================================

    @testset "evaluation accuracy" begin
        # Linear interpolation should be exact for linear data
        x = [0.0, 1.0, 2.0, 3.0]
        y1 = [0.0, 2.0, 4.0, 6.0]  # y = 2x (linear)
        y2 = [1.0, 1.0, 1.0, 1.0]  # y = 1 (constant)
        sitp = linear_interp(x, [y1, y2])

        @testset "exact at grid points" begin
            for i in eachindex(x)
                result = sitp(x[i])
                @test result[1] ≈ y1[i]
                @test result[2] ≈ y2[i]
            end
        end

        @testset "correct at midpoints" begin
            result = sitp(0.5)
            @test result[1] ≈ 1.0  # Linear interp of 0, 2
            @test result[2] ≈ 1.0  # Constant
        end

        @testset "first derivative" begin
            deriv = sitp(0.5; deriv=1)
            @test deriv[1] ≈ 2.0  # Slope of y = 2x
            @test deriv[2] ≈ 0.0 atol=1e-10  # Slope of constant
        end
    end

    # ========================================
    # Zero-Allocation Tests
    # ========================================

    @testset "zero allocation" begin
        x = collect(0.0:0.1:1.0)
        sitp = linear_interp(x, [sin.(2π .* x), cos.(2π .* x)])

        @testset "scalar in-place" begin
            output = zeros(2)
            sitp(output, 0.5)  # Warmup
            sitp(output, 0.5)  # Warmup
            allocs = @allocated sitp(output, 0.5)
            @test allocs <= ALLOC_THRESHOLD
        end

        @testset "vector in-place" begin
            xq = collect(0.0:0.05:1.0)
            outputs = [zeros(length(xq)) for _ in 1:2]
            sitp(outputs, xq)  # Warmup
            sitp(outputs, xq)  # Warmup
            allocs = @allocated sitp(outputs, xq)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Extrapolation Tests
    # ========================================

    @testset "extrapolation modes" begin
        x = collect(0.0:0.1:1.0)
        ys = [sin.(2π .* x)]

        @testset "extrap=:none throws" begin
            sitp = linear_interp(x, ys; extrap=:none)
            @test_throws DomainError sitp(-0.1)
            @test_throws DomainError sitp(1.1)
        end

        @testset "extrap=:constant returns boundary" begin
            sitp = linear_interp(x, ys; extrap=:constant)
            @test sitp(-0.1)[1] ≈ sin(0.0) atol=1e-6
            @test sitp(1.1)[1] ≈ sin(2π) atol=1e-6
        end

        @testset "extrap=:extension extrapolates" begin
            sitp = linear_interp(x, ys; extrap=:extension)
            @test sitp(-0.1) isa Vector{Float64}
            @test sitp(1.1) isa Vector{Float64}
        end
    end

    # ========================================
    # Type Stability
    # ========================================

    @testset "type stability" begin
        x32 = Float32.(collect(0.0:0.1:1.0))
        ys32 = [Float32.(sin.(2π .* Float64.(x32)))]

        x64 = collect(0.0:0.1:1.0)
        ys64 = [sin.(2π .* x64)]

        sitp32 = linear_interp(x32, ys32)
        sitp64 = linear_interp(x64, ys64)

        @test sitp32(0.5f0) isa Vector{Float32}
        @test sitp64(0.5) isa Vector{Float64}

        @test_nowarn @inferred sitp64(0.5)
    end

    # ========================================
    # Callable Signatures
    # ========================================

    @testset "callable signatures" begin
        x = collect(0.0:0.1:1.0)
        sitp = linear_interp(x, [sin.(2π .* x), cos.(2π .* x)])

        @testset "scalar out-of-place" begin
            result = sitp(0.5)
            @test length(result) == 2
            @test result isa Vector{Float64}
        end

        @testset "scalar in-place" begin
            output = zeros(2)
            result = sitp(output, 0.5)
            @test result === output
        end

        @testset "vector out-of-place" begin
            xq = [0.25, 0.5, 0.75]
            results = sitp(xq)
            @test length(results) == 2  # n_series
            @test all(r -> length(r) == 3, results)
        end

        @testset "vector in-place" begin
            xq = [0.25, 0.5, 0.75]
            outputs = [zeros(3), zeros(3)]
            result = sitp(outputs, xq)
            @test result === outputs
        end
    end

    # ========================================
    # Consistency with Single Interpolant
    # ========================================

    @testset "consistency with existing implementation" begin
        x = collect(0.0:0.01:1.0)
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        ys = [y1, y2]

        # Both should give same results
        sitp = linear_interp(x, ys)

        # Compare at multiple points
        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        for xqi in xq
            result = sitp(xqi)
            # Linear interpolation of sin(2πx) at xqi
            idx, xL, xR = FI._find_interval(x, xqi)
            t = (xqi - x[idx]) / (x[idx+1] - x[idx])
            expected1 = y1[idx] * (1 - t) + y1[idx+1] * t
            expected2 = y2[idx] * (1 - t) + y2[idx+1] * t
            @test result[1] ≈ expected1 atol=1e-10
            @test result[2] ≈ expected2 atol=1e-10
        end
    end

end  # testset "LinearSeriesInterpolant"
