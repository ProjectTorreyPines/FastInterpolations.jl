using Test
using FastInterpolations
using FastInterpolations: _search_interval, _find_interval, _create_spacing

@testset "Search Deprecation" begin

    # ========================================
    # _find_interval Still Works (Backward Compat)
    # ========================================

    @testset "_find_interval Backward Compatibility" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Basic Usage Still Works" begin
            # Deprecated but functional - should return correct results
            idx, xL, xR = _find_interval(x, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Spacing-Aware Version Still Works" begin
            x_range = range(0.0, 1.0, 101)
            spacing = _create_spacing(x_range)

            idx, xL, xR = _find_interval(x_range, spacing, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Equivalence to _search_interval" begin
            for xi in [0.0, 0.25, 0.5, 0.75, 1.0]
                r1 = _find_interval(x, xi)
                r2 = _search_interval(x, xi)
                @test r1 == r2
            end
        end
    end

    # ========================================
    # _search_interval is Recommended
    # ========================================

    @testset "_search_interval Recommended API" begin
        x = collect(range(0.0, 1.0, 101))

        @testset "Basic Usage" begin
            idx, xL, xR = _search_interval(x, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Spacing-Aware Version" begin
            x_range = range(0.0, 1.0, 101)
            spacing = _create_spacing(x_range)

            idx, xL, xR = _search_interval(x_range, spacing, 0.5)
            @test idx == 51
            @test xL ≈ 0.50 atol=1e-12
            @test xR ≈ 0.51 atol=1e-12
        end

        @testset "Type Stability" begin
            @test @inferred(_search_interval(x, 0.5)) isa Tuple{Int,Float64,Float64}
        end
    end

    # ========================================
    # Deprecation Warning Test
    # ========================================
    # Note: Base.depwarn only emits when --depwarn=yes is passed.
    # We can't easily test this in CI, but we verify the function exists
    # and has the expected behavior.

    @testset "Deprecation Function Exists" begin
        # Verify _find_interval is callable (will emit warning if --depwarn=yes)
        @test hasmethod(_find_interval, Tuple{Vector{Float64}, Float64})
        @test hasmethod(_find_interval, Tuple{StepRangeLen{Float64}, Any, Float64})
    end

end  # @testset "Search Deprecation"
