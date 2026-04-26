@testitem "Grid Spacing Types" begin

    @testset "ScalarSpacing Construction" begin
        # Test 1.1: ScalarSpacing type exists and constructs correctly
        @testset "Float64" begin
            h = 0.1
            inv_h = 10.0
            s = FastInterpolations.ScalarSpacing(h, inv_h)

            @test s isa FastInterpolations.AbstractGridSpacing{Float64}
            @test s isa FastInterpolations.ScalarSpacing{Float64}
            @test s.h == h
            @test s.inv_h == inv_h
        end

        @testset "Float32" begin
            h = 0.1f0
            inv_h = 10.0f0
            s = FastInterpolations.ScalarSpacing(h, inv_h)

            @test s isa FastInterpolations.AbstractGridSpacing{Float32}
            @test s isa FastInterpolations.ScalarSpacing{Float32}
            @test s.h == h
            @test s.inv_h == inv_h
        end

        @testset "Type parameter inference" begin
            # Mixed types should promote
            s = FastInterpolations.ScalarSpacing(0.1, 10.0f0)
            @test s isa FastInterpolations.ScalarSpacing{Float64}
        end
    end

    @testset "VectorSpacing Construction" begin
        # Test 1.2: VectorSpacing type exists and constructs correctly
        @testset "Float64" begin
            h = [0.1, 0.2, 0.15]
            inv_h = [10.0, 5.0, 1.0 / 0.15]
            s = FastInterpolations.VectorSpacing(h, inv_h)

            @test s isa FastInterpolations.AbstractGridSpacing{Float64}
            @test s isa FastInterpolations.VectorSpacing{Float64}
            @test s.h == h
            @test s.inv_h == inv_h
            @test length(s.h) == 3
            @test length(s.inv_h) == 3
        end

        @testset "Float32" begin
            h = Float32[0.1, 0.2, 0.15]
            inv_h = Float32[10.0, 5.0, 1.0 / 0.15]
            s = FastInterpolations.VectorSpacing(h, inv_h)

            @test s isa FastInterpolations.AbstractGridSpacing{Float32}
            @test s isa FastInterpolations.VectorSpacing{Float32}
        end
    end

    @testset "Accessor Functions _get_h / _get_inv_h" begin
        # Test 1.3: Accessor functions work correctly

        @testset "ScalarSpacing accessors (index ignored)" begin
            s = FastInterpolations.ScalarSpacing(0.1, 10.0)

            # Index is ignored for ScalarSpacing - always returns the same value
            @test FastInterpolations._get_h(s, 1) == 0.1
            @test FastInterpolations._get_h(s, 5) == 0.1
            @test FastInterpolations._get_h(s, 100) == 0.1

            @test FastInterpolations._get_inv_h(s, 1) == 10.0
            @test FastInterpolations._get_inv_h(s, 5) == 10.0
            @test FastInterpolations._get_inv_h(s, 100) == 10.0
        end

        @testset "VectorSpacing accessors (indexed)" begin
            h = [0.1, 0.2, 0.3]
            inv_h = [10.0, 5.0, 1.0 / 0.3]
            s = FastInterpolations.VectorSpacing(h, inv_h)

            # VectorSpacing returns the indexed value
            @test FastInterpolations._get_h(s, 1) == 0.1
            @test FastInterpolations._get_h(s, 2) == 0.2
            @test FastInterpolations._get_h(s, 3) == 0.3

            @test FastInterpolations._get_inv_h(s, 1) == 10.0
            @test FastInterpolations._get_inv_h(s, 2) == 5.0
            @test FastInterpolations._get_inv_h(s, 3) == inv_h[3]
        end

        @testset "Type stability" begin
            s_scalar = FastInterpolations.ScalarSpacing(0.1, 10.0)
            s_vector = FastInterpolations.VectorSpacing([0.1, 0.2], [10.0, 5.0])

            # Type inference should work
            @test @inferred(FastInterpolations._get_h(s_scalar, 1)) == 0.1
            @test @inferred(FastInterpolations._get_inv_h(s_scalar, 1)) == 10.0
            @test @inferred(FastInterpolations._get_h(s_vector, 1)) == 0.1
            @test @inferred(FastInterpolations._get_inv_h(s_vector, 1)) == 10.0
        end
    end

    @testset "_create_spacing Dispatch" begin
        # Test 1.4: _create_spacing correctly dispatches based on input type

        @testset "StepRangeLen -> ScalarSpacing" begin
            x = range(0.0, 1.0, 101)  # StepRangeLen
            spacing = FastInterpolations._create_spacing(x)

            @test spacing isa FastInterpolations.ScalarSpacing{Float64}
            @test spacing.h ≈ step(x)
            @test spacing.inv_h ≈ 1.0 / step(x)
        end

        @testset "LinRange -> ScalarSpacing" begin
            x = LinRange(0.0, 1.0, 101)
            spacing = FastInterpolations._create_spacing(x)

            @test spacing isa FastInterpolations.ScalarSpacing{Float64}
            # LinRange step is computed, verify it's correct
            expected_h = (last(x) - first(x)) / (length(x) - 1)
            @test spacing.h ≈ expected_h
            @test spacing.inv_h ≈ 1.0 / expected_h
        end

        @testset "Vector -> VectorSpacing" begin
            x = [0.0, 0.3, 0.7, 1.0]
            spacing = FastInterpolations._create_spacing(x)

            @test spacing isa FastInterpolations.VectorSpacing{Float64}
            @test length(spacing.h) == length(x) - 1
            @test spacing.h[1] ≈ 0.3
            @test spacing.h[2] ≈ 0.4
            @test spacing.h[3] ≈ 0.3

            # Check inv_h consistency
            for i in 1:length(spacing.h)
                @test spacing.inv_h[i] ≈ 1.0 / spacing.h[i]
            end
        end

        @testset "Float32 type preservation" begin
            x_range = range(0.0f0, 1.0f0, 101)
            spacing_range = FastInterpolations._create_spacing(x_range)
            @test spacing_range isa FastInterpolations.ScalarSpacing{Float32}

            x_vec = Float32[0.0, 0.5, 1.0]
            spacing_vec = FastInterpolations._create_spacing(x_vec)
            @test spacing_vec isa FastInterpolations.VectorSpacing{Float32}
        end

        @testset "Collected range -> VectorSpacing" begin
            # When a range is collected to Vector, it should use VectorSpacing
            x = collect(range(0.0, 1.0, 101))
            spacing = FastInterpolations._create_spacing(x)

            @test spacing isa FastInterpolations.VectorSpacing{Float64}
        end
    end

    @testset "Memory Efficiency" begin
        # ScalarSpacing should be O(1) memory
        s_scalar = FastInterpolations.ScalarSpacing(0.1, 10.0)
        @test sizeof(s_scalar) == 2 * sizeof(Float64)  # Just two scalars

        # VectorSpacing memory depends on vector size
        h = zeros(1000)
        inv_h = zeros(1000)
        s_vector = FastInterpolations.VectorSpacing(h, inv_h)
        # VectorSpacing struct itself is small (just pointers)
        @test sizeof(s_vector) < 100  # Struct overhead is small
    end

    @testset "_search_interval with Spacing (Phase 5)" begin
        # Test spacing-aware interval search (fdiv → fmul optimization)

        @testset "ScalarSpacing: boundary regression" begin
            x = range(0.0, 1.0, 1000)
            spacing = FastInterpolations._create_spacing(x)
            n = length(x)

            # Boundary tests: first point → idx=1
            idx, _, xL, xR = FastInterpolations._search_interval(x, spacing, first(x))
            @test idx == 1
            @test xL ≈ first(x)
            @test xR ≈ first(x) + step(x)

            # Boundary tests: last point → idx=n-1
            idx, _, xL, xR = FastInterpolations._search_interval(x, spacing, last(x))
            @test idx == n - 1
            @test xL ≈ last(x) - step(x)
            @test xR ≈ last(x)

            # Interior point
            idx, _, xL, xR = FastInterpolations._search_interval(x, spacing, 0.5)
            @test 1 <= idx <= n - 1
            @test xL <= 0.5 <= xR
        end

        @testset "ScalarSpacing vs base: exact grid points" begin
            x = range(0.0, 1.0, 101)
            spacing = FastInterpolations._create_spacing(x)

            # Test exact grid points (k * step)
            for k in 0:10:100
                xi = first(x) + k * step(x)
                idx_base, _, xL_base, xR_base = FastInterpolations._search_interval(x, xi)
                idx_new, _, xL_new, xR_new = FastInterpolations._search_interval(x, spacing, xi)

                @test idx_base == idx_new
                @test xL_base ≈ xL_new atol = 1.0e-14
                @test xR_base ≈ xR_new atol = 1.0e-14
            end
        end

        @testset "ScalarSpacing vs base: random points" begin
            x = range(0.0, 10.0, 1000)
            spacing = FastInterpolations._create_spacing(x)

            # Compare spacing-aware vs base for many random points
            for _ in 1:100
                xi = first(x) + rand() * (last(x) - first(x))
                idx_base, _, xL_base, xR_base = FastInterpolations._search_interval(x, xi)
                idx_new, _, xL_new, xR_new = FastInterpolations._search_interval(x, spacing, xi)

                @test idx_base == idx_new
                @test xL_base ≈ xL_new atol = 1.0e-12
                @test xR_base ≈ xR_new atol = 1.0e-12
            end
        end

        @testset "VectorSpacing: delegates to binary search" begin
            x = [0.0, 0.3, 0.7, 1.0]
            spacing = FastInterpolations._create_spacing(x)

            # VectorSpacing should delegate to base _search_interval
            idx, _, xL, xR = FastInterpolations._search_interval(x, spacing, 0.5)
            idx_base, _, xL_base, xR_base = FastInterpolations._search_interval(x, 0.5)

            @test idx == idx_base
            @test xL == xL_base
            @test xR == xR_base

            # Verify correct interval [0.3, 0.7]
            @test idx == 2
            @test xL == 0.3
            @test xR == 0.7
        end

        @testset "Float32 consistency" begin
            x = range(0.0f0, 1.0f0, 100)
            spacing = FastInterpolations._create_spacing(x)

            xi = 0.5f0
            idx, _, xL, xR = FastInterpolations._search_interval(x, spacing, xi)

            @test idx isa Int
            @test xL isa Float32
            @test xR isa Float32
            @test xL <= xi <= xR
        end

        @testset "Type stability" begin
            x_range = range(0.0, 1.0, 100)
            spacing_scalar = FastInterpolations._create_spacing(x_range)

            x_vec = collect(x_range)
            spacing_vector = FastInterpolations._create_spacing(x_vec)

            # Both should be type-stable
            @test @inferred(FastInterpolations._search_interval(x_range, spacing_scalar, 0.5)) isa Tuple{Int, Int, Float64, Float64}
            @test @inferred(FastInterpolations._search_interval(x_vec, spacing_vector, 0.5)) isa Tuple{Int, Int, Float64, Float64}
        end
    end

end
