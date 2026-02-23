# ========================================
# Plot Recipe Tests
# ========================================
# Tests recipe functionality using RecipesBase.apply_recipe
# without loading the full Plots.jl package.

using Test
using FastInterpolations
using RecipesBase

@testset "Plot Recipes" begin
    # Test data
    x = collect(range(0.0, 1.0, 11))
    y = sin.(2π .* x)
    y_matrix = [sin.(2π .* x) cos.(2π .* x)]

    # ========================================
    # Single-Series Interpolants
    # ========================================
    @testset "Single-Series Interpolants" begin
        @testset "LinearInterpolant" begin
            itp = linear_interp(x, y)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
            @test all(r -> r isa RecipesBase.RecipeData, recipes)
        end

        @testset "ConstantInterpolant" begin
            itp = constant_interp(x, y)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end

        @testset "QuadraticInterpolant" begin
            itp = quadratic_interp(x, y)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end

        @testset "CubicInterpolant" begin
            itp = cubic_interp(x, y)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "Extrapolation Modes" begin
        for extrap in [NoExtrap(), ConstExtrap(), ExtendExtrap(), WrapExtrap()]
            @testset "extrap=:$extrap" begin
                itp = cubic_interp(x, y; extrap=extrap)
                recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
                @test !isempty(recipes)
            end
        end
    end

    # ========================================
    # Recipe Options
    # ========================================
    @testset "Recipe Options" begin
        itp = cubic_interp(x, y)

        @testset "show_data=false" begin
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_data => false), itp
            )
            @test !isempty(recipes)
        end

        @testset "show_bounds=false" begin
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_bounds => false), itp
            )
            @test !isempty(recipes)
        end

        @testset "show_outside=false" begin
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_outside => false), itp
            )
            @test !isempty(recipes)
        end

        @testset "custom samples" begin
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:samples => 100), itp
            )
            @test !isempty(recipes)
        end

        @testset "custom domain_margin" begin
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:domain_margin => 0.5), itp
            )
            @test !isempty(recipes)
        end

        @testset "all options disabled" begin
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(
                    :show_data => false,
                    :show_bounds => false,
                    :show_outside => false
                ), itp
            )
            @test !isempty(recipes)
        end
    end

    # ========================================
    # Multi-Series Interpolants
    # ========================================
    @testset "Multi-Series Interpolants" begin
        @testset "LinearSeriesInterpolant" begin
            sitp = linear_interp(x, y_matrix)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sitp)
            @test !isempty(recipes)
        end

        @testset "ConstantSeriesInterpolant" begin
            sitp = constant_interp(x, y_matrix)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sitp)
            @test !isempty(recipes)
        end

        @testset "QuadraticSeriesInterpolant" begin
            sitp = quadratic_interp(x, y_matrix)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sitp)
            @test !isempty(recipes)
        end

        @testset "CubicSeriesInterpolant" begin
            sitp = cubic_interp(x, y_matrix)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), sitp)
            @test !isempty(recipes)
        end

        @testset "series_idx options" begin
            sitp = cubic_interp(x, y_matrix)

            # First series only
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:series_idx => :first), sitp
            )
            @test !isempty(recipes)

            # All series (default)
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:series_idx => :all), sitp
            )
            @test !isempty(recipes)

            # Specific series by index
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:series_idx => 2), sitp
            )
            @test !isempty(recipes)

            # Multiple specific series
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:series_idx => [1, 2]), sitp
            )
            @test !isempty(recipes)
        end
    end

    # ========================================
    # DerivativeView
    # ========================================
    @testset "DerivativeView" begin
        itp = cubic_interp(x, y)

        @testset "deriv1" begin
            d1 = deriv1(itp)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), d1)
            @test !isempty(recipes)
        end

        @testset "deriv2" begin
            d2 = deriv2(itp)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), d2)
            @test !isempty(recipes)
        end

        @testset "deriv3" begin
            d3 = deriv3(itp)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), d3)
            @test !isempty(recipes)
        end

        @testset "deriv with options" begin
            d1 = deriv1(itp)
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_bounds => false, :samples => 100), d1
            )
            @test !isempty(recipes)
        end
    end

    # ========================================
    # Float32 Support
    # ========================================
    @testset "Float32 Support" begin
        x32 = Float32.(x)
        y32 = Float32.(y)

        itp = cubic_interp(x32, y32)
        recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
        @test !isempty(recipes)

        # With custom margin (should convert to Float32)
        recipes = RecipesBase.apply_recipe(
            Dict{Symbol,Any}(:domain_margin => 0.5), itp
        )
        @test !isempty(recipes)
    end

    # ========================================
    # N-Dimensional Interpolants (2D)
    # ========================================
    @testset "N-Dimensional Interpolants (2D)" begin
        # Setup 2D grid
        x1 = range(0.0, 1.0, 6)
        x2 = range(0.0, 1.0, 5)
        data_2d = [sin(2π * xi) * cos(2π * yj) for xi in x1, yj in x2]

        @testset "LinearInterpolantND" begin
            itp = linear_interp((x1, x2), data_2d)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
            @test all(r -> r isa RecipesBase.RecipeData, recipes)
        end

        @testset "ConstantInterpolantND" begin
            itp = constant_interp((x1, x2), data_2d)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
            @test all(r -> r isa RecipesBase.RecipeData, recipes)
        end

        @testset "CubicInterpolantND" begin
            itp = cubic_interp((x1, x2), data_2d)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
            @test all(r -> r isa RecipesBase.RecipeData, recipes)
        end

        @testset "ND recipe options" begin
            itp = linear_interp((x1, x2), data_2d)

            # Test show_nodes option
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_nodes => true), itp
            )
            @test !isempty(recipes)

            # Test show_gridlines option
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_gridlines => false), itp
            )
            @test !isempty(recipes)

            # Test custom resolution
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:resolution => (20, 20)), itp
            )
            @test !isempty(recipes)

            # Test equal_aspect
            recipes = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:equal_aspect => true), itp
            )
            @test !isempty(recipes)
        end

        @testset "ND extrapolation extension" begin
            # extrap on both axes — should produce boundary rectangle series
            itp_ext = linear_interp((x1, x2), data_2d; extrap=ExtendExtrap())
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp_ext)
            @test !isempty(recipes)

            # Find boundary series: path with 5-point closed rectangle
            boundary_series = filter(recipes) do r
                d = r.plotattributes
                get(d, :seriestype, nothing) === :path &&
                    get(d, :label, nothing) == "domain"
            end
            @test length(boundary_series) == 1

            # Boundary rectangle should have 5 points (closed)
            bx, by = boundary_series[1].args
            @test length(bx) == 5
            @test length(by) == 5
            @test bx[1] == bx[end]  # closed path
            @test by[1] == by[end]

            # Heatmap should extend beyond domain
            heatmap_series = filter(recipes) do r
                get(r.plotattributes, :seriestype, nothing) === :heatmap
            end
            @test length(heatmap_series) == 1
            hm_x = heatmap_series[1].args[1]
            @test first(hm_x) < first(x1)  # extended left
            @test last(hm_x) > last(x1)    # extended right

            # extrap=NoExtrap() — should NOT produce boundary series
            itp_none = linear_interp((x1, x2), data_2d; extrap=NoExtrap())
            recipes_none = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp_none)
            boundary_none = filter(recipes_none) do r
                get(r.plotattributes, :seriestype, nothing) === :path &&
                    get(r.plotattributes, :label, nothing) == "domain"
            end
            @test isempty(boundary_none)

            # Heatmap should stay within domain when extrap=NoExtrap()
            hm_none = filter(recipes_none) do r
                get(r.plotattributes, :seriestype, nothing) === :heatmap
            end
            hm_x_none = hm_none[1].args[1]
            @test first(hm_x_none) ≈ first(x1)
            @test last(hm_x_none) ≈ last(x1)

            # show_boundary=false — should suppress boundary even with extrap
            recipes_no_bd = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:show_boundary => false), itp_ext
            )
            boundary_off = filter(recipes_no_bd) do r
                get(r.plotattributes, :seriestype, nothing) === :path &&
                    get(r.plotattributes, :label, nothing) == "domain"
            end
            @test isempty(boundary_off)

            # Custom domain_margin
            recipes_margin = RecipesBase.apply_recipe(
                Dict{Symbol,Any}(:domain_margin => 0.5), itp_ext
            )
            hm_margin = filter(recipes_margin) do r
                get(r.plotattributes, :seriestype, nothing) === :heatmap
            end
            hm_x_m = hm_margin[1].args[1]
            @test first(hm_x_m) ≈ first(x1) - 0.5
            @test last(hm_x_m) ≈ last(x1) + 0.5
        end

        @testset "ND extrap with CubicInterpolantND" begin
            itp_ext = cubic_interp((x1, x2), data_2d; extrap=ConstExtrap())
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp_ext)
            @test !isempty(recipes)
            boundary_series = filter(recipes) do r
                get(r.plotattributes, :seriestype, nothing) === :path &&
                    get(r.plotattributes, :label, nothing) == "domain"
            end
            @test length(boundary_series) == 1
        end

        @testset "Mixed grid types" begin
            # Vector and Range combination
            x1_vec = collect(range(0.0, 1.0, 6))
            x2_range = range(0.0, 1.0, 5)
            data_mixed = [xi + yj for xi in x1_vec, yj in x2_range]

            itp = linear_interp((x1_vec, x2_range), data_mixed)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end
    end

    # ========================================
    # Edge Cases
    # ========================================
    @testset "Edge Cases" begin
        @testset "Minimal grid (2 points)" begin
            x_min = [0.0, 1.0]
            y_min = [0.0, 1.0]
            itp = linear_interp(x_min, y_min)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end

        @testset "Range input (non-Vector)" begin
            x_range = range(0.0, 1.0, 11)
            y_range = sin.(2π .* collect(x_range))
            itp = linear_interp(x_range, y_range)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end

        @testset "Negative values" begin
            y_neg = y .- 0.5  # Values cross zero
            itp = cubic_interp(x, y_neg)
            recipes = RecipesBase.apply_recipe(Dict{Symbol,Any}(), itp)
            @test !isempty(recipes)
        end
    end
end
