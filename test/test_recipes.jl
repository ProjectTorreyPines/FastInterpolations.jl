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
        for extrap in [:none, :constant, :extension, :wrap]
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
