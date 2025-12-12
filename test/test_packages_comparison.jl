using Test
using FastInterpolations
import DataInterpolations as DI
import Interpolations as Itp
using Random

const APPROX_REL_TOLERANCCE = 1e-15

@testset "Package Comparison Tests" begin

    # Test functions
    target_f(x) = sin(2π * x) + 0.5 * cos(4π * x)

    # Grid configurations
    function make_grids(n=51)
        range_grid = range(0.0, 1.0, n)
        steprangelen_grid = 0.0:0.02:1.0  # StepRangeLen
        vector_grid = collect(range(0.0, 1.0, n))

        Random.seed!(42)
        random_grid = sort(rand(n)) .* 0.98 .+ 0.01  # (0.01, 0.99) to avoid exact boundaries
        random_grid[1] = 0.0
        random_grid[end] = 1.0

        return (
            range = range_grid,
            steprangelen = steprangelen_grid,
            vector = vector_grid,
            random = random_grid
        )
    end

    grids = make_grids()

    # Uniform grids only (for Interpolations.jl which requires uniform spacing)
    uniform_grids = (
        range = grids.range,
        steprangelen = grids.steprangelen,
        vector = grids.vector
    )

    # Interior query points - includes on-grid points (0.0, 0.5, 1.0)
    xq_interior = [0.0, 0.05, 0.12, 0.23, 0.31, 0.42, 0.5, 0.58, 0.67, 0.74, 0.83, 0.91, 0.97, 1.0]

    # Extrapolation query points (for DataInterpolations.jl comparison)
    xq_extrap_left = [-0.3, -0.2, -0.1, -0.05]
    xq_extrap_right = [1.05, 1.1, 1.2, 1.3]
    xq_with_extrap = vcat(xq_extrap_left, xq_interior, xq_extrap_right)

    @testset "Linear Interpolation" begin

        @testset "vs Interpolations.jl (interior only)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations
                    result_fast = linear_interp(x, y, xq_interior)

                    # Interpolations.jl
                    itp = Itp.linear_interpolation(x, y)
                    result_interp = itp(xq_interior)

                    @test isapprox(result_fast, result_interp; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end

        @testset "vs Interpolations.jl (Range input directly)" begin
            for (grid_name, x) in pairs((range=grids.range, steprangelen=grids.steprangelen))
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations
                    result_fast = linear_interp(x, y, xq_interior)

                    # Interpolations.jl - accepts Range directly
                    itp = Itp.linear_interpolation(x, y)
                    result_interp = itp(xq_interior)

                    @test isapprox(result_fast, result_interp; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end

        @testset "vs DataInterpolations.jl (interior only)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations
                    result_fast = linear_interp(x, y, xq_interior)

                    # DataInterpolations.jl
                    itp = DI.LinearInterpolation(y, x)
                    result_data = itp(xq_interior)

                    @test isapprox(result_fast, result_data; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end

        @testset "vs DataInterpolations.jl (with ExtrapolationType.Extension)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    # Use linear function for predictable extrapolation
                    y = target_f.(x)

                    # FastInterpolations (default extrapolation=:extension)
                    result_fast = linear_interp(x, y, xq_with_extrap)

                    # DataInterpolations.jl with extrapolation
                    itp = DI.LinearInterpolation(y, x; extrapolation=DI.ExtrapolationType.Extension)
                    result_data = itp(xq_with_extrap)

                    @test isapprox(result_fast, result_data; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end
    end

    @testset "Cubic Interpolation" begin

        @testset "vs Interpolations.jl (interior only, uniform grids)" begin
            # Interpolations.jl BSpline requires uniform grids
            for (grid_name, x) in pairs(uniform_grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations
                    result_fast = cubic_interp(x, y, xq_interior)

                    # Interpolations.jl - Cubic spline with natural BC
                    x_vec = collect(x)
                    y_vec = collect(y)
                    itp = Itp.interpolate(y_vec, Itp.BSpline(Itp.Cubic(Itp.Natural(Itp.OnGrid()))))
                    # Scale to original coordinates
                    n = length(x_vec)
                    scaled_itp = Itp.scale(itp, range(x_vec[1], x_vec[end], n))
                    result_interp = scaled_itp(xq_interior)
                    # Cubic splines may have slight differences due to boundary conditions
                    # Use looser tolerance
                    @test isapprox(result_fast, result_interp; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end

        @testset "vs DataInterpolations.jl (interior only)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations
                    result_fast = cubic_interp(x, y, xq_interior)

                    # DataInterpolations.jl - CubicSpline
                    itp = DI.CubicSpline(y, x)
                    result_data = itp(xq_interior)

                    # Natural cubic spline should match closely
                    @test isapprox(result_fast, result_data; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end

        @testset "vs DataInterpolations.jl (with ExtrapolationType.Extension)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations (default extrapolation=:extension)
                    result_fast = cubic_interp(x, y, xq_with_extrap)

                    # DataInterpolations.jl with extrapolation
                    itp = DI.CubicSpline(y, x; extrapolation=DI.ExtrapolationType.Extension)
                    result_data = itp(xq_with_extrap) 

                    @test isapprox(result_fast, result_data; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end
    end

end
