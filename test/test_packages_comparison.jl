using Test
using FastInterpolations
import DataInterpolations as DI
import Interpolations as Itp
using Random

# 1e-14 instead of 1e-15: accounts for FMA (fused multiply-add) differences
# between Julia versions (e.g., 1.10 LTS vs latest) that cause last-bit variations.
# Note: FMA is numerically more accurate (single rounding vs two), but produces
# slightly different results than separate multiply-then-add operations.
const APPROX_REL_TOLERANCCE = 1e-14

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

                    # FastInterpolations (explicit extrap=ExtendExtrap())
                    result_fast = linear_interp(x, y, xq_with_extrap; extrap=ExtendExtrap())

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

                    # FastInterpolations (explicit extrap=ExtendExtrap())
                    result_fast = cubic_interp(x, y, xq_with_extrap; extrap=ExtendExtrap())

                    # DataInterpolations.jl with extrapolation
                    itp = DI.CubicSpline(y, x; extrapolation=DI.ExtrapolationType.Extension)
                    result_data = itp(xq_with_extrap)

                    @test isapprox(result_fast, result_data; rtol=APPROX_REL_TOLERANCCE)
                end
            end
        end
    end

    # =========================================================================
    # Quadratic Interpolation - Comparison Skipped
    # =========================================================================
    # Direct comparison with DataInterpolations.jl QuadraticSpline is not included
    # because the two packages use fundamentally different algorithms:
    #
    # - FastInterpolations: Piecewise polynomial with explicit BC (recurrence-based, O(n))
    #   User specifies boundary condition directly (e.g., Left(Deriv2(0.0)))
    #
    # - DataInterpolations.jl: B-spline basis with implicit BC (clamped knot vector)
    #   BC is implicitly determined by the B-spline construction, not user-specified
    #
    # Both solve the same mathematical problem (quadratic spline has 1 DOF requiring 1 BC),
    # but with different BC choices. When BCs match exactly, results are identical to
    # machine precision. However, matching DataInterpolations' implicit BC would require
    # reverse-engineering their B-spline construction, which is not practical.
    #
    # Note: Linear and Cubic comparisons work because:
    # - Linear: No BC needed (unique solution)
    # - Cubic: DataInterpolations.jl CubicSpline uses tridiagonal system with Natural BC,
    #          same algorithm as FastInterpolations
    # =========================================================================

end
