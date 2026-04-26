@testitem "Package Comparison Tests" begin
    import DataInterpolations as DI
    import Dierckx
    import Interpolations as Itp
    using Random

    # 1e-14 instead of 1e-15: accounts for FMA (fused multiply-add) differences
    # between Julia versions (e.g., 1.10 LTS vs latest) that cause last-bit variations.
    # Note: FMA is numerically more accurate (single rounding vs two), but produces
    # slightly different results than separate multiply-then-add operations.
    const APPROX_REL_TOLERANCE = 1.0e-14


    # Test functions
    target_f(x) = sin(2π * x) + 0.5 * cos(4π * x)

    # Grid configurations
    function make_grids(n = 51)
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
            random = random_grid,
        )
    end

    grids = make_grids()

    # Uniform grids only (for Interpolations.jl which requires uniform spacing)
    uniform_grids = (
        range = grids.range,
        steprangelen = grids.steprangelen,
        vector = grids.vector,
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

                    @test isapprox(result_fast, result_interp; rtol = APPROX_REL_TOLERANCE)
                end
            end
        end

        @testset "vs Interpolations.jl (Range input directly)" begin
            for (grid_name, x) in pairs((range = grids.range, steprangelen = grids.steprangelen))
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations
                    result_fast = linear_interp(x, y, xq_interior)

                    # Interpolations.jl - accepts Range directly
                    itp = Itp.linear_interpolation(x, y)
                    result_interp = itp(xq_interior)

                    @test isapprox(result_fast, result_interp; rtol = APPROX_REL_TOLERANCE)
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

                    @test isapprox(result_fast, result_data; rtol = APPROX_REL_TOLERANCE)

                end
            end
        end

        @testset "vs DataInterpolations.jl (with ExtrapolationType.Extension)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    # Use linear function for predictable extrapolation
                    y = target_f.(x)

                    # FastInterpolations (explicit extrap=ExtendExtrap())
                    result_fast = linear_interp(x, y, xq_with_extrap; extrap = ExtendExtrap())

                    # DataInterpolations.jl with extrapolation
                    itp = DI.LinearInterpolation(y, x; extrapolation = DI.ExtrapolationType.Extension)
                    result_data = itp(xq_with_extrap)

                    @test isapprox(result_fast, result_data; rtol = APPROX_REL_TOLERANCE)
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

                    # FastInterpolations — explicit ZeroCurvBC to match natural BC
                    result_fast = cubic_interp(x, y, xq_interior; bc = ZeroCurvBC())

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
                    @test isapprox(result_fast, result_interp; rtol = APPROX_REL_TOLERANCE)
                end
            end
        end

        @testset "vs DataInterpolations.jl (interior only)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations — explicit ZeroCurvBC to match natural BC
                    result_fast = cubic_interp(x, y, xq_interior; bc = ZeroCurvBC())

                    # DataInterpolations.jl - CubicSpline (natural BC)
                    itp = DI.CubicSpline(y, x)
                    result_data = itp(xq_interior)

                    # ZeroCurv cubic spline should match closely
                    @test isapprox(result_fast, result_data; rtol = APPROX_REL_TOLERANCE)
                end
            end
        end

        @testset "vs DataInterpolations.jl (with ExtrapolationType.Extension)" begin
            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations — explicit ZeroCurvBC to match natural BC
                    result_fast = cubic_interp(x, y, xq_with_extrap; bc = ZeroCurvBC(), extrap = ExtendExtrap())

                    # DataInterpolations.jl with extrapolation
                    itp = DI.CubicSpline(y, x; extrapolation = DI.ExtrapolationType.Extension)
                    result_data = itp(xq_with_extrap)

                    @test isapprox(result_fast, result_data; rtol = APPROX_REL_TOLERANCE)
                end
            end
        end

        # Dierckx.jl wraps FITPACK which uses not-a-knot BC (s=0.0):
        #   - Not-a-knot removes the 2nd and (n-1)th interior knots, forcing the first
        #     two intervals (and last two) to share a single cubic polynomial.
        #   - CubicFit (PolyFit{3}) fits a cubic through the first/last 4 points and
        #     uses its derivative at the boundary as a clamped BC.
        #   - These are NOT mathematically equivalent. They converge in the interior
        #     (BC effects decay exponentially) but differ near boundaries by O(1e-4).
        #   - Extrapolation diverges substantially (~6%) so is not compared.
        @testset "vs Dierckx.jl (interior only, general function)" begin
            # Interior query points excluding near-boundary regions
            xq_deep_interior = [xi for xi in xq_interior if 0.1 ≤ xi ≤ 0.9]

            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = target_f.(x)

                    # FastInterpolations — default CubicFit BC
                    result_fast = cubic_interp(x, y, xq_deep_interior)

                    # Dierckx.jl — FITPACK not-a-knot cubic spline
                    x_vec = collect(x)
                    y_vec = collect(y)
                    itp = Dierckx.Spline1D(x_vec, y_vec; k = 3, s = 0.0)
                    result_dierckx = [itp(xi) for xi in xq_deep_interior]

                    # Different BCs → interior values converge but residual BC influence
                    # decays exponentially from boundaries (~1e-9 at 10% from edge)
                    @test isapprox(result_fast, result_dierckx; rtol = 1.0e-8)
                end
            end
        end

        # Both CubicFit and not-a-knot reproduce a global cubic polynomial exactly:
        #   - CubicFit: 4-point fit on cubic data recovers the exact polynomial → exact slope
        #   - Not-a-knot: first/last two intervals share one cubic → trivially satisfied
        #   - Natural BC (ZeroCurvBC) does NOT satisfy this since f''(boundary) ≠ 0 in general
        # Both CubicFit and not-a-knot reproduce a global cubic polynomial exactly:
        #   - CubicFit: 4-point fit on cubic data recovers the exact polynomial → exact slope
        #   - Not-a-knot: first/last two intervals share one cubic → trivially satisfied
        #   - Natural BC (ZeroCurvBC) does NOT satisfy this since f''(boundary) ≠ 0 in general
        # Interior + boundary points match at machine precision; extrapolation omitted
        # because different FP arithmetic paths (FITPACK Fortran vs Julia Horner) amplify
        # rounding errors with distance from the data range.
        @testset "vs Dierckx.jl (cubic polynomial — exact reproduction)" begin
            cubic_poly(x) = 2.0x^3 - 1.5x^2 + 0.7x - 0.3

            for (grid_name, x) in pairs(grids)
                @testset "Grid: $grid_name" begin
                    y = cubic_poly.(x)

                    # FastInterpolations — default CubicFit BC
                    result_fast = cubic_interp(x, y, xq_interior)

                    # Dierckx.jl — FITPACK not-a-knot
                    x_vec = collect(x)
                    y_vec = collect(y)
                    itp = Dierckx.Spline1D(x_vec, y_vec; k = 3, s = 0.0)
                    result_dierckx = [itp(xi) for xi in xq_interior]

                    # Both reproduce cubic polynomials exactly → machine precision match
                    @test isapprox(result_fast, result_dierckx; rtol = APPROX_REL_TOLERANCE)
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
    # - Cubic: DataInterpolations.jl CubicSpline uses tridiagonal system with Zero-Curvature BC,
    #          same algorithm as FastInterpolations
    # =========================================================================

end
