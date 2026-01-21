@testset "Show Methods" begin
    # Test data
    x = range(0.0, 1.0, 101)
    y = sin.(2π .* collect(x))
    y_matrix = [sin.(2π .* collect(x)) cos.(2π .* collect(x)) exp.(-collect(x))]

    @testset "LinearInterpolant show" begin
        itp = linear_interp(x, y)

        # Compact show
        compact_str = sprint(show, itp)
        @test occursin("LinearInterpolant", compact_str)
        @test occursin("Float64", compact_str)
        @test occursin("101 pts", compact_str)

        # Verbose show (Range grid → no Search row)
        verbose_str = sprint(show, MIME("text/plain"), itp)
        @test occursin("LinearInterpolant", verbose_str)
        @test occursin("Grid:", verbose_str)
        @test occursin("Range", verbose_str)
        @test occursin("101 points", verbose_str)
        @test occursin("Extrap:", verbose_str)
        @test !occursin("Search:", verbose_str)  # Range → no Search
    end

    @testset "ConstantInterpolant show" begin
        itp = constant_interp(x, y)

        # Compact show
        compact_str = sprint(show, itp)
        @test occursin("ConstantInterpolant", compact_str)
        @test occursin("101 pts", compact_str)

        # Verbose show (Range grid → no Search row)
        verbose_str = sprint(show, MIME("text/plain"), itp)
        @test occursin("ConstantInterpolant", verbose_str)
        @test occursin("Grid:", verbose_str)
        @test occursin("Extrap:", verbose_str)
        @test occursin("Side:", verbose_str)
        @test !occursin("Search:", verbose_str)  # Range → no Search
    end

    @testset "QuadraticInterpolant show" begin
        itp = quadratic_interp(x, y)

        # Compact show
        compact_str = sprint(show, itp)
        @test occursin("QuadraticInterpolant", compact_str)
        @test occursin("101 pts", compact_str)

        # Verbose show (Range grid → no Search row)
        verbose_str = sprint(show, MIME("text/plain"), itp)
        @test occursin("QuadraticInterpolant", verbose_str)
        @test occursin("Grid:", verbose_str)
        @test occursin("Extrap:", verbose_str)
        @test !occursin("Search:", verbose_str)  # Range → no Search
    end

    @testset "CubicInterpolant show" begin
        # Natural BC (default)
        itp_natural = cubic_interp(x, y)

        compact_str = sprint(show, itp_natural)
        @test occursin("CubicInterpolant", compact_str)
        @test occursin("101 pts", compact_str)
        @test occursin("Natural", compact_str)

        # Verbose show (Range grid → no Search row)
        verbose_str = sprint(show, MIME("text/plain"), itp_natural)
        @test occursin("CubicInterpolant", verbose_str)
        @test occursin("Grid:", verbose_str)
        @test occursin("Extrap:", verbose_str)
        @test !occursin("Search:", verbose_str)  # Range → no Search
        @test occursin("BC:", verbose_str)
        @test occursin("Natural", verbose_str)

        # Custom BC
        itp_custom = cubic_interp(x, y; bc=BCPair(Deriv1(0.5), Deriv2(0.0)))
        verbose_custom = sprint(show, MIME("text/plain"), itp_custom)
        @test occursin("Deriv1", verbose_custom)
        @test occursin("Deriv2", verbose_custom)
    end

    @testset "LinearSeriesInterpolant show" begin
        sitp = linear_interp(x, y_matrix)

        # Compact show
        compact_str = sprint(show, sitp)
        @test occursin("LinearSeriesInterpolant", compact_str)
        @test occursin("101 × 3", compact_str)

        # Verbose show
        verbose_str = sprint(show, MIME("text/plain"), sitp)
        @test occursin("LinearSeriesInterpolant", verbose_str)
        @test occursin("with 3 series", verbose_str)
        @test occursin("Grid:", verbose_str)
        @test occursin("Matrix:", verbose_str)
        @test occursin("n_points × n_series", verbose_str)
    end

    @testset "ConstantSeriesInterpolant show" begin
        sitp = constant_interp(x, y_matrix)

        # Compact show
        compact_str = sprint(show, sitp)
        @test occursin("ConstantSeriesInterpolant", compact_str)
        @test occursin("101 × 3", compact_str)

        # Verbose show
        verbose_str = sprint(show, MIME("text/plain"), sitp)
        @test occursin("ConstantSeriesInterpolant", verbose_str)
        @test occursin("with 3 series", verbose_str)
        @test occursin("Side:", verbose_str)
    end

    @testset "QuadraticSeriesInterpolant show" begin
        sitp = quadratic_interp(x, y_matrix)

        # Compact show
        compact_str = sprint(show, sitp)
        @test occursin("QuadraticSeriesInterpolant", compact_str)
        @test occursin("101 × 3", compact_str)

        # Verbose show
        verbose_str = sprint(show, MIME("text/plain"), sitp)
        @test occursin("QuadraticSeriesInterpolant", verbose_str)
        @test occursin("with 3 series", verbose_str)
    end

    @testset "CubicSeriesInterpolant show" begin
        sitp = cubic_interp(x, y_matrix)

        # Compact show
        compact_str = sprint(show, sitp)
        @test occursin("CubicSeriesInterpolant", compact_str)
        @test occursin("101 × 3", compact_str)
        @test occursin("Natural", compact_str)

        # Verbose show
        verbose_str = sprint(show, MIME("text/plain"), sitp)
        @test occursin("CubicSeriesInterpolant", verbose_str)
        @test occursin("with 3 series", verbose_str)
        @test occursin("BC:", verbose_str)
    end

    @testset "DerivativeView show" begin
        itp = cubic_interp(x, y)
        d1 = deriv1(itp)
        d2 = deriv2(itp)

        # Compact show - deriv1
        compact_d1 = sprint(show, d1)
        @test occursin("DerivativeView", compact_d1)
        @test occursin("{1}", compact_d1)
        @test occursin("CubicInterpolant", compact_d1)

        # Compact show - deriv2
        compact_d2 = sprint(show, d2)
        @test occursin("{2}", compact_d2)

        # Verbose show
        verbose_d1 = sprint(show, MIME("text/plain"), d1)
        @test occursin("DerivativeView", verbose_d1)
        @test occursin("1st derivative", verbose_d1)
        @test occursin("Parent:", verbose_d1)
        @test occursin("CubicInterpolant", verbose_d1)
        @test occursin("101 points", verbose_d1)

        verbose_d2 = sprint(show, MIME("text/plain"), d2)
        @test occursin("2nd derivative", verbose_d2)
    end

    @testset "Search policy display" begin
        x_vec = collect(x)

        # Test different search policies
        itp_binary = linear_interp(x_vec, y; search=Binary())
        itp_hinted = linear_interp(x_vec, y; search=HintedBinary())
        itp_linear_binary = linear_interp(x_vec, y; search=LinearBinary())
        itp_linear_binary_16 = linear_interp(x_vec, y; search=LinearBinary(linear_window=16))

        verbose_binary = sprint(show, MIME("text/plain"), itp_binary)
        @test occursin("Binary", verbose_binary)

        verbose_hinted = sprint(show, MIME("text/plain"), itp_hinted)
        @test occursin("HintedBinary", verbose_hinted)

        verbose_lb = sprint(show, MIME("text/plain"), itp_linear_binary)
        @test occursin("LinearBinary{8}", verbose_lb)

        verbose_lb16 = sprint(show, MIME("text/plain"), itp_linear_binary_16)
        @test occursin("LinearBinary{16}", verbose_lb16)
    end

    @testset "Extrapolation mode display" begin
        itp_none = linear_interp(x, y; extrap=:none)
        itp_const = linear_interp(x, y; extrap=:constant)
        itp_ext = linear_interp(x, y; extrap=:extension)
        itp_wrap = linear_interp(x, y; extrap=:wrap)

        @test occursin(":none", sprint(show, MIME("text/plain"), itp_none))
        @test occursin(":constant", sprint(show, MIME("text/plain"), itp_const))
        @test occursin(":extension", sprint(show, MIME("text/plain"), itp_ext))
        @test occursin(":wrap", sprint(show, MIME("text/plain"), itp_wrap))
    end

    @testset "Grid type display" begin
        x_range = range(0.0, 1.0, 101)
        x_vector = collect(x_range)

        itp_range = linear_interp(x_range, y)
        itp_vector = linear_interp(x_vector, y)

        verbose_range = sprint(show, MIME("text/plain"), itp_range)
        @test occursin("Range", verbose_range)

        verbose_vector = sprint(show, MIME("text/plain"), itp_vector)
        @test occursin("Vector", verbose_vector)
    end

    @testset "Search row conditional display" begin
        # Range grid → no Search row (O(1) index computation)
        x_range = range(0.0, 1.0, 101)
        itp_range = linear_interp(x_range, y)
        verbose_range = sprint(show, MIME("text/plain"), itp_range)
        @test !occursin("Search:", verbose_range)

        # Vector grid → Search row shown (binary search needed)
        x_vector = collect(x_range)
        itp_vector = linear_interp(x_vector, y; search=Binary())
        verbose_vector = sprint(show, MIME("text/plain"), itp_vector)
        @test occursin("Search:", verbose_vector)
        @test occursin("Binary", verbose_vector)
    end
end
