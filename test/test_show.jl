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

    @testset "LinearSeriesInterpolant Complex show (Tg/Tv display)" begin
        # Complex values: should show both type parameters
        y_complex = [exp.(2im .* π .* collect(x)) for _ in 1:2]
        sitp_complex = linear_interp(x, y_complex)

        # Compact show: should display {Float64, ComplexF64}
        compact_str = sprint(show, sitp_complex)
        @test occursin("LinearSeriesInterpolant", compact_str)
        @test occursin("Float64", compact_str)
        @test occursin("ComplexF64", compact_str)
        @test occursin("101 × 2", compact_str)

        # Verbose show
        verbose_str = sprint(show, MIME("text/plain"), sitp_complex)
        @test occursin("Float64", verbose_str)
        @test occursin("ComplexF64", verbose_str)
        @test occursin("with 2 series", verbose_str)

        # Real values: should only show single type parameter (backward compatible)
        sitp_real = linear_interp(x, y_matrix)
        compact_real = sprint(show, sitp_real)
        @test occursin("Float64", compact_real)
        @test !occursin("Float64, Float64", compact_real)  # Should NOT duplicate
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

    # ========================================
    # Additional coverage tests
    # ========================================

    @testset "Color output (IOContext :color)" begin
        itp = linear_interp(x, y)
        # Test with color-enabled IO context
        io_color = IOContext(IOBuffer(), :color => true)
        show(io_color, MIME("text/plain"), itp)
        output = String(take!(io_color.io))
        # Should still contain the text (color codes are invisible in string)
        @test occursin("LinearInterpolant", output)
        @test occursin("Grid:", output)
    end

    @testset "Integer grid (_format_num non-float)" begin
        # Integer grid triggers _format_num(x) = string(x) fallback
        x_int = 1:10
        y_int = collect(Float64, 1:10)
        itp = linear_interp(x_int, y_int)
        verbose = sprint(show, MIME("text/plain"), itp)
        @test occursin("Grid:", verbose)
        @test occursin("10 points", verbose)
        @test occursin("[1, 10]", verbose)  # Integer bounds
    end

    @testset "Side options (:left, :right)" begin
        x_vec = collect(range(0.0, 1.0, 11))
        y_short = sin.(x_vec)

        itp_left = constant_interp(x_vec, y_short; side=:left)
        itp_right = constant_interp(x_vec, y_short; side=:right)

        verbose_left = sprint(show, MIME("text/plain"), itp_left)
        verbose_right = sprint(show, MIME("text/plain"), itp_right)

        @test occursin(":left", verbose_left)
        @test occursin(":right", verbose_right)
    end

    @testset "Search policy formatting (direct)" begin
        x_vec = collect(range(0.0, 1.0, 11))
        y_short = sin.(x_vec)

        # Linear search policy
        itp_linear = linear_interp(x_vec, y_short; search=Linear())
        verbose_linear = sprint(show, MIME("text/plain"), itp_linear)
        @test occursin("Linear", verbose_linear)
        @test !occursin("LinearBinary", verbose_linear)  # Should be just "Linear"
    end

    @testset "BC type formatting" begin
        x_vec = collect(range(0.0, 1.0, 11))
        y_short = sin.(x_vec)

        # Clamped BC (Deriv1(0) at both ends)
        itp_clamped = cubic_interp(x_vec, y_short; bc=BCPair(Deriv1(0.0), Deriv1(0.0)))
        verbose_clamped = sprint(show, MIME("text/plain"), itp_clamped)
        @test occursin("Clamped", verbose_clamped)

        # Periodic BC (use PeriodicBC() API, internally becomes PeriodicData)
        y_periodic = sin.(2π .* x_vec)
        y_periodic[end] = y_periodic[1]  # Ensure periodicity
        itp_periodic = cubic_interp(x_vec, y_periodic; bc=PeriodicBC())
        verbose_periodic = sprint(show, MIME("text/plain"), itp_periodic)
        @test occursin("Periodic", verbose_periodic)

        # Custom BC with Deriv3
        itp_deriv3 = cubic_interp(x_vec, y_short; bc=BCPair(Deriv3(0.0), Deriv3(0.0)))
        verbose_deriv3 = sprint(show, MIME("text/plain"), itp_deriv3)
        @test occursin("Deriv3", verbose_deriv3)

        # Mixed BC types (tests _format_bc_point fallback path)
        itp_mixed = cubic_interp(x_vec, y_short; bc=BCPair(Deriv1(0.5), Deriv3(0.0)))
        verbose_mixed = sprint(show, MIME("text/plain"), itp_mixed)
        @test occursin("Deriv1", verbose_mixed)
        @test occursin("Deriv3", verbose_mixed)
    end

    @testset "Short BC name for compact display" begin
        x_vec = collect(range(0.0, 1.0, 11))
        y_short = sin.(x_vec)

        # Clamped (Deriv1)
        itp_clamped = cubic_interp(x_vec, y_short; bc=BCPair(Deriv1(0.0), Deriv1(0.0)))
        compact_clamped = sprint(show, itp_clamped)
        @test occursin("Clamped", compact_clamped)

        # Periodic (use PeriodicBC() API)
        y_periodic = sin.(2π .* x_vec)
        y_periodic[end] = y_periodic[1]
        itp_periodic = cubic_interp(x_vec, y_periodic; bc=PeriodicBC())
        compact_periodic = sprint(show, itp_periodic)
        @test occursin("Periodic", compact_periodic)

        # Custom (non-standard BC with mixed types)
        itp_custom = cubic_interp(x_vec, y_short; bc=BCPair(Deriv1(1.0), Deriv2(0.5)))
        compact_custom = sprint(show, itp_custom)
        @test occursin("Custom", compact_custom)

        # Note: autocache=false to avoid cache sharing with previous BC of same type
        itp_deriv2_nonzero = cubic_interp(x_vec, y_short; bc=BCPair(Deriv2(1.0), Deriv2(0.0)))
        compact_deriv2_nonzero = sprint(show, itp_deriv2_nonzero)
        @test occursin("Custom", compact_deriv2_nonzero)
        @test !occursin("Natural", compact_deriv2_nonzero)

        # Bug fix: Deriv1 type but non-zero values should show "Custom", not "Clamped"
        # Note: autocache=false to avoid cache sharing with previous BC of same type
        itp_deriv1_nonzero = cubic_interp(x_vec, y_short; bc=BCPair(Deriv1(0.5), Deriv1(1.0)))
        compact_deriv1_nonzero = sprint(show, itp_deriv1_nonzero)
        @test occursin("Custom", compact_deriv1_nonzero)
        @test !occursin("Clamped", compact_deriv1_nonzero)
    end

    @testset "3rd+ derivative order formatting" begin
        itp = cubic_interp(x, y)
        d3 = deriv3(itp)

        verbose_d3 = sprint(show, MIME("text/plain"), d3)
        @test occursin("3rd derivative", verbose_d3)
    end

    @testset "Vector grid Search row - all interpolant types" begin
        x_vec = collect(range(0.0, 1.0, 11))
        y_short = sin.(x_vec)
        y_matrix_short = [sin.(x_vec) cos.(x_vec)]

        # ConstantInterpolant with Vector grid
        itp_const = constant_interp(x_vec, y_short; search=Binary())
        verbose_const = sprint(show, MIME("text/plain"), itp_const)
        @test occursin("Search:", verbose_const)
        @test occursin("Binary", verbose_const)

        # QuadraticInterpolant with Vector grid
        itp_quad = quadratic_interp(x_vec, y_short; search=Binary())
        verbose_quad = sprint(show, MIME("text/plain"), itp_quad)
        @test occursin("Search:", verbose_quad)

        # CubicInterpolant with Vector grid
        itp_cubic = cubic_interp(x_vec, y_short; search=Binary())
        verbose_cubic = sprint(show, MIME("text/plain"), itp_cubic)
        @test occursin("Search:", verbose_cubic)

        # CubicSeriesInterpolant with Vector grid
        sitp_cubic = cubic_interp(x_vec, y_matrix_short; search=Binary())
        verbose_sitp = sprint(show, MIME("text/plain"), sitp_cubic)
        @test occursin("Search:", verbose_sitp)

        # LinearSeriesInterpolant with Vector grid
        sitp_linear = linear_interp(x_vec, y_matrix_short; search=Binary())
        verbose_sitp_linear = sprint(show, MIME("text/plain"), sitp_linear)
        @test occursin("Search:", verbose_sitp_linear)
        @test occursin("Binary", verbose_sitp_linear)

        # ConstantSeriesInterpolant with Vector grid
        sitp_const = constant_interp(x_vec, y_matrix_short; search=Binary())
        verbose_sitp_const = sprint(show, MIME("text/plain"), sitp_const)
        @test occursin("Search:", verbose_sitp_const)

        # QuadraticSeriesInterpolant with Vector grid
        sitp_quad = quadratic_interp(x_vec, y_matrix_short; search=Binary())
        verbose_sitp_quad = sprint(show, MIME("text/plain"), sitp_quad)
        @test occursin("Search:", verbose_sitp_quad)
    end

    @testset "DerivativeView with LinearInterpolant parent" begin
        # LinearInterpolant has .x directly (not .cache.x)
        itp_linear = linear_interp(x, y)
        d1_linear = deriv1(itp_linear)

        verbose = sprint(show, MIME("text/plain"), d1_linear)
        @test occursin("LinearInterpolant", verbose)
        @test occursin("101 points", verbose)
    end

    @testset "Coverage: _format_bc" begin
        x = collect(range(0.0, 2π, 11))
        y = sin.(x)    
        
        # Custom BC with Deriv3
        itp = cubic_interp(x, y; bc=Deriv3(0.0))
        @test occursin("Deriv3(0.0) | Deriv3(0.0)", sprint(show, MIME("text/plain"), itp))

        itp = cubic_interp(x, y; bc=BCPair(Deriv3(0.0), Deriv3(0.0)))
        @test occursin("Deriv3(0.0) | Deriv3(0.0)", sprint(show, MIME("text/plain"), itp))

        itp = cubic_interp(x, y; bc=BCPair(Deriv2(1.0), Deriv3(-5.0)))
        @test occursin("Deriv2(1.0) | Deriv3(-5.0)", sprint(show, MIME("text/plain"), itp))

        itp = cubic_interp(x, y; bc=NaturalBC())
        @test occursin("Natural (S''=0 at ends)", sprint(show, MIME("text/plain"), itp))

        itp = cubic_interp(x, y; bc=ClampedBC())
        @test occursin("Clamped (S'=0 at ends)", sprint(show, MIME("text/plain"), itp))

        itp = cubic_interp(x, y; bc=PeriodicBC())
        @test occursin("Periodic", sprint(show, MIME("text/plain"), itp))
    end


    @testset "Coverage: Direct call of inlined formatting functions" begin
        FI = FastInterpolations

        # Access internal formatting functions directly to ensure full coverage
        @test FI._format_extrap(Val(:unknown_mode)) == "unknown"
        @test FI._format_side(Val(:unknown_side)) == "unknown"
        @test FI._format_deriv_order(4) == "4th"

        @test FI._format_search(Linear()) == "Linear"
        @test FI._format_search(Binary()) == "Binary"
        @test FI._format_search(HintedBinary()) == "HintedBinary"
        @test FI._format_search(LinearBinary()) == "LinearBinary{8}"
        @test FI._format_search(LinearBinary(linear_window=4)) == "LinearBinary{4}"

        # DerivativeView with unknown parent type (no .x or .cache.x)
        struct DummyInterpolant{T} <: FastInterpolations.AbstractInterpolant{T, T} end
        Base.show(io::IO, ::DummyInterpolant) = print(io, "Dummy")
        d_dummy = DerivativeView{1, DummyInterpolant{Float64}}(DummyInterpolant{Float64}())
        verbose_dummy = sprint(show, MIME("text/plain"), d_dummy)
        @test occursin("?", verbose_dummy)

        # BC formatting for types not usually exposed in high-level show
        @test FI._format_bc(MinCurvFit()) == "MinCurvFit"
        @test FI._format_bc(QuadraticFit()) == "QuadraticFit"
        @test occursin("Left", FI._format_bc(Left(Deriv1(0.0))))
        @test occursin("Right", FI._format_bc(Right(Deriv1(0.0))))

        # Direct verification of single-BC formatters (bypassed by BCPair logic)
        @test FI._format_bc(NaturalBC()) == "Natural (S''=0 at ends)"
        @test FI._format_bc(ClampedBC()) == "Clamped (S'=0 at ends)"
        @test FI._format_bc(PeriodicBC()) == "Periodic"
        @test FI._format_bc(Deriv1(1.0)) == "Deriv1(1.0)"
        @test FI._format_bc(Deriv2(2.0)) == "Deriv2(2.0)"
        @test FI._format_bc(Deriv3(3.0)) == "Deriv3(3.0)"
        @test FI._format_bc(MinCurvFit()) == "MinCurvFit"
        @test FI._format_bc(LinearFit()) == "LinearFit"
        @test FI._format_bc(QuadraticFit()) == "QuadraticFit"
        @test FI._format_bc(CubicFit()) == "CubicFit"
        @test FI._format_bc(PolyFit{4}()) == "PolyFit{4}"

        @test FI._short_bc_name(PeriodicBC()) == "Periodic"

        # BC point fallback
        @test FI._format_bc_point(LinearFit()) == "LinearFit"
        @test FI._format_bc_point(QuadraticFit()) == "QuadraticFit"
        @test FI._format_bc_point(CubicFit()) == "CubicFit"
        @test FI._format_bc_point(PolyFit{4}()) == "PolyFit{4}"
        
        # Test Left/Right wrappers with PolyFit (integration test)
        @test FI._format_bc(Left(LinearFit())) == "Left(LinearFit)"
        @test FI._format_bc(Right(CubicFit())) == "Right(CubicFit)"

        struct UnknownBC end
        @test FI._format_bc_point(UnknownBC()) == "UnknownBC"
    end

    # ========================================
    # Additional Coverage Tests for show.jl
    # ========================================

    @testset "Superscript digit and integer functions" begin
        FI = FastInterpolations

        # Test _superscript_digit for all single digits 0-9
        @test FI._superscript_digit(0) == "⁰"
        @test FI._superscript_digit(1) == "¹"
        @test FI._superscript_digit(2) == "²"
        @test FI._superscript_digit(3) == "³"
        @test FI._superscript_digit(4) == "⁴"
        @test FI._superscript_digit(5) == "⁵"
        @test FI._superscript_digit(6) == "⁶"
        @test FI._superscript_digit(7) == "⁷"
        @test FI._superscript_digit(8) == "⁸"
        @test FI._superscript_digit(9) == "⁹"
        # Fallback for d >= 10
        @test FI._superscript_digit(10) == "10"
        @test FI._superscript_digit(15) == "15"

        # Test _superscript_int for various values
        @test FI._superscript_int(0) == "⁰"
        @test FI._superscript_int(5) == "⁵"
        @test FI._superscript_int(9) == "⁹"
        # Negative numbers use ^(n) fallback
        @test FI._superscript_int(-1) == "^-1"
        @test FI._superscript_int(-5) == "^-5"
    end

    @testset "Subscript digit fallback for d >= 10" begin
        FI = FastInterpolations

        # Normal single digits
        @test FI._subscript_digit(0) == "₀"
        @test FI._subscript_digit(1) == "₁"
        @test FI._subscript_digit(9) == "₉"

        # Fallback for d >= 10 (uses parentheses)
        @test FI._subscript_digit(10) == "(10)"
        @test FI._subscript_digit(15) == "(15)"
        @test FI._subscript_digit(100) == "(100)"
    end

    @testset "DerivativeView with tuple order (ND partial derivatives)" begin
        FI = FastInterpolations

        # Test _format_deriv_order with tuple input (ND partial derivatives)
        # (0, 0) - all zeros returns value form
        result_00 = FI._format_deriv_order((0, 0))
        @test occursin("value", result_00)
        @test occursin("f", result_00)

        # (1, 0) - first-order partial with respect to x₁
        result_10 = FI._format_deriv_order((1, 0))
        @test occursin("partial derivatives", result_10)
        @test occursin("∂x", result_10)

        # (0, 1) - first-order partial with respect to x₂
        result_01 = FI._format_deriv_order((0, 1))
        @test occursin("partial derivatives", result_01)

        # (2, 0) - second-order partial (exercises _superscript_int branch in push!)
        result_20 = FI._format_deriv_order((2, 0))
        @test occursin("partial derivatives", result_20)
        @test occursin("²", result_20)  # Should contain superscript 2

        # (1, 1) - mixed partial
        result_11 = FI._format_deriv_order((1, 1))
        @test occursin("partial derivatives", result_11)

        # (3, 2) - higher order mixed partial
        result_32 = FI._format_deriv_order((3, 2))
        @test occursin("partial derivatives", result_32)
        @test occursin("³", result_32)  # superscript 3
        @test occursin("²", result_32)  # superscript 2
    end

    @testset "CubicInterpolantND show with mixed BCs per axis" begin
        # Create 2D data
        x1 = range(0.0, 1.0, 11)
        x2 = range(0.0, 2.0, 15)
        f = [sin(2π * xi) * cos(π * xj) for xi in x1, xj in x2]

        # Mixed BCs: Natural on axis 1, Clamped on axis 2
        bc_natural = NaturalBC()
        bc_clamped = BCPair(Deriv1(0.0), Deriv1(0.0))  # Clamped
        itp_mixed = cubic_interp((x1, x2), f; bc=(bc_natural, bc_clamped))

        # Compact show should say "Mixed"
        compact_str = sprint(show, itp_mixed)
        @test occursin("CubicInterpolantND", compact_str)
        @test occursin("Mixed", compact_str)

        # Verbose show should display per-axis BCs hierarchically
        verbose_str = sprint(show, MIME("text/plain"), itp_mixed)
        @test occursin("BC:", verbose_str)
        @test occursin("x₁", verbose_str)
        @test occursin("x₂", verbose_str)
        @test occursin("Natural", verbose_str)
        @test occursin("Clamped", verbose_str)
    end

    @testset "CubicInterpolantND show with heterogeneous extrapolation modes" begin
        # Create 2D data
        x1 = range(0.0, 1.0, 11)
        x2 = range(0.0, 2.0, 15)
        f = [sin(2π * xi) * cos(π * xj) for xi in x1, xj in x2]

        # Different extrapolation modes per axis
        itp_mixed_extrap = cubic_interp((x1, x2), f; extrap=(:none, :constant))

        # Verbose show should display tuple format for extrapolation
        verbose_str = sprint(show, MIME("text/plain"), itp_mixed_extrap)
        @test occursin("Extrap:", verbose_str)
        @test occursin(":none", verbose_str)
        @test occursin(":constant", verbose_str)
    end

    @testset "CubicInterpolantND show with heterogeneous search policies" begin
        # Create 2D data with Vector grids (non-Range to trigger search display)
        x1 = collect(range(0.0, 1.0, 11))
        x2 = collect(range(0.0, 2.0, 15))
        f = [sin(2π * xi) * cos(π * xj) for xi in x1, xj in x2]

        # Different search policies per axis
        itp_mixed_search = cubic_interp((x1, x2), f; search=(Binary(), Linear()))

        # Verbose show should display tuple format for search policies
        verbose_str = sprint(show, MIME("text/plain"), itp_mixed_search)
        @test occursin("Search:", verbose_str)
        @test occursin("Binary", verbose_str)
        @test occursin("Linear", verbose_str)
    end

    @testset "CubicInterpolantND show with complex values (Tv ≠ Tg)" begin
        # Create 2D complex data
        x1 = range(0.0, 1.0, 11)
        x2 = range(0.0, 2.0, 15)
        f_complex = [exp(2π * im * xi) * cos(π * xj) for xi in x1, xj in x2]

        itp_complex = cubic_interp((x1, x2), f_complex)

        # Compact show should display both Tg and Tv
        compact_str = sprint(show, itp_complex)
        @test occursin("CubicInterpolantND", compact_str)
        @test occursin("Float64", compact_str)
        @test occursin("ComplexF64", compact_str)

        # Verbose show should also display both types
        verbose_str = sprint(show, MIME("text/plain"), itp_complex)
        @test occursin("Float64", verbose_str)
        @test occursin("ComplexF64", verbose_str)
    end

    @testset "DerivativeView show with CubicInterpolantND parent" begin
        # Create 2D data
        x1 = range(0.0, 1.0, 11)
        x2 = range(0.0, 2.0, 15)
        f = [sin(2π * xi) * cos(π * xj) for xi in x1, xj in x2]

        itp_nd = cubic_interp((x1, x2), f)

        # Create DerivativeView with tuple order
        d_view = deriv_view(itp_nd, (1, 0))

        # Compact show
        compact_str = sprint(show, d_view)
        @test occursin("DerivativeView", compact_str)
        @test occursin("CubicInterpolantND", compact_str)

        # Verbose show - should display tuple derivative info and ND parent info
        verbose_str = sprint(show, MIME("text/plain"), d_view)
        @test occursin("DerivativeView", verbose_str)
        @test occursin("partial derivatives", verbose_str)
        @test occursin("Parent:", verbose_str)
        @test occursin("CubicInterpolantND", verbose_str)
        @test occursin("2D", verbose_str)
        @test occursin("11×15", verbose_str)
    end

    @testset "ND grid display with Vector grids (triggers Search row)" begin
        # Create 2D data with Vector grids
        x1 = collect(range(0.0, 1.0, 11))
        x2 = collect(range(0.0, 2.0, 15))
        f = [sin(2π * xi) * cos(π * xj) for xi in x1, xj in x2]

        itp_vec = cubic_interp((x1, x2), f)

        # Verbose show should include Search row
        verbose_str = sprint(show, MIME("text/plain"), itp_vec)
        @test occursin("Grids:", verbose_str)
        @test occursin("Vector", verbose_str)
        @test occursin("Search:", verbose_str)
    end

    @testset "3D CubicInterpolantND show" begin
        # Create 3D data
        x1 = range(0.0, 1.0, 8)
        x2 = range(0.0, 2.0, 10)
        x3 = range(0.0, 3.0, 12)
        f = [sin(xi) * cos(xj) * exp(-xk/3) for xi in x1, xj in x2, xk in x3]

        itp_3d = cubic_interp((x1, x2, x3), f)

        # Compact show
        compact_str = sprint(show, itp_3d)
        @test occursin("CubicInterpolantND", compact_str)
        @test occursin("8×10×12", compact_str)

        # Verbose show
        verbose_str = sprint(show, MIME("text/plain"), itp_3d)
        @test occursin("3D", verbose_str)
        @test occursin("x₁", verbose_str)
        @test occursin("x₂", verbose_str)
        @test occursin("x₃", verbose_str)
    end

    @testset "Color output for ND interpolants" begin
        x1 = range(0.0, 1.0, 11)
        x2 = range(0.0, 2.0, 15)
        f = [sin(2π * xi) * cos(π * xj) for xi in x1, xj in x2]

        itp_nd = cubic_interp((x1, x2), f)

        # Test with color-enabled IO context
        io_color = IOContext(IOBuffer(), :color => true)
        show(io_color, MIME("text/plain"), itp_nd)
        output = String(take!(io_color.io))

        # Should contain the text (color codes are invisible in string)
        @test occursin("CubicInterpolantND", output)
        @test occursin("Grids:", output)
        @test occursin("BC:", output)
    end

    @testset "Direct test of _show_nd_config_row with different values" begin
        FI = FastInterpolations

        # Test with heterogeneous tuple (different values per axis)
        io = IOBuffer()
        configs = (Val(:none), Val(:constant), Val(:extension))
        FI._show_nd_config_row(io, false, "Extrap:", configs, FI._format_extrap; value_color=:magenta)
        output = String(take!(io))

        # Should show tuple format since values differ
        @test occursin("(", output)
        @test occursin(":none", output)
        @test occursin(":constant", output)
        @test occursin(":extension", output)

        # Test with homogeneous tuple (same values)
        io2 = IOBuffer()
        configs_same = (Val(:none), Val(:none))
        FI._show_nd_config_row(io2, true, "Extrap:", configs_same, FI._format_extrap; value_color=:magenta)
        output2 = String(take!(io2))

        # Should show single value with "(all axes)"
        @test occursin(":none", output2)
        @test occursin("(all axes)", output2)
    end

    @testset "Direct test of _short_bc_name_nd" begin
        FI = FastInterpolations

        # Same BCs on all axes (Natural = Deriv2(0) at both ends)
        bc_natural = BCPair(Deriv2(0.0), Deriv2(0.0))
        bcs_same = (bc_natural, bc_natural)
        @test FI._short_bc_name_nd(bcs_same) == "Natural"

        # Different BCs on axes (Natural vs Clamped)
        bc_clamped = BCPair(Deriv1(0.0), Deriv1(0.0))
        bcs_mixed = (bc_natural, bc_clamped)
        @test FI._short_bc_name_nd(bcs_mixed) == "Mixed"

        # Three axes, all same (Periodic)
        bcs_3d_periodic = (PeriodicBC(), PeriodicBC(), PeriodicBC())
        @test FI._short_bc_name_nd(bcs_3d_periodic) == "Periodic"

        # Three axes, mixed
        bcs_3d_mixed = (bc_natural, bc_clamped, bc_natural)
        @test FI._short_bc_name_nd(bcs_3d_mixed) == "Mixed"

        # Custom BCs (non-standard values) - both axes same
        bc_custom = BCPair(Deriv1(1.0), Deriv2(0.5))
        bcs_custom_same = (bc_custom, bc_custom)
        @test FI._short_bc_name_nd(bcs_custom_same) == "Custom"
    end
end
