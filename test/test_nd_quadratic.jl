# ========================================
# Tests for QuadraticInterpolantND
# ========================================
#
# Comprehensive test coverage for N-dimensional quadratic interpolation.
# Tests cover: polynomial reproduction, derivatives, batch queries,
# extrapolation, grid types, complex values, BCs, and error handling.

using Test
using FastInterpolations

@testset "QuadraticInterpolantND" begin

    # ========================================
    # 2D Polynomial Reproduction
    # ========================================
    @testset "2D polynomial reproduction (x²+y²)" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        # Type checks
        @test itp isa QuadraticInterpolantND
        @test ndims(itp) == 2
        @test size(itp) == (15, 11)

        # Grid point pass-through
        for i in 1:length(x)
            for j in 1:length(y)
                @test itp((x[i], y[j])) ≈ data[i, j] atol=1e-12
            end
        end

        # Interior point — exact for degree 2
        xq, yq = 1.23, 0.67
        @test itp((xq, yq)) ≈ f(xq, yq) rtol=1e-10
    end

    @testset "2D polynomial reproduction with cross term (x²+xy+y²)" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + xi*yi + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
        xq, yq = 1.0, 0.5
        @test itp((xq, yq)) ≈ f(xq, yq) rtol=1e-8
    end

    # ========================================
    # Derivative Evaluation
    # ========================================
    @testset "2D derivatives" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        xq, yq = 1.23, 0.67

        # Value
        @test itp((xq, yq); deriv=(0, 0)) ≈ f(xq, yq) rtol=1e-10

        # First derivatives: df/dx = 2x, df/dy = 2y
        @test itp((xq, yq); deriv=(1, 0)) ≈ 2xq rtol=1e-10
        @test itp((xq, yq); deriv=(0, 1)) ≈ 2yq rtol=1e-10

        # Second derivatives: d²f/dx² = 2, d²f/dy² = 2
        @test itp((xq, yq); deriv=(2, 0)) ≈ 2.0 rtol=1e-6
        @test itp((xq, yq); deriv=(0, 2)) ≈ 2.0 rtol=1e-6

        # Mixed derivative: d²f/dxdy = 0 for x²+y²
        @test abs(itp((xq, yq); deriv=(1, 1))) < 1e-10

        # Val-based derivative spec (compile-time)
        @test itp((xq, yq); deriv=Val((1, 0))) ≈ 2xq rtol=1e-10
        @test itp((xq, yq); deriv=Val((0, 1))) ≈ 2yq rtol=1e-10
        @test itp((xq, yq); deriv=Val((2, 0))) ≈ 2.0 rtol=1e-6

        # Integer derivative (all axes same order)
        @test itp((xq, yq); deriv=1) isa Float64
        @test itp((xq, yq); deriv=2) isa Float64
    end

    @testset "2D non-zero mixed derivative" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + xi*yi + yi^2
        data = [f(xi, yi) for xi in x, yi in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        xq, yq = 1.0, 0.5

        # d²f/dxdy = 1 for x² + xy + y²
        @test itp((xq, yq); deriv=(1, 1)) ≈ 1.0 atol=1e-6
    end

    # ========================================
    # Boundary Conditions
    # ========================================
    @testset "boundary conditions" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        @testset "Left(QuadraticFit())" begin
            itp = quadratic_interp((x, y), data; bc=Left(QuadraticFit()))
            @test itp((1.0, 0.5)) ≈ f(1.0, 0.5) rtol=1e-8
        end

        @testset "Right(QuadraticFit())" begin
            itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
            @test itp((1.0, 0.5)) ≈ f(1.0, 0.5) rtol=1e-10
        end

        @testset "NaturalBC()" begin
            data_sin = [sin(xi) * cos(yi) for xi in x, yi in y]
            itp = quadratic_interp((x, y), data_sin; bc=NaturalBC())
            @test isapprox(itp((1.0, 0.5)), sin(1.0) * cos(0.5); atol=2e-3)
        end

        @testset "MinCurvFit()" begin
            itp = quadratic_interp((x, y), data; bc=MinCurvFit())
            @test isapprox(itp((1.0, 0.5)), f(1.0, 0.5); rtol=1e-4)
        end

        @testset "Per-axis BC (Left + Right)" begin
            itp = quadratic_interp((x, y), data;
                bc=(Left(QuadraticFit()), Right(QuadraticFit())))
            @test itp((1.0, 0.5)) ≈ f(1.0, 0.5) rtol=1e-8
        end

        @testset "PolyFit BC conversion" begin
            itp = quadratic_interp((x, y), data; bc=CubicFit())
            @test itp((1.0, 0.5)) isa Float64
        end
    end

    # ========================================
    # Search Policies
    # ========================================
    @testset "search policies" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        @testset "uniform search" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()), search=Binary())
            @test itp((1.0, 0.5)) ≈ f(1.0, 0.5) rtol=1e-10
        end

        @testset "mixed search policies" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()),
                search=(Binary(), LinearBinary{4}()))
            @test itp((1.0, 0.5)) ≈ f(1.0, 0.5) rtol=1e-10
        end
    end

    # ========================================
    # Extrapolation Modes
    # ========================================
    @testset "extrapolation modes" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        @testset "extrap=:none (default)" begin
            itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
            @test_throws DomainError itp((-0.1, 0.5))
            @test_throws DomainError itp((0.5, -0.1))
            @test_throws DomainError itp((2.1, 0.5))
            @test_throws DomainError itp((0.5, 1.1))
        end

        @testset "extrap=:constant" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()), extrap=:constant)
            @test itp((-0.1, 0.5)) ≈ itp((0.0, 0.5))
            @test itp((2.1, 0.5)) ≈ itp((2.0, 0.5))
            @test itp((0.5, 1.1)) ≈ itp((0.5, 1.0))
            @test itp((0.5, -0.1)) ≈ itp((0.5, 0.0))
        end

        @testset "extrap=:extension" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()), extrap=:extension)
            @test isfinite(itp((-0.1, 0.5)))
            @test isfinite(itp((2.5, 1.5)))
        end

        @testset "per-axis extrap" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()),
                extrap=(:constant, :extension))
            @test itp((-0.1, 0.5)) ≈ itp((0.0, 0.5)) rtol=1e-10
            @test isfinite(itp((1.0, 1.5)))
        end
    end

    # ========================================
    # Batch Evaluation
    # ========================================
    @testset "batch evaluation" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @testset "SoA (Tuple of Vectors)" begin
            xs = [0.5, 1.0, 1.5]
            ys = [0.2, 0.5, 0.8]
            results = itp((xs, ys))

            @test results isa Vector{Float64}
            @test length(results) == 3
            for i in 1:3
                @test results[i] ≈ f(xs[i], ys[i]) rtol=1e-10
            end
        end

        @testset "SoA with derivatives" begin
            xs = [0.5, 1.0, 1.5]
            ys = [0.2, 0.5, 0.8]
            dvals = itp((xs, ys); deriv=(1, 0))
            for i in 1:3
                @test dvals[i] ≈ 2xs[i] rtol=1e-10
            end
        end

        @testset "AoS (Vector of Tuples)" begin
            queries = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
            results = itp(queries)

            @test results isa Vector{Float64}
            @test length(results) == 3
            for i in 1:3
                @test results[i] ≈ f(queries[i]...) rtol=1e-10
            end
        end

        @testset "AoS with derivatives" begin
            queries = [(0.5, 0.2), (1.0, 0.5)]
            dvals = itp(queries; deriv=(1, 0))
            @test dvals[1] ≈ 2 * 0.5 rtol=1e-10
            @test dvals[2] ≈ 2 * 1.0 rtol=1e-10
        end

        @testset "Vector input (ForwardDiff compat)" begin
            @test itp([1.0, 0.5]) ≈ f(1.0, 0.5) rtol=1e-10
            @test_throws DimensionMismatch itp([1.0])
            @test_throws DimensionMismatch itp([1.0, 0.5, 0.3])
        end
    end

    # ========================================
    # One-Shot API
    # ========================================
    @testset "one-shot API" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        @testset "single point" begin
            val = quadratic_interp((x, y), data, (1.0, 0.5);
                bc=Right(QuadraticFit()))
            @test val ≈ f(1.0, 0.5) rtol=1e-10
        end

        @testset "single point with derivative" begin
            val = quadratic_interp((x, y), data, (1.0, 0.5);
                bc=Right(QuadraticFit()), deriv=Val((1, 0)))
            @test val ≈ 2.0 rtol=1e-10  # df/dx = 2x = 2
        end

        @testset "batch one-shot" begin
            xqs = [0.5, 1.0, 1.5]
            yqs = [0.2, 0.5, 0.8]
            vals = quadratic_interp((x, y), data, (xqs, yqs);
                bc=Right(QuadraticFit()))
            for k in 1:3
                @test vals[k] ≈ f(xqs[k], yqs[k]) rtol=1e-10
            end
        end
    end

    # ========================================
    # 3D Interpolation
    # ========================================
    @testset "3D polynomial reproduction" begin
        x = range(0.0, 1.0, 8)
        y = range(0.0, 1.0, 8)
        z = range(0.0, 1.0, 8)
        f(xi, yi, zi) = xi^2 + yi^2 + zi^2
        data = [f(xi, yi, zi) for xi in x, yi in y, zi in z]
        itp = quadratic_interp((x, y, z), data; bc=Right(QuadraticFit()))

        @test ndims(itp) == 3
        @test size(itp) == (8, 8, 8)

        xq, yq, zq = 0.5, 0.3, 0.7

        # Value
        @test itp((xq, yq, zq)) ≈ f(xq, yq, zq) rtol=1e-8

        # First derivatives
        @test itp((xq, yq, zq); deriv=(1, 0, 0)) ≈ 2xq rtol=1e-6
        @test itp((xq, yq, zq); deriv=(0, 1, 0)) ≈ 2yq rtol=1e-6
        @test itp((xq, yq, zq); deriv=(0, 0, 1)) ≈ 2zq rtol=1e-6

        # Second derivatives
        @test itp((xq, yq, zq); deriv=(2, 0, 0)) ≈ 2.0 rtol=1e-4
        @test itp((xq, yq, zq); deriv=(0, 2, 0)) ≈ 2.0 rtol=1e-4
        @test itp((xq, yq, zq); deriv=(0, 0, 2)) ≈ 2.0 rtol=1e-4
    end

    # ========================================
    # Non-uniform Grids
    # ========================================
    @testset "non-uniform grids" begin
        x = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
        y = [0.0, 0.2, 0.5, 1.0]
        f(xi, yi) = xi^2 + yi^2
        data = [f(xi, yi) for xi in x, yi in y]

        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        # Grid point pass-through
        @test itp((x[3], y[2])) ≈ data[3, 2] atol=1e-12

        # Interior point — exact for degree 2
        xq, yq = 0.5, 0.3
        @test itp((xq, yq)) ≈ f(xq, yq) rtol=1e-6
    end

    # ========================================
    # Float32 Support
    # ========================================
    @testset "Float32 support" begin
        x = range(0.0f0, 2.0f0, 11)
        y = range(0.0f0, 1.0f0, 6)
        data = Float32[xi^2 + yj^2 for xi in x, yj in y]

        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @test grid_type(itp) == Float32
        @test value_type(itp) == Float32

        result = itp((1.0f0, 0.5f0))
        @test result isa Float32
        @test result ≈ 1.0f0 + 0.25f0 atol=1e-4
    end

    # ========================================
    # Complex Value Support
    # ========================================
    @testset "complex value support" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [complex(xi^2, yj^2) for xi in x, yj in y]

        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @test value_type(itp) == ComplexF64

        result = itp((1.0, 0.5))
        @test result isa ComplexF64
        @test result ≈ complex(1.0, 0.25) atol=1e-6
    end

    # ========================================
    # Type Introspection
    # ========================================
    @testset "type introspection" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = rand(11, 11)
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @test grid_type(itp) == Float64
        @test value_type(itp) == Float64
        @test ndims(itp) == 2
        @test size(itp) == (11, 11)
        @test length(axes(itp)) == 2
        @test FastInterpolations.num_partials(itp) == 4  # 2^2
        @test FastInterpolations.num_partials(typeof(itp)) == 4
    end

    # ========================================
    # Error Handling
    # ========================================
    @testset "error handling" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 6)

        @testset "dimension mismatch in construction" begin
            bad_data = rand(10, 6)  # Wrong x dimension
            @test_throws DimensionMismatch quadratic_interp((x, y), bad_data;
                bc=Right(QuadraticFit()))

            bad_data2 = rand(11, 5)  # Wrong y dimension
            @test_throws DimensionMismatch quadratic_interp((x, y), bad_data2;
                bc=Right(QuadraticFit()))
        end

        @testset "unsupported BC" begin
            data = rand(11, 6)
            @test_throws ArgumentError quadratic_interp((x, y), data;
                bc=PeriodicBC())
        end
    end

    # ========================================
    # Show Methods
    # ========================================
    @testset "show methods" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi * yj for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @testset "compact show" begin
            str = sprint(show, itp)
            @test contains(str, "QuadraticInterpolantND")
        end

        @testset "verbose show" begin
            str = sprint(show, MIME("text/plain"), itp)
            @test contains(str, "QuadraticInterpolantND")
        end
    end

    # ========================================
    # Val-based Accessors
    # ========================================
    @testset "Val-based accessors" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 2.0, 15)
        data = rand(10, 15)
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @test FastInterpolations._grid(itp, Val(1)) === itp.grids[1]
        @test FastInterpolations._grid(itp, Val(2)) === itp.grids[2]
        @test FastInterpolations._spacing(itp, Val(1)) isa FastInterpolations.AbstractGridSpacing
        @test FastInterpolations._bc(itp, Val(1)) isa FastInterpolations.AbstractBC
        @test FastInterpolations._extrap(itp, Val(1)) isa Val
        @test FastInterpolations._search(itp, Val(1)) isa FastInterpolations.AbstractSearchPolicy
    end

    # ========================================
    # Field accessors (direct tuple access)
    # ========================================
    @testset "field tuple accessors" begin
        x = range(0.0, 1.0, 10)
        y = range(0.0, 2.0, 15)
        data = rand(10, 15)
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        @test itp.grids isa Tuple
        @test length(itp.grids) == 2

        @test length(itp.spacings) == 2
        @test length(itp.extraps) == 2
        @test length(itp.searches) == 2
    end

    # ========================================
    # Zero-Allocation One-Shot Tests
    # ========================================
    #
    # Each test uses a full function barrier: setup + warmup + @allocated
    # all inside one function. This avoids @testset-scope boxing artifacts.

    function _alloc_test_quadratic_default()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query)
        quadratic_interp((x, y), data, query)
        @allocated quadratic_interp((x, y), data, query)
    end

    function _alloc_test_quadratic_deriv()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; deriv=1)
        quadratic_interp((x, y), data, query; deriv=1)
        @allocated quadratic_interp((x, y), data, query; deriv=1)
    end

    function _alloc_test_quadratic_deriv_val()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; deriv=Val((1, 0)))
        quadratic_interp((x, y), data, query; deriv=Val((1, 0)))
        @allocated quadratic_interp((x, y), data, query; deriv=Val((1, 0)))
    end

    function _alloc_test_quadratic_natural_bc()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; bc=NaturalBC())
        quadratic_interp((x, y), data, query; bc=NaturalBC())
        @allocated quadratic_interp((x, y), data, query; bc=NaturalBC())
    end

    function _alloc_test_quadratic_extrap_constant()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; extrap=:constant)
        quadratic_interp((x, y), data, query; extrap=:constant)
        @allocated quadratic_interp((x, y), data, query; extrap=:constant)
    end

    function _alloc_test_quadratic_3d()
        x = range(0.0, 2.0, 10)
        y = range(0.0, 1.0, 8)
        z = range(0.0, 3.0, 6)
        data = [xi^2 + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        quadratic_interp((x, y, z), data, query)
        quadratic_interp((x, y, z), data, query)
        @allocated quadratic_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot" begin
        @testset "zero-alloc scalar (Range grids, default BC)" begin
            @test _alloc_test_quadratic_default() == 0
        end

        @testset "zero-alloc scalar (Range grids, deriv=1)" begin
            @test _alloc_test_quadratic_deriv() == 0
        end

        @testset "zero-alloc scalar (Range grids, deriv=Val)" begin
            @test _alloc_test_quadratic_deriv_val() == 0
        end

        @testset "zero-alloc scalar (Range grids, NaturalBC)" begin
            @test _alloc_test_quadratic_natural_bc() == 0
        end

        @testset "zero-alloc scalar (Range grids, extrap=:constant)" begin
            @test _alloc_test_quadratic_extrap_constant() == 0
        end

        @testset "zero-alloc scalar (3D Range grids)" begin
            @test _alloc_test_quadratic_3d() == 0
        end
    end
end
