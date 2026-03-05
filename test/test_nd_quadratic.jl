# ========================================
# Tests for QuadraticInterpolantND
# ========================================
#
# Comprehensive test coverage for N-dimensional quadratic interpolation.
# Tests cover: polynomial reproduction, derivatives, batch queries,
# extrapolation, grid types, complex values, BCs, and error handling.

using Test
using FastInterpolations
using FastInterpolations: get_task_local_pool

# Allocation threshold (bytes) — tolerates minor LTS/GC overhead.
# Guarded for standalone execution (runtests.jl defines this globally).
if !@isdefined(ND_ALLOC_THRESHOLD)
    const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
end

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
        @test itp((xq, yq); deriv=DerivOp(0, 0)) ≈ f(xq, yq) rtol=1e-10

        # First derivatives: df/dx = 2x, df/dy = 2y
        @test itp((xq, yq); deriv=DerivOp(1, 0)) ≈ 2xq rtol=1e-10
        @test itp((xq, yq); deriv=DerivOp(0, 1)) ≈ 2yq rtol=1e-10

        # Second derivatives: d²f/dx² = 2, d²f/dy² = 2
        @test itp((xq, yq); deriv=DerivOp(2, 0)) ≈ 2.0 rtol=1e-6
        @test itp((xq, yq); deriv=DerivOp(0, 2)) ≈ 2.0 rtol=1e-6

        # Mixed derivative: d²f/dxdy = 0 for x²+y²
        @test abs(itp((xq, yq); deriv=DerivOp(1, 1))) < 1e-10

        # DerivOp derivative spec (compile-time)
        @test itp((xq, yq); deriv=DerivOp(1, 0)) ≈ 2xq rtol=1e-10
        @test itp((xq, yq); deriv=DerivOp(0, 1)) ≈ 2yq rtol=1e-10
        @test itp((xq, yq); deriv=DerivOp(2, 0)) ≈ 2.0 rtol=1e-6

        # Integer derivative (all axes same order)
        @test itp((xq, yq); deriv=DerivOp(1, 1)) isa Float64
        @test itp((xq, yq); deriv=DerivOp(2, 2)) isa Float64
    end

    @testset "2D non-zero mixed derivative" begin
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 11)
        f(xi, yi) = xi^2 + xi*yi + yi^2
        data = [f(xi, yi) for xi in x, yi in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))

        xq, yq = 1.0, 0.5

        # d²f/dxdy = 1 for x² + xy + y²
        @test itp((xq, yq); deriv=DerivOp(1, 1)) ≈ 1.0 atol=1e-6
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

        @testset "ZeroCurvBC()" begin
            data_sin = [sin(xi) * cos(yi) for xi in x, yi in y]
            itp = quadratic_interp((x, y), data_sin; bc=ZeroCurvBC())
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
                bc=Right(QuadraticFit()), search=BinarySearch())
            @test itp((1.0, 0.5)) ≈ f(1.0, 0.5) rtol=1e-10
        end

        @testset "mixed search policies" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()),
                search=(BinarySearch(), LinearBinarySearch{4}()))
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

        @testset "extrap=NoExtrap() (default)" begin
            itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
            @test_throws DomainError itp((-0.1, 0.5))
            @test_throws DomainError itp((0.5, -0.1))
            @test_throws DomainError itp((2.1, 0.5))
            @test_throws DomainError itp((0.5, 1.1))
        end

        @testset "extrap=ClampedExtrap()" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()), extrap=ClampedExtrap())
            @test itp((-0.1, 0.5)) ≈ itp((0.0, 0.5))
            @test itp((2.1, 0.5)) ≈ itp((2.0, 0.5))
            @test itp((0.5, 1.1)) ≈ itp((0.5, 1.0))
            @test itp((0.5, -0.1)) ≈ itp((0.5, 0.0))
        end

        @testset "extrap=ExtendExtrap()" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()), extrap=ExtendExtrap())
            @test isfinite(itp((-0.1, 0.5)))
            @test isfinite(itp((2.5, 1.5)))
        end

        @testset "per-axis extrap" begin
            itp = quadratic_interp((x, y), data;
                bc=Right(QuadraticFit()),
                extrap=(ClampedExtrap(), ExtendExtrap()))
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
            dvals = itp((xs, ys); deriv=DerivOp(1, 0))
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
            dvals = itp(queries; deriv=DerivOp(1, 0))
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
                bc=Right(QuadraticFit()), deriv=DerivOp(1, 0))
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
        @test itp((xq, yq, zq); deriv=DerivOp(1, 0, 0)) ≈ 2xq rtol=1e-6
        @test itp((xq, yq, zq); deriv=DerivOp(0, 1, 0)) ≈ 2yq rtol=1e-6
        @test itp((xq, yq, zq); deriv=DerivOp(0, 0, 1)) ≈ 2zq rtol=1e-6

        # Second derivatives
        @test itp((xq, yq, zq); deriv=DerivOp(2, 0, 0)) ≈ 2.0 rtol=1e-4
        @test itp((xq, yq, zq); deriv=DerivOp(0, 2, 0)) ≈ 2.0 rtol=1e-4
        @test itp((xq, yq, zq); deriv=DerivOp(0, 0, 2)) ≈ 2.0 rtol=1e-4
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
        @test FastInterpolations._extrap(itp, Val(1)) isa FastInterpolations.AbstractExtrap
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
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 1))
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 1))
        @allocated quadratic_interp((x, y), data, query; deriv=DerivOp(1, 1))
    end

    function _alloc_test_quadratic_deriv_val()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 0))
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 0))
        @allocated quadratic_interp((x, y), data, query; deriv=DerivOp(1, 0))
    end

    function _alloc_test_quadratic_natural_bc()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; bc=ZeroCurvBC())
        quadratic_interp((x, y), data, query; bc=ZeroCurvBC())
        @allocated quadratic_interp((x, y), data, query; bc=ZeroCurvBC())
    end

    function _alloc_test_quadratic_extrap_constant()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; extrap=ClampedExtrap())
        quadratic_interp((x, y), data, query; extrap=ClampedExtrap())
        @allocated quadratic_interp((x, y), data, query; extrap=ClampedExtrap())
    end

    function _alloc_test_quadratic_extrap_wrap_periodic()
        x = range(0.0, 2π, 21)
        y = range(0.0, 2π, 21)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        quadratic_interp((x, y), data, query; bc=ZeroCurvBC(), extrap=WrapExtrap())
        quadratic_interp((x, y), data, query; bc=ZeroCurvBC(), extrap=WrapExtrap())
        @allocated quadratic_interp((x, y), data, query; bc=ZeroCurvBC(), extrap=WrapExtrap())
    end

    function _alloc_test_quadratic_mixed_mode()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; extrap=(NoExtrap(), ClampedExtrap()))
        quadratic_interp((x, y), data, query; extrap=(NoExtrap(), ClampedExtrap()))
        @allocated quadratic_interp((x, y), data, query; extrap=(NoExtrap(), ClampedExtrap()))
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
            @test _alloc_test_quadratic_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, deriv=DerivOp(1, 1))" begin
            @test _alloc_test_quadratic_deriv() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, deriv=Val)" begin
            @test _alloc_test_quadratic_deriv_val() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, ZeroCurvBC)" begin
            @test _alloc_test_quadratic_natural_bc() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=ClampedExtrap())" begin
            @test _alloc_test_quadratic_extrap_constant() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=WrapExtrap+PeriodicBC)" begin
            @test _alloc_test_quadratic_extrap_wrap_periodic() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, per-axis mixed Mode)" begin
            @test _alloc_test_quadratic_mixed_mode() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Range grids)" begin
            @test _alloc_test_quadratic_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Mixed-Grid Allocation Tests (Range + Vector)
    # ========================================
    #
    # Heterogeneous grid tuples (ScalarSpacing + VectorSpacing) must be zero-allocation.
    # Catches ntuple closure boxing on heterogeneous inputs.

    function _alloc_test_quadratic_mixed_2d()
        x = range(0.0, 2.0, 20)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 15)) # Vector → VectorSpacing
        data = [xi^2 + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query)
        quadratic_interp((x, y), data, query)
        @allocated quadratic_interp((x, y), data, query)
    end

    function _alloc_test_quadratic_mixed_3d()
        x = range(0.0, 2.0, 10)          # Range → ScalarSpacing
        y = collect(range(0.0, 1.0, 8))  # Vector → VectorSpacing
        z = range(0.0, 3.0, 6)           # Range → ScalarSpacing
        data = [xi^2 + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        quadratic_interp((x, y, z), data, query)
        quadratic_interp((x, y, z), data, query)
        @allocated quadratic_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot (Mixed grids: Range + Vector)" begin
        @testset "zero-alloc scalar (2D mixed grid)" begin
            @test _alloc_test_quadratic_mixed_2d() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D mixed grid)" begin
            @test _alloc_test_quadratic_mixed_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Vector-Grid Allocation Tests
    # ========================================
    #
    # Pool-based spacing: VectorSpacing h/inv_h acquired from pool,
    # zero heap allocation for Vector grids after warmup.

    function _alloc_test_quadratic_vector_default()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query)
        quadratic_interp((x, y), data, query)
        @allocated quadratic_interp((x, y), data, query)
    end

    function _alloc_test_quadratic_vector_deriv()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 0))
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 0))
        @allocated quadratic_interp((x, y), data, query; deriv=DerivOp(1, 0))
    end

    function _alloc_test_quadratic_vector_deriv_int()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi^2 + yj^2 for xi in x, yj in y]
        query = (1.0, 0.5)
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 1))
        quadratic_interp((x, y), data, query; deriv=DerivOp(1, 1))
        @allocated quadratic_interp((x, y), data, query; deriv=DerivOp(1, 1))
    end

    function _alloc_test_quadratic_vector_3d()
        x = collect(range(0.0, 2.0, 10))
        y = collect(range(0.0, 1.0, 8))
        z = collect(range(0.0, 3.0, 6))
        data = [xi^2 + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        quadratic_interp((x, y, z), data, query)
        quadratic_interp((x, y, z), data, query)
        @allocated quadratic_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot (Vector grids)" begin
        @testset "zero-alloc scalar (Vector grids, default BC)" begin
            @test _alloc_test_quadratic_vector_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Vector grids, deriv=Val)" begin
            @test _alloc_test_quadratic_vector_deriv() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Vector grids, deriv=DerivOp(1) Int)" begin
            @test _alloc_test_quadratic_vector_deriv_int() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Vector grids)" begin
            @test _alloc_test_quadratic_vector_3d() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # In-Place Batch Allocation Tests
    # ========================================
    #
    # In-place paths write into a pre-allocated output buffer.
    # These must be truly zero-allocation (only output + THRESHOLD).

    function _alloc_test_quadratic_inplace_soa()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        itp(out, (xqs, yqs))
        itp(out, (xqs, yqs))
        @allocated itp(out, (xqs, yqs))
    end

    function _alloc_test_quadratic_inplace_aos()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; bc=Right(QuadraticFit()))
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        itp(out, points)
        itp(out, points)
        @allocated itp(out, points)
    end

    @testset "In-Place Batch Allocation Tests" begin
        @testset "in-place SoA batch (Range grids)" begin
            @test _alloc_test_quadratic_inplace_soa() <= ND_ALLOC_THRESHOLD
        end

        @testset "in-place AoS batch (Range grids)" begin
            @test _alloc_test_quadratic_inplace_aos() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place API (quadratic_interp!)
    # ========================================

    @testset "Oneshot In-Place (quadratic_interp!)" begin
        @testset "SoA correctness" begin
            x = range(0.0, 2.0, 20)
            y = range(0.0, 1.0, 15)
            data = [xi^2 + yj for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5]
            yqs = [0.2, 0.5, 0.8]
            ref = quadratic_interp((x, y), data, (xqs, yqs))
            out = similar(ref)
            quadratic_interp!(out, (x, y), data, (xqs, yqs))
            @test out ≈ ref atol=1e-14
        end

        @testset "AoS correctness" begin
            x = range(0.0, 2.0, 20)
            y = range(0.0, 1.0, 15)
            data = [xi^2 + yj for xi in x, yj in y]
            points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
            ref = quadratic_interp((x, y), data, points)
            out = similar(ref)
            quadratic_interp!(out, (x, y), data, points)
            @test out ≈ ref atol=1e-14
        end

        @testset "SoA with deriv" begin
            x = range(0.0, 2.0, 20)
            y = range(0.0, 1.0, 15)
            data = [xi^2 + yj for xi in x, yj in y]
            xqs = [0.5, 1.0, 1.5]
            yqs = [0.2, 0.5, 0.8]
            ref = quadratic_interp((x, y), data, (xqs, yqs); deriv=DerivOp(1, 1))
            out = similar(ref)
            quadratic_interp!(out, (x, y), data, (xqs, yqs); deriv=DerivOp(1, 1))
            @test out ≈ ref atol=1e-14
        end

        @testset "DimensionMismatch on wrong output length" begin
            x = range(0.0, 1.0, 10)
            y = range(0.0, 1.0, 10)
            data = [xi + yj for xi in x, yj in y]
            xqs = [0.5, 0.6, 0.7]
            yqs = [0.5, 0.6, 0.7]
            out = zeros(5)
            @test_throws DimensionMismatch quadratic_interp!(out, (x, y), data, (xqs, yqs))
        end
    end

    function _alloc_test_oneshot_inplace_soa_quadratic()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        quadratic_interp!(out, (x, y), data, (xqs, yqs))
        quadratic_interp!(out, (x, y), data, (xqs, yqs))
        @allocated quadratic_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_quadratic()
        x = range(0.0, 2.0, 20)
        y = range(0.0, 1.0, 15)
        data = [xi^2 + yj for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        quadratic_interp!(out, (x, y), data, points)
        quadratic_interp!(out, (x, y), data, points)
        @allocated quadratic_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests" begin
        @testset "oneshot in-place SoA (Range grids)" begin
            @test _alloc_test_oneshot_inplace_soa_quadratic() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Range grids)" begin
            @test _alloc_test_oneshot_inplace_aos_quadratic() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Oneshot In-Place Allocation Tests (Vector grids)
    # ========================================

    function _alloc_test_oneshot_inplace_soa_quadratic_vec()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi^2 + yj for xi in x, yj in y]
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.2, 0.5, 0.8]
        out = Vector{Float64}(undef, 3)
        quadratic_interp!(out, (x, y), data, (xqs, yqs))
        quadratic_interp!(out, (x, y), data, (xqs, yqs))
        @allocated quadratic_interp!(out, (x, y), data, (xqs, yqs))
    end

    function _alloc_test_oneshot_inplace_aos_quadratic_vec()
        x = collect(range(0.0, 2.0, 20))
        y = collect(range(0.0, 1.0, 15))
        data = [xi^2 + yj for xi in x, yj in y]
        points = [(0.5, 0.2), (1.0, 0.5), (1.5, 0.8)]
        out = Vector{Float64}(undef, 3)
        quadratic_interp!(out, (x, y), data, points)
        quadratic_interp!(out, (x, y), data, points)
        @allocated quadratic_interp!(out, (x, y), data, points)
    end

    @testset "Oneshot In-Place Allocation Tests (Vector grids)" begin
        @testset "oneshot in-place SoA (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_soa_quadratic_vec() <= ND_ALLOC_THRESHOLD
        end

        @testset "oneshot in-place AoS (Vector grids)" begin
            @test _alloc_test_oneshot_inplace_aos_quadratic_vec() <= ND_ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Pool Rewind Verification
    # ========================================

    @testset "Pool rewind after oneshot (quadratic)" begin
        xv = collect(range(0.0, 2.0, 15))
        yv = collect(range(0.0, 1.0, 11))
        data = [xi^2 + yi^2 for xi in xv, yi in yv]
        query = (1.0, 0.5)
        xqs = [0.3, 0.7, 1.1, 1.5, 1.9]
        yqs = [0.1, 0.3, 0.5, 0.7, 0.9]
        pts = [(xqs[i], yqs[i]) for i in 1:5]

        # Warmup
        quadratic_interp((xv, yv), data, query)
        quadratic_interp((xv, yv), data, (xqs, yqs))
        quadratic_interp((xv, yv), data, pts)
        out = Vector{Float64}(undef, 5)
        quadratic_interp!(out, (xv, yv), data, (xqs, yqs))
        quadratic_interp!(out, (xv, yv), data, pts)

        pool = get_task_local_pool()

        @testset "scalar oneshot" begin
            n_before = pool.float64.n_active
            quadratic_interp((xv, yv), data, query)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA batch oneshot" begin
            n_before = pool.float64.n_active
            quadratic_interp((xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS batch oneshot" begin
            n_before = pool.float64.n_active
            quadratic_interp((xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end

        @testset "SoA in-place oneshot" begin
            n_before = pool.float64.n_active
            quadratic_interp!(out, (xv, yv), data, (xqs, yqs))
            @test pool.float64.n_active == n_before
        end

        @testset "AoS in-place oneshot" begin
            n_before = pool.float64.n_active
            quadratic_interp!(out, (xv, yv), data, pts)
            @test pool.float64.n_active == n_before
        end
    end
end
