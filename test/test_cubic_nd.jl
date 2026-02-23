using Test
using FastInterpolations

# Allocation threshold (bytes) — tolerates minor LTS/GC overhead.
if !@isdefined(ND_ALLOC_THRESHOLD)
    const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
end

@testset "ND Cubic Interpolation (2D)" begin

    @testset "Basic 2D Interpolation" begin
        # Create a simple test function: f(x,y) = sin(x) * cos(y)
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # Create interpolant
        itp = cubic_interp((x, y), data)

        # Test type
        @test itp isa CubicInterpolantND
        @test ndims(itp) == 2
        @test size(itp) == (21, 11)

        # Test grid point pass-through
        for i in 1:length(x)
            for j in 1:length(y)
                @test itp((x[i], y[j])) ≈ data[i, j] atol=1e-12
            end
        end

        # Test interpolation at interior points
        xq, yq = 1.5, 0.8
        expected = sin(xq) * cos(yq)
        @test itp((xq, yq)) ≈ expected atol=1e-3  # Cubic interpolation accuracy

        # Test tuple input (same as above)
        @test itp((xq, yq)) ≈ expected atol=1e-3
    end

    @testset "Complex Value Support" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)

        # Complex-valued function: f(x,y) = sin(x)*cos(y) + i*cos(x)*sin(y)
        data = [sin(xi) * cos(yj) + im * cos(xi) * sin(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        # Check type parameters
        @test grid_type(itp) == Float64
        @test value_type(itp) == ComplexF64

        # Test grid point pass-through
        @test itp((x[5], y[3])) ≈ data[5, 3] atol=1e-12

        # Test interpolation
        xq, yq = 1.5, 0.8
        expected = sin(xq) * cos(yq) + im * cos(xq) * sin(yq)
        result = itp((xq, yq))
        @test result isa ComplexF64
        @test result ≈ expected atol=1e-3
    end

    @testset "Oneshot API" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # Single point oneshot
        xq, yq = 1.5, 0.8
        val = cubic_interp((x, y), data, (xq, yq))
        expected = sin(xq) * cos(yq)
        @test val ≈ expected atol=1e-3

        # Multiple points oneshot
        xqs = [0.5, 1.0, 1.5, 2.0]
        yqs = [0.2, 0.4, 0.6, 0.8]
        vals = cubic_interp((x, y), data, (xqs, yqs))
        @test length(vals) == 4
        for k in 1:4
            expected_k = sin(xqs[k]) * cos(yqs[k])
            @test vals[k] ≈ expected_k atol=2e-3
        end
    end

    @testset "Vector Evaluation" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        # Evaluate at multiple points
        xqs = collect(range(0.5, 2.0, 10))
        yqs = collect(range(0.2, 0.8, 10))
        vals = itp((xqs, yqs))

        @test length(vals) == 10
        for k in 1:10
            expected = sin(xqs[k]) * cos(yqs[k])
            @test vals[k] ≈ expected atol=2e-3
        end
    end

    @testset "Boundary Conditions" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        # ZeroCurvBC
        itp_natural = cubic_interp((x, y), data; bc=ZeroCurvBC())
        @test itp_natural((1.0, 0.5)) isa Float64

        # ZeroSlopeBC
        itp_clamped = cubic_interp((x, y), data; bc=ZeroSlopeBC())
        @test itp_clamped((1.0, 0.5)) isa Float64

        # Per-axis BC
        itp_mixed = cubic_interp((x, y), data; bc=(ZeroCurvBC(), ZeroSlopeBC()))
        @test itp_mixed((1.0, 0.5)) isa Float64

        # BCPair with specific derivatives
        itp_deriv = cubic_interp((x, y), data;
            bc=(BCPair(Deriv1(0.0), Deriv1(0.0)), ZeroCurvBC()))
        @test itp_deriv((1.0, 0.5)) isa Float64
    end

    @testset "CubicFit BC (Polynomial Endpoint Estimation)" begin
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data; bc=CubicFit())
        @test itp((1.0, 0.5)) isa Float64

        # Should give good accuracy at interior points
        xq, yq = 1.5, 0.8
        expected = sin(xq) * cos(yq)
        @test itp((xq, yq)) ≈ expected atol=1e-3
    end

    @testset "Extrapolation Modes" begin
        x = range(0.0, 2.0, 11)
        y = range(0.0, 1.0, 6)
        data = [xi * yj for xi in x, yj in y]

        # :none (default) - should throw outside domain
        itp_none = cubic_interp((x, y), data; extrap=NoExtrap())
        @test_throws DomainError itp_none((-0.1, 0.5))
        @test_throws DomainError itp_none((0.5, -0.1))

        # :constant - clamp to boundary
        itp_const = cubic_interp((x, y), data; extrap=ConstExtrap())
        @test itp_const((-0.1, 0.5)) ≈ itp_const((0.0, 0.5))
        @test itp_const((2.1, 0.5)) ≈ itp_const((2.0, 0.5))
    end

    @testset "Derivative Evaluation" begin
        # f(x,y) = x^2 * y^2 → ∂f/∂x = 2xy², ∂f/∂y = 2x²y, ∂²f/∂x∂y = 4xy
        x = range(0.0, 2.0, 21)
        y = range(0.0, 1.0, 11)
        data = [xi^2 * yj^2 for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        xq, yq = 1.0, 0.5

        # Value
        @test itp((xq, yq); deriv=DerivOp(0, 0)) ≈ xq^2 * yq^2 atol=1e-4

        # ∂f/∂x = 2xy²
        dfdx_expected = 2 * xq * yq^2
        @test itp((xq, yq); deriv=DerivOp(1, 0)) ≈ dfdx_expected atol=1e-3

        # ∂f/∂y = 2x²y
        dfdy_expected = 2 * xq^2 * yq
        @test itp((xq, yq); deriv=DerivOp(0, 1)) ≈ dfdy_expected atol=1e-3

        # ∂²f/∂x∂y = 4xy
        d2fdxdy_expected = 4 * xq * yq
        @test itp((xq, yq); deriv=DerivOp(1, 1)) ≈ d2fdxdy_expected atol=1e-2
    end

    @testset "Non-uniform Grids" begin
        # Non-uniform grid spacing
        x = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
        y = [0.0, 0.2, 0.5, 1.0]
        data = [sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        # Test grid point pass-through
        @test itp((x[3], y[2])) ≈ data[3, 2] atol=1e-12

        # Test interpolation
        xq, yq = 0.5, 0.3
        expected = sin(xq) * cos(yq)
        @test itp((xq, yq)) ≈ expected atol=1e-3
    end

    @testset "Float32 Support" begin
        x = range(0.0f0, 2.0f0, 11)
        y = range(0.0f0, 1.0f0, 6)
        data = Float32[sin(xi) * cos(yj) for xi in x, yj in y]

        itp = cubic_interp((x, y), data)

        @test grid_type(itp) == Float32
        @test value_type(itp) == Float32

        result = itp((1.0f0, 0.5f0))
        @test result isa Float32
        @test result ≈ sin(1.0f0) * cos(0.5f0) atol=5e-4
    end

    @testset "Type Introspection" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = rand(11, 11)

        itp = cubic_interp((x, y), data)

        @test grid_type(itp) == Float64
        @test value_type(itp) == Float64
        @test ndims(itp) == 2
        @test size(itp) == (11, 11)
        @test length(axes(itp)) == 2
    end

    @testset "Strategy Types" begin
        # PreCompute is default
        @test PreCompute() isa AbstractCoeffStrategy
        @test OnTheFly() isa AbstractCoeffStrategy

        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 11)
        data = rand(11, 11)

        # Explicit PreCompute
        itp = cubic_interp((x, y), data; coeffs=PreCompute())
        @test itp isa CubicInterpolantND

        # OnTheFly should throw (not implemented)
        @test_throws ArgumentError cubic_interp((x, y), data; coeffs=OnTheFly())
    end

    @testset "Dimension Mismatch Errors" begin
        x = range(0.0, 1.0, 11)
        y = range(0.0, 1.0, 6)
        data = rand(10, 6)  # Wrong x dimension

        @test_throws DimensionMismatch cubic_interp((x, y), data)

        data2 = rand(11, 5)  # Wrong y dimension
        @test_throws DimensionMismatch cubic_interp((x, y), data2)
    end

    @testset "PolyFit BC with insufficient grid points" begin
        # CubicFit is PolyFit{3} — requires at least 4 grid points.
        # Using 3 points triggers the ArgumentError in _validate_nd_bcs!.
        y = range(0.0, 1.0, 11)

        # Dim-1 too short (3 points < 4 required for CubicFit)
        x_short = range(0.0, 1.0, 3)
        data_short = [xi + yj for xi in x_short, yj in y]
        @test_throws ArgumentError cubic_interp((x_short, y), data_short; bc=CubicFit())

        # Dim-2 too short (per-axis BC tuple)
        x = range(0.0, 1.0, 11)
        y_short = range(0.0, 1.0, 3)
        data_short2 = [xi + yj for xi in x, yj in y_short]
        @test_throws ArgumentError cubic_interp((x, y_short), data_short2;
            bc=(ZeroCurvBC(), CubicFit()))

        # Oneshot path also validates (triggers _validate_nd_bcs! in cubic_interp)
        @test_throws ArgumentError cubic_interp(
            (x_short, y), data_short, (0.5, 0.5); bc=CubicFit()
        )
    end

    # ========================================
    # Zero-Allocation One-Shot Tests
    # ========================================
    #
    # Each test uses a full function barrier: setup + warmup + @allocated
    # all inside one function. This avoids @testset-scope boxing artifacts.

    function _alloc_test_cubic_default()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query)
        cubic_interp((x, y), data, query)
        @allocated cubic_interp((x, y), data, query)
    end

    function _alloc_test_cubic_deriv()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query; deriv=DerivOp(1, 1))
        cubic_interp((x, y), data, query; deriv=DerivOp(1, 1))
        @allocated cubic_interp((x, y), data, query; deriv=DerivOp(1, 1))
    end

    function _alloc_test_cubic_deriv_val()
        x = range(0.0, 2π, 21)
        y = range(0.0, π, 11)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query; deriv=DerivOp(1, 0))
        cubic_interp((x, y), data, query; deriv=DerivOp(1, 0))
        @allocated cubic_interp((x, y), data, query; deriv=DerivOp(1, 0))
    end

    function _alloc_test_cubic_extrap_const()
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi^2 + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; extrap=ConstExtrap())
        cubic_interp((x, y), data, query; extrap=ConstExtrap())
        @allocated cubic_interp((x, y), data, query; extrap=ConstExtrap())
    end

    function _alloc_test_cubic_extrap_extend()
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi^2 + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; extrap=ExtendExtrap())
        cubic_interp((x, y), data, query; extrap=ExtendExtrap())
        @allocated cubic_interp((x, y), data, query; extrap=ExtendExtrap())
    end

    function _alloc_test_cubic_extrap_wrap_periodic()
        x = range(0.0, 2π, 21)
        y = range(0.0, 2π, 21)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        query = (1.5, 0.8)
        cubic_interp((x, y), data, query; bc=PeriodicBC(), extrap=WrapExtrap())
        cubic_interp((x, y), data, query; bc=PeriodicBC(), extrap=WrapExtrap())
        @allocated cubic_interp((x, y), data, query; bc=PeriodicBC(), extrap=WrapExtrap())
    end

    function _alloc_test_cubic_mixed_mode()
        x = range(0.0, 2.0, 15)
        y = range(0.0, 1.0, 10)
        data = [xi^2 + yj for xi in x, yj in y]
        query = (1.0, 0.5)
        cubic_interp((x, y), data, query; extrap=(NoExtrap(), ConstExtrap()))
        cubic_interp((x, y), data, query; extrap=(NoExtrap(), ConstExtrap()))
        @allocated cubic_interp((x, y), data, query; extrap=(NoExtrap(), ConstExtrap()))
    end

    function _alloc_test_cubic_periodic_exclusive()
        x = range(0.0, step=0.1, length=20)
        y = range(0.0, step=0.2, length=10)
        data = [sin(2π*xi) * cos(2π*yj) for xi in x, yj in y]
        query = (0.5, 0.5)
        bc = PeriodicBC(; endpoint=:exclusive, period=2.0)
        cubic_interp((x, y), data, query; bc=bc, extrap=WrapExtrap())
        cubic_interp((x, y), data, query; bc=bc, extrap=WrapExtrap())
        @allocated cubic_interp((x, y), data, query; bc=bc, extrap=WrapExtrap())
    end

    function _alloc_test_cubic_3d()
        x = range(0.0, 2.0, 10)
        y = range(0.0, 1.0, 8)
        z = range(0.0, 3.0, 6)
        data = [xi^2 + yj + zk for xi in x, yj in y, zk in z]
        query = (1.0, 0.5, 1.5)
        cubic_interp((x, y, z), data, query)
        cubic_interp((x, y, z), data, query)
        @allocated cubic_interp((x, y, z), data, query)
    end

    @testset "Zero-Allocation One-Shot" begin
        @testset "zero-alloc scalar (Range grids, default)" begin
            @test _alloc_test_cubic_default() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, deriv=DerivOp(1, 1))" begin
            @test _alloc_test_cubic_deriv() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, deriv=Val)" begin
            @test _alloc_test_cubic_deriv_val() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=ConstExtrap)" begin
            @test _alloc_test_cubic_extrap_const() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (Range grids, extrap=ExtendExtrap)" begin
            @test _alloc_test_cubic_extrap_extend() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (PeriodicBC + WrapExtrap)" begin
            @test _alloc_test_cubic_extrap_wrap_periodic() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (per-axis mixed Mode)" begin
            @test _alloc_test_cubic_mixed_mode() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (PeriodicBC exclusive + WrapExtrap)" begin
            @test _alloc_test_cubic_periodic_exclusive() <= ND_ALLOC_THRESHOLD
        end

        @testset "zero-alloc scalar (3D Range grids)" begin
            @test _alloc_test_cubic_3d() <= ND_ALLOC_THRESHOLD
        end
    end

end
