# test/test_series_utils.jl
# Phase A: Validation function tests for SeriesInterpolant infrastructure

@testitem "series_utils - Validation Functions" begin
    FI = FastInterpolations

    @testset "_validate_series_outputs" begin
        @testset "valid outputs pass" begin
            outputs = [zeros(10), zeros(10)]
            @test FI._validate_series_outputs(outputs, 2, 10) === nothing
        end

        @testset "wrong n_series throws DimensionMismatch" begin
            outputs = [zeros(10), zeros(10)]
            @test_throws DimensionMismatch FI._validate_series_outputs(outputs, 3, 10)
        end

        @testset "wrong output length throws DimensionMismatch" begin
            outputs = [zeros(10), zeros(5)]  # Second has wrong length
            @test_throws DimensionMismatch FI._validate_series_outputs(outputs, 2, 10)
        end
    end

    @testset "_validate_scalar_output" begin
        @testset "valid output passes" begin
            output = zeros(3)
            @test FI._validate_scalar_output(output, 3) === nothing
        end

        @testset "wrong length throws DimensionMismatch" begin
            output = zeros(2)
            @test_throws DimensionMismatch FI._validate_scalar_output(output, 3)
        end
    end
end

# ============================================================================
# Phase 1: Extrapolation Helper Tests
# ============================================================================
@testitem "series_utils - Extrapolation Helpers" begin
    FI = FastInterpolations

    @testset "_boundary_point_index" begin
        @testset "left side (FI.OOB_LEFT) returns 1" begin
            @test FI._boundary_point_index(FI.OOB_LEFT, 10) == 1
            @test FI._boundary_point_index(FI.OOB_LEFT, 100) == 1
            @test FI._boundary_point_index(FI.OOB_LEFT, 2) == 1
        end

        @testset "right side (FI.OOB_RIGHT) returns n_pts" begin
            @test FI._boundary_point_index(FI.OOB_RIGHT, 10) == 10
            @test FI._boundary_point_index(FI.OOB_RIGHT, 100) == 100
            @test FI._boundary_point_index(FI.OOB_RIGHT, 2) == 2
        end
    end

    @testset "_throw_extrap_domain_error" begin
        @testset "throws DomainError" begin
            @test_throws DomainError FI._throw_extrap_domain_error(-0.5, 0.0, 1.0)
            @test_throws DomainError FI._throw_extrap_domain_error(1.5, 0.0, 1.0)
        end

        @testset "error message contains domain bounds" begin
            err = try
                FI._throw_extrap_domain_error(-0.5, 0.0, 1.0)
                nothing
            catch e
                e
            end
            @test err isa DomainError
            @test occursin("0.0", string(err))
            @test occursin("1.0", string(err))
            @test occursin("outside domain", string(err))
        end
    end

    # The lean OOB helpers thread the query TYPE (not a stored `xq`): pass `Tq`.
    Tq = Float64

    # …plus the grid-unit deriv scale (`inv(oneunit(h))^N`), which the callers in
    # src compute from their grid. These kernels are unit-tested on a Float64 grid,
    # where the scale is the identity — wrap it so each call below stays readable.
    _sc(op) = FI._constant_axis_deriv_scale(1.0, op)
    boundary_value(y, side, n, k, op, ext, T) =
        FI._constant_extrap_boundary_value(y, side, n, k, op, ext, T, _sc(op))
    fill_simd!(out, y_point, side, n, op, ext, T) =
        FI._fill_constant_extrap_simd!(out, y_point, side, n, op, ext, T, _sc(op))

    @testset "_constant_extrap_boundary_value" begin
        # Test matrix: 5 points × 3 series
        y = [
            1.0 10.0 100.0;
            2.0 20.0 200.0;
            3.0 30.0 300.0;
            4.0 40.0 400.0;
            5.0 50.0 500.0
        ]
        n_pts = 5

        @testset "EvalValue returns boundary value" begin
            # Left boundary (index 1)
            @test boundary_value(y, FI.OOB_LEFT, n_pts, 1, FI.EvalValue(), ClampExtrap(), Tq) == 1.0
            @test boundary_value(y, FI.OOB_LEFT, n_pts, 2, FI.EvalValue(), ClampExtrap(), Tq) == 10.0
            @test boundary_value(y, FI.OOB_LEFT, n_pts, 3, FI.EvalValue(), ClampExtrap(), Tq) == 100.0

            # Right boundary (index n_pts)
            @test boundary_value(y, FI.OOB_RIGHT, n_pts, 1, FI.EvalValue(), ClampExtrap(), Tq) == 5.0
            @test boundary_value(y, FI.OOB_RIGHT, n_pts, 2, FI.EvalValue(), ClampExtrap(), Tq) == 50.0
            @test boundary_value(y, FI.OOB_RIGHT, n_pts, 3, FI.EvalValue(), ClampExtrap(), Tq) == 500.0
        end

        @testset "EvalDeriv1 returns zero" begin
            @test boundary_value(y, FI.OOB_LEFT, n_pts, 1, FI.EvalDeriv1(), ClampExtrap(), Tq) == 0.0
            @test boundary_value(y, FI.OOB_RIGHT, n_pts, 2, FI.EvalDeriv1(), ClampExtrap(), Tq) == 0.0
        end

        @testset "EvalDeriv2 returns zero" begin
            @test boundary_value(y, FI.OOB_LEFT, n_pts, 1, FI.EvalDeriv2(), ClampExtrap(), Tq) == 0.0
            @test boundary_value(y, FI.OOB_RIGHT, n_pts, 2, FI.EvalDeriv2(), ClampExtrap(), Tq) == 0.0
        end

        @testset "FillExtrap deriv: fill_value-as-data (NaN propagates, finite × 0 = 0)" begin
            # Cell-local "OOB cell = extrap's data" contract: deriv = 0 * fill_value.
            # NaN fill_value propagates through deriv; finite fill_value × 0 = 0.
            @test isnan(boundary_value(y, FI.OOB_LEFT, n_pts, 1, FI.EvalDeriv1(), FillExtrap(NaN), Tq))
            @test isnan(boundary_value(y, FI.OOB_RIGHT, n_pts, 2, FI.EvalDeriv1(), FillExtrap(NaN), Tq))
            @test boundary_value(y, FI.OOB_LEFT, n_pts, 1, FI.EvalDeriv1(), FillExtrap(99.0), Tq) == 0.0
        end
    end

    @testset "_fill_constant_extrap_simd!" begin
        # Test matrix: 3 series × 5 points (SIMD layout: y_point[series, point])
        # This matches the transposed layout used across all series interpolants
        y_point = [
            1.0 2.0 3.0 4.0 5.0;     # Series 1 at points 1-5
            10.0 20.0 30.0 40.0 50.0; # Series 2 at points 1-5
            100.0 200.0 300.0 400.0 500.0
        ]  # Series 3 at points 1-5
        n_pts = 5

        @testset "EvalValue fills boundary values" begin
            out = zeros(3)
            # Left boundary (point 1) - should fill column 1
            fill_simd!(out, y_point, FI.OOB_LEFT, n_pts, FI.EvalValue(), ClampExtrap(), Tq)
            @test out == [1.0, 10.0, 100.0]

            # Right boundary (point n_pts) - should fill column 5
            fill_simd!(out, y_point, FI.OOB_RIGHT, n_pts, FI.EvalValue(), ClampExtrap(), Tq)
            @test out == [5.0, 50.0, 500.0]
        end

        @testset "EvalDeriv1 fills zeros" begin
            out = ones(3)  # Start with non-zero to verify fill
            fill_simd!(out, y_point, FI.OOB_LEFT, n_pts, FI.EvalDeriv1(), ClampExtrap(), Tq)
            @test out == [0.0, 0.0, 0.0]

            out = ones(3)
            fill_simd!(out, y_point, FI.OOB_RIGHT, n_pts, FI.EvalDeriv1(), ClampExtrap(), Tq)
            @test out == [0.0, 0.0, 0.0]
        end

        @testset "EvalDeriv2 fills zeros" begin
            out = ones(3)
            fill_simd!(out, y_point, FI.OOB_LEFT, n_pts, FI.EvalDeriv2(), ClampExtrap(), Tq)
            @test out == [0.0, 0.0, 0.0]
        end

        @testset "FillExtrap deriv: fill_value-as-data (NaN propagates, finite → 0)" begin
            out = ones(3)
            fill_simd!(out, y_point, FI.OOB_LEFT, n_pts, FI.EvalDeriv1(), FillExtrap(NaN), Tq)
            @test all(isnan, out)

            out = ones(3)
            fill_simd!(out, y_point, FI.OOB_LEFT, n_pts, FI.EvalDeriv1(), FillExtrap(99.0), Tq)
            @test out == [0.0, 0.0, 0.0]
        end

        @testset "returns output vector" begin
            out = zeros(3)
            result = fill_simd!(out, y_point, FI.OOB_LEFT, n_pts, FI.EvalValue(), ClampExtrap(), Tq)
            @test result === out
        end
    end
end
