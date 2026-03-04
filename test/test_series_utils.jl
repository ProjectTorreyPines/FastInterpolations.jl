# test/test_series_utils.jl
# Phase A: Validation function tests for SeriesInterpolant infrastructure

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "series_utils - Validation Functions" begin

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
@testset "series_utils - Extrapolation Helpers" begin

    @testset "_boundary_point_index" begin
        @testset "left side (0x01) returns 1" begin
            @test FI._boundary_point_index(0x01, 10) == 1
            @test FI._boundary_point_index(0x01, 100) == 1
            @test FI._boundary_point_index(0x01, 2) == 1
        end

        @testset "right side (0x02) returns n_pts" begin
            @test FI._boundary_point_index(0x02, 10) == 10
            @test FI._boundary_point_index(0x02, 100) == 100
            @test FI._boundary_point_index(0x02, 2) == 2
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

    @testset "_constant_extrap_boundary_value" begin
        # Test matrix: 5 points × 3 series
        y = [1.0 10.0 100.0;
             2.0 20.0 200.0;
             3.0 30.0 300.0;
             4.0 40.0 400.0;
             5.0 50.0 500.0]
        n_pts = 5

        @testset "EvalValue returns boundary value" begin
            # Left boundary (index 1)
            @test FI._constant_extrap_boundary_value(y, 0x01, n_pts, 1, FI.EvalValue(), ConstExtrap()) == 1.0
            @test FI._constant_extrap_boundary_value(y, 0x01, n_pts, 2, FI.EvalValue(), ConstExtrap()) == 10.0
            @test FI._constant_extrap_boundary_value(y, 0x01, n_pts, 3, FI.EvalValue(), ConstExtrap()) == 100.0

            # Right boundary (index n_pts)
            @test FI._constant_extrap_boundary_value(y, 0x02, n_pts, 1, FI.EvalValue(), ConstExtrap()) == 5.0
            @test FI._constant_extrap_boundary_value(y, 0x02, n_pts, 2, FI.EvalValue(), ConstExtrap()) == 50.0
            @test FI._constant_extrap_boundary_value(y, 0x02, n_pts, 3, FI.EvalValue(), ConstExtrap()) == 500.0
        end

        @testset "EvalDeriv1 returns zero" begin
            @test FI._constant_extrap_boundary_value(y, 0x01, n_pts, 1, FI.EvalDeriv1(), ConstExtrap()) == 0.0
            @test FI._constant_extrap_boundary_value(y, 0x02, n_pts, 2, FI.EvalDeriv1(), ConstExtrap()) == 0.0
        end

        @testset "EvalDeriv2 returns zero" begin
            @test FI._constant_extrap_boundary_value(y, 0x01, n_pts, 1, FI.EvalDeriv2(), ConstExtrap()) == 0.0
            @test FI._constant_extrap_boundary_value(y, 0x02, n_pts, 2, FI.EvalDeriv2(), ConstExtrap()) == 0.0
        end
    end

    @testset "_fill_constant_extrap_simd!" begin
        # Test matrix: 3 series × 5 points (SIMD layout: y_point[series, point])
        # This matches the transposed layout used across all series interpolants
        y_point = [1.0 2.0 3.0 4.0 5.0;     # Series 1 at points 1-5
                   10.0 20.0 30.0 40.0 50.0; # Series 2 at points 1-5
                   100.0 200.0 300.0 400.0 500.0]  # Series 3 at points 1-5
        n_pts = 5

        @testset "EvalValue fills boundary values" begin
            out = zeros(3)
            # Left boundary (point 1) - should fill column 1
            FI._fill_constant_extrap_simd!(out, y_point, 0x01, n_pts, FI.EvalValue(), ConstExtrap())
            @test out == [1.0, 10.0, 100.0]

            # Right boundary (point n_pts) - should fill column 5
            FI._fill_constant_extrap_simd!(out, y_point, 0x02, n_pts, FI.EvalValue(), ConstExtrap())
            @test out == [5.0, 50.0, 500.0]
        end

        @testset "EvalDeriv1 fills zeros" begin
            out = ones(3)  # Start with non-zero to verify fill
            FI._fill_constant_extrap_simd!(out, y_point, 0x01, n_pts, FI.EvalDeriv1(), ConstExtrap())
            @test out == [0.0, 0.0, 0.0]

            out = ones(3)
            FI._fill_constant_extrap_simd!(out, y_point, 0x02, n_pts, FI.EvalDeriv1(), ConstExtrap())
            @test out == [0.0, 0.0, 0.0]
        end

        @testset "EvalDeriv2 fills zeros" begin
            out = ones(3)
            FI._fill_constant_extrap_simd!(out, y_point, 0x01, n_pts, FI.EvalDeriv2(), ConstExtrap())
            @test out == [0.0, 0.0, 0.0]
        end

        @testset "returns output vector" begin
            out = zeros(3)
            result = FI._fill_constant_extrap_simd!(out, y_point, 0x01, n_pts, FI.EvalValue(), ConstExtrap())
            @test result === out
        end
    end
end
