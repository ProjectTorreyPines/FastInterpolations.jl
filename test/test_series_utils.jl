# test/test_series_utils.jl
# Phase A: Validation function tests for SeriesInterpolant infrastructure

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "series_utils - Validation Functions" begin

    @testset "_validate_series_inputs" begin
        x = collect(0.0:0.1:1.0)  # 11 points
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        @testset "valid inputs pass" begin
            # Should not throw
            @test FI._validate_series_inputs(x, [y1, y2]) === nothing
        end

        @testset "empty ys throws ArgumentError" begin
            @test_throws ArgumentError FI._validate_series_inputs(x, Vector{Float64}[])
        end

        @testset "mismatched length throws DimensionMismatch" begin
            y_short = y1[1:5]
            @test_throws DimensionMismatch FI._validate_series_inputs(x, [y1, y_short])
        end

        @testset "error message includes series index" begin
            y_short = y1[1:5]
            try
                FI._validate_series_inputs(x, [y1, y_short])
                @test false  # Should not reach here
            catch e
                @test occursin("2", string(e))  # Series index 2
            end
        end
    end

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
