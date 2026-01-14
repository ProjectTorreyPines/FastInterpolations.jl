# test/test_cubic_series_naming.jl
# Phase 2: Test CubicSeriesInterpolant naming and backward compatibility

using Test
using FastInterpolations

@testset "CubicSeriesInterpolant naming" begin
    x = 0.0:0.1:1.0
    y1 = sin.(x)
    y2 = cos.(x)

    @testset "CubicSeriesInterpolant is exported" begin
        @test isdefined(FastInterpolations, :CubicSeriesInterpolant)
    end

    @testset "CubicSeriesInterpolant constructor works" begin
        sitp = cubic_interp(x, [y1, y2])
        @test sitp isa CubicSeriesInterpolant
    end

    @testset "Backward compat: CubicMultiInterpolant alias" begin
        sitp = cubic_interp(x, [y1, y2])
        @test sitp isa CubicMultiInterpolant  # Should still work
        @test CubicMultiInterpolant === CubicSeriesInterpolant
    end

    @testset "Backward compat: MultiCubicInterpolant alias" begin
        sitp = cubic_interp(x, [y1, y2])
        @test sitp isa MultiCubicInterpolant  # Should still work
        @test MultiCubicInterpolant === CubicSeriesInterpolant
    end

    @testset "AbstractSeriesInterpolant supertype" begin
        @test isdefined(FastInterpolations, :AbstractSeriesInterpolant)
        sitp = cubic_interp(x, [y1, y2])
        @test sitp isa AbstractSeriesInterpolant
    end

    @testset "AbstractMultiInterpolant backward compat" begin
        sitp = cubic_interp(x, [y1, y2])
        @test sitp isa AbstractMultiInterpolant
        @test AbstractMultiInterpolant === AbstractSeriesInterpolant
    end
end
