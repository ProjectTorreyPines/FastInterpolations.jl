# test/test_cubic_series_naming.jl
# Test CubicSeriesInterpolant naming and type hierarchy

@testitem "CubicSeriesInterpolant naming" begin
    x = 0.0:0.1:1.0
    y1 = sin.(x)
    y2 = cos.(x)

    @testset "CubicSeriesInterpolant is exported" begin
        @test isdefined(FastInterpolations, :CubicSeriesInterpolant)
    end

    @testset "CubicSeriesInterpolant constructor works" begin
        sitp = cubic_interp(x, Series(y1, y2))
        @test sitp isa CubicSeriesInterpolant
    end

    @testset "AbstractSeriesInterpolant supertype" begin
        @test isdefined(FastInterpolations, :AbstractSeriesInterpolant)
        sitp = cubic_interp(x, Series(y1, y2))
        @test sitp isa AbstractSeriesInterpolant
    end
end
