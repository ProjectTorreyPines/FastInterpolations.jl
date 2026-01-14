# test/test_series_interface.jl
# Phase A: Interface trait tests for SeriesInterpolant abstraction

using Test
using FastInterpolations
const FI = FastInterpolations

@testset "series_interface - Required Traits" begin

    # Create a minimal test type to verify trait error messages
    struct _TestSeriesInterpolant{T} <: FI.AbstractSeriesInterpolant{T} end

    @testset "n_series throws error for unimplemented" begin
        sitp = _TestSeriesInterpolant{Float64}()
        @test_throws ErrorException FI.n_series(sitp)
    end

    @testset "_should_wrap throws error for unimplemented" begin
        sitp = _TestSeriesInterpolant{Float64}()
        @test_throws ErrorException FI._should_wrap(sitp)
    end

    @testset "_get_grid throws error for unimplemented" begin
        sitp = _TestSeriesInterpolant{Float64}()
        @test_throws ErrorException FI._get_grid(sitp)
    end

    @testset "_get_extrap throws error for unimplemented" begin
        sitp = _TestSeriesInterpolant{Float64}()
        @test_throws ErrorException FI._get_extrap(sitp)
    end

    @testset "_make_anchor throws error for unimplemented" begin
        sitp = _TestSeriesInterpolant{Float64}()
        @test_throws ErrorException FI._make_anchor(sitp, 0.5)
    end

    @testset "_eval_series_at_anchor! throws error for unimplemented" begin
        sitp = _TestSeriesInterpolant{Float64}()
        output = zeros(2)
        @test_throws ErrorException FI._eval_series_at_anchor!(output, sitp, nothing, FI.EvalValue())
    end
end

@testset "series_interface - CubicSeriesInterpolant traits" begin
    # Verify existing CubicSeriesInterpolant implements all traits
    x = collect(0.0:0.1:1.0)
    ys = [sin.(2π .* x), cos.(2π .* x)]
    sitp = cubic_interp(x, ys)

    @testset "n_series returns correct count" begin
        @test FI.n_series(sitp) == 2
    end

    @testset "_should_wrap returns boolean" begin
        @test FI._should_wrap(sitp) isa Bool
    end

    @testset "_get_grid returns x vector" begin
        grid = FI._get_grid(sitp)
        @test grid ≈ x
    end

    @testset "_get_extrap returns ExtrapVal" begin
        extrap = FI._get_extrap(sitp)
        @test extrap isa FI.ExtrapVal
    end
end
