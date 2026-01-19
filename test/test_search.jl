using Test
using FastInterpolations

@testset "Search Module - Phase 1 Smoke Tests" begin
    # These tests will FAIL until search.jl is implemented

    @testset "Types Exist" begin
        # Verify types are accessible (will fail: UndefVarError)
        @test isdefined(FastInterpolations, :SearchPolicy)
        @test isdefined(FastInterpolations, :BinaryAlg)
        @test isdefined(FastInterpolations, :NoHint)
        @test isdefined(FastInterpolations, :DEFAULT_SEARCH_POLICY)
    end

    @testset "Basic Dispatch" begin
        x = collect(range(0.0, 1.0, 101))
        policy = FastInterpolations.DEFAULT_SEARCH_POLICY

        # This will fail until search_interval is implemented
        idx, xL, xR = FastInterpolations.search_interval(policy, x, 0.5)
        @test idx == 51
        @test xL ≈ 0.50 atol=1e-12
        @test xR ≈ 0.51 atol=1e-12
    end

    @testset "Backward Compatibility" begin
        x = collect(range(0.0, 1.0, 101))

        # _find_interval should still work (alias chain)
        idx1, xL1, xR1 = FastInterpolations._find_interval(x, 0.5)

        # _search_interval should produce same result
        idx2, xL2, xR2 = FastInterpolations._search_interval(x, 0.5)

        @test idx1 == idx2
        @test xL1 == xL2
        @test xR1 == xR2
    end

    @testset "Type Inference (search_interval)" begin
        x_vec = collect(range(0.0, 1.0, 101))
        x_range = range(0.0, 1.0, 101)
        policy = FastInterpolations.DEFAULT_SEARCH_POLICY

        # Must be fully inferred: Tuple{Int64, Float64, Float64}
        @test @inferred(FastInterpolations.search_interval(policy, x_vec, 0.5)) isa Tuple{Int, Float64, Float64}
        @test @inferred(FastInterpolations.search_interval(policy, x_range, 0.5)) isa Tuple{Int, Float64, Float64}
    end
end
