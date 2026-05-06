# BC-aware `search_interval` 4-tuple unit tests.
#
# Exercises the seam-wrap contract that periodic oneshot/persistent paths rely
# on. Functional/end-to-end tests cover steady-state behavior; this file pins
# down the boundary semantics (xq at x[n], xq just inside last cell, xq beyond
# x[n]) so off-by-one regressions surface immediately.

@testitem "search_interval — 4-tuple BC dispatch" begin
    using FastInterpolations: search_interval, Searcher, BinarySearch, NoHint,
        NoBC, PeriodicBC

    @testset "NoBC: idx_R = idx_L + 1, no seam" begin
        # Both Range and Vector grids
        for x in (range(0.0, 1.0, length = 5), collect(range(0.0, 1.0, length = 5)))
            s = Searcher{BinarySearch, NoHint, NoBC}(NoHint(), NoBC())
            for xq in (0.1, 0.5, 0.9)
                idx_L, idx_R, xL, xR = search_interval(s, x, xq)
                @test idx_R == idx_L + 1
                @test xL == x[idx_L]
                @test xR == x[idx_R]
                @test xL ≤ xq < xR
            end
        end
    end

    @testset "PeriodicBC{:inclusive}: same as NoBC (matched endpoints)" begin
        # Inclusive grid spans the full period — within the grid, no seam wrap.
        x = range(0.0, 2π, length = 5)
        bc = PeriodicBC()
        s = Searcher{BinarySearch, NoHint, typeof(bc)}(NoHint(), bc)

        for xq in (0.1, π, 2π - 0.1)
            idx_L, idx_R, xL, xR = search_interval(s, x, xq)
            @test idx_R == idx_L + 1
            @test x[idx_L] ≤ xq ≤ x[idx_R]
        end
    end

    @testset "PeriodicBC{:exclusive}: seam wrap at last cell" begin
        # Range grid: [0.0, 0.25, 0.5, 0.75], period 1.0 → seam cell [0.75, 1.0)
        x = range(0.0, step = 0.25, length = 4)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        s = Searcher{BinarySearch, NoHint, typeof(bc)}(NoHint(), bc)

        # Inside grid, not seam: standard pair
        idx_L, idx_R, xL, xR = search_interval(s, x, 0.4)
        @test (idx_L, idx_R) == (2, 3)
        @test (xL, xR) == (0.25, 0.5)

        # Just below last grid point: standard pair (n-1, n)
        idx_L, idx_R, xL, xR = search_interval(s, x, 0.74)
        @test (idx_L, idx_R) == (3, 4)
        @test (xL, xR) == (0.5, 0.75)

        # Boundary at xq == x[n]: contract puts this in the seam cell
        idx_L, idx_R, xL, xR = search_interval(s, x, 0.75)
        @test (idx_L, idx_R) == (4, 1)
        @test xL ≈ 0.75
        @test xR ≈ 1.0                                     # x[1] + period

        # Inside seam cell: idx_R wraps to 1, xR is virtual endpoint
        idx_L, idx_R, xL, xR = search_interval(s, x, 0.85)
        @test (idx_L, idx_R) == (4, 1)
        @test xL ≈ 0.75
        @test xR ≈ 1.0
    end

    @testset "PeriodicBC{:exclusive}: Vector grid seam wrap" begin
        x = [0.0, 0.25, 0.5, 0.75]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        s = Searcher{BinarySearch, NoHint, typeof(bc)}(NoHint(), bc)

        idx_L, idx_R, xL, xR = search_interval(s, x, 0.85)
        @test (idx_L, idx_R) == (4, 1)
        @test xL ≈ 0.75
        @test xR ≈ 1.0
    end

    @testset "Float32 grid: types preserved through seam" begin
        x = range(0.0f0, step = 0.25f0, length = 4)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0f0)
        s = Searcher{BinarySearch, NoHint, typeof(bc)}(NoHint(), bc)

        idx_L, idx_R, xL, xR = search_interval(s, x, 0.85f0)
        @test (idx_L, idx_R) == (4, 1)
        @test xL ≈ 0.75f0
        @test xR ≈ 1.0f0
        @test xL isa Float32
        @test xR isa Float32
    end

    @testset "Type stability — @inferred 4-tuple" begin
        x = range(0.0, 1.0, length = 5)
        s_nobc = Searcher{BinarySearch, NoHint, NoBC}(NoHint(), NoBC())
        bc_inc = PeriodicBC()
        s_inc = Searcher{BinarySearch, NoHint, typeof(bc_inc)}(NoHint(), bc_inc)
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        s_exc = Searcher{BinarySearch, NoHint, typeof(bc_exc)}(NoHint(), bc_exc)

        @test @inferred(search_interval(s_nobc, x, 0.5)) isa Tuple{Int, Int, Float64, Float64}
        @test @inferred(search_interval(s_inc, x, 0.5)) isa Tuple{Int, Int, Float64, Float64}
        @test @inferred(search_interval(s_exc, x, 0.5)) isa Tuple{Int, Int, Float64, Float64}
        # Seam path is the same return type
        @test @inferred(search_interval(s_exc, x, 0.99)) isa Tuple{Int, Int, Float64, Float64}
    end
end
