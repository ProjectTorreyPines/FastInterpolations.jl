@testitem "_ExclusivePeriodicGrid construction + interface" begin
    using FastInterpolations: _ExclusivePeriodicGrid

    @testset "Vector inner — basic round-trip" begin
        x = [0.0, 0.25, 0.5, 0.75]
        g = _ExclusivePeriodicGrid(x, 1.0)

        @test g isa _ExclusivePeriodicGrid{Float64, Vector{Float64}, Float64}
        @test g.inner === x
        @test g.period == 1.0

        # length reports virtual extension (n+1)
        @test length(g) == 5
        @test size(g) == (5,)
        @test firstindex(g) == 1
        @test lastindex(g) == 5

        # eltype
        @test eltype(g) == Float64

        # Base.getindex forwards to inner — i ≤ n
        for i in 1:4
            @test g[i] == x[i]
        end

        # i = n+1 raises BoundsError from inner (no virtual access via `[]`).
        @test_throws BoundsError g[5]
    end

    @testset "Range inner" begin
        x = 0.0:0.25:0.75   # length 4
        g = _ExclusivePeriodicGrid(x, 1.0)

        @test g isa _ExclusivePeriodicGrid{Float64}
        @test length(g) == 5
        @test eltype(g) == Float64

        for i in 1:4
            @test g[i] == x[i]
        end
    end

    @testset "Float32 grid + Float32 period" begin
        x = Float32[0.0, 0.5, 1.0, 1.5]
        g = _ExclusivePeriodicGrid(x, Float32(2.0))
        @test g isa _ExclusivePeriodicGrid{Float32, Vector{Float32}, Float32}
        @test eltype(g) == Float32
    end

    @testset "Different period type than grid (e.g. Int period)" begin
        x = [0.0, 1.0, 2.0]
        g = _ExclusivePeriodicGrid(x, 3)  # Int period
        @test g.period == 3
        @test g.period isa Int
    end
end

@testitem "_ExclusivePeriodicGrid helpers (_getindex, _resolve_idx)" begin
    using FastInterpolations: _ExclusivePeriodicGrid, _getindex, _resolve_idx

    x = [0.0, 0.25, 0.5, 0.75]
    g = _ExclusivePeriodicGrid(x, 1.0)
    n = length(x)  # = 4

    @testset "_getindex on _ExclusivePeriodicGrid" begin
        # Normal cells: forward to inner
        for i in 1:n
            @test _getindex(g, i) == x[i]
        end

        # Virtual point at i = n+1: x[1] + period
        @test _getindex(g, n + 1) == x[1] + 1.0
        @test _getindex(g, n + 1) ≈ 1.0
    end

    @testset "_getindex on plain AbstractVector — no-op forward" begin
        v = [1.0, 2.0, 3.0]
        for i in 1:3
            @test _getindex(v, i) == v[i]
        end

        # Range
        r = 0.0:0.5:2.0
        for i in 1:length(r)
            @test _getindex(r, i) == r[i]
        end
    end

    @testset "_resolve_idx on _ExclusivePeriodicGrid" begin
        # Normal indices: unchanged
        for i in 1:n
            @test _resolve_idx(i, g) == i
        end
        # Virtual index n+1 wraps to 1
        @test _resolve_idx(n + 1, g) == 1
    end

    @testset "_resolve_idx on plain AbstractVector — identity" begin
        v = [1.0, 2.0, 3.0]
        for i in 1:5
            @test _resolve_idx(i, v) == i
        end
    end
end

@testitem "_ExclusivePeriodicGrid specialized search (PoC validation)" begin
    using FastInterpolations: _ExclusivePeriodicGrid, _search_binary, _search_direct

    @testset "Vector inner — _search_binary seam fast-path" begin
        x = collect(range(0.0, 0.75; length = 4))  # [0.0, 0.25, 0.5, 0.75], n=4
        g = _ExclusivePeriodicGrid(x, 1.0)

        # Normal in-domain queries: should match raw inner search
        for xq in [0.05, 0.3, 0.55]
            idx_g, xL_g, xR_g = _search_binary(g, xq)
            idx_v, xL_v, xR_v = _search_binary(x, xq)
            @test idx_g == idx_v
            @test xL_g == xL_v
            @test xR_g == xR_v
        end

        # Seam queries (xq >= x[n]): early-exit, return (n, x[n], x[1]+period)
        for xq in [0.75, 0.9, 1.0 - 1e-10]
            idx, xL, xR = _search_binary(g, xq)
            @test idx == 4         # n
            @test xL == x[4]       # x[n]
            @test xR == x[1] + 1.0 # x[1] + period
        end
    end

    @testset "Range inner — _search_direct seam fast-path" begin
        x = 0.0:0.25:0.75   # length 4
        g = _ExclusivePeriodicGrid(x, 1.0)

        # Normal in-domain
        for xq in [0.05, 0.3, 0.55]
            idx_g, xL_g, xR_g = _search_direct(g, xq)
            idx_x, xL_x, xR_x = _search_direct(x, xq)
            @test idx_g == idx_x
            @test xL_g == xL_x
            @test xR_g == xR_x
        end

        # Seam case
        idx, xL, xR = _search_direct(g, 0.85)
        @test idx == 4
        @test xL == 0.75
        @test xR ≈ 1.0
    end

    @testset "_CachedRange inner (via _CachedRange <: AbstractRange dispatch)" begin
        using FastInterpolations: _CachedRange, _to_float
        cr = _to_float(0.0:0.25:0.75, Float64)
        g = _ExclusivePeriodicGrid(cr, 1.0)

        # Normal: delegates to _search_direct(::_CachedRange, ...)
        idx, xL, xR = _search_direct(g, 0.3)
        @test idx == 2
        @test xL ≈ 0.25
        @test xR ≈ 0.5

        # Seam
        idx, xL, xR = _search_direct(g, 0.95)
        @test idx == 4
        @test xL ≈ 0.75
        @test xR ≈ 1.0
    end
end

@testitem "_ExclusivePeriodicGrid search_interval 4-tuple entry" begin
    using FastInterpolations:
        _ExclusivePeriodicGrid, search_interval, _resolve_search,
        AutoSearch, NoBC, BinarySearch, _to_float

    @testset "Vector grid + NoBC searcher" begin
        x = [0.0, 0.25, 0.5, 0.75]
        g = _ExclusivePeriodicGrid(x, 1.0)
        searcher = _resolve_search(g, 0.3, BinarySearch(), nothing, NoBC())

        # Normal cell
        idx, idx_R, xL, xR = search_interval(searcher, g, 0.3)
        @test idx == 2
        @test idx_R == 3
        @test xL ≈ 0.25
        @test xR ≈ 0.5

        # Seam cell — idx_R is the *virtual* n+1
        idx, idx_R, xL, xR = search_interval(searcher, g, 0.85)
        @test idx == 4
        @test idx_R == 5         # virtual!
        @test xL == 0.75
        @test xR ≈ 1.0           # virtual right endpoint = x[1] + period
    end

    @testset "Range grid + NoBC searcher" begin
        r = 0.0:0.25:0.75
        g = _ExclusivePeriodicGrid(r, 1.0)
        searcher = _resolve_search(g, 0.5, BinarySearch(), nothing, NoBC())

        idx, idx_R, xL, xR = search_interval(searcher, g, 0.5)
        @test idx == 3
        @test idx_R == 4
        @test xL ≈ 0.5
        @test xR ≈ 0.75
    end
end

@testitem "_ExclusivePeriodicGrid + _resolve_idx round-trip in eval-style usage" begin
    using FastInterpolations: _ExclusivePeriodicGrid, search_interval, _resolve_idx,
                              _resolve_search, NoBC, BinarySearch

    # Simulate an eval kernel: use search_interval + _resolve_idx to access data
    x = [0.0, 0.25, 0.5, 0.75]
    y = [1.0, 2.0, 3.0, 4.0]
    g = _ExclusivePeriodicGrid(x, 1.0)
    searcher = _resolve_search(g, 0.0, BinarySearch(), nothing, NoBC())

    @testset "Normal cell — yL, yR straight from y[idx], y[idx_R]" begin
        idx, idx_R, _, _ = search_interval(searcher, g, 0.3)
        @test idx == 2 && idx_R == 3
        # _resolve_idx no-op for non-seam
        @test _resolve_idx(idx, g) == 2
        @test _resolve_idx(idx_R, g) == 3
        @test y[_resolve_idx(idx, g)] == 2.0
        @test y[_resolve_idx(idx_R, g)] == 3.0
    end

    @testset "Seam cell — idx_R = n+1 wraps to 1 via _resolve_idx" begin
        idx, idx_R, xL, xR = search_interval(searcher, g, 0.85)
        @test idx == 4 && idx_R == 5  # virtual
        @test _resolve_idx(idx, g) == 4
        @test _resolve_idx(idx_R, g) == 1   # cyclic wrap
        @test y[_resolve_idx(idx, g)] == 4.0
        @test y[_resolve_idx(idx_R, g)] == 1.0  # wraps to y[1]
    end
end

@testitem "_periodic_fold_axis! — adjoint seam fold-back" begin
    using FastInterpolations: _periodic_fold_axis!

    @testset "1D fold-back (single axis)" begin
        # arr is length n_period+1 = 5; fold arr[5] into arr[1]
        arr = [10.0, 20.0, 30.0, 40.0, 5.0]
        n_period = 4
        _periodic_fold_axis!(arr, 1, n_period)
        @test arr[1] == 15.0   # 10 + 5
        @test arr[2:5] == [20.0, 30.0, 40.0, 5.0]  # untouched
    end

    @testset "2D fold-back along dim=1" begin
        arr = [10.0 20.0;
               30.0 40.0;
               50.0 60.0;
               1.0 2.0]   # 4×2, fold dim=1 with n_period=3
        _periodic_fold_axis!(arr, 1, 3)
        @test arr[1, :] == [11.0, 22.0]   # arr[1,:] += arr[4,:]
        @test arr[2:4, :] == [30.0 40.0; 50.0 60.0; 1.0 2.0]  # untouched
    end

    @testset "2D fold-back along dim=2" begin
        arr = [10.0 20.0 30.0 1.0;
               40.0 50.0 60.0 2.0]   # 2×4, fold dim=2 with n_period=3
        _periodic_fold_axis!(arr, 2, 3)
        @test arr[:, 1] == [11.0, 42.0]   # arr[:,1] += arr[:,4]
        @test arr[:, 2:4] == [20.0 30.0 1.0; 50.0 60.0 2.0]  # untouched
    end
end
