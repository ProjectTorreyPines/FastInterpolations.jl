@testitem "_ExclusivePeriodicAxis construction + interface" begin
    using FastInterpolations: _ExclusivePeriodicAxis

    @testset "Vector inner — basic round-trip" begin
        x = [0.0, 0.25, 0.5, 0.75]
        g = _ExclusivePeriodicAxis(x, 1.0)

        @test g isa _ExclusivePeriodicAxis{Float64, Vector{Float64}, Float64}
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

        # i = n+1 returns the virtual seam coord `inner[1] + period` via cyclic Base.getindex.
        @test g[5] == g[1] + g.period
    end

    @testset "_CachedVector inner" begin
        using FastInterpolations: _CachedVector
        x = [0.0, 0.25, 0.5, 0.75]
        cv = _CachedVector(x)
        g = _ExclusivePeriodicAxis(cv, 1.0)

        @test g isa _ExclusivePeriodicAxis{Float64, _CachedVector{Float64, Float64}, Float64}
        @test length(g) == 5
        @test eltype(g) == Float64

        for i in 1:4
            @test g[i] == x[i]
        end
    end

    @testset "Float32 grid + Float32 period" begin
        x = Float32[0.0, 0.5, 1.0, 1.5]
        g = _ExclusivePeriodicAxis(x, Float32(2.0))
        @test g isa _ExclusivePeriodicAxis{Float32, Vector{Float32}, Float32}
        @test eltype(g) == Float32
    end

    @testset "Different period type than grid (e.g. Int period)" begin
        x = [0.0, 1.0, 2.0]
        g = _ExclusivePeriodicAxis(x, 3)  # Int period
        @test g.period == 3
        @test g.period isa Int
    end

    @testset "AbstractRange / _CachedRange inner is accepted (axis-as-truth design)" begin
        # `_ExclusivePeriodicAxis` now accepts any `AbstractVector` inner so
        # the same wrapper unifies Vector and Range periodic-exclusive paths.
        # The constructor only validates `inner[end] < inner[1] + period`.
        r = 0.0:0.25:0.75
        g_r = _ExclusivePeriodicAxis(r, 1.0)
        @test length(g_r) == 5  # virtual n+1
        @test last(g_r) == 1.0  # = inner[1] + period

        cr = FastInterpolations._to_float(r, Float64)  # _CachedRange
        g_cr = _ExclusivePeriodicAxis(cr, 1.0)
        @test length(g_cr) == 5
        @test last(g_cr) == 1.0
    end
end

@testitem "_ExclusivePeriodicAxis helpers (_getindex)" begin
    # NOTE: cyclic-index `_resolve_idx` was retired in favor of
    # `_ExclusivePeriodicData` — the data wrapper handles the virtual `n+1`
    # slot automatically via `Base.getindex(data, n+1) = data.inner[1]`.
    # Eval kernels read `y[idx_R]` directly without explicit index resolution.
    using FastInterpolations: _ExclusivePeriodicAxis, _getindex

    x = [0.0, 0.25, 0.5, 0.75]
    g = _ExclusivePeriodicAxis(x, 1.0)
    n = length(x)  # = 4

    @testset "_getindex on _ExclusivePeriodicAxis" begin
        # Normal cells: forward to inner
        for i in 1:n
            @test _getindex(g, i) == x[i]
        end

        # Virtual point at i = n+1: x[1] + period (axis carries coord info)
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
end

@testitem "_ExclusivePeriodicAxis idx-aware _get_h / _get_inv_h" begin
    using FastInterpolations: _ExclusivePeriodicAxis, _CachedVector, _get_h, _get_inv_h

    x = [0.0, 0.25, 0.5, 0.75]   # h = [0.25, 0.25, 0.25], seam_h = 1.0 - 0.75 + 0.0 = 0.25
    period = 1.0

    @testset "Vector inner — normal cells delegate, seam cell computes from period" begin
        g = _ExclusivePeriodicAxis(x, period)

        # Normal cells (idx = 1..n-1): delegate to inner's on-the-fly diff
        @test _get_h(g, 1) ≈ 0.25
        @test _get_h(g, 2) ≈ 0.25
        @test _get_h(g, 3) ≈ 0.25
        @test _get_inv_h(g, 1) ≈ 4.0
        @test _get_inv_h(g, 3) ≈ 4.0

        # Seam cell (idx = n = 4): h = inner[1] + period - inner[n] = 0.25
        @test _get_h(g, 4) ≈ 0.25
        @test _get_inv_h(g, 4) ≈ 4.0
    end

    @testset "_CachedVector inner — normal cells use cached fast path" begin
        cv = _CachedVector(x)
        g = _ExclusivePeriodicAxis(cv, period)

        # Normal cells: cached lookup via inner._CachedVector
        @test _get_h(g, 1) === cv.h[1]
        @test _get_inv_h(g, 1) === cv.inv_h[1]

        # Seam cell: still computed on-the-fly (cv.h doesn't have seam slot)
        @test _get_h(g, 4) ≈ 0.25
        @test _get_inv_h(g, 4) ≈ 4.0
    end

    @testset "Asymmetric seam — period larger than data span" begin
        # x covers [0.0, 0.7], period = 1.0 → seam_h = 1.0 - 0.7 + 0.0 = 0.3
        x_asym = [0.0, 0.2, 0.5, 0.7]
        g = _ExclusivePeriodicAxis(x_asym, 1.0)

        # Normal cells
        @test _get_h(g, 1) ≈ 0.2
        @test _get_h(g, 2) ≈ 0.3
        @test _get_h(g, 3) ≈ 0.2

        # Seam cell — wider than any normal cell
        @test _get_h(g, 4) ≈ 0.3
        @test _get_inv_h(g, 4) ≈ 1 / 0.3
    end
end

@testitem "_ExclusivePeriodicAxis specialized search (PoC validation)" begin
    using FastInterpolations: _ExclusivePeriodicAxis, _search_binary

    @testset "Vector inner — _search_binary seam fast-path" begin
        x = collect(range(0.0, 0.75; length = 4))  # [0.0, 0.25, 0.5, 0.75], n=4
        g = _ExclusivePeriodicAxis(x, 1.0)

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

    @testset "_CachedVector inner — _search_binary delegates to inner.inner" begin
        using FastInterpolations: _CachedVector
        x = [0.0, 0.25, 0.5, 0.75]
        cv = _CachedVector(x)
        g = _ExclusivePeriodicAxis(cv, 1.0)

        # Normal: delegates to _search_binary(::_CachedVector — itself <:AbstractVector)
        idx, xL, xR = _search_binary(g, 0.3)
        @test idx == 2
        @test xL == 0.25
        @test xR == 0.5

        # Seam
        idx, xL, xR = _search_binary(g, 0.85)
        @test idx == 4
        @test xL == 0.75
        @test xR ≈ 1.0
    end
end

@testitem "_ExclusivePeriodicAxis search_interval 4-tuple entry" begin
    using FastInterpolations:
        _ExclusivePeriodicAxis, search_interval, _resolve_search,
        AutoSearch, NoBC, BinarySearch

    @testset "Vector grid + NoBC searcher" begin
        x = [0.0, 0.25, 0.5, 0.75]
        g = _ExclusivePeriodicAxis(x, 1.0)
        searcher = _resolve_search(g, 0.3, BinarySearch(), nothing, NoBC())

        # Normal cell
        idx, idx_R, xL, xR = search_interval(searcher, g, 0.3)
        @test idx == 2
        @test idx_R == 3
        @test xL ≈ 0.25
        @test xR ≈ 0.5

        # Seam cell — `_ExclusivePeriodicAxis` returns *post-fold* idx_R = 1
        # (so ND eval can read raw data without a separate cyclic wrapper).
        # The virtual n+1 endpoint is captured in `xR = first(g) + period`.
        idx, idx_R, xL, xR = search_interval(searcher, g, 0.85)
        @test idx == 4
        @test idx_R == 1         # post-fold (was: virtual 5)
        @test xL == 0.75
        @test xR ≈ 1.0           # virtual right endpoint = x[1] + period
    end
end

@testitem "_ExclusivePeriodicAxis + _ExclusivePeriodicData round-trip in eval-style usage" begin
    using FastInterpolations: _ExclusivePeriodicAxis, _ExclusivePeriodicData,
                              search_interval, _resolve_search, NoBC, BinarySearch

    # Simulate an eval kernel: search returns idx_R = n+1 at seam; the data
    # wrapper auto-cycles `y[n+1] → inner[1]` so kernels write `y[idx_R]`
    # uniformly without explicit index resolution.
    x = [0.0, 0.25, 0.5, 0.75]
    y_raw = [1.0, 2.0, 3.0, 4.0]
    g = _ExclusivePeriodicAxis(x, 1.0)
    y = _ExclusivePeriodicData(y_raw)
    searcher = _resolve_search(g, 0.0, BinarySearch(), nothing, NoBC())

    @testset "Normal cell — yL, yR straight from y[idx], y[idx_R]" begin
        idx, idx_R, _, _ = search_interval(searcher, g, 0.3)
        @test idx == 2 && idx_R == 3
        @test y[idx] == 2.0
        @test y[idx_R] == 3.0
    end

    @testset "Seam cell — idx_R = 1 (post-fold) reads y_raw[1] directly" begin
        idx, idx_R, xL, xR = search_interval(searcher, g, 0.85)
        @test idx == 4 && idx_R == 1            # post-fold (was: virtual 5)
        @test y[idx] == 4.0                     # last physical y
        @test y[idx_R] == 1.0                   # post-fold idx_R = 1 reads first physical y directly
        # The data wrapper still cycles `y[5]` → `y_raw[1]` for completeness,
        # but the post-fold idx_R means kernels never need that path on the
        # search-driven hot loop.
        @test y[5] == 1.0
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
