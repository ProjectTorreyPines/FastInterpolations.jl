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
        for xq in [0.75, 0.9, 1.0 - 1.0e-10]
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

@testitem "_ExclusivePeriodicAxis period validation (matches y-endpoint check)" begin
    using FastInterpolations: _ExclusivePeriodicAxis

    # Period validation for Range inner mirrors `_check_periodic_endpoints`:
    #   AbstractFloat → atol = 8*eps(T), rtol = sqrt(eps(T))
    #   Integer / Rational → default isapprox (effectively ==)
    # Float32 with a 0.03% off period was previously accepted under the old
    # `sqrt(eps(Tg))`-only rtol; the merged contract now also enforces atol so
    # the seam-cell width error stays ≤ ULP-scale.

    @testset "Float64 — accepts ULP-level rounding" begin
        x = range(0.0, step = 0.1, length = 10)
        # Period within sqrt(eps(Float64)) ≈ 1.5e-8: should pass
        _ExclusivePeriodicAxis(x, 1.0 + 1.0e-10)
        @test true   # no throw
    end

    @testset "Float64 — rejects gross mismatch" begin
        x = range(0.0, step = 0.1, length = 10)
        @test_throws ArgumentError _ExclusivePeriodicAxis(x, 2.0)
        @test_throws ArgumentError _ExclusivePeriodicAxis(x, 0.5)
    end

    @testset "Float32 — atol bound prevents near-zero false-pass" begin
        x = range(0.0f0, step = 0.1f0, length = 10)
        # Exact period passes
        _ExclusivePeriodicAxis(x, 1.0f0)
        @test true
        # 1 ULP off should still pass (atol = 8*eps(Float32))
        _ExclusivePeriodicAxis(x, nextfloat(1.0f0))
        @test true
    end

    @testset "Integer grid + Integer period — exact equality required" begin
        x = 0:1:5   # step=1, length=6, inferred period = 6
        _ExclusivePeriodicAxis(x, 6)
        @test true
        # Off-by-one should fail (default isapprox is essentially ==)
        @test_throws ArgumentError _ExclusivePeriodicAxis(x, 7)
    end

    @testset "Vector inner — no validation (period unverifiable)" begin
        x = [0.0, 0.1, 0.3, 0.6]   # non-uniform — no canonical period
        _ExclusivePeriodicAxis(x, 1.0)   # any period accepted
        @test true
        _ExclusivePeriodicAxis(x, 999.0)
        @test true
    end
end

@testitem "_ExclusivePeriodicAxis bounds contract (@boundscheck)" begin
    using FastInterpolations: _ExclusivePeriodicAxis

    # Valid indices are 1:n+1 (n = length(inner)), with i==n+1 returning the
    # virtual seam coord. Anything outside that must throw BoundsError so
    # off-by-one bugs and negative-index `@inbounds` paths cannot read
    # arbitrary memory.
    n = 5
    x = collect(0.0:0.2:0.8)   # length 5, period 1.0
    g = _ExclusivePeriodicAxis(x, 1.0)

    @testset "valid range 1:n+1" begin
        for i in 1:(n + 1)
            @test g[i] == (i <= n ? x[i] : g._x_max)
        end
    end

    @testset "out-of-range throws BoundsError" begin
        @test_throws BoundsError g[n + 2]
        @test_throws BoundsError g[100]
        @test_throws BoundsError g[0]
        @test_throws BoundsError g[-3]
    end

    @testset "@inbounds elides the check (no error path taken)" begin
        # The function must remain `@propagate_inbounds`, so callers wrapping
        # in `@inbounds` see zero overhead and no error even on the seam slot.
        f(g, i) = @inbounds g[i]
        @test f(g, 1) == 0.0
        @test f(g, n + 1) == g._x_max
    end
end

@testitem "_ExclusivePeriodicAxis search_interval seam fold for GridIdx" begin
    using FastInterpolations: _ExclusivePeriodicAxis, GridIdx, AutoSearch,
        _resolve_search, search_interval

    # search_interval(::Searcher, ::_ExclusivePeriodicAxis, ::GridIdx) must
    # apply the same seam-fold contract as the Real-query path: when the
    # resolved cell is at the seam (`idx == n_inner`), `idx_R` must be `1`
    # (post-fold), NOT the virtual `n+1`. Otherwise downstream eval kernels
    # using `_raw(y)[idx_R]` would read past the raw inner's length-n bounds.
    n = 5   # length of the raw inner
    x = collect(0.0:0.2:0.8)   # length 5
    g = _ExclusivePeriodicAxis(x, 1.0)
    @test length(g) == n + 1   # virtual length includes seam
    searcher = _resolve_search(g, 0.5, AutoSearch(), nothing)

    @testset "interior GridIdx — idx_R = idx + 1" begin
        for k in 1:(n - 1)
            idx, idx_R, _, _ = search_interval(searcher, g, GridIdx(k))
            @test idx == k
            @test idx_R == k + 1
        end
    end

    @testset "seam cell GridIdx(n) — idx_R folds to 1" begin
        idx, idx_R, xL, xR = search_interval(searcher, g, GridIdx(n))
        @test idx == n
        @test idx_R == 1     # post-fold (would be n+1 = 6 = OOB on _raw(y))
        @test xL == x[n]
        @test xR == g._x_max
    end

    @testset "virtual seam GridIdx(n+1) — clamped, still folds" begin
        # `_search_grididx` clamps `idx = min(xq.idx, n_virtual - 1) = n`,
        # so GridIdx(n+1) collapses to the seam cell rather than producing
        # idx_R = n+2.
        idx, idx_R, _, _ = search_interval(searcher, g, GridIdx(n + 1))
        @test idx == n
        @test idx_R == 1
    end
end

@testitem "GridIdx + periodic-exclusive end-to-end (no OOB on _raw(y))" begin
    using FastInterpolations

    # End-to-end smoke for the GridIdx seam-fold fix: build a 2D periodic-
    # exclusive cubic interpolant, query with GridIdx ranging over both
    # interior and seam cells of the periodic axis. With the fix, every
    # query returns a finite value; without it, GridIdx(n) / GridIdx(n+1)
    # would access `_raw(y)[n+1]` which is past the raw inner length.
    n = 5
    x = collect(0.0:0.2:0.8)
    data2d = [sin(2π * xi) * cos(2π * yj) for xi in x, yj in x]
    bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
    itp = cubic_interp((x, x), data2d; bc = (bc, bc))

    @testset "interior + seam GridIdx queries are all finite" begin
        for k in 1:(n + 1)
            v = itp((GridIdx(k), 0.5))
            @test isfinite(v)
        end
    end

    @testset "GridIdx(n+1) on the periodic axis matches the cyclic GridIdx(1)" begin
        # The virtual seam endpoint is `inner[1] + period`, which under the
        # axis-as-truth contract equals the cyclic image of `inner[1]`.
        # Evaluating at the seam endpoint must equal evaluating at index 1
        # along the same other-axis coordinate.
        v_seam_end = itp((GridIdx(n + 1), 0.3))
        v_first = itp((GridIdx(1), 0.3))
        @test v_seam_end ≈ v_first  rtol = 1.0e-12
    end
end

@testitem "_ExclusivePeriodicAxis view (seam-containing range)" begin
    using FastInterpolations: _ExclusivePeriodicAxis

    n = 5
    x = collect(0.0:0.2:0.8)
    g = _ExclusivePeriodicAxis(x, 1.0)

    @testset "full range returns wrapper itself (identity)" begin
        v = @view g[1:end]
        @test v === g
    end

    @testset "interior partial range delegates to inner" begin
        v = @view g[2:(n - 1)]
        @test parent(v) === g.inner
        @test collect(v) == x[2:(n - 1)]
    end

    @testset "seam-containing partial range builds SubArray over wrapper" begin
        # `2:end` includes the virtual seam slot — must NOT delegate to
        # `view(g.inner, 2:end)` (which would BoundsError since g.inner has
        # length n while the range ends at n+1).
        v = @view g[2:end]
        @test length(v) == n   # = (n+1) - 2 + 1
        @test v[end] == g._x_max
        @test v[1] == x[2]
    end

    @testset "out-of-range view throws BoundsError" begin
        @test_throws BoundsError view(g, 0:5)
        @test_throws BoundsError view(g, 1:(n + 5))
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
        arr = [
            10.0 20.0;
            30.0 40.0;
            50.0 60.0;
            1.0 2.0
        ]   # 4×2, fold dim=1 with n_period=3
        _periodic_fold_axis!(arr, 1, 3)
        @test arr[1, :] == [11.0, 22.0]   # arr[1,:] += arr[4,:]
        @test arr[2:4, :] == [30.0 40.0; 50.0 60.0; 1.0 2.0]  # untouched
    end

    @testset "2D fold-back along dim=2" begin
        arr = [
            10.0 20.0 30.0 1.0;
            40.0 50.0 60.0 2.0
        ]   # 2×4, fold dim=2 with n_period=3
        _periodic_fold_axis!(arr, 2, 3)
        @test arr[:, 1] == [11.0, 42.0]   # arr[:,1] += arr[:,4]
        @test arr[:, 2:4] == [20.0 30.0 1.0; 50.0 60.0 2.0]  # untouched
    end
end

# Lock-down test for the `<: AbstractVector{T}` semantic contract:
# The wrapper is **internal API** that intentionally exposes a virtual
# `(n+1)`-length view. This test pins down what generic Base operations do
# with that contract — both the *intended* iteration/reduction behavior and
# the *materialization escape hatch* of `similar`/`copy`/broadcast (which
# fall back to plain `Vector{T}` of length `n+1`). If a future contributor
# is tempted to overload `Base.similar` / `Base.copy` to return a wrapper,
# these tests force that to be a deliberate, reviewed decision.
@testitem "_ExclusivePeriodicAxis — Base.AbstractVector contract (lock-down)" begin
    using FastInterpolations: _ExclusivePeriodicAxis

    x = [0.0, 1.0, 2.0, 3.0]
    g = _ExclusivePeriodicAxis(x, 4.0)
    n = length(x)

    @testset "Iteration covers virtual span (n+1, includes seam coord)" begin
        # `for v in g` and `collect` see the seam at index n+1 with value
        # `inner[1] + period`. INTENDED — eval kernels rely on this uniformity.
        v = collect(g)
        @test length(v) == n + 1
        @test v[1:n] == x
        @test v[n + 1] ≈ x[1] + 4.0
    end

    @testset "Reductions span n+1 (seam coord included)" begin
        # `sum(g)` / `maximum(g)` iterate the virtual extension. For an
        # `_ExclusivePeriodicAxis`, the seam coord is `inner[1] + period`,
        # so `sum(g) == sum(inner) + (inner[1] + period)`. INTENDED.
        @test sum(g) == sum(x) + (x[1] + 4.0)
        @test maximum(g) == x[1] + 4.0   # seam = 4.0, larger than inner max=3.0
        @test minimum(g) == x[1]          # 0.0
    end

    @testset "similar/copy materialize as plain Vector (n+1)" begin
        # Default `similar(::AbstractVector{T}, ::Dims)` returns
        # `Vector{T}(undef, length)`. We document this: any code that calls
        # `similar(g)` or `copy(g)` gets a NON-cyclic `Vector` of length n+1
        # — the wrapper's zero-copy property is lost at that point. Caller's
        # responsibility to use `_raw(g)` or `g.inner` for raw-length buffers.
        s = similar(g)
        @test s isa Vector{Float64}
        @test length(s) == n + 1

        c = copy(g)
        @test c isa Vector{Float64}
        @test length(c) == n + 1
        @test c[1:n] == x
        @test c[n + 1] ≈ x[1] + 4.0
    end

    @testset "Broadcast materializes virtual span as plain Vector" begin
        # `g .+ 0.0` runs through the default AbstractArray broadcast path,
        # which produces a `Vector{T}` of length `n+1`. This is the same
        # rule as `similar`/`copy` — internal-API wrappers are not preserved
        # under broadcast. INTENDED escape hatch for diagnostics; do NOT use
        # in performance-critical paths.
        b = g .+ 0.0
        @test b isa Vector{Float64}
        @test length(b) == n + 1
        @test b[n + 1] ≈ x[1] + 4.0
    end

    @testset "Equality and hash differ from inner" begin
        # `g == g.inner` is `false` because lengths differ (5 vs 4). Wrappers
        # are NOT interchangeable with their inner under `==` / `Set` / `Dict`.
        @test g != g.inner
        @test hash(g) != hash(g.inner)
        # But two wrappers with the same inner+period agree.
        g2 = _ExclusivePeriodicAxis(x, 4.0)
        @test g == g2
    end
end
