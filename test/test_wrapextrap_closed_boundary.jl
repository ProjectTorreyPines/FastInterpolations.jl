# ════════════════════════════════════════════════════════════════════════════
# WrapExtrap closed-domain `[first(x), last(x)]` semantic guards
#
# After the closed-domain conversion (PR refac/wrap_closed), queries at
# exactly `last(x)` are treated as in-domain boundary queries (return
# `y[end]`) instead of wrapping to `first(x)`. This file pins down that
# behavior across 1D methods × extrap policies × grid types, plus the ND
# axis-level analog and the :exclusive/:inclusive PeriodicBC invariants.
# ════════════════════════════════════════════════════════════════════════════

@testitem "Closed boundary — Linear 1D × every extrap × every grid type" begin
    using FastInterpolations
    for x in (collect(range(0.0, 1.0, 11)), range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        # Scalar — every extrap at exact last(x) returns y[end]
        @test linear_interp(x, y, 1.0; extrap = WrapExtrap()) == y[end]
        @test linear_interp(x, y, 1.0; extrap = ClampExtrap()) == y[end]
        @test linear_interp(x, y, 1.0; extrap = FillExtrap(-1.0)) == y[end]
        @test linear_interp(x, y, 1.0; extrap = NoExtrap()) == y[end]
        # Batch (exercises the linear_oneshot fast path)
        @test linear_interp(x, y, [1.0]; extrap = WrapExtrap())[1] == y[end]
        @test linear_interp(x, y, [0.5, 1.0]; extrap = WrapExtrap())[end] == y[end]
        # Strictly OOB still wraps to the correct in-domain cell
        @test linear_interp(x, y, 1.25; extrap = WrapExtrap()) ≈
            linear_interp(x, y, 0.25; extrap = NoExtrap())  atol = 1.0e-12
    end
end

@testitem "Closed boundary — Constant 1D × every extrap" begin
    using FastInterpolations
    x = collect(range(0.0, 1.0, 11))
    y = collect(1.0:11.0)
    @test constant_interp(x, y, 1.0; extrap = WrapExtrap()) == y[end]
    @test constant_interp(x, y, [1.0]; extrap = WrapExtrap())[1] == y[end]
    @test constant_interp(x, y, 1.0; extrap = ClampExtrap()) == y[end]
end

@testitem "Closed boundary — Cubic 1D × WrapExtrap × every grid type" begin
    using FastInterpolations
    for x in (collect(range(0.0, 1.0, 11)), range(0.0, 1.0, 11))
        y = sin.(x)
        @test cubic_interp(x, y, 1.0; extrap = WrapExtrap()) ≈ y[end] atol = 1.0e-12
        @test cubic_interp(x, y, [1.0]; extrap = WrapExtrap())[1] ≈ y[end] atol = 1.0e-12
    end
end

@testitem "Closed boundary — Quadratic 1D × WrapExtrap" begin
    using FastInterpolations
    x = collect(range(0.0, 1.0, 11))
    y = x .^ 2
    @test quadratic_interp(x, y, 1.0; extrap = WrapExtrap()) ≈ y[end] atol = 1.0e-12
    @test quadratic_interp(x, y, [1.0]; extrap = WrapExtrap())[1] ≈ y[end] atol = 1.0e-12
end

@testitem "Closed boundary — Hermite (precomputed dy) × WrapExtrap" begin
    using FastInterpolations
    x = collect(range(0.0, 1.0, 11))
    y = sin.(x)
    dy = cos.(x)
    @test hermite_interp(x, y, dy, 1.0; extrap = WrapExtrap()) ≈ y[end] atol = 1.0e-12
    @test hermite_interp(x, y, dy, [1.0]; extrap = WrapExtrap())[1] ≈ y[end] atol = 1.0e-12
end

@testitem "Closed boundary — Hermite OnTheFly (PCHIP/Cardinal/Akima) × WrapExtrap" begin
    using FastInterpolations
    x = collect(range(0.0, 1.0, 11))
    y = sin.(x)
    @test pchip_interp(x, y, 1.0; extrap = WrapExtrap()) ≈ y[end] atol = 1.0e-12
    @test cardinal_interp(x, y, 1.0; tension = 0.0, extrap = WrapExtrap()) ≈ y[end] atol = 1.0e-12
    @test akima_interp(x, y, 1.0; extrap = WrapExtrap()) ≈ y[end] atol = 1.0e-12
end

@testitem "Closed boundary — :inclusive PeriodicBC still returns y[1] (= y[end])" begin
    using FastInterpolations
    x = collect(range(0.0, 2π, 11))
    y = sin.(x)
    y[end] = y[1]   # enforce closed cycle
    # Validation guarantees y[1] == y[end], so closed eval at xq=last(x)
    # (search returns last interval, α=1, eval = y[end]) gives the same value
    # as the old wrap-to-x_min path (search returns interval 1, α=0).
    @test cubic_interp(x, y, 2π; bc = PeriodicBC()) ≈ y[1] atol = 1.0e-12
    @test cubic_interp(x, y, [2π]; bc = PeriodicBC())[1] ≈ y[1] atol = 1.0e-12
    @test linear_interp(x, y, 2π; bc = PeriodicBC()) ≈ y[1] atol = 1.0e-12
end

@testitem "Closed boundary — :exclusive PeriodicBC: seam search returns y[1] without wrap" begin
    using FastInterpolations
    x = range(0.5, step = 1.0, length = 3)       # FVM cell-centered grid
    y = [10.0, 20.0, 30.0]
    bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
    # Virtual endpoint x[1] + period = 3.5
    # Old (half-open): _wrap_to_domain(3.5) → 0.5 → search → y[1]=10.0
    # New (closed):    no wrap; seam search → (n=3, idx_R=1, ...), α=1 → y[idx_R=1] = y[1]
    @test linear_interp(x, y, 3.5; bc = bc) == 10.0
    @test linear_interp(x, y, [3.5]; bc = bc)[1] == 10.0
    @test constant_interp(x, y, 3.5; bc = bc) == 10.0
end

@testitem "Closed boundary — :exclusive PeriodicBC series oneshot at virtual endpoint" begin
    using FastInterpolations
    # Pin the bug exposed by closed semantics: constant series oneshot
    # _constant_eval_at_anchor short-circuit was using y[end] (=y[n]) instead
    # of y[aq.idxR] (=y[1] for _ExclusivePeriodicAxis seam fold).
    x = collect(0.0:3.0)
    y1 = [10.0, 20.0, 30.0, 40.0]
    y2 = [1.0, 2.0, 3.0, 4.0]
    s = Series(y1, y2)
    bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
    out = constant_interp(x, s, 4.0; bc = bc)
    @test out[1] == y1[1]                                        # cyclic, NOT y1[end]
    @test out[2] == y2[1]
    # Series ↔ non-series alignment
    @test out[1] == constant_interp(x, y1, 4.0; bc = bc)
    @test out[2] == constant_interp(x, y2, 4.0; bc = bc)
end

@testitem "Closed boundary — ND per-axis WrapExtrap at last(grid)" begin
    using FastInterpolations
    gx = collect(range(0.0, 1.0, 11))
    gy = collect(range(0.0, 2.0, 21))
    data = [Float64(i + 10j) for i in 1:length(gx), j in 1:length(gy)]
    # Scalar query at the corner (last(gx), last(gy)) on BOTH axes
    val = interp(
        (gx, gy), data, (1.0, 2.0);
        method = (LinearInterp(), LinearInterp()),
        extrap = (WrapExtrap(), WrapExtrap()),
    )
    @test val == data[end, end]
    # Batch (SoA) query at the corner
    out = interp(
        (gx, gy), data, ([1.0], [2.0]);
        method = (LinearInterp(), LinearInterp()),
        extrap = (WrapExtrap(), WrapExtrap()),
    )
    @test out[1] == data[end, end]
end
