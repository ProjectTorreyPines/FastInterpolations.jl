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

@testitem "Closed boundary — ND hetero corner × WrapExtrap (homo + mixed method tuples)" begin
    using FastInterpolations
    # Splines interpolate exactly at every node, so the corner query
    # `(last(gx), last(gy))` must return `data[end, end]` for every supported
    # ND method combination under per-axis WrapExtrap. Covers method-tuple
    # branches not exercised by the Linear×Linear corner above:
    #   - homo Cubic×Cubic, Quadratic×Quadratic
    #   - hetero Cubic×Linear (different families per axis)
    #   - 3D Cubic×Cubic×Cubic (N > 2 corner)
    #   - OnTheFly Hermite via Cardinal×Cardinal (cell-local `_collapse_dims` path)
    gx = collect(range(0.0, 1.0, 11))
    gy = collect(range(0.0, 2.0, 21))
    data = [sin(xi) + cos(yj) for xi in gx, yj in gy]
    corner = (1.0, 2.0)
    ext_both = (WrapExtrap(), WrapExtrap())
    expected = data[end, end]

    # Homogeneous high-order at the corner
    @test interp((gx, gy), data, corner; method = (CubicInterp(), CubicInterp()), extrap = ext_both) ≈
        expected atol = 1.0e-12
    @test interp(
        (gx, gy), data, corner;
        method = (QuadraticInterp(), QuadraticInterp()), extrap = ext_both
    ) ≈ expected atol = 1.0e-12

    # Hetero: different families per axis (oneshot and persistent)
    @test interp((gx, gy), data, corner; method = (CubicInterp(), LinearInterp()), extrap = ext_both) ≈
        expected atol = 1.0e-12
    let itp = interp(
            (gx, gy), data;
            method = (CubicInterp(), LinearInterp()),
            extrap = ext_both, coeffs = PreCompute(),
        )
        @test itp(corner) ≈ expected atol = 1.0e-12
    end

    # OnTheFly Hermite via Cardinal×Cardinal — exercises the cell-local
    # `_collapse_dims` window-aware path from PR #112.
    @test interp(
        (gx, gy), data, corner;
        method = (CardinalInterp(), CardinalInterp()),
        extrap = ext_both,
    ) ≈ expected atol = 1.0e-12
    let itp = interp(
            (gx, gy), data;
            method = (CardinalInterp(), CardinalInterp()),
            extrap = ext_both, coeffs = OnTheFly(),
        )
        @test itp(corner) ≈ expected atol = 1.0e-12
    end

    # 3D corner (N > 2): Cubic×Cubic×Cubic, all-WrapExtrap
    gz = collect(range(0.0, 0.5, 7))
    data3 = [sin(xi) + cos(yj) + zk for xi in gx, yj in gy, zk in gz]
    corner3 = (1.0, 2.0, 0.5)
    @test interp(
        (gx, gy, gz), data3, corner3;
        method = (CubicInterp(), CubicInterp(), CubicInterp()),
        extrap = (WrapExtrap(), WrapExtrap(), WrapExtrap()),
    ) ≈ data3[end, end, end] atol = 1.0e-12
end

@testitem "Closed boundary — persistent (callable) interpolant at last(x) × WrapExtrap" begin
    using FastInterpolations
    # Spline interpolates exactly at every node, so itp(last(x)) must return
    # y[end] for every method family. Pins the closed-domain contract on the
    # `_anchor_loc`-driven persistent path (distinct from the search-driven
    # oneshot path covered above).
    for x in (collect(range(0.0, 1.0, 11)), range(0.0, 1.0, 11))
        y = sin.(x)
        @test linear_interp(x, y; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-14
        @test constant_interp(x, y; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-14
        @test cubic_interp(x, y; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-12
        @test quadratic_interp(x, y; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-12
        @test pchip_interp(x, y; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-14
        @test cardinal_interp(x, y; tension = 0.0, extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-14
        @test akima_interp(x, y; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-14
        # Hermite with explicit slopes
        dy = cos.(x)
        @test hermite_interp(x, y, dy; extrap = WrapExtrap())(1.0) ≈ y[end] atol = 1.0e-14
    end
end

@testitem "Closed boundary — adjoint at last(x) lands at f_bar[end] × WrapExtrap" begin
    using FastInterpolations
    # Under closed semantics, xq == last(x) lands at α=1 in interval n-1.
    # Spline interpolates exactly at every node, so d itp(last(x)) / d y[i]
    # is the indicator δ_{i, end}. Old half-open semantics wrapped to first(x),
    # giving δ_{i, 1} — this test pins the gradient location.
    x = collect(range(0.0, 1.0, 11))
    y = sin.(x)
    y_bar = [1.0]

    # Linear: closed-form weights, exact at nodes
    let adj = linear_adjoint(x, [1.0]; extrap = WrapExtrap()), f_bar = adj(y_bar)
        @test f_bar[end] == 1.0
        @test f_bar[1] == 0.0
        @test all(f_bar[2:(end - 1)] .== 0.0)
    end
    # Cubic, Quadratic: data-independent adjoint (no y arg)
    let adj = cubic_adjoint(x, [1.0]; extrap = WrapExtrap()), f_bar = adj(y_bar)
        @test f_bar[end] ≈ 1.0 atol = 1.0e-12
        @test f_bar[1] ≈ 0.0 atol = 1.0e-12
    end
    let adj = quadratic_adjoint(x, [1.0]; extrap = WrapExtrap()), f_bar = adj(y_bar)
        @test f_bar[end] ≈ 1.0 atol = 1.0e-12
        @test f_bar[1] ≈ 0.0 atol = 1.0e-12
    end
    # OnTheFly Hermite families: slopes are data-dependent, so adjoint takes (x, y, xq)
    let adj = pchip_adjoint(x, y, [1.0]; extrap = WrapExtrap()), f_bar = adj(y_bar)
        @test f_bar[end] ≈ 1.0 atol = 1.0e-12
        @test f_bar[1] ≈ 0.0 atol = 1.0e-12
    end
    # Cardinal adjoint: data-independent (linear in y), so no y arg
    let adj = cardinal_adjoint(x, [1.0]; tension = 0.0, extrap = WrapExtrap()),
            f_bar = adj(y_bar)
        @test f_bar[end] ≈ 1.0 atol = 1.0e-12
        @test f_bar[1] ≈ 0.0 atol = 1.0e-12
    end
    let adj = akima_adjoint(x, y, [1.0]; extrap = WrapExtrap()), f_bar = adj(y_bar)
        @test f_bar[end] ≈ 1.0 atol = 1.0e-12
        @test f_bar[1] ≈ 0.0 atol = 1.0e-12
    end
    # Hermite (user-supplied slopes): adjoint is data-independent (x, xq) only
    let adj = hermite_adjoint(x, [1.0]; extrap = WrapExtrap()), f_bar = adj(y_bar)
        @test f_bar[end] ≈ 1.0 atol = 1.0e-12
        @test f_bar[1] ≈ 0.0 atol = 1.0e-12
    end
end

@testitem "Closed boundary — zero alloc on new WrapExtrap fast paths" setup = [AllocConstants] begin
    using FastInterpolations
    # The closed-domain conversion added (1) a constant scalar oneshot right-edge
    # short-circuit and (2) widened the linear batch fast-path predicate to `<=`.
    # Both must stay within `ALLOC_THRESHOLD` (0 on Julia ≥ 1.12, ~240 B on LTS).

    # Function-barrier pattern: setup + warmup + @allocated all inside one fn,
    # so the @testset try/catch doesn't taint type inference of locals.
    function _alloc_constant_oneshot_at_xmax()
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        # warmup
        constant_interp(x, y, 1.0; extrap = WrapExtrap())
        return @allocated constant_interp(x, y, 1.0; extrap = WrapExtrap())
    end

    function _alloc_linear_batch_at_xmax()
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        out = Vector{Float64}(undef, 3)
        targets = [0.25, 0.5, 1.0]
        # warmup
        linear_interp!(out, x, y, targets; extrap = WrapExtrap())
        return @allocated linear_interp!(out, x, y, targets; extrap = WrapExtrap())
    end

    @test _alloc_constant_oneshot_at_xmax() <= ALLOC_THRESHOLD
    @test _alloc_linear_batch_at_xmax() <= ALLOC_THRESHOLD
end

@testitem "Closed boundary — derivative at last(x) matches last-segment slope × WrapExtrap" begin
    using FastInterpolations
    # Under closed semantics, xq == last(x) lands in cell n-1 at α=1, so
    # `deriv=DerivOp(1)` must return the last-segment slope. Under the old
    # half-open convention it would have wrapped to xq=first(x) and returned
    # the FIRST-segment slope — a discontinuity at the seam. Asymmetric data
    # exposes the difference.
    x = collect(range(0.0, 1.0, 11))
    h = x[2] - x[1]
    # Asymmetric so first-segment slope ≠ last-segment slope
    y = exp.(x)
    last_slope = (y[end] - y[end - 1]) / h
    first_slope = (y[2] - y[1]) / h
    @test last_slope ≉ first_slope            # data is asymmetric (sanity)

    # Linear: analytical — derivative is exactly the segment slope
    @test linear_interp(x, y, 1.0; extrap = WrapExtrap(), deriv = DerivOp(1)) ≈
        last_slope atol = 1.0e-12

    # Cubic/Quadratic/Hermite: derivative is continuous; verify it matches the
    # limit from inside the domain (closed-domain seam continuity, not a wrap jump)
    for method_call in (
            x_ -> cubic_interp(x, y, x_; extrap = WrapExtrap(), deriv = DerivOp(1)),
            x_ -> quadratic_interp(x, y, x_; extrap = WrapExtrap(), deriv = DerivOp(1)),
            x_ -> pchip_interp(x, y, x_; extrap = WrapExtrap(), deriv = DerivOp(1)),
            x_ -> akima_interp(x, y, x_; extrap = WrapExtrap(), deriv = DerivOp(1)),
        )
        d_at_max = method_call(1.0)
        d_just_below = method_call(1.0 - 1.0e-7)
        @test d_at_max ≈ d_just_below rtol = 1.0e-4
        # And NOT equal to the first-segment slope (would be the case under old wrap)
        @test !isapprox(d_at_max, first_slope; rtol = 1.0e-3)
    end
end
