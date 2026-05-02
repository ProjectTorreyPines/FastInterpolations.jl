# Tests for PeriodicBC on Linear interpolation (Phase 1).
#
# Coverage:
#   - `NoBC()` default is a no-op (regression guard against pre-change behavior).
#   - `PeriodicBC(endpoint=:inclusive)` wraps correctly with y[1] == y[end] data.
#   - `PeriodicBC(endpoint=:exclusive, period=L)` extends grid+data, forces WrapExtrap.
#   - FVM cell-centered case: x = [0.5, 1.5, 2.5], period = 3.0.
#   - Range grid (type-stable extension) vs Vector grid (vcat extension).
#   - Inclusive check raises on mismatched endpoints.

@testitem "Linear PeriodicBC" setup = [AllocConstants] begin
    using FastInterpolations: _is_periodic_bc, _CachedRange

    @testset "NoBC default is no-op" begin
        x = collect(range(0.0, 1.0, length = 11))
        y = sin.(2π .* x)

        itp_default = linear_interp(x, y)                       # no bc kwarg
        itp_nobc = linear_interp(x, y; bc = NoBC())          # explicit NoBC

        @test typeof(itp_default) === typeof(itp_nobc)
        for xq in (0.0, 0.25, 0.5, 0.75, 1.0)
            @test itp_default(xq) === itp_nobc(xq)
        end
    end

    @testset "Inclusive — Range grid" begin
        x = range(0.0, 2π, length = 11)                         # y[1] == y[end] for sin
        y = sin.(x)
        itp = linear_interp(x, y; bc = PeriodicBC())

        # Round-trip at domain endpoints
        @test itp(0.0) ≈ itp(2π) atol = 1.0e-12

        # Query outside base domain wraps by `period = 2π`
        @test itp(2π + 0.5) ≈ itp(0.5) atol = 1.0e-12
        @test itp(-0.3) ≈ itp(2π - 0.3) atol = 1.0e-12
    end

    @testset "Inclusive — endpoint mismatch raises" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]                          # y[1] != y[end]
        @test_throws ArgumentError linear_interp(x, y; bc = PeriodicBC())
    end

    @testset "Inclusive — endpoint mismatch raises on 1D oneshot (scalar + vector)" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]                          # y[1] != y[end]
        @test_throws ArgumentError linear_interp(x, y, 0.5; bc = PeriodicBC())
        @test_throws ArgumentError linear_interp(x, y, [0.25, 0.75]; bc = PeriodicBC())
    end

    @testset "Inclusive — endpoint check=false skips validation on 1D oneshot" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        bc = PeriodicBC(endpoint = :inclusive, check = false)
        @test linear_interp(x, y, 0.5; bc = bc) isa Real
        @test linear_interp(x, y, [0.25, 0.75]; bc = bc) isa AbstractVector
    end

    @testset "Non-Float wrap domains — Int grid + PeriodicBC" begin
        # Contract: y[1] ≈ y[end] is satisfied so :inclusive build/eval is legal.
        # Exercises WrapExtrap{Int} (produced by _resolve_periodic_extrap on Int grid).
        x = [0, 1, 2, 3]
        y = [1.0, 2.0, 3.0, 1.0]
        # Persistent
        itp = linear_interp(x, y; bc = PeriodicBC())
        @test itp(0.5) ≈ 1.5 atol = 1.0e-12    # no wrap
        @test itp(1.5) ≈ 2.5 atol = 1.0e-12    # no wrap
        @test itp(4.5) ≈ 2.5 atol = 1.0e-12    # wrap: 4.5 → 1.5 via period=3

        # Oneshot scalar + vector
        @test linear_interp(x, y, 0.5; bc = PeriodicBC()) ≈ 1.5 atol = 1.0e-12
        @test linear_interp(x, y, 4.5; bc = PeriodicBC()) ≈ 2.5 atol = 1.0e-12
        @test linear_interp(x, y, [0.5, 4.5]; bc = PeriodicBC()) ≈ [1.5, 2.5] atol = 1.0e-12
    end

    @testset "Non-Float wrap domains — Int grid + WrapExtrap()" begin
        # `WrapExtrap` is a tag struct: the wrap domain `[first(x), last(x))`
        # is read directly from the axis at query time. For an Int grid that
        # exercises the duck-typed 3-arg `_wrap_to_domain` (Int x_min/x_max,
        # Float xi).
        x_int = [0, 1, 2, 3, 4]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        itp = linear_interp(x_int, y; extrap = WrapExtrap())
        # wrap: 4.5 → period 4 → 0.5 → between y[1]=1.0 and y[2]=2.0 → 1.5
        @test itp(4.5) ≈ 1.5 atol = 1.0e-12
    end

    @testset "Non-Float wrap domains — Rational grid + PeriodicBC" begin
        # Rational grid preserves exact arithmetic through the wrap path.
        x = [0 // 1, 1 // 1, 2 // 1, 3 // 1]
        y = [1.0, 2.0, 3.0, 1.0]
        itp = linear_interp(x, y; bc = PeriodicBC())
        @test itp(0.5) ≈ 1.5 atol = 1.0e-12
        @test itp(4.5) ≈ 2.5 atol = 1.0e-12
        @test linear_interp(x, y, 4.5; bc = PeriodicBC()) ≈ 2.5 atol = 1.0e-12
    end

    @testset "Exclusive — FVM cell-centered, Range grid" begin
        # Cell centers x = [0.5, 1.5, 2.5], physical period L = 3.0
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 3.0))

        # Extended virtual endpoint: x[end+1] = 0.5 + 3.0 = 3.5, y = 10.0 (wraps to y[1])
        # So segment [2.5, 3.5] interpolates linearly between 30.0 → 10.0.
        @test itp(2.5) ≈ 30.0 atol = 1.0e-12
        @test itp(3.0) ≈ 20.0 atol = 1.0e-12    # midpoint of [2.5,3.5]: (30+10)/2 = 20
        @test itp(3.5) ≈ 10.0 atol = 1.0e-12    # wraps to x=0.5 via WrapExtrap

        # Query to the left of x[1] wraps into the extended segment
        @test itp(0.0) ≈ 20.0 atol = 1.0e-12    # 0.0 wraps to 3.0 → midpoint of [2.5, 3.5]
    end

    @testset "Exclusive — Vector grid (allocating path)" begin
        x_vec = [0.5, 1.5, 2.5]                                # Vector, not Range
        y = [10.0, 20.0, 30.0]
        itp = linear_interp(x_vec, y; bc = PeriodicBC(endpoint = :exclusive, period = 3.0))

        @test itp(2.5) ≈ 30.0 atol = 1.0e-12
        @test itp(3.0) ≈ 20.0 atol = 1.0e-12
        @test itp(3.5) ≈ 10.0 atol = 1.0e-12
    end

    @testset "Exclusive — auto-infer period from Range" begin
        # period auto = step(x) * length(x) = 1.0 * 3 = 3.0
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))

        @test itp(3.0) ≈ 20.0 atol = 1.0e-12
    end

    @testset "Exclusive — auto-infer period on 1D oneshot (regression for unresolved Searcher bc)" begin
        # Regression: Searcher must receive resolved PeriodicBC period. Otherwise the seam
        # branch `x[1] + s.bc.period` in `search_interval` reduces to `Float + Nothing`
        # and throws MethodError. The persistent interpolant materializes the period at
        # construction; the oneshot path used to thread the raw bc directly.
        x = range(0.5, step = 1.0, length = 3)   # period auto = 3.0
        y = [10.0, 20.0, 30.0]
        bc_auto = PeriodicBC(endpoint = :exclusive)
        bc_expl = PeriodicBC(endpoint = :exclusive, period = 3.0)

        # Scalar oneshot at seam cell (xq = 3.0 is inside [x[n]=2.5, x[1]+period=3.5))
        @test linear_interp(x, y, 3.0; bc = bc_auto) ≈ 20.0 atol = 1.0e-12
        @test linear_interp(x, y, 3.0; bc = bc_auto) ≈
            linear_interp(x, y, 3.0; bc = bc_expl) atol = 1.0e-12

        # Vector oneshot mixing seam + non-seam queries
        xq = [0.5, 1.0, 2.5, 3.0, 3.4]
        @test linear_interp(x, y, xq; bc = bc_auto) ≈
            linear_interp(x, y, xq; bc = bc_expl) atol = 1.0e-12
    end

    @testset "ND Exclusive — auto-infer period on ND oneshot (regression)" begin
        # Same class of bug as above, via _search_all_intervals_stencil → _resolve_search per axis.
        x = range(0.5, step = 1.0, length = 3)         # axis-1 periodic-exclusive (period auto)
        y = range(0.0, 1.0, length = 4)                # axis-2 NoBC
        data = [10xi + yj for xi in x, yj in y]
        bc_auto = (PeriodicBC(endpoint = :exclusive), NoBC())
        bc_expl = (PeriodicBC(endpoint = :exclusive, period = 3.0), NoBC())

        q = (3.0, 0.5)                                  # axis-1 seam, axis-2 interior
        @test linear_interp((x, y), data, q; bc = bc_auto) ≈
            linear_interp((x, y), data, q; bc = bc_expl) atol = 1.0e-12
    end

    @testset "Extrap is forced to WrapExtrap on periodic path" begin
        x = range(0.0, 2π, length = 11)
        y = sin.(x)
        # User-passed extrap is silently overridden (matches cubic 1D behavior).
        itp = linear_interp(x, y; bc = PeriodicBC(), extrap = ClampExtrap())
        @test itp.extrap isa WrapExtrap
    end

    @testset "Oneshot scalar — matches persistent interpolant" begin
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp = linear_interp(x, y; bc = bc)

        for xq in (0.5, 1.5, 2.5, 3.0, 3.4, 0.0, -0.5)
            @test linear_interp(x, y, xq; bc = bc) ≈ itp(xq) atol = 1.0e-12
        end
    end

    @testset "Oneshot vector (allocating) — matches persistent interpolant" begin
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp = linear_interp(x, y; bc = bc)

        xq = [0.5, 1.0, 2.5, 3.0, 3.4]
        out_oneshot = linear_interp(x, y, xq; bc = bc)
        out_itp = itp.(xq)
        @test out_oneshot ≈ out_itp atol = 1.0e-12
    end

    @testset "Oneshot in-place — matches persistent interpolant" begin
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp = linear_interp(x, y; bc = bc)

        xq = [0.5, 1.0, 2.5, 3.0, 3.4]
        out = zeros(length(xq))
        linear_interp!(out, x, y, xq; bc = bc)
        @test out ≈ itp.(xq) atol = 1.0e-12
    end

    @testset "Oneshot NoBC default — regression guard" begin
        x = collect(range(0.0, 1.0, length = 11))
        y = sin.(2π .* x)
        xq = [0.05, 0.45, 0.95]

        # Default (no bc kwarg) must equal explicit NoBC()
        @test linear_interp(x, y, xq) == linear_interp(x, y, xq; bc = NoBC())
        @test linear_interp(x, y, 0.5) === linear_interp(x, y, 0.5; bc = NoBC())
    end

    @testset "Exclusive — Vector grid seam at xq == x[n] exactly (T-1)" begin
        # _search_binary path must still route xq == x[n] to the seam branch
        # (xq >= x[n]) before entering the binary search. Off-by-one here could
        # read data[end+1].
        x_vec = [0.0, 0.25, 0.5, 0.75]            # non-uniform-ok? uniform here
        y = [1.0, 2.0, 4.0, 8.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        itp = linear_interp(x_vec, y; bc = bc)

        # At xq == x[n] the seam cell left edge → α = 0 → value = y[n]
        @test itp(0.75) ≈ 8.0 atol = 1.0e-12
        @test linear_interp(x_vec, y, 0.75; bc = bc) ≈ 8.0 atol = 1.0e-12
        # Strictly inside seam
        @test itp(0.85) ≈ 8.0 * 0.6 + 1.0 * 0.4 atol = 1.0e-12
        @test linear_interp(x_vec, y, 0.85; bc = bc) ≈ itp(0.85) atol = 1.0e-12
    end

    @testset "Exclusive — Clamp/Fill/Extend extrap silently overridden on 1D bundled (T-2)" begin
        # Periodic BC forces WrapExtrap regardless of user-passed extrap on 1D.
        # ND raises; 1D silently overrides (matches interpolant path).
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)

        itp_ref = linear_interp(x, y; bc = bc)
        for extrap in (ClampExtrap(), ExtendExtrap(), FillExtrap(0.0))
            itp = linear_interp(x, y; bc = bc, extrap = extrap)
            @test itp.extrap isa WrapExtrap
            @test itp(3.0) ≈ itp_ref(3.0) atol = 1.0e-12
            # Oneshot override must also match (regression for B-1 era path)
            @test linear_interp(x, y, 3.0; bc = bc, extrap = extrap) ≈ itp_ref(3.0) atol = 1.0e-12
        end
    end

    @testset "Exclusive — ND persistent rejects period-too-small at build (T-3)" begin
        # period < grid_span should trip _throw_wrap_virtual_endpoint_error at ND build.
        x = [0.0, 1.0, 2.0, 3.0]
        y = range(0.0, 1.0, length = 4)
        data = rand(4, 4)
        # period=2.5 places virtual endpoint at 2.5, behind x[end]=3.0
        @test_throws ArgumentError linear_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive, period = 2.5), NoBC()),
        )
    end

    @testset "Exclusive — seam continuity invariant (T-4)" begin
        # Left/right limits at the virtual endpoint must agree with y[1].
        x = range(0.0, step = 1.0, length = 4)
        y = [1.0, 2.0, 3.0, 4.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        itp = linear_interp(x, y; bc = bc)
        # Approach virtual endpoint 4.0 from the left (inside seam)
        @test itp(4.0 - 1.0e-9) ≈ y[1] atol = 1.0e-6
        # Approach from right — wraps to +ε, back to y[1]
        @test itp(1.0e-9) ≈ y[1] atol = 1.0e-8
        # Exactly at virtual endpoint wraps to x[1]
        @test itp(4.0) ≈ y[1] atol = 1.0e-12
    end

    @testset "Exclusive — derivative at seam (T-5)" begin
        # Seam cell derivative via Rs - Ls width on virtual endpoint.
        # y = [1, 2, 4, 8], x step = 1. Seam slope = (y[1] - y[n]) / step = (1 - 8) / 1 = -7.
        x = range(0.0, step = 1.0, length = 4)
        y = [1.0, 2.0, 4.0, 8.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        itp = linear_interp(x, y; bc = bc)
        @test itp(3.5; deriv = DerivOp(1)) ≈ -7.0 atol = 1.0e-10
        @test linear_interp(x, y, 3.5; bc = bc, deriv = DerivOp(1)) ≈ -7.0 atol = 1.0e-10
    end

    @testset "Exclusive — ND axis-2 periodic only (T-6)" begin
        # Non-leading periodic axis exercises a different recursion / dispatch pattern
        # than leading-axis periodic.
        x = range(0.0, 1.0, length = 5)                       # NoBC
        y = range(0.5, step = 1.0, length = 3)                # :exclusive
        data = [i + 10j for i in 1:5, j in 1:3]
        bc = (NoBC(), PeriodicBC(endpoint = :exclusive, period = 3.0))
        itp = linear_interp((x, y), data; bc = bc)

        # Seam-wrap equivalence along axis 2
        @test itp((0.5, 0.5)) ≈ itp((0.5, 3.5)) atol = 1.0e-12
        # Oneshot must agree
        @test linear_interp((x, y), data, (0.5, 3.0); bc = bc) ≈ itp((0.5, 3.0)) atol = 1.0e-12
    end

    @testset "Exclusive — NoBC oneshot allocation regression guard (T-9)" begin
        # After the WrapExtrap{T} materialization refactor, NoBC paths must not
        # *leak* WrapExtrap{Float64} structs to the heap. On 1.12+ escape analysis
        # stack-elides them (0 bytes); on LTS (1.10) a small Ref / struct may
        # survive (~16 B). Compare against ALLOC_THRESHOLD per the project-wide
        # convention rather than strict `== 0`.
        x = range(0.0, 1.0, length = 11)
        y = sin.(2π .* x)
        # Warmup
        for _ in 1:3
            linear_interp(x, y, 0.5)
        end
        @test (@allocated linear_interp(x, y, 0.5)) <= ALLOC_THRESHOLD
    end

    @testset "Exclusive — period conflict rejected in oneshot too (P1 review)" begin
        # Regression: WrapExtrap(x, bc::PeriodicBC{:exclusive, <:Real}) used to read
        # `bc.period` directly, skipping the `step(x) * length(x)` cross-check that
        # persistent paths route through `_resolve_exclusive_period`. That let
        # oneshot silently accept a period that disagrees with the Range's implied
        # period. Both paths must now throw on the same input.
        x = range(0.0, step = 0.1, length = 10)           # implied period = 1.0
        y = rand(10)
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.0)   # conflicts
        @test_throws ArgumentError linear_interp(x, y; bc = bc_bad)
        @test_throws ArgumentError linear_interp(x, y, 0.5; bc = bc_bad)
        @test_throws ArgumentError linear_interp(x, y, [0.2, 0.5]; bc = bc_bad)
    end

    @testset "Exclusive — ND oneshot Vector grid without period raises (3.5)" begin
        # Public ND oneshot must surface the non-Range + no-period error path.
        x = [0.0, 1.0, 2.0, 3.0]                        # Vector grid, cannot infer period
        y = range(0.0, 1.0, length = 4)
        data = rand(4, 4)
        @test_throws ArgumentError linear_interp(
            (x, y), data, (1.5, 0.5);
            bc = (PeriodicBC(endpoint = :exclusive), NoBC()),   # period::Nothing
        )
    end

    @testset "Exclusive — seam hint write-back updates RefHint (P2 review)" begin
        # Regression: the seam fast path in `search_interval` used to return
        # before `_search_interval_real`, so RefHint never got updated at the
        # seam cell. Monotone batches spending time past x[n] would keep
        # searching from the stale interior hint. After fix, the seam branch
        # writes `n` back to hint.idx so LinearBinarySearch can resume from
        # the seam position on subsequent queries.
        x = range(0.0, step = 1.0, length = 5)            # [0, 1, 2, 3, 4]
        bc = PeriodicBC(endpoint = :exclusive, period = 5.0)
        ref = Ref(1)
        s = FastInterpolations._resolve_search(x, 4.5, AutoSearch(), ref, bc)
        # Seam query: xq=4.5 > x[end]=4 → seam fires, hint must now be n=5
        FastInterpolations.search_interval(s, x, 4.5)
        @test ref[] == 5
    end

    @testset "Exclusive — adjoint with WrapExtrap() smoke (T-8)" begin
        # Adjoint doesn't accept `bc`, so periodic-shaped adjoint is exercised via
        # explicit `extrap=WrapExtrap()` (tag struct — wrap domain comes from the
        # axis at query time). Verify it flows through scatter correctly.
        x = range(0.0, 1.0, length = 5)
        y = range(0.0, 1.0, length = 4)
        q = ((0.5, 0.25),)
        adj = linear_adjoint((x, y), q; extrap = WrapExtrap())
        # Scatter a scalar gradient of 1.0 → per-corner weights sum to 1.0 at a single query
        y_bar = [1.0]
        f_bar = adj(y_bar)
        @test sum(f_bar) ≈ 1.0 atol = 1.0e-12
        @test size(f_bar) == (length(x), length(y))
    end

    # ============================================================
    # ND tests
    # ============================================================
    @testset "ND — scalar bc applied to all axes (2D)" begin
        x = range(0.5, step = 1.0, length = 3)
        y = range(0.0, 1.0, length = 5)
        data = [10.0 * i + j for i in 1:3, j in 1:5]
        # 2D periodic on both axes, auto-infer period
        bc = PeriodicBC(endpoint = :exclusive)
        itp = linear_interp((x, y), data; bc = bc)

        # ND persistent path: extended Range axes must remain _CachedRange.
        @test itp.grids[1] isa _CachedRange
        @test itp.grids[2] isa _CachedRange
        @test length(itp.grids[1]) == 4  # 3 + 1
        @test length(itp.grids[2]) == 6  # 5 + 1
        @test size(itp.data) == (4, 6)

        # Interior query agrees between persistent and oneshot
        q = (1.2, 0.5)
        @test itp(q) ≈ linear_interp((x, y), data, q; bc = bc) atol = 1.0e-12
    end

    @testset "ND — per-axis bc tuple (periodic + non-periodic mix)" begin
        x = range(0.5, step = 1.0, length = 3)                  # periodic axis
        y = collect(range(0.0, 1.0, length = 5))                # non-periodic axis
        data = [10.0 * i + j for i in 1:3, j in 1:5]
        bc = (PeriodicBC(endpoint = :exclusive, period = 3.0), NoBC())
        itp = linear_interp((x, y), data; bc = bc)

        # Query at y=0.5 should wrap correctly on axis 1 when xq goes past x[end]
        @test itp((0.5, 0.5)) ≈ itp((3.5, 0.5)) atol = 1.0e-12  # same by periodicity
    end

    @testset "ND — NoBC default is no-op" begin
        x = range(0.0, 1.0, length = 5)
        y = range(0.0, 1.0, length = 4)
        data = rand(5, 4)

        itp_default = linear_interp((x, y), data)
        itp_nobc = linear_interp((x, y), data; bc = NoBC())

        for q in ((0.3, 0.5), (0.9, 0.1), (0.0, 0.0), (1.0, 1.0))
            @test itp_default(q) === itp_nobc(q)
        end
    end

    @testset "ND — incompatible extrap with PeriodicBC raises" begin
        x = range(0.5, step = 1.0, length = 3)
        y = range(0.0, 1.0, length = 5)
        data = rand(3, 5)
        # ND strict check: ClampExtrap on a periodic axis must raise
        @test_throws ArgumentError linear_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive, period = 3.0), NoBC()),
            extrap = (ClampExtrap(), NoExtrap())
        )
    end

    @testset "ND — :inclusive slice mismatch raises on build" begin
        # `:inclusive` default requires data[1,...] ≈ data[end,...] along each
        # periodic axis. Mismatched endpoint slices must raise, mirroring 1D.
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))
        good = [sin(xi) * cos(yj) for xi in x, yj in y]  # endpoints match
        bad = copy(good)
        bad[end, :] .+= 1.0  # break periodicity along axis 1

        @test_throws ArgumentError linear_interp((x, y), bad; bc = PeriodicBC())
        @test linear_interp((x, y), good; bc = PeriodicBC()) isa LinearInterpolantND
    end

    @testset "ND — :inclusive slice mismatch raises on oneshot" begin
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))
        bad = [sin(xi) * cos(yj) for xi in x, yj in y]
        bad[:, end] .+= 1.0  # break along axis 2

        q = (1.0, 1.0)
        @test_throws ArgumentError linear_interp((x, y), bad, q; bc = PeriodicBC())
    end

    @testset "ND — :inclusive + check=false skips validation" begin
        x = range(0.0, 2π, length = 5)
        y = range(0.0, 2π, length = 5)
        bad = [sin(xi) * cos(yj) for xi in x, yj in y]
        bad[end, :] .+= 1.0
        itp = linear_interp(
            (x, y), bad;
            bc = (
                PeriodicBC(endpoint = :inclusive, check = false),
                PeriodicBC(endpoint = :inclusive, check = false),
            )
        )
        @test itp isa LinearInterpolantND
    end

    @testset "ND seam-cell — _multilinear_sum exact bilinear via wrap" begin
        # 2D, axis 1 :exclusive (period=4), axis 2 NoBC. Query lands in the seam
        # cell on axis 1, so the kernel must read corners (n, 1) on that axis
        # rather than (n, n+1). Picks data values that make the expected wrap-
        # interpolated result trivially predictable.
        x = collect(range(0.0, step = 1.0, length = 4))     # [0,1,2,3], period 4
        yy = collect(range(0.0, step = 1.0, length = 3))    # [0,1,2]
        # data[i, j] = (i-1) + 10*(j-1)  ⇒ corner-recoverable
        data = [Float64(i - 1) + 10 * Float64(j - 1) for i in 1:4, j in 1:3]

        bc = (PeriodicBC(endpoint = :exclusive, period = 4.0), NoBC())
        # NoExtrap on axis 2 is fine because q[2] is in domain.
        extrap = (NoExtrap(), NoExtrap())

        # Query in seam cell axis 1: 3.5 ∈ [3, 4=x[1]+period].
        # Axis 1 corners read data[4, j] (=3 + 10*(j-1)) and data[1, j] (=10*(j-1)).
        # α = (3.5 - 3)/(4 - 3) = 0.5 ⇒ axis-1 blend = 1.5 + 10*(j-1).
        # Axis 2 at 0.5 between j=1 (1.5) and j=2 (11.5) ⇒ β=0.5 ⇒ 6.5.
        q = (3.5, 0.5)
        @test linear_interp((x, yy), data, q; bc = bc, extrap = extrap) ≈ 6.5 atol = 1.0e-12

        # Persistent must agree at the same query (zero-copy must match extension path).
        itp = linear_interp((x, yy), data; bc = bc, extrap = extrap)
        @test itp(q) ≈ 6.5 atol = 1.0e-12

        # Query just below seam → no wrap, normal bilinear cell [2,3]×[0,1].
        # data[3,1]=2, data[4,1]=3, data[3,2]=12, data[4,2]=13.
        # At (2.9, 0.5): α=0.9, β=0.5
        # 0.1*0.5*2 + 0.9*0.5*3 + 0.1*0.5*12 + 0.9*0.5*13
        # = 0.1 + 1.35 + 0.6 + 5.85 = 7.9
        q2 = (2.9, 0.5)
        @test linear_interp((x, yy), data, q2; bc = bc, extrap = extrap) ≈ 7.9 atol = 1.0e-12

        # Both axes periodic, both in seam — kernel reads (4,1) on each axis.
        bc_both = (
            PeriodicBC(endpoint = :exclusive, period = 4.0),
            PeriodicBC(endpoint = :exclusive, period = 3.0),
        )
        # data periodic along axis 2 too: rebuild with period-3 wrap-friendly values.
        # Use data[i,j] = (i-1) + (j-1) so wrap on axis 2 gives j=1 ↔ j=4 virtual.
        data2 = [Float64(i - 1) + Float64(j - 1) for i in 1:4, j in 1:3]
        q3 = (3.5, 2.5)  # both axes in seam cells
        # Axis 1 corners: data2[4, j]=3+(j-1) and data2[1, j]=(j-1) → α=0.5 → 1.5+(j-1)
        # Axis 2 corners (via wrap): blend between j=3 (1.5+2=3.5) and j=1 (1.5+0=1.5)
        # β = (2.5 - 2)/(3 - 2) = 0.5 → result = 0.5*3.5 + 0.5*1.5 = 2.5
        @test linear_interp((x, yy), data2, q3; bc = bc_both, extrap = extrap) ≈ 2.5 atol = 1.0e-12
        itp_both = linear_interp((x, yy), data2; bc = bc_both, extrap = extrap)
        @test itp_both(q3) ≈ 2.5 atol = 1.0e-12
    end

    @testset "ND seam-cell — _multilinear_sum at N=3 with periodic axis-1 wrap" begin
        # The unified `_multilinear_sum` is @generated and unrolls to 2^N corners.
        # Existing seam-cell tests are N=2 only; this exercises the N=3 unroll
        # with a wrapped corner on axis 1, ensuring the @generated body addresses
        # `stencils[1][2] == 1` (wrap) correctly for any N.
        x = collect(range(0.0, step = 1.0, length = 4))    # axis 1 periodic, period 4
        yy = collect(range(0.0, step = 1.0, length = 3))   # axis 2 NoBC
        zz = collect(range(0.0, step = 1.0, length = 3))   # axis 3 NoBC
        # Separable data: (i-1) + 10*(j-1) + 100*(k-1) — corner-recoverable.
        data = [
            Float64(i - 1) + 10 * Float64(j - 1) + 100 * Float64(k - 1)
                for i in 1:4, j in 1:3, k in 1:3
        ]

        bc = (PeriodicBC(endpoint = :exclusive, period = 4.0), NoBC(), NoBC())
        extrap = (NoExtrap(), NoExtrap(), NoExtrap())

        # Seam on axis 1 at q1=3.5: blend axis-1 corners 3 (data[4,j,k]) and 4≡0 (data[1,j,k])
        # ⇒ axis-1 contribution = 1.5 + 10*(j-1) + 100*(k-1).
        # Axis 2 at 0.5 between j=1 (1.5) and j=2 (11.5) ⇒ 6.5 + 100*(k-1).
        # Axis 3 at 0.5 between k=1 (6.5) and k=2 (106.5) ⇒ 56.5.
        q = (3.5, 0.5, 0.5)
        @test linear_interp((x, yy, zz), data, q; bc = bc, extrap = extrap) ≈ 56.5 atol = 1.0e-12

        itp = linear_interp((x, yy, zz), data; bc = bc, extrap = extrap)
        @test itp(q) ≈ 56.5 atol = 1.0e-12

        # Persistent must match oneshot exactly (zero-copy seam ↔ extended-data path).
        @test itp(q) ≈ linear_interp((x, yy, zz), data, q; bc = bc, extrap = extrap) atol = 1.0e-12
    end

    # ============================================================
    # Edge cases (mirrors test_periodic_exclusive.jl for cubic)
    # ============================================================
    @testset "Edge — Vector grid :exclusive without period raises" begin
        # period cannot be inferred from a Vector grid — must be supplied.
        x = [0.0, 1.0, 2.0, 3.0]
        y = sin.(x)
        @test_throws ArgumentError linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
    end

    @testset "Edge — Range grid + conflicting period raises" begin
        # step(x) * length(x) = 0.1 * 10 = 1.0, but user claims period=2.0 → mismatch.
        x = range(0.0, step = 0.1, length = 10)
        y = sin.(x)
        @test_throws ArgumentError linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 2.0))
    end

    @testset "Edge — oneshot Vector grid :exclusive without period raises" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = sin.(x)
        @test_throws ArgumentError linear_interp(x, y, 1.5; bc = PeriodicBC(endpoint = :exclusive))
    end

    @testset "Edge — oneshot Vector grid :exclusive period too small raises" begin
        # first(x) + period = 0 + 2.5 = 2.5 < last(x) = 3.0 → virtual endpoint not beyond grid
        x = [0.0, 1.0, 2.0, 3.0]
        y = sin.(x)
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError linear_interp(x, y, 1.5; bc = bc_bad)
        @test_throws ArgumentError linear_interp(x, y, [1.5, 2.5]; bc = bc_bad)   # vector oneshot path
    end

    # ============================================================
    # Interpolant path — extended copy storage
    # ============================================================
    @testset "Interpolant path stores Vector grid in `_ExclusivePeriodicAxis` + y in `_ExclusivePeriodicData`" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]
        x_ref = copy(x)
        y_ref = copy(y)

        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 4.0))

        # Vector + `:exclusive` is now wrapped in `_ExclusivePeriodicAxis`
        # (axis-side, carries period) + `_ExclusivePeriodicData` (data-side,
        # auto-cyclic). Zero-copy: both wrappers reference the user's arrays.
        @test itp.x isa FastInterpolations._ExclusivePeriodicAxis
        @test itp.y isa FastInterpolations._ExclusivePeriodicData
        @test length(itp.x) == 5                # virtual extended length n+1
        @test length(itp.y) == 5                # data wrapper also reports n+1
        @test length(itp.x.inner) == 4          # physical inner grid length n
        @test length(itp.y.inner) == 4          # physical inner data length n
        @test itp.x.period ≈ 4.0
        @test itp.x.inner == x                  # inner grid is the user's original grid
        @test itp.y.inner == y                  # inner data is the user's original values

        # Virtual endpoints: axis carries coord (`inner[1] + period`),
        # data auto-cycles (`inner[1]`).
        @test itp.x[5] ≈ 4.0                    # cyclic via axis wrapper's getindex
        @test itp.y[5] == itp.y[1] == 10.0      # cyclic via data wrapper's getindex
        @test last(itp.x) ≈ 4.0                 # axis: inner[1] + period
        @test last(itp.y) == 10.0               # data: inner[1] (cyclic)

        # Original user arrays are untouched.
        @test x == x_ref
        @test y == y_ref
    end

    @testset "Interpolant path wraps Range axis (`_ExclusivePeriodicAxis(_CachedRange)`)" begin
        x = range(0.0, step = 1.0, length = 4)
        y = [10.0, 20.0, 30.0, 40.0]

        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))

        # Range input → `_ExclusivePeriodicAxis(_CachedRange, period)` (uniform with
        # the Vector path; period + virtual endpoint cached on the axis).
        @test itp.x isa FastInterpolations._ExclusivePeriodicAxis
        @test itp.x.inner isa _CachedRange
        @test length(itp.x) == 5             # virtual length n+1
        @test length(itp.x.inner) == 4       # raw n-length cached Range
        @test length(itp.y) == 5             # `_ExclusivePeriodicData` virtual n+1
        @test itp.y[end] == itp.y[1]         # cyclic
    end

    # ============================================================
    # Zero-allocation tests (one-shot, after warmup)
    # ============================================================
    # Function-barrier pattern: setup + warmup + @allocated must live in ONE
    # function so @testset's try/catch wrapping doesn't make locals type-unstable.
    # Double warmup primes both compilation and pool slab reuse (cubic pattern).
    function _alloc_linear_1d_range_scalar()
        x = range(0.0, step = 2π / 16, length = 16)
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive)
        linear_interp(x, y, 1.0; bc = bc)
        linear_interp(x, y, 1.0; bc = bc)
        return @allocated linear_interp(x, y, 1.0; bc = bc)
    end

    function _alloc_linear_1d_range_vec()
        x = range(0.0, step = 2π / 16, length = 16)
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive)
        xq = [0.5, 1.0, 2.0, 3.5]
        out = similar(xq)
        linear_interp!(out, x, y, xq; bc = bc)
        linear_interp!(out, x, y, xq; bc = bc)
        return @allocated linear_interp!(out, x, y, xq; bc = bc)
    end

    function _alloc_linear_1d_vector_scalar()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        linear_interp(x, y, 1.0; bc = bc)
        linear_interp(x, y, 1.0; bc = bc)
        return @allocated linear_interp(x, y, 1.0; bc = bc)
    end

    function _alloc_linear_1d_vector_vec()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        xq = [0.5, 1.0, 2.0, 3.5]
        out = similar(xq)
        linear_interp!(out, x, y, xq; bc = bc)
        linear_interp!(out, x, y, xq; bc = bc)
        return @allocated linear_interp!(out, x, y, xq; bc = bc)
    end

    function _alloc_linear_nd_range()
        x = range(0.0, step = 2π / 8, length = 8)
        y = range(0.0, step = 2π / 8, length = 8)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive)
        q = (1.0, 2.0)
        linear_interp((x, y), data, q; bc = bc)
        linear_interp((x, y), data, q; bc = bc)
        return @allocated linear_interp((x, y), data, q; bc = bc)
    end

    function _alloc_linear_nd_vector()
        x = collect(range(0.0, step = 2π / 8, length = 8))
        y = collect(range(0.0, step = 2π / 8, length = 8))
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        bc_axis = PeriodicBC(endpoint = :exclusive, period = 2π)
        bcs = (bc_axis, bc_axis)
        q = (1.0, 2.0)
        linear_interp((x, y), data, q; bc = bcs)
        linear_interp((x, y), data, q; bc = bcs)
        return @allocated linear_interp((x, y), data, q; bc = bcs)
    end

    function _alloc_linear_nd_mixed()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y = range(0.0, 1.0, length = 7)
        data = rand(length(x), length(y))
        bcs = (PeriodicBC(endpoint = :exclusive, period = 2π), NoBC())
        q = (1.0, 0.5)
        linear_interp((x, y), data, q; bc = bcs)
        linear_interp((x, y), data, q; bc = bcs)
        return @allocated linear_interp((x, y), data, q; bc = bcs)
    end

    @testset "One-shot :exclusive zero-alloc — 1D Range scalar" begin
        @test _alloc_linear_1d_range_scalar() <= ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — 1D Range vector in-place" begin
        @test _alloc_linear_1d_range_vec() <= ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — 1D Vector scalar (pool)" begin
        @test _alloc_linear_1d_vector_scalar() <= ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — 1D Vector vector in-place (pool)" begin
        @test _alloc_linear_1d_vector_vec() <= ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — ND Range grids" begin
        @test _alloc_linear_nd_range() <= ND_ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — ND Vector grids (pool)" begin
        @test _alloc_linear_nd_vector() <= ND_ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — ND mixed (periodic Vector + non-periodic Range)" begin
        @test _alloc_linear_nd_mixed() <= ND_ALLOC_THRESHOLD
    end

    @testset "ND — Vector grid :exclusive without period raises" begin
        x = [0.0, 1.0, 2.0, 3.0]          # Vector
        y = range(0.0, 1.0, length = 4)
        data = rand(4, 4)
        @test_throws ArgumentError linear_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive), NoBC())
        )
    end

    # ============================================================
    # Integer grid + duck-type (Dual) smoke tests.
    # The periodic extension helpers use `_PromotableValue` to decide whether
    # to `float(Tg)`-promote. Integer grids must work (previously threw
    # InexactError on `_CachedRange{Int}`), duck grids must pass through
    # with their original type (AD chains preserved).
    # ============================================================
    @testset "Int Range + PeriodicBC(:exclusive) — persistent" begin
        x = 0:10                                       # UnitRange{Int}
        y = sin.(2π .* (x ./ 10))
        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 11.0))
        @test itp.x isa FastInterpolations._ExclusivePeriodicAxis
        @test itp.x.inner isa _CachedRange{Float64}    # Int Range → Float _CachedRange (cached)
        @test length(itp.x) == 12                      # virtual N+1
        @test length(itp.x.inner) == 11                # raw N
        # Query agrees with Float-equivalent grid
        ref = linear_interp(Float64.(0:10), y; bc = PeriodicBC(endpoint = :exclusive, period = 11.0))
        for xq in (0.5, 5.5, 10.5)
            @test itp(xq) ≈ ref(xq) atol = 1.0e-12
        end
    end

    @testset "Int Range + PeriodicBC(:exclusive) — oneshot" begin
        x = 0:10
        y = sin.(2π .* (x ./ 10))
        bc = PeriodicBC(endpoint = :exclusive, period = 11.0)
        for xq in (0.5, 5.5, 10.5)
            @test linear_interp(x, y, xq; bc = bc) isa Float64
        end
    end

    @testset "Int Vector + PeriodicBC(:exclusive) with Float period — oneshot" begin
        # The edge case: Vector{Int} without Range-auto-promotion, Float period.
        x = [0, 1, 2, 3]
        y = [0.0, 1.0, 0.0, -1.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        @test linear_interp(x, y, 1.5; bc = bc) isa Float64
        ref = linear_interp(Float64.(x), y, 1.5; bc = bc)
        @test linear_interp(x, y, 1.5; bc = bc) ≈ ref atol = 1.0e-12
    end

    # Duck-type grid (ForwardDiff.Dual) + PeriodicBC coverage lives in
    # `test/ext/test_linear_dual_grid.jl` alongside the existing Dual tests,
    # so base test runs do not pull in ForwardDiff.

    # ============================================================
    # Float32 + Complex smoke tests (precision/value-type coverage).
    # ============================================================
    @testset "Float32 Range + PeriodicBC(:exclusive)" begin
        x = range(0.0f0, step = Float32(2π / 16), length = 16)
        y = sin.(x)                                                # Vector{Float32}
        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
        @test eltype(itp.x) === Float32
        @test itp(1.0f0) isa Float32
        # Wrap equivalence at the seam
        @test itp(0.0f0) ≈ itp(Float32(step(x) * length(x))) atol = Float32(1.0e-6)
    end

    @testset "ComplexF64 values + PeriodicBC(:exclusive)" begin
        x = collect(range(0.0, 2π, length = 17))
        y = @. exp(im * x)                                         # Vector{ComplexF64}, y[1] == y[end]
        # :inclusive path
        itp_inc = linear_interp(x, y; bc = PeriodicBC())
        @test itp_inc(0.0) ≈ itp_inc(2π) atol = 1.0e-12
        @test itp_inc(2π + 0.5) ≈ itp_inc(0.5) atol = 1.0e-12

        # :exclusive path, Vector of length 16 (drop repeated endpoint)
        x_ex = collect(range(0.0, step = 2π / 16, length = 16))
        y_ex = @. exp(im * x_ex)
        itp_ex = linear_interp(x_ex, y_ex; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        @test eltype(itp_ex.y) <: Complex
        @test itp_ex(0.5) isa Complex
    end

    # ============================================================
    # Series + PeriodicBC
    # ============================================================
    @testset "Series persistent + PeriodicBC — matches per-series interpolants" begin
        x = collect(range(0.0, 2π, length = 17))
        y1 = sin.(x)
        y2 = cos.(x)
        y3 = @. sin(2 * x)
        # :inclusive (endpoints match for sin/cos/sin(2x) at 0, 2π)
        sitp = linear_interp(x, Series(y1, y2, y3); bc = PeriodicBC())
        itp1 = linear_interp(x, y1; bc = PeriodicBC())
        itp2 = linear_interp(x, y2; bc = PeriodicBC())
        itp3 = linear_interp(x, y3; bc = PeriodicBC())
        for xq in (0.3, 1.7, 2π + 0.5, -0.2)
            out = sitp(xq)
            @test out[1] ≈ itp1(xq) atol = 1.0e-12
            @test out[2] ≈ itp2(xq) atol = 1.0e-12
            @test out[3] ≈ itp3(xq) atol = 1.0e-12
        end
    end

    @testset "Series persistent + PeriodicBC(:exclusive)" begin
        # Drop the repeated endpoint — both series, same length as x
        x = collect(range(0.0, step = 2π / 16, length = 16))
        y1 = sin.(x)
        y2 = cos.(x)
        sitp = linear_interp(x, Series(y1, y2); bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        # Wrap test: query past the domain should equal the wrapped query
        @test sitp(0.3)[1] ≈ sitp(2π + 0.3)[1] atol = 1.0e-12
        @test sitp(0.3)[2] ≈ sitp(2π + 0.3)[2] atol = 1.0e-12
    end

    @testset "Series persistent + PeriodicBC :inclusive mismatch raises" begin
        x = collect(range(0.0, 2π, length = 17))
        y1 = sin.(x)
        y2 = collect(0.0:16)             # y2[1] != y2[end]
        @test_throws ArgumentError linear_interp(x, Series(y1, y2); bc = PeriodicBC())
    end

    @testset "Series oneshot scalar + PeriodicBC agrees with persistent" begin
        x = collect(range(0.0, step = 2π / 16, length = 16))
        y1 = sin.(x)
        y2 = cos.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        sitp = linear_interp(x, Series(y1, y2); bc = bc)
        for xq in (0.0, 0.5, 2.0, 5.0, -0.1, 2π + 0.1)
            oneshot = linear_interp(x, Series(y1, y2), xq; bc = bc)
            @test oneshot ≈ sitp(xq) atol = 1.0e-12
        end
    end

    @testset "Series oneshot vector + PeriodicBC" begin
        x = collect(range(0.0, step = 2π / 16, length = 16))
        y1 = sin.(x)
        y2 = cos.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        sitp = linear_interp(x, Series(y1, y2); bc = bc)

        xqs = [0.0, 0.5, 2.0, -0.1]
        out_vec = linear_interp(x, Series(y1, y2), xqs; bc = bc)
        @test length(out_vec) == 2
        @test length(out_vec[1]) == length(xqs)
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            @test out_vec[1][j] ≈ ref[1] atol = 1.0e-12
            @test out_vec[2][j] ≈ ref[2] atol = 1.0e-12
        end

        # In-place variant
        outs = [similar(xqs) for _ in 1:2]
        linear_interp!(outs, x, Series(y1, y2), xqs; bc = bc)
        @test outs[1] ≈ out_vec[1] atol = 1.0e-12
        @test outs[2] ≈ out_vec[2] atol = 1.0e-12
    end

    # ============================================================
    # Series OneShot Scalar + PeriodicBC — Zero-Copy Migration (A-2)
    # ============================================================
    # These tests drive and lock down the scalar-series zero-copy refactor
    # (TODO: linear_constant_series_zero_copy.md, Stage 1).
    # Function-barrier pattern mirrors the non-series alloc-guards above.

    function _alloc_linear_series_scalar_range_exclusive()
        x = range(0.0, step = 2π / 16, length = 16)
        y1 = sin.(x)
        y2 = cos.(x)
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        out = Vector{Float64}(undef, 2)
        linear_interp!(out, x, s, 1.0; bc = bc)
        linear_interp!(out, x, s, 1.0; bc = bc)
        return @allocated linear_interp!(out, x, s, 1.0; bc = bc)
    end

    function _alloc_linear_series_scalar_vector_exclusive()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y1 = sin.(x)
        y2 = cos.(x)
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        out = Vector{Float64}(undef, 2)
        linear_interp!(out, x, s, 1.0; bc = bc)
        linear_interp!(out, x, s, 1.0; bc = bc)
        return @allocated linear_interp!(out, x, s, 1.0; bc = bc)
    end

    function _alloc_linear_series_scalar_inclusive()
        x = collect(range(0.0, 2π, length = 17))
        y1 = sin.(x)                          # y1[1] == y1[end] == 0
        y2 = cos.(x)                          # y2[1] == y2[end] == 1
        s = Series(y1, y2)
        bc = PeriodicBC()                     # :inclusive default
        out = Vector{Float64}(undef, 2)
        linear_interp!(out, x, s, 1.0; bc = bc)
        linear_interp!(out, x, s, 1.0; bc = bc)
        return @allocated linear_interp!(out, x, s, 1.0; bc = bc)
    end

    @testset "Series scalar + PeriodicBC zero-alloc — Range exclusive (T-series-alloc)" begin
        @test _alloc_linear_series_scalar_range_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series scalar + PeriodicBC zero-alloc — Vector exclusive (T-series-alloc)" begin
        @test _alloc_linear_series_scalar_vector_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series scalar + PeriodicBC zero-alloc — inclusive (T-series-alloc)" begin
        @test _alloc_linear_series_scalar_inclusive() <= ALLOC_THRESHOLD
    end

    @testset "Series scalar + PeriodicBC(:exclusive) seam semantic" begin
        # Simple 4-point grid, period = 4.0 → seam cell [x[n], x[1]+period) = [3, 4)
        x = collect(0.0:3.0)
        y1 = [10.0, 20.0, 30.0, 40.0]
        y2 = [1.0, 2.0, 3.0, 4.0]
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)

        # Inside seam cell at xq = 3.5, α = 0.5 → y[n]*(1-α) + y[1]*α
        out = linear_interp(x, s, 3.5; bc = bc)
        @test out[1] ≈ 40.0 * 0.5 + 10.0 * 0.5 atol = 1.0e-12
        @test out[2] ≈ 4.0 * 0.5 + 1.0 * 0.5 atol = 1.0e-12

        # At xq = x[n] = 3.0 (α = 0 in seam cell → yL = y[n])
        out_left = linear_interp(x, s, 3.0; bc = bc)
        @test out_left[1] ≈ 40.0 atol = 1.0e-12
        @test out_left[2] ≈ 4.0 atol = 1.0e-12

        # Cross-check: series oneshot == per-series non-series scalar
        @test out[1] ≈ linear_interp(x, y1, 3.5; bc = bc) atol = 1.0e-12
        @test out[2] ≈ linear_interp(x, y2, 3.5; bc = bc) atol = 1.0e-12

        # Cross-check: series oneshot == persistent series interpolant
        sitp = linear_interp(x, s; bc = bc)
        @test out[1] ≈ sitp(3.5)[1] atol = 1.0e-12
        @test out[2] ≈ sitp(3.5)[2] atol = 1.0e-12
    end

    @testset "Series oneshot + PeriodicBC(:exclusive) preserves cached step on large-offset Range" begin
        # On `range(1e8, step=0.1, …)`, `x[i+1] - x[i]` loses precision to float
        # cancellation (~1.5e-8 ulp at 1e8), while `_CachedRange.h` stores the
        # exact step. The periodic series one-shot paths must dispatch through
        # `_get_h`/`_get_inv_h` so they match the non-series scalar and
        # persistent series evaluators on such grids.
        x = range(1.0e8, step = 0.1, length = 10)
        y1 = Float64.(1:10)
        y2 = Float64.(11:20)
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive)
        xq = 1.0e8 + 0.95  # lands in seam cell [x[10], x[1]+period)

        v_scalar = linear_interp(x, y1, xq; bc = bc)
        v_oneshot = linear_interp(x, s, xq; bc = bc)
        v_persist = linear_interp(x, s; bc = bc)(xq)

        # Scalar oneshot uses `_alpha_of(g) → _alpha_of(g.inner::_CachedRange)`
        # which delegates to cached `inner.inv_h = inv(step)` (DCE-friendly:
        # for `EvalValue` the kernel-side `_get_inv_h` is dead-code-eliminated
        # by LLVM since only α is consumed). Series oneshot routes through the
        # wrapper's seam-aware `_get_inv_h(g, n) = inv(_x_max - inner[n])`,
        # whose `(xq - xL) / (xR - xL)`-shaped α cancels structurally and
        # gives a slightly different Float64 rounding at a 1e8 offset.
        # Persistent series builds via `_prepare_periodic` (physical n+1
        # extension), so it sees a regular `_CachedRange` and uses cached
        # step like scalar. The three routes thus agree within the
        # `eps(1e8)/0.1 ≈ 5e-8`-relative Float64 floor at this grid offset.
        @test v_oneshot[1] ≈ v_scalar rtol = 1.0e-7
        @test v_oneshot[1] ≈ v_persist[1] rtol = 1.0e-7

        # Batch path uses the wrapper-based series oneshot route, so it
        # matches `v_oneshot` (and not `v_scalar`) bit-for-bit.
        xqs = [1.0e8 + 0.95, 1.0e8 + 0.55]
        outs = [similar(xqs) for _ in 1:2]
        linear_interp!(outs, x, s, xqs; bc = bc)
        @test outs[1][1] === v_oneshot[1]
        # Interior cell (xqs[2]) — no seam, no cancellation — bit-equal across routes.
        @test outs[1][2] === linear_interp(x, y1, xqs[2]; bc = bc)
    end

    # ============================================================
    # Series OneShot Vector-Batch + PeriodicBC — Zero-Copy (Stage 2)
    # ============================================================
    # Drives the Q outer × K inner zero-pool refactor. Function-barrier +
    # double-warmup pattern matches the scalar T-series-alloc convention.

    function _alloc_linear_series_vector_range_exclusive()
        x = range(0.0, step = 2π / 16, length = 16)
        s = Series(sin.(x), cos.(x))
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        linear_interp!(outs, x, s, xqs; bc = bc)
        linear_interp!(outs, x, s, xqs; bc = bc)
        return @allocated linear_interp!(outs, x, s, xqs; bc = bc)
    end

    function _alloc_linear_series_vector_vector_exclusive()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        s = Series(sin.(x), cos.(x))
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        linear_interp!(outs, x, s, xqs; bc = bc)
        linear_interp!(outs, x, s, xqs; bc = bc)
        return @allocated linear_interp!(outs, x, s, xqs; bc = bc)
    end

    function _alloc_linear_series_vector_inclusive()
        x = collect(range(0.0, 2π, length = 17))
        s = Series(sin.(x), cos.(x))
        bc = PeriodicBC()
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        linear_interp!(outs, x, s, xqs; bc = bc)
        linear_interp!(outs, x, s, xqs; bc = bc)
        return @allocated linear_interp!(outs, x, s, xqs; bc = bc)
    end

    function _alloc_linear_series_vector_nobc()
        x = range(0.0, step = 2π / 16, length = 16)
        s = Series(sin.(x), cos.(x))
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        linear_interp!(outs, x, s, xqs)
        linear_interp!(outs, x, s, xqs)
        return @allocated linear_interp!(outs, x, s, xqs)
    end

    @testset "Series vector + PeriodicBC zero-alloc — Range exclusive (T-series-alloc)" begin
        @test _alloc_linear_series_vector_range_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series vector + PeriodicBC zero-alloc — Vector exclusive (T-series-alloc)" begin
        @test _alloc_linear_series_vector_vector_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series vector + PeriodicBC zero-alloc — inclusive (T-series-alloc)" begin
        @test _alloc_linear_series_vector_inclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series vector + NoBC zero-alloc (T-series-alloc)" begin
        @test _alloc_linear_series_vector_nobc() <= ALLOC_THRESHOLD
    end

    @testset "Series vector + PeriodicBC(:exclusive) seam cell semantic" begin
        # 4-point grid, period=4.0 → seam cell [x[n], x[1]+period) = [3, 4).
        x = collect(0.0:3.0)
        y1 = [10.0, 20.0, 30.0, 40.0]
        y2 = [1.0, 2.0, 3.0, 4.0]
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)

        # Batch of xqs covering:
        #   2.5   — interior cell [2,3], α=0.5 → y[3]*0.5 + y[4]*0.5
        #   3.0   — exactly x[n], α=0 in seam cell → yL = y[n]
        #   3.25  — seam cell, α=0.25 → y[n]*0.75 + y[1]*0.25
        #   3.5   — seam mid, α=0.5 → y[n]*0.5 + y[1]*0.5
        #   3.875 — seam cell near right, α=0.875 → y[n]*0.125 + y[1]*0.875
        xqs = [2.5, 3.0, 3.25, 3.5, 3.875]
        outs = [Vector{Float64}(undef, length(xqs)) for _ in 1:2]
        linear_interp!(outs, x, s, xqs; bc = bc)

        # Linear blend expected values, y1 series
        @test outs[1][1] ≈ 30.0 * 0.5 + 40.0 * 0.5   atol = 1.0e-12  # interior
        @test outs[1][2] ≈ 40.0                        atol = 1.0e-12  # at x[n]
        @test outs[1][3] ≈ 40.0 * 0.75 + 10.0 * 0.25  atol = 1.0e-12  # seam 25%
        @test outs[1][4] ≈ 40.0 * 0.5 + 10.0 * 0.5   atol = 1.0e-12  # seam mid
        @test outs[1][5] ≈ 40.0 * 0.125 + 10.0 * 0.875 atol = 1.0e-12  # seam 87.5%

        # Linear blend expected values, y2 series
        @test outs[2][1] ≈ 3.0 * 0.5 + 4.0 * 0.5   atol = 1.0e-12
        @test outs[2][2] ≈ 4.0                        atol = 1.0e-12
        @test outs[2][3] ≈ 4.0 * 0.75 + 1.0 * 0.25  atol = 1.0e-12
        @test outs[2][4] ≈ 4.0 * 0.5 + 1.0 * 0.5   atol = 1.0e-12
        @test outs[2][5] ≈ 4.0 * 0.125 + 1.0 * 0.875 atol = 1.0e-12

        # Cross-check: batch path agrees with per-query scalar path, series-wise.
        for j in eachindex(xqs)
            scalar_out = linear_interp(x, s, xqs[j]; bc = bc)
            @test outs[1][j] ≈ scalar_out[1] atol = 1.0e-12
            @test outs[2][j] ≈ scalar_out[2] atol = 1.0e-12
        end

        # Cross-check: batch path agrees with persistent interpolants (non-series).
        itp1 = linear_interp(x, y1; bc = bc)
        itp2 = linear_interp(x, y2; bc = bc)
        for j in eachindex(xqs)
            @test outs[1][j] ≈ itp1(xqs[j]) atol = 1.0e-12
            @test outs[2][j] ≈ itp2(xqs[j]) atol = 1.0e-12
        end
    end

    # ============================================================
    # Persistent Series callable + PeriodicBC — Zero-Copy (Stage 3)
    # ============================================================
    # `sitp(outs, xqs)` callable must be zero-alloc under Q outer × K inner
    # refactor. These guards complement the existing non-periodic alloc
    # tests in test_linear_series_interp.jl (lines 101-121).

    function _alloc_linear_persistent_vector_range_exclusive()
        x = range(0.0, step = 2π / 16, length = 16)
        s = Series(sin.(x), cos.(x))
        sitp = linear_interp(x, s; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        sitp(outs, xqs); sitp(outs, xqs)
        return @allocated sitp(outs, xqs)
    end

    function _alloc_linear_persistent_vector_vector_exclusive()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        s = Series(sin.(x), cos.(x))
        sitp = linear_interp(x, s; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        sitp(outs, xqs); sitp(outs, xqs)
        return @allocated sitp(outs, xqs)
    end

    function _alloc_linear_persistent_vector_inclusive()
        x = collect(range(0.0, 2π, length = 17))
        s = Series(sin.(x), cos.(x))
        sitp = linear_interp(x, s; bc = PeriodicBC())
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        sitp(outs, xqs); sitp(outs, xqs)
        return @allocated sitp(outs, xqs)
    end

    @testset "Persistent callable + PeriodicBC zero-alloc — Range exclusive (T-persistent-alloc)" begin
        @test _alloc_linear_persistent_vector_range_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Persistent callable + PeriodicBC zero-alloc — Vector exclusive (T-persistent-alloc)" begin
        @test _alloc_linear_persistent_vector_vector_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Persistent callable + PeriodicBC zero-alloc — inclusive (T-persistent-alloc)" begin
        @test _alloc_linear_persistent_vector_inclusive() <= ALLOC_THRESHOLD
    end

end
