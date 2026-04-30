# Tests for PeriodicBC on Constant interpolation (Phase 1).
#
# Coverage mirrors test_linear_periodic.jl, with extra per-`side` coverage
# (NearestSide/LeftSide/RightSide must all respect periodicity).

@testitem "Constant PeriodicBC" setup = [AllocConstants] begin
    using FastInterpolations: _CachedRange

    @testset "NoBC default is no-op" begin
        x = collect(range(0.0, 1.0, length = 11))
        y = sin.(2π .* x)

        itp_default = constant_interp(x, y)
        itp_nobc = constant_interp(x, y; bc = NoBC())

        @test typeof(itp_default) === typeof(itp_nobc)
        for xq in (0.0, 0.25, 0.5, 0.75, 1.0)
            @test itp_default(xq) === itp_nobc(xq)
        end
    end

    @testset "Inclusive — endpoint mismatch raises" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError constant_interp(x, y; bc = PeriodicBC())
    end

    @testset "Inclusive — endpoint mismatch raises on 1D oneshot (scalar + vector)" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError constant_interp(x, y, 0.5; bc = PeriodicBC())
        @test_throws ArgumentError constant_interp(x, y, [0.25, 0.75]; bc = PeriodicBC())
    end

    @testset "Inclusive — endpoint check=false skips validation on 1D oneshot" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        bc = PeriodicBC(endpoint = :inclusive, check = false)
        @test constant_interp(x, y, 0.5; bc = bc) isa Real
        @test constant_interp(x, y, [0.25, 0.75]; bc = bc) isa AbstractVector
    end

    @testset "Exclusive — FVM cell-centered, LeftSide" begin
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp = constant_interp(x, y; bc = bc, side = LeftSide())

        # LeftSide with extended endpoint: segment [2.5, 3.5] has left-value = 30.0
        @test itp(2.6) ≈ 30.0 atol = 1.0e-12
        @test itp(3.4) ≈ 30.0 atol = 1.0e-12   # still within [2.5, 3.5] → 30.0

        # Wrap: query outside [0.5, 3.5) wraps back
        @test itp(3.6) ≈ itp(0.6) atol = 1.0e-12
    end

    @testset "Exclusive — RightSide" begin
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp = constant_interp(x, y; bc = bc, side = RightSide())

        # RightSide on segment [2.5, 3.5] → right endpoint value = y[n+1] = y[1] = 10.0
        @test itp(2.6) ≈ 10.0 atol = 1.0e-12
    end

    @testset "Exclusive — Vector grid (allocating path)" begin
        x_vec = [0.5, 1.5, 2.5]
        y = [10.0, 20.0, 30.0]
        itp = constant_interp(x_vec, y; bc = PeriodicBC(endpoint = :exclusive, period = 3.0))

        # NearestSide default — closest cell center
        @test itp(2.5) ≈ 30.0 atol = 1.0e-12
        @test itp(3.4) ≈ 10.0 atol = 1.0e-12   # closer to virtual x=3.5 (value=10) than to 2.5
    end

    @testset "Extrap is forced to WrapExtrap on periodic path" begin
        x = range(0.0, 2π, length = 11)
        y = sin.(x)
        itp = constant_interp(x, y; bc = PeriodicBC(), extrap = ClampExtrap())
        @test itp.extrap isa WrapExtrap
    end

    @testset "Exclusive — auto-infer period on 1D + ND oneshot (regression)" begin
        # See test_linear_periodic.jl for the parent regression note.
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc_auto = PeriodicBC(endpoint = :exclusive)
        bc_expl = PeriodicBC(endpoint = :exclusive, period = 3.0)

        # 1D scalar + vector oneshot at and near seam
        @test constant_interp(x, y, 3.0; bc = bc_auto, side = LeftSide()) ≈
            constant_interp(x, y, 3.0; bc = bc_expl, side = LeftSide()) atol = 1.0e-12
        xq = [0.5, 2.5, 3.0, 3.4]
        @test constant_interp(x, y, xq; bc = bc_auto) ≈
            constant_interp(x, y, xq; bc = bc_expl) atol = 1.0e-12

        # ND oneshot with mixed BC
        x2 = range(0.0, 1.0, length = 4)
        data = [10xi + yj for xi in x, yj in x2]
        q = (3.0, 0.5)
        @test constant_interp((x, x2), data, q; bc = (bc_auto, NoBC())) ≈
            constant_interp((x, x2), data, q; bc = (bc_expl, NoBC())) atol = 1.0e-12
    end

    @testset "Exclusive — Vector grid seam at xq == x[n] exactly (T-1)" begin
        x_vec = [0.0, 0.25, 0.5, 0.75]
        y = [10.0, 20.0, 30.0, 40.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        itp = constant_interp(x_vec, y; bc = bc, side = LeftSide())
        @test itp(0.75) ≈ 40.0 atol = 1.0e-12
        @test constant_interp(x_vec, y, 0.75; bc = bc, side = LeftSide()) ≈ 40.0 atol = 1.0e-12
    end

    @testset "Exclusive — ND persistent rejects period-too-small at build (T-3)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = range(0.0, 1.0, length = 4)
        data = rand(4, 4)
        @test_throws ArgumentError constant_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive, period = 2.5), NoBC()),
        )
    end

    @testset "Exclusive — ND axis-2 periodic only (T-6)" begin
        x = range(0.0, 1.0, length = 5)
        y = range(0.5, step = 1.0, length = 3)
        data = [i + 10j for i in 1:5, j in 1:3]
        bc = (NoBC(), PeriodicBC(endpoint = :exclusive, period = 3.0))
        itp = constant_interp((x, y), data; bc = bc)
        @test itp((0.5, 0.5)) ≈ itp((0.5, 3.5)) atol = 1.0e-12
        @test constant_interp((x, y), data, (0.5, 3.0); bc = bc) ≈ itp((0.5, 3.0)) atol = 1.0e-12
    end

    @testset "Exclusive — 1D RightSide at seam edges (T-7)" begin
        # Grid-point convention: at `xq == x[i]` exactly, constant interp returns
        # `y[i]` regardless of `side` (the point is unambiguous). The seam-wrap
        # to y[1] only activates for xq STRICTLY inside the seam cell (xq > x[n]).
        # This testset pins down both edges of the seam cell for RightSide.
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp_R = constant_interp(x, y; bc = bc, side = RightSide())
        itp_L = constant_interp(x, y; bc = bc, side = LeftSide())

        # At grid point x[n]: both sides agree on y[n].
        @test itp_R(2.5) ≈ 30.0 atol = 1.0e-12
        @test itp_L(2.5) ≈ 30.0 atol = 1.0e-12
        @test constant_interp(x, y, 2.5; bc = bc, side = RightSide()) ≈ 30.0 atol = 1.0e-12

        # Strictly inside seam (xq > x[n]): RightSide reads virtual right corner = y[1].
        @test itp_R(2.7) ≈ 10.0 atol = 1.0e-12
        @test constant_interp(x, y, 2.7; bc = bc, side = RightSide()) ≈ 10.0 atol = 1.0e-12

        # Approaching virtual endpoint from inside still reads y[1] (wrap).
        @test itp_R(3.4999) ≈ 10.0 atol = 1.0e-12
    end

    @testset "Oneshot scalar + vector + in-place — matches persistent" begin
        x = range(0.5, step = 1.0, length = 3)
        y = [10.0, 20.0, 30.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 3.0)
        itp = constant_interp(x, y; bc = bc)

        for xq in (0.5, 1.5, 2.5, 3.0, 3.4, 0.0, -0.5)
            @test constant_interp(x, y, xq; bc = bc) ≈ itp(xq) atol = 1.0e-12
        end

        xq = [0.5, 1.0, 2.5, 3.0, 3.4]
        @test constant_interp(x, y, xq; bc = bc) ≈ itp.(xq) atol = 1.0e-12

        out = zeros(length(xq))
        constant_interp!(out, x, y, xq; bc = bc)
        @test out ≈ itp.(xq) atol = 1.0e-12
    end

    # ============================================================
    # ND tests
    # ============================================================
    @testset "ND — scalar bc, periodic both axes" begin
        x = range(0.5, step = 1.0, length = 3)
        y = range(0.5, step = 1.0, length = 4)
        data = [10.0 * i + j for i in 1:3, j in 1:4]

        bc = PeriodicBC(endpoint = :exclusive)
        itp = constant_interp((x, y), data; bc = bc)

        # ND persistent path: extended Range axes must remain _CachedRange.
        @test itp.grids[1] isa _CachedRange
        @test itp.grids[2] isa _CachedRange
        @test length(itp.grids[1]) == 4
        @test length(itp.grids[2]) == 5
        @test size(itp.data) == (4, 5)

        q = (1.2, 0.8)
        @test itp(q) ≈ constant_interp((x, y), data, q; bc = bc) atol = 1.0e-12
    end

    @testset "ND — per-axis mix (periodic + non-periodic)" begin
        x = range(0.5, step = 1.0, length = 3)
        y = collect(range(0.0, 1.0, length = 5))
        data = [10.0 * i + j for i in 1:3, j in 1:5]
        bc = (PeriodicBC(endpoint = :exclusive, period = 3.0), NoBC())
        itp = constant_interp((x, y), data; bc = bc)

        @test itp((0.5, 0.5)) ≈ itp((3.5, 0.5)) atol = 1.0e-12
    end

    @testset "ND — NoBC default is no-op" begin
        x = range(0.0, 1.0, length = 5)
        y = range(0.0, 1.0, length = 4)
        data = rand(5, 4)

        itp_default = constant_interp((x, y), data)
        itp_nobc = constant_interp((x, y), data; bc = NoBC())

        for q in ((0.3, 0.5), (0.9, 0.1), (0.0, 0.0), (1.0, 1.0))
            @test itp_default(q) === itp_nobc(q)
        end
    end

    @testset "ND — incompatible extrap with PeriodicBC raises" begin
        x = range(0.5, step = 1.0, length = 3)
        y = range(0.0, 1.0, length = 5)
        data = rand(3, 5)
        @test_throws ArgumentError constant_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive, period = 3.0), NoBC()),
            extrap = (ClampExtrap(), NoExtrap())
        )
    end

    @testset "ND — :inclusive slice mismatch raises on build" begin
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))
        good = [sin(xi) * cos(yj) for xi in x, yj in y]
        bad = copy(good)
        bad[end, :] .+= 1.0

        @test_throws ArgumentError constant_interp((x, y), bad; bc = PeriodicBC())
        @test constant_interp((x, y), good; bc = PeriodicBC()) isa ConstantInterpolantND
    end

    @testset "ND — :inclusive slice mismatch raises on oneshot" begin
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))
        bad = [sin(xi) * cos(yj) for xi in x, yj in y]
        bad[:, end] .+= 1.0

        q = (1.0, 1.0)
        @test_throws ArgumentError constant_interp((x, y), bad, q; bc = PeriodicBC())
    end

    @testset "ND — :inclusive + check=false skips validation" begin
        x = range(0.0, 2π, length = 5)
        y = range(0.0, 2π, length = 5)
        bad = [sin(xi) * cos(yj) for xi in x, yj in y]
        bad[end, :] .+= 1.0
        itp = constant_interp(
            (x, y), bad;
            bc = (
                PeriodicBC(endpoint = :inclusive, check = false),
                PeriodicBC(endpoint = :inclusive, check = false),
            )
        )
        @test itp isa ConstantInterpolantND
    end

    @testset "ND seam-cell — _constant_nd_kernel exact wrap" begin
        # 2D, axis 1 :exclusive (period=4), axis 2 NoBC. Constant uses one of
        # the (idx_L, idx_R) corners per axis according to `side`. Pinning down
        # LeftSide and RightSide along the seam axis is the strongest unit-level
        # check on the kernel: oneshot must agree with persistent for both.
        x = collect(range(0.0, step = 1.0, length = 4))     # period 4
        yy = collect(range(0.0, step = 1.0, length = 3))
        data = [Float64(i - 1) + 10 * Float64(j - 1) for i in 1:4, j in 1:3]

        bc = (PeriodicBC(endpoint = :exclusive, period = 4.0), NoBC())
        extrap = (NoExtrap(), NoExtrap())

        # Seam cell axis 1 at q=(3.5, 0.5):
        #   axis-1 corners are data[4, j] (left, x=3) and data[1, j] (right, wrapped to x=4)
        #   axis-2 corners are data[i, 1] (left, y=0) and data[i, 2] (right, y=1)
        q = (3.5, 0.5)

        # LeftSide on both axes ⇒ pick (4, 1) ⇒ data[4, 1] = 3.0
        side_LL = (LeftSide(), LeftSide())
        @test constant_interp((x, yy), data, q; bc = bc, extrap = extrap, side = side_LL) == 3.0
        itp_LL = constant_interp((x, yy), data; bc = bc, extrap = extrap, side = side_LL)
        @test itp_LL(q) == 3.0

        # RightSide on axis 1, LeftSide on axis 2 ⇒ pick (1, 1) ⇒ data[1, 1] = 0.0
        # (axis 1 right corner wraps to data[1, j])
        side_RL = (RightSide(), LeftSide())
        @test constant_interp((x, yy), data, q; bc = bc, extrap = extrap, side = side_RL) == 0.0
        itp_RL = constant_interp((x, yy), data; bc = bc, extrap = extrap, side = side_RL)
        @test itp_RL(q) == 0.0

        # Both RightSide ⇒ data[1, 2] = 10.0
        side_RR = (RightSide(), RightSide())
        @test constant_interp((x, yy), data, q; bc = bc, extrap = extrap, side = side_RR) == 10.0

        # Just below seam (no wrap): LeftSide picks data[3, 1] = 2.0
        q_pre = (2.9, 0.5)
        @test constant_interp((x, yy), data, q_pre; bc = bc, extrap = extrap, side = side_LL) == 2.0
    end

    @testset "ND seam-cell — _constant_nd_kernel at N=3 with periodic axis-1 wrap" begin
        # The unified `_constant_nd_kernel` is @generated and addresses one of
        # 2^N corners per query via runtime offsets. Existing seam-cell tests
        # are N=2 only; this exercises the N=3 unroll with axis-1 wrap, pinning
        # the `ifelse(offset==0, stencils[d][1], stencils[d][2])` corner-pick
        # for any N.
        x = collect(range(0.0, step = 1.0, length = 4))    # axis 1 periodic, period 4
        yy = collect(range(0.0, step = 1.0, length = 3))   # axis 2 NoBC
        zz = collect(range(0.0, step = 1.0, length = 3))   # axis 3 NoBC
        data = [
            Float64(i - 1) + 10 * Float64(j - 1) + 100 * Float64(k - 1)
                for i in 1:4, j in 1:3, k in 1:3
        ]

        bc = (PeriodicBC(endpoint = :exclusive, period = 4.0), NoBC(), NoBC())
        extrap = (NoExtrap(), NoExtrap(), NoExtrap())

        # Seam cell axis 1 at q=(3.5, 0.5, 0.5): axis-1 corners are
        # data[4, j, k] (left) and data[1, j, k] (right, wrapped).
        q = (3.5, 0.5, 0.5)

        # All-LeftSide ⇒ pick (4, 1, 1) ⇒ data[4, 1, 1] = 3.0
        side_LLL = (LeftSide(), LeftSide(), LeftSide())
        @test constant_interp((x, yy, zz), data, q; bc = bc, extrap = extrap, side = side_LLL) == 3.0
        itp_LLL = constant_interp((x, yy, zz), data; bc = bc, extrap = extrap, side = side_LLL)
        @test itp_LLL(q) == 3.0

        # RightSide axis 1 ⇒ wraps to (1, 1, 1) ⇒ data[1, 1, 1] = 0.0
        side_RLL = (RightSide(), LeftSide(), LeftSide())
        @test constant_interp((x, yy, zz), data, q; bc = bc, extrap = extrap, side = side_RLL) == 0.0

        # All-RightSide ⇒ (1, 2, 2) ⇒ data[1, 2, 2] = 0 + 10 + 100 = 110.0
        side_RRR = (RightSide(), RightSide(), RightSide())
        @test constant_interp((x, yy, zz), data, q; bc = bc, extrap = extrap, side = side_RRR) == 110.0
    end

    # ============================================================
    # Edge cases
    # ============================================================
    @testset "Edge — Vector grid :exclusive without period raises" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = sin.(x)
        @test_throws ArgumentError constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
    end

    @testset "Edge — Range grid + conflicting period raises" begin
        x = range(0.0, step = 0.1, length = 10)
        y = sin.(x)
        @test_throws ArgumentError constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 2.0))
    end

    @testset "Edge — oneshot Vector grid :exclusive without period raises" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = sin.(x)
        @test_throws ArgumentError constant_interp(x, y, 1.5; bc = PeriodicBC(endpoint = :exclusive))
    end

    @testset "Edge — oneshot Vector grid :exclusive period too small raises" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = sin.(x)
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError constant_interp(x, y, 1.5; bc = bc_bad)
        @test_throws ArgumentError constant_interp(x, y, [1.5, 2.5]; bc = bc_bad)
    end

    # ============================================================
    # Interpolant path — extended copy storage
    # ============================================================
    @testset "Interpolant path stores Vector grid in `_ExclusivePeriodicAxis` + y in `_ExclusivePeriodicData`" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]
        x_ref = copy(x)
        y_ref = copy(y)

        itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 4.0))

        # Both axis and data are wrapped — zero-copy presents virtual length n+1.
        @test itp.x isa FastInterpolations._ExclusivePeriodicAxis
        @test itp.y isa FastInterpolations._ExclusivePeriodicData
        @test length(itp.x) == 5
        @test length(itp.y) == 5
        @test itp.y[end] == itp.y[1] == 10.0    # cyclic via data wrapper
        @test last(itp.x) ≈ 4.0                  # axis: inner[1] + period (virtual coord)
        # Plain `itp.x[5]` BoundsErrors (axis wrapper has no auto-cycle on `getindex`);
        # use `_getindex` for the virtual coord:
        @test FastInterpolations._getindex(itp.x, 5) ≈ 4.0

        # Original arrays unmodified.
        @test x == x_ref
        @test y == y_ref
    end

    @testset "Interpolant path stores extended copy (Range grid → _CachedRange)" begin
        x = range(0.0, step = 1.0, length = 4)
        y = [10.0, 20.0, 30.0, 40.0]

        itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))

        # Range input → extended grid must be _CachedRange (O(1) indexing preserved).
        @test itp.x isa _CachedRange
        @test length(itp.x) == 5
        @test length(itp.y) == 5
        @test itp.y[end] == itp.y[1]
    end

    # ============================================================
    # Zero-allocation tests (one-shot, after warmup)
    # ============================================================
    # Function-barrier pattern for alloc tests (same rationale as Linear).
    function _alloc_const_1d_range_scalar(side)
        x = range(0.0, step = 2π / 16, length = 16)
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive)
        constant_interp(x, y, 1.0; bc = bc, side = side)
        constant_interp(x, y, 1.0; bc = bc, side = side)
        return @allocated constant_interp(x, y, 1.0; bc = bc, side = side)
    end

    function _alloc_const_1d_range_vec(side)
        x = range(0.0, step = 2π / 16, length = 16)
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive)
        xq = [0.5, 1.0, 2.0, 3.5]
        out = similar(xq)
        constant_interp!(out, x, y, xq; bc = bc, side = side)
        constant_interp!(out, x, y, xq; bc = bc, side = side)
        return @allocated constant_interp!(out, x, y, xq; bc = bc, side = side)
    end

    function _alloc_const_1d_vector_scalar()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        constant_interp(x, y, 1.0; bc = bc)
        constant_interp(x, y, 1.0; bc = bc)
        return @allocated constant_interp(x, y, 1.0; bc = bc)
    end

    function _alloc_const_1d_vector_vec()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y = sin.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        xq = [0.5, 1.0, 2.0, 3.5]
        out = similar(xq)
        constant_interp!(out, x, y, xq; bc = bc)
        constant_interp!(out, x, y, xq; bc = bc)
        return @allocated constant_interp!(out, x, y, xq; bc = bc)
    end

    function _alloc_const_nd_range()
        x = range(0.0, step = 2π / 8, length = 8)
        y = range(0.0, step = 2π / 8, length = 8)
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        bc = PeriodicBC(endpoint = :exclusive)
        q = (1.0, 2.0)
        constant_interp((x, y), data, q; bc = bc)
        constant_interp((x, y), data, q; bc = bc)
        return @allocated constant_interp((x, y), data, q; bc = bc)
    end

    function _alloc_const_nd_vector()
        x = collect(range(0.0, step = 2π / 8, length = 8))
        y = collect(range(0.0, step = 2π / 8, length = 8))
        data = [sin(xi) * cos(yj) for xi in x, yj in y]
        bc_axis = PeriodicBC(endpoint = :exclusive, period = 2π)
        bcs = (bc_axis, bc_axis)
        q = (1.0, 2.0)
        constant_interp((x, y), data, q; bc = bcs)
        constant_interp((x, y), data, q; bc = bcs)
        return @allocated constant_interp((x, y), data, q; bc = bcs)
    end

    function _alloc_const_nd_mixed()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y = range(0.0, 1.0, length = 7)
        data = rand(length(x), length(y))
        bcs = (PeriodicBC(endpoint = :exclusive, period = 2π), NoBC())
        q = (1.0, 0.5)
        constant_interp((x, y), data, q; bc = bcs)
        constant_interp((x, y), data, q; bc = bcs)
        return @allocated constant_interp((x, y), data, q; bc = bcs)
    end

    @testset "One-shot :exclusive zero-alloc — 1D Range (all sides)" begin
        for side in (NearestSide(), LeftSide(), RightSide())
            @test _alloc_const_1d_range_scalar(side) <= ALLOC_THRESHOLD
            @test _alloc_const_1d_range_vec(side) <= ALLOC_THRESHOLD
        end
    end
    @testset "One-shot :exclusive zero-alloc — 1D Vector scalar (pool)" begin
        @test _alloc_const_1d_vector_scalar() <= ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — 1D Vector vector in-place (pool)" begin
        @test _alloc_const_1d_vector_vec() <= ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — ND Range grids" begin
        @test _alloc_const_nd_range() <= ND_ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — ND Vector grids (pool)" begin
        @test _alloc_const_nd_vector() <= ND_ALLOC_THRESHOLD
    end
    @testset "One-shot :exclusive zero-alloc — ND mixed (periodic Vector + non-periodic Range)" begin
        @test _alloc_const_nd_mixed() <= ND_ALLOC_THRESHOLD
    end

    @testset "ND — Vector grid :exclusive without period raises" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = range(0.0, 1.0, length = 4)
        data = rand(4, 4)
        @test_throws ArgumentError constant_interp(
            (x, y), data;
            bc = (PeriodicBC(endpoint = :exclusive), NoBC())
        )
    end

    # ============================================================
    # Integer grid + duck-type (Dual) smoke tests. See
    # test_linear_periodic.jl for the rationale — same `_PromotableValue`
    # gating in `_extend_exclusive` / `_periodic_extend_1d_pooled!`.
    # ============================================================
    @testset "Int Range + PeriodicBC(:exclusive) — persistent" begin
        x = 0:10
        y = Float64.(0:10)
        itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 11.0))
        @test itp.x isa _CachedRange{Float64}
        @test length(itp.x) == 12
        ref = constant_interp(Float64.(0:10), y; bc = PeriodicBC(endpoint = :exclusive, period = 11.0))
        for xq in (0.5, 5.5, 10.5)
            @test itp(xq) ≈ ref(xq) atol = 1.0e-12
        end
    end

    @testset "Int Range + PeriodicBC(:exclusive) — oneshot" begin
        x = 0:10
        y = Float64.(0:10)
        bc = PeriodicBC(endpoint = :exclusive, period = 11.0)
        for xq in (0.5, 5.5, 10.5)
            @test constant_interp(x, y, xq; bc = bc) isa Float64
        end
    end

    @testset "Int Vector + PeriodicBC(:exclusive) with Float period — oneshot" begin
        x = [0, 1, 2, 3]
        y = [10.0, 20.0, 30.0, 40.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        @test constant_interp(x, y, 1.5; bc = bc) isa Float64
        ref = constant_interp(Float64.(x), y, 1.5; bc = bc)
        @test constant_interp(x, y, 1.5; bc = bc) ≈ ref atol = 1.0e-12
    end

    # Dual-grid + PeriodicBC coverage lives in
    # `test/ext/test_constant_quadratic_dual_grid.jl`.

    # ============================================================
    # Float32 + Complex smoke tests.
    # ============================================================
    @testset "Float32 Range + PeriodicBC(:exclusive)" begin
        x = range(0.0f0, step = Float32(2π / 8), length = 8)
        y = sin.(x)
        itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))
        @test eltype(itp.x) === Float32
        @test itp(1.0f0) isa Float32
    end

    @testset "ComplexF64 values + PeriodicBC(:exclusive)" begin
        x = collect(range(0.0, step = 2π / 8, length = 8))
        y = @. exp(im * x)
        itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        @test eltype(itp.y) <: Complex
        @test itp(0.5) isa Complex
    end

    # ============================================================
    # Series + PeriodicBC
    # ============================================================
    @testset "Series persistent + PeriodicBC — matches per-series interpolants" begin
        x = collect(range(0.0, step = 2π / 16, length = 16))
        y1 = sin.(x)
        y2 = cos.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        sitp = constant_interp(x, Series(y1, y2); bc = bc)
        itp1 = constant_interp(x, y1; bc = bc)
        itp2 = constant_interp(x, y2; bc = bc)
        for xq in (0.3, 1.7, 2π + 0.5, -0.2)
            out = sitp(xq)
            @test out[1] ≈ itp1(xq) atol = 1.0e-12
            @test out[2] ≈ itp2(xq) atol = 1.0e-12
        end
    end

    @testset "Series persistent + PeriodicBC :inclusive mismatch raises" begin
        x = collect(range(0.0, 2π, length = 17))
        y1 = sin.(x)
        y2 = collect(0.0:16)             # mismatched endpoints
        @test_throws ArgumentError constant_interp(x, Series(y1, y2); bc = PeriodicBC())
    end

    @testset "Series oneshot scalar + PeriodicBC agrees with persistent" begin
        x = collect(range(0.0, step = 2π / 16, length = 16))
        y1 = sin.(x)
        y2 = cos.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        sitp = constant_interp(x, Series(y1, y2); bc = bc)
        for xq in (0.0, 0.5, 2.0, -0.1, 2π + 0.1)
            oneshot = constant_interp(x, Series(y1, y2), xq; bc = bc)
            @test oneshot ≈ sitp(xq) atol = 1.0e-12
        end
    end

    @testset "Series oneshot vector + PeriodicBC" begin
        x = collect(range(0.0, step = 2π / 16, length = 16))
        y1 = sin.(x)
        y2 = cos.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        sitp = constant_interp(x, Series(y1, y2); bc = bc)

        xqs = [0.0, 0.5, 2.0, -0.1]
        out_vec = constant_interp(x, Series(y1, y2), xqs; bc = bc)
        @test length(out_vec) == 2
        for j in eachindex(xqs)
            ref = sitp(xqs[j])
            @test out_vec[1][j] ≈ ref[1] atol = 1.0e-12
            @test out_vec[2][j] ≈ ref[2] atol = 1.0e-12
        end

        outs = [similar(xqs) for _ in 1:2]
        constant_interp!(outs, x, Series(y1, y2), xqs; bc = bc)
        @test outs[1] ≈ out_vec[1] atol = 1.0e-12
    end

    # ============================================================
    # Series OneShot Scalar + PeriodicBC — Zero-Copy Migration (A-2)
    # ============================================================
    # Mirrors the Linear zero-copy migration. Notable constant-specific
    # behavior: `constant_interp` defaults to `NearestSide()`, which tie-breaks
    # to the left at `dL == h/2`. At `xq == x[1] + period` (exclusive right
    # endpoint), `_wrap_to_domain` maps the query back to `x[1]` — the series
    # path now returns the same value as the non-series constant path there
    # (i.e. `y[1]`), closing the former series/non-series gap.

    function _alloc_constant_series_scalar_range_exclusive()
        x = range(0.0, step = 2π / 16, length = 16)
        y1 = sin.(x)
        y2 = cos.(x)
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        out = Vector{Float64}(undef, 2)
        constant_interp!(out, x, s, 1.0; bc = bc)
        constant_interp!(out, x, s, 1.0; bc = bc)
        return @allocated constant_interp!(out, x, s, 1.0; bc = bc)
    end

    function _alloc_constant_series_scalar_vector_exclusive()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        y1 = sin.(x)
        y2 = cos.(x)
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        out = Vector{Float64}(undef, 2)
        constant_interp!(out, x, s, 1.0; bc = bc)
        constant_interp!(out, x, s, 1.0; bc = bc)
        return @allocated constant_interp!(out, x, s, 1.0; bc = bc)
    end

    function _alloc_constant_series_scalar_inclusive()
        x = collect(range(0.0, 2π, length = 17))
        y1 = sin.(x)
        y2 = cos.(x)
        s = Series(y1, y2)
        bc = PeriodicBC()
        out = Vector{Float64}(undef, 2)
        constant_interp!(out, x, s, 1.0; bc = bc)
        constant_interp!(out, x, s, 1.0; bc = bc)
        return @allocated constant_interp!(out, x, s, 1.0; bc = bc)
    end

    @testset "Series scalar + PeriodicBC zero-alloc — Range exclusive (T-series-alloc)" begin
        @test _alloc_constant_series_scalar_range_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series scalar + PeriodicBC zero-alloc — Vector exclusive (T-series-alloc)" begin
        @test _alloc_constant_series_scalar_vector_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series scalar + PeriodicBC zero-alloc — inclusive (T-series-alloc)" begin
        @test _alloc_constant_series_scalar_inclusive() <= ALLOC_THRESHOLD
    end

    @testset "Series scalar + PeriodicBC(:exclusive) seam semantic" begin
        # 4-point grid, period = 4.0 → seam cell [x[n], x[1]+period) = [3, 4)
        x = collect(0.0:3.0)
        y1 = [10.0, 20.0, 30.0, 40.0]
        y2 = [1.0, 2.0, 3.0, 4.0]
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)

        # Inside seam cell at xq = 3.5: default `NearestSide()` ties left at
        # `dL == h/2`, so the result is `y[idxL] = y[n]`.
        out = constant_interp(x, s, 3.5; bc = bc)
        @test out[1] == 40.0
        @test out[2] == 4.0

        # At xq == x[n] = 3.0: `aq.xq == x_last` short-circuit → y[end] = y[n]
        out_at_n = constant_interp(x, s, 3.0; bc = bc)
        @test out_at_n[1] == 40.0
        @test out_at_n[2] == 4.0

        # Series↔non-series alignment at the exclusive right endpoint.
        # `_wrap_to_domain` sends xq = x[1] + period back onto the base domain
        # [x[1], x[1]+period), so the resolved value is `y[1]` (the wrapped
        # endpoint), not `y[n]`. Before this refactor the pool-extended series
        # path returned `y[n]` here instead; the assertion pins down the fix.
        out_endpoint = constant_interp(x, s, 4.0; bc = bc)
        @test out_endpoint[1] == constant_interp(x, y1, 4.0; bc = bc) == y1[1]
        @test out_endpoint[2] == constant_interp(x, y2, 4.0; bc = bc) == y2[1]

        # Cross-check series↔non-series at mid-seam
        @test out[1] == constant_interp(x, y1, 3.5; bc = bc)
        @test out[2] == constant_interp(x, y2, 3.5; bc = bc)

        # Cross-check series oneshot == persistent series interpolant (mid-seam)
        sitp = constant_interp(x, s; bc = bc)
        @test out[1] == sitp(3.5)[1]
        @test out[2] == sitp(3.5)[2]
    end

    @testset "Series oneshot + PeriodicBC(:exclusive) preserves cached step on large-offset Range" begin
        # Parallel to the Linear test: `_CachedRange.h` stores the exact step
        # while `xR - xL` suffers float cancellation near 1e8. Constant uses
        # `h`/`dL` for side-offset computation, so nearest/left/right-side
        # decisions must see the cached step to match scalar/persistent paths.
        x = range(1.0e8, step = 0.1, length = 10)
        y1 = Float64.(1:10)
        y2 = Float64.(11:20)
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive)
        xq = 1.0e8 + 0.95

        for side in (LeftSide(), RightSide(), NearestSide())
            v_scalar = constant_interp(x, y1, xq; bc = bc, side = side)
            v_oneshot = constant_interp(x, s, xq; bc = bc, side = side)
            v_persist = constant_interp(x, s; bc = bc, side = side)(xq)
            @test v_oneshot[1] === v_scalar
            @test v_oneshot[1] === v_persist[1]

            xqs = [1.0e8 + 0.95, 1.0e8 + 0.55]
            outs = [similar(xqs) for _ in 1:2]
            constant_interp!(outs, x, s, xqs; bc = bc, side = side)
            @test outs[1][1] === v_scalar
            @test outs[1][2] === constant_interp(x, y1, xqs[2]; bc = bc, side = side)
        end
    end

    # ============================================================
    # Series OneShot Vector-Batch + PeriodicBC — Zero-Copy (Stage 2)
    # ============================================================

    function _alloc_constant_series_vector_range_exclusive()
        x = range(0.0, step = 2π / 16, length = 16)
        s = Series(sin.(x), cos.(x))
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        constant_interp!(outs, x, s, xqs; bc = bc)
        constant_interp!(outs, x, s, xqs; bc = bc)
        return @allocated constant_interp!(outs, x, s, xqs; bc = bc)
    end

    function _alloc_constant_series_vector_vector_exclusive()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        s = Series(sin.(x), cos.(x))
        bc = PeriodicBC(endpoint = :exclusive, period = 2π)
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        constant_interp!(outs, x, s, xqs; bc = bc)
        constant_interp!(outs, x, s, xqs; bc = bc)
        return @allocated constant_interp!(outs, x, s, xqs; bc = bc)
    end

    function _alloc_constant_series_vector_inclusive()
        x = collect(range(0.0, 2π, length = 17))
        s = Series(sin.(x), cos.(x))
        bc = PeriodicBC()
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        constant_interp!(outs, x, s, xqs; bc = bc)
        constant_interp!(outs, x, s, xqs; bc = bc)
        return @allocated constant_interp!(outs, x, s, xqs; bc = bc)
    end

    function _alloc_constant_series_vector_nobc()
        x = range(0.0, step = 2π / 16, length = 16)
        s = Series(sin.(x), cos.(x))
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        constant_interp!(outs, x, s, xqs)
        constant_interp!(outs, x, s, xqs)
        return @allocated constant_interp!(outs, x, s, xqs)
    end

    @testset "Series vector + PeriodicBC zero-alloc — Range exclusive (T-series-alloc)" begin
        @test _alloc_constant_series_vector_range_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series vector + PeriodicBC zero-alloc — Vector exclusive (T-series-alloc)" begin
        @test _alloc_constant_series_vector_vector_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series vector + PeriodicBC zero-alloc — inclusive (T-series-alloc)" begin
        @test _alloc_constant_series_vector_inclusive() <= ALLOC_THRESHOLD
    end
    @testset "Series vector + NoBC zero-alloc (T-series-alloc)" begin
        @test _alloc_constant_series_vector_nobc() <= ALLOC_THRESHOLD
    end

    @testset "Series vector + PeriodicBC(:exclusive) seam cell semantic" begin
        # 4-point grid, period=4.0 → seam cell [3, 4). NearestSide default:
        # returns yL if dL ≤ h/2, else yR. Inside seam cell: yL=y[n], yR=y[1].
        # `_constant_eval_at_anchor` short-circuits at `aq.xq == x_last`
        # (x_last = x[n] = 3.0) → returns y[end] = y[n] = 40.
        x = collect(0.0:3.0)
        y1 = [10.0, 20.0, 30.0, 40.0]
        y2 = [1.0, 2.0, 3.0, 4.0]
        s = Series(y1, y2)
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)

        # Batch covering:
        #   2.25  — interior cell [2,3], dL=0.25 ≤ 0.5 → yL = y[3] = 30
        #   3.0   — exact x[n], x_last short-circuit → y[end] = y[n] = 40
        #   3.25  — seam, dL=0.25 ≤ 0.5 → yL = y[n] = 40
        #   3.5   — seam mid, dL=0.5 = h/2 (tie) → yL = y[n] = 40
        #   3.75  — seam, dL=0.75 > 0.5 → yR = y[1] = 10 (wrap)
        xqs = [2.25, 3.0, 3.25, 3.5, 3.75]
        outs = [Vector{Float64}(undef, length(xqs)) for _ in 1:2]
        constant_interp!(outs, x, s, xqs; bc = bc)

        # y1 series
        @test outs[1][1] == 30.0   # interior, NearestSide → yL
        @test outs[1][2] == 40.0   # at x[n] (x_last short-circuit)
        @test outs[1][3] == 40.0   # seam 25%, NearestSide → yL=y[n]
        @test outs[1][4] == 40.0   # seam mid (tie-break to yL)
        @test outs[1][5] == 10.0   # seam 75%, NearestSide → yR=y[1] (wrap)

        # y2 series
        @test outs[2][1] == 3.0
        @test outs[2][2] == 4.0
        @test outs[2][3] == 4.0
        @test outs[2][4] == 4.0
        @test outs[2][5] == 1.0

        # Cross-check: batch path agrees with per-query scalar path, series-wise.
        for j in eachindex(xqs)
            scalar_out = constant_interp(x, s, xqs[j]; bc = bc)
            @test outs[1][j] == scalar_out[1]
            @test outs[2][j] == scalar_out[2]
        end

        # Cross-check: batch path agrees with non-series constant.
        for j in eachindex(xqs)
            @test outs[1][j] == constant_interp(x, y1, xqs[j]; bc = bc)
            @test outs[2][j] == constant_interp(x, y2, xqs[j]; bc = bc)
        end
    end

    # ============================================================
    # Persistent Series callable + PeriodicBC — Zero-Copy (Stage 3)
    # ============================================================

    function _alloc_constant_persistent_vector_range_exclusive()
        x = range(0.0, step = 2π / 16, length = 16)
        s = Series(sin.(x), cos.(x))
        sitp = constant_interp(x, s; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        sitp(outs, xqs); sitp(outs, xqs)
        return @allocated sitp(outs, xqs)
    end

    function _alloc_constant_persistent_vector_vector_exclusive()
        x = [0.0, 0.5, 1.5, 3.0, 5.0]
        s = Series(sin.(x), cos.(x))
        sitp = constant_interp(x, s; bc = PeriodicBC(endpoint = :exclusive, period = 2π))
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        sitp(outs, xqs); sitp(outs, xqs)
        return @allocated sitp(outs, xqs)
    end

    function _alloc_constant_persistent_vector_inclusive()
        x = collect(range(0.0, 2π, length = 17))
        s = Series(sin.(x), cos.(x))
        sitp = constant_interp(x, s; bc = PeriodicBC())
        xqs = [0.5, 1.0, 2.0, 3.5]
        outs = [similar(xqs) for _ in 1:2]
        sitp(outs, xqs); sitp(outs, xqs)
        return @allocated sitp(outs, xqs)
    end

    @testset "Persistent callable + PeriodicBC zero-alloc — Range exclusive (T-persistent-alloc)" begin
        @test _alloc_constant_persistent_vector_range_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Persistent callable + PeriodicBC zero-alloc — Vector exclusive (T-persistent-alloc)" begin
        @test _alloc_constant_persistent_vector_vector_exclusive() <= ALLOC_THRESHOLD
    end
    @testset "Persistent callable + PeriodicBC zero-alloc — inclusive (T-persistent-alloc)" begin
        @test _alloc_constant_persistent_vector_inclusive() <= ALLOC_THRESHOLD
    end

end
