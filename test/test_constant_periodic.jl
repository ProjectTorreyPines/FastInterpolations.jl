# Tests for PeriodicBC on Constant interpolation (Phase 1).
#
# Coverage mirrors test_linear_periodic.jl, with extra per-`side` coverage
# (NearestSide/LeftSide/RightSide must all respect periodicity).

using Test
using FastInterpolations
using FastInterpolations: _CachedRange

@testset "Constant PeriodicBC" begin

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

    # ============================================================
    # Interpolant path — extended copy storage
    # ============================================================
    @testset "Interpolant path stores extended copy (Vector grid)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]
        x_ref = copy(x)
        y_ref = copy(y)

        itp = constant_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 4.0))

        @test length(itp.x) == 5
        @test length(itp.y) == 5
        @test itp.y[end] == itp.y[1]
        @test itp.x[end] ≈ 4.0

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

end
