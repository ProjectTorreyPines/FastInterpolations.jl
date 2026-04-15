# Tests for PeriodicBC on Linear interpolation (Phase 1).
#
# Coverage:
#   - `NoBC()` default is a no-op (regression guard against pre-change behavior).
#   - `PeriodicBC(endpoint=:inclusive)` wraps correctly with y[1] == y[end] data.
#   - `PeriodicBC(endpoint=:exclusive, period=L)` extends grid+data, forces WrapExtrap.
#   - FVM cell-centered case: x = [0.5, 1.5, 2.5], period = 3.0.
#   - Range grid (type-stable extension) vs Vector grid (vcat extension).
#   - Inclusive check raises on mismatched endpoints.

using Test
using FastInterpolations
using FastInterpolations: _is_periodic_bc, _CachedRange

@testset "Linear PeriodicBC" begin

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

    # ============================================================
    # Interpolant path — extended copy storage
    # ============================================================
    @testset "Interpolant path stores extended copy (Vector grid)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 20.0, 30.0, 40.0]
        x_ref = copy(x)
        y_ref = copy(y)

        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive, period = 4.0))

        # Stored grid/values are length N+1 with the virtual endpoint appended.
        @test length(itp.x) == 5
        @test length(itp.y) == 5
        @test itp.y[end] == itp.y[1]  # appended y[1] at the end
        @test itp.x[end] ≈ 4.0

        # Original user arrays are untouched.
        @test x == x_ref
        @test y == y_ref
    end

    @testset "Interpolant path stores extended copy (Range grid → _CachedRange)" begin
        x = range(0.0, step = 1.0, length = 4)
        y = [10.0, 20.0, 30.0, 40.0]

        itp = linear_interp(x, y; bc = PeriodicBC(endpoint = :exclusive))

        # Range input → extended grid must be _CachedRange (preserves O(1) indexing
        # and zero-alloc lookup; _to_float_adding_endpoint guarantees this).
        @test itp.x isa _CachedRange
        @test length(itp.x) == 5
        @test length(itp.y) == 5
        @test itp.y[end] == itp.y[1]
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
        @test itp.x isa _CachedRange{Float64}          # Int Range → Float _CachedRange
        @test length(itp.x) == 12                      # N+1 extension
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

end
