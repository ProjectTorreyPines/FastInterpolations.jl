# ========================================
# ND NoExtrap OOB Domain Validation
# ========================================
#
# These tests verify that NoExtrap correctly throws DomainError for
# out-of-bounds queries on ALL ND paths:
#   1. Oneshot scalar   — e.g. linear_interp(grids, data, (x,y))
#   2. Oneshot batch    — e.g. linear_interp(grids, data, ([x1,x2], [y1,y2]))
#   3. Interpolant scalar — e.g. itp((x,y))
#   4. Interpolant batch  — e.g. itp(out, ([x1,x2], [y1,y2]))
#   5. AoS queries       — e.g. itp(out, [(x1,y1), (x2,y2)])
#
# IMPORTANT: Under Pkg.test() (--check-bounds=yes), @boundscheck is never
# elided, so these tests pass even WITHOUT the _validate_nd_domain fix.
# The subprocess test at the bottom verifies the bug under production mode
# (--check-bounds=auto), which is the actual failure scenario.
#
# BUG: _extrap_axis wraps _handle_axis_extrap in @inbounds, so
# @boundscheck _check_domain(::NoExtrap) is ALWAYS elided in production.
# Without _validate_nd_domain, OOB queries silently return garbage.

using Test
using FastInterpolations

# LTS Julia may show small boxing allocations on ND eval paths.
# Guarded for standalone execution (runtests.jl defines this globally).
if !@isdefined(ALLOC_THRESHOLD)
    const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240
end

@testset "ND NoExtrap OOB — DomainError" begin
    # ── Shared test data ──
    gx = [0.0, 1.0, 2.0]
    gy = [0.0, 1.0]
    f_2d = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2

    gx4 = [0.0, 1.0, 2.0, 3.0]
    gy4 = [0.0, 1.0, 2.0, 3.0]
    f_cubic = randn(4, 4)
    f_quad = randn(4, 4)

    # 3D data for N=3 tests
    gz4 = [0.0, 1.0, 2.0, 3.0]
    f_3d_linear = randn(3, 2, 4)   # 3×2×4 for (gx, gy, gz4)
    f_3d_cubic = randn(4, 4, 4)

    # ── Constant ND ──
    @testset "constant_interp — scalar oneshot" begin
        @test_throws DomainError constant_interp((gx, gy), f_2d, (-0.1, 0.5))
        @test_throws DomainError constant_interp((gx, gy), f_2d, (0.5, 1.1))
    end

    @testset "constant_interp — batch (SoA)" begin
        @test_throws DomainError constant_interp((gx, gy), f_2d, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError constant_interp((gx, gy), f_2d, ([0.5, 0.5], [0.5, 1.1]))
    end

    @testset "constant_interp — interpolant scalar" begin
        itp = constant_interp((gx, gy), f_2d)
        @test_throws DomainError itp((-0.1, 0.5))
        @test_throws DomainError itp((0.5, 1.1))
    end

    @testset "constant_interp — interpolant batch" begin
        itp = constant_interp((gx, gy), f_2d)
        out = zeros(2)
        @test_throws DomainError itp(out, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError itp(out, ([0.5, 0.5], [0.5, 1.1]))
    end

    # ── Linear ND ──
    @testset "linear_interp — scalar oneshot" begin
        @test_throws DomainError linear_interp((gx, gy), f_2d, (-0.1, 0.5))
        @test_throws DomainError linear_interp((gx, gy), f_2d, (0.5, 1.1))
    end

    @testset "linear_interp — batch (SoA)" begin
        @test_throws DomainError linear_interp((gx, gy), f_2d, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError linear_interp((gx, gy), f_2d, ([0.5, 0.5], [0.5, 1.1]))
    end

    @testset "linear_interp — interpolant scalar" begin
        itp = linear_interp((gx, gy), f_2d)
        @test_throws DomainError itp((-0.1, 0.5))
        @test_throws DomainError itp((0.5, 1.1))
    end

    @testset "linear_interp — interpolant batch" begin
        itp = linear_interp((gx, gy), f_2d)
        out = zeros(2)
        @test_throws DomainError itp(out, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError itp(out, ([0.5, 0.5], [0.5, 1.1]))
    end

    # ── Quadratic ND ──
    @testset "quadratic_interp — scalar oneshot" begin
        @test_throws DomainError quadratic_interp((gx4, gy4), f_quad, (-0.1, 0.5))
        @test_throws DomainError quadratic_interp((gx4, gy4), f_quad, (0.5, 3.1))
    end

    @testset "quadratic_interp — batch (SoA)" begin
        @test_throws DomainError quadratic_interp((gx4, gy4), f_quad, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError quadratic_interp((gx4, gy4), f_quad, ([0.5, 0.5], [0.5, 3.1]))
    end

    @testset "quadratic_interp — interpolant scalar" begin
        itp = quadratic_interp((gx4, gy4), f_quad)
        @test_throws DomainError itp((-0.1, 0.5))
        @test_throws DomainError itp((0.5, 3.1))
    end

    @testset "quadratic_interp — interpolant batch" begin
        itp = quadratic_interp((gx4, gy4), f_quad)
        out = zeros(2)
        @test_throws DomainError itp(out, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError itp(out, ([0.5, 0.5], [0.5, 3.1]))
    end

    # ── Cubic ND ──
    @testset "cubic_interp — scalar oneshot" begin
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, (-0.1, 0.5))
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, (0.5, 3.1))
    end

    @testset "cubic_interp — batch (SoA)" begin
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, ([0.5, 0.5], [0.5, 3.1]))
    end

    @testset "cubic_interp — interpolant scalar" begin
        itp = cubic_interp((gx4, gy4), f_cubic)
        @test_throws DomainError itp((-0.1, 0.5))
        @test_throws DomainError itp((0.5, 3.1))
    end

    @testset "cubic_interp — interpolant batch" begin
        itp = cubic_interp((gx4, gy4), f_cubic)
        out = zeros(2)
        @test_throws DomainError itp(out, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError itp(out, ([0.5, 0.5], [0.5, 3.1]))
    end

    # ── AoS queries (generic _validate_nd_domain path) ──
    @testset "AoS queries — generic path" begin
        aos_oob = [(-0.1, 0.5), (0.5, 0.5)]
        aos_ok = [(0.5, 0.5), (1.0, 0.5)]

        @test_throws DomainError linear_interp((gx, gy), f_2d, aos_oob)
        itp = linear_interp((gx, gy), f_2d)
        out = zeros(2)
        @test_throws DomainError itp(out, aos_oob)

        # In-bounds AoS should succeed
        itp(out, aos_ok)
        @test all(isfinite, out)
    end

    # ── N=3 tests (exercises map on larger tuples) ──
    @testset "N=3 — linear scalar + batch" begin
        @test_throws DomainError linear_interp(
            (gx, gy, gz4), f_3d_linear, (-0.1, 0.5, 1.0)
        )
        @test_throws DomainError linear_interp(
            (gx, gy, gz4), f_3d_linear, ([-0.1, 0.5], [0.5, 0.5], [1.0, 1.0])
        )
        # In-bounds should succeed
        val = linear_interp((gx, gy, gz4), f_3d_linear, (0.5, 0.5, 1.0))
        @test isfinite(val)
    end

    @testset "N=3 — cubic scalar + interpolant" begin
        @test_throws DomainError cubic_interp(
            (gx4, gy4, gz4), f_3d_cubic, (-0.1, 0.5, 1.0)
        )
        itp3 = cubic_interp((gx4, gy4, gz4), f_3d_cubic)
        @test_throws DomainError itp3((-0.1, 0.5, 1.0))
        @test isfinite(itp3((0.5, 0.5, 1.0)))
    end

    # ── Mixed extrap: only NoExtrap axis should throw ──
    @testset "mixed extrap — NoExtrap on one axis only" begin
        @test_throws DomainError linear_interp(
            (gx, gy), f_2d, (-0.1, 0.5);
            extrap = (NoExtrap(), ExtendExtrap())
        )
        val = linear_interp(
            (gx, gy), f_2d, (0.5, 1.5);
            extrap = (NoExtrap(), ExtendExtrap())
        )
        @test val isa Float64
    end

    # ── Subprocess test: production mode (--check-bounds=auto) ──
    # This is the TRUE red test: under default bounds checking,
    # @boundscheck inside @inbounds is elided, exposing the bug.
    @testset "production mode OOB (--check-bounds=auto)" begin
        project_dir = dirname(@__DIR__)

        # Batch oneshot path
        batch_script = raw"""
        using FastInterpolations
        gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
        f = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        try
            linear_interp((gx, gy), f, ([-0.5, 0.5], [0.5, 0.5]))
            println("BUG:no_error")
        catch e
            println(string("OK:", typeof(e)))
        end
        """
        result = readchomp(`julia --startup-file=no --project=$project_dir -e $batch_script`)
        @test result == "OK:DomainError"

        # Scalar oneshot path
        scalar_script = raw"""
        using FastInterpolations
        gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
        f = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        try
            linear_interp((gx, gy), f, (-0.5, 0.5))
            println("BUG:no_error")
        catch e
            println(string("OK:", typeof(e)))
        end
        """
        result2 = readchomp(`julia --startup-file=no --project=$project_dir -e $scalar_script`)
        @test result2 == "OK:DomainError"

        # Interpolant scalar path (the gap found in code review)
        itp_scalar_script = raw"""
        using FastInterpolations
        gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
        f = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        itp = linear_interp((gx, gy), f)
        try
            itp((-0.5, 0.5))
            println("BUG:no_error")
        catch e
            println(string("OK:", typeof(e)))
        end
        """
        result3 = readchomp(`julia --startup-file=no --project=$project_dir -e $itp_scalar_script`)
        @test result3 == "OK:DomainError"

        # AoS query path
        aos_script = raw"""
        using FastInterpolations
        gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
        f = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        itp = linear_interp((gx, gy), f)
        out = zeros(2)
        try
            itp(out, [(-0.5, 0.5), (0.5, 0.5)])
            println("BUG:no_error")
        catch e
            println(string("OK:", typeof(e)))
        end
        """
        result4 = readchomp(`julia --startup-file=no --project=$project_dir -e $aos_script`)
        @test result4 == "OK:DomainError"
    end

    # ── Zero-allocation: _validate_nd_domain must not allocate ──
    @testset "zero-allocation — _validate_nd_domain" begin
        function _test_validate_alloc()
            gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
            grids = (gx, gy)
            scalar_pt = (0.5, 0.5)
            soa = ([0.3, 0.7], [0.2, 0.8])
            extraps_homo = (NoExtrap(), NoExtrap())
            extraps_het = (NoExtrap(), ExtendExtrap())

            # warmup all paths
            FastInterpolations._validate_nd_domain(grids, scalar_pt, extraps_homo)
            FastInterpolations._validate_nd_domain(grids, scalar_pt, extraps_het)
            FastInterpolations._validate_nd_domain(grids, soa, extraps_homo)
            FastInterpolations._validate_nd_domain(grids, soa, extraps_het)

            a1 = @allocated FastInterpolations._validate_nd_domain(grids, scalar_pt, extraps_homo)
            a2 = @allocated FastInterpolations._validate_nd_domain(grids, scalar_pt, extraps_het)
            a3 = @allocated FastInterpolations._validate_nd_domain(grids, soa, extraps_homo)
            a4 = @allocated FastInterpolations._validate_nd_domain(grids, soa, extraps_het)

            (a1, a2, a3, a4)
        end
        allocs = _test_validate_alloc()
        @test all(<=(ALLOC_THRESHOLD), allocs)
    end

    # ── Zero-allocation: full eval paths with NoExtrap ──
    @testset "zero-allocation — full eval paths" begin
        function _test_eval_alloc_linear()
            gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
            f = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            itp = linear_interp((gx, gy), f)
            pt = (0.5, 0.5)

            # warmup
            itp(pt)
            itp(pt)

            @allocated itp(pt)
        end

        function _test_eval_alloc_constant()
            gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
            f = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            itp = constant_interp((gx, gy), f)
            pt = (0.5, 0.5)

            itp(pt)
            itp(pt)

            @allocated itp(pt)
        end

        function _test_eval_alloc_cubic()
            gx = [0.0, 1.0, 2.0, 3.0]; gy = [0.0, 1.0, 2.0, 3.0]
            f = randn(4, 4)
            itp = cubic_interp((gx, gy), f)
            pt = (0.5, 0.5)

            itp(pt)
            itp(pt)

            @allocated itp(pt)
        end

        function _test_eval_alloc_quadratic()
            gx = [0.0, 1.0, 2.0, 3.0]; gy = [0.0, 1.0, 2.0, 3.0]
            f = randn(4, 4)
            itp = quadratic_interp((gx, gy), f)
            pt = (0.5, 0.5)

            itp(pt)
            itp(pt)

            @allocated itp(pt)
        end

        @test _test_eval_alloc_linear() <= ALLOC_THRESHOLD
        @test _test_eval_alloc_constant() <= ALLOC_THRESHOLD
        @test _test_eval_alloc_cubic() <= ALLOC_THRESHOLD
        @test _test_eval_alloc_quadratic() <= ALLOC_THRESHOLD
    end
end
