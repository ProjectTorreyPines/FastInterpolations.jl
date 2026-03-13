# ========================================
# ND NoExtrap OOB Domain Validation
# ========================================
#
# These tests verify that NoExtrap correctly throws DomainError for
# out-of-bounds queries on ALL ND paths (oneshot scalar, oneshot batch,
# interpolant batch).
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

@testset "ND NoExtrap OOB — DomainError" begin
    # ── Shared test data ──
    gx = [0.0, 1.0, 2.0]
    gy = [0.0, 1.0]
    f_2d = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3×2

    gx4 = [0.0, 1.0, 2.0, 3.0]
    gy4 = [0.0, 1.0, 2.0, 3.0]
    f_cubic = randn(4, 4)
    f_quad = randn(4, 4)

    # ── Constant ND ──
    @testset "constant_interp — scalar oneshot" begin
        @test_throws DomainError constant_interp((gx, gy), f_2d, (-0.1, 0.5))
        @test_throws DomainError constant_interp((gx, gy), f_2d, (0.5, 1.1))
    end

    @testset "constant_interp — batch (SoA)" begin
        @test_throws DomainError constant_interp((gx, gy), f_2d, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError constant_interp((gx, gy), f_2d, ([0.5, 0.5], [0.5, 1.1]))
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

    # ── Cubic ND ──
    @testset "cubic_interp — scalar oneshot" begin
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, (-0.1, 0.5))
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, (0.5, 3.1))
    end

    @testset "cubic_interp — batch (SoA)" begin
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError cubic_interp((gx4, gy4), f_cubic, ([0.5, 0.5], [0.5, 3.1]))
    end

    @testset "cubic_interp — interpolant batch" begin
        itp = cubic_interp((gx4, gy4), f_cubic)
        out = zeros(2)
        @test_throws DomainError itp(out, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError itp(out, ([0.5, 0.5], [0.5, 3.1]))
    end

    # ── Quadratic ND interpolant ──
    @testset "quadratic_interp — interpolant batch" begin
        itp = quadratic_interp((gx4, gy4), f_quad)
        out = zeros(2)
        @test_throws DomainError itp(out, ([-0.1, 0.5], [0.5, 0.5]))
        @test_throws DomainError itp(out, ([0.5, 0.5], [0.5, 3.1]))
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
        test_script = raw"""
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
        result = readchomp(`julia --startup-file=no --project=$project_dir -e $test_script`)
        @test startswith(result, "OK:")

        # Scalar oneshot (single point, not batch)
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
        @test startswith(result2, "OK:")
    end

    # ── Zero-allocation: _validate_nd_domain must not allocate ──
    @testset "zero-allocation — _validate_nd_domain" begin
        function _test_validate_alloc()
            gx = [0.0, 1.0, 2.0]; gy = [0.0, 1.0]
            grids = (gx, gy)
            soa = ([0.3, 0.7], [0.2, 0.8])
            extraps_homo = (NoExtrap(), NoExtrap())
            extraps_het = (NoExtrap(), ExtendExtrap())

            # warmup all paths
            FastInterpolations._validate_nd_domain(grids, (0.5, 0.5), extraps_homo)
            FastInterpolations._validate_nd_domain(grids, (0.5, 0.5), extraps_het)
            FastInterpolations._validate_nd_domain(grids, soa, extraps_homo)
            FastInterpolations._validate_nd_domain(grids, soa, extraps_het)

            a_scalar_homo = @allocated FastInterpolations._validate_nd_domain(grids, (0.5, 0.5), extraps_homo)
            a_scalar_het  = @allocated FastInterpolations._validate_nd_domain(grids, (0.5, 0.5), extraps_het)
            a_soa_homo    = @allocated FastInterpolations._validate_nd_domain(grids, soa, extraps_homo)
            a_soa_het     = @allocated FastInterpolations._validate_nd_domain(grids, soa, extraps_het)

            (a_scalar_homo, a_scalar_het, a_soa_homo, a_soa_het)
        end
        allocs = _test_validate_alloc()
        @test all(==(0), allocs)
    end
end
