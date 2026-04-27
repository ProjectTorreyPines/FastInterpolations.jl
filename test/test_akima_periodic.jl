# Tests for PeriodicBC on Akima interpolation.
#
# Mirrors test_pchip_periodic.jl. Akima uses a wider 5-point stencil so the
# closed-cycle k=2/k=n-1 indices retain virtual extrapolation (one cell removed
# from the join — validated as acceptable approximation in the design).

@testitem "Akima PeriodicBC" setup = [AllocConstants] begin
    f(x) = sin(2π * x)
    fp(x) = 2π * cos(2π * x)

    @testset "NoBC default is no-op" begin
        x = collect(range(0.0, 1.0, length = 11))
        y = f.(x)

        itp_default = akima_interp(x, y)
        itp_nobc = akima_interp(x, y; bc = NoBC())
        @test typeof(itp_default) === typeof(itp_nobc)
        for xq in (0.05, 0.25, 0.5, 0.75, 0.95)
            @test itp_default(xq) === itp_nobc(xq)
        end
    end

    @testset "Inclusive — endpoint mismatch raises" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError akima_interp(x, y; bc = PeriodicBC())
    end

    @testset "C¹ at seam — exclusive, persistent, OnTheFly + PreCompute" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            itp = akima_interp(x, y; bc = bc, coeffs = cs)
            v_left = itp(1.0 - ε; deriv = DerivOp(1))
            v_right = itp(0.0 + ε; deriv = DerivOp(1))
            @test isapprox(v_left, v_right; atol = 1.0e-5)
        end
    end

    @testset "C¹ at seam — inclusive" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n))
        y = f.(x)
        @assert isapprox(y[1], y[end]; atol = 1.0e-12)
        bc = PeriodicBC(endpoint = :inclusive)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            itp = akima_interp(x, y; bc = bc, coeffs = cs)
            v_left = itp(1.0 - ε; deriv = DerivOp(1))
            v_right = itp(0.0 + ε; deriv = DerivOp(1))
            @test isapprox(v_left, v_right; atol = 1.0e-5)
        end
    end

    @testset "C¹ at seam — exclusive oneshot" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            v_left = akima_interp(x, y, 1.0 - ε; bc = bc, coeffs = cs, deriv = DerivOp(1))
            v_right = akima_interp(x, y, 0.0 + ε; bc = bc, coeffs = cs, deriv = DerivOp(1))
            @test isapprox(v_left, v_right; atol = 1.0e-5)
        end
    end

    @testset "Wrap query past seam" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = akima_interp(x, y; bc = bc)
        @test itp(1.25) ≈ itp(0.25) atol = 1.0e-12
        @test itp(-0.30) ≈ itp(0.70) atol = 1.0e-12
    end

    @testset "Closed-cycle baseline equivalence (exclusive ↔ inclusive)" begin
        n_excl = 20
        x_excl = collect(range(0.0, 1.0, length = n_excl + 1))[1:n_excl]
        y_excl = f.(x_excl)
        x_incl = vcat(x_excl, [1.0])
        y_incl = vcat(y_excl, [y_excl[1]])

        itp_excl = akima_interp(x_excl, y_excl; bc = PeriodicBC(endpoint = :exclusive, period = 1.0))
        itp_incl = akima_interp(x_incl, y_incl; bc = PeriodicBC(endpoint = :inclusive))

        for xq in (0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
            @test itp_excl(xq) ≈ itp_incl(xq) atol = 1.0e-12
            @test itp_excl(xq; deriv = DerivOp(1)) ≈ itp_incl(xq; deriv = DerivOp(1)) atol = 1.0e-10
        end
    end

    @testset "Vector oneshot — match against persistent" begin
        n = 25
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        xq = [0.05, 0.20, 0.50, 0.80, 0.95, 0.999]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        v_persistent = akima_interp(x, y; bc = bc).(xq)
        v_oneshot = akima_interp(x, y, xq; bc = bc)
        @test v_persistent ≈ v_oneshot atol = 1.0e-12
    end

    @testset "Range grid with bc.period === nothing (codex P2)" begin
        xr = range(0.0, 1.0, length = 11)[1:10]
        y = f.(xr)
        bc = PeriodicBC(endpoint = :exclusive)
        @test bc.period === nothing

        for cs in (PreCompute(), OnTheFly())
            v = akima_interp(xr, y, 0.5; bc = bc, coeffs = cs)
            @test isfinite(v)
        end
        itp = akima_interp(xr, y; bc = bc)
        @test itp(0.5) ≈ f(0.5) atol = 1.0e-3
    end

    @testset "Approximation accuracy" begin
        n = 41
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = akima_interp(x, y; bc = bc)
        for xq in (0.13, 0.37, 0.62, 0.88)
            @test itp(xq) ≈ f(xq) atol = 1.0e-3
        end
    end

    @testset "Small grid n=4 (closed-cycle k=2 guard)" begin
        # Akima boundary stencil uses min(2, n-1) — verify n=4 (smallest non-degenerate)
        # passes without index out-of-bounds.
        x = collect(range(0.0, 1.0, length = 5))[1:4]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        itp = akima_interp(x, y; bc = bc)
        @test isfinite(itp(0.5))
        @test isfinite(itp(0.5; deriv = DerivOp(1)))
    end

    @testset "n=3 special case honors PeriodicBC (codex P3)" begin
        # Before fix: n=3 special case bypassed PeriodicBC and returned the
        # one-sided endpoint slopes, breaking C¹ at the seam for inclusive
        # data like [0, 1, 0] where the endpoints match but stencil disagrees.
        x = [0.0, 0.5, 1.0]
        y = [0.0, 1.0, 0.0]    # y[1] == y[3] (inclusive)
        bc = PeriodicBC(endpoint = :inclusive)

        for cs in (PreCompute(), OnTheFly())
            itp = akima_interp(x, y; bc = bc, coeffs = cs)
            ε = 1.0e-7
            v_left = itp(1.0 - ε; deriv = DerivOp(1))
            v_right = itp(0.0 + ε; deriv = DerivOp(1))
            # |diff| at ε=1e-7 should be O(ε * d²y/dx²) ≈ O(1e-6), not O(1)
            @test abs(v_left - v_right) < 1.0e-5
        end

        # Exclusive form (n=3, virtual seam cell)
        bc_excl = PeriodicBC(endpoint = :exclusive, period = 1.5)
        x_ex = [0.0, 0.5, 1.0]
        y_ex = [0.0, 1.0, 0.5]
        for cs in (PreCompute(), OnTheFly())
            itp = akima_interp(x_ex, y_ex; bc = bc_excl, coeffs = cs)
            @test isfinite(itp(0.5))
            @test isfinite(itp(0.5; deriv = DerivOp(1)))
        end
    end
end
