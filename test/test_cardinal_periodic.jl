# Tests for PeriodicBC on Cardinal spline interpolation.
#
# Mirrors test_pchip_periodic.jl with extra `tension` coverage.

@testitem "Cardinal PeriodicBC" setup = [AllocConstants] begin
    f(x) = sin(2π * x)
    fp(x) = 2π * cos(2π * x)

    @testset "NoBC default is no-op" begin
        x = collect(range(0.0, 1.0, length = 11))
        y = f.(x)

        itp_default = cardinal_interp(x, y)
        itp_nobc = cardinal_interp(x, y; bc = NoBC())
        @test typeof(itp_default) === typeof(itp_nobc)
        for xq in (0.05, 0.25, 0.5, 0.75, 0.95)
            @test itp_default(xq) === itp_nobc(xq)
        end
    end

    @testset "Inclusive — endpoint mismatch raises" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError cardinal_interp(x, y; bc = PeriodicBC())
    end

    @testset "C¹ at seam — exclusive across all (coeffs × tension)" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            for tens in (0.0, 0.3, 0.7)
                itp = cardinal_interp(x, y; bc = bc, coeffs = cs, tension = tens)
                v_left = itp(1.0 - ε; deriv = DerivOp(1))
                v_right = itp(0.0 + ε; deriv = DerivOp(1))
                @test isapprox(v_left, v_right; atol = 1.0e-5)
            end
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
            itp = cardinal_interp(x, y; bc = bc, coeffs = cs, tension = 0.0)
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
            v_left = cardinal_interp(x, y, 1.0 - ε; bc = bc, coeffs = cs, deriv = DerivOp(1), tension = 0.5)
            v_right = cardinal_interp(x, y, 0.0 + ε; bc = bc, coeffs = cs, deriv = DerivOp(1), tension = 0.5)
            @test isapprox(v_left, v_right; atol = 1.0e-5)
        end
    end

    @testset "Wrap query past seam" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = cardinal_interp(x, y; bc = bc, tension = 0.0)
        @test itp(1.25) ≈ itp(0.25) atol = 1.0e-12
        @test itp(-0.30) ≈ itp(0.70) atol = 1.0e-12
    end

    @testset "Closed-cycle baseline equivalence (exclusive ↔ inclusive)" begin
        n_excl = 20
        x_excl = collect(range(0.0, 1.0, length = n_excl + 1))[1:n_excl]
        y_excl = f.(x_excl)
        x_incl = vcat(x_excl, [1.0])
        y_incl = vcat(y_excl, [y_excl[1]])

        for tens in (0.0, 0.5)
            itp_excl = cardinal_interp(x_excl, y_excl; bc = PeriodicBC(endpoint = :exclusive, period = 1.0), tension = tens)
            itp_incl = cardinal_interp(x_incl, y_incl; bc = PeriodicBC(endpoint = :inclusive), tension = tens)
            for xq in (0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
                @test itp_excl(xq) ≈ itp_incl(xq) atol = 1.0e-12
                @test itp_excl(xq; deriv = DerivOp(1)) ≈ itp_incl(xq; deriv = DerivOp(1)) atol = 1.0e-10
            end
        end
    end

    @testset "Vector oneshot — match against persistent" begin
        n = 25
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        xq = [0.05, 0.20, 0.50, 0.80, 0.95, 0.999]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        v_persistent = cardinal_interp(x, y; bc = bc, tension = 0.3).(xq)
        v_oneshot = cardinal_interp(x, y, xq; bc = bc, tension = 0.3)
        @test v_persistent ≈ v_oneshot atol = 1.0e-12
    end

    @testset "Range grid with bc.period === nothing (codex P2)" begin
        xr = range(0.0, 1.0, length = 11)[1:10]
        y = f.(xr)
        bc = PeriodicBC(endpoint = :exclusive)
        @test bc.period === nothing

        for cs in (PreCompute(), OnTheFly()), tens in (0.0, 0.5)
            v = cardinal_interp(xr, y, 0.5; bc = bc, coeffs = cs, tension = tens)
            @test isfinite(v)
        end
        itp = cardinal_interp(xr, y; bc = bc)
        @test itp(0.5) ≈ f(0.5) atol = 1.0e-3
    end

    @testset "Approximation accuracy" begin
        n = 41
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = cardinal_interp(x, y; bc = bc, tension = 0.0)
        for xq in (0.13, 0.37, 0.62, 0.88)
            @test itp(xq) ≈ f(xq) atol = 1.0e-3
        end
    end
end
