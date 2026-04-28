# Tests for PeriodicBC on PCHIP interpolation.
#
# Coverage:
#   - `NoBC()` default is a no-op (regression guard).
#   - C¹ continuity at the seam (the main correctness contract).
#   - Both endpoints (`:inclusive`, `:exclusive`).
#   - Both `coeffs` strategies (OnTheFly, PreCompute).
#   - Both paths (oneshot scalar/vector, persistent interpolant).
#   - Inclusive endpoint mismatch raises.
#   - Match against extended-grid baseline (closed-cycle equivalence).

@testitem "PCHIP PeriodicBC" setup = [AllocConstants] begin
    f(x) = sin(2π * x)
    fp(x) = 2π * cos(2π * x)

    @testset "NoBC default is no-op" begin
        x = collect(range(0.0, 1.0, length = 11))
        y = f.(x)

        itp_default = pchip_interp(x, y)
        itp_nobc = pchip_interp(x, y; bc = NoBC())
        @test typeof(itp_default) === typeof(itp_nobc)
        for xq in (0.05, 0.25, 0.5, 0.75, 0.95)
            @test itp_default(xq) === itp_nobc(xq)
        end
    end

    @testset "Inclusive — endpoint mismatch raises" begin
        x = collect(range(0.0, 1.0, length = 5))
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError pchip_interp(x, y; bc = PeriodicBC())
    end

    @testset "C¹ at seam — exclusive, persistent, OnTheFly + PreCompute" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            itp = pchip_interp(x, y; bc = bc, coeffs = cs)
            v_left = itp(1.0 - ε; deriv = DerivOp(1))
            v_right = itp(0.0 + ε; deriv = DerivOp(1))
            @test isapprox(v_left, v_right; atol = 1.0e-5)         # C¹ within numeric noise
        end
    end

    @testset "C¹ at seam — inclusive, persistent, OnTheFly + PreCompute" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n))
        y = f.(x)
        @assert isapprox(y[1], y[end]; atol = 1.0e-12)
        bc = PeriodicBC(endpoint = :inclusive)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            itp = pchip_interp(x, y; bc = bc, coeffs = cs)
            v_left = itp(1.0 - ε; deriv = DerivOp(1))
            v_right = itp(0.0 + ε; deriv = DerivOp(1))
            @test isapprox(v_left, v_right; atol = 1.0e-5)
        end
    end

    @testset "C¹ at seam — exclusive, oneshot, OnTheFly + PreCompute" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        ε = 1.0e-7

        for cs in (PreCompute(), OnTheFly())
            v_left = pchip_interp(x, y, 1.0 - ε; bc = bc, coeffs = cs, deriv = DerivOp(1))
            v_right = pchip_interp(x, y, 0.0 + ε; bc = bc, coeffs = cs, deriv = DerivOp(1))
            @test isapprox(v_left, v_right; atol = 1.0e-5)
        end
    end

    @testset "Wrap query past seam — value continuity" begin
        n = 21
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = pchip_interp(x, y; bc = bc)
        # Query past the seam should wrap modulo period
        @test itp(1.25) ≈ itp(0.25) atol = 1.0e-12
        @test itp(-0.30) ≈ itp(0.70) atol = 1.0e-12
        @test itp(2.5) ≈ itp(0.5) atol = 1.0e-12
    end

    @testset "Match closed-cycle baseline (exclusive ↔ inclusive equivalence)" begin
        # Exclusive (n length) and inclusive (n+1 length, with y[end]=y[1]) on the
        # same physical periodic grid must produce identical interpolants.
        n_excl = 20
        x_excl = collect(range(0.0, 1.0, length = n_excl + 1))[1:n_excl]
        y_excl = f.(x_excl)

        x_incl = vcat(x_excl, [1.0])      # close the cycle
        y_incl = vcat(y_excl, [y_excl[1]])

        itp_excl = pchip_interp(x_excl, y_excl; bc = PeriodicBC(endpoint = :exclusive, period = 1.0))
        itp_incl = pchip_interp(x_incl, y_incl; bc = PeriodicBC(endpoint = :inclusive))

        for xq in (0.05, 0.25, 0.5, 0.75, 0.95, 0.999)
            @test itp_excl(xq) ≈ itp_incl(xq) atol = 1.0e-12
            @test itp_excl(xq; deriv = DerivOp(1)) ≈ itp_incl(xq; deriv = DerivOp(1)) atol = 1.0e-10
        end
    end

    @testset "Vector oneshot — element-wise match against persistent" begin
        n = 25
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        xq = [0.05, 0.20, 0.50, 0.80, 0.95, 0.999]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        v_persistent = pchip_interp(x, y; bc = bc).(xq)
        v_oneshot = pchip_interp(x, y, xq; bc = bc)

        @test v_persistent ≈ v_oneshot atol = 1.0e-12
    end

    @testset "Range grid with bc.period === nothing (codex P2)" begin
        # `_periodic_secant` previously called `bc.period - ...` directly, which
        # threw MethodError when period was unresolved. Range-inferred period
        # via `_resolve_exclusive_period` now flows correctly through the
        # zero-copy oneshot path.
        xr = range(0.0, 1.0, length = 11)[1:10]    # AbstractRange, exclusive form
        y = f.(xr)
        bc = PeriodicBC(endpoint = :exclusive)     # period unresolved (nothing)
        @test bc.period === nothing

        for cs in (PreCompute(), OnTheFly())
            v = pchip_interp(xr, y, 0.5; bc = bc, coeffs = cs)
            @test isfinite(v)
            @test v ≈ f(0.5) atol = 1.0e-3
        end
        # Persistent path (extension copies, exclusive→inclusive normalize)
        itp = pchip_interp(xr, y; bc = bc)
        @test itp(0.5) ≈ f(0.5) atol = 1.0e-3
    end

    @testset "Approximation accuracy — periodic data" begin
        # PCHIP on smooth periodic data should match the function within
        # second-order accuracy on a moderately-fine grid.
        n = 41
        x = collect(range(0.0, 1.0, length = n + 1))[1:n]
        y = f.(x)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)

        itp = pchip_interp(x, y; bc = bc)
        for xq in (0.13, 0.37, 0.62, 0.88)
            @test itp(xq) ≈ f(xq) atol = 1.0e-3
        end
    end

    @testset "Zero-alloc 1D oneshot OnTheFly (function barrier)" begin
        # Function-barrier pattern: setup + warmup + @allocated all inside one
        # function so locals are concrete-typed (the @testset try/catch wrapper
        # would otherwise box and inflate the alloc count).
        function _alloc_check_excl()
            n = 21
            x = collect(range(0.0, 1.0, length = n + 1))[1:n]
            y = f.(x)
            bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
            pchip_interp(x, y, 0.5; bc = bc, coeffs = OnTheFly())            # warmup
            return @allocated pchip_interp(x, y, 0.5; bc = bc, coeffs = OnTheFly())
        end
        function _alloc_check_incl()
            n = 21
            x = collect(range(0.0, 1.0, length = n))
            y = f.(x); y[end] = y[1]
            bc = PeriodicBC(endpoint = :inclusive)
            pchip_interp(x, y, 0.5; bc = bc, coeffs = OnTheFly())            # warmup
            return @allocated pchip_interp(x, y, 0.5; bc = bc, coeffs = OnTheFly())
        end
        @test _alloc_check_excl() <= ALLOC_THRESHOLD
        @test _alloc_check_incl() <= ALLOC_THRESHOLD
    end

    @testset "Float32 grid + user-supplied Float64 period (codex P2)" begin
        # Regression: Float64 `bc.period` on a Float32 grid must not widen
        # results to Float64 (silent precision change) — verified at the seam
        # cell where the period participates in `xR` and `seam_h`.
        x32 = collect(Float32, range(0f0, 1f0, length = 8))[1:7]
        y32 = sin.(2f0 .* Float32(pi) .* x32)
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)              # Float64 user period

        itp = pchip_interp(x32, y32; bc = bc)
        for q in (0.42f0, 0.86f0, 0.95f0)
            r_scalar = pchip_interp(x32, y32, q; bc = bc)
            r_persist = itp(q)
            @test r_scalar isa Float32
            @test r_persist isa Float32
            @test isapprox(r_scalar, r_persist; atol = 1.0e-6)
        end
    end

    @testset "n=2 :exclusive scalar↔persistent consistency (codex P2)" begin
        # Regression: the OnTheFly + PreCompute n==2 fast paths previously
        # bypassed BC dispatch and returned the linear secant, diverging from
        # the persistent path's wrap-aware boundary slope on `:exclusive` data
        # whose seam-cell secant has opposite sign of cell 1.
        x = [0.0, 0.3]; y = [0.0, 1.0]
        bc = PeriodicBC(endpoint = :exclusive, period = 1.0)
        itp = pchip_interp(x, y; bc = bc)
        for q in (0.05, 0.15, 0.5, 0.8)
            persist = itp(q)
            for cs in (OnTheFly(), PreCompute())
                @test pchip_interp(x, y, q; bc = bc, coeffs = cs) ≈ persist atol = 1.0e-12
            end
        end
    end
end
