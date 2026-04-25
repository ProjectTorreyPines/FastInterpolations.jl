# ALLOC_THRESHOLD is defined in runtests.jl

@testitem "Integration Allocation Tests" setup=[AllocConstants] begin

    # ═══════════════════════════════════════════════════════════════
    # 1D Linear — should be 0B (no _create_spacing)
    # ═══════════════════════════════════════════════════════════════

    @testset "linear 1D: zero allocation" begin
        x = collect(range(0.0, 1.0, length = 21))
        y = @. 3x - 1
        itp = linear_interp(x, y; extrap = NoExtrap())
        a, b = 0.15, 0.85

        # Warmup
        integrate(itp, a, b)
        integrate(itp, a, b)

        allocs = @allocated integrate(itp, a, b)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ═══════════════════════════════════════════════════════════════
    # 1D Quadratic — should be 0B (no _create_spacing)
    # ═══════════════════════════════════════════════════════════════

    @testset "quadratic 1D: zero allocation" begin
        x = collect(range(0.0, 1.0, length = 21))
        y = @. 2x^2 - x + 4
        itp = quadratic_interp(x, y; extrap = NoExtrap())
        a, b = 0.1, 0.9

        # Warmup
        integrate(itp, a, b)
        integrate(itp, a, b)

        allocs = @allocated integrate(itp, a, b)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ═══════════════════════════════════════════════════════════════
    # 1D Constant — should be 0B (no _create_spacing)
    # ═══════════════════════════════════════════════════════════════

    @testset "constant 1D: zero allocation" begin
        x = collect(range(0.0, 1.0, length = 21))
        y = collect(1.0:length(x))
        for side in (LeftSide(), RightSide(), NearestSide())
            itp = constant_interp(x, y; side = side, extrap = NoExtrap())
            a, b = 0.2, 0.7

            # Warmup
            for _ in 1:3
                integrate(itp, a, b)
            end

            allocs = @allocated integrate(itp, a, b)
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # 2D Cubic — should be 0B (no Core.Box from _normalize_bounds_nd)
    # ═══════════════════════════════════════════════════════════════

    @testset "cubic 2D: zero allocation (no hint)" begin
        xg = collect(range(0.0, 1.0, length = 11))
        yg = collect(range(0.0, 1.0, length = 11))
        data = [sin(x + y) for x in xg, y in yg]
        itp = cubic_interp((xg, yg), data)
        lo = (0.2, 0.2)
        hi = (0.8, 0.8)

        # Warmup
        integrate(itp, lo, hi)
        integrate(itp, lo, hi)

        allocs = @allocated integrate(itp, lo, hi)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ═══════════════════════════════════════════════════════════════
    # 1D Cubic — should remain 0B (baseline, no regressions)
    # ═══════════════════════════════════════════════════════════════

    @testset "cubic 1D: zero allocation (baseline)" begin
        x = collect(range(0.0, 1.0, length = 21))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = NoExtrap())
        a, b = 0.15, 0.85

        # Warmup
        integrate(itp, a, b)
        integrate(itp, a, b)

        allocs = @allocated integrate(itp, a, b)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ═══════════════════════════════════════════════════════════════
    # FillExtrap fill value — integration path zero allocation
    # ═══════════════════════════════════════════════════════════════

    @testset "FillExtrap fill value: integrate zero allocation" begin
        x = collect(range(0.0, 1.0, length = 21))
        y = @. 3x - 1

        itp_clamp = linear_interp(x, y; extrap = ClampExtrap())
        itp_zero = linear_interp(x, y; extrap = FillExtrap(0.0))
        itp_42 = linear_interp(x, y; extrap = FillExtrap(42.0))

        function integrate_fill(itp, a, b)
            integrate(itp, a, b)
        end

        # Warmup
        for itp in (itp_clamp, itp_zero, itp_42)
            integrate_fill(itp, 0.15, 0.85)   # in-domain
            integrate_fill(itp, -0.5, 1.5)     # spans outside
            integrate_fill(itp, 0.15, 0.85)
            integrate_fill(itp, -0.5, 1.5)
        end

        # In-domain integration
        for itp in (itp_clamp, itp_zero, itp_42)
            allocs = @allocated integrate_fill(itp, 0.15, 0.85)
            @test allocs <= ALLOC_THRESHOLD
        end

        # Cross-boundary integration (exercises fill value arithmetic)
        for itp in (itp_clamp, itp_zero, itp_42)
            allocs = @allocated integrate_fill(itp, -0.5, 1.5)
            @test allocs <= ALLOC_THRESHOLD
        end
    end
end
