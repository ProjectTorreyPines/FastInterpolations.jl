# ═══════════════════════════════════════════════════════════════════════
# Hetero ND integrate — mixed per-axis methods through the separable engine
#
# The separable node-weight engine keys on per-axis method tags, so mixed
# tensor-product axes (Linear/Cubic/Quadratic/Constant) integrate exactly like
# their homogeneous counterparts. Oracle: the engine-independent per-fiber 1D
# composition (`comp_nd` in the NDCompositionOracle snippet).
# ═══════════════════════════════════════════════════════════════════════

@testitem "hetero ND integrate: mixed full-domain vs 1D composition" setup = [Basic, NDCompositionOracle] begin
    x = [0.0, 0.3, 0.7, 1.1, 1.8, 2.0]
    y = range(0.0, 3.0, length = 9)
    A = [sin(xi) * cos(yj) + 0.5 * xi for xi in x, yj in y]

    @testset "PreCompute mixes (rank-1 × rank-2, rank-2 × rank-2)" begin
        for ms in (
                (CubicInterp(), LinearInterp()),
                (LinearInterp(), CubicInterp()),
                (QuadraticInterp(), LinearInterp()),
                (LinearInterp(), QuadraticInterp()),
                (CubicInterp(), QuadraticInterp()),
            )
            itp = interp((x, y), A; method = ms, coeffs = PreCompute())
            @test itp isa FI.HeteroInterpolantND
            @test integrate(itp) ≈ comp_nd(ms, (x, y), A) rtol = 1.0e-11
        end
    end

    @testset "Range × Range grids" begin
        xr = range(0.0, 2.0, length = 13)
        Ar = [sin(xi) * cos(yj) for xi in xr, yj in y]
        ms = (CubicInterp(), LinearInterp())
        itp = interp((xr, y), Ar; method = ms, coeffs = PreCompute())
        @test integrate(itp) ≈ comp_nd(ms, (xr, y), Ar) rtol = 1.0e-11
    end

    @testset "OnTheFly rank-1 mixes (raw data payload)" begin
        ms = (LinearInterp(), ConstantInterp(side = LeftSide()))
        itp = interp((x, y), A; method = ms, coeffs = OnTheFly())
        @test itp isa FI.HeteroInterpolantND
        @test integrate(itp) ≈ comp_nd(ms, (x, y), A) rtol = 1.0e-12

        # NearestSide constant weights ≡ composite trapezoid ≡ linear weights
        itp_n = interp((x, y), A; method = (LinearInterp(), ConstantInterp()), coeffs = OnTheFly())
        itp_l = interp((x, y), A; method = (LinearInterp(), LinearInterp()), coeffs = OnTheFly())
        @test integrate(itp_n) ≈ integrate(itp_l) rtol = 1.0e-14
    end

    @testset "all-linear hetero == homogeneous linear" begin
        itp_h = interp((x, y), A; method = (LinearInterp(), LinearInterp()), coeffs = OnTheFly())
        itp_l = linear_interp((x, y), A)
        @test integrate(itp_h) ≈ integrate(itp_l) rtol = 1.0e-14
    end

    @testset "periodic linear axis (closed-domain storage)" begin
        xp = range(0.0, 2π, length = 17)
        P = [sin(xi) * (1.0 + 0.3 * yj) for xi in xp, yj in y]   # P[end,:] == P[1,:]
        itp_h = interp(
            (xp, y), P;
            method = (LinearInterp(bc = PeriodicBC()), LinearInterp()), coeffs = OnTheFly()
        )
        itp_l = linear_interp((xp, y), P; bc = (PeriodicBC(), NoBC()))
        @test integrate(itp_h) ≈ integrate(itp_l) rtol = 1.0e-13
    end

    @testset "3D Linear × Cubic × Quadratic" begin
        y3 = range(0.0, 1.5, length = 7)
        z = [0.0, 0.35, 0.8, 1.0]
        B = [cos(xi) + yj * zk + 0.2 * zk^2 for xi in x, yj in y3, zk in z]
        ms = (LinearInterp(), CubicInterp(), QuadraticInterp())
        itp = interp((x, y3, z), B; method = ms, coeffs = PreCompute())
        @test itp isa FI.HeteroInterpolantND
        @test integrate(itp) ≈ comp_nd(ms, (x, y3, z), B) rtol = 1.0e-10
    end
end

@testitem "hetero ND integrate: mixed bounded boxes" setup = [Basic, NDCompositionOracle] begin
    x = [0.0, 0.3, 0.7, 1.1, 1.8, 2.0]
    y = range(0.0, 3.0, length = 9)
    A = [sin(xi) * cos(yj) + 0.5 * xi for xi in x, yj in y]

    @testset "box sweep vs composition oracle" begin
        for ms in ((CubicInterp(), LinearInterp()), (CubicInterp(), QuadraticInterp()))
            itp = interp((x, y), A; method = ms, coeffs = PreCompute())
            for (lo, hi) in (
                    ((0.2, 0.4), (1.5, 2.6)),            # interior, non-node cuts
                    ((x[2], y[3]), (x[4], y[6])),        # node-aligned
                    ((first(x), first(y)), (last(x), last(y))),  # box == domain
                )
                @test integrate(itp, lo, hi) ≈ comp_nd(ms, (x, y), A, lo, hi) rtol = 1.0e-11
            end
            lo = (first(x), first(y))
            hi = (last(x), last(y))
            @test integrate(itp, lo, hi) ≈ integrate(itp) rtol = 1.0e-12
        end
    end

    @testset "axis additivity at a non-node cut" begin
        ms = (CubicInterp(), LinearInterp())
        itp = interp((x, y), A; method = ms, coeffs = PreCompute())
        lo = (0.1, 0.3)
        hi = (1.9, 2.7)
        xc = 0.95
        left = integrate(itp, lo, (xc, hi[2]))
        right = integrate(itp, (xc, lo[2]), hi)
        @test left + right ≈ integrate(itp, lo, hi) rtol = 1.0e-11
    end

    @testset "orientation sign" begin
        ms = (CubicInterp(), LinearInterp())
        itp = interp((x, y), A; method = ms, coeffs = PreCompute())
        lo = (0.2, 0.4)
        hi = (1.5, 2.6)
        ref = integrate(itp, lo, hi)
        @test integrate(itp, (hi[1], lo[2]), (lo[1], hi[2])) ≈ -ref rtol = 1.0e-11
        @test integrate(itp, hi, lo) ≈ ref rtol = 1.0e-11
    end

    @testset "OnTheFly rank-1 bounded" begin
        ms = (LinearInterp(), ConstantInterp(side = LeftSide()))
        itp = interp((x, y), A; method = ms, coeffs = OnTheFly())
        lo = (0.2, 0.4)
        hi = (1.5, 2.6)
        @test integrate(itp, lo, hi) ≈ comp_nd(ms, (x, y), A, lo, hi) rtol = 1.0e-12
    end
end

@testitem "hetero ND integrate: gates and error messages" begin
    const FI = FastInterpolations
    x = [0.0, 0.3, 0.7, 1.1, 1.8, 2.0]
    y = range(0.0, 3.0, length = 9)
    A = [sin(xi) * cos(yj) for xi in x, yj in y]

    @testset "OnTheFly with a derivative axis needs PreCompute" begin
        itp = interp((x, y), A; method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly())
        @test_throws "PreCompute" integrate(itp)
        @test_throws "PreCompute" integrate(itp, (0.2, 0.4), (1.5, 2.6))
    end

    @testset "local-Hermite axes stay unsupported" begin
        itp = interp((x, y), A; method = (PchipInterp(), LinearInterp()))
        @test_throws ArgumentError integrate(itp)
        @test_throws ArgumentError integrate(itp, (0.2, 0.4), (1.5, 2.6))
    end

    @testset "NoInterp axes stay unsupported" begin
        itp = interp((x, y), A; method = (LinearInterp(), NoInterp()))
        @test_throws ArgumentError integrate(itp)
    end

    @testset "ND with Real bounds points at tuple form" begin
        lin = linear_interp((x, y), A)
        @test_throws "tuple" integrate(lin, 0.1, 0.5)
    end

    @testset "CubicHermiteInterpolantND: clean not-implemented (not MethodError)" begin
        dfdx = [cos(xi) * cos(yj) for xi in x, yj in y]
        dfdy = [-sin(xi) * sin(yj) for xi in x, yj in y]
        d2 = [-cos(xi) * sin(yj) for xi in x, yj in y]
        p = HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)
        itp = hermite_interp((x, y), A, p)
        @test_throws ArgumentError integrate(itp)
        @test_throws ArgumentError integrate(itp, (0.2, 0.4), (1.5, 2.6))
    end
end

@testitem "hetero ND integrate: zero allocation" setup = [AllocConstants] begin
    x = collect(range(0.0, 2.0, length = 24))
    y = range(0.0, 3.0, length = 20)
    A = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp2 = interp((x, y), A; method = (CubicInterp(), LinearInterp()), coeffs = PreCompute())

    z = range(0.0, 1.0, length = 10)
    B = [cos(xi) + yj * zk for xi in x, yj in y, zk in z]
    itp3 = interp((x, y, z), B; method = (LinearInterp(), CubicInterp(), QuadraticInterp()), coeffs = PreCompute())

    lo2 = (0.2, 0.4)
    hi2 = (1.7, 2.6)
    lo3 = (0.2, 0.4, 0.1)
    hi3 = (1.7, 2.6, 0.9)
    integrate(itp2);  integrate(itp3)                    # warmup
    integrate(itp2, lo2, hi2);  integrate(itp3, lo3, hi3)
    @test @allocated(integrate(itp2)) <= ND_ALLOC_THRESHOLD
    @test @allocated(integrate(itp3)) <= ND_ALLOC_THRESHOLD
    @test @allocated(integrate(itp2, lo2, hi2)) <= ND_ALLOC_THRESHOLD
    @test @allocated(integrate(itp3, lo3, hi3)) <= ND_ALLOC_THRESHOLD
end
