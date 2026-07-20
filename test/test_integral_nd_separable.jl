# ═══════════════════════════════════════════════════════════════════════
# ND linear full-domain integrate — separable composite-trapezoid path
#
# The full-domain integral of a multilinear interpolant factorizes into a
# tensor product of 1D composite-trapezoid rules:
#
#     I = Σ_nodes (Π_d w_d(i_d)) · data[i],   w_d = ½h at ends, ½(h₋+h₊) interior
#
# so every node is read once (vs 2^N corner reads per cell in the generic
# per-cell kernel). This suite pins that path against two independent oracles
# — the analytic integral of a multilinear-exact integrand, and the generic
# bounded engine `integrate(itp, lo, hi)` — on Range and Vector grids, 2D/3D.
# ═══════════════════════════════════════════════════════════════════════

@testitem "ND linear separable: analytic + bounded parity" setup = [AllocConstants] begin
    # 2D bilinear f(x,y) = xy + 2x − 3y + 5 (reproduced exactly by bilinear itp)
    bilin_expected(lo, hi) = begin
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        X2 * Y2 + 2 * X2 * Y1 - 3 * X1 * Y2 + 5 * X1 * Y1
    end
    mk2d(x, y) = [xi * yj + 2xi - 3yj + 5 for xi in x, yj in y]

    @testset "2D bilinear — Range grid" begin
        x = range(0.0, 2.0, length = 21)
        y = range(0.0, 3.0, length = 17)
        itp = linear_interp((x, y), mk2d(x, y); extrap = NoExtrap())
        lo = (first(x), first(y));  hi = (last(x), last(y))
        @test integrate(itp) ≈ bilin_expected(lo, hi) rtol = 1.0e-12
        @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
    end

    @testset "2D bilinear — Vector grid (non-uniform)" begin
        x = [0.0, 0.3, 0.7, 1.1, 1.8, 2.0]
        y = [0.0, 0.5, 1.2, 2.0, 3.0]
        itp = linear_interp((x, y), mk2d(x, y); extrap = NoExtrap())
        lo = (first(x), first(y));  hi = (last(x), last(y))
        @test integrate(itp) ≈ bilin_expected(lo, hi) rtol = 1.0e-12   # bilinear exact on non-uniform
        @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
    end

    # 3D affine f = 2x + 3y − z + 4
    aff_expected(lo, hi) = begin
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2
        2 * X2 * Y1 * Z1 + 3 * X1 * Y2 * Z1 - X1 * Y1 * Z2 + 4 * X1 * Y1 * Z1
    end
    mk3d(x, y, z) = [2xi + 3yj - zk + 4 for xi in x, yj in y, zk in z]

    @testset "3D affine — Range grid" begin
        x = range(0.0, 2.0, length = 11)
        y = range(0.0, 3.0, length = 13)
        z = range(0.0, 1.0, length = 9)
        itp = linear_interp((x, y, z), mk3d(x, y, z); extrap = NoExtrap())
        lo = (first(x), first(y), first(z));  hi = (last(x), last(y), last(z))
        @test integrate(itp) ≈ aff_expected(lo, hi) rtol = 1.0e-12
        @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
    end

    @testset "3D affine — Vector grid (non-uniform)" begin
        x = [0.0, 0.4, 0.9, 1.5, 2.0]
        y = [0.0, 0.6, 1.3, 2.1, 3.0]
        z = [0.0, 0.35, 0.8, 1.0]
        itp = linear_interp((x, y, z), mk3d(x, y, z); extrap = NoExtrap())
        lo = (first(x), first(y), first(z));  hi = (last(x), last(y), last(z))
        @test integrate(itp) ≈ aff_expected(lo, hi) rtol = 1.0e-12
        @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
    end

    @testset "3D smooth field — Vector parity vs bounded oracle" begin
        x = [0.0, 0.4, 0.9, 1.5, 2.0]
        y = [0.0, 0.6, 1.3, 2.1, 3.0]
        z = [0.0, 0.35, 0.8, 1.0]
        data = [sin(xi) * cos(yj) + zk^2 for xi in x, yj in y, zk in z]  # not exact, oracle = bounded
        itp = linear_interp((x, y, z), data; extrap = NoExtrap())
        lo = (first(x), first(y), first(z));  hi = (last(x), last(y), last(z))
        @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
    end

    @testset "Range and Vector grids agree (same nodes)" begin
        xr = range(0.0, 2.0, length = 15);  yr = range(-1.0, 1.0, length = 11)
        data = [sin(xi) + cos(yj) for xi in xr, yj in yr]
        itp_r = linear_interp((xr, yr), data; extrap = NoExtrap())
        itp_v = linear_interp((collect(xr), collect(yr)), data; extrap = NoExtrap())
        @test integrate(itp_r) ≈ integrate(itp_v) rtol = 1.0e-13
    end

    @testset "zero allocation (Range + Vector)" begin
        xr = range(0.0, 2.0, length = 20);  yr = range(0.0, 3.0, length = 16)
        data = [sin(xi) * cos(yj) for xi in xr, yj in yr]
        itp_r = linear_interp((xr, yr), data)
        itp_v = linear_interp((collect(xr), collect(yr)), data)
        integrate(itp_r);  integrate(itp_v)   # warmup
        @test @allocated(integrate(itp_r)) <= ND_ALLOC_THRESHOLD
        @test @allocated(integrate(itp_v)) <= ND_ALLOC_THRESHOLD
    end
end

# The separable path multiplies each value by a grid-typed (Float) weight before
# summing, so fixed-point carriers widen and never modular-wrap — the ND analog
# of the 1D endpoint-wrap contract.
@testitem "ND linear separable: N0f8 carrier no-wrap" begin
    using FixedPointNumbers

    x = 0.0:0.25:1.0
    y = 0.0:0.5:1.0
    dataN = N0f8.([0.8 0.4 0.8; 0.4 0.4 0.4; 0.8 0.4 0.8; 0.4 0.8 0.4; 0.8 0.4 0.8])
    dataF = Float64.(dataN)
    for gx in (x, collect(x)), gy in (y, collect(y))
        itpN = linear_interp((gx, gy), dataN; extrap = NoExtrap())
        itpF = linear_interp((gx, gy), dataF; extrap = NoExtrap())
        @test Float64(integrate(itpN)) ≈ integrate(itpF) rtol = 1.0e-9
    end
end
