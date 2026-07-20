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

# Constant ND full-domain: the per-axis side weights also collapse on full cells
# (Left → h at the left node, Right → h at the right node, Nearest → ½h each),
# so the domain integral is a separable side-weighted Riemann sum. Oracle 1 is
# an explicit cell-sum reference implementation (independent of both engines);
# oracle 2 is the generic bounded engine at full bounds. NearestSide weights
# equal the linear composite-trapezoid weights, giving a third cross-family pin.
@testitem "ND constant separable: cell-sum + bounded parity (all sides)" setup = [AllocConstants] begin
    # reference: I = Σ_cells Σ_{s∈{0,1}^N} Π_d wpair(side_d, h_d(c_d))[s_d+1] · data[c+s]
    wpair(::LeftSide, h) = (h, zero(h))
    wpair(::RightSide, h) = (zero(h), h)
    wpair(::NearestSide, h) = (h / 2, h / 2)
    function brute(grids, data, sides)
        N = length(grids)
        total = 0.0
        for c in CartesianIndices(ntuple(d -> 1:(length(grids[d]) - 1), N))
            for s in CartesianIndices(ntuple(_ -> 0:1, N))
                w = 1.0
                for d in 1:N
                    h = Float64(grids[d][c[d] + 1] - grids[d][c[d]])
                    w *= wpair(sides[d], h)[s[d] + 1]
                end
                total += w * Float64(data[c + s])
            end
        end
        return total
    end

    xv = [0.0, 0.3, 0.7, 1.1, 1.8, 2.0]
    yv = [0.0, 0.5, 1.2, 2.0, 3.0]
    xr = range(0.0, 2.0, length = 6)
    yr = range(0.0, 3.0, length = 5)

    @testset "2D all side combos — Range + Vector" begin
        for (gx, gy) in ((xr, yr), (xv, yv))
            data = [sin(xi) + cos(yj) for xi in gx, yj in gy]
            lo = (first(gx), first(gy));  hi = (last(gx), last(gy))
            for sx in (LeftSide(), RightSide(), NearestSide()),
                    sy in (LeftSide(), RightSide(), NearestSide())

                itp = constant_interp((gx, gy), data; side = (sx, sy), extrap = NoExtrap())
                @test integrate(itp) ≈ brute((gx, gy), data, (sx, sy)) rtol = 1.0e-12
                @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
            end
        end
    end

    @testset "3D mixed sides — Vector (non-uniform)" begin
        zv = [0.0, 0.35, 0.8, 1.0]
        data = [xi + 2yj - zk for xi in xv, yj in yv, zk in zv]
        sides = (LeftSide(), NearestSide(), RightSide())
        itp = constant_interp((xv, yv, zv), data; side = sides, extrap = NoExtrap())
        lo = (first(xv), first(yv), first(zv));  hi = (last(xv), last(yv), last(zv))
        @test integrate(itp) ≈ brute((xv, yv, zv), data, sides) rtol = 1.0e-12
        @test integrate(itp) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
    end

    @testset "NearestSide ≡ linear composite trapezoid (full domain)" begin
        for (gx, gy) in ((xr, yr), (xv, yv))
            data = [xi^2 + yj for xi in gx, yj in gy]
            itp_c = constant_interp((gx, gy), data; side = NearestSide())
            itp_l = linear_interp((gx, gy), data)
            @test integrate(itp_c) ≈ integrate(itp_l) rtol = 1.0e-12
        end
    end

    @testset "Int data promotes to Float output" begin
        di = [xi + yj for xi in 1:4, yj in 1:3]
        itp = constant_interp((collect(1.0:4.0), collect(1.0:3.0)), di; side = NearestSide())
        @test integrate(itp) isa AbstractFloat
        @test integrate(itp) ≈ brute((1.0:4.0, 1.0:3.0), di, (NearestSide(), NearestSide())) rtol = 1.0e-12
    end

    @testset "zero allocation (Range + Vector, homogeneous + mixed sides)" begin
        xr2 = range(0.0, 2.0, length = 20);  yr2 = range(0.0, 3.0, length = 16)
        data = [sin(xi) * cos(yj) for xi in xr2, yj in yr2]
        for sides in ((NearestSide(), NearestSide()), (LeftSide(), RightSide()))
            itp_r = constant_interp((xr2, yr2), data; side = sides)
            itp_v = constant_interp((collect(xr2), collect(yr2)), data; side = sides)
            integrate(itp_r);  integrate(itp_v)   # warmup
            @test @allocated(integrate(itp_r)) <= ND_ALLOC_THRESHOLD
            @test @allocated(integrate(itp_v)) <= ND_ALLOC_THRESHOLD
        end
    end
end

# Bounded ND (linear/constant): separability holds on any axis-aligned box —
# per-axis weights become "clipped composites" (zero outside the cell range,
# partial `_w0_int`/`_w1_int`-style weights at the two boundary cells, full
# weights inside), so the sum visits only the node sub-box. These tests pin the
# bounded semantics through engine-independent oracles before the path lands:
# analytic exact fields, box == domain ≡ full-domain, axis additivity, sign.
@testitem "ND bounded separable: oracles (linear + constant)" setup = [AllocConstants] begin
    bilin_expected(lo, hi) = begin
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        X2 * Y2 + 2 * X2 * Y1 - 3 * X1 * Y2 + 5 * X1 * Y1
    end
    mk2d(x, y) = [xi * yj + 2xi - 3yj + 5 for xi in x, yj in y]

    xv = [0.0, 0.3, 0.7, 1.1, 1.8, 2.0]
    yv = [0.0, 0.5, 1.2, 2.0, 3.0]
    xr = range(0.0, 2.0, length = 6)
    yr = range(0.0, 3.0, length = 5)

    @testset "linear 2D analytic sub-boxes (Range + Vector)" begin
        for (gx, gy) in ((xr, yr), (xv, yv))
            itp = linear_interp((gx, gy), mk2d(gx, gy); extrap = NoExtrap())
            for (lo, hi) in (
                    ((0.25, 0.4), (1.65, 2.6)),     # interior, cuts cells mid-way
                    ((0.0, 0.0), (2.0, 3.0)),       # box == domain
                    ((0.31, 0.55), (0.65, 1.15)),   # within few cells
                    ((0.72, 1.25), (1.05, 1.95)),   # single-cell box on both axes (vector grid)
                )
                @test integrate(itp, lo, hi) ≈ bilin_expected(lo, hi) rtol = 1.0e-12
            end
            # box == domain matches the full-domain separable path
            @test integrate(itp, (0.0, 0.0), (2.0, 3.0)) ≈ integrate(itp) rtol = 1.0e-12
        end
    end

    @testset "linear 3D analytic sub-box + additivity" begin
        z = [0.0, 0.35, 0.8, 1.0]
        aff_expected(lo, hi) = begin
            X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
            Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
            Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2
            2 * X2 * Y1 * Z1 + 3 * X1 * Y2 * Z1 - X1 * Y1 * Z2 + 4 * X1 * Y1 * Z1
        end
        data = [2xi + 3yj - zk + 4 for xi in xv, yj in yv, zk in z]
        itp = linear_interp((xv, yv, z), data; extrap = NoExtrap())
        lo = (0.15, 0.4, 0.1);  hi = (1.7, 2.55, 0.9)
        @test integrate(itp, lo, hi) ≈ aff_expected(lo, hi) rtol = 1.0e-12
        # additivity along axis 1 at a non-node cut
        c = 0.95
        @test integrate(itp, lo, hi) ≈
            integrate(itp, lo, (c, hi[2], hi[3])) + integrate(itp, (c, lo[2], lo[3]), hi) rtol = 1.0e-12
    end

    @testset "per-axis orientation sign; degenerate box is zero" begin
        itp = linear_interp((xv, yv), mk2d(xv, yv))
        lo = (0.25, 0.4);  hi = (1.65, 2.6)
        # one reversed axis → −1; both reversed → (−1)² = +1
        @test integrate(itp, (hi[1], lo[2]), (lo[1], hi[2])) ≈ -integrate(itp, lo, hi) rtol = 1.0e-12
        @test integrate(itp, hi, lo) ≈ integrate(itp, lo, hi) rtol = 1.0e-12
        @test integrate(itp, (0.5, 0.4), (0.5, 2.6)) == 0
    end

    @testset "constant: box-volume identity + additivity (all sides, mixed)" begin
        ones2d = fill(1.0, length(xv), length(yv))
        for side in (
                NearestSide(), LeftSide(), RightSide(),
                (LeftSide(), NearestSide()),
            )
            itp = constant_interp((xv, yv), ones2d; side = side)
            lo = (0.2, 0.45);  hi = (1.75, 2.7)
            # unit data integrates to the box volume regardless of side
            @test integrate(itp, lo, hi) ≈ (hi[1] - lo[1]) * (hi[2] - lo[2]) rtol = 1.0e-12
        end
        data = [sin(xi) + cos(yj) for xi in xv, yj in yv]
        for side in (NearestSide(), LeftSide(), RightSide())
            itp = constant_interp((xv, yv), data; side = side)
            lo = (0.2, 0.45);  hi = (1.75, 2.7)
            c = 1.0                                     # non-node cut
            @test integrate(itp, lo, hi) ≈
                integrate(itp, lo, (c, hi[2])) + integrate(itp, (c, lo[2]), hi) rtol = 1.0e-12
            @test integrate(itp, (first(xv), first(yv)), (last(xv), last(yv))) ≈
                integrate(itp) rtol = 1.0e-12
        end
    end

    @testset "zero allocation (Range + Vector)" begin
        xr2 = range(0.0, 2.0, length = 20);  yr2 = range(0.0, 3.0, length = 16)
        data = [sin(xi) * cos(yj) for xi in xr2, yj in yr2]
        lo = (0.2, 0.4);  hi = (1.8, 2.7)
        for itp in (
                linear_interp((xr2, yr2), data),
                linear_interp((collect(xr2), collect(yr2)), data),
                constant_interp((xr2, yr2), data; side = NearestSide()),
            )
            integrate(itp, lo, hi)   # warmup
            @test @allocated(integrate(itp, lo, hi)) <= ND_ALLOC_THRESHOLD
        end
    end
end
