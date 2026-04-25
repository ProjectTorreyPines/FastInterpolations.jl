# ═══════════════════════════════════════════════════════════════════════
# Polynomial exactness tests for ND integrate
#
# Principle: a degree-n tensor-product interpolant reproduces polynomials
# of degree ≤ n in each axis exactly (up to floating-point and BC effects).
# The integral of such a function must therefore match the analytical value.
# ═══════════════════════════════════════════════════════════════════════

@testitem "integrate nd polynomial exactness" begin

    # ───────────────────────────────────────────────
    # 2D polynomial exactness
    # ───────────────────────────────────────────────

    @testset "2D linear: bilinear f(x,y) = xy + 2x − 3y + 5" begin
        x = collect(range(0.0, 2.0, length = 21))
        y = collect(range(0.0, 3.0, length = 17))
        data = [xi * yj + 2xi - 3yj + 5 for xi in x, yj in y]
        itp = linear_interp((x, y), data; extrap = NoExtrap())

        lo, hi = (0.3, 0.5), (1.7, 2.4)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        # ∫∫ (xy + 2x − 3y + 5) dy dx
        expected = X2 * Y2 + 2 * X2 * Y1 - 3 * X1 * Y2 + 5 * X1 * Y1
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-12
    end

    @testset "2D linear: full-domain bilinear" begin
        x = collect(range(0.0, 2.0, length = 21))
        y = collect(range(0.0, 3.0, length = 17))
        data = [xi * yj + 2xi - 3yj + 5 for xi in x, yj in y]
        itp = linear_interp((x, y), data; extrap = NoExtrap())

        lo = (first(x), first(y))
        hi = (last(x), last(y))
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        expected = X2 * Y2 + 2 * X2 * Y1 - 3 * X1 * Y2 + 5 * X1 * Y1
        @test integrate(itp) ≈ expected atol = 1.0e-12
    end

    @testset "2D quadratic: product (x²+1)(y²+1)" begin
        x = collect(range(0.0, 2.0, length = 31))
        y = collect(range(0.0, 3.0, length = 25))
        data = [(xi^2 + 1) * (yj^2 + 1) for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; extrap = NoExtrap())

        lo, hi = (0.2, 0.4), (1.6, 2.5)
        Ix = (hi[1]^3 - lo[1]^3) / 3 + (hi[1] - lo[1])
        Iy = (hi[2]^3 - lo[2]^3) / 3 + (hi[2] - lo[2])
        expected = Ix * Iy
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-8
    end

    @testset "2D quadratic: non-separable x² + xy + y²" begin
        x = collect(range(0.0, 2.0, length = 31))
        y = collect(range(0.0, 3.0, length = 25))
        data = [xi^2 + xi * yj + yj^2 for xi in x, yj in y]
        itp = quadratic_interp((x, y), data; extrap = NoExtrap())

        lo, hi = (0.2, 0.4), (1.6, 2.5)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2;  X3 = (hi[1]^3 - lo[1]^3) / 3
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2;  Y3 = (hi[2]^3 - lo[2]^3) / 3
        # ∫∫ (x² + xy + y²) dy dx
        expected = X3 * Y1 + X2 * Y2 + X1 * Y3
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-8
    end

    @testset "2D cubic: separable (x³ − x)(y³ − y)" begin
        x = collect(range(0.0, 2.0, length = 41))
        y = collect(range(0.0, 3.0, length = 37))
        data = [(xi^3 - xi) * (yj^3 - yj) for xi in x, yj in y]
        itp = cubic_interp((x, y), data; bc = CubicFit(), extrap = NoExtrap())

        lo, hi = (0.3, 0.4), (1.7, 2.6)
        Ix = (hi[1]^4 - lo[1]^4) / 4 - (hi[1]^2 - lo[1]^2) / 2
        Iy = (hi[2]^4 - lo[2]^4) / 4 - (hi[2]^2 - lo[2]^2) / 2
        expected = Ix * Iy
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-10
    end

    @testset "2D cubic: mixed x²y + xy² + x + y" begin
        x = collect(range(0.0, 2.0, length = 41))
        y = collect(range(0.0, 3.0, length = 37))
        data = [xi^2 * yj + xi * yj^2 + xi + yj for xi in x, yj in y]
        itp = cubic_interp((x, y), data; bc = CubicFit(), extrap = NoExtrap())

        lo, hi = (0.3, 0.4), (1.7, 2.6)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        X3 = (hi[1]^3 - lo[1]^3) / 3
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        Y3 = (hi[2]^3 - lo[2]^3) / 3
        # ∫∫ (x²y + xy² + x + y) dy dx
        expected = X3 * Y2 + X2 * Y3 + X2 * Y1 + X1 * Y2
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-10
    end

    @testset "2D cubic: high-degree product x³y³" begin
        x = collect(range(0.0, 2.0, length = 51))
        y = collect(range(0.0, 2.0, length = 51))
        data = [xi^3 * yj^3 for xi in x, yj in y]
        itp = cubic_interp((x, y), data; bc = CubicFit(), extrap = NoExtrap())

        lo, hi = (0.3, 0.3), (1.7, 1.7)
        X4 = (hi[1]^4 - lo[1]^4) / 4
        Y4 = (hi[2]^4 - lo[2]^4) / 4
        expected = X4 * Y4
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-10
    end

    # ───────────────────────────────────────────────
    # 3D polynomial exactness
    # ───────────────────────────────────────────────

    @testset "3D linear: affine f = 2x + 3y − z + 4" begin
        x = collect(range(0.0, 2.0, length = 11))
        y = collect(range(0.0, 3.0, length = 13))
        z = collect(range(0.0, 1.0, length = 9))
        data = [2xi + 3yj - zk + 4 for xi in x, yj in y, zk in z]
        itp = linear_interp((x, y, z), data; extrap = NoExtrap())

        lo, hi = (0.3, 0.5, 0.1), (1.7, 2.4, 0.8)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2
        expected = 2 * X2 * Y1 * Z1 + 3 * X1 * Y2 * Z1 - X1 * Y1 * Z2 + 4 * X1 * Y1 * Z1
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-12
    end

    @testset "3D linear: trilinear f = xyz + xy + yz + xz" begin
        x = collect(range(0.0, 2.0, length = 11))
        y = collect(range(0.0, 3.0, length = 13))
        z = collect(range(0.0, 1.0, length = 9))
        data = [xi * yj * zk + xi * yj + yj * zk + xi * zk for xi in x, yj in y, zk in z]
        itp = linear_interp((x, y, z), data; extrap = NoExtrap())

        lo, hi = (0.3, 0.5, 0.1), (1.7, 2.4, 0.8)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2
        # ∫∫∫ (xyz + xy + yz + xz) dz dy dx
        expected = X2 * Y2 * Z2 + X2 * Y2 * Z1 + X1 * Y2 * Z2 + X2 * Y1 * Z2
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-11
    end

    @testset "3D linear: full-domain parity" begin
        x = collect(range(0.0, 2.0, length = 11))
        y = collect(range(0.0, 3.0, length = 13))
        z = collect(range(0.0, 1.0, length = 9))
        data = [2xi + 3yj - zk + 4 for xi in x, yj in y, zk in z]
        itp = linear_interp((x, y, z), data; extrap = NoExtrap())

        lo = (first(x), first(y), first(z))
        hi = (last(x), last(y), last(z))
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2
        Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2
        expected = 2 * X2 * Y1 * Z1 + 3 * X1 * Y2 * Z1 - X1 * Y1 * Z2 + 4 * X1 * Y1 * Z1
        @test integrate(itp) ≈ expected atol = 1.0e-12
    end

    @testset "3D quadratic: additive x² + 2y² + 3z²" begin
        x = collect(range(0.0, 2.0, length = 17))
        y = collect(range(0.0, 3.0, length = 15))
        z = collect(range(0.0, 1.0, length = 11))
        data = [xi^2 + 2yj^2 + 3zk^2 for xi in x, yj in y, zk in z]
        itp = quadratic_interp((x, y, z), data; extrap = NoExtrap())

        lo, hi = (0.2, 0.4, 0.1), (1.8, 2.5, 0.9)
        X1 = hi[1] - lo[1];  X3 = (hi[1]^3 - lo[1]^3) / 3
        Y1 = hi[2] - lo[2];  Y3 = (hi[2]^3 - lo[2]^3) / 3
        Z1 = hi[3] - lo[3];  Z3 = (hi[3]^3 - lo[3]^3) / 3
        expected = X3 * Y1 * Z1 + 2 * X1 * Y3 * Z1 + 3 * X1 * Y1 * Z3
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-8
    end

    @testset "3D quadratic: mixed xy + yz + xz + x² + y² + z²" begin
        x = collect(range(0.0, 2.0, length = 17))
        y = collect(range(0.0, 3.0, length = 15))
        z = collect(range(0.0, 1.0, length = 11))
        data = [
            xi * yj + yj * zk + xi * zk + xi^2 + yj^2 + zk^2
                for xi in x, yj in y, zk in z
        ]
        itp = quadratic_interp((x, y, z), data; extrap = NoExtrap())

        lo, hi = (0.2, 0.4, 0.1), (1.8, 2.5, 0.9)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2;  X3 = (hi[1]^3 - lo[1]^3) / 3
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2;  Y3 = (hi[2]^3 - lo[2]^3) / 3
        Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2;  Z3 = (hi[3]^3 - lo[3]^3) / 3
        # ∫∫∫ (xy + yz + xz + x² + y² + z²) dz dy dx
        expected = X2 * Y2 * Z1 + X1 * Y2 * Z2 + X2 * Y1 * Z2 + X3 * Y1 * Z1 + X1 * Y3 * Z1 + X1 * Y1 * Z3
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-8
    end

    @testset "3D quadratic: product x²·y²·z²" begin
        x = collect(range(0.0, 2.0, length = 17))
        y = collect(range(0.0, 3.0, length = 15))
        z = collect(range(0.0, 1.0, length = 11))
        data = [xi^2 * yj^2 * zk^2 for xi in x, yj in y, zk in z]
        itp = quadratic_interp((x, y, z), data; extrap = NoExtrap())

        lo, hi = (0.2, 0.4, 0.1), (1.8, 2.5, 0.9)
        X3 = (hi[1]^3 - lo[1]^3) / 3
        Y3 = (hi[2]^3 - lo[2]^3) / 3
        Z3 = (hi[3]^3 - lo[3]^3) / 3
        expected = X3 * Y3 * Z3
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-6
    end

    @testset "3D cubic: additive x³ + y³ + z³" begin
        x = collect(range(0.0, 2.0, length = 21))
        y = collect(range(0.0, 2.0, length = 21))
        z = collect(range(0.0, 1.0, length = 17))
        data = [xi^3 + yj^3 + zk^3 for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data; bc = CubicFit(), extrap = NoExtrap())

        lo, hi = (0.2, 0.3, 0.1), (1.8, 1.7, 0.9)
        X1 = hi[1] - lo[1];  X4 = (hi[1]^4 - lo[1]^4) / 4
        Y1 = hi[2] - lo[2];  Y4 = (hi[2]^4 - lo[2]^4) / 4
        Z1 = hi[3] - lo[3];  Z4 = (hi[3]^4 - lo[3]^4) / 4
        expected = X4 * Y1 * Z1 + X1 * Y4 * Z1 + X1 * Y1 * Z4
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-10
    end

    @testset "3D cubic: mixed x²y + y²z + z²x" begin
        x = collect(range(0.0, 2.0, length = 21))
        y = collect(range(0.0, 2.0, length = 21))
        z = collect(range(0.0, 1.0, length = 17))
        data = [xi^2 * yj + yj^2 * zk + zk^2 * xi for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data; bc = CubicFit(), extrap = NoExtrap())

        lo, hi = (0.2, 0.3, 0.1), (1.8, 1.7, 0.9)
        X1 = hi[1] - lo[1];  X2 = (hi[1]^2 - lo[1]^2) / 2;  X3 = (hi[1]^3 - lo[1]^3) / 3
        Y1 = hi[2] - lo[2];  Y2 = (hi[2]^2 - lo[2]^2) / 2;  Y3 = (hi[2]^3 - lo[2]^3) / 3
        Z1 = hi[3] - lo[3];  Z2 = (hi[3]^2 - lo[3]^2) / 2;  Z3 = (hi[3]^3 - lo[3]^3) / 3
        # ∫∫∫ (x²y + y²z + z²x) dz dy dx
        expected = X3 * Y2 * Z1 + X1 * Y3 * Z2 + X2 * Y1 * Z3
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-10
    end

    @testset "3D cubic: product x²·y²·z²" begin
        x = collect(range(0.0, 2.0, length = 25))
        y = collect(range(0.0, 2.0, length = 25))
        z = collect(range(0.0, 1.0, length = 17))
        data = [xi^2 * yj^2 * zk^2 for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data; bc = CubicFit(), extrap = NoExtrap())

        lo, hi = (0.2, 0.3, 0.1), (1.8, 1.7, 0.9)
        X3 = (hi[1]^3 - lo[1]^3) / 3
        Y3 = (hi[2]^3 - lo[2]^3) / 3
        Z3 = (hi[3]^3 - lo[3]^3) / 3
        expected = X3 * Y3 * Z3
        @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-10
    end

    # ───────────────────────────────────────────────
    # 3D constant interpolation
    # ───────────────────────────────────────────────

    @testset "3D constant: exact for uniform f = 7" begin
        x = collect(range(0.0, 2.0, length = 11))
        y = collect(range(0.0, 3.0, length = 9))
        z = collect(range(0.0, 1.0, length = 7))
        data = fill(7.0, length(x), length(y), length(z))
        for side in (LeftSide(), RightSide(), NearestSide())
            itp = constant_interp((x, y, z), data; side = side, extrap = NoExtrap())
            lo, hi = (0.3, 0.5, 0.1), (1.7, 2.4, 0.8)
            expected = 7.0 * (hi[1] - lo[1]) * (hi[2] - lo[2]) * (hi[3] - lo[3])
            @test integrate(itp, lo, hi) ≈ expected atol = 1.0e-12
        end
    end

    @testset "3D constant: finite for non-trivial field" begin
        x = collect(range(0.0, 2.0, length = 11))
        y = collect(range(0.0, 3.0, length = 9))
        z = collect(range(0.0, 1.0, length = 7))
        data = [sin(xi) + cos(yj) + zk for xi in x, yj in y, zk in z]
        for side in (LeftSide(), RightSide(), NearestSide())
            itp = constant_interp((x, y, z), data; side = side, extrap = NoExtrap())
            @test isfinite(integrate(itp, (0.3, 0.5, 0.1), (1.7, 2.4, 0.8)))
        end
    end

    # ───────────────────────────────────────────────
    # 3D structural properties (cubic)
    # ───────────────────────────────────────────────

    @testset "3D cubic structural properties" begin
        x = collect(range(0.0, 2.0, length = 17))
        y = collect(range(-1.0, 1.0, length = 15))
        z = collect(range(0.0, 1.5, length = 13))
        data = [sin(xi) * cos(yj) * exp(zk / 2) for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data; extrap = NoExtrap())

        @testset "full-domain parity" begin
            lo = (first(x), first(y), first(z))
            hi = (last(x), last(y), last(z))
            @test integrate(itp) ≈ integrate(itp, lo, hi) atol = 1.0e-10
        end

        @testset "sign rule (antisymmetry)" begin
            lo, hi = (0.3, -0.5, 0.2), (1.5, 0.7, 1.1)
            # flip one axis → negate result
            @test integrate(itp, lo, hi) ≈ -integrate(
                itp, (hi[1], lo[2], lo[3]),
                (lo[1], hi[2], hi[3])
            ) atol = 1.0e-10
            # flip two axes → same sign
            @test integrate(itp, lo, hi) ≈ integrate(
                itp, (hi[1], hi[2], lo[3]),
                (lo[1], lo[2], hi[3])
            ) atol = 1.0e-10
            # flip all three → negate
            @test integrate(itp, lo, hi) ≈ -integrate(itp, hi, lo) atol = 1.0e-10
        end

        @testset "partition additivity (split along x)" begin
            lo, hi = (0.2, -0.4, 0.2), (1.7, 0.8, 1.2)
            xm = 0.9
            lhs = integrate(itp, lo, hi)
            rhs = integrate(itp, lo, (xm, hi[2], hi[3])) +
                integrate(itp, (xm, lo[2], lo[3]), hi)
            @test lhs ≈ rhs atol = 1.0e-10
        end

        @testset "partition additivity (split along y)" begin
            lo, hi = (0.2, -0.4, 0.2), (1.7, 0.8, 1.2)
            ym = 0.2
            lhs = integrate(itp, lo, hi)
            rhs = integrate(itp, lo, (hi[1], ym, hi[3])) +
                integrate(itp, (lo[1], ym, lo[3]), hi)
            @test lhs ≈ rhs atol = 1.0e-10
        end

        @testset "partition additivity (split along z)" begin
            lo, hi = (0.2, -0.4, 0.2), (1.7, 0.8, 1.2)
            zm = 0.7
            lhs = integrate(itp, lo, hi)
            rhs = integrate(itp, lo, (hi[1], hi[2], zm)) +
                integrate(itp, (lo[1], lo[2], zm), hi)
            @test lhs ≈ rhs atol = 1.0e-10
        end

        @testset "domain check" begin
            @test_throws DomainError integrate(itp, (-0.1, -0.5, 0.1), (0.5, 0.2, 0.5))
            @test_throws DomainError integrate(itp, (0.1, -0.5, 0.1), (2.1, 0.2, 0.5))
            @test_throws DomainError integrate(itp, (0.1, -0.5, -0.1), (0.5, 0.2, 0.5))
            @test_throws DomainError integrate(itp, (0.1, -0.5, 0.1), (0.5, 0.2, 1.6))
        end
    end

    # ───────────────────────────────────────────────
    # 3D structural properties (linear)
    # ───────────────────────────────────────────────

    @testset "3D linear structural properties" begin
        x = collect(range(0.0, 2.0, length = 11))
        y = collect(range(0.0, 3.0, length = 13))
        z = collect(range(0.0, 1.0, length = 9))
        data = [sin(xi) * cos(yj) + zk^2 for xi in x, yj in y, zk in z]
        itp = linear_interp((x, y, z), data; extrap = NoExtrap())

        @testset "antisymmetry" begin
            lo, hi = (0.3, 0.5, 0.1), (1.7, 2.4, 0.8)
            @test integrate(itp, lo, hi) ≈ -integrate(itp, hi, lo) atol = 1.0e-10
        end

        @testset "additivity (split along x)" begin
            lo, hi = (0.3, 0.5, 0.1), (1.7, 2.4, 0.8)
            xm = 1.0
            lhs = integrate(itp, lo, hi)
            rhs = integrate(itp, lo, (xm, hi[2], hi[3])) +
                integrate(itp, (xm, lo[2], lo[3]), hi)
            @test lhs ≈ rhs atol = 1.0e-10
        end
    end

    # ───────────────────────────────────────────────
    # 3D structural properties (quadratic)
    # ───────────────────────────────────────────────

    @testset "3D quadratic structural properties" begin
        x = collect(range(0.0, 2.0, length = 17))
        y = collect(range(0.0, 3.0, length = 15))
        z = collect(range(0.0, 1.0, length = 11))
        data = [sin(xi) * cos(yj) + zk^2 for xi in x, yj in y, zk in z]
        itp = quadratic_interp((x, y, z), data; extrap = NoExtrap())

        @testset "antisymmetry" begin
            lo, hi = (0.2, 0.4, 0.1), (1.8, 2.5, 0.9)
            @test integrate(itp, lo, hi) ≈ -integrate(itp, hi, lo) atol = 1.0e-10
        end

        @testset "additivity (split along z)" begin
            lo, hi = (0.2, 0.4, 0.1), (1.8, 2.5, 0.9)
            zm = 0.5
            lhs = integrate(itp, lo, hi)
            rhs = integrate(itp, lo, (hi[1], hi[2], zm)) +
                integrate(itp, (lo[1], lo[2], zm), hi)
            @test lhs ≈ rhs atol = 1.0e-10
        end
    end
end
