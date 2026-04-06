# ========================================
# ND OnTheFly One-Shot + AutoCoeffs Tests
# ========================================
# Validates:
# A. OnTheFly ≈ PreCompute numerical equivalence (scalar + derivatives + PeriodicBC)
# B. AutoCoeffs default behavior (scalar → OnTheFly, batch → PreCompute)
# C. Zero-allocation after warmup (function barrier pattern)

using Test
using FastInterpolations

const AAP_RUNTIME_CHECK_LOCAL = FastInterpolations.AdaptiveArrayPools.RUNTIME_CHECK
const ND_ALLOC_THRESHOLD_LOCAL = VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK_LOCAL + 1) * 240

@testset "ND OnTheFly One-Shot + AutoCoeffs" begin
    # ========================================
    # Test Data Setup
    # ========================================
    x = range(0.0, 2π, 30)
    y = range(0.0, π, 25)
    z = range(0.0, 1.0, 10)
    data_2d = [sin(xi) * cos(yj) for xi in x, yj in y]
    data_3d = [sin(xi) * cos(yj) * zk for xi in x, yj in y, zk in z]
    qx, qy, qz = 1.7, 0.8, 0.4

    # ========================================
    # A. Numerical Equivalence
    # ========================================

    @testset "Equivalence: cubic 2D scalar" begin
        val_otf = cubic_interp((x, y), data_2d, (qx, qy); coeffs = OnTheFly())
        val_pre = cubic_interp((x, y), data_2d, (qx, qy); coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: cubic 3D scalar" begin
        val_otf = cubic_interp((x, y, z), data_3d, (qx, qy, qz); coeffs = OnTheFly())
        val_pre = cubic_interp((x, y, z), data_3d, (qx, qy, qz); coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: quadratic 2D scalar" begin
        val_otf = quadratic_interp((x, y), data_2d, (qx, qy); coeffs = OnTheFly())
        val_pre = quadratic_interp((x, y), data_2d, (qx, qy); coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: interp hetero Cubic×Linear" begin
        val_otf = interp((x, y), data_2d, (qx, qy); method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly())
        val_pre = interp((x, y), data_2d, (qx, qy); method = (CubicInterp(), LinearInterp()), coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: interp hetero Cubic×Quadratic" begin
        val_otf = interp((x, y), data_2d, (qx, qy); method = (CubicInterp(), QuadraticInterp()), coeffs = OnTheFly())
        val_pre = interp((x, y), data_2d, (qx, qy); method = (CubicInterp(), QuadraticInterp()), coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: derivatives ∂f/∂x, ∂f/∂y" begin
        d = (DerivOp(1), DerivOp(0))
        val_otf = cubic_interp((x, y), data_2d, (qx, qy); deriv = d, coeffs = OnTheFly())
        val_pre = cubic_interp((x, y), data_2d, (qx, qy); deriv = d, coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10

        d2 = (DerivOp(0), DerivOp(1))
        val_otf2 = cubic_interp((x, y), data_2d, (qx, qy); deriv = d2, coeffs = OnTheFly())
        val_pre2 = cubic_interp((x, y), data_2d, (qx, qy); deriv = d2, coeffs = PreCompute())
        @test val_otf2 ≈ val_pre2 rtol = 1.0e-10
    end

    @testset "Equivalence: PeriodicBC on one axis" begin
        data_p = [sin(xi) * cos(yj) for xi in x, yj in y]
        data_p[:, end] .= data_p[:, 1]  # ensure periodicity on y-axis
        val_otf = cubic_interp(
            (x, y), data_p, (qx, qy);
            bc = (CubicFit(), PeriodicBC()), extrap = (NoExtrap(), WrapExtrap()), coeffs = OnTheFly()
        )
        val_pre = cubic_interp(
            (x, y), data_p, (qx, qy);
            bc = (CubicFit(), PeriodicBC()), extrap = (NoExtrap(), WrapExtrap()), coeffs = PreCompute()
        )
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: PeriodicBC on both axes" begin
        xp = range(0.0, 2π, 30)
        yp = range(0.0, 2π, 25)
        data_pp = [sin(xi) * sin(yj) for xi in xp, yj in yp]
        data_pp[end, :] .= data_pp[1, :]
        data_pp[:, end] .= data_pp[:, 1]
        data_pp[end, end] = data_pp[1, 1]
        val_otf = cubic_interp(
            (xp, yp), data_pp, (1.5, 2.0);
            bc = PeriodicBC(), extrap = WrapExtrap(), coeffs = OnTheFly()
        )
        val_pre = cubic_interp(
            (xp, yp), data_pp, (1.5, 2.0);
            bc = PeriodicBC(), extrap = WrapExtrap(), coeffs = PreCompute()
        )
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: mixed extrap (Clamp + NoExtrap)" begin
        val_otf = cubic_interp(
            (x, y), data_2d, (qx, qy);
            extrap = (ClampExtrap(), NoExtrap()), coeffs = OnTheFly()
        )
        val_pre = cubic_interp(
            (x, y), data_2d, (qx, qy);
            extrap = (ClampExtrap(), NoExtrap()), coeffs = PreCompute()
        )
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    # ========================================
    # B. AutoCoeffs Default Behavior
    # ========================================

    @testset "AutoCoeffs: cubic scalar matches PreCompute" begin
        val_auto = cubic_interp((x, y), data_2d, (qx, qy))  # AutoCoeffs default
        val_pre = cubic_interp((x, y), data_2d, (qx, qy); coeffs = PreCompute())
        @test val_auto ≈ val_pre rtol = 1.0e-10
    end

    @testset "AutoCoeffs: quadratic scalar matches PreCompute" begin
        val_auto = quadratic_interp((x, y), data_2d, (qx, qy))  # AutoCoeffs default
        val_pre = quadratic_interp((x, y), data_2d, (qx, qy); coeffs = PreCompute())
        @test val_auto ≈ val_pre rtol = 1.0e-10
    end

    @testset "AutoCoeffs: interp scalar matches PreCompute" begin
        val_auto = interp((x, y), data_2d, (qx, qy); method = CubicInterp())  # AutoCoeffs default
        val_pre = interp((x, y), data_2d, (qx, qy); method = CubicInterp(), coeffs = PreCompute())
        @test val_auto ≈ val_pre rtol = 1.0e-10
    end

    @testset "AutoCoeffs: batch still uses PreCompute path" begin
        queries = [(1.0, 0.5), (2.0, 1.0), (3.0, 1.5)]
        val_auto = interp((x, y), data_2d, queries; method = CubicInterp())
        val_pre = interp((x, y), data_2d, queries; method = CubicInterp(), coeffs = PreCompute())
        @test val_auto ≈ val_pre rtol = 1.0e-14
    end

    # ========================================
    # C. Zero-Allocation Tests
    # ========================================

    @testset "Zero-alloc: OnTheFly oneshot cubic 2D" begin
        function _alloc_test_otf_cubic_2d()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            q = (1.7, 0.8)
            cubic_interp((xg, yg), d, q; coeffs = OnTheFly())
            cubic_interp((xg, yg), d, q; coeffs = OnTheFly())
            return @allocated cubic_interp((xg, yg), d, q; coeffs = OnTheFly())
        end
        @test _alloc_test_otf_cubic_2d() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: OnTheFly oneshot cubic 3D" begin
        function _alloc_test_otf_cubic_3d()
            xg = collect(range(0.0, 2π, 15))
            yg = collect(range(0.0, π, 12))
            zg = collect(range(0.0, 1.0, 8))
            d = [sin(xi) * cos(yj) * zk for xi in xg, yj in yg, zk in zg]
            q = (1.7, 0.8, 0.4)
            cubic_interp((xg, yg, zg), d, q; coeffs = OnTheFly())
            cubic_interp((xg, yg, zg), d, q; coeffs = OnTheFly())
            return @allocated cubic_interp((xg, yg, zg), d, q; coeffs = OnTheFly())
        end
        @test _alloc_test_otf_cubic_3d() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: OnTheFly oneshot quadratic 2D" begin
        function _alloc_test_otf_quad_2d()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            q = (1.7, 0.8)
            quadratic_interp((xg, yg), d, q; coeffs = OnTheFly())
            quadratic_interp((xg, yg), d, q; coeffs = OnTheFly())
            return @allocated quadratic_interp((xg, yg), d, q; coeffs = OnTheFly())
        end
        @test _alloc_test_otf_quad_2d() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: OnTheFly oneshot interp hetero" begin
        function _alloc_test_otf_hetero()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            q = (1.7, 0.8)
            m = (CubicInterp(), LinearInterp())
            interp((xg, yg), d, q; method = m, coeffs = OnTheFly())
            interp((xg, yg), d, q; method = m, coeffs = OnTheFly())
            return @allocated interp((xg, yg), d, q; method = m, coeffs = OnTheFly())
        end
        @test _alloc_test_otf_hetero() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: AutoCoeffs default cubic scalar" begin
        function _alloc_test_auto_cubic()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            q = (1.7, 0.8)
            cubic_interp((xg, yg), d, q)
            cubic_interp((xg, yg), d, q)
            return @allocated cubic_interp((xg, yg), d, q)
        end
        @test _alloc_test_auto_cubic() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: OnTheFly interpolant eval" begin
        function _alloc_test_otf_itp_eval()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp((xg, yg), d; method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly())
            itp((1.7, 0.8))
            itp((1.7, 0.8))
            return @allocated itp((1.7, 0.8))
        end
        @test _alloc_test_otf_itp_eval() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: OnTheFly gradient" begin
        function _alloc_test_otf_gradient()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            itp = interp((xg, yg), d; method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly())
            gradient(itp, (1.7, 0.8))
            gradient(itp, (1.7, 0.8))
            return @allocated gradient(itp, (1.7, 0.8))
        end
        @test _alloc_test_otf_gradient() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    @testset "Zero-alloc: OnTheFly PeriodicBC oneshot" begin
        function _alloc_test_otf_periodic()
            xg = collect(range(0.0, 2π, 30))
            yg = collect(range(0.0, π, 25))
            d = [sin(xi) * cos(yj) for xi in xg, yj in yg]
            d[:, end] .= d[:, 1]
            q = (1.7, 0.8)
            cubic_interp(
                (xg, yg), d, q;
                bc = (CubicFit(), PeriodicBC()), extrap = (NoExtrap(), WrapExtrap()), coeffs = OnTheFly()
            )
            cubic_interp(
                (xg, yg), d, q;
                bc = (CubicFit(), PeriodicBC()), extrap = (NoExtrap(), WrapExtrap()), coeffs = OnTheFly()
            )
            return @allocated cubic_interp(
                (xg, yg), d, q;
                bc = (CubicFit(), PeriodicBC()), extrap = (NoExtrap(), WrapExtrap()), coeffs = OnTheFly()
            )
        end
        @test _alloc_test_otf_periodic() <= ND_ALLOC_THRESHOLD_LOCAL
    end

    # ========================================
    # D. Additional Coverage (review feedback)
    # ========================================

    @testset "Equivalence: quadratic derivatives via OnTheFly" begin
        d = (DerivOp(1), DerivOp(0))
        val_otf = quadratic_interp((x, y), data_2d, (qx, qy); deriv = d, coeffs = OnTheFly())
        val_pre = quadratic_interp((x, y), data_2d, (qx, qy); deriv = d, coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0e-10
    end

    @testset "Equivalence: Float32 data" begin
        x32 = range(0.0f0, Float32(2π), 30)
        y32 = range(0.0f0, Float32(π), 25)
        d32 = [sin(xi) * cos(yj) for xi in x32, yj in y32]
        q32 = (1.7f0, 0.8f0)
        val_otf = cubic_interp((x32, y32), d32, q32; coeffs = OnTheFly())
        val_pre = cubic_interp((x32, y32), d32, q32; coeffs = PreCompute())
        @test val_otf ≈ val_pre rtol = 1.0f-5
    end

    @testset "Equivalence: explicit PreCompute roundtrip" begin
        val_pre = cubic_interp((x, y), data_2d, (qx, qy); coeffs = PreCompute())
        val_interp = interp((x, y), data_2d, (qx, qy); method = CubicInterp(), coeffs = PreCompute())
        @test val_pre ≈ val_interp rtol = 1.0e-14
    end

    @testset "FillExtrap short-circuit with OnTheFly" begin
        val = cubic_interp(
            (x, y), data_2d, (-999.0, 0.5);
            extrap = FillExtrap(NaN), coeffs = OnTheFly()
        )
        @test isnan(val)
    end

    @testset "Equivalence: Hermite family via interp OnTheFly" begin
        val_otf = interp(
            (x, y), data_2d, (qx, qy);
            method = (PchipInterp(), PchipInterp()), coeffs = OnTheFly()
        )
        val_pre_itp = interp((x, y), data_2d; method = PchipInterp(), coeffs = OnTheFly())
        @test val_otf ≈ val_pre_itp((qx, qy)) rtol = 1.0e-12
    end
end
