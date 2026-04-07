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

    # ========================================
    # E. Regression Fixes (codex review)
    # ========================================

    @testset "AD: ForwardDiff.Dual scalar query (native OnTheFly support)" begin
        # OnTheFly now natively supports Dual queries: the _collapse_dims pool buffer
        # type is promoted via `_promote_query_eltype(Tv, q_eval)` at each entry point,
        # so query-dependent intermediates can be Dual-typed. No PreCompute fallback.
        import ForwardDiff

        # 1. Default path (AutoCoeffs → OnTheFly for scalar)
        g_cubic = ForwardDiff.gradient(v -> cubic_interp((x, y), data_2d, (v[1], v[2])), [qx, qy])
        @test length(g_cubic) == 2
        @test all(isfinite, g_cubic)

        g_quad = ForwardDiff.gradient(v -> quadratic_interp((x, y), data_2d, (v[1], v[2])), [qx, qy])
        @test length(g_quad) == 2
        @test all(isfinite, g_quad)

        g_interp = ForwardDiff.gradient(
            v -> interp((x, y), data_2d, (v[1], v[2]); method = CubicInterp()),
            [qx, qy]
        )
        @test length(g_interp) == 2
        @test all(isfinite, g_interp)

        # 2. Explicit coeffs=OnTheFly() must also work (was broken before the _promote_query_eltype fix)
        g_cubic_otf = ForwardDiff.gradient(
            v -> cubic_interp((x, y), data_2d, (v[1], v[2]); coeffs = OnTheFly()), [qx, qy]
        )
        @test g_cubic_otf ≈ g_cubic rtol = 1.0e-12

        g_quad_otf = ForwardDiff.gradient(
            v -> quadratic_interp((x, y), data_2d, (v[1], v[2]); coeffs = OnTheFly()), [qx, qy]
        )
        @test g_quad_otf ≈ g_quad rtol = 1.0e-12

        # 3. Hetero oneshot with mixed methods
        g_hetero = ForwardDiff.gradient(
            v -> interp(
                (x, y), data_2d, (v[1], v[2]);
                method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly()
            ),
            [qx, qy]
        )
        @test length(g_hetero) == 2
        @test all(isfinite, g_hetero)

        # 4. OnTheFly interpolant callable path (via gradient over query)
        itp_otf = interp((x, y), data_2d; method = (CubicInterp(), LinearInterp()), coeffs = OnTheFly())
        g_itp_cbl = ForwardDiff.gradient(v -> itp_otf((v[1], v[2])), [qx, qy])
        @test length(g_itp_cbl) == 2
        @test all(isfinite, g_itp_cbl)
        @test g_itp_cbl ≈ g_hetero rtol = 1.0e-12

        # 5. Accuracy check: cubic gradient vs analytical for sin(x)*cos(y)
        # ∂f/∂x = cos(x)*cos(y), ∂f/∂y = -sin(x)*sin(y)
        analytical = [cos(qx) * cos(qy), -sin(qx) * sin(qy)]
        @test g_cubic ≈ analytical rtol = 1.0e-4  # cubic spline accuracy on 30x25 grid
    end

    @testset "Validation: PreCompute + local Hermite rejected in oneshot" begin
        # Scalar oneshot with PreCompute + PchipInterp should raise ArgumentError
        # (not a MethodError from reaching _compute_nd_partials_hetero!).
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, qy);
            method = (PchipInterp(), AkimaInterp()), coeffs = PreCompute()
        )
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, qy);
            method = PchipInterp(), coeffs = PreCompute()
        )
        # Hetero mixing global + local with PreCompute also rejected
        @test_throws ArgumentError interp(
            (x, y), data_2d, (qx, qy);
            method = (CubicInterp(), PchipInterp()), coeffs = PreCompute()
        )
    end

    @testset "ND batch one-shot with local Hermite (AutoCoeffs → OnTheFly loop)" begin
        # `_resolve_coeffs_nd_oneshot` returns OnTheFly for batch + local Hermite
        # because PreCompute backend is not implemented for these methods. The
        # heterogeneous batch dispatch loops `_interp_nd_oneshot_onthefly` per query
        # — verify the result matches scalar one-shot per query (the ground truth).
        qs = [(0.4, 0.7), (1.5, 1.2), (2.9, 0.3)]
        nq = length(qs)
        out = zeros(nq)

        # Homogeneous Pchip
        ref_pchip = [interp((x, y), data_2d, q; method = (PchipInterp(), PchipInterp())) for q in qs]
        interp!(out, (x, y), data_2d, qs; method = (PchipInterp(), PchipInterp()))
        @test out ≈ ref_pchip rtol = 1.0e-12

        # Explicit OnTheFly should give the same answer
        fill!(out, 0.0)
        interp!(out, (x, y), data_2d, qs; method = (PchipInterp(), PchipInterp()), coeffs = OnTheFly())
        @test out ≈ ref_pchip rtol = 1.0e-12

        # Homogeneous Akima
        ref_akima = [interp((x, y), data_2d, q; method = (AkimaInterp(), AkimaInterp())) for q in qs]
        fill!(out, 0.0)
        interp!(out, (x, y), data_2d, qs; method = (AkimaInterp(), AkimaInterp()))
        @test out ≈ ref_akima rtol = 1.0e-12

        # Heterogeneous: global + local
        ref_mixed = [interp((x, y), data_2d, q; method = (CubicInterp(), PchipInterp())) for q in qs]
        fill!(out, 0.0)
        interp!(out, (x, y), data_2d, qs; method = (CubicInterp(), PchipInterp()))
        @test out ≈ ref_mixed rtol = 1.0e-12

        # Heterogeneous: local + trivial
        ref_pl = [interp((x, y), data_2d, q; method = (PchipInterp(), LinearInterp())) for q in qs]
        fill!(out, 0.0)
        interp!(out, (x, y), data_2d, qs; method = (PchipInterp(), LinearInterp()))
        @test out ≈ ref_pl rtol = 1.0e-12

        # Explicit `coeffs = PreCompute()` with local Hermite in batch must reject
        # (mirrors the scalar validation; user-supplied intent is honored as error).
        @test_throws ArgumentError interp!(
            out, (x, y), data_2d, qs;
            method = (PchipInterp(), PchipInterp()), coeffs = PreCompute()
        )
        @test_throws ArgumentError interp!(
            out, (x, y), data_2d, qs;
            method = (CubicInterp(), PchipInterp()), coeffs = PreCompute()
        )
    end

    # ========================================
    # F. Coverage Gap Closure (from re-audit)
    # ========================================

    @testset "Quadratic OnTheFly with all supported BCs" begin
        # Exercise _to_quadratic_bc for each BC variant.
        #
        # NOTE: PreCompute and OnTheFly use different effective BCs for MIXED
        # partials (`d²f/dxdy`). The PreCompute path, in
        # `_get_effective_bc_quadratic` (quadratic_nd_build.jl), forces
        # `Right(QuadraticFit())` whenever `p_src > 1` (i.e., computing a
        # cross-derivative from an already-differentiated array). OnTheFly,
        # being a sequential 1D composition, cannot replicate that switch
        # and applies the user's BC uniformly at every 1D step.
        #
        # Both paths are self-consistent and converge at the expected order,
        # but for BCs that don't match the data's actual boundary behavior
        # (e.g., ZeroCurvBC forcing d²/dy²=0 on smooth non-zero-curvature
        # data) the two paths diverge at ~1e-6 relative. For BCs that are a
        # good fit for the data, the paths give bit-identical results.
        # See brainstorm discussion in plans/keen-jumping-dijkstra.md.
        for bc in (Left(QuadraticFit()), Right(QuadraticFit()), MinCurvFit(), ZeroCurvBC(), ZeroSlopeBC())
            val_otf = quadratic_interp((x, y), data_2d, (qx, qy); bc = bc, coeffs = OnTheFly())
            val_pre = quadratic_interp((x, y), data_2d, (qx, qy); bc = bc, coeffs = PreCompute())
            @test val_otf ≈ val_pre rtol = 1.0e-5
        end
    end

    @testset "Quadratic BC-exact data: OnTheFly ≡ PreCompute bit-identical" begin
        # Proof that OnTheFly/PreCompute divergence is purely due to BC-data
        # mismatch (PreCompute's mixed-partial BC switch), not a numerical bug.
        # When the BC exactly matches the data's boundary behavior, both paths
        # produce bit-identical results even for ZeroCurvBC.
        # f(x,y) = sin(x)*(y - π) has d²f/dy² = 0 everywhere → ZeroCurvBC exact.
        data_exact = [sin(xi) * (yj - π) for xi in x, yj in y]
        for bc in (Left(QuadraticFit()), ZeroCurvBC(), MinCurvFit())
            val_otf = quadratic_interp((x, y), data_exact, (qx, qy); bc = bc, coeffs = OnTheFly())
            val_pre = quadratic_interp((x, y), data_exact, (qx, qy); bc = bc, coeffs = PreCompute())
            @test val_otf == val_pre  # bit-identical
        end
    end

    @testset "Error: _to_quadratic_bc fallback on unsupported BC" begin
        # The @noinline ArgumentError fallback must trigger for BCs that aren't
        # in the QuadraticBC family (e.g., BCPair with a non-quadratic combination).
        # Use PeriodicBC which is not a QuadraticBC.
        @test_throws ArgumentError quadratic_interp(
            (x, y), data_2d, (qx, qy); bc = PeriodicBC(), coeffs = OnTheFly()
        )
    end

    @testset "Vector calculus: hessian + laplacian with OnTheFly interpolant" begin
        # Only gradient was covered in C6; add hessian and laplacian.
        itp_otf = interp((x, y), data_2d; method = CubicInterp(), coeffs = OnTheFly())
        itp_pre = interp((x, y), data_2d; method = CubicInterp(), coeffs = PreCompute())
        @test hessian(itp_otf, (qx, qy)) ≈ hessian(itp_pre, (qx, qy)) rtol = 1.0e-10
        @test laplacian(itp_otf, (qx, qy)) ≈ laplacian(itp_pre, (qx, qy)) rtol = 1.0e-10
    end

    @testset "Trivial methods (Linear/Constant) → PreCompute path" begin
        # _resolve_coeffs_nd_oneshot with all-trivial methods must return PreCompute
        # (OnTheFly not beneficial for methods without global solve).
        # Result should match whichever code path they take.
        val_linear = interp((x, y), data_2d, (qx, qy); method = LinearInterp())
        val_linear_ref = linear_interp((x, y), data_2d, (qx, qy))
        @test val_linear ≈ val_linear_ref rtol = 1.0e-14

        val_const = interp((x, y), data_2d, (qx, qy); method = ConstantInterp())
        val_const_ref = constant_interp((x, y), data_2d, (qx, qy))
        @test val_const ≈ val_const_ref rtol = 1.0e-14
    end

    @testset "Zero-alloc: OnTheFly 3D interpolant callable" begin
        # Earlier zero-alloc tests covered 2D interpolant callable; add 3D.
        function _alloc_test_otf_itp_eval_3d()
            xg = collect(range(0.0, 2π, 15))
            yg = collect(range(0.0, π, 12))
            zg = collect(range(0.0, 1.0, 8))
            d = [sin(xi) * cos(yj) * zk for xi in xg, yj in yg, zk in zg]
            itp = interp(
                (xg, yg, zg), d;
                method = (CubicInterp(), LinearInterp(), QuadraticInterp()), coeffs = OnTheFly()
            )
            itp((1.7, 0.8, 0.4))
            itp((1.7, 0.8, 0.4))
            return @allocated itp((1.7, 0.8, 0.4))
        end
        @test _alloc_test_otf_itp_eval_3d() <= ND_ALLOC_THRESHOLD_LOCAL
    end
end
