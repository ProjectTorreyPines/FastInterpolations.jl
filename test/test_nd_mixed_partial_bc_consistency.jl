# ==============================================================================
# ND Mixed-Partial BC Consistency Tests (regression suite)
# ==============================================================================
# These tests pin the mathematical invariants that the mixed-partial BC fix
# restores. They were authored as the RED phase of the TDD cycle for the fix
# in `_get_effective_bc` / `_get_effective_bc_quadratic`. On the pre-fix commit
# they FAIL by the bug magnitudes documented in
#   claudedocs/TODO/00_HIGHEST_PRIORITY_mixed_partial_bc_fix.md
# (cubic ~1e-10, quadratic ~1e-5). After the fix they pass to machine epsilon.
#
# Three groups, each cubic + quadratic, 2D + 3D where applicable:
#
#   A. PreCompute ↔ OnTheFly bit-equivalence
#      The two strategies should produce identical results for every user BC,
#      because both ultimately apply the user BC to the same composition.
#
#   B. Clairaut / axis-swap symmetry
#      Building on `(xs, ys)` with `data` and on `(ys, xs)` with `permutedims(data)`
#      then querying at `(qx, qy)` vs `(qy, qx)` must give the same value. This
#      catches any "hidden BC switch" that breaks tensor-product symmetry.
#
#   C. Hessian off-diagonal symmetry
#      For PreCompute interpolants (gradient/hessian path), `H[1,2] == H[2,1]`
#      to machine epsilon, even with non-default BCs.
#
#   D. Cubic periodic propagation regression-pin
#      PeriodicBC on one axis must continue to propagate through mixed partials
#      (cubic Rule 2 in `_get_effective_bc`). PeriodicBC is cubic-only.
#
# Tolerance philosophy: assertions use `atol = 50*eps(Float64)` (≈ 1.1e-14),
# which is generous enough to absorb legitimate floating-point reordering in
# tridiagonal/recurrence solves, but ≫ tighter than the bug magnitudes (1e-10
# cubic, 1e-5 quadratic) so the RED→GREEN flip is unambiguous.
# ==============================================================================

using Test
using FastInterpolations

const _MP_ATOL = 50 * eps(Float64)

@testset "ND Mixed-Partial BC Consistency" begin

    # ----- shared fixtures -----
    xs = range(0.0, 2π, 30)
    ys = range(0.0, π, 25)
    zs = range(0.0, 1.0, 12)
    qx, qy, qz = 1.7, 0.8, 0.4

    # Three data functions chosen to exercise different BC mismatch regimes
    # (per the empirical table in the design doc):
    data_2d_sincos = [sin(x) * cos(y) for x in xs, y in ys]            # smooth, periodic-ish
    data_2d_siny3 = [sin(x) * y^3   for x in xs, y in ys]             # large d²f/dy² at boundary
    data_2d_sinlin = [sin(x) * (y - π) for x in xs, y in ys]           # exact zero d²f/dy²
    data_3d_sincos = [sin(x) * cos(y) * z for x in xs, y in ys, z in zs]

    cubic_bcs_2d = (
        ("ZeroCurvBC", ZeroCurvBC()),
        ("ZeroSlopeBC", ZeroSlopeBC()),
        ("CubicFit", CubicFit()),
    )

    quad_bcs_2d = (
        ("ZeroCurvBC", ZeroCurvBC()),
        ("ZeroSlopeBC", ZeroSlopeBC()),
        ("MinCurvFit", MinCurvFit()),
        ("Right(QuadraticFit)", Right(QuadraticFit())),
    )

    data_set_2d = (
        ("sin*cos", data_2d_sincos),
        ("sin*y^3", data_2d_siny3),
        ("sin*(y-π)", data_2d_sinlin),
    )

    # ==========================================================================
    # A. PreCompute ↔ OnTheFly bit-equivalence
    # ==========================================================================

    @testset "A. PreCompute ≡ OnTheFly (cubic 2D)" begin
        for (dname, data) in data_set_2d, (bcname, bc) in cubic_bcs_2d
            v_pre = cubic_interp((xs, ys), data, (qx, qy); bc = bc, coeffs = PreCompute())
            v_otf = cubic_interp((xs, ys), data, (qx, qy); bc = bc, coeffs = OnTheFly())
            @test isapprox(v_pre, v_otf; atol = _MP_ATOL)
        end
    end

    @testset "A. PreCompute ≡ OnTheFly (quadratic 2D)" begin
        for (dname, data) in data_set_2d, (bcname, bc) in quad_bcs_2d
            v_pre = quadratic_interp((xs, ys), data, (qx, qy); bc = bc, coeffs = PreCompute())
            v_otf = quadratic_interp((xs, ys), data, (qx, qy); bc = bc, coeffs = OnTheFly())
            @test isapprox(v_pre, v_otf; atol = _MP_ATOL)
        end
    end

    @testset "A. PreCompute ≡ OnTheFly (cubic 3D, ZeroCurvBC)" begin
        v_pre = cubic_interp((xs, ys, zs), data_3d_sincos, (qx, qy, qz); bc = ZeroCurvBC(), coeffs = PreCompute())
        v_otf = cubic_interp((xs, ys, zs), data_3d_sincos, (qx, qy, qz); bc = ZeroCurvBC(), coeffs = OnTheFly())
        @test isapprox(v_pre, v_otf; atol = _MP_ATOL)
    end

    @testset "A. PreCompute ≡ OnTheFly (quadratic 3D, ZeroCurvBC)" begin
        v_pre = quadratic_interp((xs, ys, zs), data_3d_sincos, (qx, qy, qz); bc = ZeroCurvBC(), coeffs = PreCompute())
        v_otf = quadratic_interp((xs, ys, zs), data_3d_sincos, (qx, qy, qz); bc = ZeroCurvBC(), coeffs = OnTheFly())
        @test isapprox(v_pre, v_otf; atol = _MP_ATOL)
    end

    # ==========================================================================
    # B. Clairaut / axis-swap symmetry
    #
    # If we swap axes — build on (ys, xs) with permutedims(data) and query at
    # (qy, qx) instead of (qx, qy) — the result must be identical. This catches
    # any axis-asymmetric BC override in the mixed-partial build.
    # ==========================================================================

    @testset "B. Axis-swap symmetry (cubic 2D PreCompute)" begin
        for (dname, data) in data_set_2d, (bcname, bc) in cubic_bcs_2d
            v_xy = cubic_interp((xs, ys), data, (qx, qy); bc = bc, coeffs = PreCompute())
            v_yx = cubic_interp((ys, xs), permutedims(data), (qy, qx); bc = bc, coeffs = PreCompute())
            @test isapprox(v_xy, v_yx; atol = _MP_ATOL)
        end
    end

    @testset "B. Axis-swap symmetry (quadratic 2D PreCompute)" begin
        for (dname, data) in data_set_2d, (bcname, bc) in quad_bcs_2d
            v_xy = quadratic_interp((xs, ys), data, (qx, qy); bc = bc, coeffs = PreCompute())
            v_yx = quadratic_interp((ys, xs), permutedims(data), (qy, qx); bc = bc, coeffs = PreCompute())
            @test isapprox(v_xy, v_yx; atol = _MP_ATOL)
        end
    end

    @testset "B. Axis-swap symmetry (cubic 3D PreCompute)" begin
        # 3D permutation: (x,y,z) → (z,y,x)
        data_perm = permutedims(data_3d_sincos, (3, 2, 1))
        v_xyz = cubic_interp((xs, ys, zs), data_3d_sincos, (qx, qy, qz); bc = ZeroCurvBC(), coeffs = PreCompute())
        v_zyx = cubic_interp((zs, ys, xs), data_perm, (qz, qy, qx); bc = ZeroCurvBC(), coeffs = PreCompute())
        @test isapprox(v_xyz, v_zyx; atol = _MP_ATOL)
    end

    @testset "B. Axis-swap symmetry (quadratic 3D PreCompute)" begin
        data_perm = permutedims(data_3d_sincos, (3, 2, 1))
        v_xyz = quadratic_interp((xs, ys, zs), data_3d_sincos, (qx, qy, qz); bc = ZeroCurvBC(), coeffs = PreCompute())
        v_zyx = quadratic_interp((zs, ys, xs), data_perm, (qz, qy, qx); bc = ZeroCurvBC(), coeffs = PreCompute())
        @test isapprox(v_xyz, v_zyx; atol = _MP_ATOL)
    end

    # ==========================================================================
    # C. Hessian off-diagonal symmetry
    #
    # For an interpolant built with non-default BC, hessian(itp, q)[1,2] must
    # equal hessian(itp, q)[2,1] to machine epsilon. The current code violates
    # this for quadratic by ~1e-5 because the mixed partial uses the wrong BC.
    # ==========================================================================

    @testset "C. Hessian symmetry (cubic 2D, non-default BC)" begin
        for (dname, data) in data_set_2d, (bcname, bc) in cubic_bcs_2d
            itp = interp((xs, ys), data; method = CubicInterp(bc = bc), coeffs = PreCompute())
            H = hessian(itp, (qx, qy))
            @test isapprox(H[1, 2], H[2, 1]; atol = _MP_ATOL)
        end
    end

    @testset "C. Hessian symmetry (quadratic 2D, non-default BC)" begin
        for (dname, data) in data_set_2d, (bcname, bc) in quad_bcs_2d
            itp = interp((xs, ys), data; method = QuadraticInterp(bc = bc), coeffs = PreCompute())
            H = hessian(itp, (qx, qy))
            @test isapprox(H[1, 2], H[2, 1]; atol = _MP_ATOL)
        end
    end

    # ==========================================================================
    # D. Cubic periodic propagation regression-pin
    #
    # PeriodicBC on one axis must propagate through mixed partials. Cubic Rule 2
    # in `_get_effective_bc` already handles this; the test pins the behavior so
    # the upcoming fix does not regress it. PeriodicBC is cubic-only.
    # ==========================================================================

    # ==========================================================================
    # E. Short-grid fallback emits an informative one-shot warning
    #
    # The mixed-partial helpers fall back to a default BC when the grid is too
    # short to safely apply the user's non-PolyFit BC to a differentiated nodal
    # array. The fallback is silent in the original code; the fix adds a
    # `@warn ... maxlog=1` so users notice. We test the helpers directly
    # because the public ND build pipeline may reject very short grids before
    # the warning would fire.
    # ==========================================================================

    @testset "E. Short-grid fallback warning (cubic)" begin
        # 3-point grid + ZeroSlopeBC + p_src > 1 → warning + ZeroCurvBC fallback
        short_grid = [0.0, 0.5, 1.0]
        @test_logs (:warn, r"Cubic ND mixed-partial build") begin
            FastInterpolations._get_effective_bc(ZeroSlopeBC(), 2, short_grid)
        end
        result = FastInterpolations._get_effective_bc(ZeroSlopeBC(), 2, short_grid)
        @test result === ZeroCurvBC()
    end

    @testset "E. Short-grid fallback warning (quadratic)" begin
        # 2-point grid + ZeroSlopeBC + p_src > 1 → warning + MinCurvFit fallback
        short_grid = [0.0, 1.0]
        @test_logs (:warn, r"Quadratic ND mixed-partial build") begin
            FastInterpolations._get_effective_bc_quadratic(ZeroSlopeBC(), 2, short_grid)
        end
        result = FastInterpolations._get_effective_bc_quadratic(ZeroSlopeBC(), 2, short_grid)
        @test result === MinCurvFit()
    end

    @testset "D. Cubic periodic-on-one-axis propagation" begin
        xp = range(0.0, 2π, 30)
        yp = range(0.0, π, 25)
        # data periodic in x (axis 1)
        data_p = [sin(x) * cos(y) for x in xp, y in yp]
        data_p[end, :] .= data_p[1, :]  # ensure exact periodicity at endpoints

        v_pre = cubic_interp(
            (xp, yp), data_p, (1.7, 0.8);
            bc = (PeriodicBC(), ZeroCurvBC()),
            extrap = (WrapExtrap(), NoExtrap()),
            coeffs = PreCompute(),
        )
        v_otf = cubic_interp(
            (xp, yp), data_p, (1.7, 0.8);
            bc = (PeriodicBC(), ZeroCurvBC()),
            extrap = (WrapExtrap(), NoExtrap()),
            coeffs = OnTheFly(),
        )
        @test isapprox(v_pre, v_otf; atol = _MP_ATOL)
    end

end
