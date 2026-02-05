# ========================================
# Heterogeneous Grid + Mixed Derivative Tests
# ========================================
#
# Regression tests for the NTuple type constraint bug:
#   - Heterogeneous grids (Range + Vector) create Tuple{ScalarSpacing, VectorSpacing}
#     which doesn't match NTuple{N, <:AbstractGridSpacing} (requires same concrete type)
#   - Mixed derivatives (e.g., deriv=(2,0)) create Tuple{EvalDeriv2, EvalValue}
#     which doesn't match NTuple{N, <:AbstractEvalOp}
#
# These tests cover the specific combination that was previously untested and
# caused MethodErrors in _compute_all_local_params and N=2 specialization dispatch.

using Test
using FastInterpolations

# ========================================
# 2D CUBIC: HETEROGENEOUS GRID + MIXED DERIVATIVES (ANALYTIC)
# ========================================

@testset "Heterogeneous Grid + Mixed Derivative Regression" begin

    @testset "2D Cubic: Range × Vector with CubicFit — polynomial analytic" begin
        # f(x,y) = x³ + x²y + xy² + y³
        # Tensor-product cubic spline with CubicFit BC should reproduce this exactly
        # since each variable's degree ≤ 3.
        f(x, y)       = x^3 + x^2*y + x*y^2 + y^3
        df_dx(x, y)   = 3x^2 + 2x*y + y^2
        df_dy(x, y)   = x^2 + 2x*y + 3y^2
        d2f_dx2(x, y) = 6x + 2y
        d2f_dy2(x, y) = 2x + 6y
        d2f_dxdy(x, y) = 2x + 2y
        d3f_dx3(x, y) = 6.0
        d3f_dy3(x, y) = 6.0

        # Heterogeneous grids: Range (uniform) × Vector (non-uniform)
        x_range = range(0.0, 2.0, 21)              # ScalarSpacing
        y_vec   = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]  # VectorSpacing
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        itp = cubic_interp((x_range, y_vec), data; bc=CubicFit())

        # Query at interior point (away from boundaries for best accuracy)
        xq, yq = 1.0, 0.5

        # Value
        @test itp((xq, yq); deriv=(0, 0)) ≈ f(xq, yq) atol=1e-10

        # First derivatives (mixed derivative orders: one axis differentiated, one not)
        @test itp((xq, yq); deriv=(1, 0)) ≈ df_dx(xq, yq)   atol=1e-8
        @test itp((xq, yq); deriv=(0, 1)) ≈ df_dy(xq, yq)   atol=1e-6

        # Second derivatives — these are the cases that triggered the original bug
        @test itp((xq, yq); deriv=(2, 0)) ≈ d2f_dx2(xq, yq) atol=1e-6
        @test itp((xq, yq); deriv=(0, 2)) ≈ d2f_dy2(xq, yq) atol=1e-4
        @test itp((xq, yq); deriv=(1, 1)) ≈ d2f_dxdy(xq, yq) atol=1e-6

        # Third derivatives
        @test itp((xq, yq); deriv=(3, 0)) ≈ d3f_dx3(xq, yq) atol=1e-4
        @test itp((xq, yq); deriv=(0, 3)) ≈ d3f_dy3(xq, yq) atol=1e-2
    end

    @testset "2D Cubic: Vector × Range with CubicFit — reversed heterogeneous" begin
        # Same polynomial, but grids swapped: Vector x (non-uniform) × Range y (uniform)
        f(x, y)       = x^3 + x^2*y + x*y^2 + y^3
        df_dx(x, y)   = 3x^2 + 2x*y + y^2
        df_dy(x, y)   = x^2 + 2x*y + 3y^2
        d2f_dx2(x, y) = 6x + 2y
        d2f_dy2(x, y) = 2x + 6y
        d2f_dxdy(x, y) = 2x + 2y

        x_vec   = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.3, 1.6, 2.0]  # VectorSpacing
        y_range = range(0.0, 1.0, 15)                                 # ScalarSpacing
        data = [f(xi, yj) for xi in x_vec, yj in y_range]

        itp = cubic_interp((x_vec, y_range), data; bc=CubicFit())
        xq, yq = 0.8, 0.6

        @test itp((xq, yq); deriv=(0, 0)) ≈ f(xq, yq)         atol=1e-10
        @test itp((xq, yq); deriv=(1, 0)) ≈ df_dx(xq, yq)     atol=1e-6
        @test itp((xq, yq); deriv=(0, 1)) ≈ df_dy(xq, yq)     atol=1e-8
        @test itp((xq, yq); deriv=(2, 0)) ≈ d2f_dx2(xq, yq)   atol=1e-4
        @test itp((xq, yq); deriv=(0, 2)) ≈ d2f_dy2(xq, yq)   atol=1e-6
        @test itp((xq, yq); deriv=(1, 1)) ≈ d2f_dxdy(xq, yq)  atol=1e-6
    end

    @testset "2D Cubic: Vector × Vector (both non-uniform)" begin
        # Both axes non-uniform — also uses VectorSpacing for both,
        # but tests that non-NTuple paths work even with same spacing type
        f(x, y)       = x^2 * y + x * y^2
        df_dx(x, y)   = 2x*y + y^2
        df_dy(x, y)   = x^2 + 2x*y
        d2f_dxdy(x, y) = 2x + 2y

        x_vec = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]
        y_vec = [0.0, 0.2, 0.5, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_vec, yj in y_vec]

        itp = cubic_interp((x_vec, y_vec), data; bc=CubicFit())
        xq, yq = 0.8, 0.6

        @test itp((xq, yq); deriv=(1, 0)) ≈ df_dx(xq, yq)     atol=1e-6
        @test itp((xq, yq); deriv=(0, 1)) ≈ df_dy(xq, yq)     atol=1e-6
        @test itp((xq, yq); deriv=(2, 0)) ≈ 2yq                atol=1e-4
        @test itp((xq, yq); deriv=(0, 2)) ≈ 2xq                atol=1e-4
        @test itp((xq, yq); deriv=(1, 1)) ≈ d2f_dxdy(xq, yq)  atol=1e-4
    end

    # ========================================
    # 3D CUBIC: HETEROGENEOUS GRID + MIXED DERIVATIVES
    # ========================================

    @testset "3D Cubic: Range × Vector × Range with CubicFit — analytic" begin
        # f(x,y,z) = x²y + yz² + xz  (degree ≤ 2 per axis, well within cubic capacity)
        f(x, y, z)         = x^2*y + y*z^2 + x*z
        df_dx(x, y, z)     = 2x*y + z
        df_dy(x, y, z)     = x^2 + z^2
        df_dz(x, y, z)     = 2y*z + x
        d2f_dx2(x, y, z)   = 2y
        d2f_dy2(x, y, z)   = 0.0
        d2f_dz2(x, y, z)   = 2y
        d2f_dxdy(x, y, z)  = 2x
        d2f_dxdz(x, y, z)  = 1.0
        d2f_dydz(x, y, z)  = 2z

        x_range = range(0.0, 2.0, 11)                         # ScalarSpacing
        y_vec   = [0.0, 0.15, 0.35, 0.6, 0.85, 1.0]          # VectorSpacing
        z_range = range(0.0, 1.5, 9)                           # ScalarSpacing
        data = [f(xi, yj, zk) for xi in x_range, yj in y_vec, zk in z_range]

        itp = cubic_interp((x_range, y_vec, z_range), data; bc=CubicFit())
        xq, yq, zq = 1.0, 0.5, 0.7

        # Value
        @test itp((xq, yq, zq)) ≈ f(xq, yq, zq) atol=1e-10

        # First derivatives — each is a mixed derivative order tuple like (1,0,0)
        @test itp((xq, yq, zq); deriv=(1, 0, 0)) ≈ df_dx(xq, yq, zq) atol=1e-6
        @test itp((xq, yq, zq); deriv=(0, 1, 0)) ≈ df_dy(xq, yq, zq) atol=1e-6
        @test itp((xq, yq, zq); deriv=(0, 0, 1)) ≈ df_dz(xq, yq, zq) atol=1e-6

        # Second derivatives — mixed orders across 3 axes
        @test itp((xq, yq, zq); deriv=(2, 0, 0)) ≈ d2f_dx2(xq, yq, zq) atol=1e-4
        @test itp((xq, yq, zq); deriv=(0, 2, 0)) ≈ d2f_dy2(xq, yq, zq) atol=1e-4
        @test itp((xq, yq, zq); deriv=(0, 0, 2)) ≈ d2f_dz2(xq, yq, zq) atol=1e-4
    end

    # ========================================
    # LINEAR ND: HETEROGENEOUS GRID DERIVATIVES
    # ========================================

    @testset "2D Linear: Range × Vector — value and derivative" begin
        # f(x,y) = 3x + 2y + 1 → ∂f/∂x = 3, ∂f/∂y = 2
        f(x, y) = 3x + 2y + 1.0

        x_range = range(0.0, 2.0, 11)
        y_vec   = [0.0, 0.3, 0.6, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        itp = linear_interp((x_range, y_vec), data)
        xq, yq = 1.0, 0.5

        @test itp((xq, yq))              ≈ f(xq, yq) atol=1e-12
        @test itp((xq, yq); deriv=(1, 0)) ≈ 3.0       atol=1e-10
        @test itp((xq, yq); deriv=(0, 1)) ≈ 2.0       atol=1e-10
    end

    # ========================================
    # CONSTANT ND: HETEROGENEOUS GRID
    # ========================================

    @testset "2D Constant: Range × Vector — value evaluation" begin
        x_range = range(0.0, 2.0, 11)
        y_vec   = [0.0, 0.3, 0.6, 0.8, 1.0]
        data = [xi + yj for xi in x_range, yj in y_vec]

        itp = constant_interp((x_range, y_vec), data)
        @test itp((1.0, 0.5)) isa Float64
    end

    # ========================================
    # MIXED SEARCH POLICIES WITH HETEROGENEOUS GRIDS
    # ========================================

    @testset "2D Cubic: heterogeneous grid + mixed search policies" begin
        f(x, y) = x^2 * y
        x_range = range(0.0, 2.0, 15)
        y_vec   = [0.0, 0.2, 0.5, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        # Mixed search: Binary for uniform axis, LinearBinary for non-uniform
        itp = cubic_interp((x_range, y_vec), data;
                           bc=CubicFit(),
                           search=(Binary(), LinearBinary{4}()))

        xq, yq = 1.0, 0.6
        @test itp((xq, yq))              ≈ f(xq, yq)     atol=1e-10
        @test itp((xq, yq); deriv=(1, 0)) ≈ 2xq * yq     atol=1e-6
        @test itp((xq, yq); deriv=(0, 1)) ≈ xq^2          atol=1e-6
        @test itp((xq, yq); deriv=(2, 0)) ≈ 2yq           atol=1e-4
        @test itp((xq, yq); deriv=(0, 2)) ≈ 0.0           atol=1e-4
    end

    # ========================================
    # MIXED EXTRAPOLATION MODES
    # ========================================

    @testset "2D Cubic: heterogeneous grid + mixed extrap modes" begin
        f(x, y) = x * y
        x_range = range(0.0, 2.0, 11)
        y_vec   = [0.0, 0.3, 0.6, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        # Per-axis extrapolation: :constant on x, :extension on y
        itp = cubic_interp((x_range, y_vec), data;
                           bc=CubicFit(),
                           extrap=(:constant, :extension))

        # Interior should work normally
        @test itp((1.0, 0.5); deriv=(1, 0)) ≈ 0.5 atol=1e-6
        @test itp((1.0, 0.5); deriv=(0, 1)) ≈ 1.0 atol=1e-6
    end

    # ========================================
    # MIXED BC TYPES
    # ========================================

    @testset "2D Cubic: heterogeneous grid + per-axis BC" begin
        f(x, y) = x^2 * y^2
        x_range = range(0.0, 2.0, 21)
        y_vec   = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        # Mixed BC: CubicFit on x-axis, NaturalBC on y-axis
        itp = cubic_interp((x_range, y_vec), data; bc=(CubicFit(), QuadraticFit()))
        xq, yq = 1.0, 0.5

        @test itp((xq, yq))              ≈ f(xq, yq)         rtol=1e-10
        @test itp((xq, yq); deriv=(1, 0)) ≈ 2xq * yq^2       rtol=1e-10
        @test itp((xq, yq); deriv=(0, 1)) ≈ 2xq^2 * yq       rtol=1e-10
        @test itp((xq, yq); deriv=(2, 0)) ≈ 2yq^2            rtol=1e-10
    end

    # ========================================
    # VAL-BASED DERIVATIVE SPECIFICATION
    # ========================================

    @testset "2D Cubic: heterogeneous grid + Val derivative spec" begin
        f(x, y) = x^2 + y^2
        x_range = range(0.0, 2.0, 15)
        y_vec   = [0.0, 0.2, 0.5, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        itp = cubic_interp((x_range, y_vec), data; bc=CubicFit())
        xq, yq = 1.0, 0.5

        # Val-based (compile-time) mixed derivatives
        @test itp((xq, yq); deriv=Val((1, 0))) ≈ 2xq   atol=1e-8
        @test itp((xq, yq); deriv=Val((0, 1))) ≈ 2yq   atol=1e-6
        @test itp((xq, yq); deriv=Val((2, 0))) ≈ 2.0    atol=1e-4
        @test itp((xq, yq); deriv=Val((0, 2))) ≈ 2.0    atol=1e-4
    end

    # ========================================
    # BATCH EVALUATION WITH HETEROGENEOUS GRIDS
    # ========================================

    @testset "2D Cubic: heterogeneous grid + batch evaluation" begin
        f(x, y) = x^2 * y
        x_range = range(0.0, 2.0, 15)
        y_vec   = [0.0, 0.2, 0.5, 0.8, 1.0]
        data = [f(xi, yj) for xi in x_range, yj in y_vec]

        itp = cubic_interp((x_range, y_vec), data; bc=CubicFit())

        # Batch query
        xqs = [0.5, 1.0, 1.5]
        yqs = [0.3, 0.6, 0.8]
        vals = itp((xqs, yqs))

        for k in 1:3
            @test vals[k] ≈ f(xqs[k], yqs[k]) atol=1e-10
        end

        # Batch derivative
        dvals = itp((xqs, yqs); deriv=(1, 0))
        for k in 1:3
            @test dvals[k] ≈ 2xqs[k] * yqs[k] atol=1e-6
        end
    end
end
