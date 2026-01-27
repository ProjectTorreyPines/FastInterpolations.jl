# ========================================
# PolyFit and LagrangeBC (Estimated Derivative BC) Tests
# ========================================
# Tests for PolyFit{D} (polynomial fitting boundary conditions)
# which include:
#   - LinearFit   = PolyFit{1} (2-point, O(h))
#   - ParabolaFit = PolyFit{2} (3-point, O(h²))
#   - CubicFit    = PolyFit{3} (4-point, O(h³))
#   - LagrangeBC  = PolyFit{3} (deprecated alias)
#
# These BCs compute endpoint derivatives from data automatically,
# unlike Deriv1/Deriv2 which require user-specified values.

using Test
using FastInterpolations

# Tolerance constants for numerical comparisons (local to this test file)
const LAGRANGE_RTOL = 1e-12
const LAGRANGE_ATOL = 1e-12

# ========================================
# PolyFit{D} Type System Tests
# ========================================

@testset "PolyFit{D} Type System" begin

    @testset "Type Aliases" begin
        # Verify type aliases are truly identical to PolyFit{D}
        @test LinearFit{Float64} === PolyFit{1, Float64}
        @test ParabolaFit{Float64} === PolyFit{2, Float64}
        @test CubicFit{Float64} === PolyFit{3, Float64}
        @test LagrangeBC{Float64} === PolyFit{3, Float64}  # Deprecated alias

        @test LinearFit{Float32} === PolyFit{1, Float32}
        @test ParabolaFit{Float32} === PolyFit{2, Float32}
        @test CubicFit{Float32} === PolyFit{3, Float32}
    end

    @testset "Type Hierarchy" begin
        # All PolyFit should be PointBC subtypes
        @test PolyFit{1, Float64} <: FastInterpolations.PointBC{Float64}
        @test PolyFit{2, Float64} <: FastInterpolations.PointBC{Float64}
        @test PolyFit{3, Float64} <: FastInterpolations.PointBC{Float64}

        # PointBC <: AbstractBC
        @test FastInterpolations.PointBC{Float64} <: AbstractBC{Float64}
    end

    @testset "Default Constructors" begin
        # Default constructors should return Float64 variant
        @test LinearFit() isa PolyFit{1, Float64}
        @test ParabolaFit() isa PolyFit{2, Float64}
        @test CubicFit() isa PolyFit{3, Float64}
        @test PolyFit{3}() isa CubicFit{Float64}
    end

    @testset "Explicit Type Constructors" begin
        @test LinearFit{Float32}() isa PolyFit{1, Float32}
        @test ParabolaFit{Float32}() isa PolyFit{2, Float32}
        @test CubicFit{Float32}() isa PolyFit{3, Float32}

        @test PolyFit{1, Float32}() isa LinearFit{Float32}
        @test PolyFit{2, Float32}() isa ParabolaFit{Float32}
        @test PolyFit{3, Float32}() isa CubicFit{Float32}
    end

    @testset "Degree Validation" begin
        # Polynomial degree must be >= 1
        @test_throws ArgumentError PolyFit{0, Float64}()
        @test_throws ArgumentError PolyFit{-1, Float64}()

        # Valid degrees should work
        @test PolyFit{1, Float64}() isa PolyFit{1, Float64}
        @test PolyFit{4, Float64}() isa PolyFit{4, Float64}  # Higher orders
    end

    @testset "Type Promotion" begin
        # _promote_pointbc should work with generic PolyFit
        @test FastInterpolations._promote_pointbc(PolyFit{1}(), Float32) isa PolyFit{1, Float32}
        @test FastInterpolations._promote_pointbc(PolyFit{2}(), Float32) isa PolyFit{2, Float32}
        @test FastInterpolations._promote_pointbc(PolyFit{3}(), Float32) isa PolyFit{3, Float32}

        # Round-trip promotion
        pf = PolyFit{2, Float64}()
        pf32 = FastInterpolations._promote_pointbc(pf, Float32)
        pf64 = FastInterpolations._promote_pointbc(pf32, Float64)
        @test pf64 isa PolyFit{2, Float64}
    end

    @testset "Not Periodic" begin
        # PolyFit is never periodic
        @test FastInterpolations._is_periodic_bc(PolyFit{1}()) == false
        @test FastInterpolations._is_periodic_bc(PolyFit{2}()) == false
        @test FastInterpolations._is_periodic_bc(PolyFit{3}()) == false
    end

end

@testset "PolyFit Normalization and BCPair" begin

    @testset "Single PolyFit → Symmetric BCPair" begin
        bc1 = FastInterpolations._normalize_bc(LinearFit(), Float64)
        @test bc1 isa BCPair{Float64, PolyFit{1, Float64}, PolyFit{1, Float64}}

        bc2 = FastInterpolations._normalize_bc(ParabolaFit(), Float64)
        @test bc2 isa BCPair{Float64, PolyFit{2, Float64}, PolyFit{2, Float64}}

        bc3 = FastInterpolations._normalize_bc(CubicFit(), Float64)
        @test bc3 isa BCPair{Float64, PolyFit{3, Float64}, PolyFit{3, Float64}}
    end

    @testset "Type Promotion in Normalization" begin
        bc = FastInterpolations._normalize_bc(CubicFit{Float64}(), Float32)
        @test bc isa BCPair{Float32, PolyFit{3, Float32}, PolyFit{3, Float32}}
    end

    @testset "Mixed BCPair with PolyFit" begin
        bc1 = BCPair(LinearFit(), Deriv1(0.0))
        @test bc1 isa BCPair{Float64, PolyFit{1, Float64}, Deriv1{Float64}}

        bc2 = BCPair(Deriv1(1.0), ParabolaFit())
        @test bc2 isa BCPair{Float64, Deriv1{Float64}, PolyFit{2, Float64}}

        bc3 = BCPair(CubicFit(), Deriv2(0.0))
        @test bc3 isa BCPair{Float64, PolyFit{3, Float64}, Deriv2{Float64}}
    end

end

@testset "CubicFit Integration (as CubicFit, not LagrangeBC)" begin

    @testset "Cubic Polynomial Reproduction with CubicFit" begin
        # f(x) = x³ - 2x² + x - 1, f'(x) = 3x² - 4x + 1
        f_cubic(x) = x^3 - 2x^2 + x - 1

        x = range(0.0, 2.0, 21)
        y = f_cubic.(x)
        xi = [0.15, 0.5, 1.0, 1.5, 1.85]

        result = cubic_interp(x, y, xi; bc=CubicFit())
        expected = f_cubic.(xi)
        @test result ≈ expected rtol=LAGRANGE_RTOL atol=LAGRANGE_ATOL
    end

    @testset "CubicFit with CubicInterpolant" begin
        x = range(0.0, 1.0, 21)
        y = sin.(π .* x)

        itp = cubic_interp(x, y; bc=CubicFit())
        @test itp isa CubicInterpolant
        @test isfinite(itp(0.5))
    end

    @testset "CubicFit Requires 4 Points" begin
        x_short = range(0.0, 1.0, 3)
        y_short = sin.(x_short)
        @test_throws ArgumentError cubic_interp(x_short, y_short, 0.5; bc=CubicFit())

        x_ok = range(0.0, 1.0, 4)
        y_ok = sin.(x_ok)
        @test isfinite(cubic_interp(x_ok, y_ok, 0.5; bc=CubicFit()))
    end

end

# ========================================
# Phase 1: LagrangeBC Type System Tests (Backward Compatibility)
# ========================================

@testset "LagrangeBC Type System" begin

    @testset "Type Hierarchy" begin
        # LagrangeBC should be a PointBC subtype (like Deriv1, Deriv2)
        @test LagrangeBC{Float64} <: FastInterpolations.PointBC{Float64}
        @test LagrangeBC{Float32} <: FastInterpolations.PointBC{Float32}

        # PointBC <: AbstractBC
        @test FastInterpolations.PointBC{Float64} <: AbstractBC{Float64}
    end

    @testset "Default Construction" begin
        # Default constructor should return Float64 variant
        lbc = LagrangeBC()
        @test lbc isa LagrangeBC{Float64}
    end

    @testset "Explicit Type Construction" begin
        lbc32 = LagrangeBC{Float32}()
        @test lbc32 isa LagrangeBC{Float32}

        lbc64 = LagrangeBC{Float64}()
        @test lbc64 isa LagrangeBC{Float64}
    end

    @testset "Type Conversion Constructor" begin
        # LagrangeBC{T}(::LagrangeBC) should convert between float types
        lbc_f64 = LagrangeBC()
        lbc_f32 = LagrangeBC{Float32}(lbc_f64)
        @test lbc_f32 isa LagrangeBC{Float32}

        # Round-trip conversion
        lbc_back = LagrangeBC{Float64}(lbc_f32)
        @test lbc_back isa LagrangeBC{Float64}
    end

end

@testset "LagrangeBC Normalization" begin

    @testset "Single LagrangeBC → Symmetric BCPair" begin
        # When a single LagrangeBC is provided, it should be applied to both ends
        bc = FastInterpolations._normalize_bc(LagrangeBC(), Float64)
        @test bc isa BCPair{Float64, LagrangeBC{Float64}, LagrangeBC{Float64}}
    end

    @testset "Type Promotion in Normalization" begin
        # Float64 LagrangeBC → Float32 target should promote
        bc_promoted = FastInterpolations._normalize_bc(LagrangeBC(), Float32)
        @test bc_promoted isa BCPair{Float32, LagrangeBC{Float32}, LagrangeBC{Float32}}
    end

    @testset "_promote_pointbc for LagrangeBC" begin
        # Direct promotion helper
        lbc_promoted = FastInterpolations._promote_pointbc(LagrangeBC(), Float32)
        @test lbc_promoted isa LagrangeBC{Float32}

        lbc_promoted64 = FastInterpolations._promote_pointbc(LagrangeBC{Float32}(), Float64)
        @test lbc_promoted64 isa LagrangeBC{Float64}
    end

    @testset "_is_periodic_bc for LagrangeBC" begin
        # LagrangeBC is not periodic
        @test FastInterpolations._is_periodic_bc(LagrangeBC()) == false
    end

end

@testset "LagrangeBC BCPair Construction" begin

    @testset "Symmetric LagrangeBC BCPair" begin
        bc = BCPair(LagrangeBC(), LagrangeBC())
        @test bc isa BCPair{Float64, LagrangeBC{Float64}, LagrangeBC{Float64}}
    end

    @testset "Mixed BCPair: LagrangeBC + Deriv1" begin
        bc = BCPair(LagrangeBC(), Deriv1(0.0))
        @test bc isa BCPair{Float64, LagrangeBC{Float64}, Deriv1{Float64}}
    end

    @testset "Mixed BCPair: Deriv1 + LagrangeBC" begin
        bc = BCPair(Deriv1(1.0), LagrangeBC())
        @test bc isa BCPair{Float64, Deriv1{Float64}, LagrangeBC{Float64}}
    end

    @testset "Mixed BCPair: LagrangeBC + Deriv2" begin
        bc = BCPair(LagrangeBC(), Deriv2(0.0))
        @test bc isa BCPair{Float64, LagrangeBC{Float64}, Deriv2{Float64}}
    end

    @testset "Mixed BCPair: Deriv2 + LagrangeBC" begin
        bc = BCPair(Deriv2(-1.0), LagrangeBC())
        @test bc isa BCPair{Float64, Deriv2{Float64}, LagrangeBC{Float64}}
    end

    @testset "Float32 BCPair" begin
        bc = BCPair(LagrangeBC{Float32}(), Deriv1(0.0f0))
        @test bc isa BCPair{Float32, LagrangeBC{Float32}, Deriv1{Float32}}
    end

end

# ========================================
# Phase 3: Lagrange Kernel Tests
# ========================================

@testset "Lagrange Kernel Correctness" begin

    @testset "Left Endpoint: f(x) = x³" begin
        # f(x) = x³, f'(x) = 3x²
        # f'(0) = 0
        h = 0.1
        inv_h = 1 / h
        f = x -> x^3

        # First 4 points: x = 0.0, 0.1, 0.2, 0.3
        f1, f2, f3, f4 = f(0.0), f(0.1), f(0.2), f(0.3)

        d_left = FastInterpolations._lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h)
        @test d_left ≈ 0.0 atol=1e-10  # f'(0) = 0
    end

    @testset "Right Endpoint: f(x) = x³" begin
        # f(x) = x³, f'(x) = 3x²
        # f'(1) = 3
        h = 0.1
        inv_h = 1 / h
        f = x -> x^3

        # Last 4 points: x = 0.7, 0.8, 0.9, 1.0
        fnm3, fnm2, fnm1, fn = f(0.7), f(0.8), f(0.9), f(1.0)

        d_right = FastInterpolations._lagrange_d1_right_uniform(fnm3, fnm2, fnm1, fn, inv_h)
        @test d_right ≈ 3.0 atol=1e-10  # f'(1) = 3
    end

    @testset "Left Endpoint: f(x) = x² - 2x + 1" begin
        # f(x) = x² - 2x + 1, f'(x) = 2x - 2
        # f'(0) = -2
        h = 0.25
        inv_h = 1 / h
        f = x -> x^2 - 2x + 1

        f1, f2, f3, f4 = f(0.0), f(0.25), f(0.5), f(0.75)
        d_left = FastInterpolations._lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h)
        @test d_left ≈ -2.0 atol=1e-10
    end

    @testset "Right Endpoint: f(x) = x² - 2x + 1" begin
        # f'(2) = 2*2 - 2 = 2
        h = 0.25
        inv_h = 1 / h
        f = x -> x^2 - 2x + 1

        fnm3, fnm2, fnm1, fn = f(1.25), f(1.5), f(1.75), f(2.0)
        d_right = FastInterpolations._lagrange_d1_right_uniform(fnm3, fnm2, fnm1, fn, inv_h)
        @test d_right ≈ 2.0 atol=1e-10
    end

    @testset "Linear Function (Exact)" begin
        # f(x) = 3x + 2, f'(x) = 3
        h = 0.5
        inv_h = 1 / h
        f = x -> 3x + 2

        f1, f2, f3, f4 = f(0.0), f(0.5), f(1.0), f(1.5)
        d_left = FastInterpolations._lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h)
        @test d_left ≈ 3.0 atol=1e-14  # Exact for linear

        fnm3, fnm2, fnm1, fn = f(1.5), f(2.0), f(2.5), f(3.0)
        d_right = FastInterpolations._lagrange_d1_right_uniform(fnm3, fnm2, fnm1, fn, inv_h)
        @test d_right ≈ 3.0 atol=1e-14
    end

    @testset "Float32 Precision" begin
        h = Float32(0.1)
        inv_h = 1 / h
        f = x -> x^2

        f1, f2, f3, f4 = Float32.(f.([0.0, 0.1, 0.2, 0.3]))
        d_left = FastInterpolations._lagrange_d1_left_uniform(f1, f2, f3, f4, inv_h)
        @test d_left isa Float32
        @test d_left ≈ Float32(0.0) atol=1e-5  # f'(0) = 0
    end

end

# ========================================
# Non-Uniform Grid Kernel Tests (Precomputed Coefficients)
# ========================================

@testset "Non-Uniform Lagrange Kernels" begin

    @testset "Coefficient Precomputation: Left Endpoint" begin
        # Non-uniform grid: [0, 0.1, 0.3, 0.6]
        x1, x2, x3, x4 = 0.0, 0.1, 0.3, 0.6
        c1, c2, c3, c4 = FastInterpolations._lagrange_coeffs_left(x1, x2, x3, x4)

        # Test: coefficients should be finite
        @test all(isfinite, (c1, c2, c3, c4))

        # Test: sum of coefficients for constant function should be 0
        # f(x) = 1, f'(x) = 0
        f1, f2, f3, f4 = 1.0, 1.0, 1.0, 1.0
        deriv = FastInterpolations._lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
        @test deriv ≈ 0.0 atol=1e-14
    end

    @testset "Coefficient Precomputation: Right Endpoint" begin
        # Non-uniform grid: [0.4, 0.7, 0.9, 1.0]
        x1, x2, x3, x4 = 0.4, 0.7, 0.9, 1.0
        c1, c2, c3, c4 = FastInterpolations._lagrange_coeffs_right(x1, x2, x3, x4)

        # Test: coefficients should be finite
        @test all(isfinite, (c1, c2, c3, c4))

        # Test: constant function f(x) = 5, f'(x) = 0
        f1, f2, f3, f4 = 5.0, 5.0, 5.0, 5.0
        deriv = FastInterpolations._lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
        @test deriv ≈ 0.0 atol=1e-14
    end

    @testset "Linear Function (Exact)" begin
        # f(x) = 2x + 3, f'(x) = 2
        # Non-uniform grid
        x_left = [0.0, 0.1, 0.25, 0.5]
        x_right = [0.5, 0.75, 0.9, 1.0]
        f_lin(x) = 2x + 3

        # Left endpoint
        c_left = FastInterpolations._lagrange_coeffs_left(x_left...)
        f_left = f_lin.(x_left)
        d_left = FastInterpolations._lagrange_d1_nonuniform(c_left..., f_left...)
        @test d_left ≈ 2.0 atol=1e-13

        # Right endpoint
        c_right = FastInterpolations._lagrange_coeffs_right(x_right...)
        f_right = f_lin.(x_right)
        d_right = FastInterpolations._lagrange_d1_nonuniform(c_right..., f_right...)
        @test d_right ≈ 2.0 atol=1e-13
    end

    @testset "Quadratic Function (Exact)" begin
        # f(x) = x² - 3x + 2, f'(x) = 2x - 3
        # f'(0) = -3, f'(1) = -1
        x_left = [0.0, 0.15, 0.35, 0.6]
        x_right = [0.4, 0.65, 0.85, 1.0]
        f_quad(x) = x^2 - 3x + 2

        # Left endpoint: f'(0) = -3
        c_left = FastInterpolations._lagrange_coeffs_left(x_left...)
        f_left = f_quad.(x_left)
        d_left = FastInterpolations._lagrange_d1_nonuniform(c_left..., f_left...)
        @test d_left ≈ -3.0 atol=1e-12

        # Right endpoint: f'(1) = -1
        c_right = FastInterpolations._lagrange_coeffs_right(x_right...)
        f_right = f_quad.(x_right)
        d_right = FastInterpolations._lagrange_d1_nonuniform(c_right..., f_right...)
        @test d_right ≈ -1.0 atol=1e-12
    end

    @testset "Cubic Function (Exact)" begin
        # f(x) = x³, f'(x) = 3x²
        # f'(0) = 0, f'(1) = 3
        x_left = [0.0, 0.2, 0.4, 0.7]
        x_right = [0.3, 0.6, 0.8, 1.0]
        f_cub(x) = x^3

        # Left endpoint: f'(0) = 0
        c_left = FastInterpolations._lagrange_coeffs_left(x_left...)
        f_left = f_cub.(x_left)
        d_left = FastInterpolations._lagrange_d1_nonuniform(c_left..., f_left...)
        @test d_left ≈ 0.0 atol=1e-12

        # Right endpoint: f'(1) = 3
        c_right = FastInterpolations._lagrange_coeffs_right(x_right...)
        f_right = f_cub.(x_right)
        d_right = FastInterpolations._lagrange_d1_nonuniform(c_right..., f_right...)
        @test d_right ≈ 3.0 atol=1e-11
    end

    @testset "Equivalence: Precomputed vs On-the-fly (Non-uniform)" begin
        # Verify precomputed kernel matches the on-the-fly calculation
        xs = [0.0, 0.12, 0.31, 0.55, 0.72, 0.88, 1.0]
        f_test(x) = sin(2π * x)
        ys = f_test.(xs)

        # Left endpoint using precomputed coefficients
        c_left = FastInterpolations._lagrange_coeffs_left(xs[1], xs[2], xs[3], xs[4])
        d_left_precomp = FastInterpolations._lagrange_d1_nonuniform(c_left..., ys[1], ys[2], ys[3], ys[4])

        # Left endpoint using on-the-fly (existing _estimate_endpoint_derivative)
        d_left_onfly = FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:left))

        @test d_left_precomp ≈ d_left_onfly rtol=1e-14

        # Right endpoint
        n = length(xs)
        c_right = FastInterpolations._lagrange_coeffs_right(xs[n-3], xs[n-2], xs[n-1], xs[n])
        d_right_precomp = FastInterpolations._lagrange_d1_nonuniform(c_right..., ys[n-3], ys[n-2], ys[n-1], ys[n])
        d_right_onfly = FastInterpolations._estimate_endpoint_derivative(xs, ys, Val(:right))

        @test d_right_precomp ≈ d_right_onfly rtol=1e-14
    end

    @testset "Equivalence: Uniform Grid - Precomputed vs Direct" begin
        # For uniform grids, precomputed coefficients should give same result as direct formula
        h = 0.25
        xs_uniform = [0.0, 0.25, 0.5, 0.75, 1.0]
        f_test(x) = x^3 - x
        ys = f_test.(xs_uniform)

        # Precomputed (non-uniform kernel with uniform data)
        c_left = FastInterpolations._lagrange_coeffs_left(xs_uniform[1:4]...)
        d_precomp = FastInterpolations._lagrange_d1_nonuniform(c_left..., ys[1:4]...)

        # Direct uniform kernel
        inv_h = 1 / h
        d_direct = FastInterpolations._lagrange_d1_left_uniform(ys[1], ys[2], ys[3], ys[4], inv_h)

        @test d_precomp ≈ d_direct rtol=1e-13
    end

    @testset "Float32 Precision" begin
        x1, x2, x3, x4 = Float32.([0.0, 0.1, 0.3, 0.6])
        c1, c2, c3, c4 = FastInterpolations._lagrange_coeffs_left(x1, x2, x3, x4)

        @test c1 isa Float32
        @test c2 isa Float32
        @test c3 isa Float32
        @test c4 isa Float32

        f1, f2, f3, f4 = Float32.([0.0, 0.1, 0.3, 0.6].^2)  # f(x) = x²
        deriv = FastInterpolations._lagrange_d1_nonuniform(c1, c2, c3, c4, f1, f2, f3, f4)
        @test deriv isa Float32
        @test deriv ≈ Float32(0.0) atol=1e-5  # f'(0) = 0
    end

end

# ========================================
# Phase 4: LagrangeBC Integration Tests
# ========================================

@testset "LagrangeBC Mathematical Correctness" begin

    @testset "Cubic Polynomial Reproduction" begin
        # f(x) = x³ - 2x² + x - 1
        # f'(x) = 3x² - 4x + 1
        # f'(0) = 1, f'(2) = 5
        # Cubic splines with correct endpoint derivatives reproduce cubics exactly
        f_cubic(x) = x^3 - 2x^2 + x - 1

        x_cubic = range(0.0, 2.0, 21)  # Uniform grid, 21 points
        y_cubic = f_cubic.(x_cubic)
        xi_cubic = [0.15, 0.5, 1.0, 1.5, 1.85]

        result = cubic_interp(x_cubic, y_cubic, xi_cubic; bc=LagrangeBC())
        expected = f_cubic.(xi_cubic)
        @test result ≈ expected rtol=LAGRANGE_RTOL atol=LAGRANGE_ATOL
    end

    @testset "Quadratic Polynomial Reproduction" begin
        # f(x) = 2x² - 3x + 1
        # f'(x) = 4x - 3
        f_quad(x) = 2x^2 - 3x + 1

        x_quad = range(-1.0, 2.0, 31)
        y_quad = f_quad.(x_quad)
        xi_quad = [-0.7, 0.0, 0.5, 1.3, 1.9]

        result = cubic_interp(x_quad, y_quad, xi_quad; bc=LagrangeBC())
        expected = f_quad.(xi_quad)
        @test result ≈ expected rtol=LAGRANGE_RTOL atol=LAGRANGE_ATOL
    end

    @testset "Linear Function Reproduction" begin
        # f(x) = 3x + 2
        f_lin(x) = 3x + 2

        x_lin = range(0.0, 5.0, 11)
        y_lin = f_lin.(x_lin)
        xi_lin = [0.5, 1.5, 2.5, 3.5, 4.5]

        result = cubic_interp(x_lin, y_lin, xi_lin; bc=LagrangeBC())
        expected = f_lin.(xi_lin)
        @test result ≈ expected rtol=LAGRANGE_RTOL atol=LAGRANGE_ATOL
    end

    @testset "Smooth Function (sin)" begin
        # sin(x) is smooth; LagrangeBC should give reasonable results
        x_sin = range(0.0, 2π, 33)
        y_sin = sin.(x_sin)
        xi_sin = [0.5, 1.0, 2.0, 4.0, 5.5]

        result = cubic_interp(x_sin, y_sin, xi_sin; bc=LagrangeBC())

        # Should be close to sin (not exact, but reasonable)
        expected = sin.(xi_sin)
        @test result ≈ expected rtol=1e-4  # Looser tolerance for non-polynomial
    end

end

@testset "LagrangeBC Mixed with Other BCs" begin

    @testset "LagrangeBC Left, Deriv2(0) Right (Natural)" begin
        f_mixed1(x) = sin(π * x)
        x_m1 = range(0.0, 1.0, 17)
        y_m1 = f_mixed1.(x_m1)

        bc = BCPair(LagrangeBC(), Deriv2(0.0))
        result = cubic_interp(x_m1, y_m1, 0.5; bc=bc)
        @test isfinite(result)

        # Compare with known approximate value
        @test abs(result - sin(π * 0.5)) < 0.01
    end

    @testset "Deriv1(exact) Left, LagrangeBC Right" begin
        # f(x) = x², f'(0) = 0
        f_mixed2(x) = x^2
        x_m2 = range(0.0, 2.0, 21)
        y_m2 = f_mixed2.(x_m2)

        bc = BCPair(Deriv1(0.0), LagrangeBC())  # Exact left, estimated right
        result = cubic_interp(x_m2, y_m2, [0.5, 1.0, 1.5]; bc=bc)
        expected = f_mixed2.([0.5, 1.0, 1.5])
        @test result ≈ expected rtol=1e-10
    end

    @testset "LagrangeBC Left, Deriv1(exact) Right" begin
        # f(x) = x², f'(2) = 4
        f_mixed3(x) = x^2
        x_m3 = range(0.0, 2.0, 21)
        y_m3 = f_mixed3.(x_m3)

        bc = BCPair(LagrangeBC(), Deriv1(4.0))  # Estimated left, exact right
        result = cubic_interp(x_m3, y_m3, [0.5, 1.0, 1.5]; bc=bc)
        expected = f_mixed3.([0.5, 1.0, 1.5])
        @test result ≈ expected rtol=1e-10
    end

    @testset "LagrangeBC Left, Deriv3(exact) Right" begin
        # f(x) = x³, f'''(x) = 6
        f_mixed4(x) = x^3
        x_m4 = range(0.0, 1.0, 21)
        y_m4 = f_mixed4.(x_m4)

        bc = BCPair(LagrangeBC(), Deriv3(6.0))
        result = cubic_interp(x_m4, y_m4, 0.5; bc=bc)
        @test abs(result - f_mixed4(0.5)) < 1e-10
    end

end

@testset "LagrangeBC Error Handling" begin

    @testset "Requires Minimum 4 Points" begin
        # LagrangeBC uses 4-point stencil, needs at least 4 points
        x_short = range(0.0, 1.0, 3)  # Only 3 points
        y_short = sin.(x_short)

        @test_throws ArgumentError cubic_interp(x_short, y_short, 0.5; bc=LagrangeBC())
    end

    @testset "Exactly 4 Points Should Work" begin
        x = range(0.0, 1.0, 4)  # Minimum: 4 points
        y = sin.(x)

        result = cubic_interp(x, y, 0.5; bc=LagrangeBC())
        @test isfinite(result)
    end

end

@testset "LagrangeBC with CubicSplineCache" begin

    @testset "Cache Creation with LagrangeBC" begin
        # Use Range directly (not collect) - LagrangeBC requires uniform grid
        x = range(0.0, 1.0, 21)
        cache = CubicSplineCache(x; bc=LagrangeBC())
        @test cache isa CubicSplineCache
    end

    @testset "Cache Reuse for Multiple y Vectors" begin
        # Use Range directly (not collect) - LagrangeBC requires uniform grid
        x = range(0.0, 2.0, 21)
        cache = CubicSplineCache(x; bc=LagrangeBC())

        y1 = sin.(π .* collect(x))
        y2 = cos.(π .* collect(x))

        result1 = cubic_interp(cache, y1, 0.5)
        result2 = cubic_interp(cache, y2, 0.5)

        @test isfinite(result1)
        @test isfinite(result2)
        @test result1 != result2  # Different y should give different results
    end

    @testset "In-Place API with LagrangeBC" begin
        # Use Range directly (not collect) - LagrangeBC requires uniform grid
        x = range(0.0, 1.0, 21)
        y = sin.(π .* collect(x))
        cache = CubicSplineCache(x; bc=LagrangeBC())

        xi = [0.1, 0.3, 0.5, 0.7, 0.9]
        output = zeros(5)
        cubic_interp!(output, cache, y, xi)

        @test all(isfinite, output)
    end

end

@testset "LagrangeBC with CubicInterpolant" begin

    @testset "2-arg Form Creates Interpolant" begin
        x = range(0.0, 1.0, 21)
        y = sin.(π .* x)

        itp = cubic_interp(x, y; bc=LagrangeBC())
        @test itp isa CubicInterpolant
    end

    @testset "Callable Interface" begin
        x = range(0.0, 2.0, 21)
        y = sin.(π .* x)

        itp = cubic_interp(x, y; bc=LagrangeBC())

        @test isfinite(itp(0.5))
        @test isfinite(itp(1.0))
        @test isfinite(itp(1.5))
    end

    @testset "Broadcast Evaluation" begin
        x = range(0.0, 1.0, 21)
        y = sin.(π .* x)

        itp = cubic_interp(x, y; bc=LagrangeBC())
        xi = [0.2, 0.4, 0.6, 0.8]

        results = itp.(xi)
        @test length(results) == 4
        @test all(isfinite, results)
    end

end

@testset "LagrangeBC with CubicSeriesInterpolant" begin

    @testset "Multiple Series" begin
        x = range(0.0, 1.0, 21)
        Y = [sin.(π .* x) cos.(π .* x) x.^2]  # 3 series

        itp = cubic_interp(x, Y; bc=LagrangeBC())
        @test itp isa CubicSeriesInterpolant

        result = itp(0.5)
        @test length(result) == 3
        @test all(isfinite, result)
    end

end

@testset "LagrangeBC Float32 Support" begin

    @testset "Float32 Grid and Values" begin
        x = range(0.0f0, 1.0f0, 21)
        y = Float32.(sin.(Float64.(x) .* π))

        result = cubic_interp(x, y, 0.5f0; bc=LagrangeBC())
        @test result isa Float32
        @test isfinite(result)
    end

    @testset "Float32 Cache" begin
        # Use Range directly (not collect) - LagrangeBC requires uniform grid
        x = range(0.0f0, 1.0f0, 21)
        cache = CubicSplineCache(x; bc=LagrangeBC{Float32}())

        @test eltype(cache.x) == Float32
    end

end

# ========================================
# Note: FDMBC Tests Not Needed
# ========================================
# Analysis showed that Lagrange polynomial derivative estimation
# is mathematically equivalent to Finite Difference Method (FDM):
#   - Same coefficients for uniform grids
#   - Same formulas for non-uniform grids
# Therefore, no separate FDMBC type is implemented.
# Use PolyFit{D} for all polynomial fitting boundary conditions.
