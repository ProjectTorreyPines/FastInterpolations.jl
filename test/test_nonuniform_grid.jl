# ========================================
# Comprehensive Non-uniform Grid Tests
# ========================================
#
# This file provides rigorous testing of all interpolation methods on non-uniform grids.
# Non-uniform grids expose bugs that uniform grids mask (e.g., h[n] = h[n-1] on uniform).
#
# Organization (separate top-level testsets for reduced compilation overhead):
# 1. Linear Interpolation
# 2. Cubic - Zero-Curvature BC
# 3. Cubic - Periodic BC
# 4. Cubic - ZeroSlope/Deriv1 BC
# 5. Cubic - Deriv2 BC
# 6. Cubic - Mixed BCPair combinations
# 7. Interpolant callable forms
# 8. Edge cases and regression tests

using Test
using FastInterpolations
using FastInterpolations: Deriv1, Deriv2, Deriv3, BCPair

# ========================================
# Test Utilities
# ========================================

# Tolerances for different precision levels
const POLY_RTOL = 100 * eps(Float64)   # Polynomial reproduction (~2.2e-14)
const POLY_ATOL = 100 * eps(Float64)
const FUNC_RTOL = 1e-6                  # General function approximation
const FUNC_ATOL = 1e-10

# Standard non-uniform grids for testing
# These grids have deliberately different spacings at edges

"""Create grid with last interval much larger than previous."""
function grid_large_last(::Type{T}=Float64) where T
    # h = [1, 1, 1, 1, 10] → h[n]=10, h[n-1]=1
    T[0, 1, 2, 3, 4, 14]
end

"""Create grid with last interval much smaller than previous."""
function grid_small_last(::Type{T}=Float64) where T
    # h = [1, 1, 1, 10, 1] → h[n]=1, h[n-1]=10
    T[0, 1, 2, 3, 13, 14]
end

"""Create asymmetric grid with small-large-small pattern."""
function grid_asymmetric(::Type{T}=Float64) where T
    # h = [0.1, 5.0, 5.0, 0.2]
    T[0, 0.1, 5.1, 10.1, 10.3]
end

"""Create geometric progression grid (exponentially increasing h)."""
function grid_geometric(::Type{T}=Float64; h1=T(0.1), ratio=T(1.5), n=8) where T
    x = Vector{T}(undef, n + 1)
    x[1] = zero(T)
    for i in 1:n
        x[i+1] = x[i] + h1 * ratio^(i-1)
    end
    x
end

"""Create clustered grid (dense at boundaries, sparse in middle)."""
function grid_clustered(::Type{T}=Float64) where T
    vcat(
        range(T(0), T(0.5), 6),      # Dense at start
        range(T(1), T(9), 5),         # Sparse in middle
        range(T(9.5), T(10), 6)       # Dense at end
    ) |> collect |> sort |> unique
end

# Test polynomials with known exact derivatives
struct TestPolynomial{T}
    f::Function
    f_prime::Function
    f_double_prime::Function
    f_triple_prime::Function  # Third derivative (constant for cubic, 0 for quadratic/linear)
    name::String
end

const QUADRATIC = TestPolynomial{Float64}(
    t -> 2t^2 - 3t + 1,
    t -> 4t - 3,
    t -> 4.0,
    t -> 0.0,  # f'''(x) = 0 for quadratic
    "quadratic"
)

const CUBIC = TestPolynomial{Float64}(
    t -> t^3 - 2t^2 + 3t - 1,
    t -> 3t^2 - 4t + 3,
    t -> 6t - 4,
    t -> 6.0,  # f'''(x) = 6 (constant for cubic)
    "cubic"
)

const LINEAR = TestPolynomial{Float64}(
    t -> 2.5t - 1.0,
    t -> 2.5,
    t -> 0.0,
    t -> 0.0,  # f'''(x) = 0 for linear
    "linear"
)

# ========================================
# 1. Linear Interpolation
# ========================================
@testset "Non-uniform Grid: Linear Interpolation" begin

    @testset "Basic interpolation accuracy" begin
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("small_last", grid_small_last),
            ("asymmetric", grid_asymmetric),
            ("geometric", grid_geometric),
            ("clustered", grid_clustered),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = LINEAR.f.(x)

                # Query points avoiding exact grid points
                x_min, x_max = extrema(x)
                xi = range(x_min + 0.01, x_max - 0.01, 10) |> collect

                result = linear_interp(x, y, xi)
                expected = LINEAR.f.(xi)

                # Linear function should be exactly reproduced
                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "Knot passage (interpolation through data points)" begin
        x = grid_geometric()
        y = sin.(x)

        itp = linear_interp(x, y)
        for (xi, yi) in zip(x, y)
            @test itp(xi) ≈ yi rtol=POLY_RTOL atol=POLY_ATOL
        end
    end

    @testset "Extrapolation modes" begin
        x = grid_large_last()
        y = QUADRATIC.f.(x)
        x_min, x_max = extrema(x)

        @testset "extension" begin
            xi_extrap = [x_min - 1.0, x_max + 1.0]
            result = linear_interp(x, y, xi_extrap; extrap=ExtendExtrap())
            @test all(isfinite, result)
        end

        @testset "constant" begin
            xi_extrap = [x_min - 1.0, x_max + 1.0]
            result = linear_interp(x, y, xi_extrap; extrap=ConstExtrap())
            @test result[1] ≈ y[1]
            @test result[2] ≈ y[end]
        end
    end

    @testset "Derivative evaluation" begin
        x = grid_small_last()
        y = LINEAR.f.(x)
        itp = linear_interp(x, y)

        x_min, x_max = extrema(x)
        xi = range(x_min + 0.1, x_max - 0.1, 5) |> collect

        for t in xi
            @test itp(t; deriv=DerivOp(1)) ≈ LINEAR.f_prime(t) rtol=POLY_RTOL atol=POLY_ATOL
        end
    end

    @testset "Float32 support" begin
        x = grid_large_last(Float32)
        y = Float32[t^2 for t in x]
        xi = Float32[2.5, 7.5, 12.5]

        result = linear_interp(x, y, xi)
        @test eltype(result) == Float32
        @test all(isfinite, result)
    end
end

# ========================================
# 2. Cubic Interpolation - Zero-Curvature BC
# ========================================
@testset "Non-uniform Grid: Cubic - ZeroCurvBC" begin

    # Note: ZeroCurvBC sets f''(x₀) = f''(xₙ) = 0 at boundaries.
    # This ONLY guarantees exact polynomial reproduction when the polynomial
    # itself has f'' = 0 at the boundaries (i.e., LINEAR polynomial).
    # For QUADRATIC (f''=4) and CUBIC (f''=6t-4), ZeroCurvBC is an approximation.

    @testset "Linear polynomial reproduction (exact)" begin
        # LINEAR has f''=0 everywhere, so ZeroCurvBC matches perfectly
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("small_last", grid_small_last),
            ("geometric", grid_geometric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = LINEAR.f.(x)

                x_min, x_max = extrema(x)
                xi = range(x_min + 0.1, x_max - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=ZeroCurvBC())
                expected = LINEAR.f.(xi)

                # Should exactly reproduce (f''=0 matches ZeroCurvBC)
                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "Higher-order polynomial approximation quality" begin
        # QUADRATIC and CUBIC have non-zero f'' at boundaries,
        # so ZeroCurvBC only provides an approximation (not exact reproduction).
        # On extreme grids (large spacing differences), errors can be significant.
        for poly in [QUADRATIC, CUBIC]
            @testset "$(poly.name) polynomial" begin
                for (grid_name, grid_fn) in [
                    ("large_last", grid_large_last),
                    ("small_last", grid_small_last),
                    ("geometric", grid_geometric),
                ]
                    @testset "Grid: $grid_name" begin
                        x = grid_fn()
                        y = poly.f.(x)

                        x_min, x_max = extrema(x)
                        xi = range(x_min + 0.1, x_max - 0.1, 10) |> collect

                        result = cubic_interp(x, y, xi; bc=ZeroCurvBC())
                        expected = poly.f.(xi)

                        # ZeroCurvBC is an approximation - only test basic sanity:
                        # 1. Results are finite
                        # 2. Results are in the same order of magnitude
                        # On extreme grids with conflicting f'' values, errors can exceed 50%
                        @test all(isfinite, result)

                        # Check results are roughly in the right range (order of magnitude)
                        for (r, e) in zip(result, expected)
                            if abs(e) > 1.0
                                @test abs(r - e) / abs(e) < 1.0  # Same order of magnitude
                            else
                                @test abs(r - e) < 2.0  # Absolute error bound for small values
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Knot passage" begin
        x = grid_asymmetric()
        y = sin.(x)

        cache = CubicSplineCache(x; bc=ZeroCurvBC())
        for (xi, yi) in zip(x, y)
            @test cubic_interp(cache, y, xi) ≈ yi rtol=POLY_RTOL atol=POLY_ATOL
        end
    end

    @testset "Derivative evaluation (approximation)" begin
        # Note: ZeroCurvBC with QUADRATIC won't give exact derivatives since f''≠0
        x = grid_geometric()
        y = QUADRATIC.f.(x)
        itp = cubic_interp(x, y; bc=ZeroCurvBC())

        x_min, x_max = extrema(x)
        xi = range(x_min + 0.1, x_max - 0.1, 5) |> collect

        for t in xi
            # Looser tolerance since ZeroCurvBC is an approximation
            @test itp(t; deriv=DerivOp(1)) ≈ QUADRATIC.f_prime(t) rtol=0.1 atol=0.5
            @test isfinite(itp(t; deriv=DerivOp(2)))
        end
    end

    @testset "Linear derivative evaluation (exact)" begin
        # LINEAR has f''=0, so ZeroCurvBC should give exact derivatives
        x = grid_geometric()
        y = LINEAR.f.(x)
        itp = cubic_interp(x, y; bc=ZeroCurvBC())

        x_min, x_max = extrema(x)
        xi = range(x_min + 0.1, x_max - 0.1, 5) |> collect

        for t in xi
            @test itp(t; deriv=DerivOp(1)) ≈ LINEAR.f_prime(t) rtol=POLY_RTOL atol=POLY_ATOL
            @test itp(t; deriv=DerivOp(2)) ≈ LINEAR.f_double_prime(t) atol=POLY_ATOL  # f''=0
        end
    end

    @testset "Piecewise cubic with f''=0 at boundaries (exact)" begin
        # Construct a piecewise cubic that is:
        # 1. Globally C2 continuous
        # 2. Has f''(x₀) = 0 and f''(xₙ) = 0 at boundaries
        # 3. Analytically known
        #
        # This IS exactly what ZeroCurvBC produces, so it should reproduce exactly!
        #
        # Grid: [0, 1, 2, 3, 13]
        # Piece 1 [0,1]:  f(x) = x³                              → f''(0) = 0
        # Piece 2 [1,2]:  f(x) = -(x-1)³ + 3(x-1)² + 3(x-1) + 1  → C2 at x=1
        # Piece 3 [2,3]:  f(x) = (x-2)³ + 6(x-2) + 6             → C2 at x=2
        # Piece 4 [3,13]: f(x) = -0.1(x-3)³ + 3(x-3)² + 9(x-3) + 13 → f''(13) = 0

        x = [0.0, 1.0, 2.0, 3.0, 13.0]

        # Define the piecewise function
        function natural_piecewise(t)
            if t <= 1.0
                return t^3
            elseif t <= 2.0
                s = t - 1.0
                return -s^3 + 3s^2 + 3s + 1
            elseif t <= 3.0
                s = t - 2.0
                return s^3 + 6s + 6
            else
                s = t - 3.0
                return -0.1s^3 + 3s^2 + 9s + 13
            end
        end

        # First derivative
        function natural_piecewise_deriv1(t)
            if t <= 1.0
                return 3t^2
            elseif t <= 2.0
                s = t - 1.0
                return -3s^2 + 6s + 3
            elseif t <= 3.0
                s = t - 2.0
                return 3s^2 + 6
            else
                s = t - 3.0
                return -0.3s^2 + 6s + 9
            end
        end

        # Second derivative
        function natural_piecewise_deriv2(t)
            if t <= 1.0
                return 6t
            elseif t <= 2.0
                s = t - 1.0
                return -6s + 6
            elseif t <= 3.0
                s = t - 2.0
                return 6s
            else
                s = t - 3.0
                return -0.6s + 6
            end
        end

        # Verify our construction: f''(0) = 0 and f''(13) = 0
        @test natural_piecewise_deriv2(0.0) ≈ 0.0 atol=1e-15
        @test natural_piecewise_deriv2(13.0) ≈ 0.0 atol=1e-15

        # Get y values at grid points
        y = natural_piecewise.(x)

        # ZeroCurvBC should exactly reproduce this piecewise cubic
        xi = [0.5, 1.5, 2.5, 5.0, 10.0, 12.5]

        result = cubic_interp(x, y, xi; bc=ZeroCurvBC())
        expected = natural_piecewise.(xi)

        @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL

        # Test derivatives too
        itp = cubic_interp(x, y; bc=ZeroCurvBC())
        for t in xi
            @test itp(t; deriv=DerivOp(1)) ≈ natural_piecewise_deriv1(t) rtol=POLY_RTOL atol=POLY_ATOL
            @test itp(t; deriv=DerivOp(2)) ≈ natural_piecewise_deriv2(t) rtol=POLY_RTOL atol=POLY_ATOL
        end
    end
end

# ========================================
# 3. Cubic Interpolation - Periodic BC
# ========================================
@testset "Non-uniform Grid: Cubic - PeriodicBC" begin

    @testset "Basic periodic spline sanity" begin
        # Test that periodic BC produces finite, reasonable results on non-uniform grids
        # Note: Sparse non-uniform grids can have significant approximation error

        x_base = sort([0.0, 0.3, 0.7, 1.2, 1.8, 2π])

        f(t) = sin(t)
        y = f.(x_base)
        y[end] = y[1]  # Periodic: wrap-around

        xi = range(0.1, 2π - 0.1, 10) |> collect

        result = cubic_interp(x_base, y, xi; bc=PeriodicBC())

        # Results should be finite and bounded by [-1, 1] (sin range)
        @test all(isfinite, result)
        @test all(r -> -1.5 <= r <= 1.5, result)  # Allow some overshoot
    end

    @testset "Knot passage" begin
        # Periodic spline MUST pass through data points
        x = sort([0.0, 0.5, 1.2, 1.8, 2.5, 3.0])
        y = sin.(x)
        y[end] = y[1]

        cache = CubicSplineCache(x; bc=PeriodicBC())

        # Test interior knots (not boundary wrap-around point)
        for i in 1:length(x)-1
            @test cubic_interp(cache, y, x[i]) ≈ y[i] rtol=POLY_RTOL atol=POLY_ATOL
        end
    end

    @testset "C2 continuity at boundaries" begin
        # Use more points for better conditioning
        x = collect(range(0.0, 4.0, 9))
        period = x[end] - x[1]

        f(t) = sin(2π * t / period)
        y = f.(x)
        y[end] = y[1]

        itp = cubic_interp(x, y; bc=PeriodicBC())

        # Check continuity near boundaries
        h = 1e-6

        # Value continuity - use absolute tolerance since values are near zero
        val_left = itp(x[1] + h)
        val_right = itp(x[end] - h)
        @test abs(val_left - val_right) < 1e-3

        # First derivative continuity
        deriv_left = itp(x[1] + h; deriv=DerivOp(1))
        deriv_right = itp(x[end] - h; deriv=DerivOp(1))
        @test abs(deriv_left - deriv_right) < 0.1  # Relaxed for numerical precision
    end

    @testset "Wrap-around query points" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = sin.(x)
        y[end] = y[1]

        itp = cubic_interp(x, y; bc=PeriodicBC(), extrap=WrapExtrap())

        # Query outside domain should wrap
        period = x[end] - x[1]
        @test itp(0.5) ≈ itp(0.5 + period) rtol=1e-10
        @test itp(1.5) ≈ itp(1.5 - period) rtol=1e-10
    end
end

# ========================================
# 4. Cubic Interpolation - ZeroSlope/Deriv1 BC
# ========================================
@testset "Non-uniform Grid: Cubic - Deriv1 BC" begin

    @testset "Polynomial reproduction with exact derivatives" begin
        for poly in [LINEAR, QUADRATIC, CUBIC]
            @testset "$(poly.name) polynomial" begin
                for (grid_name, grid_fn) in [
                    ("large_last", grid_large_last),
                    ("small_last", grid_small_last),
                    ("asymmetric", grid_asymmetric),
                    ("geometric", grid_geometric),
                ]
                    @testset "Grid: $grid_name" begin
                        x = grid_fn()
                        y = poly.f.(x)
                        x0, xn = first(x), last(x)

                        # Provide exact derivatives at boundaries
                        bc = BCPair(Deriv1(poly.f_prime(x0)), Deriv1(poly.f_prime(xn)))

                        xi = range(x0 + 0.1, xn - 0.1, 15) |> collect

                        result = cubic_interp(x, y, xi; bc=bc)
                        expected = poly.f.(xi)

                        # Should exactly reproduce polynomial
                        @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
                    end
                end
            end
        end
    end

    @testset "ZeroSlopeBC() equivalence (zero slope at boundaries)" begin
        x = grid_large_last()
        y = sin.(x)

        xi = [2.0, 7.0, 12.0]

        result_clamped = cubic_interp(x, y, xi; bc=ZeroSlopeBC())
        result_d1_zero = cubic_interp(x, y, xi; bc=BCPair(Deriv1(0.0), Deriv1(0.0)))

        @test result_clamped ≈ result_d1_zero rtol=POLY_RTOL atol=POLY_ATOL
    end

    @testset "Boundary derivative accuracy" begin
        x = grid_small_last()
        y = CUBIC.f.(x)
        x0, xn = first(x), last(x)

        bc = BCPair(Deriv1(CUBIC.f_prime(x0)), Deriv1(CUBIC.f_prime(xn)))
        itp = cubic_interp(x, y; bc=bc)

        # Derivative at boundaries should match specified values
        h = 1e-10
        @test itp(x0 + h; deriv=DerivOp(1)) ≈ CUBIC.f_prime(x0) rtol=1e-6 atol=1e-8
        @test itp(xn - h; deriv=DerivOp(1)) ≈ CUBIC.f_prime(xn) rtol=1e-6 atol=1e-8
    end

    @testset "Single Deriv1 BC (symmetric)" begin
        x = grid_geometric()
        y = QUADRATIC.f.(x)

        # Single Deriv1 applies to both ends
        slope = 1.5
        result_single = cubic_interp(x, y, [1.0, 2.0]; bc=Deriv1(slope))
        result_pair = cubic_interp(x, y, [1.0, 2.0]; bc=BCPair(Deriv1(slope), Deriv1(slope)))

        @test result_single ≈ result_pair rtol=POLY_RTOL atol=POLY_ATOL
    end
end

# ========================================
# 5. Cubic Interpolation - Deriv2 BC
# ========================================
@testset "Non-uniform Grid: Cubic - Deriv2 BC" begin

    @testset "Polynomial reproduction with exact second derivatives" begin
        for poly in [LINEAR, QUADRATIC]  # Cubic has varying f''
            @testset "$(poly.name) polynomial" begin
                for (grid_name, grid_fn) in [
                    ("large_last", grid_large_last),
                    ("small_last", grid_small_last),
                    ("geometric", grid_geometric),
                ]
                    @testset "Grid: $grid_name" begin
                        x = grid_fn()
                        y = poly.f.(x)
                        x0, xn = first(x), last(x)

                        # Provide exact second derivatives at boundaries
                        bc = BCPair(Deriv2(poly.f_double_prime(x0)), Deriv2(poly.f_double_prime(xn)))

                        xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                        result = cubic_interp(x, y, xi; bc=bc)
                        expected = poly.f.(xi)

                        @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
                    end
                end
            end
        end
    end

    @testset "ZeroCurvBC() equivalence (zero curvature at boundaries)" begin
        x = grid_asymmetric()
        y = sin.(x)

        xi = [0.05, 5.0, 10.2]

        result_natural = cubic_interp(x, y, xi; bc=ZeroCurvBC())
        result_d2_zero = cubic_interp(x, y, xi; bc=BCPair(Deriv2(0.0), Deriv2(0.0)))

        @test result_natural ≈ result_d2_zero rtol=POLY_RTOL atol=POLY_ATOL
    end

    @testset "Single Deriv2 BC (symmetric)" begin
        x = grid_large_last()
        y = QUADRATIC.f.(x)

        curv = 4.0  # Quadratic has constant second derivative
        result_single = cubic_interp(x, y, [2.0, 8.0]; bc=Deriv2(curv))
        result_pair = cubic_interp(x, y, [2.0, 8.0]; bc=BCPair(Deriv2(curv), Deriv2(curv)))

        @test result_single ≈ result_pair rtol=POLY_RTOL atol=POLY_ATOL
    end
end

# ========================================
# 5b. Cubic Interpolation - Deriv3 BC
# ========================================
# Deriv3 BC specifies the third derivative at endpoints.
# For cubic polynomials, f'''(x) = constant, so Deriv3 BC should
# exactly reproduce any cubic polynomial on any non-uniform grid.

@testset "Non-uniform Grid: Cubic - Deriv3 BC" begin

    @testset "Cubic polynomial reproduction with exact third derivative" begin
        # CUBIC polynomial has constant f'''(x) = 6 - perfect for Deriv3 BC exactness test
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("small_last", grid_small_last),
            ("asymmetric", grid_asymmetric),
            ("geometric", grid_geometric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = CUBIC.f.(x)
                x0, xn = first(x), last(x)

                # Provide exact third derivative at both endpoints
                bc = BCPair(Deriv3(CUBIC.f_triple_prime(x0)), Deriv3(CUBIC.f_triple_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 15) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = CUBIC.f.(xi)

                # Should exactly reproduce polynomial
                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "Linear/Quadratic polynomial reproduction with Deriv3(0)" begin
        # LINEAR and QUADRATIC have f'''(x) = 0
        for poly in [LINEAR, QUADRATIC]
            @testset "$(poly.name) polynomial" begin
                for (grid_name, grid_fn) in [
                    ("large_last", grid_large_last),
                    ("geometric", grid_geometric),
                ]
                    @testset "Grid: $grid_name" begin
                        x = grid_fn()
                        y = poly.f.(x)
                        x0, xn = first(x), last(x)

                        bc = BCPair(Deriv3(0.0), Deriv3(0.0))  # f'''(x) = 0 for lower-order polys

                        xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                        result = cubic_interp(x, y, xi; bc=bc)
                        expected = poly.f.(xi)

                        @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
                    end
                end
            end
        end
    end

    @testset "Single Deriv3 BC (symmetric)" begin
        x = grid_large_last()
        y = CUBIC.f.(x)

        third_deriv = 6.0  # CUBIC has constant f'''(x) = 6
        result_single = cubic_interp(x, y, [2.0, 8.0]; bc=Deriv3(third_deriv))
        result_pair = cubic_interp(x, y, [2.0, 8.0]; bc=BCPair(Deriv3(third_deriv), Deriv3(third_deriv)))

        @test result_single ≈ result_pair rtol=POLY_RTOL atol=POLY_ATOL
    end

    @testset "Deriv3 boundary derivative accuracy" begin
        # Verify that deriv=DerivOp(3) evaluation matches the specified BC value
        x = grid_geometric()
        y = CUBIC.f.(x)
        x0, xn = first(x), last(x)

        bc = BCPair(Deriv3(6.0), Deriv3(6.0))
        itp = cubic_interp(x, y; bc=bc)

        # Third derivative should be constant = 6 in first and last intervals
        h = 1e-10
        @test itp(x0 + h; deriv=DerivOp(3)) ≈ 6.0 rtol=1e-6 atol=1e-8
        @test itp(xn - h; deriv=DerivOp(3)) ≈ 6.0 rtol=1e-6 atol=1e-8
    end
end

# ========================================
# 6. Cubic Interpolation - Mixed BCPair
# ========================================
@testset "Non-uniform Grid: Cubic - Mixed BCPair" begin

    @testset "BCPair(Deriv1, Deriv2)" begin
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("small_last", grid_small_last),
            ("asymmetric", grid_asymmetric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = QUADRATIC.f.(x)
                x0, xn = first(x), last(x)

                bc = BCPair(Deriv1(QUADRATIC.f_prime(x0)), Deriv2(QUADRATIC.f_double_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = QUADRATIC.f.(xi)

                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "BCPair(Deriv2, Deriv1)" begin
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("small_last", grid_small_last),
            ("geometric", grid_geometric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = QUADRATIC.f.(x)
                x0, xn = first(x), last(x)

                bc = BCPair(Deriv2(QUADRATIC.f_double_prime(x0)), Deriv1(QUADRATIC.f_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = QUADRATIC.f.(xi)

                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "BCPair(Deriv3, Deriv1) - cubic polynomial" begin
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("asymmetric", grid_asymmetric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = CUBIC.f.(x)
                x0, xn = first(x), last(x)

                bc = BCPair(Deriv3(CUBIC.f_triple_prime(x0)), Deriv1(CUBIC.f_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = CUBIC.f.(xi)

                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "BCPair(Deriv1, Deriv3) - cubic polynomial" begin
        for (grid_name, grid_fn) in [
            ("small_last", grid_small_last),
            ("geometric", grid_geometric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = CUBIC.f.(x)
                x0, xn = first(x), last(x)

                bc = BCPair(Deriv1(CUBIC.f_prime(x0)), Deriv3(CUBIC.f_triple_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = CUBIC.f.(xi)

                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "BCPair(Deriv3, Deriv2) - cubic polynomial" begin
        for (grid_name, grid_fn) in [
            ("large_last", grid_large_last),
            ("geometric", grid_geometric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = CUBIC.f.(x)
                x0, xn = first(x), last(x)

                bc = BCPair(Deriv3(CUBIC.f_triple_prime(x0)), Deriv2(CUBIC.f_double_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = CUBIC.f.(xi)

                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "BCPair(Deriv2, Deriv3) - cubic polynomial" begin
        for (grid_name, grid_fn) in [
            ("small_last", grid_small_last),
            ("asymmetric", grid_asymmetric),
        ]
            @testset "Grid: $grid_name" begin
                x = grid_fn()
                y = CUBIC.f.(x)
                x0, xn = first(x), last(x)

                bc = BCPair(Deriv2(CUBIC.f_double_prime(x0)), Deriv3(CUBIC.f_triple_prime(xn)))

                xi = range(x0 + 0.1, xn - 0.1, 10) |> collect

                result = cubic_interp(x, y, xi; bc=bc)
                expected = CUBIC.f.(xi)

                @test result ≈ expected rtol=POLY_RTOL atol=POLY_ATOL
            end
        end
    end

    @testset "All BCPair combinations give different results" begin
        x = grid_large_last()
        y = sin.(x)
        xi = [5.0]

        results = Dict{String, Float64}()

        for (name, bc) in [
            ("ZeroCurv", ZeroCurvBC()),
            ("ZeroSlope", ZeroSlopeBC()),
            ("D1-D1", BCPair(Deriv1(1.0), Deriv1(0.5))),
            ("D2-D2", BCPair(Deriv2(0.0), Deriv2(-1.0))),
            ("D3-D3", BCPair(Deriv3(0.0), Deriv3(1.0))),
            ("D1-D2", BCPair(Deriv1(1.0), Deriv2(0.0))),
            ("D2-D1", BCPair(Deriv2(0.0), Deriv1(0.5))),
            ("D3-D1", BCPair(Deriv3(0.0), Deriv1(0.5))),
            ("D1-D3", BCPair(Deriv1(1.0), Deriv3(0.0))),
            ("D3-D2", BCPair(Deriv3(0.0), Deriv2(-1.0))),
            ("D2-D3", BCPair(Deriv2(0.0), Deriv3(1.0))),
        ]
            results[name] = cubic_interp(x, y, xi; bc=bc)[1]
        end

        # All should be finite
        @test all(isfinite, values(results))

        # Different BC should give different results (not all the same)
        unique_vals = unique(round.(values(results), digits=10))
        @test length(unique_vals) >= 5  # At least 5 meaningfully different (more combinations now)
    end
end

# ========================================
# 7. Interpolant Callable Forms
# ========================================
@testset "Non-uniform Grid: Interpolant Callable" begin

    @testset "LinearInterpolant" begin
        x = grid_geometric()
        y = LINEAR.f.(x)

        itp = linear_interp(x, y)
        @test itp isa LinearInterpolant

        # Scalar evaluation
        @test itp(1.0) ≈ LINEAR.f(1.0) rtol=POLY_RTOL

        # Broadcast
        xi = [0.5, 1.5, 2.5]
        @test itp.(xi) ≈ LINEAR.f.(xi) rtol=POLY_RTOL

        # Derivative
        @test itp(1.0; deriv=DerivOp(1)) ≈ LINEAR.f_prime(1.0) rtol=POLY_RTOL
    end

    @testset "CubicInterpolant with various BC" begin
        x = grid_large_last()
        y = CUBIC.f.(x)
        x0, xn = first(x), last(x)

        for (name, bc) in [
            ("ZeroCurv", ZeroCurvBC()),
            ("ZeroSlope", ZeroSlopeBC()),
            ("Deriv1", BCPair(Deriv1(CUBIC.f_prime(x0)), Deriv1(CUBIC.f_prime(xn)))),
            ("Deriv2", BCPair(Deriv2(CUBIC.f_double_prime(x0)), Deriv2(CUBIC.f_double_prime(xn)))),
        ]
            @testset "BC: $name" begin
                itp = cubic_interp(x, y; bc=bc)
                @test itp isa CubicInterpolant

                # Scalar evaluation
                @test isfinite(itp(5.0))

                # Broadcast
                xi = [2.0, 7.0, 12.0]
                results = itp.(xi)
                @test all(isfinite, results)

                # Derivatives
                @test isfinite(itp(5.0; deriv=DerivOp(1)))
                @test isfinite(itp(5.0; deriv=DerivOp(2)))
            end
        end
    end

    @testset "CubicSplineCache reuse" begin
        x = grid_small_last()
        cache = CubicSplineCache(x; bc=ZeroCurvBC())

        # Multiple y vectors on same grid
        y1 = sin.(x)
        y2 = cos.(x)
        y3 = x .^ 2

        xi = [2.0, 8.0, 13.5]

        r1 = cubic_interp(cache, y1, xi)
        r2 = cubic_interp(cache, y2, xi)
        r3 = cubic_interp(cache, y3, xi)

        @test all(isfinite, r1)
        @test all(isfinite, r2)
        @test all(isfinite, r3)
        @test r1 != r2  # Different y should give different results
    end
end

# ========================================
# 8. Edge Cases and Regression Tests
# ========================================
@testset "Non-uniform Grid: Edge Cases" begin

    @testset "Extreme spacing ratios" begin
        # 100x difference in spacing
        x = [0.0, 0.01, 0.02, 0.03, 3.03]
        y = x .^ 2

        @test isfinite(linear_interp(x, y, 1.5))
        @test isfinite(cubic_interp(x, y, 1.5; bc=ZeroCurvBC()))
        @test isfinite(cubic_interp(x, y, 1.5; bc=ZeroSlopeBC()))
    end

    @testset "Very small intervals" begin
        x = [0.0, 1e-10, 1.0, 2.0]
        y = x .^ 2

        result = linear_interp(x, y, 0.5)
        @test isfinite(result)
    end

    @testset "Query at grid points (knot passage)" begin
        x = grid_clustered()
        y = sin.(x)

        for bc in [ZeroCurvBC(), ZeroSlopeBC(), PeriodicBC()]
            if bc isa PeriodicBC
                y_periodic = copy(y)
                y_periodic[end] = y_periodic[1]
                result = cubic_interp(x, y_periodic, x; bc=bc)
            else
                result = cubic_interp(x, y, x; bc=bc)
            end

            for (i, xi) in enumerate(x)
                if bc isa PeriodicBC && i == length(x)
                    @test result[i] ≈ y[1] rtol=POLY_RTOL atol=POLY_ATOL
                else
                    @test result[i] ≈ y[i] rtol=POLY_RTOL atol=POLY_ATOL
                end
            end
        end
    end

    @testset "Float32 precision across BC types" begin
        x = grid_large_last(Float32)
        y = Float32.(sin.(x))

        for bc in [
            ZeroCurvBC(),
            ZeroSlopeBC(),
            BCPair(Deriv1(Float32(0.5)), Deriv2(Float32(0.0))),
            BCPair(Deriv3(Float32(0.0)), Deriv1(Float32(0.5))),
            Deriv3(Float32(0.0)),
        ]
            result = cubic_interp(x, y, Float32(5.0); bc=bc)
            @test result isa Float32
            @test isfinite(result)
        end
    end

    @testset "In-place interpolation" begin
        x = grid_geometric()
        y = QUADRATIC.f.(x)
        xi = [0.5, 1.5, 2.5]

        output = zeros(3)
        linear_interp!(output, x, y, xi)
        @test all(isfinite, output)

        output_cubic = zeros(3)
        cubic_interp!(output_cubic, x, y, xi; bc=ZeroCurvBC())
        @test all(isfinite, output_cubic)
    end

    @testset "Vector vs scalar query consistency" begin
        x = grid_asymmetric()
        y = CUBIC.f.(x)

        xi_scalar = 5.0
        xi_vec = [5.0]

        for bc in [ZeroCurvBC(), ZeroSlopeBC()]
            result_scalar = cubic_interp(x, y, xi_scalar; bc=bc)
            result_vec = cubic_interp(x, y, xi_vec; bc=bc)

            @test result_scalar ≈ result_vec[1] rtol=POLY_RTOL atol=POLY_ATOL
        end
    end
end
