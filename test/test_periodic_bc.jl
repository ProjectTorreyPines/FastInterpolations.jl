using Test
using FastInterpolations

@testset "Periodic BC Tests" begin

    @testset "Linear Periodic BC" begin
        # Periodic function: sin on [0, 2π]
        N = 101
        x = range(0.0, 2π, N)
        y = sin.(x)

        @testset "Basic wrapping" begin
            # Interior point
            @test linear_interp(x, y, π/4; bc=:periodic) ≈ sin(π/4) atol=1e-3

            # Outside domain - should wrap
            # xi = 2π + 0.5 wraps to 0.5
            @test linear_interp(x, y, 2π + 0.5; bc=:periodic) ≈ linear_interp(x, y, 0.5; bc=:periodic) atol=1e-10

            # Negative - should wrap
            # xi = -0.5 wraps to 2π - 0.5
            @test linear_interp(x, y, -0.5; bc=:periodic) ≈ linear_interp(x, y, 2π - 0.5; bc=:periodic) atol=1e-10

            # Multiple periods
            @test linear_interp(x, y, 4π + 1.0; bc=:periodic) ≈ linear_interp(x, y, 1.0; bc=:periodic) atol=1e-10
        end

        @testset "Vector interface" begin
            x_query = [0.5, 2π + 0.5, -0.5, 4π + 1.0]
            result = linear_interp(x, y, x_query; bc=:periodic)

            @test result[1] ≈ sin(0.5) atol=1e-3
            @test result[2] ≈ sin(0.5) atol=1e-3  # 2π + 0.5 wraps to 0.5
            @test result[3] ≈ sin(2π - 0.5) atol=1e-3  # -0.5 wraps to 2π - 0.5 = -sin(0.5)
        end

        @testset "In-place interface" begin
            out = Vector{Float64}(undef, 3)
            x_query = [0.5, 2π + 0.5, -0.5]
            linear_interp!(out, x, y, x_query; bc=:periodic)

            @test out[1] ≈ sin(0.5) atol=1e-3
            @test out[2] ≈ out[1] atol=1e-10  # Wrapped to same position
        end

        @testset "LinearInterpolant with periodic BC" begin
            itp = linear_interp(x, y; bc=:periodic)

            # Test scalar calls
            @test itp(0.5) ≈ sin(0.5) atol=1e-3
            @test itp(2π + 0.5) ≈ itp(0.5) atol=1e-10

            # Test vector calls
            vals = itp.([0.5, 2π + 0.5, -0.5])
            @test vals[1] ≈ vals[2] atol=1e-10
        end

        @testset "Continuity at boundary" begin
            # Check values just before and after boundary
            ε = 1e-6
            val_before = linear_interp(x, y, 2π - ε; bc=:periodic)
            val_after = linear_interp(x, y, 0.0 + ε; bc=:periodic)

            # For sin, both should be close to 0
            @test abs(val_before) < 1e-3
            @test abs(val_after) < 1e-3
        end
    end

    @testset "Cubic Periodic BC" begin
        # Periodic function: sin on [0, 2π]
        N = 101
        x = range(0.0, 2π, N)
        y = sin.(x)

        @testset "Basic cache construction" begin
            cache = CubicSplineCache(x; bc=:periodic)
            @test cache.bc_data isa FastInterpolations.PeriodicData
            @test cache.bc_data.period ≈ 2π atol=1e-10
        end

        @testset "Basic interpolation" begin
            cache = CubicSplineCache(x; bc=:periodic)

            # Interior point - should match sin closely
            result = cubic_interp(cache, y, [π/4])
            @test result[1] ≈ sin(π/4) atol=1e-4

            # Outside domain - should wrap
            result_wrapped = cubic_interp(cache, y, [2π + 0.5])
            result_interior = cubic_interp(cache, y, [0.5])
            @test result_wrapped[1] ≈ result_interior[1] atol=1e-10
        end

        @testset "C2 Continuity at boundary" begin
            cache = CubicSplineCache(x; bc=:periodic)

            # Evaluate at points near the boundary
            ε = 1e-4
            x_near_boundary = [-ε, 0.0, ε, 2π - ε, 2π, 2π + ε]
            result = cubic_interp(cache, y, x_near_boundary)

            # Values should be continuous and smooth
            # sin(0) = 0, sin(2π) = 0, sin(-ε) ≈ sin(2π - ε)
            @test abs(result[2]) < 1e-4  # sin(0) ≈ 0
            @test abs(result[5]) < 1e-4  # sin(2π) ≈ 0

            # Wrapped values should match
            @test result[1] ≈ result[4] atol=1e-4  # -ε wraps to 2π - ε
            @test result[3] ≈ result[6] atol=1e-4  # ε and 2π + ε should match

            # Check numerical derivatives for smoothness
            # Using finite differences to approximate first derivative
            h = 1e-5
            deriv_before = (cubic_interp(cache, y, [2π - h/2])[1] - cubic_interp(cache, y, [2π - 3h/2])[1]) / h
            deriv_after = (cubic_interp(cache, y, [h/2])[1] - cubic_interp(cache, y, [-h/2])[1]) / h

            # cos(0) = 1, so derivative at boundary should be close to 1
            @test deriv_before ≈ 1.0 atol=0.1
            @test deriv_after ≈ 1.0 atol=0.1
        end

        @testset "True C2 continuity (S' and S'' match at boundaries)" begin
            # Use denser grid for more accurate derivative estimates
            N_dense = 201
            x_dense = range(0.0, 2π, N_dense)
            y_dense = sin.(x_dense)
            cache = CubicSplineCache(x_dense; bc=:periodic)

            # Finite difference parameters
            h = 1e-6

            # Helper to evaluate spline
            f(t) = cubic_interp(cache, y_dense, [t])[1]

            # ===== C0: Value continuity =====
            # S(x_min) should equal S(x_max) (wrapped)
            val_start = f(0.0)
            val_end = f(2π - h)  # Just before end
            @test val_start ≈ 0.0 atol=1e-4  # sin(0) = 0
            @test val_end ≈ 0.0 atol=1e-3    # sin(2π) ≈ 0

            # ===== C1: First derivative continuity =====
            # S'(x_min⁺) ≈ S'(x_max⁻) using central differences
            # At boundary: approaching from left of 2π vs right of 0
            deriv1_left = (f(2π - h) - f(2π - 2h)) / h   # S'(x_max⁻)
            deriv1_right = (f(h) - f(0.0)) / h           # S'(x_min⁺)

            # For sin(x): S'(0) = cos(0) = 1
            @test deriv1_left ≈ deriv1_right atol=0.01   # S'(x₁) = S'(xₙ)
            @test deriv1_left ≈ 1.0 atol=0.1             # Should be cos(0) = 1

            # ===== C2: Second derivative continuity =====
            # S''(x_min⁺) ≈ S''(x_max⁻) using central differences
            deriv2_left = (f(2π - h) - 2*f(2π - 2h) + f(2π - 3h)) / h^2
            deriv2_right = (f(2h) - 2*f(h) + f(0.0)) / h^2

            # For sin(x): S''(0) = -sin(0) = 0
            @test deriv2_left ≈ deriv2_right atol=0.5    # S''(x₁) = S''(xₙ) - KEY C2 TEST
            @test abs(deriv2_left) < 1.0                 # Should be -sin(0) ≈ 0

            # ===== Compare with natural BC (should differ at boundaries) =====
            cache_natural = CubicSplineCache(x_dense; bc=:natural)
            f_nat(t) = cubic_interp(cache_natural, y_dense, [t])[1]

            # Natural BC forces S''(x_min) = S''(x_max) = 0, which matches sin(x)
            # but the derivatives approaching the boundary may differ
            # For a true periodic function, periodic BC should be more accurate overall
            deriv2_natural_left = (f_nat(2π - h) - 2*f_nat(2π - 2h) + f_nat(2π - 3h)) / h^2
            deriv2_natural_right = (f_nat(2h) - 2*f_nat(h) + f_nat(0.0)) / h^2

            # Natural BC also happens to give S''≈0 at boundaries for sin, so check interior accuracy
            # At x = π, sin''(π) = -sin(π) = 0, both should match well
            π_f = Float64(π)
            deriv2_periodic_mid = (f(π_f + h) - 2*f(π_f) + f(π_f - h)) / h^2
            deriv2_natural_mid = (f_nat(π_f + h) - 2*f_nat(π_f) + f_nat(π_f - h)) / h^2

            @test deriv2_periodic_mid ≈ 0.0 atol=0.1  # sin''(π) = 0
            @test deriv2_natural_mid ≈ 0.0 atol=0.1   # Both should work for interior
        end

        @testset "CubicInterpolant with periodic cache" begin
            cache = CubicSplineCache(x; bc=:periodic)
            itp = cubic_interp(cache, y)

            # Test scalar calls
            @test itp(π/4) ≈ sin(π/4) atol=1e-4
            @test itp(2π + 0.5) ≈ itp(0.5) atol=1e-10

            # Test vector calls
            vals = itp.([0.5, 2π + 0.5, -0.5])
            @test vals[1] ≈ vals[2] atol=1e-10
        end

        @testset "Multiple periods wrapping" begin
            cache = CubicSplineCache(x; bc=:periodic)

            # Test multiple full periods
            x_query = [0.5, 2π + 0.5, 4π + 0.5, 6π + 0.5, -2π + 0.5]
            result = cubic_interp(cache, y, x_query)

            @test result[1] ≈ result[2] atol=1e-10
            @test result[1] ≈ result[3] atol=1e-10
            @test result[1] ≈ result[4] atol=1e-10
            @test result[1] ≈ result[5] atol=1e-10
        end
    end

    @testset "Periodic BC vs Natural BC" begin
        N = 51
        x = range(0.0, 2π, N)
        y = sin.(x)

        cache_natural = CubicSplineCache(x; bc=:natural)
        cache_periodic = CubicSplineCache(x; bc=:periodic)

        # Interior values should be similar
        x_interior = [π/4, π/2, π, 3π/2]
        result_natural = cubic_interp(cache_natural, y, x_interior)
        result_periodic = cubic_interp(cache_periodic, y, x_interior)

        # Both should approximate sin well in interior
        for (r_nat, r_per, xi) in zip(result_natural, result_periodic, x_interior)
            @test r_nat ≈ sin(xi) atol=1e-3
            @test r_per ≈ sin(xi) atol=1e-3
        end
    end

    @testset "Non-uniform grid periodic" begin
        # Non-uniform but valid periodic grid
        x_base = sort([0.0, 0.3, 0.7, 1.2, 1.8, 2π])  # Non-uniform
        y_base = sin.(x_base)

        # Linear periodic should still work
        @test linear_interp(x_base, y_base, 2π + 0.5; bc=:periodic) ≈ linear_interp(x_base, y_base, 0.5; bc=:periodic) atol=1e-10

        # Cubic periodic should work with Vector grid
        cache = CubicSplineCache(collect(x_base); bc=:periodic)
        result1 = cubic_interp(cache, y_base, [0.5])
        result2 = cubic_interp(cache, y_base, [2π + 0.5])
        @test result1[1] ≈ result2[1] atol=1e-10
    end

end
