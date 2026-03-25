using Test
using FastInterpolations

@testset "Periodic BC Tests" begin

    @testset "Linear Wrap Extrapolation" begin
        # Periodic function: sin on [0, 2π]
        N = 101
        x = range(0.0, 2π, N)
        y = sin.(x)

        @testset "Basic wrapping" begin
            # Interior point
            @test linear_interp(x, y, π / 4; extrap = WrapExtrap()) ≈ sin(π / 4) atol = 1.0e-3

            # Outside domain - should wrap
            # xi = 2π + 0.5 wraps to 0.5
            @test linear_interp(x, y, 2π + 0.5; extrap = WrapExtrap()) ≈ linear_interp(x, y, 0.5; extrap = WrapExtrap()) atol = 1.0e-10

            # Negative - should wrap
            # xi = -0.5 wraps to 2π - 0.5
            @test linear_interp(x, y, -0.5; extrap = WrapExtrap()) ≈ linear_interp(x, y, 2π - 0.5; extrap = WrapExtrap()) atol = 1.0e-10

            # Multiple periods
            @test linear_interp(x, y, 4π + 1.0; extrap = WrapExtrap()) ≈ linear_interp(x, y, 1.0; extrap = WrapExtrap()) atol = 1.0e-10
        end

        @testset "Vector interface" begin
            x_query = [0.5, 2π + 0.5, -0.5, 4π + 1.0]
            result = linear_interp(x, y, x_query; extrap = WrapExtrap())

            @test result[1] ≈ sin(0.5) atol = 1.0e-3
            @test result[2] ≈ sin(0.5) atol = 1.0e-3  # 2π + 0.5 wraps to 0.5
            @test result[3] ≈ sin(2π - 0.5) atol = 1.0e-3  # -0.5 wraps to 2π - 0.5 = -sin(0.5)

            # Fast path: all queries inside domain [x_min, x_max) → uses extension path
            inside_query = [0.5, 1.0, 2.0]
            inside_result = linear_interp(x, y, inside_query; extrap = WrapExtrap())
            @test inside_result ≈ sin.(inside_query) atol = 1.0e-3
        end

        @testset "In-place interface" begin
            out = Vector{Float64}(undef, 3)
            x_query = [0.5, 2π + 0.5, -0.5]
            linear_interp!(out, x, y, x_query; extrap = WrapExtrap())

            @test out[1] ≈ sin(0.5) atol = 1.0e-3
            @test out[2] ≈ out[1] atol = 1.0e-10  # Wrapped to same position
        end

        @testset "LinearInterpolant with wrap extrap" begin
            itp = linear_interp(x, y; extrap = WrapExtrap())

            # Test scalar calls
            @test itp(0.5) ≈ sin(0.5) atol = 1.0e-3
            @test itp(2π + 0.5) ≈ itp(0.5) atol = 1.0e-10

            # Test vector calls
            vals = itp.([0.5, 2π + 0.5, -0.5])
            @test vals[1] ≈ vals[2] atol = 1.0e-10
        end

        @testset "Continuity at boundary" begin
            # Check values just before and after boundary
            ε = 1.0e-6
            val_before = linear_interp(x, y, 2π - ε; extrap = WrapExtrap())
            val_after = linear_interp(x, y, 0.0 + ε; extrap = WrapExtrap())

            # For sin, both should be close to 0
            @test abs(val_before) < 1.0e-3
            @test abs(val_after) < 1.0e-3
        end
    end

    @testset "Cubic Zero-Curvature BC with Wrap Extrapolation" begin
        # bc=ZeroCurvBC() uses zero-curvature BC coefficients, but extrap=WrapExtrap() wraps coordinates
        # Unlike bc=PeriodicBC(), this does NOT check y[1] ≈ y[end]

        N = 101
        x = range(0.0, 2π, N)
        y_sin = sin.(x)
        y_sin[end] = y_sin[1]  # Ensure exact periodicity for PeriodicBC tests below

        @testset "Basic wrapping with ZeroCurv BC" begin
            # Interior point
            @test cubic_interp(x, y_sin, π / 4; bc = ZeroCurvBC(), extrap = WrapExtrap()) ≈ sin(π / 4) atol = 1.0e-4

            # Outside domain - should wrap
            val_in = cubic_interp(x, y_sin, 0.5; bc = ZeroCurvBC(), extrap = WrapExtrap())
            val_wrapped = cubic_interp(x, y_sin, 2π + 0.5; bc = ZeroCurvBC(), extrap = WrapExtrap())
            @test val_in ≈ val_wrapped atol = 1.0e-10

            # Negative - should wrap
            val_neg = cubic_interp(x, y_sin, -0.5; bc = ZeroCurvBC(), extrap = WrapExtrap())
            val_equiv = cubic_interp(x, y_sin, 2π - 0.5; bc = ZeroCurvBC(), extrap = WrapExtrap())
            @test val_neg ≈ val_equiv atol = 1.0e-10

            # Multiple periods
            @test cubic_interp(x, y_sin, 4π + 1.0; bc = ZeroCurvBC(), extrap = WrapExtrap()) ≈ cubic_interp(x, y_sin, 1.0; bc = ZeroCurvBC(), extrap = WrapExtrap()) atol = 1.0e-10
        end

        @testset "Vector and in-place interface" begin
            x_query = [0.5, 2π + 0.5, -0.5, 4π + 1.0]
            result = cubic_interp(x, y_sin, x_query; bc = ZeroCurvBC(), extrap = WrapExtrap())

            @test result[1] ≈ sin(0.5) atol = 1.0e-4
            @test result[2] ≈ result[1] atol = 1.0e-10  # 2π + 0.5 wraps to 0.5

            # In-place
            cache = CubicSplineCache(x; bc = ZeroCurvBC())
            out = similar(result)
            cubic_interp!(out, cache, collect(y_sin), x_query; extrap = WrapExtrap())
            @test out ≈ result atol = 1.0e-10
        end

        @testset "CubicInterpolant with wrap extrap" begin
            itp = cubic_interp(x, y_sin; bc = ZeroCurvBC(), extrap = WrapExtrap())

            # Test scalar calls
            @test itp(π / 4) ≈ sin(π / 4) atol = 1.0e-4
            @test itp(2π + 0.5) ≈ itp(0.5) atol = 1.0e-10

            # Test vector calls
            vals = itp.([0.5, 2π + 0.5, -0.5])
            @test vals[1] ≈ vals[2] atol = 1.0e-10
        end

        @testset "Sawtooth/triangle wave pattern (y[1] != y[end])" begin
            # Linear ramp: y[1] = 0, y[end] = 2π (NOT equal)
            # This is the key use case for bc=ZeroCurvBC() + extrap=WrapExtrap()
            x_ramp = range(0.0, 1.0, 51)
            y_ramp = collect(x_ramp)  # [0, 0.02, ..., 1.0]

            @test y_ramp[1] != y_ramp[end]  # Confirm endpoints differ

            # Should NOT throw (unlike bc=PeriodicBC() which requires y[1] ≈ y[end])
            itp = cubic_interp(x_ramp, y_ramp; bc = ZeroCurvBC(), extrap = WrapExtrap())

            # Test wrap behavior
            @test itp(0.5) ≈ 0.5 atol = 1.0e-4
            @test itp(1.5) ≈ itp(0.5) atol = 1.0e-10  # 1.5 wraps to 0.5
            @test itp(-0.5) ≈ itp(0.5) atol = 1.0e-10  # -0.5 wraps to 0.5

            # At boundary: discontinuity expected (wraps from ~1 to ~0)
            ε = 1.0e-10
            val_before = itp(1.0 - ε)
            val_after = itp(1.0 + ε)  # wraps to ε

            @test val_before ≈ 1.0 atol = 1.0e-3  # Just before end
            @test val_after ≈ 0.0 atol = 1.0e-3   # Wraps to beginning
        end

        @testset "Comparison: bc=ZeroCurvBC()+wrap vs bc=PeriodicBC()" begin
            # For truly periodic functions (sin), both give similar results
            # but they use DIFFERENT spline coefficients (different BC)

            # Create interpolants with different BCs
            itp_nat = cubic_interp(x, y_sin; bc = ZeroCurvBC(), extrap = WrapExtrap())
            itp_per = cubic_interp(x, y_sin; bc = PeriodicBC())

            # Interior values are similar for periodic functions
            @test itp_nat(π / 2) ≈ itp_per(π / 2) atol = 1.0e-3
            @test itp_nat(π) ≈ itp_per(π) atol = 1.0e-3

            # Wrap behavior is the same
            @test itp_nat(2π + 0.5) ≈ itp_per(2π + 0.5) atol = 1.0e-3

            # However, at boundaries the second derivatives differ:
            # - ZeroCurv BC forces S''(x₁) = S''(xₙ) = 0
            # - periodic BC forces S''(x₁) = S''(xₙ) and continuity
            # For sin(x), ZeroCurv BC happens to be close, but they're computed differently
        end

        @testset "bc=PeriodicBC() ignores extrap parameter" begin
            # When bc=PeriodicBC(), extrap is ignored (always wraps)
            cache = CubicSplineCache(x; bc = PeriodicBC())

            # All extrap values should give the same result
            itp_none = cubic_interp(cache, collect(y_sin); extrap = NoExtrap())
            itp_const = cubic_interp(cache, collect(y_sin); extrap = ClampExtrap())
            itp_ext = cubic_interp(cache, collect(y_sin); extrap = ExtendExtrap())
            itp_wrap = cubic_interp(cache, collect(y_sin); extrap = WrapExtrap())

            # Test outside domain - all should wrap
            xi_outside = 2π + 0.5
            @test itp_none(xi_outside) ≈ itp_wrap(xi_outside) atol = 1.0e-10
            @test itp_const(xi_outside) ≈ itp_wrap(xi_outside) atol = 1.0e-10
            @test itp_ext(xi_outside) ≈ itp_wrap(xi_outside) atol = 1.0e-10
        end
    end

    @testset "Cubic Periodic BC" begin
        # Periodic function: sin on [0, 2π]
        N = 101
        x = range(0.0, 2π, N)
        y = sin.(x)
        y[end] = y[1]  # Ensure exact periodicity (strict == check)

        @testset "Basic cache construction" begin
            cache = CubicSplineCache(x; bc = PeriodicBC())
            @test cache.bc_config isa FastInterpolations.PeriodicData
            @test cache.bc_config.period ≈ 2π atol = 1.0e-10
        end

        @testset "Basic interpolation" begin
            cache = CubicSplineCache(x; bc = PeriodicBC())

            # Interior point - should match sin closely
            result = cubic_interp(cache, y, [π / 4])
            @test result[1] ≈ sin(π / 4) atol = 1.0e-4

            # Outside domain - should wrap
            result_wrapped = cubic_interp(cache, y, [2π + 0.5])
            result_interior = cubic_interp(cache, y, [0.5])
            @test result_wrapped[1] ≈ result_interior[1] atol = 1.0e-10
        end

        @testset "C2 Continuity at boundary" begin
            cache = CubicSplineCache(x; bc = PeriodicBC())

            # Evaluate at points near the boundary
            ε = 1.0e-4
            x_near_boundary = [-ε, 0.0, ε, 2π - ε, 2π, 2π + ε]
            result = cubic_interp(cache, y, x_near_boundary)

            # Values should be continuous and smooth
            # sin(0) = 0, sin(2π) = 0, sin(-ε) ≈ sin(2π - ε)
            @test abs(result[2]) < 1.0e-4  # sin(0) ≈ 0
            @test abs(result[5]) < 1.0e-4  # sin(2π) ≈ 0

            # Wrapped values should match
            @test result[1] ≈ result[4] atol = 1.0e-4  # -ε wraps to 2π - ε
            @test result[3] ≈ result[6] atol = 1.0e-4  # ε and 2π + ε should match

            # Check numerical derivatives for smoothness
            # Using finite differences to approximate first derivative
            h = 1.0e-5
            deriv_before = (cubic_interp(cache, y, [2π - h / 2])[1] - cubic_interp(cache, y, [2π - 3h / 2])[1]) / h
            deriv_after = (cubic_interp(cache, y, [h / 2])[1] - cubic_interp(cache, y, [-h / 2])[1]) / h

            # cos(0) = 1, so derivative at boundary should be close to 1
            @test deriv_before ≈ 1.0 atol = 0.1
            @test deriv_after ≈ 1.0 atol = 0.1
        end

        @testset "True C2 continuity (S' and S'' match at boundaries)" begin
            # Use denser grid for more accurate derivative estimates
            N_dense = 201
            x_dense = range(0.0, 2π, N_dense)
            y_dense = sin.(x_dense)
            y_dense[end] = y_dense[1]  # Ensure exact periodicity
            cache = CubicSplineCache(x_dense; bc = PeriodicBC())

            # Finite difference parameters
            h = 1.0e-6

            # Helper to evaluate spline
            f(t) = cubic_interp(cache, y_dense, [t])[1]

            # ===== C0: Value continuity =====
            # S(x_min) should equal S(x_max) (wrapped)
            val_start = f(0.0)
            val_end = f(2π - h)  # Just before end
            @test val_start ≈ 0.0 atol = 1.0e-4  # sin(0) = 0
            @test val_end ≈ 0.0 atol = 1.0e-3    # sin(2π) ≈ 0

            # ===== C1: First derivative continuity =====
            # S'(x_min⁺) ≈ S'(x_max⁻) using central differences
            # At boundary: approaching from left of 2π vs right of 0
            deriv1_left = (f(2π - h) - f(2π - 2h)) / h   # S'(x_max⁻)
            deriv1_right = (f(h) - f(0.0)) / h           # S'(x_min⁺)

            # For sin(x): S'(0) = cos(0) = 1
            @test deriv1_left ≈ deriv1_right atol = 0.01   # S'(x₁) = S'(xₙ)
            @test deriv1_left ≈ 1.0 atol = 0.1             # Should be cos(0) = 1

            # ===== C2: Second derivative continuity =====
            # S''(x_min⁺) ≈ S''(x_max⁻) using central differences
            deriv2_left = (f(2π - h) - 2 * f(2π - 2h) + f(2π - 3h)) / h^2
            deriv2_right = (f(2h) - 2 * f(h) + f(0.0)) / h^2

            # For sin(x): S''(0) = -sin(0) = 0
            @test deriv2_left ≈ deriv2_right atol = 0.5    # S''(x₁) = S''(xₙ) - KEY C2 TEST
            @test abs(deriv2_left) < 1.0                 # Should be -sin(0) ≈ 0

            # ===== Compare with ZeroCurv BC (should differ at boundaries) =====
            cache_natural = CubicSplineCache(x_dense; bc = ZeroCurvBC())
            f_nat(t) = cubic_interp(cache_natural, y_dense, [t])[1]

            # Zero-Curvature BC forces S''(x_min) = S''(x_max) = 0, which matches sin(x)
            # but the derivatives approaching the boundary may differ
            # For a true periodic function, periodic BC should be more accurate overall
            deriv2_natural_left = (f_nat(2π - h) - 2 * f_nat(2π - 2h) + f_nat(2π - 3h)) / h^2
            deriv2_natural_right = (f_nat(2h) - 2 * f_nat(h) + f_nat(0.0)) / h^2

            # Zero-Curvature BC also happens to give S''≈0 at boundaries for sin, so check interior accuracy
            # At x = π, sin''(π) = -sin(π) = 0, both should match well
            π_f = Float64(π)
            deriv2_periodic_mid = (f(π_f + h) - 2 * f(π_f) + f(π_f - h)) / h^2
            deriv2_natural_mid = (f_nat(π_f + h) - 2 * f_nat(π_f) + f_nat(π_f - h)) / h^2

            @test deriv2_periodic_mid ≈ 0.0 atol = 0.1  # sin''(π) = 0
            @test deriv2_natural_mid ≈ 0.0 atol = 0.1   # Both should work for interior
        end

        @testset "CubicInterpolant with periodic cache" begin
            cache = CubicSplineCache(x; bc = PeriodicBC())
            itp = cubic_interp(cache, y)

            # Test scalar calls
            @test itp(π / 4) ≈ sin(π / 4) atol = 1.0e-4
            @test itp(2π + 0.5) ≈ itp(0.5) atol = 1.0e-10

            # Test vector calls
            vals = itp.([0.5, 2π + 0.5, -0.5])
            @test vals[1] ≈ vals[2] atol = 1.0e-10
        end

        @testset "Multiple periods wrapping" begin
            cache = CubicSplineCache(x; bc = PeriodicBC())

            # Test multiple full periods
            x_query = [0.5, 2π + 0.5, 4π + 0.5, 6π + 0.5, -2π + 0.5]
            result = cubic_interp(cache, y, x_query)

            @test result[1] ≈ result[2] atol = 1.0e-10
            @test result[1] ≈ result[3] atol = 1.0e-10
            @test result[1] ≈ result[4] atol = 1.0e-10
            @test result[1] ≈ result[5] atol = 1.0e-10
        end
    end

    @testset "Periodic BC vs Zero-Curvature BC" begin
        N = 51
        x = range(0.0, 2π, N)
        y = sin.(x)
        y[end] = y[1]  # Ensure exact periodicity

        cache_natural = CubicSplineCache(x; bc = ZeroCurvBC())
        cache_periodic = CubicSplineCache(x; bc = PeriodicBC())

        # Interior values should be similar
        x_interior = [π / 4, π / 2, π, 3π / 2]
        result_natural = cubic_interp(cache_natural, y, x_interior)
        result_periodic = cubic_interp(cache_periodic, y, x_interior)

        # Both should approximate sin well in interior
        for (r_nat, r_per, xi) in zip(result_natural, result_periodic, x_interior)
            @test r_nat ≈ sin(xi) atol = 1.0e-3
            @test r_per ≈ sin(xi) atol = 1.0e-3
        end
    end

    @testset "Non-uniform grid periodic" begin
        # Non-uniform but valid periodic grid
        x_base = sort([0.0, 0.3, 0.7, 1.2, 1.8, 2π])  # Non-uniform
        y_base = sin.(x_base)
        y_base[end] = y_base[1]  # Ensure exact periodicity

        # Linear wrap should still work
        @test linear_interp(x_base, y_base, 2π + 0.5; extrap = WrapExtrap()) ≈ linear_interp(x_base, y_base, 0.5; extrap = WrapExtrap()) atol = 1.0e-10

        # Cubic periodic should work with Vector grid
        cache = CubicSplineCache(collect(x_base); bc = PeriodicBC())
        result1 = cubic_interp(cache, y_base, [0.5])
        result2 = cubic_interp(cache, y_base, [2π + 0.5])
        @test result1[1] ≈ result2[1] atol = 1.0e-10
    end

    @testset "_check_periodic_endpoints validation (Cubic only)" begin
        # NOTE: Linear interpolation with extrap=WrapExtrap() does NOT check endpoints!
        # Only cubic bc=PeriodicBC() checks that y[1] ≈ y[end] (isapprox for _PromotableValue)
        x = range(0.0, 2π, 101)

        @testset "Valid periodic data — exact endpoints (Float64)" begin
            y_sin = collect(sin.(x))
            y_sin[end] = y_sin[1]  # Exact equality
            @test y_sin[1] == y_sin[end]

            @test linear_interp(x, y_sin, 0.5; extrap = WrapExtrap()) isa Float64
            @test cubic_interp(x, y_sin, 0.5; bc = PeriodicBC()) isa Float64

            # cos(0) = cos(2π) = 1.0 exactly in Float64
            y_cos = collect(cos.(x))
            @test cubic_interp(x, y_cos, 0.5; bc = PeriodicBC()) isa Float64
        end

        @testset "Valid periodic data — exact endpoints (Float32)" begin
            x_f32 = range(0.0f0, 2.0f0 * Float32(π), 101)
            y_f32 = collect(sin.(x_f32))
            y_f32[end] = y_f32[1]  # Exact equality
            @test y_f32[1] == y_f32[end]

            @test linear_interp(x_f32, y_f32, 0.5f0; extrap = WrapExtrap()) isa Float32
            @test cubic_interp(x_f32, y_f32, 0.5f0; bc = PeriodicBC()) isa Float32
        end

        @testset "Approximate endpoints accepted via isapprox (_PromotableValue)" begin
            # cos(0) vs cos(2π) — both 1.0, isapprox passes (non-zero, rtol works)
            y_cos = cos.(x)
            @test isapprox(y_cos[1], y_cos[end])
            @test cubic_interp(x, y_cos, 0.5; bc = PeriodicBC()) isa Float64

            # Float32 cos — same story
            x_f32 = range(0.0f0, 2.0f0 * Float32(π), 101)
            y_cos_f32 = cos.(x_f32)
            @test isapprox(y_cos_f32[1], y_cos_f32[end])
            @test cubic_interp(x_f32, y_cos_f32, 0.5f0; bc = PeriodicBC()) isa Float32
        end

        @testset "Near-zero endpoints accepted via atol = 8eps" begin
            # sin(0) vs sin(2π): diff ≈ 1.1 eps, covered by atol = 8eps noise floor
            y_sin_raw = sin.(x)
            @test !isapprox(y_sin_raw[1], y_sin_raw[end])  # default isapprox fails
            @test cubic_interp(x, y_sin_raw, 0.5; bc = PeriodicBC()) isa Float64  # but 8eps atol saves it

            # Float32 sin
            x_f32 = range(0.0f0, 2.0f0 * Float32(π), 101)
            y_sin_f32 = sin.(x_f32)
            @test cubic_interp(x_f32, y_sin_f32, 0.5f0; bc = PeriodicBC()) isa Float32
        end

        @testset "Large-magnitude endpoints — rtol covers relative noise" begin
            # When values are far from zero, the absolute diff can exceed 8eps(T)
            # while remaining tiny *relative* to the magnitude.  rtol = √eps handles this.
            #
            # Inject noise at ~1e-14 relative scale (well within √eps ≈ 1.5e-8,
            # but 1001 * 1e-14 ≈ 1e-11 >> 8eps ≈ 1.8e-15 so atol alone would reject).

            # Float64
            y_large = collect(1000.0 .+ cos.(x))
            y_large[end] = y_large[1] * (1.0 + 1.0e-14)
            @test abs(y_large[1] - y_large[end]) > 8 * eps(Float64)  # atol alone would reject
            @test cubic_interp(x, y_large, 0.5; bc = PeriodicBC()) isa Float64

            # Float32: inject noise at ~1e-6 relative scale (within √eps(F32) ≈ 3.5e-4)
            x_f32 = range(0.0f0, 2.0f0 * Float32(π), 101)
            y_large_f32 = collect(1000.0f0 .+ cos.(x_f32))
            y_large_f32[end] = y_large_f32[1] * (1.0f0 + 1.0f-6)
            @test abs(y_large_f32[1] - y_large_f32[end]) > 8 * eps(Float32)
            @test cubic_interp(x_f32, y_large_f32, 0.5f0; bc = PeriodicBC()) isa Float32
        end

        @testset "Scaled near-zero — atol=8eps not enough, requires y[end]=y[1] or check=false" begin
            # 1e6 * sin(x): noise ≈ 1e6 * eps, exceeds 8eps floor
            y_scaled = 1.0e6 .* sin.(x)
            @test_throws ArgumentError cubic_interp(x, y_scaled, 0.5; bc = PeriodicBC())

            # Fix 1: set endpoint explicitly
            y_fixed = copy(y_scaled)
            y_fixed[end] = y_fixed[1]
            @test cubic_interp(x, y_fixed, 0.5; bc = PeriodicBC()) isa Float64

            # Fix 2: skip check via check=false
            @test cubic_interp(x, y_scaled, 0.5; bc = PeriodicBC(check = false)) isa Float64
        end

        @testset "Clearly different endpoints — rejected" begin
            y_tiny = collect(cos.(x))
            y_tiny[end] = y_tiny[1] + 1.0e-6  # Well beyond isapprox tolerance
            @test_throws ArgumentError cubic_interp(x, y_tiny, 0.5; bc = PeriodicBC())
        end

        @testset "Non-matching endpoints — Cubic throws, Linear wrap works" begin
            # Non-periodic data: y[1] != y[end] (sawtooth wave use case)
            y_invalid = collect(x)  # Linear function: y[1] = 0, y[end] = 2π
            @test y_invalid[1] != y_invalid[end]

            # Linear wrap does NOT check endpoints — works fine (sawtooth pattern)
            @test linear_interp(x, y_invalid, 0.5; extrap = WrapExtrap()) isa Float64
            @test LinearInterpolant(collect(x), y_invalid; extrap = WrapExtrap()) isa LinearInterpolant

            # Cubic bc=PeriodicBC() DOES check endpoints — throws ArgumentError
            @test_throws ArgumentError cubic_interp(x, y_invalid, 0.5; bc = PeriodicBC())
            @test_throws ArgumentError cubic_interp(collect(x), y_invalid; bc = PeriodicBC())
        end

        @testset "Error message contains useful info and tip" begin
            y_invalid = collect(x)

            try
                cubic_interp(x, y_invalid, 0.5; bc = PeriodicBC())
                @test false  # Should not reach here
            catch e
                @test e isa ArgumentError
                msg = e.msg
                @test occursin("PeriodicBC", msg)
                @test occursin("y[1]", msg)
                @test occursin("y[end]", msg)
                @test occursin("check=false", msg)  # Helpful tip
            end
        end

        @testset "ND large-magnitude endpoints — rtol covers relative noise" begin
            # Same as 1D test but for ND _check_periodic_data_noalloc! path
            x = range(0.0, 2π, 31)
            y = range(0.0, 2π, 21)
            data = [1000.0 + cos(xi) * cos(yj) for xi in x, yj in y]
            # Inject relative noise on periodic boundaries (dim 1: first/last row)
            for j in axes(data, 2)
                data[end, j] = data[1, j] * (1.0 + 1.0e-14)
            end
            @test abs(data[1, 1] - data[end, 1]) > 8 * eps(Float64)
            bc = (PeriodicBC(), CubicFit())
            @test cubic_interp((x, y), data, (0.5, 0.5); bc = bc) isa Float64
        end

        @testset "PeriodicBC(check=false) — type stability (@inferred)" begin
            using Test: @inferred
            x_r = range(0.0, 2π, 101)
            y_scaled = 1.0e6 .* sin.(x_r)
            # check=false must not introduce type instability or allocation
            bc_nocheck = PeriodicBC(check = false)
            @test @inferred(cubic_interp(x_r, y_scaled, 0.5; bc = bc_nocheck)) isa Float64

            # check=true (default) with valid data
            y_cos = cos.(x_r)
            bc_check = PeriodicBC()
            @test @inferred(cubic_interp(x_r, y_cos, 0.5; bc = bc_check)) isa Float64

            # Interpolant construction also type-stable
            @test @inferred(cubic_interp(collect(x_r), y_cos; bc = bc_check)) isa CubicInterpolant
            @test @inferred(cubic_interp(collect(x_r), y_scaled; bc = bc_nocheck)) isa CubicInterpolant
        end
    end

end
