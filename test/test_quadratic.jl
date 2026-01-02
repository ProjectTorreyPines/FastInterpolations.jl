# Tests for quadratic (C1 piecewise quadratic) spline interpolation
#
# This file follows TDD: tests are written BEFORE implementation.
# Phase 1: BC Tags (Left/Right types)
# Phase 2: Kernels + Coefficient Computation
# Phase 3: Public API + QuadraticInterpolant

# ============================================================================
# Group 1: BC Type Tests (Phase 1)
# ============================================================================
@testset "Quadratic Interpolation - BC Types" begin

    @testset "Left BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Left(Deriv1(0.5))
            @test bc1 isa Left{Float64, Deriv1{Float64}}

            bc2 = Left(Deriv2(1.0))
            @test bc2 isa Left{Float64, Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Left(Deriv2(1.0f0))
            @test bc isa Left{Float32, Deriv2{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Left(Deriv1(1))
            @test bc isa Left{Float64, Deriv1{Float64}}
        end

        @testset "accessor (inner BC value)" begin
            bc = Left(Deriv1(0.5))
            @test bc.bc.val == 0.5

            bc2 = Left(Deriv2(2.0))
            @test bc2.bc.val == 2.0
        end

        @testset "type stability" begin
            @test @inferred(Left(Deriv1(0.0))) isa Left
            @test @inferred(Left(Deriv2(1.0))) isa Left
        end

        @testset "subtype relationship" begin
            @test Left{Float64, Deriv1{Float64}} <: AbstractBC{Float64}
            @test Left{Float64, Deriv2{Float64}} <: AbstractBC{Float64}
            @test Left{Float32, Deriv1{Float32}} <: AbstractBC{Float32}
        end
    end

    @testset "Right BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Right(Deriv1(-0.5))
            @test bc1 isa Right{Float64, Deriv1{Float64}}

            bc2 = Right(Deriv2(0.0))
            @test bc2 isa Right{Float64, Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Right(Deriv1(1.0f0))
            @test bc isa Right{Float32, Deriv1{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Right(Deriv2(0))
            @test bc isa Right{Float64, Deriv2{Float64}}
        end

        @testset "accessor (inner BC value)" begin
            bc = Right(Deriv1(-0.5))
            @test bc.bc.val == -0.5

            bc2 = Right(Deriv2(1.0))
            @test bc2.bc.val == 1.0
        end

        @testset "type stability" begin
            @test @inferred(Right(Deriv1(0.0))) isa Right
            @test @inferred(Right(Deriv2(1.0))) isa Right
        end

        @testset "subtype relationship" begin
            @test Right{Float64, Deriv1{Float64}} <: AbstractBC{Float64}
            @test Right{Float64, Deriv2{Float64}} <: AbstractBC{Float64}
            @test Right{Float32, Deriv2{Float32}} <: AbstractBC{Float32}
        end
    end

    @testset "Left/Right distinctness" begin
        # Left and Right should be distinct types
        @test Left(Deriv1(0.0)) isa Left
        @test Right(Deriv1(0.0)) isa Right
        @test !(Left(Deriv1(0.0)) isa Right)
        @test !(Right(Deriv1(0.0)) isa Left)
    end

end

# ============================================================================
# Group 2: Kernel Tests
# ============================================================================
@testset "Quadratic Interpolation - Kernels" begin
    using FastInterpolations: _quadratic_kernel, EvalValue, EvalDeriv1, EvalDeriv2

    @testset "quadratic kernel value" begin
        # S(x) = a*(x-x_i)² + d*(x-x_i) + y
        # Test f(x) = x² on [0, 1], with d=0 at x=0
        # s = (1-0)/1 = 1, a = (s-d)/h = 1
        a = 1.0
        d = 0.0
        y = 0.0

        # At dt=0.5: value = 1*0.25 + 0*0.5 + 0 = 0.25
        @test _quadratic_kernel(EvalValue(), a, d, y, 0.5) ≈ 0.25

        # At dt=0: value = 0 (at interval start)
        @test _quadratic_kernel(EvalValue(), a, d, y, 0.0) ≈ 0.0

        # At dt=1: value = 1 (at interval end)
        @test _quadratic_kernel(EvalValue(), a, d, y, 1.0) ≈ 1.0
    end

    @testset "quadratic kernel deriv1" begin
        # S'(x) = 2*a*(x-x_i) + d
        a = 1.0
        d = 0.0
        y = 0.0

        # At dt=0.5: deriv1 = 2*1*0.5 + 0 = 1.0
        @test _quadratic_kernel(EvalDeriv1(), a, d, y, 0.5) ≈ 1.0

        # At dt=0: deriv1 = d = 0
        @test _quadratic_kernel(EvalDeriv1(), a, d, y, 0.0) ≈ 0.0

        # At dt=1: deriv1 = 2*1 + 0 = 2
        @test _quadratic_kernel(EvalDeriv1(), a, d, y, 1.0) ≈ 2.0
    end

    @testset "quadratic kernel deriv2" begin
        # S''(x) = 2*a (constant)
        a = 1.0
        d = 0.0
        y = 0.0

        # deriv2 = 2*a = 2.0 at any dt
        @test _quadratic_kernel(EvalDeriv2(), a, d, y, 0.0) ≈ 2.0
        @test _quadratic_kernel(EvalDeriv2(), a, d, y, 0.5) ≈ 2.0
        @test _quadratic_kernel(EvalDeriv2(), a, d, y, 1.0) ≈ 2.0
    end

    @testset "kernel edge cases" begin
        # Constant function (a=0, d=0)
        @test _quadratic_kernel(EvalValue(), 0.0, 0.0, 5.0, 0.5) ≈ 5.0
        @test _quadratic_kernel(EvalDeriv1(), 0.0, 0.0, 5.0, 0.5) ≈ 0.0
        @test _quadratic_kernel(EvalDeriv2(), 0.0, 0.0, 5.0, 0.5) ≈ 0.0

        # Linear function (a=0, d=2)
        @test _quadratic_kernel(EvalValue(), 0.0, 2.0, 0.0, 0.5) ≈ 1.0
        @test _quadratic_kernel(EvalDeriv1(), 0.0, 2.0, 0.0, 0.5) ≈ 2.0
        @test _quadratic_kernel(EvalDeriv2(), 0.0, 2.0, 0.0, 0.5) ≈ 0.0
    end

    @testset "kernel type stability" begin
        @test @inferred(_quadratic_kernel(EvalValue(), 1.0, 0.0, 0.0, 0.5)) isa Float64
        @test @inferred(_quadratic_kernel(EvalDeriv1(), 1.0, 0.0, 0.0, 0.5)) isa Float64
        @test @inferred(_quadratic_kernel(EvalDeriv2(), 1.0, 0.0, 0.0, 0.5)) isa Float64

        # Float32
        @test @inferred(_quadratic_kernel(EvalValue(), 1.0f0, 0.0f0, 0.0f0, 0.5f0)) isa Float32
    end

end

# ============================================================================
# Group 3: Coefficient Computation Tests
# ============================================================================
@testset "Quadratic Interpolation - Coefficient Computation" begin
    using FastInterpolations: _compute_quadratic_secants!, _fill_slopes!,
                               _forward_recurrence!, _backward_recurrence!,
                               _compute_quadratic_coefficients!

    @testset "secant computation" begin
        y = [0.0, 1.0, 4.0, 9.0]  # x²
        inv_h = [1.0, 1.0, 1.0]   # uniform grid h=1
        s = zeros(3)

        _compute_quadratic_secants!(s, y, inv_h)

        # s[i] = (y[i+1] - y[i]) * inv_h[i]
        @test s[1] ≈ 1.0   # (1-0)/1
        @test s[2] ≈ 3.0   # (4-1)/1
        @test s[3] ≈ 5.0   # (9-4)/1
    end

    @testset "_fill_slopes! with Left(Deriv1)" begin
        # d[1] given directly, forward recurrence
        bc = Left(Deriv1(3.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        d = zeros(4)

        _fill_slopes!(d, s, h, bc)

        # d[1] = 3 (given)
        # d[2] = 2*1 - 3 = -1
        # d[3] = 2*3 - (-1) = 7
        # d[4] = 2*5 - 7 = 3
        @test d[1] ≈ 3.0
        @test d[2] ≈ -1.0
        @test d[3] ≈ 7.0
        @test d[4] ≈ 3.0
    end

    @testset "_fill_slopes! with Left(Deriv2)" begin
        # d[1] = s[1] - (κ/2)*h[1], forward recurrence
        bc = Left(Deriv2(2.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        d = zeros(4)

        _fill_slopes!(d, s, h, bc)

        # d[1] = 1 - (2/2)*1 = 0
        # d[2] = 2*1 - 0 = 2
        # d[3] = 2*3 - 2 = 4
        # d[4] = 2*5 - 4 = 6
        @test d[1] ≈ 0.0
        @test d[2] ≈ 2.0
        @test d[3] ≈ 4.0
        @test d[4] ≈ 6.0
    end

    @testset "_fill_slopes! with Right(Deriv1)" begin
        # d[n] given directly, backward recurrence
        bc = Right(Deriv1(7.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        d = zeros(4)

        _fill_slopes!(d, s, h, bc)

        # d[4] = 7 (given)
        # d[3] = 2*5 - 7 = 3
        # d[2] = 2*3 - 3 = 3
        # d[1] = 2*1 - 3 = -1
        @test d[1] ≈ -1.0
        @test d[2] ≈ 3.0
        @test d[3] ≈ 3.0
        @test d[4] ≈ 7.0
    end

    @testset "_fill_slopes! with Right(Deriv2)" begin
        # d[n] = s[end] + (κ/2)*h[end], backward recurrence
        bc = Right(Deriv2(2.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        d = zeros(4)

        _fill_slopes!(d, s, h, bc)

        # d[4] = 5 + (2/2)*1 = 6
        # d[3] = 2*5 - 6 = 4
        # d[2] = 2*3 - 4 = 2
        # d[1] = 2*1 - 2 = 0
        @test d[1] ≈ 0.0
        @test d[2] ≈ 2.0
        @test d[3] ≈ 4.0
        @test d[4] ≈ 6.0
    end

    @testset "forward recurrence" begin
        s = [1.0, 3.0, 5.0]
        d = zeros(4)
        d1 = 0.0

        _forward_recurrence!(d, s, d1)

        # d[1] = 0
        # d[2] = 2*1 - 0 = 2
        # d[3] = 2*3 - 2 = 4
        # d[4] = 2*5 - 4 = 6
        @test d[1] ≈ 0.0
        @test d[2] ≈ 2.0
        @test d[3] ≈ 4.0
        @test d[4] ≈ 6.0
    end

    @testset "backward recurrence" begin
        s = [1.0, 3.0, 5.0]
        d = zeros(4)
        dn = 6.0

        _backward_recurrence!(d, s, dn)

        # d[4] = 6
        # d[3] = 2*5 - 6 = 4
        # d[2] = 2*3 - 4 = 2
        # d[1] = 2*1 - 2 = 0
        @test d[1] ≈ 0.0
        @test d[2] ≈ 2.0
        @test d[3] ≈ 4.0
        @test d[4] ≈ 6.0
    end

    @testset "coefficient computation" begin
        # a[i] = (s[i] - d[i]) * inv_h[i]
        s = [1.0, 3.0, 5.0]
        d = [0.0, 2.0, 4.0, 6.0]
        inv_h = [1.0, 1.0, 1.0]
        a = zeros(3)

        _compute_quadratic_coefficients!(a, d, s, inv_h)

        # a[1] = (1 - 0) * 1 = 1
        # a[2] = (3 - 2) * 1 = 1
        # a[3] = (5 - 4) * 1 = 1
        @test a[1] ≈ 1.0
        @test a[2] ≈ 1.0
        @test a[3] ≈ 1.0
    end

end

# ============================================================================
# Group 4: Public API Tests
# ============================================================================
@testset "Quadratic Interpolation - Public API" begin

    @testset "quadratic_interp scalar - grid points" begin
        # Grid point interpolation is always exact regardless of BC
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        @test quadratic_interp(x, y, 0.0) ≈ 0.0
        @test quadratic_interp(x, y, 1.0) ≈ 1.0
        @test quadratic_interp(x, y, 2.0) ≈ 4.0
        @test quadratic_interp(x, y, 3.0) ≈ 9.0
    end

    @testset "quadratic_interp scalar - exact with correct BC" begin
        # f(x) = x² on [0, 3]
        # For exact interpolation, use BC that matches f:
        # f'(3) = 6, so Right(Deriv1(6)) gives exact x² interpolation
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        # With Right(Deriv1(6)), quadratic spline exactly interpolates x²
        @test quadratic_interp(x, y, 0.5; bc=Right(Deriv1(6.0))) ≈ 0.25 rtol=1e-10
        @test quadratic_interp(x, y, 1.5; bc=Right(Deriv1(6.0))) ≈ 2.25 rtol=1e-10
        @test quadratic_interp(x, y, 2.5; bc=Right(Deriv1(6.0))) ≈ 6.25 rtol=1e-10
    end

    @testset "quadratic_interp BC variants" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        # Left(Deriv2(0)) - natural at left (default)
        v1 = quadratic_interp(x, y, 0.5; bc=Left(Deriv2(0.0)))
        @test isfinite(v1)
        @test 0.0 < v1 < 1.0  # between y[1] and y[2]

        # Left(Deriv1(0)) - zero slope at left
        v2 = quadratic_interp(x, y, 0.5; bc=Left(Deriv1(0.0)))
        @test isfinite(v2)

        # Right(Deriv2(0)) - natural at right
        v3 = quadratic_interp(x, y, 0.5; bc=Right(Deriv2(0.0)))
        @test isfinite(v3)

        # Right(Deriv1(6)) - specified slope at right (S'(3) = 6 for x²)
        v4 = quadratic_interp(x, y, 0.5; bc=Right(Deriv1(6.0)))
        @test v4 ≈ 0.25 rtol=1e-10
    end

    @testset "quadratic_interp! in-place" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]
        xq = [0.5, 1.5, 2.5]
        out = zeros(3)

        # Use correct BC for exact x² interpolation
        quadratic_interp!(out, x, y, xq; bc=Right(Deriv1(6.0)))

        @test out[1] ≈ 0.25 rtol=1e-10
        @test out[2] ≈ 2.25 rtol=1e-10
        @test out[3] ≈ 6.25 rtol=1e-10
    end

    @testset "quadratic_interp vector (allocating)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]
        xq = [0.5, 1.5, 2.5]

        # Use correct BC for exact x² interpolation
        result = quadratic_interp(x, y, xq; bc=Right(Deriv1(6.0)))

        @test result isa Vector{Float64}
        @test length(result) == 3
        @test result[1] ≈ 0.25 rtol=1e-10
        @test result[2] ≈ 2.25 rtol=1e-10
        @test result[3] ≈ 6.25 rtol=1e-10
    end

    @testset "quadratic_interp derivatives" begin
        # f(x) = x², with correct BC for exact interpolation
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        # deriv=1: S'(1.5) = 3.0 (for f(x)=x², f'(x)=2x)
        d1 = quadratic_interp(x, y, 1.5; bc=Right(Deriv1(6.0)), deriv=1)
        @test d1 ≈ 3.0 rtol=1e-10

        # deriv=2: S''(x) = 2 for f(x)=x²
        d2 = quadratic_interp(x, y, 1.5; bc=Right(Deriv1(6.0)), deriv=2)
        @test d2 ≈ 2.0 rtol=1e-10
    end

    @testset "quadratic_interp extrapolation" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 4.0]

        # :none (default) should throw for out-of-domain
        @test_throws DomainError quadratic_interp(x, y, -0.5)
        @test_throws DomainError quadratic_interp(x, y, 2.5)

        # :constant - clamp to boundary values (outside domain)
        @test quadratic_interp(x, y, -0.5; extrap=:constant) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap=:constant) ≈ 4.0

        # :constant - inside domain should work normally (coverage for eval_core path)
        @test quadratic_interp(x, y, 1.0; extrap=:constant) ≈ 1.0

        # :constant - derivatives return zero outside domain
        @test quadratic_interp(x, y, -0.5; extrap=:constant, deriv=1) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap=:constant, deriv=1) ≈ 0.0
        @test quadratic_interp(x, y, -0.5; extrap=:constant, deriv=2) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap=:constant, deriv=2) ≈ 0.0

        # :extension - extend the polynomial (right side)
        v_ext_right = quadratic_interp(x, y, 2.5; extrap=:extension)
        @test isfinite(v_ext_right)

        # :extension - extend the polynomial (left side)
        v_ext_left = quadratic_interp(x, y, -0.5; extrap=:extension)
        @test isfinite(v_ext_left)

        # :extension derivatives
        d1_left = quadratic_interp(x, y, -0.5; extrap=:extension, deriv=1)
        d2_left = quadratic_interp(x, y, -0.5; extrap=:extension, deriv=2)
        @test isfinite(d1_left)
        @test isfinite(d2_left)
    end

    @testset "quadratic_interp Float32" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        y32 = Float32[0.0, 1.0, 4.0, 9.0]

        # Use correct BC for exact x² interpolation
        result = quadratic_interp(x32, y32, 1.5f0; bc=Right(Deriv1(6.0f0)))
        @test result isa Float32
        @test result ≈ 2.25f0 rtol=1e-5
    end

    @testset "quadratic_interp type stability" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        @test @inferred(quadratic_interp(x, y, 0.5)) isa Float64
        @test @inferred(quadratic_interp(x, y, 0.5; deriv=1)) isa Float64
        @test @inferred(quadratic_interp(x, y, 0.5; deriv=2)) isa Float64
    end

    @testset "quadratic_interp non-uniform grid" begin
        x = [0.0, 0.5, 1.5, 3.0]
        y = x.^2  # [0, 0.25, 2.25, 9]

        # At grid points (always exact)
        @test quadratic_interp(x, y, 0.0) ≈ 0.0
        @test quadratic_interp(x, y, 0.5) ≈ 0.25
        @test quadratic_interp(x, y, 1.5) ≈ 2.25
        @test quadratic_interp(x, y, 3.0) ≈ 9.0

        # Midpoints with correct BC (f'(3) = 6 for x²)
        @test quadratic_interp(x, y, 0.25; bc=Right(Deriv1(6.0))) ≈ 0.0625 rtol=1e-10
        @test quadratic_interp(x, y, 2.0; bc=Right(Deriv1(6.0))) ≈ 4.0 rtol=1e-10
    end

end

# ============================================================================
# Group 5: QuadraticInterpolant Tests
# ============================================================================
@testset "Quadratic Interpolation - Interpolant" begin

    @testset "QuadraticInterpolant construction" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        # 2-argument form returns QuadraticInterpolant
        itp = quadratic_interp(x, y)
        @test itp isa QuadraticInterpolant

        # With BC option
        itp2 = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
        @test itp2 isa QuadraticInterpolant
    end

    @testset "QuadraticInterpolant scalar call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        # Grid points
        @test itp(0.0) ≈ 0.0
        @test itp(1.0) ≈ 1.0
        @test itp(2.0) ≈ 4.0
        @test itp(3.0) ≈ 9.0

        # Midpoints (exact with correct BC)
        @test itp(0.5) ≈ 0.25 rtol=1e-10
        @test itp(1.5) ≈ 2.25 rtol=1e-10
        @test itp(2.5) ≈ 6.25 rtol=1e-10
    end

    @testset "QuadraticInterpolant broadcast" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        # Broadcast
        result = itp.([0.5, 1.5, 2.5])
        @test result ≈ [0.25, 2.25, 6.25] rtol=1e-10
    end

    @testset "QuadraticInterpolant vector call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        # Vector call
        result = itp([0.5, 1.5, 2.5])
        @test result isa Vector{Float64}
        @test result ≈ [0.25, 2.25, 6.25] rtol=1e-10
    end

    @testset "QuadraticInterpolant in-place call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        out = zeros(3)
        itp(out, [0.5, 1.5, 2.5])
        @test out ≈ [0.25, 2.25, 6.25] rtol=1e-10
    end

    @testset "QuadraticInterpolant derivative call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        # deriv keyword
        @test itp(1.5; deriv=1) ≈ 3.0 rtol=1e-10
        @test itp(1.5; deriv=2) ≈ 2.0 rtol=1e-10
    end

    @testset "QuadraticInterpolant Float32" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        y32 = x32.^2

        itp = quadratic_interp(x32, y32; bc=Right(Deriv1(6.0f0)))
        @test itp isa QuadraticInterpolant{Float32}
        @test itp(1.5f0) isa Float32
        @test itp(1.5f0) ≈ 2.25f0 rtol=1e-5
    end

    @testset "QuadraticInterpolant type stability" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y)
        @test @inferred(itp(0.5)) isa Float64
    end

end

# ============================================================================
# Group 6: Allocation Tests
# ============================================================================
@testset "Quadratic Interpolation - Allocations" begin

    # ALLOC_THRESHOLD is defined in runtests.jl

    @testset "interpolant scalar zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2

        # Create interpolant (precomputes coefficients)
        itp = quadratic_interp(x, y; bc=Right(Deriv1(2.0)))

        # Prime JIT
        for _ in 1:10
            itp(0.5)
        end

        # Scalar call should be zero-allocation
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Derivative calls should also be zero-allocation
        for _ in 1:10
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)
        end
        allocs_d1 = @allocated itp(0.5; deriv=1)
        allocs_d2 = @allocated itp(0.5; deriv=2)
        @test allocs_d1 <= ALLOC_THRESHOLD
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    @testset "interpolant in-place vector zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2
        xq = collect(range(0.1, 0.9, 100))
        out = zeros(100)

        itp = quadratic_interp(x, y; bc=Right(Deriv1(2.0)))

        # Prime JIT
        for _ in 1:10
            itp(out, xq)
        end

        # In-place should be zero-allocation
        allocs = @allocated itp(out, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "DerivativeView zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(2.0)))
        d1 = deriv1(itp)
        d2 = deriv2(itp)

        # Prime
        for _ in 1:10
            d1(0.5)
            d2(0.5)
        end

        allocs_d1 = @allocated d1(0.5)
        allocs_d2 = @allocated d2(0.5)
        @test allocs_d1 <= ALLOC_THRESHOLD
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    # ========================================
    # One-shot API Allocation Tests
    # ========================================
    @testset "one-shot scalar zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2

        # warm-up all code paths (deriv=0, 1, 2)
        quadratic_interp(x, y, 0.5)
        quadratic_interp(x, y, 0.5; deriv=1)
        quadratic_interp(x, y, 0.5; deriv=2)

        # Measure allocation for scalar one-shot
        allocs = @allocated quadratic_interp(x, y, 0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Different query points should have same allocation
        allocs_other = @allocated quadratic_interp(x, y, 0.3)
        @test allocs_other <= ALLOC_THRESHOLD

        # Derivative calls should also be zero-allocation
        allocs_d1 = @allocated quadratic_interp(x, y, 0.5; deriv=1)
        allocs_d2 = @allocated quadratic_interp(x, y, 0.5; deriv=2)
        @test allocs_d1 <= ALLOC_THRESHOLD
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    @testset "one-shot vector in-place zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2
        xq = collect(range(0.1, 0.9, 100))
        out = similar(xq)

        # warmup all BC types and deriv values (single call each)
        quadratic_interp!(out, x, y, xq)
        quadratic_interp!(out, x, y, xq; bc = Left(Deriv1(2.0)))
        quadratic_interp!(out, x, y, xq; bc = Left(Deriv2(1.0)))
        quadratic_interp!(out, x, y, xq; bc = Right(Deriv1(2.0)))
        quadratic_interp!(out, x, y, xq; bc = Right(Deriv2(1.0)))
        quadratic_interp!(out, x, y, xq; deriv=1)
        quadratic_interp!(out, x, y, xq; deriv=2)

        # In-place version - all BC types
        alloc1 = @allocated quadratic_interp!(out, x, y, xq)
        alloc2 = @allocated quadratic_interp!(out, x, y, xq; bc = Left(Deriv1(2.0)))
        alloc3 = @allocated quadratic_interp!(out, x, y, xq; bc = Left(Deriv2(1.0)))
        alloc4 = @allocated quadratic_interp!(out, x, y, xq; bc = Right(Deriv1(2.0)))
        alloc5 = @allocated quadratic_interp!(out, x, y, xq; bc = Right(Deriv2(1.0)))

        @test alloc1 <= ALLOC_THRESHOLD
        @test alloc2 <= ALLOC_THRESHOLD
        @test alloc3 <= ALLOC_THRESHOLD
        @test alloc4 <= ALLOC_THRESHOLD
        @test alloc5 <= ALLOC_THRESHOLD

        # With different BC run-time value (should not affect allocation)
        alloc2 = @allocated quadratic_interp!(out, x, y, xq; bc = Left(Deriv1(-2.0)))
        alloc3 = @allocated quadratic_interp!(out, x, y, xq; bc = Left(Deriv2(-1.0)))
        alloc4 = @allocated quadratic_interp!(out, x, y, xq; bc = Right(Deriv1(-2.0)))
        alloc5 = @allocated quadratic_interp!(out, x, y, xq; bc = Right(Deriv2(-1.0)))

        @test alloc2 <= ALLOC_THRESHOLD
        @test alloc3 <= ALLOC_THRESHOLD
        @test alloc4 <= ALLOC_THRESHOLD
        @test alloc5 <= ALLOC_THRESHOLD

        # Derivative evaluations
        alloc_d1 = @allocated quadratic_interp!(out, x, y, xq; deriv=1)
        alloc_d2 = @allocated quadratic_interp!(out, x, y, xq; deriv=2)

        @test alloc_d1 <= ALLOC_THRESHOLD
        @test alloc_d2 <= ALLOC_THRESHOLD
    end
end

# ============================================================================
# Group 7: Type Conversion Tests (Real → Float wrappers)
# ============================================================================
@testset "Quadratic Interpolation - Type Conversion" begin
    using FastInterpolations: _promote_bc

    @testset "_promote_bc same-type passthrough" begin
        # Same-type should return the same object (zero-cost)
        bc_left = Left(Deriv1(1.0))
        bc_right = Right(Deriv2(0.0))

        @test _promote_bc(bc_left, Float64) === bc_left
        @test _promote_bc(bc_right, Float64) === bc_right

        # Float32 same-type passthrough
        bc_left32 = Left(Deriv1(1.0f0))
        bc_right32 = Right(Deriv2(0.0f0))

        @test _promote_bc(bc_left32, Float32) === bc_left32
        @test _promote_bc(bc_right32, Float32) === bc_right32
    end

    @testset "_promote_bc type conversion" begin
        # Float32 → Float64 conversion
        bc_left32 = Left(Deriv1(1.0f0))
        bc_right32 = Right(Deriv2(0.0f0))

        bc_left64 = _promote_bc(bc_left32, Float64)
        bc_right64 = _promote_bc(bc_right32, Float64)

        @test bc_left64 isa Left{Float64}
        @test bc_right64 isa Right{Float64}
        @test bc_left64.bc.val ≈ 1.0
        @test bc_right64.bc.val ≈ 0.0

        # Float64 → Float32 conversion
        bc_left_f64 = Left(Deriv2(2.0))
        bc_right_f64 = Right(Deriv1(3.0))

        bc_left_f32 = _promote_bc(bc_left_f64, Float32)
        bc_right_f32 = _promote_bc(bc_right_f64, Float32)

        @test bc_left_f32 isa Left{Float32}
        @test bc_right_f32 isa Right{Float32}
    end

    @testset "quadratic_interp with Integer arrays (Real → Float)" begin
        # Integer arrays trigger the Real → Float wrapper
        x = [0, 1, 2, 3]  # Int64
        y = [0, 1, 4, 9]  # Int64 (x²)

        # Scalar interpolation with Int data
        result = quadratic_interp(x, y, 1.5)
        @test result isa Float64
        @test isfinite(result)

        # With explicit BC
        result2 = quadratic_interp(x, y, 1.5; bc=Right(Deriv1(6.0)))
        @test result2 ≈ 2.25 rtol=1e-10

        # Scalar with Int query point
        result3 = quadratic_interp(x, y, 2)
        @test result3 ≈ 4.0

        # Derivatives with Int data
        d1 = quadratic_interp(x, y, 1.5; bc=Right(Deriv1(6.0)), deriv=1)
        @test d1 ≈ 3.0 rtol=1e-10
    end

    @testset "quadratic_interp vector with Integer arrays" begin
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]
        xq = [0.5, 1.5, 2.5]

        # Allocating version
        result = quadratic_interp(x, y, xq; bc=Right(Deriv1(6.0)))
        @test result isa Vector{Float64}
        @test result ≈ [0.25, 2.25, 6.25] rtol=1e-10

        # With Integer query points
        xq_int = [1, 2]
        result_int = quadratic_interp(x, y, xq_int)
        @test result_int ≈ [1.0, 4.0]
    end

    @testset "quadratic_interp! with Integer arrays (in-place Real → Float)" begin
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]
        xq = [0.5, 1.5, 2.5]
        out = zeros(3)

        quadratic_interp!(out, x, y, xq; bc=Right(Deriv1(6.0)))
        @test out ≈ [0.25, 2.25, 6.25] rtol=1e-10

        # Integer query points
        xq_int = [1, 2]
        out2 = zeros(2)
        quadratic_interp!(out2, x, y, xq_int)
        @test out2 ≈ [1.0, 4.0]
    end

    @testset "QuadraticInterpolant Real scalar wrapper" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        # Call with Int (triggers Real wrapper, not the T method)
        result = itp(2)  # Int64, not Float64
        @test result isa Float64
        @test result ≈ 4.0

        # Derivative with Int query
        d1 = itp(2; deriv=1)
        @test d1 ≈ 4.0 rtol=1e-10
    end

    @testset "QuadraticInterpolant in-place with type conversion" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))

        # In-place with Integer query points (triggers type conversion path)
        out = zeros(3)
        xq_int = [1, 2, 3]  # Int64
        itp(out, xq_int)
        @test out ≈ [1.0, 4.0, 9.0]

        # Derivative with type conversion
        out2 = zeros(2)
        itp(out2, [1, 2]; deriv=1)
        @test out2 ≈ [2.0, 4.0] rtol=1e-10
    end

    @testset "QuadraticInterpolant from Integer arrays (2-arg Real wrapper)" begin
        # Integer arrays trigger the Real wrapper for 2-argument form
        x = [0, 1, 2, 3]  # Int64
        y = [0, 1, 4, 9]  # Int64

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
        @test itp isa QuadraticInterpolant{Float64}

        @test itp(1.5) ≈ 2.25 rtol=1e-10
        @test itp(0.5) ≈ 0.25 rtol=1e-10
    end

    @testset "BC type promotion with Real data" begin
        # Int data with Int BC (both promoted to Float64)
        x_int = [0, 1, 2, 3]
        y_int = [0, 1, 4, 9]

        # BC with Int value triggers promotion
        result = quadratic_interp(x_int, y_int, 1.5; bc=Left(Deriv2(0)))
        @test result isa Float64
        @test isfinite(result)

        # BC with Float64 value for Int data
        result2 = quadratic_interp(x_int, y_int, 1.5; bc=Right(Deriv1(6.0)))
        @test result2 ≈ 2.25 rtol=1e-10
    end

end

@testset "Quadratic Interpolation - DerivativeView" begin

    @testset "deriv1 view" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
        d1 = deriv1(itp)

        # d1(x) = S'(x) = 2x for f(x)=x²
        @test d1(0.0) ≈ 0.0 rtol=1e-10
        @test d1(1.0) ≈ 2.0 rtol=1e-10
        @test d1(1.5) ≈ 3.0 rtol=1e-10
        @test d1(2.0) ≈ 4.0 rtol=1e-10
        @test d1(3.0) ≈ 6.0 rtol=1e-10
    end

    @testset "deriv2 view" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
        d2 = deriv2(itp)

        # d2(x) = S''(x) = 2 (constant) for f(x)=x²
        @test d2(0.5) ≈ 2.0 rtol=1e-10
        @test d2(1.5) ≈ 2.0 rtol=1e-10
        @test d2(2.5) ≈ 2.0 rtol=1e-10
    end

    @testset "deriv1 broadcast" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2

        itp = quadratic_interp(x, y; bc=Right(Deriv1(6.0)))
        d1 = deriv1(itp)

        result = d1.([0.5, 1.5, 2.5])
        @test result ≈ [1.0, 3.0, 5.0] rtol=1e-10
    end

end

# ============================================================================
# Group 9: Mathematical Correctness Tests (Non-uniform Grid)
# ============================================================================
@testset "Quadratic Interpolation - Mathematical Correctness" begin
    # For a quadratic polynomial f(x) = ax² + bx + c, the quadratic spline
    # should be EXACT regardless of which BC is used (Left/Right, Deriv1/Deriv2),
    # as long as the BC value matches the true derivative.

    @testset "BC equivalence on quadratic polynomial (non-uniform grid)" begin
        # Non-uniform grid with varying spacing
        x = [0.0, 0.3, 0.7, 1.2, 2.0, 3.5]

        # f(x) = 2x² - 3x + 1
        a, b, c = 2.0, -3.0, 1.0
        f(t) = a*t^2 + b*t + c
        f_d1(t) = 2*a*t + b  # f'(x) = 4x - 3
        f_d2 = 2*a           # f''(x) = 4 (constant)

        y = f.(x)

        # Compute true BC values
        x_left, x_right = first(x), last(x)
        d1_left = f_d1(x_left)    # f'(0) = -3
        d1_right = f_d1(x_right)  # f'(3.5) = 11
        d2_val = f_d2             # f''(x) = 4

        # All four BC variants
        bc_left_d1 = Left(Deriv1(d1_left))
        bc_left_d2 = Left(Deriv2(d2_val))
        bc_right_d1 = Right(Deriv1(d1_right))
        bc_right_d2 = Right(Deriv2(d2_val))

        # Query points (including edge cases near boundaries)
        xq = [0.0, 0.1, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
        expected = f.(xq)
        expected_d1 = f_d1.(xq)
        expected_d2 = fill(f_d2, length(xq))

        # Test all BC variants produce exact results
        for (name, bc) in [
            ("Left(Deriv1)", bc_left_d1),
            ("Left(Deriv2)", bc_left_d2),
            ("Right(Deriv1)", bc_right_d1),
            ("Right(Deriv2)", bc_right_d2)
        ]
            @testset "$name" begin
                # Value interpolation
                result = quadratic_interp(x, y, xq; bc=bc)
                @test result ≈ expected rtol=1e-12 atol=1e-14

                # First derivative
                result_d1 = quadratic_interp(x, y, xq; bc=bc, deriv=1)
                @test result_d1 ≈ expected_d1 rtol=1e-12 atol=1e-14

                # Second derivative
                result_d2 = quadratic_interp(x, y, xq; bc=bc, deriv=2)
                @test result_d2 ≈ expected_d2 rtol=1e-12 atol=1e-14
            end
        end
    end

    @testset "all BCs produce identical spline for quadratic" begin
        # Highly non-uniform grid
        x = [0.0, 0.1, 0.5, 2.0, 2.1, 5.0]

        # f(x) = x² (simpler case)
        y = x.^2

        # True derivatives
        d1_left = 2 * first(x)   # 0
        d1_right = 2 * last(x)   # 10
        d2_val = 2.0             # constant

        # Create interpolants with all BC variants
        itp_left_d1 = quadratic_interp(x, y; bc=Left(Deriv1(d1_left)))
        itp_left_d2 = quadratic_interp(x, y; bc=Left(Deriv2(d2_val)))
        itp_right_d1 = quadratic_interp(x, y; bc=Right(Deriv1(d1_right)))
        itp_right_d2 = quadratic_interp(x, y; bc=Right(Deriv2(d2_val)))

        # Query at many points
        xq = range(0.0, 5.0, 50)

        # All should produce identical results
        result_ld1 = itp_left_d1.(xq)
        result_ld2 = itp_left_d2.(xq)
        result_rd1 = itp_right_d1.(xq)
        result_rd2 = itp_right_d2.(xq)

        @test result_ld1 ≈ result_ld2 rtol=1e-12
        @test result_ld1 ≈ result_rd1 rtol=1e-12
        @test result_ld1 ≈ result_rd2 rtol=1e-12

        # All should match x²
        expected = collect(xq).^2
        @test result_ld1 ≈ expected rtol=1e-12
    end

    @testset "edge cases: boundary evaluation" begin
        # Grid with extreme spacing variation
        x = [0.0, 0.01, 1.0, 1.01, 10.0]  # tiny + large intervals

        # f(x) = -x² + 5x
        f(t) = -t^2 + 5*t
        f_d1(t) = -2*t + 5
        f_d2 = -2.0

        y = f.(x)

        bc = Right(Deriv1(f_d1(last(x))))  # f'(10) = -15

        # Test exact boundary points
        @test quadratic_interp(x, y, 0.0; bc=bc) ≈ f(0.0) rtol=1e-12
        @test quadratic_interp(x, y, 10.0; bc=bc) ≈ f(10.0) rtol=1e-12

        # Test points very close to boundaries
        @test quadratic_interp(x, y, 1e-10; bc=bc) ≈ f(1e-10) rtol=1e-10
        @test quadratic_interp(x, y, 10.0 - 1e-10; bc=bc) ≈ f(10.0 - 1e-10) rtol=1e-10

        # Test mid-interval points
        @test quadratic_interp(x, y, 0.005; bc=bc) ≈ f(0.005) rtol=1e-12
        @test quadratic_interp(x, y, 5.0; bc=bc) ≈ f(5.0) atol=1e-14  # f(5)=0, need atol
    end

    @testset "Float32 precision on non-uniform grid" begin
        x32 = Float32[0.0, 0.5, 1.5, 3.0]
        y32 = x32.^2

        d1_right = 2 * last(x32)  # 6.0f0

        itp = quadratic_interp(x32, y32; bc=Right(Deriv1(d1_right)))

        # Should be exact within Float32 precision
        @test itp(1.0f0) ≈ 1.0f0 rtol=1e-6
        @test itp(2.0f0) ≈ 4.0f0 rtol=1e-6
        @test itp(0.25f0) ≈ 0.0625f0 rtol=1e-6
    end

    @testset "consistency: scalar vs vector API" begin
        x = [0.0, 0.4, 1.1, 2.0, 3.3]
        y = x.^2

        bc = Left(Deriv2(2.0))
        xq = [0.2, 0.8, 1.5, 2.5, 3.0]

        # Scalar API
        results_scalar = [quadratic_interp(x, y, xi; bc=bc) for xi in xq]

        # Vector API (allocating)
        results_vector = quadratic_interp(x, y, xq; bc=bc)

        # In-place API
        results_inplace = zeros(length(xq))
        quadratic_interp!(results_inplace, x, y, xq; bc=bc)

        # Interpolant API
        itp = quadratic_interp(x, y; bc=bc)
        results_itp = itp.(xq)

        # All should be identical (not just approximately equal)
        @test results_scalar == results_vector
        @test results_scalar == results_inplace
        @test results_scalar == results_itp
    end

end

# ============================================================================
# Group 10: Extension Extrapolation Accuracy Tests
# ============================================================================
@testset "Quadratic Interpolation - Extension Extrapolation Accuracy" begin
    # For quadratic polynomials, extension extrapolation should be EXACT
    # because extending a quadratic polynomial is still the same quadratic.

    @testset "extension extrapolation exact for ax² + bx + c" begin
        # Non-uniform grid
        x = [0.0, 0.3, 0.7, 1.2, 2.0, 3.5]

        # f(x) = 2x² - 3x + 1
        a, b, c = 2.0, -3.0, 1.0
        f(t) = a*t^2 + b*t + c
        f_d1(t) = 2*a*t + b  # f'(x) = 4x - 3
        f_d2 = 2*a           # f''(x) = 4 (constant)

        y = f.(x)

        # Compute true BC values
        x_left, x_right = first(x), last(x)
        d1_left = f_d1(x_left)    # f'(0) = -3
        d1_right = f_d1(x_right)  # f'(3.5) = 11
        d2_val = f_d2             # f''(x) = 4

        # All four BC variants
        bcs = [
            ("Left(Deriv1)", Left(Deriv1(d1_left))),
            ("Left(Deriv2)", Left(Deriv2(d2_val))),
            ("Right(Deriv1)", Right(Deriv1(d1_right))),
            ("Right(Deriv2)", Right(Deriv2(d2_val)))
        ]

        # Extrapolation query points (outside domain)
        xq_left = [-1.0, -0.5, -0.1]   # left of x[1]=0
        xq_right = [4.0, 5.0, 6.0]      # right of x[end]=3.5

        for (name, bc) in bcs
            @testset "$name - one-shot API" begin
                # Left extrapolation - value
                for xi in xq_left
                    result = quadratic_interp(x, y, xi; bc=bc, extrap=:extension)
                    @test result ≈ f(xi) rtol=1e-12 atol=1e-14
                end

                # Right extrapolation - value
                for xi in xq_right
                    result = quadratic_interp(x, y, xi; bc=bc, extrap=:extension)
                    @test result ≈ f(xi) rtol=1e-12 atol=1e-14
                end

                # Left extrapolation - derivatives
                for xi in xq_left
                    d1 = quadratic_interp(x, y, xi; bc=bc, extrap=:extension, deriv=1)
                    d2 = quadratic_interp(x, y, xi; bc=bc, extrap=:extension, deriv=2)
                    @test d1 ≈ f_d1(xi) rtol=1e-12 atol=1e-14
                    @test d2 ≈ f_d2 rtol=1e-12 atol=1e-14
                end

                # Right extrapolation - derivatives
                for xi in xq_right
                    d1 = quadratic_interp(x, y, xi; bc=bc, extrap=:extension, deriv=1)
                    d2 = quadratic_interp(x, y, xi; bc=bc, extrap=:extension, deriv=2)
                    @test d1 ≈ f_d1(xi) rtol=1e-12 atol=1e-14
                    @test d2 ≈ f_d2 rtol=1e-12 atol=1e-14
                end
            end

            @testset "$name - interpolant API" begin
                itp = quadratic_interp(x, y; bc=bc, extrap=:extension)
                d1_view = deriv1(itp)
                d2_view = deriv2(itp)

                # Left extrapolation
                for xi in xq_left
                    @test itp(xi) ≈ f(xi) rtol=1e-12 atol=1e-14
                    @test d1_view(xi) ≈ f_d1(xi) rtol=1e-12 atol=1e-14
                    @test d2_view(xi) ≈ f_d2 rtol=1e-12 atol=1e-14
                end

                # Right extrapolation
                for xi in xq_right
                    @test itp(xi) ≈ f(xi) rtol=1e-12 atol=1e-14
                    @test d1_view(xi) ≈ f_d1(xi) rtol=1e-12 atol=1e-14
                    @test d2_view(xi) ≈ f_d2 rtol=1e-12 atol=1e-14
                end
            end
        end
    end

    @testset "extension C0 continuity at boundary" begin
        # The extension polynomial must match exactly at the boundary
        x = [0.0, 1.0, 2.0, 3.0]
        y = x.^2  # f(x) = x²

        bc = Right(Deriv1(6.0))  # f'(3) = 6

        # Value at boundary should match from both sides
        val_inside = quadratic_interp(x, y, 3.0; bc=bc)
        val_outside = quadratic_interp(x, y, 3.0 + 1e-10; bc=bc, extrap=:extension)
        @test val_inside ≈ val_outside rtol=1e-8

        # Left boundary (value is 0.0, so use atol instead of rtol)
        val_inside_left = quadratic_interp(x, y, 0.0; bc=bc)
        val_outside_left = quadratic_interp(x, y, -1e-10; bc=bc, extrap=:extension)
        @test val_inside_left ≈ val_outside_left atol=1e-8
    end

    @testset "extension vector API" begin
        x = [0.0, 1.0, 2.0, 3.0]
        f(t) = t^2
        y = f.(x)

        bc = Right(Deriv1(6.0))

        # Query points spanning inside and outside
        xq = [-0.5, 0.5, 1.5, 2.5, 3.5]
        expected = f.(xq)

        # One-shot vector API
        result = quadratic_interp(x, y, xq; bc=bc, extrap=:extension)
        @test result ≈ expected rtol=1e-12

        # In-place API
        out = zeros(length(xq))
        quadratic_interp!(out, x, y, xq; bc=bc, extrap=:extension)
        @test out ≈ expected rtol=1e-12

        # Interpolant vector call
        itp = quadratic_interp(x, y; bc=bc, extrap=:extension)
        @test itp(xq) ≈ expected rtol=1e-12
    end

end
