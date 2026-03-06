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
            @test bc1 isa Left{Deriv1{Float64}}

            bc2 = Left(Deriv2(1.0))
            @test bc2 isa Left{Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Left(Deriv2(1.0f0))
            @test bc isa Left{Deriv2{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Left(Deriv1(1))
            @test bc isa Left{Deriv1{Float64}}
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
            # Type-Free design: Left/Right wrappers are AbstractBC (no type parameter)
            @test Left{Deriv1{Float64}} <: AbstractBC
            @test Left{Deriv2{Float64}} <: AbstractBC
            @test Left{Deriv1{Float32}} <: AbstractBC
        end
    end

    @testset "Right BC wrapper" begin
        @testset "construction with Float64" begin
            bc1 = Right(Deriv1(-0.5))
            @test bc1 isa Right{Deriv1{Float64}}

            bc2 = Right(Deriv2(0.0))
            @test bc2 isa Right{Deriv2{Float64}}
        end

        @testset "construction with Float32" begin
            bc = Right(Deriv1(1.0f0))
            @test bc isa Right{Deriv1{Float32}}
        end

        @testset "type promotion (Int → Float64)" begin
            bc = Right(Deriv2(0))
            @test bc isa Right{Deriv2{Float64}}
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
            # Type-Free design: Left/Right wrappers are AbstractBC (no type parameter)
            @test Right{Deriv1{Float64}} <: AbstractBC
            @test Right{Deriv2{Float64}} <: AbstractBC
            @test Right{Deriv2{Float32}} <: AbstractBC
        end
    end

    @testset "Left/Right distinctness" begin
        # Left and Right should be distinct types
        @test Left(Deriv1(0.0)) isa Left
        @test Right(Deriv1(0.0)) isa Right
        @test !(Left(Deriv1(0.0)) isa Right)
        @test !(Right(Deriv1(0.0)) isa Left)
    end

    @testset "MinCurvFit type" begin
        @testset "construction (non-parametric singleton)" begin
            bc = MinCurvFit()
            @test bc isa MinCurvFit
            # Type-Free design: MinCurvFit is AbstractBC (no type parameter)
            @test bc isa AbstractBC
        end

        @testset "singleton property" begin
            # All MinCurvFit() calls return the same type
            bc1 = MinCurvFit()
            bc2 = MinCurvFit()
            @test typeof(bc1) === typeof(bc2)
        end

        @testset "type stability" begin
            @test @inferred(MinCurvFit()) isa MinCurvFit
        end

        @testset "subtype relationship" begin
            # Type-Free design: MinCurvFit is AbstractBC (no type parameter)
            @test MinCurvFit <: AbstractBC
        end

        @testset "distinctness from Left/Right" begin
            @test !(MinCurvFit() isa Left)
            @test !(MinCurvFit() isa Right)
            @test !(Left(Deriv1(0.0)) isa MinCurvFit)
            @test !(Right(Deriv1(0.0)) isa MinCurvFit)
        end
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
        # Dummy x, y (unused by Deriv1)
        x_dummy = [0.0, 1.0, 2.0, 3.0]
        y_dummy = [0.0, 1.0, 4.0, 9.0]

        _fill_slopes!(d, s, h, bc, x_dummy, y_dummy)

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
        # Dummy x, y (unused by Deriv2)
        x_dummy = [0.0, 1.0, 2.0, 3.0]
        y_dummy = [0.0, 1.0, 4.0, 9.0]

        _fill_slopes!(d, s, h, bc, x_dummy, y_dummy)

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
        # Dummy x, y (unused by Deriv1)
        x_dummy = [0.0, 1.0, 2.0, 3.0]
        y_dummy = [0.0, 1.0, 4.0, 9.0]

        _fill_slopes!(d, s, h, bc, x_dummy, y_dummy)

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
        # Dummy x, y (unused by Deriv2)
        x_dummy = [0.0, 1.0, 2.0, 3.0]
        y_dummy = [0.0, 1.0, 4.0, 9.0]

        _fill_slopes!(d, s, h, bc, x_dummy, y_dummy)

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

    @testset "_fill_slopes! with MinCurvFit" begin
        @testset "uniform grid - recurrence satisfied" begin
            # MinCurvFit minimizes total curvature Σ(s[i]-d[i])²/h[i]
            # For s = [1, 3, 5], h = [1, 1, 1]:
            # Optimal d[1] = 1/3 (gives lower curvature than d[1]=0)
            s = [1.0, 3.0, 5.0]
            h = [1.0, 1.0, 1.0]
            d = zeros(4)
            # Dummy x, y (unused by MinCurvFit)
            x_dummy = [0.0, 1.0, 2.0, 3.0]
            y_dummy = [0.0, 1.0, 4.0, 9.0]

            _fill_slopes!(d, s, h, MinCurvFit(), x_dummy, y_dummy)

            # Verify recurrence relation is satisfied: d[i+1] = 2*s[i] - d[i]
            for i in 1:3
                @test d[i + 1] ≈ 2 * s[i] - d[i] rtol = 1.0e-12
            end

            # Verify all values are finite
            @test all(isfinite, d)

            # Verify d[1] is optimal: 1/3 for this case
            # (minimizes Σ(s[i] - d[i])²/h[i])
            @test d[1] ≈ 1 / 3 rtol = 1.0e-12
        end

        @testset "non-uniform grid - produces finite values" begin
            # Test that optimization produces finite values on non-uniform grid
            s = [2.0, 1.0, 3.0]  # varying secants
            h = [0.5, 1.5, 1.0]  # non-uniform spacing
            d = zeros(4)
            # Dummy x, y (unused by MinCurvFit)
            x_dummy = [0.0, 0.5, 2.0, 3.0]
            y_dummy = [0.0, 1.0, 2.5, 5.5]

            _fill_slopes!(d, s, h, MinCurvFit(), x_dummy, y_dummy)

            @test all(isfinite, d)
            # Verify recurrence relation: d[i+1] = 2*s[i] - d[i]
            for i in 1:3
                @test d[i + 1] ≈ 2 * s[i] - d[i] rtol = 1.0e-12
            end
        end

        @testset "single segment (n=2)" begin
            # n=2: only one segment
            s = [2.0]  # single secant
            h = [1.0]
            d = zeros(2)
            # Dummy x, y (unused by MinCurvFit)
            x_dummy = [0.0, 1.0]
            y_dummy = [0.0, 2.0]

            _fill_slopes!(d, s, h, MinCurvFit(), x_dummy, y_dummy)

            # For single segment, minimizing curvature means a = 0
            # a = (s - d[1]) / h => d[1] = s[1] for a = 0
            @test d[1] ≈ s[1] rtol = 1.0e-12
            # d[2] = 2*s[1] - d[1] = 2*2 - 2 = 2
            @test d[2] ≈ 2 * s[1] - d[1] rtol = 1.0e-12
        end

        @testset "Float32 support" begin
            s = Float32[1.0, 3.0, 5.0]
            h = Float32[1.0, 1.0, 1.0]
            d = zeros(Float32, 4)
            # Dummy x, y (unused by MinCurvFit)
            x_dummy = Float32[0.0, 1.0, 2.0, 3.0]
            y_dummy = Float32[0.0, 1.0, 4.0, 9.0]

            _fill_slopes!(d, s, h, MinCurvFit(), x_dummy, y_dummy)

            # Verify finite values and recurrence
            @test all(isfinite, d)
            for i in 1:3
                @test d[i + 1] ≈ 2 * s[i] - d[i] rtol = 1.0e-5
            end
        end

        @testset "curvature lower than Left(Deriv2(0))" begin
            # MinCurvFit should produce lower or equal total curvature
            # compared to Left(Deriv2(0)) which forces first interval linear
            s = [1.0, 3.0, 5.0]
            h = [1.0, 1.0, 1.0]
            # Dummy x, y (unused by MinCurvFit and Deriv2)
            x_dummy = [0.0, 1.0, 2.0, 3.0]
            y_dummy = [0.0, 1.0, 4.0, 9.0]

            d_smooth = zeros(4)
            d_left = zeros(4)

            _fill_slopes!(d_smooth, s, h, MinCurvFit(), x_dummy, y_dummy)
            _fill_slopes!(d_left, s, h, Left(Deriv2(0.0)), x_dummy, y_dummy)

            # Compute total curvature: Σ (s[i] - d[i])² / h[i]
            curvature_smooth = sum((s[i] - d_smooth[i])^2 / h[i] for i in 1:3)
            curvature_left = sum((s[i] - d_left[i])^2 / h[i] for i in 1:3)

            # MinCurvFit should have lower or equal curvature (it's the optimum)
            @test curvature_smooth <= curvature_left + 1.0e-10
        end
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
        @test quadratic_interp(x, y, 0.5; bc = Right(Deriv1(6.0))) ≈ 0.25 rtol = 1.0e-10
        @test quadratic_interp(x, y, 1.5; bc = Right(Deriv1(6.0))) ≈ 2.25 rtol = 1.0e-10
        @test quadratic_interp(x, y, 2.5; bc = Right(Deriv1(6.0))) ≈ 6.25 rtol = 1.0e-10
    end

    @testset "quadratic_interp BC variants" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        # Left(Deriv2(0)) - zero curvature at left (default)
        v1 = quadratic_interp(x, y, 0.5; bc = Left(Deriv2(0.0)))
        @test isfinite(v1)
        @test 0.0 < v1 < 1.0  # between y[1] and y[2]

        # Left(Deriv1(0)) - zero slope at left
        v2 = quadratic_interp(x, y, 0.5; bc = Left(Deriv1(0.0)))
        @test isfinite(v2)

        # Right(Deriv2(0)) - zero curvature at right
        v3 = quadratic_interp(x, y, 0.5; bc = Right(Deriv2(0.0)))
        @test isfinite(v3)

        # Right(Deriv1(6)) - specified slope at right (S'(3) = 6 for x²)
        v4 = quadratic_interp(x, y, 0.5; bc = Right(Deriv1(6.0)))
        @test v4 ≈ 0.25 rtol = 1.0e-10
    end

    @testset "quadratic_interp! in-place" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]
        xq = [0.5, 1.5, 2.5]
        out = zeros(3)

        # Use correct BC for exact x² interpolation
        quadratic_interp!(out, x, y, xq; bc = Right(Deriv1(6.0)))

        @test out[1] ≈ 0.25 rtol = 1.0e-10
        @test out[2] ≈ 2.25 rtol = 1.0e-10
        @test out[3] ≈ 6.25 rtol = 1.0e-10
    end

    @testset "quadratic_interp vector (allocating)" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]
        xq = [0.5, 1.5, 2.5]

        # Use correct BC for exact x² interpolation
        result = quadratic_interp(x, y, xq; bc = Right(Deriv1(6.0)))

        @test result isa Vector{Float64}
        @test length(result) == 3
        @test result[1] ≈ 0.25 rtol = 1.0e-10
        @test result[2] ≈ 2.25 rtol = 1.0e-10
        @test result[3] ≈ 6.25 rtol = 1.0e-10
    end

    @testset "quadratic_interp derivatives" begin
        # f(x) = x², with correct BC for exact interpolation
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        # deriv=DerivOp(1): S'(1.5) = 3.0 (for f(x)=x², f'(x)=2x)
        d1 = quadratic_interp(x, y, 1.5; bc = Right(Deriv1(6.0)), deriv = DerivOp(1))
        @test d1 ≈ 3.0 rtol = 1.0e-10

        # deriv=DerivOp(2): S''(x) = 2 for f(x)=x²
        d2 = quadratic_interp(x, y, 1.5; bc = Right(Deriv1(6.0)), deriv = DerivOp(2))
        @test d2 ≈ 2.0 rtol = 1.0e-10
    end

    @testset "quadratic_interp extrapolation" begin
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 4.0]

        # :none (default) should throw for out-of-domain
        @test_throws DomainError quadratic_interp(x, y, -0.5)
        @test_throws DomainError quadratic_interp(x, y, 2.5)

        # :constant - clamp to boundary values (outside domain)
        @test quadratic_interp(x, y, -0.5; extrap = ClampExtrap()) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap = ClampExtrap()) ≈ 4.0

        # :constant - inside domain should work normally (coverage for eval_core path)
        @test quadratic_interp(x, y, 1.0; extrap = ClampExtrap()) ≈ 1.0

        # :constant - derivatives return zero outside domain
        @test quadratic_interp(x, y, -0.5; extrap = ClampExtrap(), deriv = DerivOp(1)) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap = ClampExtrap(), deriv = DerivOp(1)) ≈ 0.0
        @test quadratic_interp(x, y, -0.5; extrap = ClampExtrap(), deriv = DerivOp(2)) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap = ClampExtrap(), deriv = DerivOp(2)) ≈ 0.0

        # :extension - extend the polynomial (right side)
        v_ext_right = quadratic_interp(x, y, 2.5; extrap = ExtendExtrap())
        @test isfinite(v_ext_right)

        # :extension - extend the polynomial (left side)
        v_ext_left = quadratic_interp(x, y, -0.5; extrap = ExtendExtrap())
        @test isfinite(v_ext_left)

        # :extension derivatives
        d1_left = quadratic_interp(x, y, -0.5; extrap = ExtendExtrap(), deriv = DerivOp(1))
        d2_left = quadratic_interp(x, y, -0.5; extrap = ExtendExtrap(), deriv = DerivOp(2))
        @test isfinite(d1_left)
        @test isfinite(d2_left)
    end

    @testset "quadratic_interp Float32" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        y32 = Float32[0.0, 1.0, 4.0, 9.0]

        # Use correct BC for exact x² interpolation
        result = quadratic_interp(x32, y32, 1.5f0; bc = Right(Deriv1(6.0f0)))
        @test result isa Float32
        @test result ≈ 2.25f0 rtol = 1.0e-5
    end

    @testset "quadratic_interp type stability" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = [0.0, 1.0, 4.0, 9.0]

        @test @inferred(quadratic_interp(x, y, 0.5)) isa Float64
        @test @inferred(quadratic_interp(x, y, 0.5; deriv = DerivOp(1))) isa Float64
        @test @inferred(quadratic_interp(x, y, 0.5; deriv = DerivOp(2))) isa Float64
    end

    @testset "quadratic_interp non-uniform grid" begin
        x = [0.0, 0.5, 1.5, 3.0]
        y = x .^ 2  # [0, 0.25, 2.25, 9]

        # At grid points (always exact)
        @test quadratic_interp(x, y, 0.0) ≈ 0.0
        @test quadratic_interp(x, y, 0.5) ≈ 0.25
        @test quadratic_interp(x, y, 1.5) ≈ 2.25
        @test quadratic_interp(x, y, 3.0) ≈ 9.0

        # Midpoints with correct BC (f'(3) = 6 for x²)
        @test quadratic_interp(x, y, 0.25; bc = Right(Deriv1(6.0))) ≈ 0.0625 rtol = 1.0e-10
        @test quadratic_interp(x, y, 2.0; bc = Right(Deriv1(6.0))) ≈ 4.0 rtol = 1.0e-10
    end

end

# ============================================================================
# Group 5: QuadraticInterpolant Tests
# ============================================================================
@testset "Quadratic Interpolation - Interpolant" begin

    @testset "QuadraticInterpolant construction" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        # 2-argument form returns QuadraticInterpolant
        itp = quadratic_interp(x, y)
        @test itp isa QuadraticInterpolant

        # With BC option
        itp2 = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))
        @test itp2 isa QuadraticInterpolant
    end

    @testset "QuadraticInterpolant scalar call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        # Grid points
        @test itp(0.0) ≈ 0.0
        @test itp(1.0) ≈ 1.0
        @test itp(2.0) ≈ 4.0
        @test itp(3.0) ≈ 9.0

        # Midpoints (exact with correct BC)
        @test itp(0.5) ≈ 0.25 rtol = 1.0e-10
        @test itp(1.5) ≈ 2.25 rtol = 1.0e-10
        @test itp(2.5) ≈ 6.25 rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant broadcast" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        # Broadcast
        result = itp.([0.5, 1.5, 2.5])
        @test result ≈ [0.25, 2.25, 6.25] rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant vector call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        # Vector call
        result = itp([0.5, 1.5, 2.5])
        @test result isa Vector{Float64}
        @test result ≈ [0.25, 2.25, 6.25] rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant in-place call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        out = zeros(3)
        itp(out, [0.5, 1.5, 2.5])
        @test out ≈ [0.25, 2.25, 6.25] rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant derivative call" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        # deriv keyword
        @test itp(1.5; deriv = DerivOp(1)) ≈ 3.0 rtol = 1.0e-10
        @test itp(1.5; deriv = DerivOp(2)) ≈ 2.0 rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant Float32" begin
        x32 = Float32[0.0, 1.0, 2.0, 3.0]
        y32 = x32 .^ 2

        itp = quadratic_interp(x32, y32; bc = Right(Deriv1(6.0f0)))
        @test itp isa QuadraticInterpolant{Float32}
        @test itp(1.5f0) isa Float32
        @test itp(1.5f0) ≈ 2.25f0 rtol = 1.0e-5
    end

    @testset "QuadraticInterpolant type stability" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

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
        y = x .^ 2

        # Create interpolant (precomputes coefficients)
        itp = quadratic_interp(x, y; bc = Right(Deriv1(2.0)))

        # Prime JIT
        for _ in 1:10
            itp(0.5)
        end

        # Scalar call should be zero-allocation
        allocs = @allocated itp(0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Derivative calls should also be zero-allocation
        for _ in 1:10
            itp(0.5; deriv = DerivOp(1))
            itp(0.5; deriv = DerivOp(2))
        end
        allocs_d1 = @allocated itp(0.5; deriv = DerivOp(1))
        allocs_d2 = @allocated itp(0.5; deriv = DerivOp(2))
        @test allocs_d1 <= ALLOC_THRESHOLD
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    @testset "interpolant in-place vector zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        xq = collect(range(0.1, 0.9, 100))
        out = zeros(100)

        itp = quadratic_interp(x, y; bc = Right(Deriv1(2.0)))

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
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(2.0)))
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
        y = x .^ 2

        # warm-up all code paths (deriv=DerivOp(0), 1, 2)
        quadratic_interp(x, y, 0.5)
        quadratic_interp(x, y, 0.5; deriv = DerivOp(1))
        quadratic_interp(x, y, 0.5; deriv = DerivOp(2))

        # Measure allocation for scalar one-shot
        allocs = @allocated quadratic_interp(x, y, 0.5)
        @test allocs <= ALLOC_THRESHOLD

        # Different query points should have same allocation
        allocs_other = @allocated quadratic_interp(x, y, 0.3)
        @test allocs_other <= ALLOC_THRESHOLD

        # Derivative calls should also be zero-allocation
        allocs_d1 = @allocated quadratic_interp(x, y, 0.5; deriv = DerivOp(1))
        allocs_d2 = @allocated quadratic_interp(x, y, 0.5; deriv = DerivOp(2))
        @test allocs_d1 <= ALLOC_THRESHOLD
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    @testset "one-shot vector in-place zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        y = x .^ 2
        xq = collect(range(0.1, 0.9, 100))
        out = similar(xq)

        # warmup all BC types and deriv values (single call each)
        quadratic_interp!(out, x, y, xq)
        quadratic_interp!(out, x, y, xq; bc = Left(Deriv1(2.0)))
        quadratic_interp!(out, x, y, xq; bc = Left(Deriv2(1.0)))
        quadratic_interp!(out, x, y, xq; bc = Right(Deriv1(2.0)))
        quadratic_interp!(out, x, y, xq; bc = Right(Deriv2(1.0)))
        quadratic_interp!(out, x, y, xq; deriv = DerivOp(1))
        quadratic_interp!(out, x, y, xq; deriv = DerivOp(2))

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
        alloc_d1 = @allocated quadratic_interp!(out, x, y, xq; deriv = DerivOp(1))
        alloc_d2 = @allocated quadratic_interp!(out, x, y, xq; deriv = DerivOp(2))

        @test alloc_d1 <= ALLOC_THRESHOLD
        @test alloc_d2 <= ALLOC_THRESHOLD
    end
end

# ============================================================================
# Group 7: Type Conversion Tests (Real → Float wrappers)
# ============================================================================
@testset "Quadratic Interpolation - Type Conversion" begin
    using FastInterpolations: _normalize_bc

    @testset "_normalize_bc same-type passthrough" begin
        # Same-type should return the same object (zero-cost)
        bc_left = Left(Deriv1(1.0))
        bc_right = Right(Deriv2(0.0))

        @test _normalize_bc(bc_left, Float64) === bc_left
        @test _normalize_bc(bc_right, Float64) === bc_right

        # Float32 same-type passthrough
        bc_left32 = Left(Deriv1(1.0f0))
        bc_right32 = Right(Deriv2(0.0f0))

        @test _normalize_bc(bc_left32, Float32) === bc_left32
        @test _normalize_bc(bc_right32, Float32) === bc_right32
    end

    @testset "_normalize_bc type conversion" begin
        # Float32 → Float64 conversion
        bc_left32 = Left(Deriv1(1.0f0))
        bc_right32 = Right(Deriv2(0.0f0))

        bc_left64 = _normalize_bc(bc_left32, Float64)
        bc_right64 = _normalize_bc(bc_right32, Float64)

        # Left/Right now only have B parameter (the inner BC type)
        @test bc_left64 isa Left{Deriv1{Float64}}
        @test bc_right64 isa Right{Deriv2{Float64}}
        @test bc_left64.bc.val ≈ 1.0
        @test bc_right64.bc.val ≈ 0.0

        # Float64 → Float32 conversion
        bc_left_f64 = Left(Deriv2(2.0))
        bc_right_f64 = Right(Deriv1(3.0))

        bc_left_f32 = _normalize_bc(bc_left_f64, Float32)
        bc_right_f32 = _normalize_bc(bc_right_f64, Float32)

        @test bc_left_f32 isa Left{Deriv2{Float32}}
        @test bc_right_f32 isa Right{Deriv1{Float32}}
    end

    @testset "_normalize_bc for QuadraticFit" begin
        using FastInterpolations: _promote_pointbc

        # QuadraticFit is now a non-parametric singleton (PolyFit{2})
        # _promote_pointbc returns the same type (no T parameter to promote)
        pf = QuadraticFit()
        pf_promoted = _promote_pointbc(pf, Float64)
        @test pf_promoted isa QuadraticFit
        @test pf_promoted === pf  # Same singleton instance

        pf32 = _promote_pointbc(pf, Float32)
        @test pf32 isa QuadraticFit
        @test pf32 === pf  # Same singleton instance

        # Left(QuadraticFit) promotion - Left only has B parameter
        bc_left = Left(QuadraticFit())
        bc_left32 = _normalize_bc(bc_left, Float32)
        @test bc_left32 isa Left{QuadraticFit}

        # Right(QuadraticFit) promotion
        bc_right = Right(QuadraticFit())
        bc_right32 = _normalize_bc(bc_right, Float32)
        @test bc_right32 isa Right{QuadraticFit}
    end

    @testset "_normalize_bc for MinCurvFit" begin
        # MinCurvFit is now a non-parametric singleton
        # _normalize_bc returns the same type
        mc = MinCurvFit()
        mc_promoted = _normalize_bc(mc, Float64)
        @test mc_promoted isa MinCurvFit
        @test mc_promoted === mc  # Same singleton instance

        mc32 = _normalize_bc(mc, Float32)
        @test mc32 isa MinCurvFit
        @test mc32 === mc  # Same singleton instance
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
        result2 = quadratic_interp(x, y, 1.5; bc = Right(Deriv1(6.0)))
        @test result2 ≈ 2.25 rtol = 1.0e-10

        # Scalar with Int query point
        result3 = quadratic_interp(x, y, 2)
        @test result3 ≈ 4.0

        # Derivatives with Int data
        d1 = quadratic_interp(x, y, 1.5; bc = Right(Deriv1(6.0)), deriv = DerivOp(1))
        @test d1 ≈ 3.0 rtol = 1.0e-10
    end

    @testset "quadratic_interp vector with Integer arrays" begin
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]
        xq = [0.5, 1.5, 2.5]

        # Allocating version
        result = quadratic_interp(x, y, xq; bc = Right(Deriv1(6.0)))
        @test result isa Vector{Float64}
        @test result ≈ [0.25, 2.25, 6.25] rtol = 1.0e-10

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

        quadratic_interp!(out, x, y, xq; bc = Right(Deriv1(6.0)))
        @test out ≈ [0.25, 2.25, 6.25] rtol = 1.0e-10

        # Integer query points
        xq_int = [1, 2]
        out2 = zeros(2)
        quadratic_interp!(out2, x, y, xq_int)
        @test out2 ≈ [1.0, 4.0]
    end

    @testset "QuadraticInterpolant Real scalar wrapper" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        # Call with Int (triggers Real wrapper, not the T method)
        result = itp(2)  # Int64, not Float64
        @test result isa Float64
        @test result ≈ 4.0

        # Derivative with Int query
        d1 = itp(2; deriv = DerivOp(1))
        @test d1 ≈ 4.0 rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant in-place with type conversion" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))

        # In-place with Integer query points (triggers type conversion path)
        out = zeros(3)
        xq_int = [1, 2, 3]  # Int64
        itp(out, xq_int)
        @test out ≈ [1.0, 4.0, 9.0]

        # Derivative with type conversion
        out2 = zeros(2)
        itp(out2, [1, 2]; deriv = DerivOp(1))
        @test out2 ≈ [2.0, 4.0] rtol = 1.0e-10
    end

    @testset "QuadraticInterpolant from Integer arrays (2-arg Real wrapper)" begin
        # Integer arrays trigger the Real wrapper for 2-argument form
        x = [0, 1, 2, 3]  # Int64
        y = [0, 1, 4, 9]  # Int64

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))
        @test itp isa QuadraticInterpolant{Float64}

        @test itp(1.5) ≈ 2.25 rtol = 1.0e-10
        @test itp(0.5) ≈ 0.25 rtol = 1.0e-10
    end

    @testset "BC type promotion with Real data" begin
        # Int data with Int BC (both promoted to Float64)
        x_int = [0, 1, 2, 3]
        y_int = [0, 1, 4, 9]

        # BC with Int value triggers promotion
        result = quadratic_interp(x_int, y_int, 1.5; bc = Left(Deriv2(0)))
        @test result isa Float64
        @test isfinite(result)

        # BC with Float64 value for Int data
        result2 = quadratic_interp(x_int, y_int, 1.5; bc = Right(Deriv1(6.0)))
        @test result2 ≈ 2.25 rtol = 1.0e-10
    end

end

@testset "Quadratic Interpolation - DerivativeView" begin

    @testset "deriv1 view" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))
        d1 = deriv1(itp)

        # d1(x) = S'(x) = 2x for f(x)=x²
        @test d1(0.0) ≈ 0.0 rtol = 1.0e-10
        @test d1(1.0) ≈ 2.0 rtol = 1.0e-10
        @test d1(1.5) ≈ 3.0 rtol = 1.0e-10
        @test d1(2.0) ≈ 4.0 rtol = 1.0e-10
        @test d1(3.0) ≈ 6.0 rtol = 1.0e-10
    end

    @testset "deriv2 view" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))
        d2 = deriv2(itp)

        # d2(x) = S''(x) = 2 (constant) for f(x)=x²
        @test d2(0.5) ≈ 2.0 rtol = 1.0e-10
        @test d2(1.5) ≈ 2.0 rtol = 1.0e-10
        @test d2(2.5) ≈ 2.0 rtol = 1.0e-10
    end

    @testset "deriv1 broadcast" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = Right(Deriv1(6.0)))
        d1 = deriv1(itp)

        result = d1.([0.5, 1.5, 2.5])
        @test result ≈ [1.0, 3.0, 5.0] rtol = 1.0e-10
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
        f(t) = a * t^2 + b * t + c
        f_d1(t) = 2 * a * t + b  # f'(x) = 4x - 3
        f_d2 = 2 * a           # f''(x) = 4 (constant)

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
                ("Right(Deriv2)", bc_right_d2),
            ]
            @testset "$name" begin
                # Value interpolation
                result = quadratic_interp(x, y, xq; bc = bc)
                @test result ≈ expected rtol = 1.0e-12 atol = 1.0e-14

                # First derivative
                result_d1 = quadratic_interp(x, y, xq; bc = bc, deriv = DerivOp(1))
                @test result_d1 ≈ expected_d1 rtol = 1.0e-12 atol = 1.0e-14

                # Second derivative
                result_d2 = quadratic_interp(x, y, xq; bc = bc, deriv = DerivOp(2))
                @test result_d2 ≈ expected_d2 rtol = 1.0e-12 atol = 1.0e-14
            end
        end
    end

    @testset "all BCs produce identical spline for quadratic" begin
        # Highly non-uniform grid
        x = [0.0, 0.1, 0.5, 2.0, 2.1, 5.0]

        # f(x) = x² (simpler case)
        y = x .^ 2

        # True derivatives
        d1_left = 2 * first(x)   # 0
        d1_right = 2 * last(x)   # 10
        d2_val = 2.0             # constant

        # Create interpolants with all BC variants
        itp_left_d1 = quadratic_interp(x, y; bc = Left(Deriv1(d1_left)))
        itp_left_d2 = quadratic_interp(x, y; bc = Left(Deriv2(d2_val)))
        itp_right_d1 = quadratic_interp(x, y; bc = Right(Deriv1(d1_right)))
        itp_right_d2 = quadratic_interp(x, y; bc = Right(Deriv2(d2_val)))

        # Query at many points
        xq = range(0.0, 5.0, 50)

        # All should produce identical results
        result_ld1 = itp_left_d1.(xq)
        result_ld2 = itp_left_d2.(xq)
        result_rd1 = itp_right_d1.(xq)
        result_rd2 = itp_right_d2.(xq)

        @test result_ld1 ≈ result_ld2 rtol = 1.0e-12
        @test result_ld1 ≈ result_rd1 rtol = 1.0e-12
        @test result_ld1 ≈ result_rd2 rtol = 1.0e-12

        # All should match x²
        expected = collect(xq) .^ 2
        @test result_ld1 ≈ expected rtol = 1.0e-12
    end

    @testset "edge cases: boundary evaluation" begin
        # Grid with extreme spacing variation
        x = [0.0, 0.01, 1.0, 1.01, 10.0]  # tiny + large intervals

        # f(x) = -x² + 5x
        f(t) = -t^2 + 5 * t
        f_d1(t) = -2 * t + 5
        f_d2 = -2.0

        y = f.(x)

        bc = Right(Deriv1(f_d1(last(x))))  # f'(10) = -15

        # Test exact boundary points
        @test quadratic_interp(x, y, 0.0; bc = bc) ≈ f(0.0) rtol = 1.0e-12
        @test quadratic_interp(x, y, 10.0; bc = bc) ≈ f(10.0) rtol = 1.0e-12

        # Test points very close to boundaries
        @test quadratic_interp(x, y, 1.0e-10; bc = bc) ≈ f(1.0e-10) rtol = 1.0e-10
        @test quadratic_interp(x, y, 10.0 - 1.0e-10; bc = bc) ≈ f(10.0 - 1.0e-10) rtol = 1.0e-10

        # Test mid-interval points
        @test quadratic_interp(x, y, 0.005; bc = bc) ≈ f(0.005) rtol = 1.0e-12
        @test quadratic_interp(x, y, 5.0; bc = bc) ≈ f(5.0) atol = 1.0e-14  # f(5)=0, need atol
    end

    @testset "Float32 precision on non-uniform grid" begin
        x32 = Float32[0.0, 0.5, 1.5, 3.0]
        y32 = x32 .^ 2

        d1_right = 2 * last(x32)  # 6.0f0

        itp = quadratic_interp(x32, y32; bc = Right(Deriv1(d1_right)))

        # Should be exact within Float32 precision
        @test itp(1.0f0) ≈ 1.0f0 rtol = 1.0e-6
        @test itp(2.0f0) ≈ 4.0f0 rtol = 1.0e-6
        @test itp(0.25f0) ≈ 0.0625f0 rtol = 1.0e-6
    end

    @testset "consistency: scalar vs vector API" begin
        x = [0.0, 0.4, 1.1, 2.0, 3.3]
        y = x .^ 2

        bc = Left(Deriv2(2.0))
        xq = [0.2, 0.8, 1.5, 2.5, 3.0]

        # Scalar API
        results_scalar = [quadratic_interp(x, y, xi; bc = bc) for xi in xq]

        # Vector API (allocating)
        results_vector = quadratic_interp(x, y, xq; bc = bc)

        # In-place API
        results_inplace = zeros(length(xq))
        quadratic_interp!(results_inplace, x, y, xq; bc = bc)

        # Interpolant API
        itp = quadratic_interp(x, y; bc = bc)
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
        f(t) = a * t^2 + b * t + c
        f_d1(t) = 2 * a * t + b  # f'(x) = 4x - 3
        f_d2 = 2 * a           # f''(x) = 4 (constant)

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
            ("Right(Deriv2)", Right(Deriv2(d2_val))),
        ]

        # Extrapolation query points (outside domain)
        xq_left = [-1.0, -0.5, -0.1]   # left of x[1]=0
        xq_right = [4.0, 5.0, 6.0]      # right of x[end]=3.5

        for (name, bc) in bcs
            @testset "$name - one-shot API" begin
                # Left extrapolation - value
                for xi in xq_left
                    result = quadratic_interp(x, y, xi; bc = bc, extrap = ExtendExtrap())
                    @test result ≈ f(xi) rtol = 1.0e-12 atol = 1.0e-14
                end

                # Right extrapolation - value
                for xi in xq_right
                    result = quadratic_interp(x, y, xi; bc = bc, extrap = ExtendExtrap())
                    @test result ≈ f(xi) rtol = 1.0e-12 atol = 1.0e-14
                end

                # Left extrapolation - derivatives
                for xi in xq_left
                    d1 = quadratic_interp(x, y, xi; bc = bc, extrap = ExtendExtrap(), deriv = DerivOp(1))
                    d2 = quadratic_interp(x, y, xi; bc = bc, extrap = ExtendExtrap(), deriv = DerivOp(2))
                    @test d1 ≈ f_d1(xi) rtol = 1.0e-12 atol = 1.0e-14
                    @test d2 ≈ f_d2 rtol = 1.0e-12 atol = 1.0e-14
                end

                # Right extrapolation - derivatives
                for xi in xq_right
                    d1 = quadratic_interp(x, y, xi; bc = bc, extrap = ExtendExtrap(), deriv = DerivOp(1))
                    d2 = quadratic_interp(x, y, xi; bc = bc, extrap = ExtendExtrap(), deriv = DerivOp(2))
                    @test d1 ≈ f_d1(xi) rtol = 1.0e-12 atol = 1.0e-14
                    @test d2 ≈ f_d2 rtol = 1.0e-12 atol = 1.0e-14
                end
            end

            @testset "$name - interpolant API" begin
                itp = quadratic_interp(x, y; bc = bc, extrap = ExtendExtrap())
                d1_view = deriv1(itp)
                d2_view = deriv2(itp)

                # Left extrapolation
                for xi in xq_left
                    @test itp(xi) ≈ f(xi) rtol = 1.0e-12 atol = 1.0e-14
                    @test d1_view(xi) ≈ f_d1(xi) rtol = 1.0e-12 atol = 1.0e-14
                    @test d2_view(xi) ≈ f_d2 rtol = 1.0e-12 atol = 1.0e-14
                end

                # Right extrapolation
                for xi in xq_right
                    @test itp(xi) ≈ f(xi) rtol = 1.0e-12 atol = 1.0e-14
                    @test d1_view(xi) ≈ f_d1(xi) rtol = 1.0e-12 atol = 1.0e-14
                    @test d2_view(xi) ≈ f_d2 rtol = 1.0e-12 atol = 1.0e-14
                end
            end
        end
    end

    @testset "extension C0 continuity at boundary" begin
        # The extension polynomial must match exactly at the boundary
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2  # f(x) = x²

        bc = Right(Deriv1(6.0))  # f'(3) = 6

        # Value at boundary should match from both sides
        val_inside = quadratic_interp(x, y, 3.0; bc = bc)
        val_outside = quadratic_interp(x, y, 3.0 + 1.0e-10; bc = bc, extrap = ExtendExtrap())
        @test val_inside ≈ val_outside rtol = 1.0e-8

        # Left boundary (value is 0.0, so use atol instead of rtol)
        val_inside_left = quadratic_interp(x, y, 0.0; bc = bc)
        val_outside_left = quadratic_interp(x, y, -1.0e-10; bc = bc, extrap = ExtendExtrap())
        @test val_inside_left ≈ val_outside_left atol = 1.0e-8
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
        result = quadratic_interp(x, y, xq; bc = bc, extrap = ExtendExtrap())
        @test result ≈ expected rtol = 1.0e-12

        # In-place API
        out = zeros(length(xq))
        quadratic_interp!(out, x, y, xq; bc = bc, extrap = ExtendExtrap())
        @test out ≈ expected rtol = 1.0e-12

        # Interpolant vector call
        itp = quadratic_interp(x, y; bc = bc, extrap = ExtendExtrap())
        @test itp(xq) ≈ expected rtol = 1.0e-12
    end

end

# ============================================================================
# Group 11: MinCurvFit API Integration Tests (Phase 3)
# ============================================================================
@testset "Quadratic Interpolation - MinCurvFit API" begin

    @testset "quadratic_interp scalar with MinCurvFit" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        # Grid points should be exact
        @test quadratic_interp(x, y, 0.0; bc = MinCurvFit()) ≈ 0.0
        @test quadratic_interp(x, y, 1.0; bc = MinCurvFit()) ≈ 1.0
        @test quadratic_interp(x, y, 2.0; bc = MinCurvFit()) ≈ 4.0
        @test quadratic_interp(x, y, 3.0; bc = MinCurvFit()) ≈ 9.0

        # Interior points
        @test isfinite(quadratic_interp(x, y, 0.5; bc = MinCurvFit()))
        @test isfinite(quadratic_interp(x, y, 1.5; bc = MinCurvFit()))
        @test isfinite(quadratic_interp(x, y, 2.5; bc = MinCurvFit()))
    end

    @testset "quadratic_interp vector with MinCurvFit" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2
        xq = [0.5, 1.5, 2.5]

        result = quadratic_interp(x, y, xq; bc = MinCurvFit())
        @test length(result) == 3
        @test all(isfinite, result)
    end

    @testset "quadratic_interp! with MinCurvFit" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2
        xq = [0.5, 1.5, 2.5]
        out = zeros(3)

        quadratic_interp!(out, x, y, xq; bc = MinCurvFit())
        @test all(isfinite, out)
    end

    @testset "QuadraticInterpolant with MinCurvFit" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        itp = quadratic_interp(x, y; bc = MinCurvFit())
        @test itp isa QuadraticInterpolant

        # Scalar evaluation
        @test isfinite(itp(0.5))
        @test isfinite(itp(1.5))

        # Grid points exact
        @test itp(0.0) ≈ 0.0
        @test itp(1.0) ≈ 1.0
        @test itp(3.0) ≈ 9.0
    end

    @testset "MinCurvFit derivatives" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        # First derivative (should be finite)
        d1 = quadratic_interp(x, y, 1.5; bc = MinCurvFit(), deriv = DerivOp(1))
        @test isfinite(d1)

        # Second derivative (should be finite)
        d2 = quadratic_interp(x, y, 1.5; bc = MinCurvFit(), deriv = DerivOp(2))
        @test isfinite(d2)

        # Interpolant derivatives
        itp = quadratic_interp(x, y; bc = MinCurvFit())
        @test isfinite(itp(1.5; deriv = DerivOp(1)))
        @test isfinite(itp(1.5; deriv = DerivOp(2)))
    end

    @testset "MinCurvFit type promotion (Real → Float)" begin
        # Int arrays
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]

        result = quadratic_interp(x, y, 1.5; bc = MinCurvFit())
        @test isfinite(result)
        @test result isa Float64

        # Interpolant with Int arrays
        itp = quadratic_interp(x, y; bc = MinCurvFit())
        @test itp isa QuadraticInterpolant{Float64}
    end

    @testset "MinCurvFit Float32 support" begin
        x = Float32[0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        result = quadratic_interp(x, y, 1.5f0; bc = MinCurvFit())
        @test result isa Float32
        @test isfinite(result)

        itp = quadratic_interp(x, y; bc = MinCurvFit())
        @test itp isa QuadraticInterpolant{Float32}
    end

    @testset "MinCurvFit extrapolation modes" begin
        x = [0.0, 1.0, 2.0, 3.0]
        y = x .^ 2

        # :extension mode
        itp_ext = quadratic_interp(x, y; bc = MinCurvFit(), extrap = ExtendExtrap())
        @test isfinite(itp_ext(-0.5))  # outside left
        @test isfinite(itp_ext(3.5))   # outside right

        # :constant mode
        itp_const = quadratic_interp(x, y; bc = MinCurvFit(), extrap = ClampExtrap())
        @test itp_const(-0.5) ≈ 0.0    # clamps to y[1]
        @test itp_const(3.5) ≈ 9.0     # clamps to y[end]
    end

end

# ============================================================================
# Group 12: MinCurvFit Mathematical Verification (Phase 4)
# ============================================================================
@testset "Quadratic Interpolation - MinCurvFit Mathematical Verification" begin
    using FastInterpolations: _fill_slopes!, _compute_quadratic_secants!

    # ========================================
    # Test 1: d[1] Optimality Verification
    # ========================================
    # User requirement: Perturb d[1] by ±δ and verify curvature increases
    @testset "d[1] optimality - perturbation increases curvature" begin
        # Curvature objective function: Σ (s[i] - d[i])²/h[i]
        # MinCurvFit finds the d[1] that minimizes this

        @testset "uniform grid" begin
            s = [1.0, 3.0, 5.0]  # secants for x² on [0,1,2,3]
            h = [1.0, 1.0, 1.0]
            d_opt = zeros(4)
            x_dummy = [0.0, 1.0, 2.0, 3.0]
            y_dummy = x_dummy .^ 2

            _fill_slopes!(d_opt, s, h, MinCurvFit(), x_dummy, y_dummy)

            # Compute optimal curvature
            curvature_opt = sum((s[i] - d_opt[i])^2 / h[i] for i in 1:3)

            # Perturb d[1] and verify curvature increases
            for δ in [0.01, 0.1, 0.5, 1.0, -0.01, -0.1, -0.5, -1.0]
                d_perturbed = zeros(4)
                # Forward recurrence with perturbed d[1]
                d_perturbed[1] = d_opt[1] + δ
                for i in 1:3
                    d_perturbed[i + 1] = 2 * s[i] - d_perturbed[i]
                end

                curvature_perturbed = sum((s[i] - d_perturbed[i])^2 / h[i] for i in 1:3)

                @test curvature_perturbed >= curvature_opt - 1.0e-12  # allow tiny numerical error
            end
        end

        @testset "non-uniform grid" begin
            s = [2.0, 1.0, 3.0, 0.5]  # varying secants
            h = [0.5, 1.5, 1.0, 2.0]  # non-uniform spacing
            n = length(h) + 1
            d_opt = zeros(n)
            x_dummy = cumsum([0.0; h])  # [0, 0.5, 2.0, 3.0, 5.0]
            y_dummy = zeros(n)

            _fill_slopes!(d_opt, s, h, MinCurvFit(), x_dummy, y_dummy)

            # Compute optimal curvature
            curvature_opt = sum((s[i] - d_opt[i])^2 / h[i] for i in 1:length(s))

            # Perturb d[1] and verify curvature increases
            for δ in [0.1, 0.5, -0.1, -0.5]
                d_perturbed = zeros(n)
                d_perturbed[1] = d_opt[1] + δ
                for i in 1:length(s)
                    d_perturbed[i + 1] = 2 * s[i] - d_perturbed[i]
                end

                curvature_perturbed = sum((s[i] - d_perturbed[i])^2 / h[i] for i in 1:length(s))

                @test curvature_perturbed >= curvature_opt - 1.0e-12
            end
        end

        @testset "gradient is zero at optimum" begin
            # At the optimal d[1], the gradient of the objective should be zero
            # df/d(d[1]) = -2 * Σ α[i]*(s[i] - d[i])/h[i] where α[i] = (-1)^(i+1)
            s = [1.0, 3.0, 5.0, 7.0]
            h = [0.5, 1.0, 1.5, 0.8]
            n = length(h) + 1
            d_opt = zeros(n)
            x_dummy = cumsum([0.0; h])
            y_dummy = zeros(n)

            _fill_slopes!(d_opt, s, h, MinCurvFit(), x_dummy, y_dummy)

            # Compute gradient at optimal point
            gradient = 0.0
            sign = 1.0  # α[1] = (-1)^(1+1) = +1
            for i in 1:length(s)
                gradient += sign * (s[i] - d_opt[i]) / h[i]
                sign = -sign
            end
            gradient *= -2

            @test abs(gradient) < 1.0e-12  # gradient should be (near) zero
        end
    end

    # ========================================
    # Test 2: MinCurvFit Curvature Minimization Behavior
    # ========================================
    # IMPORTANT: MinCurvFit does NOT reproduce exact quadratic polynomials!
    # It minimizes Σ(s[i] - d[i])²/h[i], which trades exactness for smoothness.
    # For f(x) = x² on [0,1,2,3]: exact d[1]=0 gives curvature=12,
    # but MinCurvFit optimal d[1]=1/3 gives curvature≈10.67 (lower!)
    @testset "MinCurvFit curvature minimization behavior" begin

        @testset "f(x) = x² - MinCurvFit trades exactness for smoothness" begin
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = x .^ 2

            # MinCurvFit does NOT give exact x² - it optimizes curvature instead
            xq = [0.5, 1.5, 2.5, 3.5]

            result = quadratic_interp(x, y, xq; bc = MinCurvFit())

            # Should be finite and reasonable
            @test all(isfinite, result)
            # Grid points are always exact
            @test quadratic_interp(x, y, 1.0; bc = MinCurvFit()) ≈ 1.0
            @test quadratic_interp(x, y, 2.0; bc = MinCurvFit()) ≈ 4.0
        end

        @testset "MinCurvFit has lower curvature than exact BC for quadratic data" begin
            # This test demonstrates that MinCurvFit minimizes curvature,
            # which means it does NOT reproduce exact quadratic polynomials
            x = [0.0, 0.3, 0.8, 1.5, 2.5, 3.0]  # non-uniform
            a, b, c = 2.0, -3.0, 1.0
            f(t) = a * t^2 + b * t + c
            f_d1(t) = 2 * a * t + b
            f_d2 = 2 * a

            y = f.(x)

            # Compute secants and spacing
            h = diff(x)
            inv_h = 1.0 ./ h
            s = diff(y) .* inv_h
            n = length(x)

            # MinCurvFit curvature
            d_smooth = zeros(n)
            _fill_slopes!(d_smooth, s, h, MinCurvFit(), x, y)
            curvature_smooth = sum((s[i] - d_smooth[i])^2 / h[i] for i in 1:length(s))

            # Exact BC (Left(Deriv1)) curvature - uses correct d[1] = f'(x[1])
            d_exact = zeros(n)
            _fill_slopes!(d_exact, s, h, Left(Deriv1(f_d1(first(x)))), x, y)
            curvature_exact = sum((s[i] - d_exact[i])^2 / h[i] for i in 1:length(s))

            # MinCurvFit should have lower or equal curvature
            @test curvature_smooth <= curvature_exact + 1.0e-10
        end

        @testset "linear function f(x) = 2x + 1 (exact, zero curvature)" begin
            # Linear function: s[i] = d[i] for all i, so curvature = 0
            # MinCurvFit will give exact results because minimizing zero is zero
            x = [0.0, 1.0, 2.0, 3.0]
            y = 2.0 .* x .+ 1.0

            xq = [0.5, 1.5, 2.5]
            expected = 2.0 .* xq .+ 1.0

            result = quadratic_interp(x, y, xq; bc = MinCurvFit())
            @test result ≈ expected rtol = 1.0e-12

            # Second derivative should be zero (linear function)
            d2 = quadratic_interp(x, y, xq; bc = MinCurvFit(), deriv = DerivOp(2))
            @test all(abs.(d2) .< 1.0e-12)
        end

        @testset "constant function f(x) = 5 (exact, zero curvature)" begin
            # Constant function: all s[i] = 0, all d[i] = 0, curvature = 0
            x = [0.0, 1.0, 2.0, 3.0]
            y = fill(5.0, 4)

            xq = [0.5, 1.5, 2.5]
            expected = fill(5.0, 3)

            result = quadratic_interp(x, y, xq; bc = MinCurvFit())
            @test result ≈ expected rtol = 1.0e-12

            # All derivatives should be zero
            d1 = quadratic_interp(x, y, xq; bc = MinCurvFit(), deriv = DerivOp(1))
            d2 = quadratic_interp(x, y, xq; bc = MinCurvFit(), deriv = DerivOp(2))
            @test all(abs.(d1) .< 1.0e-12)
            @test all(abs.(d2) .< 1.0e-12)
        end
    end

    # ========================================
    # Test 3: Correct BCs Give Exact Quadratic (MinCurvFit Differs)
    # ========================================
    @testset "correct BCs give exact quadratic, MinCurvFit minimizes curvature" begin
        # For a quadratic polynomial, explicit BCs with correct derivative values
        # produce exact results. MinCurvFit optimizes curvature instead.
        x = [0.0, 0.3, 0.8, 1.5, 2.5, 3.0]

        # f(x) = 2x² - 3x + 1
        a, b, c = 2.0, -3.0, 1.0
        f(t) = a * t^2 + b * t + c
        f_d1(t) = 2 * a * t + b
        f_d2 = 2 * a

        y = f.(x)

        # Correct BC values
        d1_left = f_d1(first(x))
        d1_right = f_d1(last(x))

        # Explicit BCs with correct values should give exact results
        bc_left_d1 = Left(Deriv1(d1_left))
        bc_left_d2 = Left(Deriv2(f_d2))
        bc_right_d1 = Right(Deriv1(d1_right))
        bc_right_d2 = Right(Deriv2(f_d2))

        xq = range(0.0, 3.0, 20)
        expected = f.(xq)

        # Explicit BCs with correct values produce exact results
        for bc in [bc_left_d1, bc_left_d2, bc_right_d1, bc_right_d2]
            result = quadratic_interp(x, y, collect(xq); bc = bc)
            @test result ≈ expected rtol = 1.0e-12
        end

        # MinCurvFit produces finite, reasonable results (but not exact)
        result_smooth = quadratic_interp(x, y, collect(xq); bc = MinCurvFit())
        @test all(isfinite, result_smooth)
    end

    # ========================================
    # Test 4: MinCurvFit vs Default BC Difference
    # ========================================
    @testset "MinCurvFit differs from Left(Deriv2(0)) for curved data" begin
        # Non-quadratic data where MinCurvFit should differ from default Left(Deriv2(0))
        x = [0.0, 0.3, 0.8, 1.5, 2.5, 3.0, 4.0]
        y = [0.0, 0.8, 1.2, 0.9, 0.3, 0.6, 1.0]  # curved data

        # Default BC: Left(Deriv2(0)) forces first interval to be linear
        val_default = quadratic_interp(x, y, 0.15; bc = Left(Deriv2(0.0)))

        # MinCurvFit: globally smooth, doesn't force linearity
        val_smooth = quadratic_interp(x, y, 0.15; bc = MinCurvFit())

        # They should be different for non-quadratic data
        @test val_default != val_smooth

        # Both should be within reasonable interpolation bounds
        @test 0.0 < val_default < 1.0
        @test 0.0 < val_smooth < 1.0
    end

    # ========================================
    # Test 5: MinCurvFit Edge Cases
    # ========================================
    @testset "MinCurvFit edge cases" begin

        @testset "n=2 (minimum grid - single segment)" begin
            x = [0.0, 1.0]
            y = [0.0, 1.0]

            # Should not error
            result = quadratic_interp(x, y, 0.5; bc = MinCurvFit())
            @test isfinite(result)
            @test result ≈ 0.5  # linear interpolation (zero curvature optimal)

            # Create interpolant
            itp = quadratic_interp(x, y; bc = MinCurvFit())
            @test itp(0.5) ≈ 0.5
        end

        @testset "extreme spacing variation" begin
            # Tiny + large intervals
            x = [0.0, 0.001, 1.0, 1.001, 10.0]
            y = x .^ 2  # f(x) = x²

            # Should handle extreme spacing without numerical issues
            result = quadratic_interp(x, y, 5.0; bc = MinCurvFit())
            @test isfinite(result)
            # MinCurvFit optimizes curvature, not exact x² reproduction
            # Just verify it's in a reasonable range
            @test 20.0 < result < 30.0

            # Query near tiny intervals - should be finite and reasonable
            result_tiny = quadratic_interp(x, y, 0.0005; bc = MinCurvFit())
            @test isfinite(result_tiny)
            @test result_tiny >= 0.0  # should be non-negative for x² data
        end

        @testset "Float32 precision" begin
            x32 = Float32[0.0, 1.0, 2.0, 3.0]
            y32 = x32 .^ 2

            itp = quadratic_interp(x32, y32; bc = MinCurvFit())
            @test itp isa QuadraticInterpolant{Float32}

            # MinCurvFit doesn't give exact x², but should be finite and reasonable
            @test isfinite(itp(1.5f0))
            @test isfinite(itp(0.5f0))
            # Grid points are always exact
            @test itp(1.0f0) ≈ 1.0f0
            @test itp(2.0f0) ≈ 4.0f0
        end

        @testset "many points (large n)" begin
            x = collect(range(0.0, 10.0, 101))  # n = 101 points
            y = x .^ 2

            # With many points, MinCurvFit should be reasonably accurate
            # (curvature optimization converges toward exact for dense grids)
            xq = [2.5, 5.0, 7.5]
            expected = xq .^ 2

            result = quadratic_interp(x, y, xq; bc = MinCurvFit())
            # Allow some tolerance since MinCurvFit trades exactness for smoothness
            @test all(isfinite, result)
            # With dense grid, should be quite close to exact
            @test result ≈ expected rtol = 0.01  # 1% tolerance
        end
    end

    # ========================================
    # Test 6: MinCurvFit Curvature is Minimal
    # ========================================
    @testset "MinCurvFit produces minimal or equal curvature" begin
        # MinCurvFit should produce total curvature <= any other BC
        x = [0.0, 0.5, 1.5, 2.5, 3.0]
        y = [1.0, 2.5, 1.8, 3.2, 2.0]  # arbitrary curved data

        # Compute curvature for various BCs
        # Curvature = Σ a[i]² * h[i] where a[i] = (s[i] - d[i]) / h[i]
        # Equivalent to Σ (s[i] - d[i])² / h[i]

        h = diff(x)
        inv_h = 1.0 ./ h
        s = diff(y) .* inv_h
        n = length(x)

        # MinCurvFit curvature
        d_smooth = zeros(n)
        _fill_slopes!(d_smooth, s, h, MinCurvFit(), x, y)
        curvature_smooth = sum((s[i] - d_smooth[i])^2 / h[i] for i in 1:length(s))

        # Left(Deriv2(0)) curvature
        d_left = zeros(n)
        _fill_slopes!(d_left, s, h, Left(Deriv2(0.0)), x, y)
        curvature_left = sum((s[i] - d_left[i])^2 / h[i] for i in 1:length(s))

        # Right(Deriv2(0)) curvature
        d_right = zeros(n)
        _fill_slopes!(d_right, s, h, Right(Deriv2(0.0)), x, y)
        curvature_right = sum((s[i] - d_right[i])^2 / h[i] for i in 1:length(s))

        # MinCurvFit should have minimal curvature
        @test curvature_smooth <= curvature_left + 1.0e-10
        @test curvature_smooth <= curvature_right + 1.0e-10
    end

end


# ============================================================================
# Group 13: QuadraticFit BC Type Tests (Phase 2)
# ============================================================================
@testset "Quadratic Interpolation - QuadraticFit BC Type" begin
    using FastInterpolations: _fill_slopes!

    @testset "QuadraticFit type construction" begin
        @testset "constructor (non-parametric singleton)" begin
            bc = QuadraticFit()
            # QuadraticFit is now PolyFit{2} - a non-parametric singleton
            @test bc isa QuadraticFit
            @test bc isa PolyFit{2}
            # Type-Free design: PointBC and AbstractBC have no type parameters
            @test bc isa PointBC
            @test bc isa AbstractBC
        end

        @testset "type alias identity" begin
            # QuadraticFit is exactly PolyFit{2}
            @test QuadraticFit === PolyFit{2}
            @test QuadraticFit() isa PolyFit{2}
        end

        @testset "type stability" begin
            @test @inferred(QuadraticFit()) isa QuadraticFit
            @test @inferred(PolyFit{2}()) isa QuadraticFit
        end
    end

    @testset "Left/Right wrappers accept QuadraticFit" begin
        # Left/Right now only have B parameter (the inner BC type)
        @test Left(QuadraticFit()) isa Left{QuadraticFit}
        @test Right(QuadraticFit()) isa Right{QuadraticFit}
        @test Left(QuadraticFit()) isa Left{PolyFit{2}}
        @test Right(QuadraticFit()) isa Right{PolyFit{2}}
    end
end


# ============================================================================
# Group 14: QuadraticFit Polynomial Reproduction Tests (Phase 2)
# ============================================================================
@testset "Quadratic Interpolation - QuadraticFit Polynomial Reproduction" begin
    using FastInterpolations: _fill_slopes!

    @testset "Left(QuadraticFit()) uniform grid" begin
        # f(x) = x² on uniform grid
        @testset "f(x) = x² reproduction" begin
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = x .^ 2
            itp = quadratic_interp(x, y; bc = Left(QuadraticFit()))

            # Should reproduce exactly at midpoints
            @test itp(0.5) ≈ 0.5^2 atol = 1.0e-12
            @test itp(1.5) ≈ 1.5^2 atol = 1.0e-12
            @test itp(2.5) ≈ 2.5^2 atol = 1.0e-12
            @test itp(3.5) ≈ 3.5^2 atol = 1.0e-12

            # And at arbitrary points
            @test itp(0.25) ≈ 0.25^2 atol = 1.0e-12
            @test itp(2.7) ≈ 2.7^2 atol = 1.0e-12
        end

        # General quadratic: f(x) = 2x² - 3x + 1
        @testset "f(x) = 2x² - 3x + 1 reproduction" begin
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = @. 2 * x^2 - 3 * x + 1
            itp = quadratic_interp(x, y; bc = Left(QuadraticFit()))

            f(t) = 2 * t^2 - 3 * t + 1
            @test itp(0.5) ≈ f(0.5) atol = 1.0e-12
            @test itp(1.5) ≈ f(1.5) atol = 1.0e-12
            @test itp(2.5) ≈ f(2.5) atol = 1.0e-12
        end
    end

    @testset "Right(QuadraticFit()) uniform grid" begin
        @testset "f(x) = x² reproduction" begin
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = x .^ 2
            itp = quadratic_interp(x, y; bc = Right(QuadraticFit()))

            @test itp(0.5) ≈ 0.5^2 atol = 1.0e-12
            @test itp(1.5) ≈ 1.5^2 atol = 1.0e-12
            @test itp(2.5) ≈ 2.5^2 atol = 1.0e-12
            @test itp(3.5) ≈ 3.5^2 atol = 1.0e-12
        end
    end

    @testset "Non-uniform grid polynomial reproduction" begin
        @testset "f(x) = x² on non-uniform grid" begin
            x = [0.0, 0.5, 1.5, 3.0, 5.0]
            y = x .^ 2
            itp_left = quadratic_interp(x, y; bc = Left(QuadraticFit()))
            itp_right = quadratic_interp(x, y; bc = Right(QuadraticFit()))

            # Test at various points
            for t in [0.25, 0.75, 1.0, 2.0, 4.0]
                @test itp_left(t) ≈ t^2 atol = 1.0e-11
                @test itp_right(t) ≈ t^2 atol = 1.0e-11
            end
        end

        @testset "f(x) = -x² + 4x on non-uniform grid" begin
            x = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0]
            f(t) = -t^2 + 4 * t
            y = f.(x)
            itp = quadratic_interp(x, y; bc = Left(QuadraticFit()))

            for t in [0.5, 1.5, 3.0, 5.0, 6.5]
                @test itp(t) ≈ f(t) atol = 1.0e-11
            end
        end
    end

    @testset "Edge cases" begin
        @testset "n=3 (minimum for QuadraticFit)" begin
            x = [0.0, 1.0, 2.0]
            y = x .^ 2
            itp_left = quadratic_interp(x, y; bc = Left(QuadraticFit()))
            itp_right = quadratic_interp(x, y; bc = Right(QuadraticFit()))

            @test itp_left(0.5) ≈ 0.5^2 atol = 1.0e-12
            @test itp_left(1.5) ≈ 1.5^2 atol = 1.0e-12
            @test itp_right(0.5) ≈ 0.5^2 atol = 1.0e-12
            @test itp_right(1.5) ≈ 1.5^2 atol = 1.0e-12
        end

        @testset "n=2 requires Deriv1/Deriv2 (QuadraticFit needs 3+ points)" begin
            x = [0.0, 1.0]
            y = [0.0, 1.0]
            # QuadraticFit (PolyFit{2}) requires 3 points to estimate derivative
            @test_throws ArgumentError quadratic_interp(x, y; bc = Left(QuadraticFit()))
            @test_throws ArgumentError quadratic_interp(x, y; bc = Right(QuadraticFit()))

            # With explicit Deriv1 BC, n=2 works fine (linear interpolation)
            itp_left = quadratic_interp(x, y; bc = Left(Deriv1(1.0)))  # slope 1
            itp_right = quadratic_interp(x, y; bc = Right(Deriv1(1.0)))
            @test itp_left(0.5) ≈ 0.5 atol = 1.0e-12
            @test itp_right(0.5) ≈ 0.5 atol = 1.0e-12
        end
    end

    @testset "Derivatives on polynomial data" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = x .^ 2  # f(x) = x², f'(x) = 2x, f''(x) = 2

        itp = quadratic_interp(x, y; bc = Left(QuadraticFit()))

        # First derivative should match 2x
        @test itp(1.5; deriv = DerivOp(1)) ≈ 2 * 1.5 atol = 1.0e-11
        @test itp(2.5; deriv = DerivOp(1)) ≈ 2 * 2.5 atol = 1.0e-11

        # Second derivative should be constant = 2
        @test itp(1.5; deriv = DerivOp(2)) ≈ 2.0 atol = 1.0e-11
        @test itp(2.5; deriv = DerivOp(2)) ≈ 2.0 atol = 1.0e-11
    end

    @testset "Float32 support" begin
        x = Float32[0.0, 1.0, 2.0, 3.0, 4.0]
        y = x .^ 2
        itp = quadratic_interp(x, y; bc = Left(QuadraticFit()))

        @test itp(1.5f0) isa Float32
        @test itp(1.5f0) ≈ 1.5f0^2 atol = 1.0e-5
    end
end


# ============================================================================
# Group 15: QuadraticFit _fill_slopes! Direct Tests (Phase 2)
# ============================================================================
@testset "Quadratic Interpolation - QuadraticFit _fill_slopes!" begin
    using FastInterpolations: _fill_slopes!

    @testset "Left(QuadraticFit) slope computation" begin
        # For f(x) = x² on uniform grid [0,1,2,3,4]
        # f'(0) = 0, so d[1] should be 0
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = x .^ 2
        h = diff(x)
        s = diff(y) ./ h  # [1, 3, 5, 7]
        d = zeros(5)

        _fill_slopes!(d, s, h, Left(QuadraticFit()), x, y)

        # For x², the derivative at x=0 is 0
        @test d[1] ≈ 0.0 atol = 1.0e-12

        # Forward recurrence: d[i+1] = 2*s[i] - d[i]
        # So: d[2] = 2*1 - 0 = 2, d[3] = 2*3 - 2 = 4, etc.
        @test d[2] ≈ 2.0 atol = 1.0e-12
        @test d[3] ≈ 4.0 atol = 1.0e-12
        @test d[4] ≈ 6.0 atol = 1.0e-12
        @test d[5] ≈ 8.0 atol = 1.0e-12
    end

    @testset "Right(QuadraticFit) slope computation" begin
        # For f(x) = x² on uniform grid [0,1,2,3,4]
        # f'(4) = 8, so d[5] should be 8
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = x .^ 2
        h = diff(x)
        s = diff(y) ./ h  # [1, 3, 5, 7]
        d = zeros(5)

        _fill_slopes!(d, s, h, Right(QuadraticFit()), x, y)

        # For x², the derivative at x=4 is 8
        @test d[5] ≈ 8.0 atol = 1.0e-12

        # Backward recurrence from d[5]=8
        # d[4] = 2*s[4] - d[5] = 2*7 - 8 = 6
        # d[3] = 2*s[3] - d[4] = 2*5 - 6 = 4, etc.
        @test d[4] ≈ 6.0 atol = 1.0e-12
        @test d[3] ≈ 4.0 atol = 1.0e-12
        @test d[2] ≈ 2.0 atol = 1.0e-12
        @test d[1] ≈ 0.0 atol = 1.0e-12
    end

    @testset "Non-uniform grid 3-point formula" begin
        # For f(x) = x² on non-uniform grid [0, 0.5, 1.5]
        # f'(0) = 0, f'(1.5) = 3
        x = [0.0, 0.5, 1.5]
        y = x .^ 2  # [0, 0.25, 2.25]
        h = diff(x)  # [0.5, 1.0]
        s = diff(y) ./ h  # [0.5, 2.0]
        d = zeros(3)

        _fill_slopes!(d, s, h, Left(QuadraticFit()), x, y)
        @test d[1] ≈ 0.0 atol = 1.0e-12

        # Test Right as well
        d_right = zeros(3)
        _fill_slopes!(d_right, s, h, Right(QuadraticFit()), x, y)
        @test d_right[3] ≈ 3.0 atol = 1.0e-12
    end
end
