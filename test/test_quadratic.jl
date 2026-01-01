# Tests for quadratic (C1 piecewise quadratic) spline interpolation
#
# This file follows TDD: tests are written BEFORE implementation.
# Phase 1: BC Tags (Left/Right types)
# Phase 2: QuadraticSplineCache + Autocache
# Phase 3: Kernels + Coefficient Computation

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
# Group 2: Cache Tests (Phase 2)
# ============================================================================
@testset "Quadratic Interpolation - Cache" begin

    @testset "QuadraticSplineCache construction" begin
        x = collect(range(0.0, 1.0, 11))

        cache = QuadraticSplineCache(x)
        @test cache isa QuadraticSplineCache{Float64}
        @test length(cache.h) == 10
        @test length(cache.inv_h) == 10
        @test cache.h[1] ≈ 0.1
        @test cache.inv_h[1] ≈ 10.0
    end

    @testset "QuadraticSplineCache Float32" begin
        x32 = Float32.(collect(range(0.0, 1.0, 11)))
        cache32 = QuadraticSplineCache(x32)
        @test cache32 isa QuadraticSplineCache{Float32}
        @test eltype(cache32.h) === Float32
    end

    @testset "QuadraticSplineCache edge cases" begin
        # Too few points
        @test_throws ArgumentError QuadraticSplineCache([1.0])

        # Not strictly increasing
        @test_throws ArgumentError QuadraticSplineCache([1.0, 1.0, 2.0])
        @test_throws ArgumentError QuadraticSplineCache([1.0, 0.5, 2.0])

        # Minimum valid (n=2)
        cache_min = QuadraticSplineCache([0.0, 1.0])
        @test length(cache_min.h) == 1
    end

    @testset "QuadraticSplineCache non-uniform grid" begin
        x_nu = [0.0, 0.1, 0.3, 0.6, 1.0]
        cache = QuadraticSplineCache(x_nu)
        @test cache.h[1] ≈ 0.1
        @test cache.h[2] ≈ 0.2
        @test cache.h[3] ≈ 0.3
        @test cache.h[4] ≈ 0.4
    end

end

@testset "Quadratic Interpolation - Autocache" begin
    using FastInterpolations: _get_quadratic_cache

    @testset "autocache basic" begin
        x = collect(range(0.0, 1.0, 11))
        clear_quadratic_cache!()

        # First call creates cache
        cache1 = _get_quadratic_cache(x)
        @test cache1 isa QuadraticSplineCache

        # Second call returns same cache (RCU hit)
        cache2 = _get_quadratic_cache(x)
        @test cache1 === cache2  # identity check
    end

    @testset "autocache different grids" begin
        clear_quadratic_cache!()

        x1 = collect(range(0.0, 1.0, 11))
        x2 = collect(range(0.0, 2.0, 11))

        cache1 = _get_quadratic_cache(x1)
        cache2 = _get_quadratic_cache(x2)

        @test cache1 !== cache2
    end

    @testset "autocache disabled" begin
        x = collect(range(0.0, 1.0, 11))
        clear_quadratic_cache!()

        # With autocache disabled, should create new cache each time
        cache1 = _get_quadratic_cache(x; autocache=false)
        cache2 = _get_quadratic_cache(x; autocache=false)
        @test cache1 !== cache2
    end

    @testset "autocache zero-allocation" begin
        x = collect(range(0.0, 1.0, 51))
        clear_quadratic_cache!()
        _get_quadratic_cache(x)  # prime

        allocs = @allocated _get_quadratic_cache(x)
        @test allocs == 0
    end

    @testset "get/set cache size" begin
        @test get_quadratic_cache_size() > 0
        @test get_quadratic_cache_size() isa Int
    end

end

# ============================================================================
# Group 3: Kernel Tests (Phase 3)
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
# Group 4: Coefficient Computation Tests (Phase 3)
# ============================================================================
@testset "Quadratic Interpolation - Coefficient Computation" begin
    using FastInterpolations: _compute_quadratic_secants!, _compute_d1_from_bc,
                               _forward_recurrence!, _compute_quadratic_coefficients!

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

    @testset "d1 from Left(Deriv1)" begin
        bc = Left(Deriv1(3.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        n = 4

        d1 = _compute_d1_from_bc(bc, s, h, n)
        @test d1 ≈ 3.0  # given directly
    end

    @testset "d1 from Left(Deriv2)" begin
        # Left(Deriv2(κ)): a[1] = κ/2, d[1] = s[1] - a[1]*h[1]
        bc = Left(Deriv2(2.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        n = 4

        d1 = _compute_d1_from_bc(bc, s, h, n)
        # a[1] = 2/2 = 1, d[1] = 1 - 1*1 = 0
        @test d1 ≈ 0.0
    end

    @testset "d1 from Right(Deriv1)" begin
        # d[n] = v, backward recurrence to d[1]
        bc = Right(Deriv1(7.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        n = 4

        d1 = _compute_d1_from_bc(bc, s, h, n)
        # backward: d[i] = 2*s[i] - d[i+1]
        # d[4] = 7
        # d[3] = 2*5 - 7 = 3
        # d[2] = 2*3 - 3 = 3
        # d[1] = 2*1 - 3 = -1
        @test d1 ≈ -1.0
    end

    @testset "d1 from Right(Deriv2)" begin
        # a[n-1] = κ/2, d[n-1] = s[n-1] - a[n-1]*h[n-1]
        # d[n] = 2*a[n-1]*h[n-1] + d[n-1], then backward
        bc = Right(Deriv2(2.0))
        s = [1.0, 3.0, 5.0]
        h = [1.0, 1.0, 1.0]
        n = 4

        d1 = _compute_d1_from_bc(bc, s, h, n)
        # a[3] = 2/2 = 1
        # d[3] = 5 - 1*1 = 4
        # d[4] = 2*1*1 + 4 = 6
        # d[3] = 2*5 - 6 = 4 (verify)
        # d[2] = 2*3 - 4 = 2
        # d[1] = 2*1 - 2 = 0
        @test d1 ≈ 0.0
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
# Group 5: Public API Tests (Phase 4)
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

        # :constant - clamp to boundary values
        @test quadratic_interp(x, y, -0.5; extrap=:constant) ≈ 0.0
        @test quadratic_interp(x, y, 2.5; extrap=:constant) ≈ 4.0

        # :extension - extend the polynomial
        v_ext = quadratic_interp(x, y, 2.5; extrap=:extension)
        @test isfinite(v_ext)
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
# Group 6: QuadraticInterpolant Tests (Phase 5)
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
# Group 7: DerivativeView Tests (Phase 5)
# ============================================================================
# ============================================================================
# Group 7: Allocation Tests (Phase 6)
# ============================================================================
@testset "Quadratic Interpolation - Allocations" begin

    @testset "scalar interpolation zero-allocation" begin
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
        @test allocs == 0

        # Derivative calls should also be zero-allocation
        for _ in 1:10
            itp(0.5; deriv=1)
            itp(0.5; deriv=2)
        end
        allocs_d1 = @allocated itp(0.5; deriv=1)
        allocs_d2 = @allocated itp(0.5; deriv=2)
        @test allocs_d1 == 0
        @test allocs_d2 == 0
    end

    @testset "in-place vector zero-allocation" begin
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
        @test allocs == 0
    end

    @testset "3-arg API with autocache" begin
        x = collect(range(0.0, 1.0, 51))
        y = x.^2
        clear_quadratic_cache!()

        # Prime cache and JIT
        for _ in 1:10
            quadratic_interp(x, y, 0.5; bc=Right(Deriv1(2.0)))
        end

        # After cache is warmed, scalar call allocates for coefficients (expected)
        # But autocache lookup itself should be zero-alloc
        # Note: 3-arg form computes coefficients each call, so allocations expected
        # This is different from 2-arg form which precomputes
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
        @test allocs_d1 == 0
        @test allocs_d2 == 0
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
