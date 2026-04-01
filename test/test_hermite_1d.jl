# ALLOC_THRESHOLD is defined in runtests.jl

@testset "Cubic Hermite 1D" begin

    # ========================================
    # Test data: cubic polynomial p(x) = 2x³ - x² + 3x - 1
    # Hermite with exact derivatives should reproduce exactly.
    # ========================================
    p(x) = 2x^3 - x^2 + 3x - 1
    dp(x) = 6x^2 - 2x + 3
    d2p(x) = 12x - 2
    d3p(x::T) where {T} = T(12)

    @testset "Cubic polynomial exactness — oneshot scalar" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            val = cubic_interp(x, Hermite(y, dy), xq)
            @test val ≈ p(xq) atol = 1e-12
        end
    end

    @testset "Cubic polynomial exactness — oneshot vector" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]

        result = cubic_interp(x, Hermite(y, dy), xq)
        @test result ≈ p.(xq) atol = 1e-12
    end

    @testset "Cubic polynomial exactness — in-place" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)
        xq = [0.15, 0.7, 1.33, 2.5, 2.99]
        out = zeros(length(xq))

        cubic_interp!(out, x, Hermite(y, dy), xq)
        @test out ≈ p.(xq) atol = 1e-12
    end

    @testset "Cubic polynomial exactness — callable interpolant" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        itp = cubic_interp(x, Hermite(y, dy))
        @test itp isa CubicHermiteInterpolant1D

        xq_pts = [0.15, 0.7, 1.33, 2.5, 2.99]
        for xq in xq_pts
            @test itp(xq) ≈ p(xq) atol = 1e-12
        end

        # Vector call
        xq_vec = [0.15, 0.7, 1.33, 2.5, 2.99]
        @test itp(xq_vec) ≈ p.(xq_vec) atol = 1e-12

        # In-place call
        out = zeros(length(xq_vec))
        itp(out, xq_vec)
        @test out ≈ p.(xq_vec) atol = 1e-12
    end

    @testset "DerivOp correctness" begin
        x = collect(range(0.0, 3.0, 20))
        y = p.(x)
        dy = dp.(x)

        xq = 1.5

        # Value
        @test cubic_interp(x, Hermite(y, dy), xq; deriv = EvalValue()) ≈ p(xq) atol = 1e-12

        # First derivative
        @test cubic_interp(x, Hermite(y, dy), xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 1e-10

        # Second derivative
        @test cubic_interp(x, Hermite(y, dy), xq; deriv = DerivOp(2)) ≈ d2p(xq) atol = 1e-8

        # Third derivative (constant 12 for cubic polynomial)
        @test cubic_interp(x, Hermite(y, dy), xq; deriv = DerivOp(3)) ≈ d3p(xq) atol = 1e-6

        # Fourth+ derivative → zero
        @test cubic_interp(x, Hermite(y, dy), xq; deriv = DerivOp(4)) ≈ 0.0 atol = 1e-14

        # Via callable
        itp = cubic_interp(x, Hermite(y, dy))
        @test itp(xq; deriv = DerivOp(1)) ≈ dp(xq) atol = 1e-10
        @test itp(xq; deriv = DerivOp(2)) ≈ d2p(xq) atol = 1e-8
    end

    @testset "Extrapolation modes" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)

        # NoExtrap — throws outside domain
        @test_throws DomainError cubic_interp(x, Hermite(y, dy), -0.1)
        @test_throws DomainError cubic_interp(x, Hermite(y, dy), 1.1)

        # ClampExtrap — returns boundary values
        val_left = cubic_interp(x, Hermite(y, dy), -0.5; extrap = ClampExtrap())
        val_right = cubic_interp(x, Hermite(y, dy), 1.5; extrap = ClampExtrap())
        @test val_left ≈ y[1]
        @test val_right ≈ y[end]

        # ExtendExtrap — no error outside domain
        val_ext = cubic_interp(x, Hermite(y, dy), 1.2; extrap = ExtendExtrap())
        @test isfinite(val_ext)

        # WrapExtrap — wraps to domain
        val_wrap = cubic_interp(x, Hermite(y, dy), 1.2; extrap = WrapExtrap())
        @test isfinite(val_wrap)

        # Callable with extrap
        itp = cubic_interp(x, Hermite(y, dy); extrap = ClampExtrap())
        @test itp(-0.5) ≈ y[1]
        @test itp(1.5) ≈ y[end]
    end

    @testset "Type stability — @inferred" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)
        xq_scalar = 0.5
        xq_vec = [0.2, 0.5, 0.8]

        # Scalar oneshot
        @test @inferred(cubic_interp(x, Hermite(y, dy), xq_scalar)) isa Float64

        # Vector oneshot
        @test @inferred(cubic_interp(x, Hermite(y, dy), xq_vec)) isa Vector{Float64}

        # Callable scalar
        itp = cubic_interp(x, Hermite(y, dy))
        @test @inferred(itp(xq_scalar)) isa Float64

        # Callable vector
        @test @inferred(itp(xq_vec)) isa Vector{Float64}
    end

    @testset "Zero allocation" begin
        x = collect(range(0.0, 1.0, 50))
        y = sin.(x)
        dy = cos.(x)
        h = Hermite(y, dy)

        # Scalar oneshot — function barrier for accurate measurement
        function test_scalar_alloc(x, h, xq)
            cubic_interp(x, h, xq)
            return @allocated cubic_interp(x, h, xq)
        end
        @test test_scalar_alloc(x, h, 0.5) <= ALLOC_THRESHOLD

        # Callable scalar
        itp = cubic_interp(x, Hermite(y, dy))
        function test_callable_alloc(itp, xq)
            itp(xq)
            return @allocated itp(xq)
        end
        @test test_callable_alloc(itp, 0.5) == 0
    end

    @testset "Uniform grid (Range input)" begin
        x = range(0.0, 3.0, 20)
        y = p.(collect(x))
        dy = dp.(collect(x))

        # Range grid → O(1) direct search
        @test cubic_interp(x, Hermite(y, dy), 1.5) ≈ p(1.5) atol = 1e-12

        # Callable with range grid — verify Range preserved as _CachedRange
        itp = cubic_interp(x, Hermite(y, dy))
        @test itp.x isa AbstractRange
        @test itp.x isa FastInterpolations._CachedRange
        @test itp(1.5) ≈ p(1.5) atol = 1e-12
    end

    @testset "Non-uniform grid" begin
        x = sort([0.0, 0.1, 0.3, 0.7, 1.2, 2.0, 2.5, 3.0])
        y = p.(x)
        dy = dp.(x)

        xq = 1.5
        @test cubic_interp(x, Hermite(y, dy), xq) ≈ p(xq) atol = 1e-12
    end

    @testset "2-point grid (minimum)" begin
        x = [0.0, 1.0]
        y = p.(x)
        dy = dp.(x)

        xq = 0.5
        @test cubic_interp(x, Hermite(y, dy), xq) ≈ p(xq) atol = 1e-12
    end

    @testset "Complex values" begin
        x = collect(range(0.0, 2π, 20))
        y = exp.(1im .* x)
        dy = 1im .* exp.(1im .* x)

        itp = cubic_interp(x, Hermite(y, dy))
        xq = 1.0
        @test itp(xq) ≈ exp(1im * xq) atol = 1e-4
    end

    @testset "Integer input promotion" begin
        x_int = 0:10
        y_int = collect(x_int) .^ 2
        dy_int = 2 .* collect(x_int)

        # Should auto-promote to Float
        val = cubic_interp(collect(x_int), Hermite(y_int, dy_int), 5.5)
        @test val ≈ 5.5^2 atol = 1e-10
    end

    @testset "Error conditions" begin
        x = collect(range(0.0, 1.0, 10))
        y = sin.(x)
        dy = cos.(x)

        # Length mismatch at Hermite construction
        @test_throws ArgumentError Hermite(y, cos.(x[1:5]))

        # Rejected kwargs
        @test_throws ArgumentError cubic_interp(x, Hermite(y, dy), 0.5; bc = CubicFit())
        @test_throws ArgumentError cubic_interp(x, Hermite(y, dy), 0.5; autocache = true)

        # Rejected kwargs on callable
        @test_throws ArgumentError cubic_interp(x, Hermite(y, dy); bc = CubicFit())
        @test_throws ArgumentError cubic_interp(x, Hermite(y, dy); autocache = false)
    end

    @testset "Hermite matches spline on cubic polynomial data" begin
        # For a true cubic polynomial, both spline and Hermite should agree
        x = collect(range(0.0, 3.0, 15))
        y = p.(x)
        dy = dp.(x)
        xq = [0.3, 1.0, 1.7, 2.5]

        hermite_vals = cubic_interp(x, Hermite(y, dy), xq)
        spline_vals = cubic_interp(x, y, xq)

        # Both should reproduce the cubic polynomial exactly
        @test hermite_vals ≈ p.(xq) atol = 1e-10
        @test spline_vals ≈ p.(xq) atol = 1e-10
    end
end
