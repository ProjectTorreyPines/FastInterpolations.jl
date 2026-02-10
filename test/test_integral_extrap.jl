using Test
using FastInterpolations

@testset "integrate extrap semantics" begin
    x = collect(range(0.0, 1.0, length=31))

    @testset ":none throws outside domain" begin
        y = @. sin(2pi*x) + 0.1x
        itp = cubic_interp(x, y; extrap=:none)
        @test_throws DomainError integrate(itp, -0.2, 0.3)
        @test_throws DomainError integrate(itp, 0.2, 1.2)
        # fully in-domain should still work
        @test integrate(itp, 0.1, 0.9) isa Real
    end

    @testset ":constant tails (linear)" begin
        y = @. x^2 + 1.0  # y[1] = 1.0, y[end] = 2.0
        itp = linear_interp(x, y; extrap=:constant)
        # pure left tail
        @test integrate(itp, -0.4, 0.0) ≈ y[1] * 0.4 atol=1e-12
        # pure right tail
        @test integrate(itp, 1.0, 1.6) ≈ y[end] * 0.6 atol=1e-12
        # mixed: left tail + in-domain
        in_part = integrate(linear_interp(x, y; extrap=:none), 0.0, 0.5)
        @test integrate(itp, -0.3, 0.5) ≈ y[1] * 0.3 + in_part atol=1e-12
    end

    @testset ":constant tails (cubic)" begin
        y = @. x^2 + 1.0
        itp = cubic_interp(x, y; extrap=:constant)
        @test integrate(itp, -0.5, 0.0) ≈ y[1] * 0.5 atol=1e-12
        @test integrate(itp, 1.0, 1.3) ≈ y[end] * 0.3 atol=1e-12
    end

    @testset ":constant tails (constant interp, side=:left)" begin
        # y[1]=1.0, y[2]≈1.001, y[end-1]≈1.999, y[end]=2.0
        # With side=:left, extrap=:constant must use y[1] (left) and y[end] (right)
        y = @. x^2 + 1.0
        itp = constant_interp(x, y; side=:left, extrap=:constant)
        # pure left tail
        @test integrate(itp, -0.4, 0.0) ≈ y[1] * 0.4 atol=1e-12
        # pure right tail
        @test integrate(itp, 1.0, 1.6) ≈ y[end] * 0.6 atol=1e-12
        # mixed: left tail + in-domain
        in_part = integrate(constant_interp(x, y; side=:left, extrap=:none), 0.0, 0.5)
        @test integrate(itp, -0.3, 0.5) ≈ y[1] * 0.3 + in_part atol=1e-12
    end

    @testset ":constant tails (constant interp, side=:right)" begin
        y = @. x^2 + 1.0
        itp = constant_interp(x, y; side=:right, extrap=:constant)
        # pure left tail — must use y[1], NOT y[2]
        @test integrate(itp, -0.4, 0.0) ≈ y[1] * 0.4 atol=1e-12
        # pure right tail — must use y[end], NOT y[end-1]
        @test integrate(itp, 1.0, 1.6) ≈ y[end] * 0.6 atol=1e-12
        # mixed: right tail + in-domain
        in_part = integrate(constant_interp(x, y; side=:right, extrap=:none), 0.5, 1.0)
        @test integrate(itp, 0.5, 1.3) ≈ in_part + y[end] * 0.3 atol=1e-12
    end

    @testset ":constant signed orientation" begin
        y = @. x^2 + 1.0
        itp = linear_interp(x, y; extrap=:constant)
        @test integrate(itp, 0.0, -0.4) ≈ -(y[1] * 0.4) atol=1e-12
    end

    @testset ":wrap periodic consistency" begin
        y = @. sin(2pi*x) + 0.1x
        itp = cubic_interp(x, y; extrap=:wrap)
        p = x[end] - x[1]
        Iperiod = integrate(itp, x[1], x[end])
        Ilong = integrate(itp, -0.2, -0.2 + 3p)
        @test Ilong ≈ 3 * Iperiod atol=1e-10
    end

    @testset ":wrap boundary crossing" begin
        y = @. sin(2pi*x) + 0.1x
        itp = cubic_interp(x, y; extrap=:wrap)
        # crossing one boundary
        I1 = integrate(itp, 0.8, 1.0)
        I2 = integrate(itp, 0.0, 0.3)
        @test integrate(itp, 0.8, 1.3) ≈ I1 + I2 atol=1e-10
    end

    @testset ":extension linear exactness" begin
        # For f(x) = 3x - 1 on [0,1], extension IS the same line,
        # so the integral over extended domain matches the analytical value.
        y = @. 3x - 1
        itp = linear_interp(x, y; extrap=:extension)
        a, b = -0.5, 1.6
        expected = 1.5*b^2 - b - (1.5*a^2 - a)   # ∫(3x-1)dx = 3x²/2 - x
        @test integrate(itp, a, b) ≈ expected atol=1e-12
    end

    @testset ":extension cubic exactness (CubicFit)" begin
        # With CubicFit BC, cubic polynomial is reproduced exactly,
        # so the boundary polynomial IS the original function.
        y = @. x^3 - 2x + 1
        itp = cubic_interp(x, y; bc=CubicFit(), extrap=:extension)
        a, b = -0.3, 1.4
        expected = (b^4/4 - b^2 + b) - (a^4/4 - a^2 + a)   # ∫(x³-2x+1)dx
        @test integrate(itp, a, b) ≈ expected atol=1e-8
    end

    @testset ":extension additivity across boundary" begin
        y = @. sin(2pi*x) + 0.1x
        itp = cubic_interp(x, y; extrap=:extension)
        # left boundary crossing
        @test integrate(itp, -0.3, 0.5) ≈
              integrate(itp, -0.3, 0.0) + integrate(itp, 0.0, 0.5) atol=1e-12
        # right boundary crossing
        @test integrate(itp, 0.7, 1.4) ≈
              integrate(itp, 0.7, 1.0) + integrate(itp, 1.0, 1.4) atol=1e-12
    end

    @testset ":extension antisymmetry" begin
        y = @. sin(2pi*x) + 0.1x
        itp = cubic_interp(x, y; extrap=:extension)
        @test integrate(itp, -0.3, 1.4) ≈ -integrate(itp, 1.4, -0.3) atol=1e-12
    end

    @testset ":extension agrees with :none in-domain" begin
        y = @. x^2 + 1.0
        itp_ext = cubic_interp(x, y; extrap=:extension)
        itp_none = cubic_interp(x, y; extrap=:none)
        @test integrate(itp_ext, 0.2, 0.8) ≈ integrate(itp_none, 0.2, 0.8) atol=1e-12
    end
end
