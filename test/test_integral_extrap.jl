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

    @testset ":extension throws not-yet-implemented" begin
        y = @. sin(2pi*x) + 0.1x
        itp = cubic_interp(x, y; extrap=:extension)
        @test_throws ArgumentError integrate(itp, -0.2, 0.5)
    end
end
