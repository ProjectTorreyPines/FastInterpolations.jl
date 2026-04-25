@testitem "integrate 1d linear/quadratic/constant" begin
    x = collect(range(0.0, 1.0, length = 31))

    @testset "linear affine exact" begin
        y = @. 3x - 1
        itp = linear_interp(x, y; extrap = NoExtrap())
        a, b = 0.15, 0.85
        expected = (1.5 * b^2 - b) - (1.5 * a^2 - a)
        @test integrate(itp, a, b) ≈ expected atol = 1.0e-12
    end

    @testset "quadratic polynomial exact" begin
        y = @. 2x^2 - x + 4
        itp = quadratic_interp(x, y; extrap = NoExtrap())
        a, b = 0.1, 0.9
        expected = ((2 / 3) * b^3 - 0.5 * b^2 + 4b) - ((2 / 3) * a^3 - 0.5 * a^2 + 4a)
        @test integrate(itp, a, b) ≈ expected atol = 1.0e-10
    end

    @testset "constant finite by side mode" begin
        y = collect(1.0:length(x))
        for side in (LeftSide(), RightSide(), NearestSide())
            itp = constant_interp(x, y; side = side, extrap = NoExtrap())
            I = integrate(itp, 0.2, 0.7)
            @test isfinite(I)
        end
    end

    @testset "common invariants" begin
        y = @. sin(2pi * x)
        for itp in (linear_interp(x, y), quadratic_interp(x, y), constant_interp(x, y))
            a, m, b = 0.12, 0.48, 0.93
            @test integrate(itp, a, b) ≈ -integrate(itp, b, a) atol = 1.0e-12
            @test integrate(itp, a, b) ≈ integrate(itp, a, m) + integrate(itp, m, b) atol = 1.0e-12
        end
    end
end
