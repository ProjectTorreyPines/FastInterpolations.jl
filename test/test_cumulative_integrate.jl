using Test
using FastInterpolations

@testset "cumulative_integrate" begin
    x = collect(range(0.0, 2.0, length=21))

    @testset "1D scalar" begin
        y_cubic = @. x^2 + x + 1   # always positive on [0,2]
        y_linear = @. 3x + 1
        y_quad = @. 2x^2 - x + 4

        itp_c = cubic_interp(x, y_cubic; extrap=NoExtrap())
        itp_l = linear_interp(x, y_linear; extrap=NoExtrap())
        itp_q = quadratic_interp(x, y_quad; extrap=NoExtrap())

        for (name, itp) in [("cubic", itp_c), ("linear", itp_l), ("quadratic", itp_q)]
            @testset "$name" begin
                cum = cumulative_integrate(itp)
                @test cum isa Vector
                @test length(cum) == length(x)
                @test cum[1] == 0.0
                @test cum[end] ≈ integrate(itp) atol=1e-12
                # monotonicity check: differences should be non-negative for positive function on [0,2]
                diffs = diff(cum)
                @test all(d -> d >= -1e-15, diffs)
            end
        end

        @testset "constant" begin
            y_const = collect(1.0:length(x))
            for side in (LeftSide(), RightSide(), NearestSide())
                itp_k = constant_interp(x, y_const; side=side, extrap=NoExtrap())
                cum = cumulative_integrate(itp_k)
                @test cum[1] == 0.0
                @test cum[end] ≈ integrate(itp_k) atol=1e-12
            end
        end
    end

    @testset "1D analytical" begin
        # f(x) = 3x + 1, F(x) = 1.5x² + x
        y_linear = @. 3x + 1
        itp_l = linear_interp(x, y_linear; extrap=NoExtrap())
        cum = cumulative_integrate(itp_l)
        for (j, xj) in enumerate(x)
            expected = 1.5 * xj^2 + xj - (1.5 * first(x)^2 + first(x))
            @test cum[j] ≈ expected atol=1e-12
        end
    end

    @testset "1D Series" begin
        y1 = sin.(x)
        y2 = cos.(x)

        for (name, mk_scalar, mk_series) in [
            ("cubic",     (x, y) -> cubic_interp(x, y),     (x, ys) -> cubic_interp(x, ys)),
            ("linear",    (x, y) -> linear_interp(x, y),    (x, ys) -> linear_interp(x, ys)),
            ("quadratic", (x, y) -> quadratic_interp(x, y), (x, ys) -> quadratic_interp(x, ys)),
        ]
            @testset "$name" begin
                sitp = mk_series(x, [y1, y2])
                itp1 = mk_scalar(x, y1)
                itp2 = mk_scalar(x, y2)
                cum = cumulative_integrate(sitp)
                @test cum isa Matrix
                @test size(cum) == (length(x), 2)

                cum1 = cumulative_integrate(itp1)
                cum2 = cumulative_integrate(itp2)
                @test cum[:, 1] ≈ cum1 atol=1e-12
                @test cum[:, 2] ≈ cum2 atol=1e-12

                # endpoint matches integrate
                result = integrate(sitp)
                @test cum[end, 1] ≈ result[1] atol=1e-12
                @test cum[end, 2] ≈ result[2] atol=1e-12
            end
        end

        @testset "constant series" begin
            y_c1 = collect(1.0:length(x))
            y_c2 = collect(length(x):-1.0:1.0)
            for side in (LeftSide(), RightSide(), NearestSide())
                sitp = constant_interp(x, [y_c1, y_c2]; side=side)
                itp1 = constant_interp(x, y_c1; side=side)
                itp2 = constant_interp(x, y_c2; side=side)
                cum = cumulative_integrate(sitp)
                @test cum[:, 1] ≈ cumulative_integrate(itp1) atol=1e-12
                @test cum[:, 2] ≈ cumulative_integrate(itp2) atol=1e-12
            end
        end
    end

    @testset "ND fallback" begin
        xg = collect(range(0.0, 1.0, length=5))
        yg = collect(range(0.0, 1.0, length=5))
        data = [xi * yj for xi in xg, yj in yg]
        itp_nd = linear_interp((xg, yg), data)
        @test_throws ArgumentError cumulative_integrate(itp_nd)
    end
end
