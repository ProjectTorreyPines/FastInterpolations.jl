using Test
using FastInterpolations

@testset "Full-domain fast path" begin
    x = collect(range(0.0, 2.0, length=21))

    @testset "1D scalar parity" begin
        y_cubic = @. x^3 - 2x + 1
        y_linear = @. 3x + 1
        y_quad = @. 2x^2 - x + 4
        y_const = collect(1.0:length(x))

        itp_c = cubic_interp(x, y_cubic; extrap=:none)
        itp_l = linear_interp(x, y_linear; extrap=:none)
        itp_q = quadratic_interp(x, y_quad; extrap=:none)

        @test integrate(itp_c) ≈ integrate(itp_c, first(x), last(x)) atol=1e-14
        @test integrate(itp_l) ≈ integrate(itp_l, first(x), last(x)) atol=1e-14
        @test integrate(itp_q) ≈ integrate(itp_q, first(x), last(x)) atol=1e-14

        for side in (:left, :right, :nearest)
            itp_k = constant_interp(x, y_const; side=side, extrap=:none)
            @test integrate(itp_k) ≈ integrate(itp_k, first(x), last(x)) atol=1e-14
        end
    end

    @testset "1D analytical exactness" begin
        y_linear = @. 3x + 1
        itp_l = linear_interp(x, y_linear; extrap=:none)
        expected_linear = 1.5 * last(x)^2 + last(x) - (1.5 * first(x)^2 + first(x))
        @test integrate(itp_l) ≈ expected_linear atol=1e-12

        y_cubic = @. x^3 - 2x + 1
        itp_c = cubic_interp(x, y_cubic; bc=CubicFit(), extrap=:none)
        expected_cubic = (last(x)^4/4 - last(x)^2 + last(x)) - (first(x)^4/4 - first(x)^2 + first(x))
        @test integrate(itp_c) ≈ expected_cubic atol=1e-10
    end

    @testset "1D Series parity" begin
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
                result = integrate(sitp)
                @test result isa Vector
                @test length(result) == 2
                @test result[1] ≈ integrate(itp1) atol=1e-12
                @test result[2] ≈ integrate(itp2) atol=1e-12
            end
        end

        @testset "constant series" begin
            y_c = hcat(collect(1.0:length(x)), collect(length(x):-1.0:1.0))
            for side in (:left, :right, :nearest)
                sitp = constant_interp(x, [y_c[:, 1], y_c[:, 2]]; side=side)
                itp1 = constant_interp(x, y_c[:, 1]; side=side)
                itp2 = constant_interp(x, y_c[:, 2]; side=side)
                result = integrate(sitp)
                @test result[1] ≈ integrate(itp1) atol=1e-12
                @test result[2] ≈ integrate(itp2) atol=1e-12
            end
        end
    end

    @testset "ND parity" begin
        xg = collect(range(0.0, 1.0, length=11))
        yg = collect(range(-1.0, 1.0, length=9))
        data_2d = [sin(xi) * cos(yj) for xi in xg, yj in yg]

        @testset "cubic ND" begin
            itp = cubic_interp((xg, yg), data_2d; extrap=(:none, :none))
            lo = (first(xg), first(yg))
            hi = (last(xg), last(yg))
            @test integrate(itp) ≈ integrate(itp, lo, hi) atol=1e-10
        end

        @testset "linear ND" begin
            itp = linear_interp((xg, yg), data_2d; extrap=(:none, :none))
            lo = (first(xg), first(yg))
            hi = (last(xg), last(yg))
            @test integrate(itp) ≈ integrate(itp, lo, hi) atol=1e-10
        end

        @testset "quadratic ND" begin
            itp = quadratic_interp((xg, yg), data_2d; extrap=(:none, :none))
            lo = (first(xg), first(yg))
            hi = (last(xg), last(yg))
            @test integrate(itp) ≈ integrate(itp, lo, hi) atol=1e-10
        end

        @testset "constant ND" begin
            for side in ((:left, :left), (:right, :right), (:nearest, :nearest))
                itp = constant_interp((xg, yg), data_2d; side=side, extrap=(:none, :none))
                lo = (first(xg), first(yg))
                hi = (last(xg), last(yg))
                @test integrate(itp) ≈ integrate(itp, lo, hi) atol=1e-10
            end
        end
    end

    @testset "1D zero-allocation" begin
        y = @. 3x + 1
        itp_l = linear_interp(x, y; extrap=:none)
        integrate(itp_l)  # warmup
        alloc = @allocated integrate(itp_l)
        @test alloc == 0

        y_c = @. x^3 - 2x + 1
        itp_c = cubic_interp(x, y_c; extrap=:none)
        integrate(itp_c)
        alloc_c = @allocated integrate(itp_c)
        @test alloc_c == 0
    end
end
