@testitem "integrate scaffold api" begin
    x = collect(range(0.0, 1.0, length = 11))
    y = @. x^2
    itp_lin = linear_interp(x, y)
    itp_cub = cubic_interp(x, y)

    @testset "1d integrate returns value (all methods)" begin
        for itp in (itp_lin, itp_cub)
            @test integrate(itp) isa Real
            @test integrate(itp, 0.2, 0.7) isa Real
        end
    end

    @testset "normalization helpers" begin
        s1, a1, b1 = FastInterpolations._normalize_bounds_1d(0.2, 0.8)
        @test s1 == 1
        @test a1 == 0.2
        @test b1 == 0.8

        s2, a2, b2 = FastInterpolations._normalize_bounds_1d(0.8, 0.2)
        @test s2 == -1
        @test a2 == 0.2
        @test b2 == 0.8

        s3, a3, b3 = FastInterpolations._normalize_bounds_1d(0.5, 0.5)
        @test s3 == 0
        @test a3 == 0.5
        @test b3 == 0.5
    end

    @testset "nd normalization sign" begin
        sign, lo, hi = FastInterpolations._normalize_bounds_nd((2.0, -1.0), (0.0, 3.0))
        @test sign == -1
        @test lo == (0.0, -1.0)
        @test hi == (2.0, 3.0)
    end
end

# One-shot quadrature: integrate(x, y; method) — unified-API routing (the same
# `_interp1d_route` trait as `interp(x, y; method=…)`), built internally with
# `StorePolicy(copy=false, cache_axis=false)` so construction cost ≈ 0.
@testitem "one-shot integrate(x, y; method)" begin
    x_vec = collect(range(0.0, 2.0, length = 21))
    x_rng = range(0.0, 2.0, length = 21)
    y = @. x_vec^2 + 1.0

    @testset "matches the persistent build" begin
        for x in (x_vec, x_rng)
            @test integrate(x, y; method = LinearInterp()) ≈ integrate(linear_interp(x, y)) atol = 1.0e-12
            @test integrate(x, y; method = CubicInterp()) ≈ integrate(cubic_interp(x, y)) atol = 1.0e-12
            @test integrate(x, y; method = QuadraticInterp()) ≈ integrate(quadratic_interp(x, y)) atol = 1.0e-12
        end
    end

    @testset "method options forwarded" begin
        @test integrate(x_vec, y; method = ConstantInterp(side = LeftSide())) ≈
            integrate(constant_interp(x_vec, y; side = LeftSide())) atol = 1.0e-12
        @test integrate(x_vec, y; method = CubicInterp(bc = ZeroCurvBC())) ≈
            integrate(cubic_interp(x_vec, y; bc = ZeroCurvBC())) atol = 1.0e-12
    end

    @testset "contract: method is required; unsupported methods reject" begin
        @test_throws UndefKeywordError integrate(x_vec, y)
        @test_throws ArgumentError integrate(x_vec, y; method = NoInterp())
    end

    @testset "inputs are not mutated" begin
        xb = copy(x_vec)
        yb = copy(y)
        integrate(x_vec, y; method = LinearInterp())
        @test x_vec == xb && y == yb
    end
end

# ND one-shot quadrature: integrate(grids, data; method) — the ND mirror of the
# 1-D form. A single method is applied to every axis, building a homogeneous
# specialized ND interpolant; trivial methods (linear/constant) use raw reference
# storage. Mixed/Hermite methods build a HeteroInterpolantND (integrate unsupported).
@testitem "one-shot integrate(grids, data; method) — ND" setup = [AllocConstants] begin
    xr = range(0.0, Float64(π), length = 20)
    yr = range(0.0, 2.0, length = 16)
    xv = collect(xr)
    yv = collect(yr)
    f(xi, yj) = sin(xi) * cos(yj)

    @testset "matches the persistent build (2D, Range + Vector)" begin
        for (gx, gy) in ((xr, yr), (xv, yv))
            data = [f(xi, yj) for xi in gx, yj in gy]
            for m in (LinearInterp(), CubicInterp(), QuadraticInterp(), ConstantInterp())
                ref = integrate(interp((gx, gy), data; method = m))
                @test integrate((gx, gy), data; method = m) ≈ ref rtol = 1.0e-12
            end
        end
    end

    @testset "3D linear parity" begin
        x = range(0.0, 1.0, length = 9)
        y = range(0.0, 2.0, length = 8)
        z = range(0.0, 3.0, length = 7)
        data = [xi + 2yj - zk for xi in x, yj in y, zk in z]
        ref = integrate(linear_interp((x, y, z), data))
        @test integrate((x, y, z), data; method = LinearInterp()) ≈ ref rtol = 1.0e-12
    end

    @testset "trivial methods build with near-zero allocation" begin
        # Measured through a function barrier: a bare `@allocated` at test scope
        # boxes the captured globals and reports noise, not the API's real cost.
        data = [f(xi, yj) for xi in xv, yj in yv]
        alloc_oneshot(g, d, m) = @allocated integrate(g, d; method = m)
        alloc_oneshot((xv, yv), data, LinearInterp())          # warmup
        @test alloc_oneshot((xv, yv), data, LinearInterp()) <= ND_ALLOC_THRESHOLD
        @test alloc_oneshot((xv, yv), data, ConstantInterp()) <= ND_ALLOC_THRESHOLD
    end

    @testset "ND integrates fewer methods than 1D — reject the rest up front" begin
        data = [f(xi, yj) for xi in xr, yj in yr]
        @test_throws UndefKeywordError integrate((xr, yr), data)
        # 1-D one-shot integrates the Hermite family (Pchip/Akima/Cardinal); ND
        # does not (they build a HeteroInterpolantND with no ND integral). Reject
        # them with a clear method-named error, not the internal Hetero message.
        for m in (PchipInterp(), AkimaInterp(), CardinalInterp(), NoInterp())
            @test_throws "ND full-domain integration is implemented" integrate((xr, yr), data; method = m)
        end
        # …while the same methods DO integrate in 1-D:
        xv1 = collect(xr)
        yv1 = sin.(xv1)
        for m in (PchipInterp(), AkimaInterp(), CardinalInterp())
            @test integrate(xv1, yv1; method = m) isa Real
        end
    end

    @testset "inputs are not mutated" begin
        data = [f(xi, yj) for xi in xv, yj in yv]
        xb = copy(xv);  yb = copy(yv);  db = copy(data)
        integrate((xv, yv), data; method = LinearInterp())
        @test xv == xb && yv == yb && data == db
    end
end
