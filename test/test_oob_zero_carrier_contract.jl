# ========================================
# OOB / constant-derivative ZERO carrier contract
# ========================================
# The flat-extrap and constant-derivative kernels build their result as
# `0 * oob_data * <carriers>` rather than `zero(T_out)`. That single expression
# carries THREE contracts at once:
#   1. NaN propagation — `0 * NaN === NaN` (ClampExtrap extends a NaN boundary
#      value; FillExtrap(NaN) makes NaN the OOB cell's data). `zero(T)` would
#      silently turn those into 0.0.
#   2. Unit/order scaling — a derivative zero must carry `value/coordᴺ`
#      (`m/s`, `m/s²`), NOT the value unit.
#   3. Carrier width — value queries keep their Int carrier; derivative queries
#      float (Int/Int is a rational, not an Int).
# These pins exist so any rework of the zero path (e.g. threading the output
# TYPE instead of a scale VALUE) is provably behaviour-preserving. The
# unit × NaN × derivative intersection had no coverage before.

@testitem "OOB zero contract: NaN propagates through derivative OOB (Real)" begin
    x = collect(0.0:1.0:4.0)
    y = [1.0, 2.0, 3.0, 4.0, 5.0]

    @testset "FillExtrap(NaN): fill value IS the OOB cell data" begin
        itp = linear_interp(x, y; extrap = FillExtrap(NaN))
        @test isnan(itp(-1.0; deriv = DerivOp(1)))
        @test isnan(itp(9.0; deriv = DerivOp(1)))
        # a finite fill is data too → 0 * finite = 0 (not NaN)
        finite = linear_interp(x, y; extrap = FillExtrap(99.0))
        @test finite(-1.0; deriv = DerivOp(1)) == 0.0
    end

    @testset "ClampExtrap: a NaN boundary value extends into the OOB cell" begin
        yn = copy(y)
        yn[end] = NaN
        itp = linear_interp(x, yn; extrap = ClampExtrap())
        @test isnan(itp(9.0; deriv = DerivOp(1)))     # right boundary is NaN
        @test itp(-1.0; deriv = DerivOp(1)) == 0.0    # left boundary is finite
    end

    @testset "series mirrors the scalar contract" begin
        s = linear_interp(x, Series(hcat(y, reverse(y))); extrap = FillExtrap(NaN))
        d = s(-1.0; deriv = DerivOp(1))
        @test all(isnan, d)
        @test eltype(d) === Float64
    end
end

@testitem "OOB zero contract: NaN × units × derivative keeps the derivative unit" begin
    using Unitful
    xu = range(1.0u"s", 10.0u"s", length = 9)
    yu = (xu ./ u"s") .^ 2 .* u"m"
    q_oob = 20.0u"s"
    ms = u"m" / u"s"

    @testset "FillExtrap(NaN·m) — NaN survives AND is scaled to m/s" begin
        itp = linear_interp(xu, yu; extrap = FillExtrap(NaN * u"m"))
        d = itp(q_oob; deriv = DerivOp(1))
        @test isnan(ustrip(d))
        @test unit(d) == ms                       # value unit (m) would be wrong
    end

    @testset "ClampExtrap with a NaN boundary value" begin
        yn = collect(yu)
        yn[end] = NaN * u"m"
        d = linear_interp(xu, yn; extrap = ClampExtrap())(q_oob; deriv = DerivOp(1))
        @test isnan(ustrip(d))
        @test unit(d) == ms
    end

    @testset "series: NaN + unit derivative" begin
        s = linear_interp(xu, Series(hcat(yu, 2 .* yu)); extrap = FillExtrap(NaN * u"m"))
        d = s(q_oob; deriv = DerivOp(1))
        @test all(v -> isnan(ustrip(v)), d)
        @test unit(eltype(d)) == ms
    end

    @testset "order > 1 scales by coordᴺ" begin
        d2 = cubic_interp(xu, yu; extrap = ClampExtrap())(q_oob; deriv = DerivOp(2))
        @test iszero(ustrip(d2))
        @test unit(d2) == u"m" / u"s"^2
    end
end

@testitem "OOB zero contract: carrier width (Int values stay Int; derivatives float)" begin
    xi = collect(0:10)
    yi = collect(10:20)

    @testset "1D scalar" begin
        itp = constant_interp(xi, yi; extrap = ClampExtrap())
        @test itp(3) === 13                       # value keeps the Int carrier
        @test itp(-2) === 10                      # OOB value keeps it too
        # Int value / Int coordinate is a rational → the derivative carrier floats
        @test itp(3; deriv = DerivOp(1)) === 0.0
        @test itp(-2; deriv = DerivOp(1)) === 0.0
    end

    @testset "series mirrors the scalar carrier" begin
        s = constant_interp(xi, Series(hcat(yi, yi .+ 10)); extrap = ClampExtrap())
        @test eltype(s(3)) === Int
        @test eltype(s(3; deriv = DerivOp(1))) === Float64
        @test eltype(s(-2; deriv = DerivOp(1))) === Float64
    end
end
