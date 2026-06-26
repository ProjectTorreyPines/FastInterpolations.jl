# Guards for the OOB flat-extrapolation carrier idiom (ClampExtrap/FillExtrap).
#
# `_promote_extrap_val` (`val + zero(xq)*zero(val)`) and `_promote_extrap_zero` (`0*val + …`)
# in src/core/utils.jl look like they carry a pointless `+ 0.0`, but those arithmetic forms are
# LOAD-BEARING: the trailing add normalizes signed zero (`-0.0 → +0.0`) and keeps Unitful
# dimension-correctness, while `0*val` propagates NaN/Inf. Folding them to `oftype`/`convert`
# (to drop the `+0.0`) silently regresses these. These guards fail loudly if that happens.

@testitem "OOB flat-extrap carrier idiom — load-bearing guards" begin
    using FastInterpolations: _promote_extrap_val, _promote_extrap_zero
    using ForwardDiff

    @testset "signed-zero normalization — value path" begin
        # `val + zero(xq)*zero(val)` flushes -0.0 → +0.0; an oftype/convert fold would keep -0.0.
        @test !signbit(_promote_extrap_val(-0.0, 1.0))
        @test !signbit(_promote_extrap_val(-0.0f0, 1.0f0))
        # end-to-end: ClampExtrap returning a -0.0 boundary sample
        itp = linear_interp([0.0, 1.0, 2.0], [-0.0, 1.0, 2.0]; extrap = ClampExtrap())
        @test !signbit(itp(-5.0))                       # OOB-left → y[1] = -0.0 → +0.0
        # FillExtrap with a -0.0 fill value
        itpf = linear_interp([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]; extrap = FillExtrap(-0.0))
        @test !signbit(itpf(-5.0))
    end

    @testset "signed-zero normalization — deriv path" begin
        # `0*val` of a negative sample is -0.0; the trailing `+ zero*zero` normalizes to +0.0.
        @test iszero(_promote_extrap_zero(-3.5, 1.0))
        @test !signbit(_promote_extrap_zero(-3.5, 1.0))
    end

    @testset "NaN/Inf propagation — deriv path keeps `0*val`" begin
        # 0*NaN = NaN, 0*Inf = NaN: a non-finite boundary sample must surface, not a clean 0.
        @test isnan(_promote_extrap_zero(NaN, 1.0))
        @test isnan(_promote_extrap_zero(Inf, 1.0))
        @test isnan(_promote_extrap_val(NaN, 1.0))
        @test isinf(_promote_extrap_val(Inf, 1.0))
        # end-to-end via the native derivative API (routes through _promote_extrap_zero)
        itpn = linear_interp([0.0, 1.0, 2.0], [NaN, 1.0, 2.0]; extrap = ClampExtrap())
        @test isnan(itpn(-5.0; deriv = DerivOp(1)))     # OOB-left, NaN boundary → NaN derivative
    end

    @testset "AD carrier — OOB value-path derivative is zero, value correct, type stable" begin
        itp = linear_interp([0.0, 1.0, 2.0], [10.0, 20.0, 30.0]; extrap = ClampExtrap())
        @test ForwardDiff.derivative(itp, -5.0) == 0.0  # clamp is flat outside the domain
        @test ForwardDiff.derivative(itp, 7.0) == 0.0
        @test itp(-5.0) == 10.0
        @test itp(7.0) == 30.0
        @test (@inferred itp(-5.0)) isa Float64
    end
end
