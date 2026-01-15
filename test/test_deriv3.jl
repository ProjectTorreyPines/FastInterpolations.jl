# ========================================
# Deriv3 Tests for FastInterpolations.jl
# ========================================
# Phase 1: Core Types and Dispatch

using Test
using FastInterpolations
using FastInterpolations: EvalDeriv3, AbstractEvalOp

@testset "Deriv3 - Phase 1: Core Types" begin

    @testset "EvalDeriv3 type exists and is subtype of AbstractEvalOp" begin
        @test isdefined(FastInterpolations, :EvalDeriv3)
        @test EvalDeriv3 <: AbstractEvalOp
        @test EvalDeriv3() isa AbstractEvalOp
    end

    @testset "@_dispatch_deriv handles deriv=3" begin
        result = FastInterpolations.@_dispatch_deriv 3 => op begin
            op
        end
        @test result isa EvalDeriv3
    end

    @testset "@_dispatch_deriv throws for deriv=4" begin
        @test_throws ArgumentError begin
            FastInterpolations.@_dispatch_deriv 4 => op begin
                op
            end
        end
    end

    @testset "@_dispatch_deriv throws for deriv=-1" begin
        @test_throws ArgumentError begin
            FastInterpolations.@_dispatch_deriv -1 => op begin
                op
            end
        end
    end
end

@testset "Deriv3 - Phase 2: Kernel Implementations" begin

    @testset "Cubic kernel - S'''(x) = (zR - zL) / h" begin
        zL, zR = 1.0, 3.0
        yL, yR = 0.0, 1.0
        h = 0.5
        inv_h = 1.0 / h
        dL, dR = 0.2, 0.3

        result = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, yL, yR, h, inv_h, dL, dR
        )

        expected = (zR - zL) / h
        @test result ≈ expected
    end

    @testset "Cubic kernel - result is constant within interval" begin
        zL, zR = 2.0, 5.0
        h, inv_h = 0.1, 10.0

        result1 = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0, 0.0, h, inv_h, 0.02, 0.08
        )
        result2 = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0, 0.0, h, inv_h, 0.05, 0.05
        )
        result3 = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0, 0.0, h, inv_h, 0.09, 0.01
        )

        @test result1 ≈ result2 ≈ result3
    end

    @testset "Cubic kernel - Float32 support" begin
        zL, zR = 1.0f0, 2.0f0
        h, inv_h = 0.5f0, 2.0f0

        result = FastInterpolations._cubic_kernel(
            EvalDeriv3(), zL, zR, 0.0f0, 0.0f0, h, inv_h, 0.1f0, 0.4f0
        )

        @test result isa Float32
        @test result ≈ 2.0f0
    end

    @testset "Linear kernel - returns zero" begin
        yL, yR = 1.0, 5.0
        h, dL = 0.5, 0.2

        result = FastInterpolations._linear_kernel(EvalDeriv3(), yL, yR, h, dL)
        @test result === zero(Float64)
    end

    @testset "Linear kernel - Float32 returns Float32 zero" begin
        result = FastInterpolations._linear_kernel(
            EvalDeriv3(), 1.0f0, 2.0f0, 0.5f0, 0.1f0
        )
        @test result === zero(Float32)
    end

    @testset "Quadratic kernel - returns zero" begin
        a, d, y = 1.0, 2.0, 3.0
        dL = 0.5

        result = FastInterpolations._quadratic_kernel(EvalDeriv3(), a, d, y, dL)
        @test result === zero(Float64)
    end

    @testset "Quadratic kernel - Float32 returns Float32 zero" begin
        result = FastInterpolations._quadratic_kernel(
            EvalDeriv3(), 1.0f0, 2.0f0, 3.0f0, 0.5f0
        )
        @test result === zero(Float32)
    end

    @testset "Constant kernel - returns zero" begin
        yL, yR = 5.0, 5.0
        h, dL = 0.5, 0.2

        result = FastInterpolations._constant_kernel(
            EvalDeriv3(), yL, yR, h, dL, Val(:left)
        )
        @test result === zero(Float64)
    end

    @testset "Constant kernel - Float32 returns Float32 zero" begin
        result = FastInterpolations._constant_kernel(
            EvalDeriv3(), 5.0f0, 5.0f0, 0.5f0, 0.2f0, Val(:left)
        )
        @test result === zero(Float32)
    end
end

@testset "Deriv3 - Phase 3: Anchor Infrastructure" begin
    FI = FastInterpolations

    @testset "Weight computation - _compute_anchor_weights(::EvalDeriv3)" begin
        h, inv_h = 0.1, 10.0
        dL, dR = 0.03, 0.07

        w3 = FI._compute_anchor_weights(EvalDeriv3(), h, inv_h, dL, dR)

        @test w3[1] === 0.0
        @test w3[2] === 0.0
        @test w3[3] === -10.0
        @test w3[4] === 10.0
    end

    @testset "Weight computation - Float32 support" begin
        h, inv_h = 0.1f0, 10.0f0
        dL, dR = 0.03f0, 0.07f0

        w3 = FI._compute_anchor_weights(EvalDeriv3(), h, inv_h, dL, dR)

        @test w3[1] isa Float32
        @test w3[3] === -10.0f0
        @test w3[4] === 10.0f0
    end

    @testset "Anchor struct has w3 field" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        aq = FI._anchor_query(x, 0.5)

        @test hasfield(typeof(aq), :w3)
        @test aq.w3 isa NTuple{4, Float64}
    end

    @testset "Anchored weights dispatch for EvalDeriv3" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        aq = FI._anchor_query(x, 0.5)

        w3 = FI._anchored_weights(aq, EvalDeriv3())
        @test w3 === aq.w3
    end

    @testset "Anchored evaluation with deriv=3" begin
        x = collect(range(0.0, 1.0, 11))
        y = x.^3
        itp = cubic_interp(x, y)

        direct_val = itp(0.5; deriv=3)

        aq = FI._anchor_query(x, 0.5)
        anchored_val = itp(aq; deriv=3)

        @test direct_val ≈ anchored_val
    end

    @testset "Anchored evaluation - constant within interval" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        xq1, xq2, xq3 = 0.51, 0.55, 0.59

        aq1 = FI._anchor_query(x, xq1)
        aq2 = FI._anchor_query(x, xq2)
        aq3 = FI._anchor_query(x, xq3)

        if aq1.idx == aq2.idx && aq2.idx == aq3.idx
            @test itp(aq1; deriv=3) ≈ itp(aq2; deriv=3) ≈ itp(aq3; deriv=3)
        end
    end

    @testset "Anchored extrapolation - :constant returns zero" begin
        x = collect(range(0.0, 1.0, 11))
        y = x.^3
        itp = cubic_interp(x, y; extrap=:constant)

        aq_lo = FI._anchor_query(x, -0.5)
        @test itp(aq_lo; deriv=3) === 0.0

        aq_hi = FI._anchor_query(x, 1.5)
        @test itp(aq_hi; deriv=3) === 0.0
    end

    @testset "Anchored evaluation - type stability" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        aq = FI._anchor_query(x, 0.5)

        @inferred itp(aq; deriv=3)
    end

    @testset "Anchored evaluation - Float32" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        y = sin.(2f0 * Float32(π) .* x)
        itp = cubic_interp(x, y)

        aq = FI._anchor_query(x, 0.5f0)
        val = itp(aq; deriv=3)

        @test val isa Float32
        @test aq.w3 isa NTuple{4, Float32}
    end
end

@testset "Deriv3 - Phase 4: DerivativeView & Exports" begin
    FI = FastInterpolations

    @testset "deriv3() factory - CubicInterpolant" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        d3 = FI.deriv3(itp)

        @test d3 isa FI.DerivativeView{3}
        @test d3.parent === itp
    end

    @testset "deriv3() factory - LinearInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = 2.0 .* x .+ 1.0
        itp = linear_interp(x, y)

        d3 = FI.deriv3(itp)

        @test d3 isa FI.DerivativeView{3}
        @test d3(0.5) === 0.0
    end

    @testset "deriv3() factory - QuadraticInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = x.^2
        itp = quadratic_interp(x, y)

        d3 = FI.deriv3(itp)

        @test d3 isa FI.DerivativeView{3}
        @test d3(0.5) === 0.0
    end

    @testset "deriv3() factory - ConstantInterpolant" begin
        x = collect(range(0.0, 1.0, 11))
        y = fill(5.0, length(x))
        itp = constant_interp(x, y)

        d3 = FI.deriv3(itp)

        @test d3 isa FI.DerivativeView{3}
        @test d3(0.5) === 0.0
    end

    @testset "DerivativeView{3} callable - scalar evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        d3 = FI.deriv3(itp)

        @test d3(0.5) == itp(0.5; deriv=3)
        @test d3(0.25) == itp(0.25; deriv=3)
    end

    @testset "DerivativeView{3} callable - broadcasting" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        d3 = FI.deriv3(itp)
        xs = [0.1, 0.5, 0.9]

        results = d3.(xs)
        expected = [itp(xi; deriv=3) for xi in xs]

        @test results ≈ expected
    end

    @testset "DerivativeView{3} - type stability" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y)

        d3 = FI.deriv3(itp)

        @inferred d3(0.5)
        @inferred FI.deriv3(itp)
    end

    @testset "DerivativeView{3} - Float32 support" begin
        x = collect(range(0.0f0, 1.0f0, 101))
        y = sin.(2f0 * Float32(π) .* x)
        itp = cubic_interp(x, y)

        d3 = FI.deriv3(itp)
        val = d3(0.5f0)

        @test val isa Float32
    end

    @testset "Exports - EvalDeriv3 is exported" begin
        @test :EvalDeriv3 in names(FastInterpolations)
    end

    @testset "Exports - deriv3 is exported" begin
        @test :deriv3 in names(FastInterpolations)
    end

    @testset "deriv3 matches itp(x; deriv=3) exactly" begin
        x = collect(range(0.0, 1.0, 101))
        y = x.^3
        itp = cubic_interp(x, y)

        d3 = FI.deriv3(itp)

        for xq in [0.1, 0.25, 0.5, 0.75, 0.9]
            @test d3(xq) === itp(xq; deriv=3)
        end
    end
end

@testset "Deriv3 - Phase 5: Series & Extrapolation" begin

    @testset "Extrapolation - :constant returns zero" begin
        x = collect(range(0.0, 1.0, 11))
        y = x.^3
        itp = cubic_interp(x, y; extrap=:constant)

        @test itp(-0.5; deriv=3) === 0.0
        @test itp(-1.0; deriv=3) === 0.0
        @test itp(1.5; deriv=3) === 0.0
        @test itp(2.0; deriv=3) === 0.0
    end

    @testset "Extrapolation - :extension uses boundary polynomial" begin
        x = collect(range(0.0, 1.0, 11))
        y = x.^3
        itp = cubic_interp(x, y; extrap=:extension)

        val_below = itp(-0.5; deriv=3)
        val_first = itp(0.05; deriv=3)
        @test val_below ≈ val_first

        val_above = itp(1.5; deriv=3)
        val_last = itp(0.95; deriv=3)
        @test val_above ≈ val_last
    end

    @testset "Extrapolation - :none throws DomainError" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:none)

        @test_throws DomainError itp(-0.5; deriv=3)
        @test_throws DomainError itp(1.5; deriv=3)
    end

    @testset "Extrapolation - :wrap (periodic)" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:wrap)

        val_below = itp(-0.2; deriv=3)
        val_equiv = itp(0.8; deriv=3)

        @test isfinite(val_below)
        @test isfinite(val_equiv)
    end

    @testset "CubicSeriesInterpolant - scalar evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        vals = sitp(0.5; deriv=3)

        @test length(vals) == 2
        @test vals isa Vector
    end

    @testset "CubicSeriesInterpolant - in-place scalar" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        vals = sitp(0.5; deriv=3)
        out = similar(vals)
        sitp(out, 0.5; deriv=3)

        @test out ≈ vals
    end

    @testset "CubicSeriesInterpolant - vector evaluation" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        xq = [0.1, 0.5, 0.9]
        results = sitp(xq; deriv=3)

        @test length(results) == 2
        @test length(results[1]) == 3
        @test length(results[2]) == 3
    end

    @testset "CubicSeriesInterpolant - in-place vector" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        xq = [0.1, 0.5, 0.9]
        results = sitp(xq; deriv=3)

        outputs = [similar(xq) for _ in 1:2]
        sitp(outputs, xq; deriv=3)

        @test outputs[1] ≈ results[1]
        @test outputs[2] ≈ results[2]
    end

    @testset "CubicSeriesInterpolant - extrapolation :constant" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = x.^3
        y2 = x.^2
        sitp = cubic_interp(x, [y1, y2]; extrap=:constant)

        vals_below = sitp(-0.5; deriv=3)
        vals_above = sitp(1.5; deriv=3)

        @test vals_below[1] === 0.0
        @test vals_below[2] === 0.0
        @test vals_above[1] === 0.0
        @test vals_above[2] === 0.0
    end

    @testset "LinearSeriesInterpolant - deriv=3 returns zero" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = 2.0 .* x
        y2 = 3.0 .* x
        sitp = linear_interp(x, [y1, y2])

        vals = sitp(0.5; deriv=3)

        @test all(v === 0.0 for v in vals)
    end

    @testset "Series interpolant - type stability" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        sitp = cubic_interp(x, [y1, y2])

        @inferred sitp(0.5; deriv=3)
    end

    @testset "Series interpolant - Float32" begin
        x = collect(range(0.0f0, 1.0f0, 101))
        y1 = sin.(2f0 * Float32(π) .* x)
        y2 = cos.(2f0 * Float32(π) .* x)
        sitp = cubic_interp(x, [y1, y2])

        vals = sitp(0.5f0; deriv=3)

        @test eltype(vals) === Float32
    end
end
