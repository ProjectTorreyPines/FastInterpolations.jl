@testitem "no-wrap helpers" begin
    using FastInterpolations: _fielddiff, _fieldsum, _linear_value_blend

    # promote-operands difference: wrap-free for UInt8, bit-identical for Float
    @test _fielddiff(Float64, UInt8(50), UInt8(200)) == -150.0   # raw UInt8 sub would be +106
    @test _fielddiff(Float64, 3.0, 1.0) === 2.0                  # Float identity
    @test _fieldsum(Float64, UInt8(200), UInt8(200)) == 400.0    # raw UInt8 add would wrap
    @test _fieldsum(Float64, 1.5, 2.5) === 4.0

    # signed-overflow (Int8) is also fixed: raw Int8(-128) - Int8(1) wraps to +127
    @test _fielddiff(Float64, Int8(-128), Int8(1)) == -129.0
    @test _fieldsum(Float64, Int8(100), Int8(100)) == 200.0   # raw Int8 add would wrap to -56

    # convex linear blend: endpoint-exact + correct on descending finite cell
    @test _linear_value_blend(1.0, UInt8(200), UInt8(50)) == 50.0   # t=1 → yR exactly
    @test _linear_value_blend(0.0, UInt8(200), UInt8(50)) == 200.0  # t=0 → yL exactly
    @test _linear_value_blend(0.5, UInt8(200), UInt8(50)) == 125.0  # true midpoint (raw wrap → 253)
    @test _linear_value_blend(0.5, 0.2, 0.8) ≈ 0.5
end

@testitem "linear value kernel no-wrap" begin
    using FastInterpolations
    const FI = FastInterpolations

    # Raw 1D kernel with native UInt8 (external-consumer style): descending cell.
    @test FI._linear_kernel(FI.EvalValue(), UInt8(200), UInt8(50), 1.0, 0.5) == 125.0

    # Anchored value kernel: build a real anchor on a unit cell (alpha=0.5,
    # inv_h=1.0) via the internal builder, then feed native UInt8 corner values.
    aq = FI._anchor_query([1.0, 2.0], 1.5, Val(:linear))   # alpha=0.5, inv_h=1.0
    @test FI._linear_kernel(FI.EvalValue(), UInt8(200), UInt8(50), aq) == 125.0

    # Bilinear (ND) collapses through the 1D value kernel — spot-check the helper
    # form is endpoint-exact so an N0f8-range write stays in-range.
    @test FI._linear_kernel(FI.EvalValue(), UInt8(200), UInt8(50), 1.0, 1.0) == 50.0
end

@testitem "linear value convex: Float behavior" begin
    using FastInterpolations
    const FI = FastInterpolations
    # Endpoint-exact (the reason for convex over slope form)
    @test FI._linear_kernel(FI.EvalValue(), 0.2, 0.9, 1.0, 1.0) === 0.9
    @test FI._linear_kernel(FI.EvalValue(), 0.2, 0.9, 1.0, 0.0) === 0.2
    # Bounded: interior stays within [yL,yR]
    for α in 0.0:0.1:1.0
        v = FI._linear_kernel(FI.EvalValue(), 0.2, 0.9, 1.0, α)
        @test 0.2 <= v <= 0.9
    end
end

@testitem "linear deriv kernel no-wrap" begin
    using FastInterpolations
    const FI = FastInterpolations
    # slope across a descending UInt8 cell: (50-200)/1 = -150
    @test FI._linear_kernel(FI.EvalDeriv1(), UInt8(200), UInt8(50), 1.0, 0.5) == -150.0
    aq = FI._anchor_query([1.0, 2.0], 1.5, Val(:linear))   # inv_h=1.0
    @test FI._linear_kernel(FI.EvalDeriv1(), UInt8(200), UInt8(50), aq) == -150.0
    # Float bit-identical
    @test FI._linear_kernel(FI.EvalDeriv1(), 0.2, 0.9, 2.0, 0.5) === (0.9 - 0.2) * 2.0 * one(0.5)
end

@testitem "linear deriv kernel inferred" begin
    using FastInterpolations, Test
    const FI = FastInterpolations
    @test (@inferred FI._linear_kernel(FI.EvalDeriv1(), 0.2, 0.9, 2.0, 0.5)) isa Float64
end

@testitem "no-wrap helpers preserve natural promotion (no forced convert)" begin
    using FastInterpolations: _fielddiff, _fieldsum
    using ForwardDiff: Dual

    # Field types: the fast-path method (a::Tc, b::Tc) is plain `a - b`/`a + b` —
    # byte-for-byte identical to the old code, NO convert.
    @test _fielddiff(Float64, 3.0, 1.0) === 3.0 - 1.0
    @test _fieldsum(Float64, 3.0, 1.0) === 3.0 + 1.0
    @test _fielddiff(Float32, 3.0f0, 1.0f0) === 3.0f0 - 1.0f0
    @test _fielddiff(ComplexF64, 2.0 + 1im, 1.0 + 0im) === (2.0 + 1im) - (1.0 + 0im)

    # Duck/AD types take the natural-promotion path (Tc === the duck type) — the
    # result is the exact Dual, partials intact, never flattened through Float.
    d1 = Dual(3.0, 1.0); d2 = Dual(1.0, 0.0)
    Td = typeof(d1)
    @test _fielddiff(Td, d1, d2) === d1 - d2          # identity, partials preserved
    @test _fieldsum(Td, d1, d2) === d1 + d2

    # Mixed field lift (e.g. Float operand in a Dual coefficient field, as in
    # AD-wrt-grid): convert lifts to the field exactly as natural promotion would.
    @test _fielddiff(Td, 3.0, 1.0) === Td(3.0) - Td(1.0)
end

@testitem "build-overflow: PolyFit/BC stencils" begin
    using FastInterpolations
    const FI = FastInterpolations
    # Use AbstractRange grid so _compute_deriv1 (not _weighted_sum) is exercised.
    x = 1.0:1.0:5.0
    yU = UInt8[200, 50, 150, 40, 210]
    # CubicFit / QuadraticFit / LinearFit endpoint-derivative BCs exercise the PolyFit kernels.
    for bc in (FI.CubicFit(), FI.QuadraticFit(), FI.LinearFit())
        itpN = FI.cubic_interp(x, yU; bc = bc)
        itpF = FI.cubic_interp(x, Float64.(yU); bc = bc)
        for q in (1.5, 2.5, 3.5, 4.5)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

@testitem "build-overflow: quadratic" begin
    using FastInterpolations
    const FI = FastInterpolations
    x = [1.0, 2.0, 3.0, 4.0]
    yU = UInt8[200, 50, 100, 30]
    yI = Int8[100, -50, 60, -30]
    for yN in (yU, yI)
        itpN = FI.quadratic_interp(x, yN); itpF = FI.quadratic_interp(x, Float64.(yN))
        for q in (1.5, 2.5, 3.5, 2.25, 3.75)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

@testitem "build-overflow: cubic" begin
    using FastInterpolations
    const FI = FastInterpolations

    # Float64 reference must match the narrow-eltype build (no wrap).
    function assert_no_wrap(ctor, x, yN; qs = (1.5, 2.5, 3.5, 2.25, 3.75), atol = 1.0e-9)
        itpN = ctor(x, yN)
        itpF = ctor(x, Float64.(yN))
        for q in qs
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = atol)
        end
        return itpN
    end

    x = [1.0, 2.0, 3.0, 4.0]
    # Descending cells so yR < yL triggers unsigned/overflow wrap.
    yU = UInt8[200, 50, 100, 30]
    yI = Int8[100, -50, 60, -30]

    assert_no_wrap(FI.cubic_interp, x, yU)
    assert_no_wrap(FI.cubic_interp, x, yI)

    # Direct deriv1 kernel wrap pin (z already coeff-typed; yR-yL is the wrap site).
    # cubic EvalDeriv1: with zL=zR=0, slope reduces to (yR-yL)*inv_h.
    @test FI._cubic_kernel(FI.EvalDeriv1(), 0.0, 0.0, UInt8(200), UInt8(50), 1.0, 1.0, 0.0, 1.0) == -150.0
end

@testitem "build-overflow: pchip (precompute + onthefly)" begin
    using FastInterpolations
    const FI = FastInterpolations
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    yU = UInt8[200, 50, 150, 40, 210]
    yI = Int8[100, -50, 60, -30, 90]
    for yN in (yU, yI), strat in (FI.PreCompute(), FI.OnTheFly())
        itpN = FI.pchip_interp(x, yN; coeffs = strat)
        itpF = FI.pchip_interp(x, Float64.(yN); coeffs = strat)
        for q in (1.5, 2.5, 3.5, 4.5, 2.25)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

@testitem "build-overflow: akima (precompute + onthefly)" begin
    using FastInterpolations
    const FI = FastInterpolations
    x = collect(1.0:1.0:8.0)
    yU = UInt8[200, 50, 150, 40, 210, 30, 180, 60]
    yI = Int8[100, -50, 60, -30, 90, -40, 70, -20]
    for yN in (yU, yI), strat in (FI.PreCompute(), FI.OnTheFly())
        itpN = FI.akima_interp(x, yN; coeffs = strat)
        itpF = FI.akima_interp(x, Float64.(yN); coeffs = strat)
        for q in (1.5, 3.5, 5.5, 6.5, 4.25)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

@testitem "build-overflow: cardinal (precompute + onthefly)" begin
    using FastInterpolations
    const FI = FastInterpolations
    x = collect(1.0:1.0:6.0)
    yU = UInt8[200, 50, 150, 40, 210, 30]
    yI = Int8[100, -50, 60, -30, 90, -40]
    for yN in (yU, yI), strat in (FI.PreCompute(), FI.OnTheFly())
        itpN = FI.cardinal_interp(x, yN; coeffs = strat)
        itpF = FI.cardinal_interp(x, Float64.(yN); coeffs = strat)
        for q in (1.5, 2.5, 3.5, 4.5, 3.25)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

@testitem "build-overflow: integrals" begin
    using FastInterpolations
    const FI = FastInterpolations

    # The integral kernels are called directly with raw corner values (yL, yR).
    # For the stored interpolant, UInt8 is promoted to Float64 by the constructor,
    # so the high-level integrate() path does not exercise the raw-UInt8 arithmetic.
    # These tests validate the kernel functions directly — the same code paths that
    # external consumers (e.g. duck-typed colorant grids) would exercise.

    # ── linear partial: (yR - yL) wrap ─────────────────────────────────────────
    # UInt8(50) - UInt8(200) = UInt8(106) wraps; correct result is -150 → integral = 125
    # Full-span partial (u0=0, u1=h=1): du*(half_slope*(u1+u0) + yL)
    @test FI._linear_integral_kernel(FI._EvalIntegralPartial(), UInt8(200), UInt8(50), 1.0, 0.0, 1.0) ≈ 125.0
    @test FI._linear_integral_kernel(FI._EvalIntegralPartial(), Int8(100), Int8(-30), 1.0, 0.0, 0.5) ≈
        FI._linear_integral_kernel(FI._EvalIntegralPartial(), 100.0, -30.0, 1.0, 0.0, 0.5)

    # ── linear full-cell: (yL + yR) wrap ────────────────────────────────────────
    # UInt8(200) + UInt8(220) = 420 mod 256 = 164; correct is 420 → h/2*420 = 210
    @test FI._linear_integral_kernel(FI._EvalIntegralCell(), UInt8(200), UInt8(220), 1.0) ≈ 210.0
    @test FI._linear_integral_kernel(FI._EvalIntegralCell(), Int8(100), Int8(100), 1.0) ≈ 100.0

    # ── cubic partial: (yR - yL) wrap ───────────────────────────────────────────
    # zL=zR=0 → c2 = (yR-yL)*(inv_h/2); with UInt8 yR=50, yL=200: wrap → 53*inv_h
    # With z=0, full span: integral = h/2*(yL+yR) = 125 (same as linear cell)
    @test FI._cubic_integral_kernel(FI._EvalIntegralPartial(), 0.0, 0.0, UInt8(200), UInt8(50), 1.0, 0.0, 1.0) ≈ 125.0
    @test FI._cubic_integral_kernel(FI._EvalIntegralPartial(), 0.0, 0.0, Int8(100), Int8(-30), 1.0, 0.0, 1.0) ≈
        FI._cubic_integral_kernel(FI._EvalIntegralPartial(), 0.0, 0.0, 100.0, -30.0, 1.0, 0.0, 1.0)

    # ── cubic full-cell: (yL + yR) wrap ─────────────────────────────────────────
    # h/2 * muladd(-(h²/12)*(zL+zR), yL+yR); with z=0: h/2*(yL+yR)
    @test FI._cubic_integral_kernel(FI._EvalIntegralCell(), 0.0, 0.0, UInt8(200), UInt8(220), 1.0) ≈ 210.0
    @test FI._cubic_integral_kernel(FI._EvalIntegralCell(), 0.0, 0.0, Int8(100), Int8(100), 1.0) ≈ 100.0

    # ── Float64 bit-identical (fast path) ───────────────────────────────────────
    @test FI._linear_integral_kernel(FI._EvalIntegralPartial(), 200.0, 50.0, 1.0, 0.0, 1.0) === 125.0
    @test FI._linear_integral_kernel(FI._EvalIntegralCell(), 200.0, 220.0, 1.0) === 210.0
    @test FI._cubic_integral_kernel(FI._EvalIntegralPartial(), 0.0, 0.0, 200.0, 50.0, 1.0, 0.0, 1.0) === 125.0
    @test FI._cubic_integral_kernel(FI._EvalIntegralCell(), 0.0, 0.0, 200.0, 220.0, 1.0) === 210.0

    # ── quadratic ND kernel: (fR - fL) wrap ─────────────────────────────────────
    # _quadratic_integral_kernel_nd: s = (fR - fL) * inv_h; with dfL=0, u0=0, u1=1
    @test FI._quadratic_integral_kernel_nd(UInt8(200), UInt8(50), 0.0, 1.0, 1.0, 0.0, 1.0) ≈
        FI._quadratic_integral_kernel_nd(200.0, 50.0, 0.0, 1.0, 1.0, 0.0, 1.0)
end

@testitem "build-overflow: periodic seams" begin
    using FastInterpolations
    const FI = FastInterpolations
    x = collect(1.0:1.0:6.0)
    # closed cycle: y[1] == y[end] for :inclusive
    yU = UInt8[200, 50, 150, 40, 210, 200]
    bc = FI.PeriodicBC()   # :inclusive by default
    for ctor in (FI.pchip_interp, FI.akima_interp, FI.cardinal_interp),
            strat in (FI.PreCompute(), FI.OnTheFly())
        itpN = ctor(x, yU; bc = bc, coeffs = strat)
        itpF = ctor(x, Float64.(yU); bc = bc, coeffs = strat)
        for q in (1.5, 2.5, 4.5, 5.5)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end
