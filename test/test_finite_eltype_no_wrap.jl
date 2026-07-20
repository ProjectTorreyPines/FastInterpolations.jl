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

# Direct-kernel pin for the PolyFit *homogeneous* deriv overload (f::NTuple{N,T},
# inv_h::T). The public API always supplies a float inv_h → mixed overload, so the
# homogeneous narrow path needs its own pin: its coefficient-field type must widen a
# narrow T (the footgun was `Base.promote_op(*, T, T) === UInt8`, which did not widen
# → the divided difference wrapped: (50-200) read as 106). Degree-1 only — for it the
# field type is the sole determinant (no `-inv_h/2` coefficient on a raw narrow inv_h).
@testitem "no-wrap: PolyFit homogeneous deriv field type (degree 1)" begin
    using FastInterpolations
    const FI = FastInterpolations
    for side in (FI.LeftSide(), FI.RightSide())
        @test FI._compute_deriv1(FI.PolyFit{1}(), side, (UInt8(200), UInt8(50)), UInt8(1)) ==
            FI._compute_deriv1(FI.PolyFit{1}(), side, (200.0, 50.0), 1.0)
        @test FI._compute_deriv1(FI.PolyFit{1}(), side, (Int8(100), Int8(-50)), Int8(1)) ==
            FI._compute_deriv1(FI.PolyFit{1}(), side, (100.0, -50.0), 1.0)
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

    # Quadratic ND no longer has a raw-corner kernel: the separable engine
    # multiplies each node value by a grid-typed (Float) weight before summing,
    # so a finite-eltype quadratic payload widens and never modular-wraps. Pinned
    # end-to-end by the N0f8/Gray carrier tests below.
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

@testitem "no-wrap invariant property sweep" setup = [] begin
    using FastInterpolations, Random
    const FI = FastInterpolations
    rng = MersenneTwister(20260627)
    ctors = (
        FI.linear_interp,
        FI.constant_interp,
        FI.quadratic_interp,
        FI.cubic_interp,
        FI.pchip_interp,
        FI.akima_interp,
        FI.cardinal_interp,
    )
    for _ in 1:25
        n = rand(rng, 5:9)
        x = collect(1.0:1.0:n)
        yU = rand(rng, UInt8, n)               # random ⇒ many descending cells
        for ctor in ctors
            itpN = ctor(x, yU)
            itpF = ctor(x, Float64.(yU))
            for q in range(1.0, Float64(n); length = 13)
                @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-7)
            end
        end
    end
end

# Adjoint audit: narrow-eltype build must equal float build (constructors promote y).
# pchip_adjoint / akima_adjoint call _promote_itp_inputs(x, y) which lifts y
# to Float64 before any secant arithmetic.  Build from narrow UInt8 data and
# verify the adjoint operator gives the same result as the float-built adjoint.
# cardinal_adjoint has NO y parameter (slopes are linear in y; it is data-free)
# so it is excluded — there is no narrow-data path to audit for it.
@testitem "adjoint narrow-data audit" begin
    using FastInterpolations, LinearAlgebra
    const FI = FastInterpolations

    x = collect(1.0:1.0:6.0)
    yU = UInt8[200, 50, 150, 40, 210, 30]
    qs = [1.5, 2.5, 3.5, 4.5, 5.5]
    v = [0.1, -0.3, 0.7, -0.5, 0.2]   # fixed seed so test is deterministic

    for ctor_adj in (FI.pchip_adjoint, FI.akima_adjoint)
        AN = ctor_adj(x, yU, qs)
        AF = ctor_adj(x, Float64.(yU), qs)
        # adj(v) maps co-tangent vector v ∈ ℝ^|qs| → f_bar ∈ ℝ^|x|
        fN = Float64.(AN(v))
        fF = Float64.(AF(v))
        @test isapprox(fN, fF; atol = 1.0e-7)
    end
end

# Fixed-point `Real` (N0f8) is NOT in `_PromotableValue`, so `_promote_itp_inputs`
# does NOT float it at the adjoint boundary (unlike UInt8 above). The adjoint's own
# secant recomputation must therefore be wrap-free — else the raw `y[k+1]-y[k]` on
# N0f8 wraps, flipping `sign(δ)` monotonicity branches and corrupting the gradient.
@testitem "no-wrap: N0f8 adjoint (un-promoted carrier)" begin
    using FastInterpolations
    using FixedPointNumbers
    const FI = FastInterpolations
    x = collect(1.0:1.0:6.0)
    yN = N0f8.([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])   # descending cells exercise the wrap
    yf = Float64.(yN)
    qs = [1.5, 2.5, 3.5, 4.5, 5.5]
    v = [0.1, -0.3, 0.7, -0.5, 0.2]
    for ctor_adj in (FI.pchip_adjoint, FI.akima_adjoint)
        AN = ctor_adj(x, yN, qs)
        AF = ctor_adj(x, yf, qs)
        @test isapprox(Float64.(AN(v)), Float64.(AF(v)); atol = 1.0e-7)
    end
end

@testitem "Float64 determinism + finiteness at promoted-difference sites" begin
    using FastInterpolations, Random
    const FI = FastInterpolations
    rng = MersenneTwister(1)
    # Non-linear methods route their Float difference sites through the `_fielddiff`
    # fast path (`a::Tc, b::Tc → a - b`), so Float results are unchanged. This smoke
    # only checks determinism + finiteness; the linear VALUE path uses the convex
    # form (intentionally not bit-identical) and is pinned separately below.
    x = collect(1.0:1.0:8.0)
    y = randn(rng, 8)
    for ctor in (
            FI.quadratic_interp, FI.cubic_interp, FI.pchip_interp,
            FI.akima_interp, FI.cardinal_interp,
        )
        itp = ctor(x, y)
        for q in range(1.0, 8.0; length = 21)
            v = itp(q)
            @test v === itp(q)                 # deterministic
            @test isfinite(v)
        end
    end
end

@testitem "no-wrap: inference + allocation" setup = [AllocConstants] begin
    using FastInterpolations, Test
    const FI = FastInterpolations
    x = collect(1.0:1.0:8.0); y = randn(8)
    itp = FI.cubic_interp(x, y)
    @test (@inferred itp(3.3)) isa Float64
    itp(3.3)                                  # warmup
    # 0 on 1.12+; LTS inference leaves a tiny residual box on scalar eval, so use
    # the shared version-aware threshold (0 on 1.12, slack on 1.10) like the ND tests.
    @test @allocated(itp(3.3)) <= ALLOC_THRESHOLD
    # narrow-eltype build is type-stable too
    yU = rand(UInt8, 8)
    itpU = FI.cubic_interp(x, yU)
    @test (@inferred itpU(3.3)) isa Float64
end

# Convex value-kernel contract: endpoint-exact at α=0,1 and bounded within
# [min(yL,yR), max(yL,yR)] for α∈[0,1] — i.e. the result can never overshoot the
# endpoints (the old slope form was not endpoint-exact; a narrow-eltype subtraction
# could wrap past the range, e.g. midpoint of UInt8[200,50] returned 253).
@testitem "linear convex value: endpoint-exact + bounded (no overshoot)" begin
    using FastInterpolations
    const FI = FastInterpolations
    cells = ((0.2, 0.9), (0.9, 0.2), (-0.3, 0.7), (UInt8(50), UInt8(200)), (UInt8(200), UInt8(50)))
    for (yL, yR) in cells
        lo = min(Float64(yL), Float64(yR)); hi = max(Float64(yL), Float64(yR))
        @test FI._linear_kernel(FI.EvalValue(), yL, yR, 1.0, 0.0) == Float64(yL)   # α=0 → yL exact
        @test FI._linear_kernel(FI.EvalValue(), yL, yR, 1.0, 1.0) == Float64(yR)   # α=1 → yR exact
        for α in 0.0:0.05:1.0
            v = Float64(FI._linear_kernel(FI.EvalValue(), yL, yR, 1.0, α))
            @test lo <= v <= hi                                                    # never overshoots
        end
    end

    # End-to-end on a descending UInt8 grid: every value stays in the data range.
    x = [1.0, 2.0, 3.0, 4.0]
    yU = UInt8[200, 50, 150, 40]
    itp = FI.linear_interp(x, yU)
    lo = Float64(minimum(yU)); hi = Float64(maximum(yU))
    for q in range(1.0, 4.0; length = 31)
        @test lo <= Float64(itp(q)) <= hi
    end
end

# n=2 / n=3 use dedicated `_fielddiff` secant branches, distinct from the general
# interior loop and the n∈5:9 sweep. Use Gray{N0f8}, NOT UInt8 — the constructors
# promote integers to Float before the slope kernels, which would hide a `_fielddiff`
# revert; colorant carriers reach the slope builders un-promoted, so reverting
# `_fielddiff` would wrap the n=2/3 secants. (pchip excluded: monotonicity needs
# `sign`, undefined on Gray.) Narrow build must equal the float-channel build.
@testitem "no-wrap: n=2 / n=3 special-case slope paths" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations
    for (x, g) in (
            ([1.0, 2.0], Gray{N0f8}.([0.9, 0.1])),
            ([1.0, 2.0, 3.0], Gray{N0f8}.([0.9, 0.1, 0.8])),
        )
        gf = Float64.(gray.(g))
        for ctor in (FI.linear_interp, FI.constant_interp, FI.akima_interp, FI.cardinal_interp)
            itpG = ctor(x, g); itpF = ctor(x, gf)
            for q in range(first(x), last(x); length = 7)
                @test isapprox(Float64(gray(itpG(q))), Float64(itpF(q)); atol = 1.0e-9)
            end
        end
    end
end

# Cubic PeriodicBC seam RHS (`compute_rhs_periodic!`) `_fielddiff` sites: narrow
# build must equal float (cubic is omitted from the seam loop above — no `coeffs`).
@testitem "build-overflow: cubic periodic seam" begin
    using FastInterpolations
    const FI = FastInterpolations
    x = collect(1.0:1.0:6.0)
    yU = UInt8[200, 50, 150, 40, 210, 200]   # closed cycle (y[1] == y[end])
    itpN = FI.cubic_interp(x, yU; bc = FI.PeriodicBC())
    itpF = FI.cubic_interp(x, Float64.(yU); bc = FI.PeriodicBC())
    for q in (1.5, 2.5, 4.5, 5.5)
        @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
    end
end

# Convex value-FORM pin: the public linear path must use the convex blend
# `muladd(α, yR, muladd(-α, yL, yL))` exactly. A revert to the slope form
# `muladd(α, yR-yL, yL)` perturbs the low bits, so pin the public eval bit-exactly
# against the convex helper (with inv_h=1 so α = q - xL is exact).
@testitem "linear convex value: public path uses convex form (bit-exact)" begin
    using FastInterpolations
    const FI = FastInterpolations
    for (yL, yR, q) in ((0.1, 0.7, 1.3), (0.2, 0.8, 1.25), (-0.3, 0.9, 1.6), (1.7, 0.4, 1.8))
        itp = FI.linear_interp([1.0, 2.0], [yL, yR])
        α = q - 1.0                                  # inv_h = 1 → α = q - xL exactly
        @test itp(q) === FI._linear_value_blend(α, yL, yR)
    end
end

# Quadratic ND eval `_fielddiff` (quadratic_nd_eval.jl): un-promoted N0f8 data
# reaches the eval kernel; the secant `(fR-fL)` must not wrap. UInt8 would be
# floated by the constructor, so use N0f8 — it stays narrow into the kernel.
@testitem "no-wrap: quadratic ND eval (N0f8)" begin
    using FastInterpolations
    using FixedPointNumbers
    const FI = FastInterpolations
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 4.0]
    AN = N0f8.([0.9 0.1 0.8 0.2; 0.1 0.9 0.2 0.8; 0.85 0.15 0.75 0.25; 0.2 0.7 0.3 0.95])
    itpN = FI.quadratic_interp((x, y), AN)
    itpF = FI.quadratic_interp((x, y), Float64.(AN))
    for qx in (1.5, 2.5, 3.25), qy in (1.5, 2.5, 3.75)
        @test isapprox(Float64(itpN(qx, qy)), Float64(itpF(qx, qy)); atol = 1.0e-9)
    end
end

# OnTheFly pchip/akima/cardinal route through `hermite_local_slopes.jl` `_local_slope`
# (a DISTINCT path from the PreCompute `*_slopes.jl` builders). Un-promoted N0f8 must
# not wrap those secants. UInt8 is floated by the constructor → use N0f8.
@testitem "no-wrap: OnTheFly hermite local slopes (N0f8)" begin
    using FastInterpolations
    using FixedPointNumbers
    const FI = FastInterpolations
    x = collect(1.0:1.0:7.0)
    yN = N0f8.([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.85])   # descending cells exercise the wrap
    yF = Float64.(yN)
    for ctor in (FI.pchip_interp, FI.akima_interp, FI.cardinal_interp)
        itpN = ctor(x, yN; coeffs = FI.OnTheFly())
        itpF = ctor(x, yF; coeffs = FI.OnTheFly())
        for q in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 3.25)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

# OnTheFly + PeriodicBC routes through `hermite_periodic_slopes.jl` `_periodic_secant`
# (seam secant `(y[1]-y[n])/seam_h`, plus `mod1`-wrapped cycle secants). Un-promoted
# N0f8 must not wrap. The PreCompute "periodic seams" test above excludes hermite.
@testitem "no-wrap: OnTheFly hermite periodic slopes (N0f8)" begin
    using FastInterpolations
    using FixedPointNumbers
    const FI = FastInterpolations
    x = collect(1.0:1.0:6.0)
    yN = N0f8.([0.9, 0.1, 0.8, 0.2, 0.7, 0.9])   # closed cycle: y[1] == y[end] for :inclusive
    yF = Float64.(yN)
    bc = FI.PeriodicBC()
    for ctor in (FI.pchip_interp, FI.akima_interp, FI.cardinal_interp)
        itpN = ctor(x, yN; bc = bc, coeffs = FI.OnTheFly())
        itpF = ctor(x, yF; bc = bc, coeffs = FI.OnTheFly())
        for q in (1.5, 2.5, 4.5, 5.5)
            @test isapprox(Float64(itpN(q)), Float64(itpF(q)); atol = 1.0e-9)
        end
    end
end

# AD-through-query on a narrow-carrier interpolant: the eval kernels must carry
# ForwardDiff partials (α is a Dual, the data is N0f8) — the convex blend / `_fielddiff`
# sites take the natural-promotion path, never flattening through Float. The derivative
# must be finite and match the float-built interpolant's.
@testitem "no-wrap: ForwardDiff through query (N0f8 carrier)" begin
    using FastInterpolations
    using FixedPointNumbers, ForwardDiff
    const FI = FastInterpolations
    x = collect(1.0:1.0:6.0)
    yN = N0f8.([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    yF = Float64.(yN)
    for ctor in (FI.linear_interp, FI.pchip_interp, FI.cubic_interp)
        itpN = ctor(x, yN)
        itpF = ctor(x, yF)
        for q in (1.5, 2.5, 3.5, 4.5)
            dN = ForwardDiff.derivative(itpN, q)
            dF = ForwardDiff.derivative(itpF, q)
            @test isfinite(dN)
            @test isapprox(dN, dF; atol = 1.0e-7)
        end
    end
end

# Batch eval dispatches through a separate (pre-allocated output) path; a narrow
# carrier must stay wrap-free, match the float build elementwise, and stay bounded.
@testitem "no-wrap: batch eval (N0f8 carrier)" begin
    using FastInterpolations
    using FixedPointNumbers
    const FI = FastInterpolations
    x = collect(1.0:1.0:5.0)
    qs = [1.5, 2.5, 3.5, 4.5, 2.25, 3.75]
    yN = N0f8.([0.9, 0.1, 0.8, 0.2, 0.7])
    itpN = FI.linear_interp(x, yN)
    itpF = FI.linear_interp(x, Float64.(yN))
    rN = Float64.(itpN(qs))
    @test all(isapprox.(rN, Float64.(itpF(qs)); atol = 1.0e-9))
    lo = Float64(minimum(yN)); hi = Float64(maximum(yN))
    @test all(lo - 1.0e-9 .<= rN .<= hi + 1.0e-9)   # bounded — no overshoot/wrap
end

# Full-domain integrate on an UN-promoted fixed-point carrier: N0f8 is outside
# `_PromotableValue`, so it stays narrow in `itp.y` and any endpoint sum computed
# outside the `_fieldsum`-protected kernels wraps mod 1. Pins BOTH grid arms of
# `_integrate_1d_fulldomain` (Range = telescoped closed form, Vector = cellwise engine).
@testitem "no-wrap: full-domain integrate (N0f8 carrier)" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    yN = N0f8.([0.8, 0.4, 0.4, 0.4, 0.8])           # y[1] + y[n] = 1.6 wraps in N0f8
    yF = Float64.(yN)
    for x in (0.0:0.25:1.0, collect(0.0:0.25:1.0))  # Range (fast path) + Vector (generic)
        itpN = FI.linear_interp(x, yN)
        itpF = FI.linear_interp(x, yF)
        @test isapprox(Float64(integrate(itpN)), integrate(itpF); atol = 1.0e-9)
    end

    gN = Gray{N0f8}.([0.8, 0.5, 0.8])
    gF = Float64.(gray.(gN))
    for x in (0.0:1.0:2.0, [0.0, 1.0, 2.0])
        itpG = FI.linear_interp(x, gN)
        itpF = FI.linear_interp(x, gF)
        @test isapprox(Float64(gray(integrate(itpG))), integrate(itpF); atol = 1.0e-9)
    end
end
