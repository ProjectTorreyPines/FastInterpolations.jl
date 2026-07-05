# Regression guard: ND scalar one-shot on Int *vector* grids with an explicit
# non-integer `:exclusive` period threw `InexactError: Int64(2.5)`. The raw-grid
# one-shot path (#166) converted the period through the Int grid eltype instead
# of the float coefficient type, so the virtual seam point (`x[1] + period`) was
# forced into `Int`. The persistent path always worked (it floats grids first);
# one-shot must agree with it.
#
#   cubic   : `_ExclusivePeriodicAxis` ctor (axis_types.jl) forced `Tg(period)`.
#   hermite : pooled nodal-deriv grid extension (hermite_nd_build.jl) forced it.

@testitem "ND :exclusive one-shot on Int vector grid — cubic" begin
    using FastInterpolations
    const FI = FastInterpolations

    x = collect(0:2)
    y = collect(0:2)                         # Int Vector grids
    data = Float64[1 2 3; 4 5 6; 7 8 9]
    bc = FI.PeriodicBC(endpoint = :exclusive, period = 2.5)

    itp = FI.cubic_interp((x, y), data; bc = bc)   # persistent reference (always worked)
    xf = Float64.(x)
    yf = Float64.(y)

    for q in ((0.5, 0.5), (1.5, 0.5), (2.25, 1.75), (0.0, 2.4))
        # one-shot must (a) not throw and (b) match persistent. `atol` (not bare `≈`):
        # the value is ~0 for some q, where one-shot/persistent take slightly different
        # FP-reduction orders on some LTS arches (≤1 ULP; `≈` vs 0.0 has no rtol).
        @test isapprox(FI.cubic_interp((x, y), data, q; bc = bc), itp(q...); atol = 1.0e-12)
        # and equal the same one-shot built from a float grid (no Int wrap/inexact)
        @test isapprox(
            FI.cubic_interp((x, y), data, q; bc = bc),
            FI.cubic_interp((xf, yf), data, q; bc = bc); atol = 1.0e-12,
        )
    end
end

@testitem "ND :exclusive one-shot on Int vector grid — hermite" begin
    using FastInterpolations
    const FI = FastInterpolations

    x = collect(0:2)
    y = collect(0:2)
    data = Float64[1 2 3; 4 5 6; 7 8 9]
    dfdx = zeros(3, 3)
    dfdy = zeros(3, 3)
    d2 = zeros(3, 3)
    p = FI.HermitePartials((1, 0) => dfdx, (0, 1) => dfdy, (1, 1) => d2)
    bc = FI.PeriodicBC(endpoint = :exclusive, period = 2.5)

    itp = FI.hermite_interp((x, y), data, p; bc = bc)   # persistent reference
    xf = Float64.(x)
    yf = Float64.(y)

    for q in ((0.5, 0.5), (1.5, 0.5), (2.25, 1.75), (0.0, 2.4))
        @test isapprox(FI.hermite_interp((x, y), data, p, q; bc = bc), itp(q...); atol = 1.0e-12)
        @test isapprox(
            FI.hermite_interp((x, y), data, p, q; bc = bc),
            FI.hermite_interp((xf, yf), data, p, q; bc = bc); atol = 1.0e-12,
        )
    end
end

# Unit pin at the fix's source: the `_ExclusivePeriodicAxis` outer ctor must widen an
# Int Vector grid against a non-integer period (zero-copy when the grid is already
# float). Targets `axis_types.jl` directly — a small guard for the regression that the
# ND integration tests cover with ~50 lines of setup.
@testitem "_ExclusivePeriodicAxis widens Int vector grid for float period" begin
    using FastInterpolations
    const FI = FastInterpolations
    ax = FI._ExclusivePeriodicAxis(collect(0:2), 2.5)   # Int grid + non-integer period
    @test eltype(ax) === Float64
    @test ax.inner isa Vector{Float64}
    @test ax[end] ≈ 2.5                                  # virtual seam = inner[1] + period
    @test length(ax) == 4                                # n + 1
    # Float grid + float period stays zero-copy (same inner object, no widen).
    g = [0.0, 1.0, 2.0]
    @test FI._ExclusivePeriodicAxis(g, 2.5).inner === g
end

# The `_ExclusivePeriodicAxis` ctor widening is shared by every ND method on the
# raw-grid one-shot path. The fix landed for cubic/hermite (above) plus linear/constant;
# pin those two here too (quadratic ND has no `:exclusive` support).
@testitem "ND :exclusive one-shot on Int vector grid — linear & constant" begin
    using FastInterpolations
    const FI = FastInterpolations

    x = collect(0:2)
    y = collect(0:2)                         # Int Vector grids
    data = Float64[1 2 3; 4 5 6; 7 8 9]
    bc = FI.PeriodicBC(endpoint = :exclusive, period = 2.5)
    xf = Float64.(x)
    yf = Float64.(y)

    for ctor in (FI.linear_interp, FI.constant_interp)
        itp = ctor((x, y), data; bc = bc)    # persistent reference (always worked)
        for q in ((0.5, 0.5), (1.5, 0.5), (2.25, 1.75), (0.0, 2.4))
            # one-shot == persistent; no Int wrap/inexact (≤1 ULP on some LTS arches)
            @test isapprox(ctor((x, y), data, q; bc = bc), itp(q...); atol = 1.0e-12)
            @test isapprox(
                ctor((x, y), data, q; bc = bc),
                ctor((xf, yf), data, q; bc = bc); atol = 1.0e-12,
            )
        end
    end
end

# RED PIN (#1): narrow-float (Float32) exclusive-periodic on an Int Vector grid. The arithmetic
# kernels value-match `Tg` to the DATA width (Int grid + Float32 data → Float32), but the one-shot
# `_resolve_axis` exclusive-Vector arm resolves the period against the RAW Int grid → Float64 period
# → Float64 axis → the Float32 witness `::Tr` throws. Persistent already returns Float32 (verified);
# the one-shot must match. (Range grids already work via the convert-first arm; only Vector regresses.)
@testitem "ND :exclusive one-shot narrow-float (Float32) Int-vec grid — linear matches persistent" begin
    using FastInterpolations
    const FI = FastInterpolations

    x = collect(0:2)
    y = collect(0:2)                                     # Int Vector grids
    data = Float32[1 2 3; 4 5 6; 7 8 9]                  # narrow float → value-matched Tg = Float32
    bc = FI.PeriodicBC(endpoint = :exclusive, period = 2.5)
    xf = Float32.(x)
    yf = Float32.(y)

    itp = FI.linear_interp((x, y), data; bc = bc)        # persistent reference → Float32 today
    @test itp(0.5f0, 0.5f0) isa Float32

    for q in ((0.5f0, 0.5f0), (1.5f0, 0.5f0), (2.25f0, 1.75f0), (0.0f0, 2.4f0))
        r = FI.linear_interp((x, y), data, q; bc = bc)   # RED: currently TypeError (Float64 axis vs Float32 witness)
        @test r isa Float32                              # value-matched output type
        @test isapprox(r, itp(q...); atol = 1.0f-5)      # one-shot == persistent
        @test isapprox(r, FI.linear_interp((xf, yf), data, q; bc = bc); atol = 1.0f-5)  # == Float32 grid
    end
end
