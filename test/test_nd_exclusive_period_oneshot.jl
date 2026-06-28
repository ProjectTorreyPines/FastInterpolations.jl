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
        # one-shot must (a) not throw and (b) match persistent
        @test FI.cubic_interp((x, y), data, q; bc = bc) ≈ itp(q...)
        # and equal the same one-shot built from a float grid (no Int wrap/inexact)
        @test FI.cubic_interp((x, y), data, q; bc = bc) ≈
            FI.cubic_interp((xf, yf), data, q; bc = bc)
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
        @test FI.hermite_interp((x, y), data, p, q; bc = bc) ≈ itp(q...)
        @test FI.hermite_interp((x, y), data, p, q; bc = bc) ≈
            FI.hermite_interp((xf, yf), data, p, q; bc = bc)
    end
end
