# Direct unit coverage for `WrapExtrap` constructor layering and the unified
# `_resolve_extrap` family (primitive per-axis + 1D bundled + ND). All layers
# share the same function name — `extrap` is always the first argument.

using Test
using FastInterpolations
using FastInterpolations: WrapExtrap, NoBC, PeriodicBC, NoExtrap, ClampExtrap,
    _resolve_extrap

@testset "WrapExtrap + _resolve_extrap" begin

    @testset "WrapExtrap constructors" begin
        # Zero-arg legacy singleton — build-time placeholder only.
        e = WrapExtrap()
        @test e isa WrapExtrap{Nothing}
        @test e._x_min === nothing
        @test e._x_max === nothing

        # Primary factory: grid span.
        for x in (range(0.0, 2π, length = 5), collect(range(0.0, 2π, length = 5)))
            e = WrapExtrap(x)
            @test e isa WrapExtrap
            @test e._x_min ≈ 0.0
            @test e._x_max ≈ 2π
        end

        # Int grid preserved (no eager Float promotion).
        e_int = WrapExtrap([0, 1, 2, 3])
        @test e_int isa WrapExtrap{Int}
        @test (e_int._x_min, e_int._x_max) == (0, 3)

        # BC-aware delegate: non-exclusive BCs route to WrapExtrap(x).
        x = collect(range(0.0, 2π, length = 5))
        @test WrapExtrap(x, NoBC()) == WrapExtrap(x)
        @test WrapExtrap(x, PeriodicBC(endpoint = :inclusive)) == WrapExtrap(x)

        # :exclusive with explicit period: [first(x), first(x) + period).
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        e = WrapExtrap([0.0, 1.0, 2.0, 3.0], bc_exc)
        @test e._x_min === 0.0
        @test e._x_max ≈ 4.0

        # :exclusive with period that places virtual endpoint inside the grid.
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError WrapExtrap([0.0, 1.0, 2.0, 3.0], bc_bad)

        # :exclusive Nothing + AbstractRange → auto-infer period = step × length.
        bc_auto = PeriodicBC(endpoint = :exclusive)
        x_r = range(0.0, step = 1.0, length = 4)
        e = WrapExtrap(x_r, bc_auto)
        @test e._x_min ≈ 0.0
        @test e._x_max ≈ 4.0

        # :exclusive Nothing + Vector grid → cannot infer, must error.
        @test_throws ArgumentError WrapExtrap([0.0, 1.0, 2.0, 3.0], bc_auto)
    end

    @testset "_resolve_extrap — primitive 2-arg (extrap, x)" begin
        x = 0.0:0.5:2.0

        # WrapExtrap{Nothing} → upgrade via grid span.
        e = _resolve_extrap(WrapExtrap(), x)
        @test e isa WrapExtrap
        @test !(e isa WrapExtrap{Nothing})
        @test e._x_min ≈ 0.0 && e._x_max ≈ 2.0

        # Non-Wrap extraps passthrough.
        @test _resolve_extrap(NoExtrap(), x) === NoExtrap()
        @test _resolve_extrap(ClampExtrap(), x) === ClampExtrap()
    end

    @testset "_resolve_extrap — primitive 3-arg (extrap, bc, x)" begin
        x = 0.0:0.5:2.0

        # Rule 1: non-periodic BC + non-Wrap extrap → passthrough (===).
        @test _resolve_extrap(NoExtrap(), NoBC(), x) === NoExtrap()
        @test _resolve_extrap(ClampExtrap(), NoBC(), x) === ClampExtrap()

        # Rule 2: non-periodic BC + WrapExtrap{Nothing} → upgrade via WrapExtrap(x).
        e = _resolve_extrap(WrapExtrap(), NoBC(), x)
        @test e isa WrapExtrap
        @test !(e isa WrapExtrap{Nothing})
        @test e._x_min ≈ 0.0
        @test e._x_max ≈ 2.0

        # Rule 3: PeriodicBC forces WrapExtrap via (x, bc) constructor, overriding
        # any user-supplied extrap (including NoExtrap).
        e = _resolve_extrap(NoExtrap(), PeriodicBC(endpoint = :inclusive), x)
        @test e isa WrapExtrap
        @test e._x_min ≈ 0.0 && e._x_max ≈ 2.0

        bc_exc = PeriodicBC(endpoint = :exclusive, period = 2.5)
        e = _resolve_extrap(NoExtrap(), bc_exc, x)
        @test e isa WrapExtrap
        @test e._x_min ≈ 0.0 && e._x_max ≈ 2.5
    end

    @testset "_resolve_extrap — 1D bundled (extrap, bc, x, y)" begin
        x = collect(range(0.0, 1.0, length = 5))
        y_good = [0.0, 1.0, 2.0, 1.0, 0.0]                  # y[1] == y[end] = 0
        y_bad = [0.0, 1.0, 2.0, 3.0, 4.0]                   # y[1] != y[end]

        # NoBC: y not validated, extrap returned unchanged.
        @test _resolve_extrap(NoExtrap(), NoBC(), x, y_bad) === NoExtrap()
        @test _resolve_extrap(ClampExtrap(), NoBC(), x, y_bad) === ClampExtrap()

        # NoBC + legacy singleton: upgraded to typed WrapExtrap even without periodic BC.
        e = _resolve_extrap(WrapExtrap(), NoBC(), x, y_bad)
        @test e isa WrapExtrap
        @test !(e isa WrapExtrap{Nothing})
        @test e._x_min ≈ 0.0 && e._x_max ≈ 1.0

        # :inclusive valid: typed WrapExtrap returned.
        e = _resolve_extrap(NoExtrap(), PeriodicBC(), x, y_good)
        @test e isa WrapExtrap
        @test e._x_min ≈ 0.0 && e._x_max ≈ 1.0

        # :inclusive invalid: bundled gateway fires the validator.
        @test_throws ArgumentError _resolve_extrap(
            NoExtrap(), PeriodicBC(), x, y_bad
        )

        # :inclusive + check=false: validation skipped (escape hatch).
        bc_skip = PeriodicBC(endpoint = :inclusive, check = false)
        @test _resolve_extrap(NoExtrap(), bc_skip, x, y_bad) isa WrapExtrap

        # :exclusive: y carries no endpoint contract, never validated.
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 1.25)
        @test _resolve_extrap(NoExtrap(), bc_exc, x, y_bad) isa WrapExtrap
    end

    @testset "_resolve_extrap — ND bundled (extraps, bcs, grids, data, Val(N))" begin
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))
        good = [sin(xi) * cos(yj) for xi in x, yj in y]     # endpoints match on both axes
        bad_axis1 = copy(good)
        bad_axis1[end, :] .+= 1.0                            # break axis 1
        bad_axis2 = copy(good)
        bad_axis2[:, end] .+= 1.0                            # break axis 2

        extraps = (NoExtrap(), NoExtrap())

        # All NoBC: per-axis passthrough — slice validation is a no-op.
        out = _resolve_extrap(
            extraps, (NoBC(), NoBC()), (x, y), bad_axis1, Val(2)
        )
        @test out === extraps

        # :inclusive on axis 1 with valid data.
        bcs_p1 = (PeriodicBC(), NoBC())
        out = _resolve_extrap(extraps, bcs_p1, (x, y), good, Val(2))
        @test out[1] isa WrapExtrap
        @test out[2] === NoExtrap()

        # :inclusive on axis 1 with mismatch on axis 1: gateway fires validator.
        @test_throws ArgumentError _resolve_extrap(
            extraps, bcs_p1, (x, y), bad_axis1, Val(2)
        )

        # :inclusive on axis 2 with mismatch on axis 2: gateway catches per-axis.
        bcs_p2 = (NoBC(), PeriodicBC())
        @test_throws ArgumentError _resolve_extrap(
            extraps, bcs_p2, (x, y), bad_axis2, Val(2)
        )

        # :inclusive + check=false on axis 1: validation skipped.
        bcs_skip = (PeriodicBC(endpoint = :inclusive, check = false), NoBC())
        out = _resolve_extrap(extraps, bcs_skip, (x, y), bad_axis1, Val(2))
        @test out[1] isa WrapExtrap

        # ND with legacy singleton on one axis: gets upgraded to typed form.
        out = _resolve_extrap(
            (WrapExtrap(), NoExtrap()), (NoBC(), NoBC()), (x, y), good, Val(2)
        )
        @test out[1] isa WrapExtrap
        @test !(out[1] isa WrapExtrap{Nothing})
        @test out[2] === NoExtrap()
    end

    @testset "_resolve_extrap — ND 5-arg (extrap, bcs, grids, Val, Tv) one-step" begin
        # 1-liner collapse: expand + promote + per-axis materialize.
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))

        # Periodic on axis 1 — BC-aware materialize (bc.period used for exclusive;
        # grid span for inclusive).
        bcs = (PeriodicBC(endpoint = :inclusive), NoBC())
        out = _resolve_extrap(NoExtrap(), bcs, (x, y), Val(2), Float64)
        @test out[1] isa WrapExtrap
        @test out[2] === NoExtrap()

        # `nothing` bcs: expand + per-axis 2-arg materialize (grid-span only).
        out = _resolve_extrap(WrapExtrap(), nothing, (x, y), Val(2), Float64)
        @test out[1] isa WrapExtrap && !(out[1] isa WrapExtrap{Nothing})
        @test out[2] isa WrapExtrap && !(out[2] isa WrapExtrap{Nothing})
    end
end
