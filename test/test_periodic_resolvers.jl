# Direct unit coverage for periodic resolver gateways and the `WrapExtrap`
# parametric constructor — paths that previously had only end-to-end coverage.

using Test
using FastInterpolations
using FastInterpolations: WrapExtrap, NoBC, PeriodicBC, NoExtrap, ClampExtrap,
    _resolve_periodic_extrap, _resolve_periodic_extrap_1d, _resolve_periodic_extraps_nd

@testset "Periodic resolvers and gateways" begin

    @testset "WrapExtrap parametric constructor" begin
        # Backward-compat singleton
        e = WrapExtrap()
        @test e isa WrapExtrap{Nothing}
        @test e._x_min === nothing
        @test e._x_max === nothing

        # Advisory @warn fires on direct construction (TestLogger captures it).
        # All subsequent WrapExtrap(x,y) calls in this session are silenced by
        # the constructor's `maxlog=1`, so we don't pollute the test output.
        e = @test_logs (:warn, r"WrapExtrap") match_mode = :any WrapExtrap(0.0, 2π)
        @test e isa WrapExtrap{Float64}
        @test e._x_min === 0.0
        @test e._x_max ≈ 2π

        # Explicit Int — preserves Int (duck-typing through to 3-arg _wrap_to_domain)
        e_int = WrapExtrap(0, 4)
        @test e_int isa WrapExtrap{Int}
        @test (e_int._x_min, e_int._x_max) == (0, 4)

        # Mixed numeric promotion: Int + Float → Float
        e_mix = WrapExtrap(0, 4.0)
        @test e_mix isa WrapExtrap{Float64}
        @test (e_mix._x_min, e_mix._x_max) == (0.0, 4.0)

        # Guards: x_max ≤ x_min must throw before WrapExtrap is constructed
        @test_throws ArgumentError WrapExtrap(2.0, 1.0)
        @test_throws ArgumentError WrapExtrap(1.0, 1.0)
    end

    @testset "_resolve_periodic_extrap — per-method truth table" begin
        # NoBC: passthrough — must return the user's extrap unchanged (===)
        x = 0.0:0.5:2.0
        @test _resolve_periodic_extrap(NoBC(), NoExtrap(), x) === NoExtrap()
        @test _resolve_periodic_extrap(NoBC(), ClampExtrap(), x) === ClampExtrap()

        # :inclusive — domain comes from grid span (first/last)
        bc_inc = PeriodicBC(endpoint = :inclusive)
        for x in (range(0.0, 2π, length = 5), collect(range(0.0, 2π, length = 5)))
            e = _resolve_periodic_extrap(bc_inc, NoExtrap(), x)
            @test e isa WrapExtrap
            @test e._x_min ≈ 0.0
            @test e._x_max ≈ 2π
        end

        # :inclusive on Int grid — Int preserved (no eager Float promotion)
        e_int = _resolve_periodic_extrap(bc_inc, NoExtrap(), [0, 1, 2, 3])
        @test e_int isa WrapExtrap{Int}
        @test (e_int._x_min, e_int._x_max) == (0, 3)

        # :exclusive with explicit period
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        e = _resolve_periodic_extrap(bc_exc, NoExtrap(), [0.0, 1.0, 2.0, 3.0])
        @test e._x_min === 0.0
        @test e._x_max ≈ 4.0

        # :exclusive with period that places virtual endpoint inside the grid
        bc_bad = PeriodicBC(endpoint = :exclusive, period = 2.5)
        @test_throws ArgumentError _resolve_periodic_extrap(
            bc_bad, NoExtrap(), [0.0, 1.0, 2.0, 3.0]
        )

        # :exclusive Nothing + AbstractRange → auto-infer period = step×length
        bc_auto = PeriodicBC(endpoint = :exclusive)        # period defaults to Nothing
        x_r = range(0.0, step = 1.0, length = 4)
        e = _resolve_periodic_extrap(bc_auto, NoExtrap(), x_r)
        @test e._x_min ≈ 0.0
        @test e._x_max ≈ 4.0                                # 1.0 × 4

        # :exclusive Nothing + Vector grid → cannot infer period, must error
        @test_throws ArgumentError _resolve_periodic_extrap(
            bc_auto, NoExtrap(), [0.0, 1.0, 2.0, 3.0]
        )
    end

    @testset "_resolve_periodic_extrap_1d gateway" begin
        x = collect(range(0.0, 1.0, length = 5))
        y_good = [0.0, 1.0, 2.0, 1.0, 0.0]                  # y[1] == y[end] = 0
        y_bad = [0.0, 1.0, 2.0, 3.0, 4.0]                   # y[1] != y[end]

        # NoBC: y not validated, extrap returned unchanged
        @test _resolve_periodic_extrap_1d(NoBC(), NoExtrap(), x, y_bad) === NoExtrap()
        @test _resolve_periodic_extrap_1d(NoBC(), ClampExtrap(), x, y_bad) === ClampExtrap()

        # :inclusive valid: typed WrapExtrap
        e = _resolve_periodic_extrap_1d(PeriodicBC(), NoExtrap(), x, y_good)
        @test e isa WrapExtrap
        @test e._x_min ≈ 0.0 && e._x_max ≈ 1.0

        # :inclusive invalid: gateway fires the validator
        @test_throws ArgumentError _resolve_periodic_extrap_1d(
            PeriodicBC(), NoExtrap(), x, y_bad
        )

        # :inclusive + check=false: validation skipped (escape hatch)
        bc_skip = PeriodicBC(endpoint = :inclusive, check = false)
        @test _resolve_periodic_extrap_1d(bc_skip, NoExtrap(), x, y_bad) isa WrapExtrap

        # :exclusive: y carries no contract, never validated
        bc_exc = PeriodicBC(endpoint = :exclusive, period = 1.25)
        @test _resolve_periodic_extrap_1d(bc_exc, NoExtrap(), x, y_bad) isa WrapExtrap
    end

    @testset "_resolve_periodic_extraps_nd gateway" begin
        x = collect(range(0.0, 2π, length = 5))
        y = collect(range(0.0, 2π, length = 5))
        good = [sin(xi) * cos(yj) for xi in x, yj in y]     # endpoints match on both axes
        bad_axis1 = copy(good)
        bad_axis1[end, :] .+= 1.0                            # break axis 1
        bad_axis2 = copy(good)
        bad_axis2[:, end] .+= 1.0                            # break axis 2

        extraps = (NoExtrap(), NoExtrap())

        # All NoBC: per-axis passthrough — slice validation should be a no-op
        out = _resolve_periodic_extraps_nd(
            (NoBC(), NoBC()), extraps, (x, y), bad_axis1, Val(2)
        )
        @test out === extraps

        # :inclusive on axis 1 with valid data
        bcs_p1 = (PeriodicBC(), NoBC())
        out = _resolve_periodic_extraps_nd(bcs_p1, extraps, (x, y), good, Val(2))
        @test out[1] isa WrapExtrap
        @test out[2] === NoExtrap()

        # :inclusive on axis 1 with mismatch on axis 1: gateway fires validator
        @test_throws ArgumentError _resolve_periodic_extraps_nd(
            bcs_p1, extraps, (x, y), bad_axis1, Val(2)
        )

        # :inclusive on axis 2 with mismatch on axis 2: gateway catches per-axis
        bcs_p2 = (NoBC(), PeriodicBC())
        @test_throws ArgumentError _resolve_periodic_extraps_nd(
            bcs_p2, extraps, (x, y), bad_axis2, Val(2)
        )

        # :inclusive + check=false on axis 1: validation skipped
        bcs_skip = (PeriodicBC(endpoint = :inclusive, check = false), NoBC())
        out = _resolve_periodic_extraps_nd(bcs_skip, extraps, (x, y), bad_axis1, Val(2))
        @test out[1] isa WrapExtrap
    end
end
