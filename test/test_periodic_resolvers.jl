# Direct unit coverage for the unified `_resolve_extrap` family. After the
# `WrapExtrap` → tag-struct refactor, the helper is purely BC-aware passthrough:
# PeriodicBC forces `WrapExtrap()`, otherwise the user extrap flows through.
# Wrap-domain extraction happens at query time via `_wrap_to_domain(xq, x)`,
# which reads `(first(x), last(x))` directly from the (possibly-wrapped) axis.

@testitem "WrapExtrap + _resolve_extrap" begin
    using FastInterpolations: WrapExtrap, NoBC, PeriodicBC, NoExtrap, ClampExtrap,
        _resolve_extrap

    @testset "WrapExtrap is a tag struct" begin
        e = WrapExtrap()
        @test e isa WrapExtrap
        # Backward-compat shim: `WrapExtrap(x)` discards the axis arg.
        @test WrapExtrap([0.0, 1.0, 2.0]) === WrapExtrap()
        @test WrapExtrap(0:3) === WrapExtrap()
    end

    @testset "_resolve_extrap — primitive 2-arg (extrap, x)" begin
        x = 0.0:0.5:2.0
        # WrapExtrap passes through (no materialization).
        @test _resolve_extrap(WrapExtrap(), x) isa WrapExtrap
        @test _resolve_extrap(NoExtrap(), x) === NoExtrap()
        @test _resolve_extrap(ClampExtrap(), x) === ClampExtrap()
    end

    @testset "_resolve_extrap — primitive 3-arg (extrap, bc, x)" begin
        x = 0.0:0.5:2.0

        # Non-periodic BC: passthrough (===).
        @test _resolve_extrap(NoExtrap(), NoBC(), x) === NoExtrap()
        @test _resolve_extrap(ClampExtrap(), NoBC(), x) === ClampExtrap()
        @test _resolve_extrap(WrapExtrap(), NoBC(), x) isa WrapExtrap

        # PeriodicBC forces WrapExtrap (overrides user extrap).
        @test _resolve_extrap(NoExtrap(), PeriodicBC(endpoint = :inclusive), x) isa WrapExtrap
        @test _resolve_extrap(ClampExtrap(), PeriodicBC(endpoint = :inclusive), x) isa WrapExtrap
        @test _resolve_extrap(NoExtrap(), PeriodicBC(endpoint = :exclusive, period = 2.5), x) isa WrapExtrap
    end

    @testset "_resolve_extrap — 1D bundled (extrap, bc, x, y) :inclusive validation" begin
        x = collect(range(0.0, 1.0, length = 5))
        y_good = [0.0, 1.0, 2.0, 1.0, 0.0]                  # y[1] == y[end] = 0
        y_bad = [0.0, 1.0, 2.0, 3.0, 4.0]                   # y[1] != y[end]

        # NoBC: y not validated, extrap returned unchanged.
        @test _resolve_extrap(NoExtrap(), NoBC(), x, y_bad) === NoExtrap()
        @test _resolve_extrap(ClampExtrap(), NoBC(), x, y_bad) === ClampExtrap()

        # :inclusive valid: WrapExtrap returned.
        @test _resolve_extrap(NoExtrap(), PeriodicBC(), x, y_good) isa WrapExtrap

        # :inclusive invalid: bundled gateway fires the endpoint validator.
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
end
