# ========================================
# Adjoint operators on unit-carrying grids: explicit guard
# ========================================
# The adjoint families build `Wᵀ` from grid geometry and are not part of the
# duck-typed-grid work (see the exclusion note in test_unitful_api_smoke.jl).
# Until now a `Unitful` grid produced an internal `MethodError` naming a private
# helper (`_linear_anchor_query_impl`, `_compute_quadratic_adjoint…`) or a bare
# `DimensionError` — nothing that says whether it is a bug or a boundary.
#
# `Dual` grids DO work and must keep working: `ForwardDiff.Dual <: Real`, so the
# guard tests `<: Real` rather than "is it a plain float".

@testitem "Adjoint: unit-carrying grids are refused with an explanation" begin
    using Unitful

    xu = (0.0:1.0:4.0)u"s"
    qu = [1.5u"s", 2.5u"s"]

    # The slope-from-data families take `(x, y, x_query)`; the rest `(x, x_query)`.
    yu = [1.0, 2.0, 4.0, 8.0, 16.0]u"m"
    families = (
        ("linear_adjoint", () -> linear_adjoint(xu, qu)),
        ("cubic_adjoint", () -> cubic_adjoint(xu, qu)),
        ("constant_adjoint", () -> constant_adjoint(xu, qu)),
        ("quadratic_adjoint", () -> quadratic_adjoint(xu, qu)),
        ("cardinal_adjoint", () -> cardinal_adjoint(xu, qu)),
        ("hermite_adjoint", () -> hermite_adjoint(xu, qu)),
        ("pchip_adjoint", () -> pchip_adjoint(xu, yu, qu)),
        ("akima_adjoint", () -> akima_adjoint(xu, yu, qu)),
    )

    for (name, call) in families
        e = try
            call()
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("adjoint", msg)
        @test occursin("not supported", msg)
        @test occursin("ustrip", msg)          # names the workaround
    end

    # Scalar-query entries route through the same guard.
    @test_throws ArgumentError linear_adjoint(xu, 1.5u"s")
    @test_throws ArgumentError cubic_adjoint(xu, 1.5u"s")
end

@testitem "Adjoint: Real and Dual grids keep working (guard must not over-fire)" begin
    using ForwardDiff: Dual

    xr = collect(0.0:1.0:4.0)
    qr = [1.5, 2.5]
    xd = Dual.(xr, 1.0)
    qd = Dual.(qr, 1.0)

    for f in (linear_adjoint, cubic_adjoint, constant_adjoint, quadratic_adjoint)
        @test size(f(xr, qr)) == (5, 2)
        @test size(f(xd, qd)) == (5, 2)   # Dual <: Real — must pass the guard
    end
end
