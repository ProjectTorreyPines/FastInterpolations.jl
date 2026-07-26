# ========================================
# Bounded `integrate` with bounds in a different (same-dimension) unit
# ========================================
# Forward evaluation already accepts a query whose unit differs from the grid's
# as long as the dimension matches (`itp(0.015u"m")` on a `cm` grid) — Unitful
# converts and the value is right. Bounded integration did not: the partial-cell
# kernel pins its two offsets to one type `Td`, and mixing a user bound with a
# grid node produced `(…, ::cm, ::m, ::cm)` → MethodError.
#
# The result must not depend on which unit the caller spells the bounds in.

@testitem "integrate: bounds may use any same-dimension unit" begin
    using Unitful

    x = (0.0:1.0:4.0)u"cm"
    y = [1.0, 2.0, 4.0, 8.0, 16.0]u"K"

    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        itp = f(x, y)
        ref = integrate(itp, 1.0u"cm", 3.0u"cm")

        @testset "$(nameof(f))" begin
            # same physical interval, spelled in metres
            @test integrate(itp, 0.01u"m", 0.03u"m") ≈ ref
            # one bound in each unit
            @test integrate(itp, 1.0u"cm", 0.03u"m") ≈ ref
            @test integrate(itp, 0.01u"m", 3.0u"cm") ≈ ref
            # a coarser unit on both sides
            @test integrate(itp, 1.0e-5u"km", 3.0e-5u"km") ≈ ref
            # partial cells (bounds strictly inside cells) exercise the offset kernel
            @test integrate(itp, 0.005u"m", 0.025u"m") ≈ integrate(itp, 0.5u"cm", 2.5u"cm")

            # BOTH bounds inside ONE cell takes the `i0 == i1` branch, which the
            # multi-cell `promote` never touches (there both bounds are caller
            # bounds — only the end cells pair a bound with a grid node).
            @test integrate(itp, 5.0u"mm", 8.0u"mm") ≈ integrate(itp, 0.5u"cm", 0.8u"cm")
            @test integrate(itp, 0.005u"m", 0.008u"m") ≈ integrate(itp, 0.5u"cm", 0.8u"cm")
            # reversed single-cell bounds keep the sign
            @test integrate(itp, 8.0u"mm", 5.0u"mm") ≈ -integrate(itp, 0.5u"cm", 0.8u"cm")
        end
    end
end

@testitem "integrate: same-unit bounds and Real grids are unchanged" begin
    using Unitful

    # Real canary — must stay bit-identical.
    xr = collect(0.0:1.0:4.0)
    yr = [1.0, 2.0, 4.0, 8.0, 16.0]
    @test integrate(linear_interp(xr, yr), 1.0, 3.0) === 9.0   # (2+4)/2 + (4+8)/2
    @test integrate(constant_interp(xr, yr), 0.5, 2.5) isa Float64

    # Unit grid with matching bound units keeps the grid's unit in the result.
    x = (0.0:1.0:4.0)u"cm"
    y = [1.0, 2.0, 4.0, 8.0, 16.0]u"K"
    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        itp = f(x, y)
        @test unit(integrate(itp)) === u"K" * u"cm"
        @test unit(integrate(itp, 1.0u"cm", 3.0u"cm")) === u"K" * u"cm"
    end
end
