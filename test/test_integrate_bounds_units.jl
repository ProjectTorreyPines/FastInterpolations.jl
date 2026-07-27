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

# ========================================
# Bounds that do not share a type
# ========================================
# The partial-cell kernels take BOTH in-cell offsets as one type `Td`. The
# multi-cell path promotes each end cell's (bound, node) pair, so it is safe —
# but the single-cell `i0 == i1` branch hands `lo`/`hi` straight through, and
# after subtracting the same node they keep whatever types the caller gave.
#
# This is not a units problem: `integrate(itp, 1, 1.5)` on an Int grid produces
# an Int offset beside a Float one. It only surfaces where the grid is NOT
# promoted past the bound types — Constant keeps Int grids as Int (it selects,
# it does not blend), and an Int/Float32 grid sits below a Float64 bound.
@testitem "integrate: single-cell bounds need not share a type" begin
    y = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]

    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        @testset "$(nameof(f))" begin
            itp = f(0:1:10, y)
            ref = integrate(itp, 1.0, 1.5)
            @test integrate(itp, 1, 1.5) ≈ ref        # Int lower, Float upper
            @test integrate(itp, 1.0, 1.5) ≈ ref
            @test integrate(itp, 1.5, 2) ≈ integrate(itp, 1.5, 2.0)
            # reversed keeps the sign
            @test integrate(itp, 1.5, 1) ≈ -ref

            # Int grid + Float32 values: the grid resolves to Float32, below a
            # Float64 bound.
            itp32 = f(0:1:10, Float32.(y))
            @test integrate(itp32, 1, 1.5) ≈ integrate(itp32, 1.0f0, 1.5f0) rtol = 1.0e-6

            # Float32 grid with one Float64 bound.
            itpg32 = f(Float32.(collect(0.0:1.0:10.0)), Float32.(y))
            @test integrate(itpg32, 1.0, 1.5f0) ≈ integrate(itpg32, 1.0f0, 1.5f0) rtol = 1.0e-6
        end
    end

    # Multi-cell (already promoted) must keep agreeing.
    itp = constant_interp(0:1:10, y)
    @test integrate(itp, 1, 3.5) ≈ integrate(itp, 1.0, 3.5)
end

@testitem "integrate: single-cell bounds in different same-dimension units" begin
    using Unitful

    x = (0.0:1.0:4.0)u"cm"
    y = [1.0, 2.0, 4.0, 8.0, 16.0]u"K"
    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        itp = f(x, y)
        ref = integrate(itp, 0.5u"cm", 0.8u"cm")
        @test integrate(itp, 5.0u"mm", 0.8u"cm") ≈ ref     # one bound in each unit
        @test integrate(itp, 0.5u"cm", 8.0u"mm") ≈ ref
        @test integrate(itp, 0.005u"m", 8.0u"mm") ≈ ref
    end
end
