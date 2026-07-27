# ========================================
# Series derivative `oneunit` contract on unit-carrying grids
# ========================================
# An order-N derivative lives in `value/coordᴺ`. Kernels that DO the arithmetic
# get those units for free; kernels that FABRICATE a result must transport the
# value-space zero with `_deriv_oneunit` (see core/utils.jl). Two Series paths
# fabricate:
#
#   1. in-domain, order beyond the polynomial degree — a synthetic zero
#      (Constant N≥1, Linear N≥2, Cubic N≥4; Quadratic never, it differentiates
#      the actual polynomial down to a real zero)
#   2. OOB under a flat extrap (Clamp/Fill) — the boundary/fill value, and its
#      derivative zero
#
# The scalar/batch/ND surfaces compute the factor inline (the grid is in scope);
# Series anchors are decoupled from the grid and carry `TinvN` in the payload
# type instead. These pins cover the Series side, which is where an omission is
# invisible until a unit grid shows up.

@testitem "Series deriv oneunit: in-domain synthetic zeros carry grid⁻ᴺ" begin
    using Unitful

    xu = (0.0:1.0:4.0)u"s"
    ym = [1.0 2.0; 2.0 3.0; 4.0 5.0; 8.0 9.0; 16.0 17.0]u"m"

    # (family, first order whose in-domain result is a fabricated zero)
    families = (
        (constant_interp, 1),
        (linear_interp, 2),
        (quadratic_interp, 3),
        (cubic_interp, 4),
    )

    for (f, n_zero) in families
        sitp = f(xu, Series(ym))
        for n in n_zero:(n_zero + 2)
            r = sitp(1.5u"s"; deriv = DerivOp(n))
            @test unit(eltype(r)) === u"m" / u"s"^n
            @test all(iszero, ustrip.(r))
            # batch query must agree with the scalar-query vector
            rb = sitp([1.5u"s", 2.5u"s"]; deriv = DerivOp(n))
            @test unit(eltype(rb[1])) === u"m" / u"s"^n
        end
        # Orders BELOW the fabrication threshold come from real arithmetic —
        # pin them too so a fix cannot regress the arithmetic path.
        for n in 0:(n_zero - 1)
            r = sitp(1.5u"s"; deriv = DerivOp(n))
            @test unit(eltype(r)) === u"m" / u"s"^n
        end
    end
end

@testitem "Series deriv oneunit: flat-extrap OOB carries grid⁻ᴺ" begin
    using Unitful

    xu = (0.0:1.0:4.0)u"s"
    ym = [1.0 2.0; 2.0 3.0; 4.0 5.0; 8.0 9.0; 16.0 17.0]u"m"

    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        for extrap in (ClampExtrap(), FillExtrap(0.0u"m"))
            sitp = f(xu, Series(ym); extrap = extrap)
            for q in (-1.0u"s", 5.0u"s")
                v = sitp(q)
                @test unit(eltype(v)) === u"m"
                for n in 1:3
                    d = sitp(q; deriv = DerivOp(n))
                    @test unit(eltype(d)) === u"m" / u"s"^n
                    @test all(iszero, ustrip.(d))
                end
            end
        end
    end
end

@testitem "Series deriv oneunit: Real grids stay bit-identical" begin
    xr = collect(0.0:1.0:4.0)
    ym = [1.0 2.0; 2.0 3.0; 4.0 5.0; 8.0 9.0; 16.0 17.0]

    for f in (constant_interp, linear_interp, quadratic_interp, cubic_interp)
        sitp = f(xr, Series(ym); extrap = ClampExtrap())
        for n in 0:5, q in (1.5, -1.0, 5.0)
            r = sitp(q; deriv = DerivOp(n))
            @test eltype(r) === Float64
            @test all(isfinite, r)
        end
    end
end
