# ========================================
# GriddedQuery on unit-carrying ND grids
# ========================================
# The separable gridded engine multiplies one per-axis kernel per dimension, so
# every axis must contribute its own `coord⁻ᴺ` — the whole-array fold used by the
# Fill/OOB post-pass (`_nd_fill_deriv_scale`) never runs on the in-domain path.
#
# Two consequences pinned here:
#
#  1. Linear's DerivOp{N≥2} anchor stores no geometry, so its kernel fabricated a
#     zero in VALUE units while the output array was sized in derivative units.
#     `(DerivOp(1), DerivOp(1))` happened to work (each axis carries its own
#     `inv_h`), which is why only the zero arm was wrong.
#
#  2. The output eltype was joined from ONE grid type and ONE target type. On
#     mixed-unit axes both collapse to an abstract `Quantity{Float64}`, so the
#     result array boxed every element — the gridded sibling of the point-wise
#     buffer fix in `_nd_value_eltype`.
#
# 1-D is unaffected: a 1-axis `GriddedQuery` on a 1-D interpolant routes to the
# ordinary vector batch, never to the separable engine.

@testitem "GriddedQuery ND: zero-derivative axes carry the grid's derivative units" begin
    using Unitful

    xs = (0.0:1.0:3.0)u"s"
    ys = (0.0:1.0:3.0)u"s"
    V = [1.0i + 2.0j for i in 1:4, j in 1:4]u"K"
    gq = GriddedQuery(([0.5, 1.5]u"s", [0.5, 1.5]u"s"))

    for f in (linear_interp, constant_interp)
        itp = f((xs, ys), V)
        @testset "$(nameof(f))" begin
            # Every op combination must agree with the point-wise scalar eval,
            # which is the reference for both value AND unit.
            for ops in (
                    (DerivOp(0), DerivOp(0)),
                    (DerivOp(1), DerivOp(0)),
                    (DerivOp(0), DerivOp(1)),
                    (DerivOp(2), DerivOp(0)),   # zero arm — the regression
                    (DerivOp(0), DerivOp(3)),
                    (DerivOp(1), DerivOp(1)),
                    (DerivOp(2), DerivOp(2)),
                )
                ref = itp(0.5u"s", 0.5u"s"; deriv = ops)
                got = itp(gq; deriv = ops)
                @test unit(eltype(got)) === unit(ref)
                @test got[1, 1] ≈ ref
            end
        end
    end
end

@testitem "GriddedQuery ND: mixed-unit axes allocate a concrete output" begin
    using Unitful

    xs = (0.0:1.0:3.0)u"s"
    ym = (0.0:1.0:3.0)u"m"
    V = [1.0i + 2.0j for i in 1:4, j in 1:4]u"K"
    gq = GriddedQuery(([0.5, 1.5]u"s", [0.5, 1.5]u"m"))

    for f in (linear_interp, constant_interp)
        itp = f((xs, ym), V)
        out = itp(gq)
        @testset "$(nameof(f))" begin
            # An abstract `Quantity{Float64}` eltype boxes every element.
            @test isconcretetype(eltype(out))
            @test eltype(out) === typeof(itp(0.5u"s", 0.5u"m"))
            @test out[1, 1] ≈ itp(0.5u"s", 0.5u"m")
            @test out[2, 2] ≈ itp(1.5u"s", 1.5u"m")
        end
    end

    # Same-dimension but different units on one axis: the query unit must not
    # break the buffer either (`cm` target over an `m` grid).
    itp2 = linear_interp((xs, ym), V)
    gq2 = GriddedQuery(([0.5, 1.5]u"s", [50.0, 150.0]u"cm"))
    out2 = itp2(gq2)
    @test isconcretetype(eltype(out2))
    @test out2[1, 1] ≈ itp2(0.5u"s", 0.5u"m")
end

@testitem "GriddedQuery ND: Real grids are unchanged" begin
    # The per-axis fold must be the identity on Real: same eltype, same values,
    # and the Int-grid → Float promotion still happens.
    xs = 0.0:1.0:3.0
    ys = 0.0:1.0:3.0
    V = [1.0i + 2.0j for i in 1:4, j in 1:4]
    itp = linear_interp((xs, ys), V)
    gq = GriddedQuery(([0.5, 1.5], [0.5, 1.5]))
    @test itp(gq) isa Matrix{Float64}
    @test itp(gq; deriv = (DerivOp(2), DerivOp(0))) isa Matrix{Float64}
    @test itp(gq)[1, 1] ≈ itp(0.5, 0.5)

    # Int grid + Float data, and a Float32 grid beside a Float64 one.
    itp_i = linear_interp((0:3, 0:3), V)
    @test itp_i(GriddedQuery(([0.5, 1.5], [0.5, 1.5]))) isa Matrix{Float64}

    x32 = Float32.(0.0:1.0:3.0)
    itp_m = linear_interp((x32, ys), V)
    @test eltype(itp_m(GriddedQuery(([0.5, 1.5], [0.5, 1.5])))) === Float64
end
