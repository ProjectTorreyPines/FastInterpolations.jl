# ========================================
# ND one-shot derivative queries on unit-carrying grids
# ========================================
# The one-shot entries compute their result witness `Tr` in VALUE space and the
# scalar form asserts it (`::Tr`). Without folding the derivative order in, a
# `deriv = DerivOp(1, 0)` query returns `W/s` against a `W` assertion (scalar) or
# allocates a `W` buffer for `W/s` values (batch).
#
# ⚠️ The axes must share ONE unit. On mixed-unit axes the joined `Tg` degrades to
# an abstract `Quantity{Float64}`, the assertion becomes vacuous and the bug
# hides — the "less broken" case is the one that reports it.
#
# Linear is the only family this reaches: cubic/quadratic/hetero ND one-shots
# refuse unit grids up front (`_check_nd_solver_grid` / `_check_nd_hetero_grid`)
# and Constant already folds through `_interp_nd_output_eltype`.

@testitem "ND one-shot: derivative queries keep the grid's derivative units" begin
    using Unitful

    grids = (collect((0:1.0:4)u"s"), collect((0:1.0:3)u"s"))
    data = [Float64(i + j)u"W" for i in 1:5, j in 1:4]
    q = (2.5u"s", 1.5u"s")

    # The persistent interpolant is the reference — it has always been right.
    ref_val = interp(grids, data; method = LinearInterp())(q)
    ref_d10 = interp(grids, data; method = LinearInterp())(q; deriv = DerivOp(1, 0))
    ref_d01 = interp(grids, data; method = LinearInterp())(q; deriv = DerivOp(0, 1))
    @test unit(ref_d10) === u"W" / u"s"

    @testset "linear_interp scalar" begin
        @test linear_interp(grids, data, q) ≈ ref_val
        @test linear_interp(grids, data, q; deriv = DerivOp(1, 0)) ≈ ref_d10
        @test linear_interp(grids, data, q; deriv = DerivOp(0, 1)) ≈ ref_d01
        # second order is identically zero, but still in ∂-units
        @test unit(linear_interp(grids, data, q; deriv = DerivOp(2, 0))) === u"W" / u"s"^2
    end

    @testset "linear_interp batch" begin
        out = linear_interp(grids, data, [q]; deriv = DerivOp(1, 0))
        @test eltype(out) === typeof(ref_d10)
        @test out[1] ≈ ref_d10
        @test eltype(linear_interp(grids, data, [q])) === typeof(ref_val)
    end

    @testset "interp(...; method) scalar" begin
        @test interp(grids, data, q; method = LinearInterp(), deriv = DerivOp(1, 0)) ≈ ref_d10
        # an all-Linear method TUPLE routes through the hetero front, which
        # computes its own scalar witness
        @test interp(
            grids, data, q;
            method = (LinearInterp(), LinearInterp()), deriv = DerivOp(1, 0)
        ) ≈ ref_d10
    end

    @testset "interp(...; method) batch" begin
        out = interp(grids, data, [q]; method = LinearInterp(), deriv = DerivOp(1, 0))
        @test eltype(out) === typeof(ref_d10)
        @test out[1] ≈ ref_d10
    end

    @testset "constant_interp batch" begin
        out = constant_interp(grids, data, [q]; deriv = DerivOp(1, 0))
        @test eltype(out) === typeof(ref_d10)
        @test unit(out[1]) === u"W" / u"s"
        @test iszero(out[1])
    end

    # Constant is the sibling that already worked on the scalar path — pin it so
    # the shared helper cannot regress it.
    @test unit(constant_interp(grids, data, q; deriv = DerivOp(1, 0))) === u"W" / u"s"
end

@testitem "ND one-shot: Real grids and mixed-unit axes are unchanged" begin
    using Unitful

    # Real: the derivative fold is the identity, values and types unchanged.
    gr = (collect(0.0:1.0:4.0), collect(0.0:1.0:3.0))
    dr = [Float64(i + j) for i in 1:5, j in 1:4]
    qr = (2.5, 1.5)
    itp = interp(gr, dr; method = LinearInterp())
    @test linear_interp(gr, dr, qr) === itp(qr)
    @test linear_interp(gr, dr, qr; deriv = DerivOp(1, 0)) === itp(qr; deriv = DerivOp(1, 0))
    @test linear_interp(gr, dr, [qr]; deriv = DerivOp(1, 0)) isa Vector{Float64}
    @test interp(gr, dr, qr; method = (LinearInterp(), LinearInterp()), deriv = DerivOp(1, 0)) ===
        itp(qr; deriv = DerivOp(1, 0))

    # Mixed-unit axes: the joined witness is abstract, so this never asserted —
    # it must keep working and agree with the persistent path.
    gm = (collect((0:1.0:4)u"s"), collect((0:1.0:3)u"m"))
    dm = [Float64(i + j)u"W" for i in 1:5, j in 1:4]
    qm = (2.5u"s", 1.5u"m")
    itp_m = interp(gm, dm; method = LinearInterp())
    @test linear_interp(gm, dm, qm; deriv = DerivOp(1, 0)) ≈ itp_m(qm; deriv = DerivOp(1, 0))
    @test linear_interp(gm, dm, qm; deriv = DerivOp(0, 1)) ≈ itp_m(qm; deriv = DerivOp(0, 1))
end
