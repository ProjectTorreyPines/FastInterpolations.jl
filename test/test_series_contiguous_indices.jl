@testitem "Series anchors use _ContiguousIndices on ordinary grids" begin
    const FI = FastInterpolations
    x = collect(range(0.0, 1.0, 101))           # ordinary AbstractVector grid

    for tag in (Val(:linear), Val(:cubic), Val(:constant))
        aq = FI._anchor_query(x, 0.37, tag)
        # ordinary grid ⇒ compact representation
        @test getfield(aq, :interval) isa FI._ContiguousIndices{2}
        # virtual props still resolve the physical cell
        @test aq.idxL == searchsortedlast(x, 0.37)
        @test aq.idxR == aq.idxL + 1
    end
end

@testitem "Series wrap/periodic seam keeps _ExplicitIndices" begin
    const FI = FastInterpolations
    x = collect(range(0.0, 1.0, 11))
    # exclusive-periodic axis: seam cell (n, 1) must stay explicit + preserve idxR
    xax = FI._resolve_axis(x, PeriodicBC(endpoint = :exclusive, period = 1.1))
    idxL, idxR, _, _ = FI.search_interval(FI.DEFAULT_SEARCHER, xax, 0.999 * last(xax))
    iv = FI._interval_indices(xax, idxL, idxR)
    @test iv isa FI._ExplicitIndices{2}
    @test iv[Val(1)] == idxL && iv[Val(2)] == idxR
end

@testitem "Linear anchor: contiguous is bit-identical to explicit + smaller" begin
    const FI = FastInterpolations
    x = collect(range(0.0, 1.0, 101)); y = sin.(x)
    itp = FI.linear_interp(x, y)
    aqc = FI._anchor_query(x, 0.37, Val(:linear))                       # contiguous
    aqe = FI._LinearAnchoredQuery(
        FI._ExplicitIndices(aqc.idxL, aqc.idxR),
        aqc.xq, aqc.state, aqc.xL, aqc.h, aqc.inv_h, aqc.alpha
    )
    @test itp(aqc) === itp(aqe)                                         # bitwise
    @test itp(aqc; deriv = FI.DerivOp(1)) === itp(aqe; deriv = FI.DerivOp(1))
    @test sizeof(aqc) < sizeof(aqe)                                     # 56 < 64
end

@testitem "Cubic anchor: contiguous is bit-identical to explicit + smaller" begin
    const FI = FastInterpolations
    x = collect(range(0.0, 1.0, 101)); y = sin.(x)
    itp = FI.cubic_interp(x, y)
    aqc = FI._anchor_query(x, 0.37, Val(:cubic))
    aqe = FI._CubicAnchoredQuery(
        FI._ExplicitIndices(aqc.idxL, aqc.idxR),
        aqc.xq, aqc.state, aqc.w0, aqc.w1, aqc.w2, aqc.w3, eltype(x)
    )
    @test itp(aqc) === itp(aqe)
    @test itp(aqc; deriv = FI.DerivOp(1)) === itp(aqe; deriv = FI.DerivOp(1))
    @test sizeof(aqc) < sizeof(aqe)                                     # 120 < 128
end

@testitem "Constant anchor: contiguous is bit-identical to explicit + smaller" begin
    const FI = FastInterpolations
    x = collect(range(0.0, 1.0, 101)); y = sin.(x)
    itp = FI.constant_interp(x, y; side = FI.NearestSide())
    aqc = FI._anchor_query(x, 0.37, Val(:constant))
    aqe = FI._ConstantAnchoredQuery(
        FI._ExplicitIndices(aqc.idxL, aqc.idxR),
        aqc.xq, aqc.state, aqc.h, aqc.dL
    )
    @test itp(aqc) === itp(aqe)
    @test sizeof(aqc) < sizeof(aqe)                                     # 40 < 48
end

@testitem "Series adjoints pin anchor representation" begin
    const FI = FastInterpolations
    # ordinary grid ⇒ adjoint storage uses compact _ContiguousIndices
    x = collect(range(0.0, 1.0, 21)); xq = collect(range(0.05, 0.95, 15))
    ladj = linear_adjoint(x, xq)
    cadj = cubic_adjoint(x, xq)                          # default bc = CubicFit()
    @test eltype(ladj.anchors) === FI._LinearAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}
    @test eltype(cadj.anchors) === FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}

    # Exclusive-periodic representation is method-specific — both correct, same seam
    # cell: Linear wraps the axis (`_ExclusivePeriodicAxis` ⇒ explicit seam pair),
    # while Cubic materializes the extended inclusive grid so the seam is an ordinary
    # contiguous interior cell (n, n+1), folded to index 1 at scatter time.
    xper = collect(range(0.0, 2π, 21))[1:(end - 1)]; xqp = collect(range(0.3, 5.5, 15))
    bcp = PeriodicBC(endpoint = :exclusive, period = 2π)
    ladjp = linear_adjoint(xper, xqp; bc = bcp)
    cadjp = cubic_adjoint(xper, xqp; bc = bcp)
    @test eltype(ladjp.anchors) === FI._LinearAnchoredQuery{Float64, Float64, FI._ExplicitIndices{2}}
    @test eltype(cadjp.anchors) === FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}
end
