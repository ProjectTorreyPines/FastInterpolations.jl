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
