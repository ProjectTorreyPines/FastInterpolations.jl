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
