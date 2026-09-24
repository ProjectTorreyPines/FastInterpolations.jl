@testitem "Series _ContiguousIndices representation contract" setup = [Basic] begin
    x = collect(range(0.0, 1.0, 101)); y = sin.(x)     # shared ordinary grid

    @testset "forward anchor: ordinary grid ⇒ compact _ContiguousIndices ($sym)" for sym in
        (:linear, :cubic, :constant)
        aq = FI._anchor_query(x, 0.37, Val(sym))
        @test getfield(aq, :interval) isa FI._ContiguousIndices{2}
        @test aq.idxL == searchsortedlast(x, 0.37)      # virtual props still resolve the cell
        @test aq.idxR == aq.idxL + 1
    end

    @testset "forward anchor: exclusive-periodic seam keeps _ExplicitIndices (idxR wraps)" begin
        xax = FI._resolve_axis(x, PeriodicBC(endpoint = :exclusive, period = 1.1))
        idxL, idxR, _, _ = FI.search_interval(FI.DEFAULT_SEARCHER, xax, 0.999 * last(xax))
        iv = FI._interval_indices(xax, idxL, idxR)
        @test iv isa FI._ExplicitIndices{2}
        @test iv[Val(1)] == idxL && iv[Val(2)] == idxR
    end

    @testset "eval: contiguous anchor bit-identical to an explicit rebuild + smaller" begin
        litp = FI.linear_interp(x, y)
        la = FI._anchor_query(x, 0.37, Val(:linear))
        le = FI._LinearAnchoredQuery(
            FI._ExplicitIndices(la.idxL, la.idxR), la.xq, la.state, la.xL, la.h, la.inv_h, la.alpha
        )
        @test litp(la) === litp(le)
        @test litp(la; deriv = FI.DerivOp(1)) === litp(le; deriv = FI.DerivOp(1))
        @test sizeof(la) < sizeof(le)                   # 56 < 64

        # Cubic anchor forward eval is now adjoint-internal (no `itp(aq)`); the
        # contiguous vs explicit anchor differ only in the indices field, so the
        # taps match by construction and the size win is the check here.
        ca = FI._anchor_query(x, 0.37, Val(:cubic))
        ce = FI._CubicAdjointAnchor(
            FI._ExplicitIndices(ca.idxL, ca.idxR), ca.xq, ca.state, ca.w0, ca.w1, ca.w2, ca.w3, eltype(x)
        )
        @test (ca.idxL, ca.idxR) === (ce.idxL, ce.idxR)  # same taps
        @test sizeof(ca) < sizeof(ce)                   # 120 < 128

        kitp = FI.constant_interp(x, y; side = FI.NearestSide())
        ka = FI._anchor_query(x, 0.37, Val(:constant))
        ke = FI._ConstantAnchoredQuery(
            FI._ExplicitIndices(ka.idxL, ka.idxR), ka.xq, ka.state, ka.h, ka.dL
        )
        @test kitp(ka) === kitp(ke)
        @test sizeof(ka) < sizeof(ke)                   # 40 < 48
    end

    @testset "adjoint storage pins the representation" begin
        xa = collect(range(0.0, 1.0, 21)); xqa = collect(range(0.05, 0.95, 15))
        ladj = linear_adjoint(xa, xqa)
        cadj = cubic_adjoint(xa, xqa)                   # default bc = CubicFit()
        @test eltype(ladj.anchors) === FI._LinearAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}
        @test eltype(cadj.anchors) === FI._CubicAdjointAnchor{Float64, Float64, FI._ContiguousIndices{2}}

        # Exclusive-periodic representation is method-specific — both correct, same seam
        # cell: Linear wraps the axis (`_ExclusivePeriodicAxis` ⇒ explicit seam pair),
        # while Cubic materializes the extended inclusive grid so the seam is an ordinary
        # contiguous interior cell (n, n+1), folded to index 1 at scatter time.
        xper = collect(range(0.0, 2π, 21))[1:(end - 1)]; xqp = collect(range(0.3, 5.5, 15))
        bcp = PeriodicBC(endpoint = :exclusive, period = 2π)
        ladjp = linear_adjoint(xper, xqp; bc = bcp)
        cadjp = cubic_adjoint(xper, xqp; bc = bcp)
        @test eltype(ladjp.anchors) === FI._LinearAnchoredQuery{Float64, Float64, FI._ExplicitIndices{2}}
        @test eltype(cadjp.anchors) === FI._CubicAdjointAnchor{Float64, Float64, FI._ContiguousIndices{2}}
    end
end
