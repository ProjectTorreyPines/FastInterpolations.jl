@testitem "Anchor Common: _AnchorLoc + _anchor_loc" begin
    FI = FastInterpolations
    using FastInterpolations: _ContiguousIndices, _ExplicitIndices

    # ========================================
    # _anchor_loc: basic functionality
    # ========================================

    @testset "_anchor_loc — inside domain" begin
        x = collect(range(0.0, 1.0, 11))  # 0.0, 0.1, ..., 1.0
        loc = FI._anchor_loc(x, 0.35, false)
        # Ordinary axis → contiguous interval; canonical layout has no redundant idxR::Int.
        @test loc isa FI._AnchorLoc{_ContiguousIndices{2}, Float64, Float64}
        @test loc.interval isa _ContiguousIndices{2}
        @test loc.interval == _ContiguousIndices{2}(4)
        @test loc.idxL == 4        # interval [0.3, 0.4)
        @test loc.idxR == 5
        @test loc.xq == 0.35
        @test loc.state == FI.IN_DOMAIN
        @test loc.xL == 0.3
        @test loc.xR ≈ 0.4
        # Ordinary Float64 anchor: interval(8) + xq(8) + state(8) + xL(8) + xR(8).
        @test sizeof(loc) == 40
    end

    @testset "_anchor_loc — below domain" begin
        x = collect(range(0.0, 1.0, 11))
        loc = FI._anchor_loc(x, -0.5, false)
        @test loc.state == FI.OOB_LEFT
        @test loc.interval isa _ContiguousIndices{2}
        @test loc.idxL == 1
        @test loc.idxR == 2
        @test loc.xL == 0.0
        @test loc.xR ≈ 0.1
        @test loc.xq == -0.5
    end

    @testset "_anchor_loc — above domain" begin
        x = collect(range(0.0, 1.0, 11))
        loc = FI._anchor_loc(x, 1.5, false)
        @test loc.state == FI.OOB_RIGHT
        @test loc.interval isa _ContiguousIndices{2}
        @test loc.idxL == 10        # last interval
        @test loc.idxR == 11
        @test loc.xL ≈ 0.9
        @test loc.xR == 1.0
        @test loc.xq == 1.5
    end

    @testset "_anchor_loc — wrap mode" begin
        x = collect(range(0.0, 1.0, 11))
        # Below domain → wrapped inside
        loc = FI._anchor_loc(x, -0.15, true)
        @test loc.state == FI.IN_DOMAIN
        @test loc.xq ≈ 0.85       # wrapped: -0.15 + 1.0 = 0.85

        # Above domain → wrapped inside
        loc2 = FI._anchor_loc(x, 1.25, true)
        @test loc2.state == FI.IN_DOMAIN
        @test loc2.xq ≈ 0.25      # wrapped: 1.25 - 1.0 = 0.25
    end

    @testset "_anchor_loc — boundary points" begin
        x = collect(range(0.0, 1.0, 11))
        # Left boundary: inside
        loc_left = FI._anchor_loc(x, 0.0, false)
        @test loc_left.state == FI.IN_DOMAIN
        @test loc_left.idxL == 1

        # Right boundary: inside (xq == x_max → state=IN_DOMAIN)
        loc_right = FI._anchor_loc(x, 1.0, false)
        @test loc_right.state == FI.IN_DOMAIN
    end

    @testset "_anchor_loc — closed boundary at x_max (no wrap)" begin
        x = collect(range(0.0, 1.0, 11))
        # Closed semantics: xq == x_max stays in-domain, lands on the last cell.
        loc = FI._anchor_loc(x, 1.0, true)
        @test loc.state == FI.IN_DOMAIN
        @test loc.xq ≈ 1.0
        @test loc.idxL == length(x) - 1   # last cell (n-1)
        @test loc.xR ≈ 1.0
    end

    @testset "_anchor_loc — strictly OOB right still wraps" begin
        x = collect(range(0.0, 1.0, 11))
        # q = 1.25 = x_min + 1.25*period → wraps to 0.25
        loc = FI._anchor_loc(x, 1.25, true)
        @test loc.state == FI.IN_DOMAIN
        @test loc.xq ≈ 0.25
    end

    @testset "_anchor_loc — _ExclusivePeriodicAxis preserves search idxR" begin
        x = collect(0.0:3.0)
        bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
        xper = FI._cache_axis(x, bc)
        searcher = FI._to_searcher(BinarySearch())

        # The seam cell's right tap is the physical first node. `_anchor_loc`
        # should carry the physical `search_interval` interval as an
        # `_ExplicitIndices{2}`, not manufacture a virtual idxR by hand.
        expected = FI.search_interval(searcher, xper, 4.25)
        loc = FI._anchor_loc(xper, 4.25, false, searcher)
        @test expected[1] == 4
        @test expected[2] == 1
        @test loc.state == FI.OOB_RIGHT
        @test loc.interval isa _ExplicitIndices{2}
        @test loc.interval == _ExplicitIndices(4, 1)
        @test loc.idxL == expected[1]
        @test loc.idxR == expected[2]
        @test loc.xL == expected[3]
        @test loc.xR == expected[4]
        # Explicit Float64 anchor keeps both indices: interval(16) + 8 + 8 + 8 + 8.
        @test sizeof(loc) == 48

        aq_linear = FI._anchor_query(xper, 4.25, Val(:linear), false, searcher)
        aq_constant = FI._anchor_query(xper, 4.25, Val(:constant), false, searcher)
        aq_cubic = FI._anchor_query(xper, 4.25, Val(:cubic), false, searcher)
        @test aq_linear.idxL == 4 && aq_linear.idxR == 1
        @test aq_constant.idxL == 4 && aq_constant.idxR == 1
        @test aq_cubic.idxL == 4 && aq_cubic.idxR == 1

        # Representation is selected per axis type, not per query value: even an
        # interior query on an exclusive-periodic axis carries `_ExplicitIndices`,
        # so one anchor vector always has one concrete element type.
        loc_wrap = FI._anchor_loc(xper, 4.25, true, searcher)
        @test loc_wrap.state == FI.IN_DOMAIN
        @test loc_wrap.xq ≈ 0.25
        @test loc_wrap.interval isa _ExplicitIndices{2}
        @test loc_wrap.interval == _ExplicitIndices(1, 2)
        @test loc_wrap.idxL == 1
        @test loc_wrap.idxR == 2
    end

    # ========================================
    # _anchor_loc with _CachedRange
    # ========================================

    @testset "_anchor_loc — _CachedRange (uniform grid)" begin
        x_range = range(0.0, 1.0, 101)
        x_cached = FI._to_float(x_range, Float64)
        loc = FI._anchor_loc(x_cached, 0.355, false)
        @test loc.state == FI.IN_DOMAIN
        @test loc.interval isa _ContiguousIndices{2}
        @test loc.idxL >= 1
        @test loc.idxR == loc.idxL + 1
        @test loc.xL <= 0.355 <= loc.xR
    end

    # ========================================
    # AD support (ForwardDiff.Dual)
    # ========================================

    @testset "_anchor_loc — ForwardDiff.Dual preservation" begin
        using ForwardDiff
        x = collect(range(0.0, 1.0, 11))
        xq_dual = ForwardDiff.Dual(0.35, 1.0)
        loc = FI._anchor_loc(x, xq_dual, false)
        @test loc.interval isa _ContiguousIndices{2}
        @test loc.xq isa ForwardDiff.Dual
        @test ForwardDiff.value(loc.xq) ≈ 0.35
        @test ForwardDiff.partials(loc.xq)[1] ≈ 1.0
        @test loc.state == FI.IN_DOMAIN
    end

    # ========================================
    # Float32 support
    # ========================================

    @testset "_anchor_loc — Float32" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        loc = FI._anchor_loc(x, 0.35f0, false)
        @test loc isa FI._AnchorLoc{_ContiguousIndices{2}, Float32, Float32}
        @test loc.state == FI.IN_DOMAIN
    end

    # ========================================
    # Custom Searcher policy
    # ========================================

    @testset "_anchor_loc — explicit BinarySearch" begin
        x = collect(range(0.0, 1.0, 101))
        searcher = FI._to_searcher(BinarySearch())
        loc = FI._anchor_loc(x, 0.355, false, searcher)
        @test loc.state == FI.IN_DOMAIN
        @test loc.xL <= 0.355 <= loc.xR
    end

    @testset "_anchor_loc — explicit LinearBinarySearch" begin
        x = collect(range(0.0, 1.0, 101))
        searcher = FI._to_searcher(LinearBinarySearch())
        loc = FI._anchor_loc(x, 0.355, false, searcher)
        @test loc.state == FI.IN_DOMAIN
        @test loc.xL <= 0.355 <= loc.xR

        # Non-uniform grid
        x_nu = [0.0, 0.1, 0.4, 0.5, 1.0]
        loc_nu = FI._anchor_loc(x_nu, 0.25, false, searcher)
        @test loc_nu.state == FI.IN_DOMAIN
        @test loc_nu.idxL == 2  # interval [0.1, 0.4)
        @test loc_nu.xL == 0.1
        @test loc_nu.xR == 0.4
    end

    # ========================================
    # Inference: both representations concrete
    # ========================================

    @testset "_anchor_loc — inference for both representations" begin
        x = collect(range(0.0, 1.0, 11))
        searcher = FI._to_searcher(BinarySearch())
        @test @inferred(FI._anchor_loc(x, 0.35, false, searcher)) isa
            FI._AnchorLoc{_ContiguousIndices{2}, Float64, Float64}

        xper = FI._cache_axis(collect(0.0:3.0), PeriodicBC(endpoint = :exclusive, period = 4.0))
        loc_per = @inferred FI._anchor_loc(xper, 1.5, false, searcher)
        @test loc_per.interval isa _ExplicitIndices{2}
    end

end
