@testset "Anchor Common: _AnchorLoc + _anchor_loc" begin
    FI = FastInterpolations

    # ========================================
    # _anchor_loc: basic functionality
    # ========================================

    @testset "_anchor_loc — inside domain" begin
        x = collect(range(0.0, 1.0, 11))  # 0.0, 0.1, ..., 1.0
        loc = FI._anchor_loc(x, 0.35, false)
        @test loc isa FI._AnchorLoc{Float64, Float64}
        @test loc.idx == 4         # interval [0.3, 0.4)
        @test loc.xq == 0.35
        @test loc.state == FI.IN_DOMAIN
        @test loc.xL == 0.3
        @test loc.xR ≈ 0.4
    end

    @testset "_anchor_loc — below domain" begin
        x = collect(range(0.0, 1.0, 11))
        loc = FI._anchor_loc(x, -0.5, false)
        @test loc.state == FI.OOB_LEFT
        @test loc.idx == 1
        @test loc.xL == 0.0
        @test loc.xR ≈ 0.1
        @test loc.xq == -0.5
    end

    @testset "_anchor_loc — above domain" begin
        x = collect(range(0.0, 1.0, 11))
        loc = FI._anchor_loc(x, 1.5, false)
        @test loc.state == FI.OOB_RIGHT
        @test loc.idx == 10         # last interval
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
        @test loc_left.idx == 1

        # Right boundary: inside (xq == x_max → state=IN_DOMAIN)
        loc_right = FI._anchor_loc(x, 1.0, false)
        @test loc_right.state == FI.IN_DOMAIN
    end

    @testset "_anchor_loc — wrap at x_max" begin
        x = collect(range(0.0, 1.0, 11))
        # xq == x_max with wrap → should wrap to x_min
        loc = FI._anchor_loc(x, 1.0, true)
        @test loc.state == FI.IN_DOMAIN
        @test loc.xq ≈ 0.0
    end

    # ========================================
    # _anchor_loc with _CachedRange
    # ========================================

    @testset "_anchor_loc — _CachedRange (uniform grid)" begin
        x_range = range(0.0, 1.0, 101)
        x_cached = FI._to_float(x_range, Float64)
        loc = FI._anchor_loc(x_cached, 0.355, false)
        @test loc.state == FI.IN_DOMAIN
        @test loc.idx >= 1
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
        @test loc isa FI._AnchorLoc{Float32, Float32}
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
        @test loc_nu.idx == 2  # interval [0.1, 0.4)
        @test loc_nu.xL == 0.1
        @test loc_nu.xR == 0.4
    end

end
