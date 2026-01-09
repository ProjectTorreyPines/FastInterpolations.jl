@testset "Cubic Anchored Query" begin

    # ========================================
    # Phase 1: Core Types & Weight Computation
    # ========================================

    @testset "CubicAnchoredQuery struct" begin
        x = collect(range(0.0, 1.0, 101))
        xq = 0.35

        # Basic construction
        aq = anchor_query(x, xq)
        @test aq isa CubicAnchoredQuery{Float64, EvalValue}

        # Derivative variants
        aq1 = anchor_query(x, xq; deriv=1)
        @test aq1 isa CubicAnchoredQuery{Float64, EvalDeriv1}

        aq2 = anchor_query(x, xq; deriv=2)
        @test aq2 isa CubicAnchoredQuery{Float64, EvalDeriv2}

        # Float32 support
        x32 = Float32.(x)
        xq32 = Float32(0.35)
        aq32 = anchor_query(x32, xq32)
        @test aq32 isa CubicAnchoredQuery{Float32, EvalValue}
    end

    @testset "Grid ID computation" begin
        x1 = collect(range(0.0, 1.0, 101))
        x2 = collect(range(0.0, 1.0, 101))  # same content
        x3 = collect(range(0.0, 2.0, 101))  # different content, same length
        x4 = collect(range(0.0, 1.0, 51))   # different length

        # _grid_id returns (length, hash) tuple
        gid1 = FastInterpolations._grid_id(x1)
        gid2 = FastInterpolations._grid_id(x2)
        gid3 = FastInterpolations._grid_id(x3)
        gid4 = FastInterpolations._grid_id(x4)

        @test gid1 isa Tuple{Int, UInt}
        @test gid1[1] == length(x1)
        @test gid1[2] == hash(x1)

        # Same content → same grid_id
        @test gid1 == gid2

        # Different content → different hash (almost certainly)
        @test gid1 != gid3

        # Different length → different grid_id
        @test gid1 != gid4
    end

    @testset "Anchor idx field" begin
        x = collect(range(0.0, 1.0, 11))  # 10 intervals, h=0.1

        # Interior points
        aq1 = anchor_query(x, 0.05)
        @test aq1.idx == 1  # [0.0, 0.1)

        aq2 = anchor_query(x, 0.15)
        @test aq2.idx == 2  # [0.1, 0.2)

        aq3 = anchor_query(x, 0.95)
        @test aq3.idx == 10  # [0.9, 1.0]

        # At grid points
        aq_left = anchor_query(x, 0.0)
        @test aq_left.idx == 1

        aq_right = anchor_query(x, 1.0)
        @test aq_right.idx == 10  # last interval
    end

    @testset "Anchor side field" begin
        x = collect(range(0.0, 1.0, 11))

        # Interior: side = 0
        aq_inside = anchor_query(x, 0.5)
        @test aq_inside.side == 0x00

        # At boundaries: side = 0 (still inside domain)
        aq_left_bound = anchor_query(x, 0.0)
        @test aq_left_bound.side == 0x00

        aq_right_bound = anchor_query(x, 1.0)
        @test aq_right_bound.side == 0x00

        # Below minimum: side = 1
        aq_below = anchor_query(x, -0.5)
        @test aq_below.side == 0x01

        # Above maximum: side = 2
        aq_above = anchor_query(x, 1.5)
        @test aq_above.side == 0x02
    end

    @testset "Anchor xq field preserved" begin
        x = collect(range(0.0, 1.0, 101))

        xq_values = [0.0, 0.35, 0.5, 1.0, -0.5, 1.5]
        for xq in xq_values
            aq = anchor_query(x, xq)
            @test aq.xq == xq
        end
    end

    @testset "Anchor weights tuple" begin
        x = collect(range(0.0, 1.0, 101))
        xq = 0.35

        aq = anchor_query(x, xq)
        @test aq.w isa NTuple{4, Float64}

        aq32 = anchor_query(Float32.(x), Float32(xq))
        @test aq32.w isa NTuple{4, Float32}
    end

    @testset "Periodic anchor wrapping" begin
        x = collect(range(0.0, 2π, 101))

        # Query outside domain with periodic=true should wrap
        xq_outside = 2π + 1.0  # wraps to ~1.0
        aq_periodic = anchor_query(x, xq_outside; periodic=true)

        # Should be inside after wrapping
        @test aq_periodic.side == 0x00
        @test aq_periodic.xq != xq_outside  # xq is wrapped value

        # Without periodic, should be outside
        aq_nonperiodic = anchor_query(x, xq_outside; periodic=false)
        @test aq_nonperiodic.side == 0x02  # above max
    end

    @testset "Invalid deriv argument" begin
        x = collect(range(0.0, 1.0, 101))
        @test_throws ArgumentError anchor_query(x, 0.5; deriv=-1)
        @test_throws ArgumentError anchor_query(x, 0.5; deriv=3)
    end

end
