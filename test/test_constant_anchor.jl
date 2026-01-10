# Test suite for Constant Anchored Query functionality
# Phase 0B of AbstractMultiInterpolant implementation

using Test
using FastInterpolations

@testset "Constant Anchored Query" begin

    # ========================================
    # Struct Fields Tests
    # ========================================
    @testset "struct _ConstantAnchoredQuery fields" begin
        x = collect(range(0.0, 1.0, 11))  # 0.0, 0.1, ..., 1.0
        xq = 0.35

        aq = FastInterpolations._anchor_query(x, xq, Val(:constant))

        @test aq isa FastInterpolations._ConstantAnchoredQuery{Float64}
        @test hasfield(typeof(aq), :idx)
        @test hasfield(typeof(aq), :xq)
        @test hasfield(typeof(aq), :side)
        @test hasfield(typeof(aq), :h)
        @test hasfield(typeof(aq), :dL)

        # Verify field types
        @test aq.idx isa Int
        @test aq.xq isa Float64
        @test aq.side isa UInt8
        @test aq.h isa Float64
        @test aq.dL isa Float64
    end

    # ========================================
    # Scalar Construction Tests
    # ========================================
    @testset "_anchor_query scalar construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1

        # Query inside domain
        aq = FastInterpolations._anchor_query(x, 0.35, Val(:constant))
        @test aq.idx == 4  # interval [0.3, 0.4]
        @test aq.xq == 0.35
        @test aq.side == 0x00  # inside
        @test aq.h ≈ 0.1
        @test aq.dL ≈ 0.05  # 0.35 - 0.3

        # Query at left boundary
        aq_left = FastInterpolations._anchor_query(x, 0.0, Val(:constant))
        @test aq_left.idx == 1
        @test aq_left.side == 0x00
        @test aq_left.dL ≈ 0.0

        # Query at right boundary
        aq_right = FastInterpolations._anchor_query(x, 1.0, Val(:constant))
        @test aq_right.side == 0x00
    end

    # ========================================
    # Vector Construction Tests
    # ========================================
    @testset "_anchor_query vector construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1
        xq_vec = [0.15, 0.35, 0.75]

        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))

        @test length(aq_vec) == 3
        @test aq_vec[1].idx == 2   # interval [0.1, 0.2]
        @test aq_vec[2].idx == 4   # interval [0.3, 0.4]
        @test aq_vec[3].idx == 8   # interval [0.7, 0.8]

        @test all(aq.h ≈ 0.1 for aq in aq_vec)
    end

    # ========================================
    # Evaluation Tests - :nearest mode
    # ========================================
    @testset "itp(aq) for :nearest mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; side=:nearest, extrap=:extension)

        xq_points = [0.05, 0.15, 0.35, 0.65, 0.95]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:constant))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - :left mode
    # ========================================
    @testset "itp(aq) for :left mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; side=:left, extrap=:extension)

        xq_points = [0.05, 0.15, 0.35, 0.65, 0.95]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:constant))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - :right mode
    # ========================================
    @testset "itp(aq) for :right mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; side=:right, extrap=:extension)

        xq_points = [0.05, 0.15, 0.35, 0.65, 0.95]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:constant))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Domain Boundary Handling Tests
    # ========================================
    @testset "domain boundary handling" begin
        x = collect(range(0.0, 1.0, 11))

        # Inside domain
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:constant))
        @test aq_inside.side == 0x00

        # Below domain
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:constant))
        @test aq_below.side == 0x01

        # Above domain
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:constant))
        @test aq_above.side == 0x02

        # Verify extrapolation works with anchors
        y = collect(1.0:11.0)
        itp_ext = constant_interp(x, y; extrap=:extension)
        itp_const = constant_interp(x, y; extrap=:constant)

        @test itp_ext(aq_below) ≈ itp_ext(-0.5)
        @test itp_ext(aq_above) ≈ itp_ext(1.5)
        @test itp_const(aq_below) ≈ itp_const(-0.5)
        @test itp_const(aq_above) ≈ itp_const(1.5)
    end

    # ========================================
    # Wrap Mode Tests
    # ========================================
    @testset "wrap mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; extrap=:wrap)

        # Query outside domain with wrap=true
        aq_wrap = FastInterpolations._anchor_query(x, 1.5, Val(:constant); wrap=true)
        @test aq_wrap.xq ≈ 0.5
        @test aq_wrap.side == 0x00

        # Verify wrapped evaluation matches
        @test itp(aq_wrap) ≈ itp(1.5)
    end

    # ========================================
    # Type Promotion Tests
    # ========================================
    @testset "type promotion Real → Float" begin
        x = collect(range(0.0, 1.0, 11))

        # Int query should be promoted
        aq_int = FastInterpolations._anchor_query(x, 0, Val(:constant))
        @test aq_int.xq isa Float64
        @test aq_int.xq ≈ 0.0
    end

    # ========================================
    # Float32 Support Tests
    # ========================================
    @testset "Float32 support" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        xq = 0.35f0

        aq = FastInterpolations._anchor_query(x, xq, Val(:constant))
        @test aq isa FastInterpolations._ConstantAnchoredQuery{Float32}
        @test aq.xq isa Float32
        @test aq.h isa Float32
        @test aq.dL isa Float32
    end

    # ========================================
    # In-Place Vector Evaluation Tests
    # ========================================
    @testset "in-place vector evaluation with anchors" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; extrap=:extension)

        xq_vec = [0.05, 0.15, 0.35, 0.65, 0.95]
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))

        # Test out-of-place vector evaluation
        result = itp(aq_vec)
        expected = itp(xq_vec)
        @test result ≈ expected

        # Test in-place vector evaluation
        output = zeros(Float64, 5)
        itp(output, aq_vec)
        @test output ≈ expected
    end

    # ========================================
    # Non-Uniform Grid Tests
    # ========================================
    @testset "non-uniform grid support" begin
        # Non-uniform grid
        x = [0.0, 0.1, 0.3, 0.6, 1.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        itp = constant_interp(x, y; extrap=:extension)

        xq = 0.45  # interval [0.3, 0.6]
        aq = FastInterpolations._anchor_query(x, xq, Val(:constant))

        @test aq.idx == 3  # interval [0.3, 0.6]
        @test aq.xq == 0.45
        @test aq.h ≈ 0.3  # 0.6 - 0.3
        @test aq.dL ≈ 0.15  # 0.45 - 0.3

        # Verify evaluation matches
        @test itp(aq) ≈ itp(xq)
    end

    # ========================================
    # Zero-Allocation Test
    # ========================================
    @testset "zero-allocation with pre-built anchors" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(x)
        itp = constant_interp(x, y; extrap=:extension)

        xq_vec = collect(range(0.01, 0.99, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec)

        # Allocation test
        allocs = @allocated itp(output, aq_vec)
        @test allocs == 0
    end

end
