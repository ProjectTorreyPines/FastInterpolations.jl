# Test suite for Quadratic Anchored Query functionality
# Phase 0C of AbstractMultiInterpolant implementation

using Test
using FastInterpolations

@testset "Quadratic Anchored Query" begin

    # ========================================
    # Struct Fields Tests
    # ========================================
    @testset "struct _QuadraticAnchoredQuery fields" begin
        x = collect(range(0.0, 1.0, 11))  # 0.0, 0.1, ..., 1.0
        xq = 0.35

        aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))

        @test aq isa FastInterpolations._QuadraticAnchoredQuery{Float64}
        @test hasfield(typeof(aq), :idx)
        @test hasfield(typeof(aq), :xq)
        @test hasfield(typeof(aq), :side)
        @test hasfield(typeof(aq), :dL)

        # Verify field types
        @test aq.idx isa Int
        @test aq.xq isa Float64
        @test aq.side isa UInt8
        @test aq.dL isa Float64
    end

    # ========================================
    # Scalar Construction Tests
    # ========================================
    @testset "_anchor_query scalar construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1

        # Query inside domain
        aq = FastInterpolations._anchor_query(x, 0.35, Val(:quadratic))
        @test aq.idx == 4  # interval [0.3, 0.4]
        @test aq.xq == 0.35
        @test aq.side == 0x00  # inside
        @test aq.dL ≈ 0.05  # 0.35 - 0.3

        # Query at left boundary
        aq_left = FastInterpolations._anchor_query(x, 0.0, Val(:quadratic))
        @test aq_left.idx == 1
        @test aq_left.side == 0x00
        @test aq_left.dL ≈ 0.0

        # Query at right boundary
        aq_right = FastInterpolations._anchor_query(x, 1.0, Val(:quadratic))
        @test aq_right.idx == 10  # last interval
        @test aq_right.side == 0x00
    end

    # ========================================
    # Vector Construction Tests
    # ========================================
    @testset "_anchor_query vector construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1
        xq_vec = [0.15, 0.35, 0.75]

        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))

        @test length(aq_vec) == 3
        @test aq_vec[1].idx == 2   # interval [0.1, 0.2]
        @test aq_vec[2].idx == 4   # interval [0.3, 0.4]
        @test aq_vec[3].idx == 8   # interval [0.7, 0.8]

        @test aq_vec[1].dL ≈ 0.05  # 0.15 - 0.1
        @test aq_vec[2].dL ≈ 0.05  # 0.35 - 0.3
        @test aq_vec[3].dL ≈ 0.05  # 0.75 - 0.7
    end

    # ========================================
    # Evaluation Tests - Value
    # ========================================
    @testset "itp(aq) evaluation matches itp(xq)" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap=:extension)

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - Derivative 1
    # ========================================
    @testset "itp(aq; deriv=1) derivative evaluation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap=:extension)

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq; deriv=1) ≈ itp(xq; deriv=1)
        end
    end

    # ========================================
    # Evaluation Tests - Derivative 2
    # ========================================
    @testset "itp(aq; deriv=2) derivative evaluation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap=:extension)

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq; deriv=2) ≈ itp(xq; deriv=2)
        end
    end

    # ========================================
    # Extrapolation Mode Tests
    # ========================================
    @testset "extrapolation modes with anchors" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2

        # Extension mode
        itp_ext = quadratic_interp(x, y; extrap=:extension)
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:quadratic))
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic))

        @test itp_ext(aq_below) ≈ itp_ext(-0.5)
        @test itp_ext(aq_above) ≈ itp_ext(1.5)

        # Constant mode
        itp_const = quadratic_interp(x, y; extrap=:constant)
        @test itp_const(aq_below) ≈ itp_const(-0.5)
        @test itp_const(aq_above) ≈ itp_const(1.5)
    end

    # ========================================
    # Domain Boundary Detection Tests
    # ========================================
    @testset "domain boundary detection (side field)" begin
        x = collect(range(0.0, 1.0, 11))

        # Inside domain
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:quadratic))
        @test aq_inside.side == 0x00

        # Below domain
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:quadratic))
        @test aq_below.side == 0x01

        # Above domain
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic))
        @test aq_above.side == 0x02
    end

    # ========================================
    # Type Promotion Tests
    # ========================================
    @testset "type promotion Real → Float" begin
        x = collect(range(0.0, 1.0, 11))

        # Int query should be promoted
        aq_int = FastInterpolations._anchor_query(x, 0, Val(:quadratic))
        @test aq_int.xq isa Float64
        @test aq_int.xq ≈ 0.0

        # Rational query should be promoted
        aq_rat = FastInterpolations._anchor_query(x, 1//2, Val(:quadratic))
        @test aq_rat.xq isa Float64
        @test aq_rat.xq ≈ 0.5
    end

    # ========================================
    # Float32 Support Tests
    # ========================================
    @testset "Float32 support" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        xq = 0.35f0

        aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
        @test aq isa FastInterpolations._QuadraticAnchoredQuery{Float32}
        @test aq.xq isa Float32
        @test aq.dL isa Float32
    end

    # ========================================
    # In-Place Vector Evaluation Tests
    # ========================================
    @testset "in-place vector evaluation with anchors" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap=:extension)

        xq_vec = [0.5, 1.0, 2.0, 3.0, 5.5]
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))

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
        y = x .^ 2
        itp = quadratic_interp(x, y; extrap=:extension)

        xq = 0.45  # interval [0.3, 0.6]
        aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))

        @test aq.idx == 3  # interval [0.3, 0.6]
        @test aq.xq == 0.45
        @test aq.dL ≈ 0.15  # 0.45 - 0.3

        # Verify evaluation matches
        @test itp(aq) ≈ itp(xq)
    end

    # ========================================
    # Zero-Allocation Test
    # ========================================
    @testset "zero-allocation with pre-built anchors" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap=:extension)

        xq_vec = collect(range(0.1, 6.0, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec)

        # Allocation test
        allocs = @allocated itp(output, aq_vec)
        @test allocs == 0
    end

    # ========================================
    # BC Mode Tests
    # ========================================
    @testset "different BC modes" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2

        # Test with different BCs
        for bc in [Left(ParabolaFit{Float64}()), Right(ParabolaFit{Float64}()), MinCurvFit{Float64}()]
            itp = quadratic_interp(x, y; bc=bc, extrap=:extension)
            xq = 0.35
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq) ≈ itp(xq)
        end
    end

end
