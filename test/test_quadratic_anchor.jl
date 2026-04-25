# Test suite for Quadratic Anchored Query functionality
#
# ALLOC_THRESHOLD is defined in runtests.jl

@testitem "Quadratic Anchored Query" setup = [AllocConstants] begin

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
        @test hasfield(typeof(aq), :state)
        @test hasfield(typeof(aq), :dL)

        # Verify field types
        @test aq.idx isa Int
        @test aq.xq isa Float64
        @test aq.state isa UInt8
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
        @test aq.state == FastInterpolations.IN_DOMAIN  # inside
        @test aq.dL ≈ 0.05  # 0.35 - 0.3

        # Query at left boundary
        aq_left = FastInterpolations._anchor_query(x, 0.0, Val(:quadratic))
        @test aq_left.idx == 1
        @test aq_left.state == FastInterpolations.IN_DOMAIN
        @test aq_left.dL ≈ 0.0

        # Query at right boundary
        aq_right = FastInterpolations._anchor_query(x, 1.0, Val(:quadratic))
        @test aq_right.idx == 10  # last interval
        @test aq_right.state == FastInterpolations.IN_DOMAIN
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
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - Derivative 1
    # ========================================
    @testset "itp(aq; deriv=DerivOp(1)) derivative evaluation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq; deriv = DerivOp(1)) ≈ itp(xq; deriv = DerivOp(1))
        end
    end

    # ========================================
    # Evaluation Tests - Derivative 2
    # ========================================
    @testset "itp(aq; deriv=DerivOp(2)) derivative evaluation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq; deriv = DerivOp(2)) ≈ itp(xq; deriv = DerivOp(2))
        end
    end

    # ========================================
    # Extrapolation Mode Tests
    # ========================================
    @testset "extrapolation modes with anchors" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2

        # Extension mode
        itp_ext = quadratic_interp(x, y; extrap = ExtendExtrap())
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:quadratic))
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic))

        @test itp_ext(aq_below) ≈ itp_ext(-0.5)
        @test itp_ext(aq_above) ≈ itp_ext(1.5)

        # Constant mode
        itp_const = quadratic_interp(x, y; extrap = ClampExtrap())
        @test itp_const(aq_below) ≈ itp_const(-0.5)
        @test itp_const(aq_above) ≈ itp_const(1.5)
    end

    # ========================================
    # Domain Boundary Detection Tests
    # ========================================
    @testset "domain boundary detection (state field)" begin
        x = collect(range(0.0, 1.0, 11))

        # Inside domain
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:quadratic))
        @test aq_inside.state == FastInterpolations.IN_DOMAIN

        # Below domain
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:quadratic))
        @test aq_below.state == FastInterpolations.OOB_LEFT

        # Above domain
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic))
        @test aq_above.state == FastInterpolations.OOB_RIGHT
    end

    # ========================================
    # Type Promotion Tests
    # ========================================
    @testset "type preservation Real types" begin
        x = collect(range(0.0, 1.0, 11))

        # Int/Rational queries are widened to match dL type (grid arithmetic promotes)
        # This is consistent with Linear anchor behavior
        aq_int = FastInterpolations._anchor_query(x, 0, Val(:quadratic))
        @test aq_int.xq ≈ 0.0
        @test aq_int isa FastInterpolations._QuadraticAnchoredQuery{Float64, Float64}

        aq_rat = FastInterpolations._anchor_query(x, 1 // 2, Val(:quadratic))
        @test aq_rat.xq ≈ 0.5
        @test aq_rat isa FastInterpolations._QuadraticAnchoredQuery{Float64, Float64}
    end

    # ========================================
    # Float32 Support Tests
    # ========================================
    @testset "Float32 support" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        xq = 0.35f0

        aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
        @test aq isa FastInterpolations._QuadraticAnchoredQuery{Float32, Float32}
        @test aq.xq isa Float32
        @test aq.dL isa Float32
    end

    # ========================================
    # In-Place Vector Evaluation Tests
    # ========================================
    @testset "in-place vector evaluation with anchors" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

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
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

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
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_vec = collect(range(0.1, 6.0, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec)

        # Allocation test
        allocs = @allocated itp(output, aq_vec)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # BC Mode Tests
    # ========================================
    @testset "different BC modes" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2

        # Test with different BCs
        for bc in [Left(QuadraticFit()), Right(QuadraticFit()), MinCurvFit()]
            itp = quadratic_interp(x, y; bc = bc, extrap = ExtendExtrap())
            xq = 0.35
            aq = FastInterpolations._anchor_query(x, xq, Val(:quadratic))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # extrap=NoExtrap() DomainError Tests
    # ========================================
    @testset "extrap=NoExtrap() throws DomainError via anchor" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2
        itp = quadratic_interp(x, y; extrap = NoExtrap())

        # Inside domain works
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:quadratic))
        @test isfinite(itp(aq_inside))

        # Outside domain throws DomainError
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:quadratic))
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic))

        @test_throws DomainError itp(aq_below)
        @test_throws DomainError itp(aq_above)

        # Derivatives also throw
        @test_throws DomainError itp(aq_below; deriv = DerivOp(1))
        @test_throws DomainError itp(aq_above; deriv = DerivOp(2))
    end

    # ========================================
    # extrap=ClampExtrap() Tests
    # ========================================
    @testset "extrap=ClampExtrap() via anchor" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2
        itp = quadratic_interp(x, y; extrap = ClampExtrap())

        # Below domain returns first y
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:quadratic))
        @test itp(aq_below) ≈ y[1]

        # Above domain returns last y
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic))
        @test itp(aq_above) ≈ y[end]

        # Inside domain still interpolates
        aq_mid = FastInterpolations._anchor_query(x, 0.35, Val(:quadratic))
        @test itp(aq_mid) ≈ itp(0.35)
    end

    # ========================================
    # Vector anchor with different extrap modes
    # ========================================
    @testset "vector anchor with different extrap modes" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2

        for extrap in [ExtendExtrap(), ClampExtrap()]
            itp = quadratic_interp(x, y; extrap = extrap)
            xq_vec = [-0.2, 0.3, 0.7, 1.2]  # Mix of inside/outside
            aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))

            result = itp(aq_vec)
            expected = itp(xq_vec)
            @test result ≈ expected
        end
    end

    # ========================================
    # In-place output length assertion
    # ========================================
    @testset "in-place output length assertion" begin
        x = collect(range(0.0, 1.0, 11))
        y = x .^ 2
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_vec = [0.2, 0.5, 0.8]
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))

        # Wrong size output throws assertion
        output_wrong = zeros(Float64, 5)
        @test_throws AssertionError itp(output_wrong, aq_vec)
    end

    # ========================================
    # Zero-Allocation with derivatives
    # ========================================
    @testset "zero-allocation with deriv=DerivOp(1)" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_vec = collect(range(0.1, 6.0, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec; deriv = DerivOp(1))

        # Allocation test
        allocs = @allocated itp(output, aq_vec; deriv = DerivOp(1))
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "zero-allocation with deriv=DerivOp(2)" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = quadratic_interp(x, y; extrap = ExtendExtrap())

        xq_vec = collect(range(0.1, 6.0, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:quadratic))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec; deriv = DerivOp(2))

        # Allocation test
        allocs = @allocated itp(output, aq_vec; deriv = DerivOp(2))
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # Wrap mode for anchor construction
    # ========================================
    @testset "wrap mode anchor construction" begin
        x = collect(range(0.0, 1.0, 11))

        # Query outside domain with wrap=true
        aq_wrap = FastInterpolations._anchor_query(x, 1.5, Val(:quadratic), true)
        @test aq_wrap.xq ≈ 0.5  # 1.5 wraps to 0.5
        @test aq_wrap.state == FastInterpolations.IN_DOMAIN  # inside after wrap

        # Below domain with wrap=true
        aq_wrap_neg = FastInterpolations._anchor_query(x, -0.3, Val(:quadratic), true)
        @test aq_wrap_neg.xq ≈ 0.7  # -0.3 + 1.0 = 0.7
        @test aq_wrap_neg.state == FastInterpolations.IN_DOMAIN
    end

    # ========================================
    # _fill_anchors! In-Place API
    # ========================================

    @testset "_fill_anchors! in-place" begin
        FI = FastInterpolations

        @testset "fills buffer with correct anchor values" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]

            expected = FI._anchor_query(x, xq, Val(:quadratic))
            buffer = Vector{FI._QuadraticAnchoredQuery{Float64, Float64}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:quadratic))

            for i in eachindex(xq)
                @test buffer[i].idx == expected[i].idx
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
                @test buffer[i].dL == expected[i].dL
            end
        end

        @testset "wrap mode works correctly" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [-0.3, 0.5, 1.3, 2.5]

            expected = FI._anchor_query(x, xq, Val(:quadratic), true)
            buffer = Vector{FI._QuadraticAnchoredQuery{Float64, Float64}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:quadratic), true)

            for i in eachindex(xq)
                @test buffer[i].idx == expected[i].idx
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
            end
        end

        @testset "length assertion when buffer too small" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.15, 0.35, 0.5, 0.75]
            buffer = Vector{FI._QuadraticAnchoredQuery{Float64, Float64}}(undef, 2)

            @test_throws AssertionError FI._fill_anchors!(buffer, x, xq, Val(:quadratic))
        end

        @testset "zero allocation after warmup" begin
            x = collect(range(0.0, 1.0, 101))
            xq = collect(range(0.1, 0.9, 50))
            buffer = Vector{FI._QuadraticAnchoredQuery{Float64, Float64}}(undef, length(xq))

            FI._fill_anchors!(buffer, x, xq, Val(:quadratic))
            allocs = @allocated FI._fill_anchors!(buffer, x, xq, Val(:quadratic))
            @test allocs <= ALLOC_THRESHOLD
        end
    end

end
