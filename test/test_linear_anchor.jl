# Test suite for Linear Anchored Query functionality
#
# ALLOC_THRESHOLD is defined in test/setup.jl

@testitem "Linear Anchored Query" setup = [AllocConstants] begin

    # ========================================
    # Struct Fields Tests
    # ========================================
    @testset "struct _LinearAnchoredQuery fields" begin
        # Test that struct exists and has correct fields
        x = collect(range(0.0, 1.0, 11))  # 0.0, 0.1, ..., 1.0
        xq = 0.35

        aq = FastInterpolations._anchor_query(x, xq, Val(:linear))

        @test aq isa FastInterpolations._LinearAnchoredQuery{Float64}
        # `idxL` and `idxR` are virtual properties backed by
        # `interval::_ExplicitIndices{2}` — `hasproperty` handles both real and
        # virtual fields.
        @test hasproperty(aq, :idxL)
        @test hasproperty(aq, :idxR)
        @test hasfield(typeof(aq), :interval)
        @test hasfield(typeof(aq), :xq)
        @test hasfield(typeof(aq), :state)
        @test hasfield(typeof(aq), :h)
        @test hasfield(typeof(aq), :inv_h)
        @test hasfield(typeof(aq), :alpha)

        # Verify field types
        @test aq.idxL isa Int
        @test aq.idxR isa Int
        @test aq.xq isa Float64
        @test aq.state isa UInt8
        @test aq.h isa Float64
        @test aq.inv_h isa Float64
        @test aq.alpha isa Float64
    end

    @testset "virtual property type-stability + zero-alloc" begin
        # Pins the Val-dispatched `getproperty` contract: every virtual access
        # (idxL/idxR backed by stencil) and direct field access (alpha/h)
        # must infer to a concrete type and allocate zero bytes inside a
        # function barrier. A regression to single-method `getproperty` with
        # `s === :idxL && return ...` branches would defeat inference (union
        # return) and silently slow hot consumers.
        # `@inferred` requires a call expression — wrap each access in a
        # 1-arg helper so the macro sees a function call.
        _get_idxL(aq) = aq.idxL
        _get_idxR(aq) = aq.idxR
        _get_alpha(aq) = aq.alpha
        _get_h(aq) = aq.h

        x = collect(range(0.0, 1.0, 11))
        aq = FastInterpolations._anchor_query(x, 0.35, Val(:linear))

        @test @inferred(_get_idxL(aq)) isa Int
        @test @inferred(_get_idxR(aq)) isa Int
        @test @inferred(_get_alpha(aq)) isa Float64
        @test @inferred(_get_h(aq)) isa Float64

        function _bench_idx(aq)
            return aq.idxL + aq.idxR
        end
        _bench_idx(aq); _bench_idx(aq)  # warmup
        @test (@allocated _bench_idx(aq)) == 0
    end

    # ========================================
    # Scalar Construction Tests
    # ========================================
    @testset "_anchor_query scalar construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1

        # Query inside domain
        aq = FastInterpolations._anchor_query(x, 0.35, Val(:linear))
        @test aq.idxL == 4  # interval [0.3, 0.4]
        @test aq.idxR == 5
        @test aq.xq == 0.35
        @test aq.state == FastInterpolations.IN_DOMAIN  # inside
        @test aq.h ≈ 0.1            # interval width
        @test aq.inv_h ≈ 10.0       # 1/h
        @test aq.alpha ≈ 0.5        # (0.35 - 0.3) / 0.1 = 0.5

        # Query at left boundary
        aq_left = FastInterpolations._anchor_query(x, 0.0, Val(:linear))
        @test aq_left.idxL == 1
        @test aq_left.idxR == 2
        @test aq_left.state == FastInterpolations.IN_DOMAIN  # inside (at boundary)
        @test aq_left.alpha ≈ 0.0   # at left boundary, alpha = 0

        # Query at right boundary
        aq_right = FastInterpolations._anchor_query(x, 1.0, Val(:linear))
        @test aq_right.idxL == 10  # last interval
        @test aq_right.idxR == 11
        @test aq_right.state == FastInterpolations.IN_DOMAIN
        @test aq_right.alpha ≈ 1.0  # (1.0 - 0.9) / 0.1 = 1.0
    end

    # ========================================
    # Vector Construction Tests
    # ========================================
    @testset "_anchor_query vector construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1
        xq_vec = [0.15, 0.35, 0.75]

        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:linear))

        @test length(aq_vec) == 3
        @test aq_vec[1].idxL == 2   # interval [0.1, 0.2]
        @test aq_vec[2].idxL == 4   # interval [0.3, 0.4]
        @test aq_vec[3].idxL == 8   # interval [0.7, 0.8]

        # alpha = (xq - xL) / h = 0.05 / 0.1 = 0.5 for all
        @test aq_vec[1].alpha ≈ 0.5  # (0.15 - 0.1) / 0.1
        @test aq_vec[2].alpha ≈ 0.5  # (0.35 - 0.3) / 0.1
        @test aq_vec[3].alpha ≈ 0.5  # (0.75 - 0.7) / 0.1
    end

    # ========================================
    # Evaluation Tests - Value
    # ========================================
    @testset "itp(aq) evaluation matches itp(xq)" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:linear))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - Derivative
    # ========================================
    @testset "itp(aq; deriv=DerivOp(1)) derivative evaluation" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq_points = [0.5, 1.0, 2.0, 3.0, 5.5]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:linear))
            @test itp(aq; deriv = DerivOp(1)) ≈ itp(xq; deriv = DerivOp(1))
        end
    end

    # ========================================
    # Wrap Mode Tests
    # ========================================
    @testset "wrap mode for extrap=WrapExtrap()" begin
        x = collect(range(0.0, 1.0, 11))  # domain [0, 1]

        # Query outside domain with wrap=true
        aq_wrap = FastInterpolations._anchor_query(x, 1.5, Val(:linear), true)
        @test aq_wrap.xq ≈ 0.5  # 1.5 wraps to 0.5
        @test aq_wrap.state == FastInterpolations.IN_DOMAIN  # inside after wrap

        # Query below domain with wrap=true
        aq_wrap_neg = FastInterpolations._anchor_query(x, -0.3, Val(:linear), true)
        @test aq_wrap_neg.xq ≈ 0.7  # -0.3 + 1.0 = 0.7
        @test aq_wrap_neg.state == FastInterpolations.IN_DOMAIN

        # Verify wrapped evaluation matches
        y = sin.(2π .* x)
        itp = linear_interp(x, y; extrap = WrapExtrap())
        aq = FastInterpolations._anchor_query(x, 1.5, Val(:linear), true)
        @test itp(aq) ≈ itp(1.5)
    end

    # ========================================
    # Domain Boundary Detection Tests
    # ========================================
    @testset "domain boundary detection (state field)" begin
        x = collect(range(0.0, 1.0, 11))

        # Inside domain
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:linear))
        @test aq_inside.state == FastInterpolations.IN_DOMAIN

        # Below domain (no wrap)
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:linear))
        @test aq_below.state == FastInterpolations.OOB_LEFT  # below

        # Above domain (no wrap)
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:linear))
        @test aq_above.state == FastInterpolations.OOB_RIGHT  # above
    end

    # ========================================
    # Type Promotion Tests
    # ========================================
    @testset "type promotion Real → Float" begin
        x = collect(range(0.0, 1.0, 11))

        # Int query should be promoted
        aq_int = FastInterpolations._anchor_query(x, 0, Val(:linear))
        @test aq_int.xq isa Float64
        @test aq_int.xq ≈ 0.0

        # Rational query should be promoted
        aq_rat = FastInterpolations._anchor_query(x, 1 // 2, Val(:linear))
        @test aq_rat.xq isa Float64
        @test aq_rat.xq ≈ 0.5
    end

    # ========================================
    # Float32 Support Tests
    # ========================================
    @testset "Float32 support" begin
        x = collect(range(0.0f0, 1.0f0, 11))
        xq = 0.35f0

        aq = FastInterpolations._anchor_query(x, xq, Val(:linear))
        @test aq isa FastInterpolations._LinearAnchoredQuery{Float32}
        @test aq.xq isa Float32
        @test aq.h isa Float32
        @test aq.inv_h isa Float32
        @test aq.alpha isa Float32
    end

    # ========================================
    # In-Place Vector Evaluation Tests
    # ========================================
    @testset "in-place vector evaluation with anchors" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq_vec = [0.5, 1.0, 2.0, 3.0, 5.5]
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:linear))

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
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq = 0.45  # interval [0.3, 0.6]
        aq = FastInterpolations._anchor_query(x, xq, Val(:linear))

        @test aq.idxL == 3  # interval [0.3, 0.6]
        @test aq.xq == 0.45
        @test aq.h ≈ 0.3           # 0.6 - 0.3 = 0.3
        @test aq.alpha ≈ 0.5       # (0.45 - 0.3) / 0.3 = 0.5

        # Verify evaluation matches
        @test itp(aq) ≈ itp(xq)
    end

    # ========================================
    # Zero-Allocation Test
    # ========================================
    @testset "zero-allocation with pre-built anchors" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq_vec = collect(range(0.1, 6.0, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:linear))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec)

        # Allocation test
        allocs = @allocated itp(output, aq_vec)
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # extrap=NoExtrap() DomainError Tests
    # ========================================
    @testset "extrap=NoExtrap() throws DomainError via anchor" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = linear_interp(x, y; extrap = NoExtrap())

        # Inside domain works
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:linear))
        @test isfinite(itp(aq_inside))

        # Outside domain throws DomainError
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:linear))
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:linear))

        @test_throws DomainError itp(aq_below)
        @test_throws DomainError itp(aq_above)

        # Derivatives also throw
        @test_throws DomainError itp(aq_below; deriv = DerivOp(1))
        @test_throws DomainError itp(aq_above; deriv = DerivOp(1))
    end

    # ========================================
    # extrap=ClampExtrap() Tests
    # ========================================
    @testset "extrap=ClampExtrap() via anchor" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)
        itp = linear_interp(x, y; extrap = ClampExtrap())

        # Below domain returns first y
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:linear))
        @test itp(aq_below) ≈ y[1]
        @test itp(aq_below) ≈ itp(-0.5)

        # Above domain returns last y.
        # The scalar path clamps the coordinate to last(x) then runs the kernel
        # (muladd at α=1), so it equals y[end] only to ≤1 ULP; near a zero-valued
        # endpoint (sin(2π)≈0) that gap is large relative to the value, so compare
        # with an absolute tolerance rather than the default relative ≈.
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:linear))
        @test itp(aq_above) ≈ y[end]
        @test itp(aq_above) ≈ itp(1.5) atol = 1.0e-14

        # Inside domain still interpolates
        aq_mid = FastInterpolations._anchor_query(x, 0.35, Val(:linear))
        @test itp(aq_mid) ≈ itp(0.35)
    end

    # ========================================
    # Vector Anchor with extrap modes
    # ========================================
    @testset "vector anchor with different extrap modes" begin
        x = collect(range(0.0, 1.0, 11))
        y = sin.(2π .* x)

        for extrap in [ExtendExtrap(), ClampExtrap()]
            itp = linear_interp(x, y; extrap = extrap)
            xq_vec = [-0.2, 0.3, 0.7, 1.2]  # Mix of inside/outside
            aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:linear))

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
        y = sin.(2π .* x)
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq_vec = [0.2, 0.5, 0.8]
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:linear))

        # Wrong size output throws assertion
        output_wrong = zeros(Float64, 5)
        @test_throws AssertionError itp(output_wrong, aq_vec)
    end

    # ========================================
    # Zero-Allocation with deriv=1
    # ========================================
    @testset "zero-allocation with deriv=DerivOp(1)" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        itp = linear_interp(x, y; extrap = ExtendExtrap())

        xq_vec = collect(range(0.1, 6.0, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:linear))
        output = zeros(Float64, 100)

        # Warm-up call
        itp(output, aq_vec; deriv = DerivOp(1))

        # Allocation test
        allocs = @allocated itp(output, aq_vec; deriv = DerivOp(1))
        @test allocs <= ALLOC_THRESHOLD
    end

    # ========================================
    # _fill_anchors! In-Place API
    # ========================================

    @testset "_fill_anchors! in-place" begin
        FI = FastInterpolations

        @testset "fills buffer with correct anchor values" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]

            # Reference: allocating version
            expected = FI._anchor_query(x, xq, Val(:linear))

            # In-place version - now uses {Tg, Tq} type parameters
            buffer = Vector{FI._LinearAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:linear))

            # Verify all fields match exactly (bit-wise)
            for i in eachindex(xq)
                @test buffer[i].idxL == expected[i].idxL
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
                @test buffer[i].h == expected[i].h
                @test buffer[i].inv_h == expected[i].inv_h
                @test buffer[i].alpha == expected[i].alpha
            end
        end

        @testset "wrap mode works correctly" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [-0.3, 0.5, 1.3, 2.5]

            expected = FI._anchor_query(x, xq, Val(:linear), true)
            buffer = Vector{FI._LinearAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:linear), true)

            for i in eachindex(xq)
                @test buffer[i].idxL == expected[i].idxL
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
            end
        end

        @testset "length assertion when buffer too small" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.15, 0.35, 0.5, 0.75]
            buffer = Vector{FI._LinearAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, 2)

            @test_throws AssertionError FI._fill_anchors!(buffer, x, xq, Val(:linear))
        end

        @testset "zero allocation after warmup" begin
            x = collect(range(0.0, 1.0, 101))
            xq = collect(range(0.1, 0.9, 50))
            buffer = Vector{FI._LinearAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, length(xq))

            FI._fill_anchors!(buffer, x, xq, Val(:linear))
            allocs = @allocated FI._fill_anchors!(buffer, x, xq, Val(:linear))
            @test allocs <= ALLOC_THRESHOLD
        end
    end

end
