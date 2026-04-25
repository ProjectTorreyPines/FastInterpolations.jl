# Test suite for Constant Anchored Query functionality
#
# ALLOC_THRESHOLD is defined in runtests.jl

@testitem "Constant Anchored Query" setup = [AllocConstants] begin

    # ========================================
    # Struct Fields Tests
    # ========================================
    @testset "struct _ConstantAnchoredQuery fields" begin
        x = collect(range(0.0, 1.0, 11))  # 0.0, 0.1, ..., 1.0
        xq = 0.35

        aq = FastInterpolations._anchor_query(x, xq, Val(:constant))

        @test aq isa FastInterpolations._ConstantAnchoredQuery{Float64}
        # `idxL` and `idxR` are virtual properties backed by `stencil::_IdxStencil{2}`
        # since the _IdxStencil migration — `hasproperty` handles both real and
        # virtual fields.
        @test hasproperty(aq, :idxL)
        @test hasproperty(aq, :idxR)
        @test hasfield(typeof(aq), :stencil)
        @test hasfield(typeof(aq), :xq)
        @test hasfield(typeof(aq), :state)
        @test hasfield(typeof(aq), :h)
        @test hasfield(typeof(aq), :dL)

        # Verify field types
        @test aq.idxL isa Int
        @test aq.idxR isa Int
        @test aq.xq isa Float64
        @test aq.state isa UInt8
        @test aq.h isa Float64
        @test aq.dL isa Float64
    end

    @testset "virtual property type-stability + zero-alloc" begin
        # Pins the Val-dispatched `getproperty` contract: every virtual access
        # (idxL/idxR backed by stencil) and direct field access (h/dL) must
        # infer to a concrete type and allocate zero bytes inside a function
        # barrier. A regression to single-method `getproperty` with
        # `s === :idxL && return ...` branches would defeat inference and
        # silently slow hot consumers.
        # `@inferred` requires a call expression — wrap each access in a
        # 1-arg helper so the macro sees a function call.
        _get_idxL(aq) = aq.idxL
        _get_idxR(aq) = aq.idxR
        _get_h(aq) = aq.h
        _get_dL(aq) = aq.dL

        x = collect(range(0.0, 1.0, 11))
        aq = FastInterpolations._anchor_query(x, 0.35, Val(:constant))

        @test @inferred(_get_idxL(aq)) isa Int
        @test @inferred(_get_idxR(aq)) isa Int
        @test @inferred(_get_h(aq)) isa Float64
        @test @inferred(_get_dL(aq)) isa Float64

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
        aq = FastInterpolations._anchor_query(x, 0.35, Val(:constant))
        @test aq.idxL == 4  # interval [0.3, 0.4]
        @test aq.idxR == 5
        @test aq.xq == 0.35
        @test aq.state == FastInterpolations.IN_DOMAIN  # inside
        @test aq.h ≈ 0.1
        @test aq.dL ≈ 0.05  # 0.35 - 0.3

        # Query at left boundary
        aq_left = FastInterpolations._anchor_query(x, 0.0, Val(:constant))
        @test aq_left.idxL == 1
        @test aq_left.idxR == 2
        @test aq_left.state == FastInterpolations.IN_DOMAIN
        @test aq_left.dL ≈ 0.0

        # Query at right boundary
        aq_right = FastInterpolations._anchor_query(x, 1.0, Val(:constant))
        @test aq_right.state == FastInterpolations.IN_DOMAIN
    end

    # ========================================
    # Vector Construction Tests
    # ========================================
    @testset "_anchor_query vector construction" begin
        x = collect(range(0.0, 1.0, 11))  # h = 0.1
        xq_vec = [0.15, 0.35, 0.75]

        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))

        @test length(aq_vec) == 3
        @test aq_vec[1].idxL == 2   # interval [0.1, 0.2]
        @test aq_vec[2].idxL == 4   # interval [0.3, 0.4]
        @test aq_vec[3].idxL == 8   # interval [0.7, 0.8]

        @test all(aq.h ≈ 0.1 for aq in aq_vec)
    end

    # ========================================
    # Evaluation Tests - NearestSide() mode
    # ========================================
    @testset "itp(aq) for NearestSide() mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; side = NearestSide(), extrap = ExtendExtrap())

        xq_points = [0.05, 0.15, 0.35, 0.65, 0.95]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:constant))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - LeftSide() mode
    # ========================================
    @testset "itp(aq) for LeftSide() mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; side = LeftSide(), extrap = ExtendExtrap())

        xq_points = [0.05, 0.15, 0.35, 0.65, 0.95]

        for xq in xq_points
            aq = FastInterpolations._anchor_query(x, xq, Val(:constant))
            @test itp(aq) ≈ itp(xq)
        end
    end

    # ========================================
    # Evaluation Tests - RightSide() mode
    # ========================================
    @testset "itp(aq) for RightSide() mode" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; side = RightSide(), extrap = ExtendExtrap())

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
        @test aq_inside.state == FastInterpolations.IN_DOMAIN

        # Below domain
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:constant))
        @test aq_below.state == FastInterpolations.OOB_LEFT

        # Above domain
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:constant))
        @test aq_above.state == FastInterpolations.OOB_RIGHT

        # Verify extrapolation works with anchors
        y = collect(1.0:11.0)
        itp_ext = constant_interp(x, y; extrap = ExtendExtrap())
        itp_const = constant_interp(x, y; extrap = ClampExtrap())

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
        itp = constant_interp(x, y; extrap = WrapExtrap())

        # Query outside domain with wrap=true
        aq_wrap = FastInterpolations._anchor_query(x, 1.5, Val(:constant), true)
        @test aq_wrap.xq ≈ 0.5
        @test aq_wrap.state == FastInterpolations.IN_DOMAIN

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
        itp = constant_interp(x, y; extrap = ExtendExtrap())

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
        itp = constant_interp(x, y; extrap = ExtendExtrap())

        xq = 0.45  # interval [0.3, 0.6]
        aq = FastInterpolations._anchor_query(x, xq, Val(:constant))

        @test aq.idxL == 3  # interval [0.3, 0.6]
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
        itp = constant_interp(x, y; extrap = ExtendExtrap())

        xq_vec = collect(range(0.01, 0.99, 100))
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))
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
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; extrap = NoExtrap())

        # Inside domain works
        aq_inside = FastInterpolations._anchor_query(x, 0.5, Val(:constant))
        @test isfinite(itp(aq_inside))

        # At boundary works
        aq_boundary = FastInterpolations._anchor_query(x, 1.0, Val(:constant))
        @test itp(aq_boundary) ≈ y[end]

        # Outside domain throws DomainError
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:constant))
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:constant))

        @test_throws DomainError itp(aq_below)
        @test_throws DomainError itp(aq_above)
    end

    # ========================================
    # extrap=ClampExtrap() Tests
    # ========================================
    @testset "extrap=ClampExtrap() via anchor with boundary check" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; extrap = ClampExtrap())

        # At exact right boundary
        aq_right = FastInterpolations._anchor_query(x, 1.0, Val(:constant))
        @test itp(aq_right) ≈ y[end]

        # Below domain returns first y
        aq_below = FastInterpolations._anchor_query(x, -0.5, Val(:constant))
        @test itp(aq_below) ≈ y[1]

        # Above domain returns last y
        aq_above = FastInterpolations._anchor_query(x, 1.5, Val(:constant))
        @test itp(aq_above) ≈ y[end]
    end

    # ========================================
    # Vector anchor with extrap modes
    # ========================================
    @testset "vector anchor with different extrap modes" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)

        for extrap in [ExtendExtrap(), ClampExtrap()]
            itp = constant_interp(x, y; extrap = extrap)
            xq_vec = [-0.2, 0.3, 0.7, 1.2]  # Mix of inside/outside
            aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))

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
        y = collect(1.0:11.0)
        itp = constant_interp(x, y; extrap = ExtendExtrap())

        xq_vec = [0.2, 0.5, 0.8]
        aq_vec = FastInterpolations._anchor_query(x, xq_vec, Val(:constant))

        # Wrong size output throws assertion
        output_wrong = zeros(Float64, 5)
        @test_throws AssertionError itp(output_wrong, aq_vec)
    end

    # ========================================
    # Boundary handling - exact x_max via anchor
    # ========================================
    @testset "boundary handling at x_max via anchor" begin
        x = collect(range(0.0, 1.0, 11))
        y = collect(1.0:11.0)

        # Test all modes at exact boundary
        for mode in [ExtendExtrap(), ClampExtrap()]
            itp = constant_interp(x, y; extrap = mode)
            aq = FastInterpolations._anchor_query(x, 1.0, Val(:constant))
            @test itp(aq) ≈ y[end]
        end
    end

    # ========================================
    # _fill_anchors! In-Place API
    # ========================================

    @testset "_fill_anchors! in-place" begin
        FI = FastInterpolations

        @testset "fills buffer with correct anchor values" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]

            expected = FI._anchor_query(x, xq, Val(:constant))
            buffer = Vector{FI._ConstantAnchoredQuery{Float64, Float64}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:constant))

            for i in eachindex(xq)
                @test buffer[i].idxL == expected[i].idxL
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
                @test buffer[i].h == expected[i].h
                @test buffer[i].dL == expected[i].dL
            end
        end

        @testset "wrap mode works correctly" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [-0.3, 0.5, 1.3, 2.5]

            expected = FI._anchor_query(x, xq, Val(:constant), true)
            buffer = Vector{FI._ConstantAnchoredQuery{Float64, Float64}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:constant), true)

            for i in eachindex(xq)
                @test buffer[i].idxL == expected[i].idxL
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
            end
        end

        @testset "length assertion when buffer too small" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.15, 0.35, 0.5, 0.75]
            buffer = Vector{FI._ConstantAnchoredQuery{Float64, Float64}}(undef, 2)

            @test_throws AssertionError FI._fill_anchors!(buffer, x, xq, Val(:constant))
        end

        @testset "zero allocation after warmup" begin
            x = collect(range(0.0, 1.0, 101))
            xq = collect(range(0.1, 0.9, 50))
            buffer = Vector{FI._ConstantAnchoredQuery{Float64, Float64}}(undef, length(xq))

            FI._fill_anchors!(buffer, x, xq, Val(:constant))
            allocs = @allocated FI._fill_anchors!(buffer, x, xq, Val(:constant))
            @test allocs <= ALLOC_THRESHOLD
        end
    end

end
