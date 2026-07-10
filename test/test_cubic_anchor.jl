# ALLOC_THRESHOLD is defined in test/setup.jl

@testitem "Cubic Anchored Query" setup = [AllocConstants] begin
    FI = FastInterpolations

    # ========================================
    # Phase 1: Anchored Evaluation
    # ========================================

    @testset "Anchored evaluation - exact agreement" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        # Test multiple query points
        for xq in [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]
            aq = FI._anchor_query(x, xq, Val(:cubic))
            @test itp(aq) ≈ itp(xq) atol = 1.0e-14
        end
    end

    @testset "Anchored evaluation - derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        for xq in [0.15, 0.5, 0.85]
            aq = FI._anchor_query(x, xq, Val(:cubic))

            # First derivative (1e-13: anchor precomputes weights, direct computes
            # on-the-fly — different FMA fusion across JIT sessions → sub-ULP drift)
            @test itp(aq; deriv = DerivOp(1)) ≈ itp(xq; deriv = DerivOp(1)) atol = 1.0e-13

            # Second derivative (same reasoning as above)
            @test itp(aq; deriv = DerivOp(2)) ≈ itp(xq; deriv = DerivOp(2)) atol = 1.0e-13
        end
    end

    @testset "Anchored evaluation - extrap NoExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = NoExtrap())

        # Inside domain should work
        aq_inside = FI._anchor_query(x, 0.5, Val(:cubic))
        @test isfinite(itp(aq_inside))

        # Outside domain should throw DomainError
        aq_below = FI._anchor_query(x, -0.1, Val(:cubic))
        @test_throws DomainError itp(aq_below)

        aq_above = FI._anchor_query(x, 1.1, Val(:cubic))
        @test_throws DomainError itp(aq_above)
    end

    @testset "Anchored evaluation - extrap ClampExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ClampExtrap())

        # Below domain returns y[1]
        aq_below = FI._anchor_query(x, -0.5, Val(:cubic))
        @test itp(aq_below) ≈ y[1]

        # Above domain returns y[end]
        aq_above = FI._anchor_query(x, 1.5, Val(:cubic))
        @test itp(aq_above) ≈ y[end]

        # Derivatives of constant are zero
        @test itp(aq_below; deriv = DerivOp(1)) == 0.0
        @test itp(aq_above; deriv = DerivOp(2)) == 0.0
    end

    @testset "Anchored evaluation - extrap ExtendExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        # Outside domain uses boundary polynomial
        aq_below = FI._anchor_query(x, -0.1, Val(:cubic))
        @test itp(aq_below) ≈ itp(-0.1) atol = 1.0e-14

        aq_above = FI._anchor_query(x, 1.1, Val(:cubic))
        @test itp(aq_above) ≈ itp(1.1) atol = 1.0e-14
    end

    @testset "Anchored evaluation - extrap WrapExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = WrapExtrap())

        # Wrap extrapolation - anchor must be created with wrap=true
        # to pre-wrap coordinates for :wrap mode
        aq_above = FI._anchor_query(x, 1.3, Val(:cubic), true)
        wrapped_xq = mod(1.3, 1.0)  # 0.3
        @test itp(aq_above) ≈ itp(wrapped_xq) atol = 1.0e-14
    end

    @testset "Anchored evaluation - PeriodicBC with wrap" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        itp = cubic_interp(x, y; bc = PeriodicBC())

        # Wrap at anchor construction (for extrap=WrapExtrap() mode)
        aq_wrapped = FI._anchor_query(x, 2π + 1.0, Val(:cubic), true)
        @test itp(aq_wrapped) ≈ itp(1.0) atol = 1.0e-10
    end

    @testset "Anchored evaluation - Float32" begin
        x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
        y32 = sin.(Float32(2π) .* x32)
        itp = cubic_interp(x32, y32; extrap = ExtendExtrap())

        xq = Float32(0.35)
        aq = FI._anchor_query(x32, xq, Val(:cubic))

        @test itp(aq) isa Float32
        @test itp(aq) ≈ itp(xq) atol = 1.0f-6
    end

    @testset "Anchored evaluation - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())
        aq = FI._anchor_query(x, 0.5, Val(:cubic))

        # Warmup
        itp(aq)

        # Measure allocation
        allocs = @allocated itp(aq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Multi-interpolant use case" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-3 .* x)

        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        # Create anchor once
        aq = FI._anchor_query(x, 0.35, Val(:cubic))

        # All three should work with same anchor
        @test itp1(aq) ≈ itp1(0.35) atol = 1.0e-14
        @test itp2(aq) ≈ itp2(0.35) atol = 1.0e-14
        @test itp3(aq) ≈ itp3(0.35) atol = 1.0e-14
    end

    @testset "_CubicAnchoredQuery struct" begin
        x = collect(range(0.0, 1.0, 101))
        xq = 0.35

        # Basic construction
        aq = FI._anchor_query(x, xq, Val(:cubic))
        @test aq isa FI._CubicAnchoredQuery{Float64, Float64}

        # Float32 support
        x32 = Float32.(x)
        xq32 = Float32(0.35)
        aq32 = FI._anchor_query(x32, xq32, Val(:cubic))
        @test aq32 isa FI._CubicAnchoredQuery{Float32, Float32}
    end

    @testset "Anchor idx field" begin
        x = collect(range(0.0, 1.0, 11))  # 10 intervals, h=0.1

        # Interior points
        aq1 = FI._anchor_query(x, 0.05, Val(:cubic))
        @test aq1.idx == 1  # [0.0, 0.1)

        aq2 = FI._anchor_query(x, 0.15, Val(:cubic))
        @test aq2.idx == 2  # [0.1, 0.2)

        aq3 = FI._anchor_query(x, 0.95, Val(:cubic))
        @test aq3.idx == 10  # [0.9, 1.0]

        # At grid points
        aq_left = FI._anchor_query(x, 0.0, Val(:cubic))
        @test aq_left.idx == 1

        aq_right = FI._anchor_query(x, 1.0, Val(:cubic))
        @test aq_right.idx == 10  # last interval
    end

    @testset "Anchor state field" begin
        x = collect(range(0.0, 1.0, 11))

        # Interior: side = 0
        aq_inside = FI._anchor_query(x, 0.5, Val(:cubic))
        @test aq_inside.state == FI.IN_DOMAIN

        # At boundaries: side = 0 (still inside domain)
        aq_left_bound = FI._anchor_query(x, 0.0, Val(:cubic))
        @test aq_left_bound.state == FI.IN_DOMAIN

        aq_right_bound = FI._anchor_query(x, 1.0, Val(:cubic))
        @test aq_right_bound.state == FI.IN_DOMAIN

        # Below minimum: side = 1
        aq_below = FI._anchor_query(x, -0.5, Val(:cubic))
        @test aq_below.state == FI.OOB_LEFT

        # Above maximum: side = 2
        aq_above = FI._anchor_query(x, 1.5, Val(:cubic))
        @test aq_above.state == FI.OOB_RIGHT
    end

    @testset "Anchor xq field preserved" begin
        x = collect(range(0.0, 1.0, 101))

        xq_values = [0.0, 0.35, 0.5, 1.0, -0.5, 1.5]
        for xq in xq_values
            aq = FI._anchor_query(x, xq, Val(:cubic))
            @test aq.xq == xq
        end
    end

    @testset "type promotion Real → Float" begin
        x = collect(range(0.0, 1.0, 11))

        # Int query should be promoted
        aq_int = FI._anchor_query(x, 0, Val(:cubic))
        @test aq_int.xq isa Float64
        @test aq_int.xq ≈ 0.0

        # Rational query should be promoted
        aq_rat = FI._anchor_query(x, 1 // 2, Val(:cubic))
        @test aq_rat.xq isa Float64
        @test aq_rat.xq ≈ 0.5
    end

    @testset "Anchor weights tuple" begin
        x = collect(range(0.0, 1.0, 101))
        xq = 0.35

        aq = FI._anchor_query(x, xq, Val(:cubic))
        @test aq.w0 isa NTuple{4, Float64}
        @test aq.w1 isa NTuple{4, Float64}
        @test aq.w2 isa NTuple{2, Float64}  # Optimized: only (wzL, wzR)
        @test aq.w3 isa NTuple{2, Float64}  # Optimized: only (wzL, wzR)

        aq32 = FI._anchor_query(Float32.(x), Float32(xq), Val(:cubic))
        @test aq32.w0 isa NTuple{4, Float32}
        @test aq32.w1 isa NTuple{4, Float32}
        @test aq32.w2 isa NTuple{2, Float32}  # Optimized: only (wzL, wzR)
        @test aq32.w3 isa NTuple{2, Float32}  # Optimized: only (wzL, wzR)
    end

    @testset "Anchor wrapping (wrap=true)" begin
        x = collect(range(0.0, 2π, 101))

        # Query outside domain with wrap=true should wrap
        xq_outside = 2π + 1.0  # wraps to ~1.0
        aq_wrapped = FI._anchor_query(x, xq_outside, Val(:cubic), true)

        # Should be inside after wrapping
        @test aq_wrapped.state == FI.IN_DOMAIN
        @test aq_wrapped.xq != xq_outside  # xq is wrapped value

        # Without wrap, should be outside
        aq_nowrap = FI._anchor_query(x, xq_outside, Val(:cubic), false)
        @test aq_nowrap.state == FI.OOB_RIGHT  # above max
    end

    @testset "Invalid deriv argument" begin
        x = collect(range(0.0, 1.0, 101))
        aq = FI._anchor_query(x, 0.5, Val(:cubic))
        itp = cubic_interp(x, sin.(2π .* x))
        @test_throws TypeError itp(aq; deriv = -1)
        @test_throws TypeError itp(aq; deriv = 4)
    end

    # ========================================
    # Vector Anchored Queries
    # ========================================

    @testset "Vector anchor construction" begin
        x = collect(range(0.0, 1.0, 101))
        xq = [0.15, 0.35, 0.5, 0.75]

        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        @test length(aq_vec) == 4
        @test all(aq -> aq isa FI._CubicAnchoredQuery{Float64, Float64}, aq_vec)
        @test all(aq -> aq.state == FI.IN_DOMAIN, aq_vec)  # All inside domain
    end

    @testset "Vector anchor - type promotion" begin
        x = collect(range(0.0, 1.0, 101))
        xq_f32 = Float32[0.15, 0.35, 0.5]
        aq_vec = FI._anchor_query(x, xq_f32, Val(:cubic))

        @test all(aq -> aq isa FI._CubicAnchoredQuery{Float64, Float64}, aq_vec)
    end

    @testset "Vector anchor - wrap=true" begin
        x = collect(range(0.0, 1.0, 101))
        xq = [-0.3, 1.3, 2.5]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic), true)

        @test all(aq -> aq.state == FI.IN_DOMAIN, aq_vec)  # All wrapped to inside
    end

    @testset "Vector anchor - empty input" begin
        x = collect(range(0.0, 1.0, 101))
        aq_vec = FI._anchor_query(x, Float64[], Val(:cubic))

        @test length(aq_vec) == 0
    end

    # ========================================
    # Vector Anchored Query Evaluation
    # ========================================

    @testset "Vector evaluation - allocating" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        xq = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        vals = itp(aq_vec)

        @test length(vals) == length(xq)
        for (i, xq_i) in enumerate(xq)
            @test vals[i] ≈ itp(xq_i) atol = 1.0e-14
        end
    end

    @testset "Vector evaluation - extrap NoExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        itp = cubic_interp(x, sin.(x); extrap = NoExtrap())

        xq = [-0.1, 0.5, 1.1]  # First is out of domain
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        @test_throws DomainError itp(aq_vec)
    end

    @testset "Vector evaluation - extrap ClampExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ClampExtrap())

        xq = [-0.5, 0.5, 1.5]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        vals = itp(aq_vec)

        @test vals[1] ≈ y[1]
        @test vals[2] ≈ itp(0.5)
        @test vals[3] ≈ y[end]
    end

    @testset "Vector evaluation - extrap ExtendExtrap()" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        xq = [-0.1, 0.5, 1.1]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        vals = itp(aq_vec)

        for (i, xq_i) in enumerate(xq)
            @test vals[i] ≈ itp(xq_i) atol = 1.0e-14
        end
    end

    @testset "Vector evaluation - in-place" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        xq = [0.15, 0.35, 0.5, 0.75]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        output = Vector{Float64}(undef, 4)

        result = itp(output, aq_vec)

        @test result === output
        for (i, xq_i) in enumerate(xq)
            @test output[i] ≈ itp(xq_i) atol = 1.0e-14
        end
    end

    @testset "Vector evaluation - derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        xq = [0.15, 0.5, 0.85]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        for d in [DerivOp(1), DerivOp(2)]
            vals = itp(aq_vec; deriv = d)
            for (i, xq_i) in enumerate(xq)
                @test vals[i] ≈ itp(xq_i; deriv = d) atol = 1.0e-14
            end
        end
    end

    @testset "Vector evaluation - length mismatch" begin
        x = collect(range(0.0, 1.0, 101))
        itp = cubic_interp(x, sin.(x))

        aq_vec = FI._anchor_query(x, [0.15, 0.35, 0.5], Val(:cubic))
        output = Vector{Float64}(undef, 2)  # Wrong size

        @test_throws AssertionError itp(output, aq_vec)
    end

    @testset "Vector evaluation - extrap ClampExtrap() derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ClampExtrap())

        xq = [-0.5, 0.5, 1.5]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        derivs = itp(aq_vec; deriv = DerivOp(1))
        @test derivs[1] == 0.0
        @test derivs[3] == 0.0
    end

    @testset "Vector evaluation - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        itp = cubic_interp(x, sin.(x))

        xq = collect(range(0.1, 0.9, 50))
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        output = similar(xq)

        # Warmup
        itp(output, aq_vec)

        allocs = @allocated itp(output, aq_vec)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Vector evaluation - multi-interpolant reuse" begin
        x = collect(range(0.0, 1.0, 101))

        itp1 = cubic_interp(x, sin.(2π .* x))
        itp2 = cubic_interp(x, cos.(2π .* x))
        itp3 = cubic_interp(x, exp.(-3 .* x))

        xq = [0.15, 0.35, 0.5, 0.75]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))  # Anchors computed once

        @test itp1(aq_vec) ≈ [itp1(q) for q in xq] atol = 1.0e-14
        @test itp2(aq_vec) ≈ [itp2(q) for q in xq] atol = 1.0e-14
        @test itp3(aq_vec) ≈ [itp3(q) for q in xq] atol = 1.0e-14
    end

    @testset "Vector evaluation - wrap=true" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = WrapExtrap())

        xq = [-0.3, 1.3, 2.5]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic), true)
        vals = itp(aq_vec)

        for (i, xq_i) in enumerate(xq)
            wrapped = mod(xq_i, 1.0)
            @test vals[i] ≈ itp(wrapped) atol = 1.0e-14
        end
    end

    # ========================================
    # Zero-Allocation Tests for Derivatives
    # ========================================

    @testset "Scalar anchored derivatives - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())
        aq = FI._anchor_query(x, 0.5, Val(:cubic))

        # Warmup all derivative orders
        itp(aq)
        itp(aq; deriv = DerivOp(1))
        itp(aq; deriv = DerivOp(2))

        # Value (deriv=DerivOp(0)) - already tested, but include for completeness
        allocs_d0 = @allocated itp(aq)
        @test allocs_d0 <= ALLOC_THRESHOLD

        # First derivative
        allocs_d1 = @allocated itp(aq; deriv = DerivOp(1))
        @test allocs_d1 <= ALLOC_THRESHOLD

        # Second derivative
        allocs_d2 = @allocated itp(aq; deriv = DerivOp(2))
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    @testset "Vector in-place derivatives - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap = ExtendExtrap())

        xq = collect(range(0.1, 0.9, 50))
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        output = similar(xq)

        # Warmup all derivative orders
        itp(output, aq_vec)
        itp(output, aq_vec; deriv = DerivOp(1))
        itp(output, aq_vec; deriv = DerivOp(2))

        # Value (deriv=DerivOp(0))
        allocs_d0 = @allocated itp(output, aq_vec)
        @test allocs_d0 <= ALLOC_THRESHOLD

        # First derivative - zero allocation
        allocs_d1 = @allocated itp(output, aq_vec; deriv = DerivOp(1))
        @test allocs_d1 <= ALLOC_THRESHOLD

        # Second derivative - zero allocation
        allocs_d2 = @allocated itp(output, aq_vec; deriv = DerivOp(2))
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    # ========================================
    # _fill_anchors! In-Place API
    # ========================================

    @testset "_fill_anchors! in-place" begin
        @testset "fills buffer with correct anchor values" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]

            # Reference: allocating version
            expected = FI._anchor_query(x, xq, Val(:cubic))

            # In-place version
            buffer = Vector{FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:cubic))

            # Verify all fields match exactly (bit-wise)
            for i in eachindex(xq)
                @test buffer[i].idx == expected[i].idx
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
                @test buffer[i].w0 == expected[i].w0
                @test buffer[i].w1 == expected[i].w1
                @test buffer[i].w2 == expected[i].w2
            end
        end

        @testset "wrap mode works correctly" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [-0.3, 0.5, 1.3, 2.5]  # First and last two outside domain

            # Reference
            expected = FI._anchor_query(x, xq, Val(:cubic), true)

            # In-place
            buffer = Vector{FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:cubic), true)

            for i in eachindex(xq)
                @test buffer[i].idx == expected[i].idx
                @test buffer[i].xq == expected[i].xq
                @test buffer[i].state == expected[i].state
                @test buffer[i].w0 == expected[i].w0
            end
        end

        @testset "length assertion when buffer too small" begin
            x = collect(range(0.0, 1.0, 101))
            xq = [0.15, 0.35, 0.5, 0.75]  # 4 points
            buffer = Vector{FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, 2)  # Only 2 slots

            @test_throws AssertionError FI._fill_anchors!(buffer, x, xq, Val(:cubic))
        end

        @testset "empty vector case" begin
            x = collect(range(0.0, 1.0, 101))
            xq = Float64[]
            buffer = Vector{FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, 0)

            # Should not throw
            FI._fill_anchors!(buffer, x, xq, Val(:cubic))
            @test length(buffer) == 0
        end

        @testset "type matching (Float32 grid and queries)" begin
            x = Float32.(collect(range(0.0f0, 1.0f0, 101)))
            xq = Float32[0.15f0, 0.35f0, 0.5f0]  # Float32 queries

            buffer = Vector{FI._CubicAnchoredQuery{Float32, Float32, FI._ContiguousIndices{2}}}(undef, length(xq))
            FI._fill_anchors!(buffer, x, xq, Val(:cubic))

            @test all(aq -> aq isa FI._CubicAnchoredQuery{Float32, Float32}, buffer)
        end

        @testset "zero allocation after warmup" begin
            x = collect(range(0.0, 1.0, 101))
            xq = collect(range(0.1, 0.9, 50))
            buffer = Vector{FI._CubicAnchoredQuery{Float64, Float64, FI._ContiguousIndices{2}}}(undef, length(xq))

            # Warmup
            FI._fill_anchors!(buffer, x, xq, Val(:cubic))

            # Measure
            allocs = @allocated FI._fill_anchors!(buffer, x, xq, Val(:cubic))
            @test allocs <= ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Phase 1: Cubic Weight Optimization (RED - Failing Tests)
    # ========================================
    # These tests encode the expected behavior after optimization:
    # - w2 and w3 should be NTuple{2,T} (only wzL, wzR needed)
    # - Deriv2/3 evaluation should remain correct
    # - Type stability must be preserved

    @testset "Cubic Weight Optimization - Reduced Weights" begin
        # Test 1.1: Anchor weight tuple shapes (w2/w3 are 2-tuples)
        @testset "w2 and w3 reduced to 2-tuples" begin
            x = collect(range(0.0, 1.0, 101))
            xq = 0.35

            aq = FI._anchor_query(x, xq, Val(:cubic))

            # After optimization, w2 and w3 should only store (wzL, wzR)
            # These will FAIL until Phase 2 implementation
            @test length(aq.w2) == 2  # Expected: 2, Current: 4
            @test length(aq.w3) == 2  # Expected: 2, Current: 4

            # w0 and w1 should remain 4-tuples (they need all weights)
            @test length(aq.w0) == 4
            @test length(aq.w1) == 4

            # Float32 should work the same way
            aq32 = FI._anchor_query(Float32.(x), Float32(xq), Val(:cubic))
            @test length(aq32.w2) == 2
            @test length(aq32.w3) == 2
            @test aq32.w2 isa NTuple{2, Float32}
            @test aq32.w3 isa NTuple{2, Float32}
        end

        # Test 1.2: Anchored deriv2/3 evaluation correctness
        @testset "Deriv2/3 correctness with anchored query" begin
            x = collect(range(0.0, 1.0, 101))
            y = sin.(2π .* x)
            itp = cubic_interp(x, y; extrap = ExtendExtrap())

            # Test multiple query points including boundaries and interior
            test_points = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]

            for xq in test_points
                aq = FI._anchor_query(x, xq, Val(:cubic))

                # Second derivative
                anchored_d2 = itp(aq; deriv = DerivOp(2))
                direct_d2 = itp(xq; deriv = DerivOp(2))
                @test anchored_d2 ≈ direct_d2 atol = 1.0e-14

                # Third derivative
                anchored_d3 = itp(aq; deriv = DerivOp(3))
                direct_d3 = itp(xq; deriv = DerivOp(3))
                @test anchored_d3 ≈ direct_d3 atol = 1.0e-12
            end
        end

        # Test 1.3: Type stability for anchored deriv2/3 entry points
        @testset "Type stability for deriv2/3" begin
            x = collect(range(0.0, 1.0, 101))
            y = sin.(2π .* x)
            itp = cubic_interp(x, y; extrap = ExtendExtrap())
            aq = FI._anchor_query(x, 0.5, Val(:cubic))

            # Scalar anchored evaluation should be type-stable
            @test (@inferred itp(aq; deriv = DerivOp(2))) isa Float64
            @test (@inferred itp(aq; deriv = DerivOp(3))) isa Float64

            # Vector evaluation should also be type-stable
            xq_vec = [0.15, 0.5, 0.85]
            aq_vec = FI._anchor_query(x, xq_vec, Val(:cubic))

            d2_vec = @inferred itp(aq_vec; deriv = DerivOp(2))
            @test d2_vec isa Vector{Float64}
            @test length(d2_vec) == 3

            d3_vec = @inferred itp(aq_vec; deriv = DerivOp(3))
            @test d3_vec isa Vector{Float64}
            @test length(d3_vec) == 3
        end

        # Additional test: Extrapolation modes with deriv2/3
        @testset "Deriv2/3 with extrapolation modes" begin
            x = collect(range(0.0, 1.0, 101))
            y = sin.(2π .* x)

            # Test :extension mode
            itp_ext = cubic_interp(x, y; extrap = ExtendExtrap())
            aq_below = FI._anchor_query(x, -0.1, Val(:cubic))
            aq_above = FI._anchor_query(x, 1.1, Val(:cubic))

            @test itp_ext(aq_below; deriv = DerivOp(2)) ≈ itp_ext(-0.1; deriv = DerivOp(2)) atol = 1.0e-12
            @test itp_ext(aq_above; deriv = DerivOp(3)) ≈ itp_ext(1.1; deriv = DerivOp(3)) atol = 1.0e-12

            # Test :constant mode (derivatives should be zero outside domain)
            itp_const = cubic_interp(x, y; extrap = ClampExtrap())
            @test itp_const(aq_below; deriv = DerivOp(2)) == 0.0
            @test itp_const(aq_above; deriv = DerivOp(3)) == 0.0
        end

        # Additional test: Zero allocation for deriv2/3
        @testset "Zero allocation for deriv2/3 anchored evaluation" begin
            x = collect(range(0.0, 1.0, 101))
            y = sin.(2π .* x)
            itp = cubic_interp(x, y; extrap = ExtendExtrap())
            aq = FI._anchor_query(x, 0.5, Val(:cubic))

            # Warmup
            itp(aq; deriv = DerivOp(2))
            itp(aq; deriv = DerivOp(3))

            # Third derivative should have zero allocation
            allocs_d3 = @allocated itp(aq; deriv = DerivOp(3))
            @test allocs_d3 <= ALLOC_THRESHOLD

            # Vector in-place should also have zero allocation
            xq_vec = collect(range(0.1, 0.9, 50))
            aq_vec = FI._anchor_query(x, xq_vec, Val(:cubic))
            output = similar(xq_vec)

            # Warmup
            itp(output, aq_vec; deriv = DerivOp(3))

            # Measure
            allocs_vec_d3 = @allocated itp(output, aq_vec; deriv = DerivOp(3))
            @test allocs_vec_d3 <= ALLOC_THRESHOLD
        end
    end

end
