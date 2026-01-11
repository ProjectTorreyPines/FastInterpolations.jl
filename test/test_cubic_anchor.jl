# ALLOC_THRESHOLD is defined in runtests.jl

@testset "Cubic Anchored Query" begin
    FI = FastInterpolations

    # ========================================
    # Phase 1: Anchored Evaluation
    # ========================================

    @testset "Anchored evaluation - exact agreement" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        # Test multiple query points
        for xq in [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]
            aq = FI._anchor_query(x, xq)
            @test itp(aq) ≈ itp(xq) atol=1e-14
        end
    end

    @testset "Anchored evaluation - derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        for xq in [0.15, 0.5, 0.85]
            aq = FI._anchor_query(x, xq)

            # First derivative
            @test itp(aq; deriv=1) ≈ itp(xq; deriv=1) atol=1e-14

            # Second derivative
            @test itp(aq; deriv=2) ≈ itp(xq; deriv=2) atol=1e-14
        end
    end

    @testset "Anchored evaluation - extrap :none" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:none)

        # Inside domain should work
        aq_inside = FI._anchor_query(x, 0.5)
        @test isfinite(itp(aq_inside))

        # Outside domain should throw DomainError
        aq_below = FI._anchor_query(x, -0.1)
        @test_throws DomainError itp(aq_below)

        aq_above = FI._anchor_query(x, 1.1)
        @test_throws DomainError itp(aq_above)
    end

    @testset "Anchored evaluation - extrap :constant" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:constant)

        # Below domain returns y[1]
        aq_below = FI._anchor_query(x, -0.5)
        @test itp(aq_below) ≈ y[1]

        # Above domain returns y[end]
        aq_above = FI._anchor_query(x, 1.5)
        @test itp(aq_above) ≈ y[end]

        # Derivatives of constant are zero
        @test itp(aq_below; deriv=1) == 0.0
        @test itp(aq_above; deriv=2) == 0.0
    end

    @testset "Anchored evaluation - extrap :extension" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        # Outside domain uses boundary polynomial
        aq_below = FI._anchor_query(x, -0.1)
        @test itp(aq_below) ≈ itp(-0.1) atol=1e-14

        aq_above = FI._anchor_query(x, 1.1)
        @test itp(aq_above) ≈ itp(1.1) atol=1e-14
    end

    @testset "Anchored evaluation - extrap :wrap" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:wrap)

        # Wrap extrapolation - anchor must be created with wrap=true
        # to pre-wrap coordinates for :wrap mode
        aq_above = FI._anchor_query(x, 1.3; wrap=true)
        wrapped_xq = mod(1.3, 1.0)  # 0.3
        @test itp(aq_above) ≈ itp(wrapped_xq) atol=1e-14
    end

    @testset "Anchored evaluation - PeriodicBC with wrap" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        itp = cubic_interp(x, y; bc=PeriodicBC())

        # Wrap at anchor construction (for extrap=:wrap mode)
        aq_wrapped = FI._anchor_query(x, 2π + 1.0; wrap=true)
        @test itp(aq_wrapped) ≈ itp(1.0) atol=1e-10
    end

    @testset "Anchored evaluation - Float32" begin
        x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
        y32 = sin.(Float32(2π) .* x32)
        itp = cubic_interp(x32, y32; extrap=:extension)

        xq = Float32(0.35)
        aq = FI._anchor_query(x32, xq)

        @test itp(aq) isa Float32
        @test itp(aq) ≈ itp(xq) atol=1f-6
    end

    @testset "Anchored evaluation - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)
        aq = FI._anchor_query(x, 0.5)

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

        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        # Create anchor once
        aq = FI._anchor_query(x, 0.35)

        # All three should work with same anchor
        @test itp1(aq) ≈ itp1(0.35) atol=1e-14
        @test itp2(aq) ≈ itp2(0.35) atol=1e-14
        @test itp3(aq) ≈ itp3(0.35) atol=1e-14
    end

    @testset "_CubicAnchoredQuery struct" begin
        x = collect(range(0.0, 1.0, 101))
        xq = 0.35

        # Basic construction
        aq = FI._anchor_query(x, xq)
        @test aq isa FI._CubicAnchoredQuery{Float64}

        # Float32 support
        x32 = Float32.(x)
        xq32 = Float32(0.35)
        aq32 = FI._anchor_query(x32, xq32)
        @test aq32 isa FI._CubicAnchoredQuery{Float32}
    end

    @testset "Anchor idx field" begin
        x = collect(range(0.0, 1.0, 11))  # 10 intervals, h=0.1

        # Interior points
        aq1 = FI._anchor_query(x, 0.05)
        @test aq1.idx == 1  # [0.0, 0.1)

        aq2 = FI._anchor_query(x, 0.15)
        @test aq2.idx == 2  # [0.1, 0.2)

        aq3 = FI._anchor_query(x, 0.95)
        @test aq3.idx == 10  # [0.9, 1.0]

        # At grid points
        aq_left = FI._anchor_query(x, 0.0)
        @test aq_left.idx == 1

        aq_right = FI._anchor_query(x, 1.0)
        @test aq_right.idx == 10  # last interval
    end

    @testset "Anchor side field" begin
        x = collect(range(0.0, 1.0, 11))

        # Interior: side = 0
        aq_inside = FI._anchor_query(x, 0.5)
        @test aq_inside.side == 0x00

        # At boundaries: side = 0 (still inside domain)
        aq_left_bound = FI._anchor_query(x, 0.0)
        @test aq_left_bound.side == 0x00

        aq_right_bound = FI._anchor_query(x, 1.0)
        @test aq_right_bound.side == 0x00

        # Below minimum: side = 1
        aq_below = FI._anchor_query(x, -0.5)
        @test aq_below.side == 0x01

        # Above maximum: side = 2
        aq_above = FI._anchor_query(x, 1.5)
        @test aq_above.side == 0x02
    end

    @testset "Anchor xq field preserved" begin
        x = collect(range(0.0, 1.0, 101))

        xq_values = [0.0, 0.35, 0.5, 1.0, -0.5, 1.5]
        for xq in xq_values
            aq = FI._anchor_query(x, xq)
            @test aq.xq == xq
        end
    end

    @testset "Anchor weights tuple" begin
        x = collect(range(0.0, 1.0, 101))
        xq = 0.35

        aq = FI._anchor_query(x, xq)
        @test aq.w0 isa NTuple{4, Float64}
        @test aq.w1 isa NTuple{4, Float64}
        @test aq.w2 isa NTuple{4, Float64}

        aq32 = FI._anchor_query(Float32.(x), Float32(xq))
        @test aq32.w0 isa NTuple{4, Float32}
        @test aq32.w1 isa NTuple{4, Float32}
        @test aq32.w2 isa NTuple{4, Float32}
    end

    @testset "Anchor wrapping (wrap=true)" begin
        x = collect(range(0.0, 2π, 101))

        # Query outside domain with wrap=true should wrap
        xq_outside = 2π + 1.0  # wraps to ~1.0
        aq_wrapped = FI._anchor_query(x, xq_outside; wrap=true)

        # Should be inside after wrapping
        @test aq_wrapped.side == 0x00
        @test aq_wrapped.xq != xq_outside  # xq is wrapped value

        # Without wrap, should be outside
        aq_nowrap = FI._anchor_query(x, xq_outside; wrap=false)
        @test aq_nowrap.side == 0x02  # above max
    end

    @testset "Invalid deriv argument" begin
        x = collect(range(0.0, 1.0, 101))
        aq = FI._anchor_query(x, 0.5)
        itp = cubic_interp(x, sin.(2π .* x))
        @test_throws ArgumentError itp(aq; deriv=-1)
        @test_throws ArgumentError itp(aq; deriv=3)
    end

    # ========================================
    # Vector Anchored Queries
    # ========================================

    @testset "Vector anchor construction" begin
        x = collect(range(0.0, 1.0, 101))
        xq = [0.15, 0.35, 0.5, 0.75]

        aq_vec = FI._anchor_query(x, xq)

        @test length(aq_vec) == 4
        @test all(aq -> aq isa FI._CubicAnchoredQuery{Float64}, aq_vec)
        @test all(aq -> aq.side == 0x00, aq_vec)  # All inside domain
    end

    @testset "Vector anchor - type promotion" begin
        x = collect(range(0.0, 1.0, 101))
        xq_f32 = Float32[0.15, 0.35, 0.5]
        aq_vec = FI._anchor_query(x, xq_f32)

        @test all(aq -> aq isa FI._CubicAnchoredQuery{Float64}, aq_vec)
    end

    @testset "Vector anchor - wrap=true" begin
        x = collect(range(0.0, 1.0, 101))
        xq = [-0.3, 1.3, 2.5]
        aq_vec = FI._anchor_query(x, xq; wrap=true)

        @test all(aq -> aq.side == 0x00, aq_vec)  # All wrapped to inside
    end

    @testset "Vector anchor - empty input" begin
        x = collect(range(0.0, 1.0, 101))
        aq_vec = FI._anchor_query(x, Float64[])

        @test length(aq_vec) == 0
    end

    # ========================================
    # Vector Anchored Query Evaluation
    # ========================================

    @testset "Vector evaluation - allocating" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        xq = [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]
        aq_vec = FI._anchor_query(x, xq)
        vals = itp(aq_vec)

        @test length(vals) == length(xq)
        for (i, xq_i) in enumerate(xq)
            @test vals[i] ≈ itp(xq_i) atol=1e-14
        end
    end

    @testset "Vector evaluation - extrap :none" begin
        x = collect(range(0.0, 1.0, 101))
        itp = cubic_interp(x, sin.(x); extrap=:none)

        xq = [-0.1, 0.5, 1.1]  # First is out of domain
        aq_vec = FI._anchor_query(x, xq)

        @test_throws DomainError itp(aq_vec)
    end

    @testset "Vector evaluation - extrap :constant" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:constant)

        xq = [-0.5, 0.5, 1.5]
        aq_vec = FI._anchor_query(x, xq)
        vals = itp(aq_vec)

        @test vals[1] ≈ y[1]
        @test vals[2] ≈ itp(0.5)
        @test vals[3] ≈ y[end]
    end

    @testset "Vector evaluation - extrap :extension" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        xq = [-0.1, 0.5, 1.1]
        aq_vec = FI._anchor_query(x, xq)
        vals = itp(aq_vec)

        for (i, xq_i) in enumerate(xq)
            @test vals[i] ≈ itp(xq_i) atol=1e-14
        end
    end

    @testset "Vector evaluation - in-place" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        xq = [0.15, 0.35, 0.5, 0.75]
        aq_vec = FI._anchor_query(x, xq)
        output = Vector{Float64}(undef, 4)

        result = itp(output, aq_vec)

        @test result === output
        for (i, xq_i) in enumerate(xq)
            @test output[i] ≈ itp(xq_i) atol=1e-14
        end
    end

    @testset "Vector evaluation - derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        xq = [0.15, 0.5, 0.85]
        aq_vec = FI._anchor_query(x, xq)

        for deriv in [1, 2]
            vals = itp(aq_vec; deriv=deriv)
            for (i, xq_i) in enumerate(xq)
                @test vals[i] ≈ itp(xq_i; deriv=deriv) atol=1e-14
            end
        end
    end

    @testset "Vector evaluation - length mismatch" begin
        x = collect(range(0.0, 1.0, 101))
        itp = cubic_interp(x, sin.(x))

        aq_vec = FI._anchor_query(x, [0.15, 0.35, 0.5])
        output = Vector{Float64}(undef, 2)  # Wrong size

        @test_throws AssertionError itp(output, aq_vec)
    end

    @testset "Vector evaluation - extrap :constant derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:constant)

        xq = [-0.5, 0.5, 1.5]
        aq_vec = FI._anchor_query(x, xq)

        derivs = itp(aq_vec; deriv=1)
        @test derivs[1] == 0.0
        @test derivs[3] == 0.0
    end

    @testset "Vector evaluation - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        itp = cubic_interp(x, sin.(x))

        xq = collect(range(0.1, 0.9, 50))
        aq_vec = FI._anchor_query(x, xq)
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
        aq_vec = FI._anchor_query(x, xq)  # Anchors computed once

        @test itp1(aq_vec) ≈ [itp1(q) for q in xq] atol=1e-14
        @test itp2(aq_vec) ≈ [itp2(q) for q in xq] atol=1e-14
        @test itp3(aq_vec) ≈ [itp3(q) for q in xq] atol=1e-14
    end

    @testset "Vector evaluation - wrap=true" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:wrap)

        xq = [-0.3, 1.3, 2.5]
        aq_vec = FI._anchor_query(x, xq; wrap=true)
        vals = itp(aq_vec)

        for (i, xq_i) in enumerate(xq)
            wrapped = mod(xq_i, 1.0)
            @test vals[i] ≈ itp(wrapped) atol=1e-14
        end
    end

    # ========================================
    # Zero-Allocation Tests for Derivatives
    # ========================================

    @testset "Scalar anchored derivatives - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)
        aq = FI._anchor_query(x, 0.5)

        # Warmup all derivative orders
        itp(aq)
        itp(aq; deriv=1)
        itp(aq; deriv=2)

        # Value (deriv=0) - already tested, but include for completeness
        allocs_d0 = @allocated itp(aq)
        @test allocs_d0 <= ALLOC_THRESHOLD

        # First derivative
        allocs_d1 = @allocated itp(aq; deriv=1)
        @test allocs_d1 <= ALLOC_THRESHOLD

        # Second derivative
        allocs_d2 = @allocated itp(aq; deriv=2)
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

    @testset "Vector in-place derivatives - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        xq = collect(range(0.1, 0.9, 50))
        aq_vec = FI._anchor_query(x, xq)
        output = similar(xq)

        # Warmup all derivative orders
        itp(output, aq_vec)
        itp(output, aq_vec; deriv=1)
        itp(output, aq_vec; deriv=2)

        # Value (deriv=0)
        allocs_d0 = @allocated itp(output, aq_vec)
        @test allocs_d0 <= ALLOC_THRESHOLD

        # First derivative - zero allocation
        allocs_d1 = @allocated itp(output, aq_vec; deriv=1)
        @test allocs_d1 <= ALLOC_THRESHOLD

        # Second derivative - zero allocation
        allocs_d2 = @allocated itp(output, aq_vec; deriv=2)
        @test allocs_d2 <= ALLOC_THRESHOLD
    end

end
