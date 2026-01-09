@testset "Cubic Anchored Query" begin

    # ========================================
    # Phase 1: Core Types & Weight Computation
    # ========================================

    # ========================================
    # Phase 2: CubicInterpolant Grid ID
    # ========================================

    @testset "CubicInterpolant has grid_id field" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)

        itp = cubic_interp(x, y)

        # grid_id field exists and matches _grid_id(x)
        @test hasfield(CubicInterpolant, :grid_id)
        @test itp.grid_id == FastInterpolations._grid_id(x)
        @test itp.grid_id isa Tuple{Int, UInt}
    end

    @testset "Same grid produces same grid_id" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)
        y3 = exp.(-3 .* x)

        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        itp3 = cubic_interp(x, y3)

        # All three should have identical grid_id
        @test itp1.grid_id == itp2.grid_id
        @test itp2.grid_id == itp3.grid_id
    end

    @testset "Different grids produce different grid_id" begin
        x1 = collect(range(0.0, 1.0, 101))
        x2 = collect(range(0.0, 2.0, 101))  # different range
        x3 = collect(range(0.0, 1.0, 51))   # different length
        y = sin.(2π .* x1)

        itp1 = cubic_interp(x1, y)
        itp2 = cubic_interp(x2, sin.(π .* x2))
        itp3 = cubic_interp(x3, sin.(2π .* x3))

        @test itp1.grid_id != itp2.grid_id
        @test itp1.grid_id != itp3.grid_id
    end

    @testset "Grid ID with different BC types" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]  # periodic

        # NaturalBC
        itp_natural = cubic_interp(x, y; bc=NaturalBC())

        # PeriodicBC
        itp_periodic = cubic_interp(x, y; bc=PeriodicBC())

        # Same grid → same grid_id
        @test itp_natural.grid_id == itp_periodic.grid_id
    end

    @testset "Grid ID with cache reuse" begin
        x = collect(range(0.0, 1.0, 101))
        cache = CubicSplineCache(x)

        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        itp1 = cubic_interp(cache, y1)
        itp2 = cubic_interp(cache, y2)

        # Both should have same grid_id (from cache's grid)
        @test itp1.grid_id == itp2.grid_id
        @test itp1.grid_id == FastInterpolations._grid_id(x)
    end

    # ========================================
    # Phase 3: Grid Validation
    # ========================================

    @testset "GridMismatchError type" begin
        # GridMismatchError should be defined and be an Exception
        @test isdefined(FastInterpolations, :GridMismatchError)
        @test FastInterpolations.GridMismatchError <: Exception
    end

    @testset "_validate_grid matching grids" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)

        itp = cubic_interp(x, y)
        aq = anchor_query(x, 0.5)

        # Same grid → validation passes (returns nothing)
        @test FastInterpolations._validate_grid(aq, itp) === nothing
    end

    @testset "_validate_grid different length" begin
        x1 = collect(range(0.0, 1.0, 101))
        x2 = collect(range(0.0, 1.0, 51))  # different length

        itp = cubic_interp(x1, sin.(2π .* x1))
        aq = anchor_query(x2, 0.5)

        # Different length → GridMismatchError
        @test_throws FastInterpolations.GridMismatchError FastInterpolations._validate_grid(aq, itp)
    end

    @testset "_validate_grid different content same length" begin
        x1 = collect(range(0.0, 1.0, 101))
        x2 = collect(range(0.0, 2.0, 101))  # same length, different content

        itp = cubic_interp(x1, sin.(2π .* x1))
        aq = anchor_query(x2, 0.5)

        # Different hash → GridMismatchError
        @test_throws FastInterpolations.GridMismatchError FastInterpolations._validate_grid(aq, itp)
    end

    @testset "GridMismatchError message quality" begin
        x1 = collect(range(0.0, 1.0, 101))
        x2 = collect(range(0.0, 2.0, 51))  # different length AND content

        itp = cubic_interp(x1, sin.(2π .* x1))
        aq = anchor_query(x2, 0.5)

        # Capture the error and check message
        err = try
            FastInterpolations._validate_grid(aq, itp)
            nothing
        catch e
            e
        end

        @test err isa FastInterpolations.GridMismatchError
        @test err.anchor_grid_id == aq.grid_id
        @test err.interp_grid_id == itp.grid_id

        # Error message should contain grid info
        msg = sprint(showerror, err)
        @test occursin("GridMismatchError", msg)
        @test occursin("length", msg)
    end

    # ========================================
    # Phase 4: Anchored Evaluation
    # ========================================

    @testset "Anchored evaluation - exact agreement" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        # Test multiple query points
        for xq in [0.0, 0.15, 0.35, 0.5, 0.75, 0.99, 1.0]
            aq = anchor_query(x, xq)
            @test itp(aq) ≈ itp(xq) atol=1e-14
        end
    end

    @testset "Anchored evaluation - derivatives" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        for xq in [0.15, 0.5, 0.85]
            # First derivative
            aq1 = anchor_query(x, xq; deriv=1)
            @test itp(aq1) ≈ itp(xq; deriv=1) atol=1e-14

            # Second derivative
            aq2 = anchor_query(x, xq; deriv=2)
            @test itp(aq2) ≈ itp(xq; deriv=2) atol=1e-14
        end
    end

    @testset "Anchored evaluation - extrap :none" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:none)

        # Inside domain should work
        aq_inside = anchor_query(x, 0.5)
        @test isfinite(itp(aq_inside))

        # Outside domain should throw DomainError
        aq_below = anchor_query(x, -0.1)
        @test_throws DomainError itp(aq_below)

        aq_above = anchor_query(x, 1.1)
        @test_throws DomainError itp(aq_above)
    end

    @testset "Anchored evaluation - extrap :constant" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:constant)

        # Below domain returns y[1]
        aq_below = anchor_query(x, -0.5)
        @test itp(aq_below) ≈ y[1]

        # Above domain returns y[end]
        aq_above = anchor_query(x, 1.5)
        @test itp(aq_above) ≈ y[end]

        # Derivatives of constant are zero
        aq_below_d1 = anchor_query(x, -0.5; deriv=1)
        @test itp(aq_below_d1) == 0.0

        aq_above_d2 = anchor_query(x, 1.5; deriv=2)
        @test itp(aq_above_d2) == 0.0
    end

    @testset "Anchored evaluation - extrap :extension" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)

        # Outside domain uses boundary polynomial
        aq_below = anchor_query(x, -0.1)
        @test itp(aq_below) ≈ itp(-0.1) atol=1e-14

        aq_above = anchor_query(x, 1.1)
        @test itp(aq_above) ≈ itp(1.1) atol=1e-14
    end

    @testset "Anchored evaluation - extrap :wrap" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:wrap)

        # Wrap extrapolation - anchor must be created with periodic=true
        # to pre-wrap coordinates for :wrap mode
        aq_above = anchor_query(x, 1.3; periodic=true)
        wrapped_xq = mod(1.3, 1.0)  # 0.3
        @test itp(aq_above) ≈ itp(wrapped_xq) atol=1e-14
    end

    @testset "Anchored evaluation - PeriodicBC" begin
        x = collect(range(0.0, 2π, 101))
        y = sin.(x)
        y[end] = y[1]  # Ensure periodic
        itp = cubic_interp(x, y; bc=PeriodicBC())

        # Periodic wrapping at anchor construction
        aq_wrapped = anchor_query(x, 2π + 1.0; periodic=true)
        @test itp(aq_wrapped) ≈ itp(1.0) atol=1e-10
    end

    @testset "Anchored evaluation - grid mismatch throws" begin
        x1 = collect(range(0.0, 1.0, 101))
        x2 = collect(range(0.0, 2.0, 101))

        itp = cubic_interp(x1, sin.(2π .* x1))
        aq_wrong = anchor_query(x2, 0.5)

        # Should throw GridMismatchError
        @test_throws FastInterpolations.GridMismatchError itp(aq_wrong)
    end

    @testset "Anchored evaluation - Float32" begin
        x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
        y32 = sin.(Float32(2π) .* x32)
        itp = cubic_interp(x32, y32; extrap=:extension)

        xq = Float32(0.35)
        aq = anchor_query(x32, xq)

        @test itp(aq) isa Float32
        @test itp(aq) ≈ itp(xq) atol=1f-6
    end

    @testset "Anchored evaluation - zero allocation" begin
        x = collect(range(0.0, 1.0, 101))
        y = sin.(2π .* x)
        itp = cubic_interp(x, y; extrap=:extension)
        aq = anchor_query(x, 0.5)

        # Warmup
        itp(aq)

        # Measure allocation
        allocs = @allocated itp(aq)
        @test allocs == 0
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
        aq = anchor_query(x, 0.35)

        # All three should work with same anchor
        @test itp1(aq) ≈ itp1(0.35) atol=1e-14
        @test itp2(aq) ≈ itp2(0.35) atol=1e-14
        @test itp3(aq) ≈ itp3(0.35) atol=1e-14
    end

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

    # ========================================
    # Phase 5: Integration & Documentation
    # ========================================

    @testset "Public API exports" begin
        # anchor_query should be exported (no module prefix needed)
        @test isdefined(FastInterpolations, :anchor_query)
        @test isdefined(Main, :anchor_query)  # available in test scope

        # CubicAnchoredQuery type should be exported
        @test isdefined(FastInterpolations, :CubicAnchoredQuery)
        @test isdefined(Main, :CubicAnchoredQuery)

        # EvalOp types should be exported for type checking
        @test isdefined(FastInterpolations, :AbstractEvalOp)
        @test isdefined(FastInterpolations, :EvalValue)
        @test isdefined(FastInterpolations, :EvalDeriv1)
        @test isdefined(FastInterpolations, :EvalDeriv2)
        @test isdefined(Main, :EvalValue)
    end

    @testset "Documentation exists" begin
        # anchor_query should have a docstring
        doc_str = string(@doc anchor_query)
        @test occursin("anchor_query", doc_str)
        @test occursin("Arguments", doc_str) || occursin("xq", doc_str)
        @test occursin("Example", doc_str)
    end

end
