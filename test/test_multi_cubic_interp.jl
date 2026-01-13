# Tests for MultiCubicInterpolant - Multi-Y cubic interpolation
#
# Multi-Y interpolation: multiple y-data series sharing the same x-grid.
# After unification: Uses matrix storage with adaptive layout for optimal performance.
#
# ALLOC_THRESHOLD is defined in runtests.jl

using FastInterpolations: _ensure_point_layout!

# ============================================================================
# Phase 1: Unified Type Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testset "MultiCubicInterpolant - TransposeSnapshot Type" begin
    FI = FastInterpolations

    @testset "Empty snapshot creation" begin
        snap = FI.TransposeSnapshot{Float64}()
        @test snap.y_point === nothing
        @test snap.z_point === nothing
    end

    @testset "Snapshot with matrices" begin
        y_point = rand(3, 10)  # 3 series × 10 points
        z_point = rand(3, 10)
        snap = FI.TransposeSnapshot{Float64}(y_point, z_point)
        @test snap.y_point === y_point
        @test snap.z_point === z_point
    end

    @testset "Float32 snapshot" begin
        snap32 = FI.TransposeSnapshot{Float32}()
        @test snap32.y_point === nothing
        @test snap32.z_point === nothing

        y_pt = rand(Float32, 2, 5)
        z_pt = rand(Float32, 2, 5)
        snap32_full = FI.TransposeSnapshot{Float32}(y_pt, z_pt)
        @test snap32_full.y_point === y_pt
    end
end

@testset "MultiCubicInterpolant - Unified Struct Fields" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3])

    @testset "Has matrix storage fields (not itps)" begin
        # After unification: has matrix storage
        @test hasfield(typeof(mitp), :y)
        @test hasfield(typeof(mitp), :z)
        @test hasfield(typeof(mitp), :cache)
        @test hasfield(typeof(mitp), :_point_snapshot)
        @test hasfield(typeof(mitp), :extrap)

        # Should NOT have itps field anymore
        @test !hasfield(typeof(mitp), :itps)
    end

    @testset "Matrix dimensions correct (n_points × n_series)" begin
        n_points = length(x)
        n_series_count = 3

        @test mitp.y isa Matrix{Float64}
        @test size(mitp.y) == (n_points, n_series_count)
        @test size(mitp.z) == (n_points, n_series_count)
    end

    @testset "y values stored correctly in columns" begin
        @test mitp.y[:, 1] ≈ y1
        @test mitp.y[:, 2] ≈ y2
        @test mitp.y[:, 3] ≈ y3
    end

    @testset "z coefficients match individual interpolants" begin
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        itp3 = cubic_interp(x, y3)

        @test mitp.z[:, 1] ≈ itp1.z rtol=1e-12
        @test mitp.z[:, 2] ≈ itp2.z rtol=1e-12
        @test mitp.z[:, 3] ≈ itp3.z rtol=1e-12
    end

    @testset "Cache is CubicSplineCache" begin
        @test mitp.cache isa FI.CubicSplineCache{Float64}
    end

    @testset "Extrap is Val type" begin
        mitp_ext = cubic_interp(x, [y1, y2]; extrap=:extension)
        @test mitp_ext.extrap === Val(:extension)

        mitp_const = cubic_interp(x, [y1, y2]; extrap=:constant)
        @test mitp_const.extrap === Val(:constant)
    end
end

@testset "MultiCubicInterpolant - Helper Functions" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "n_series and n_points helpers" begin
        mitp = cubic_interp(x, [y1, y2, y3])
        @test FI.n_series(mitp) == 3
        @test FI.n_points(mitp) == 101
    end

    @testset "n_series with different counts" begin
        mitp1 = cubic_interp(x, [y1])
        @test FI.n_series(mitp1) == 1

        mitp5 = cubic_interp(x, [y1, y2, y3, y1, y2])
        @test FI.n_series(mitp5) == 5
    end
end

@testset "MultiCubicInterpolant - Type Parameters" begin
    FI = FastInterpolations

    @testset "Float64 type parameter" begin
        x64 = collect(range(0.0, 1.0, 101))
        y1_64 = sin.(2π .* x64)
        y2_64 = cos.(2π .* x64)

        mitp64 = cubic_interp(x64, [y1_64, y2_64])
        @test mitp64 isa FI.CubicMultiInterpolant{Float64}
    end

    @testset "Float32 type parameter" begin
        x32 = Float32.(collect(range(0.0f0, 1.0f0, 101)))
        y1_32 = sin.(2f0 * Float32(π) .* x32)
        y2_32 = cos.(2f0 * Float32(π) .* x32)

        mitp32 = cubic_interp(x32, [y1_32, y2_32])
        @test mitp32 isa FI.CubicMultiInterpolant{Float32}
    end
end

# ============================================================================
# Phase 2: Constructor Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testset "MultiCubicInterpolant - Constructor Migration" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 51))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "precompute_transpose option" begin
        mitp = cubic_interp(x, [y1, y2]; precompute_transpose=true)

        # _point_snapshot should be populated immediately
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point !== nothing
        @test snap.z_point !== nothing
        @test size(snap.y_point) == (2, length(x))  # (n_series × n_points)
        @test size(snap.z_point) == (2, length(x))
    end

    @testset "Lazy transpose (precompute_transpose=false)" begin
        mitp = cubic_interp(x, [y1, y2])  # default precompute_transpose=false

        # Initially empty
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point === nothing
        @test snap.z_point === nothing
    end

    @testset "Range grid preserved in cache" begin
        x_range = range(0.0, 1.0, 51)
        y1_r = sin.(2π .* collect(x_range))
        y2_r = cos.(2π .* collect(x_range))

        mitp = cubic_interp(x_range, [y1_r, y2_r])
        @test mitp.cache.x isa AbstractRange
    end

    @testset "PeriodicBC forces :wrap extrap" begin
        # Create periodic data (endpoints match)
        y_periodic = sin.(2π .* x)  # sin(0) = sin(2π) = 0

        mitp = cubic_interp(x, [y_periodic]; bc=PeriodicBC(), extrap=:none)
        @test mitp.extrap === Val(:wrap)

        # Even if user requests :extension, periodic BC should override to :wrap
        mitp2 = cubic_interp(x, [y_periodic]; bc=PeriodicBC(), extrap=:extension)
        @test mitp2.extrap === Val(:wrap)
    end
end

# ============================================================================
# Phase 3: Scalar Kernel Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testset "MultiCubicInterpolant - Scalar Kernel Migration" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Grid knot exactness" begin
        mitp = cubic_interp(x, [y1, y2])
        for (i, xi) in enumerate(x)
            vals = mitp(xi)
            @test vals[1] ≈ y1[i] atol=1e-14
            @test vals[2] ≈ y2[i] atol=1e-14
        end
    end

    @testset "Lazy point-layout triggered by scalar eval" begin
        mitp = cubic_interp(x, [y1, y2])

        # Initially empty
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point === nothing

        # After scalar eval → populated
        _ = mitp(0.5)
        snap_after = @atomic :acquire mitp._point_snapshot
        @test snap_after.y_point !== nothing
        @test size(snap_after.y_point) == (2, length(x))
    end

    @testset "_ensure_point_layout! idempotency" begin
        mitp = cubic_interp(x, [y1, y2])
        y_pt1, z_pt1 = _ensure_point_layout!(mitp)
        y_pt2, z_pt2 = _ensure_point_layout!(mitp)
        @test y_pt1 === y_pt2  # Same reference (idempotent)
        @test z_pt1 === z_pt2
    end

    @testset "DimensionMismatch for wrong output size" begin
        mitp = cubic_interp(x, [y1, y2])
        out_wrong = zeros(5)  # Wrong size (should be 2)
        @test_throws DimensionMismatch mitp(out_wrong, 0.5)
    end

    @testset "Scalar zero-alloc after precompute" begin
        mitp = cubic_interp(x, [y1, y2]; precompute_transpose=true)
        out = zeros(2)
        mitp(out, 0.5)  # Warmup

        allocs = @allocated mitp(out, 0.5)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Invalid deriv=3 throws" begin
        mitp = cubic_interp(x, [y1])
        @test_throws Exception mitp(0.5; deriv=3)
    end
end

# ============================================================================
# Phase 4: Vector Kernel Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testset "MultiCubicInterpolant - Vector Kernel Migration" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Output layout series-first" begin
        mitp = cubic_interp(x, [y1, y2])
        xq_vec = [0.2, 0.4, 0.6]
        outputs = mitp(xq_vec)

        for (j, xq) in enumerate(xq_vec)
            scalar_result = mitp(xq)
            @test outputs[1][j] ≈ scalar_result[1] atol=1e-14
            @test outputs[2][j] ≈ scalar_result[2] atol=1e-14
        end
    end

    @testset "DimensionMismatch for wrong container size" begin
        mitp = cubic_interp(x, [y1, y2])
        xq_vec = [0.1, 0.3, 0.5]

        outputs_wrong = [zeros(3)]  # Only 1 series (should be 2)
        @test_throws DimensionMismatch mitp(outputs_wrong, xq_vec)
    end

    @testset "Anchor reuse produces identical results" begin
        mitp = cubic_interp(x, [y1, y2])
        xq_vec = [0.2, 0.5, 0.8]
        aq_vec = FI._anchor_query(x, xq_vec)

        outputs1 = [zeros(3) for _ in 1:2]
        outputs2 = [zeros(3) for _ in 1:2]

        mitp(outputs1, aq_vec)
        mitp(outputs2, aq_vec)

        @test outputs1[1] ≈ outputs2[1] atol=1e-14
        @test outputs1[2] ≈ outputs2[2] atol=1e-14
    end

    @testset "Finite difference derivative verification" begin
        y_poly = x .^ 3 .- 2 .* x .^ 2 .+ x
        mitp = cubic_interp(x, [y_poly])
        xq = 0.4
        h = 1e-6

        fd1 = (mitp(xq + h) .- mitp(xq - h)) ./ (2h)
        d1 = mitp(xq; deriv=1)
        @test d1 ≈ fd1 atol=1e-5
    end

    @testset "Derivatives with all BC types" begin
        for bc in [NaturalBC(), ClampedBC()]
            mitp = cubic_interp(x, [y1, y2]; bc=bc)
            itp1 = cubic_interp(x, y1; bc=bc)

            d1_multi = mitp(0.5; deriv=1)
            @test d1_multi[1] ≈ itp1(0.5; deriv=1) atol=1e-14
        end

        # PeriodicBC requires matching endpoints
        y1_periodic = sin.(2π .* x)
        y2_periodic = cos.(2π .* x)
        mitp_periodic = cubic_interp(x, [y1_periodic, y2_periodic]; bc=PeriodicBC())
        itp1_periodic = cubic_interp(x, y1_periodic; bc=PeriodicBC())

        d1_multi = mitp_periodic(0.5; deriv=1)
        @test d1_multi[1] ≈ itp1_periodic(0.5; deriv=1) atol=1e-14
    end
end

# ============================================================================
# Phase 5: Memory & Allocation Verification Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testset "MultiCubicInterpolant - Memory & Allocation" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Memory footprint - lazy transpose" begin
        ys = [rand(101) for _ in 1:10]
        mitp = cubic_interp(x, ys)

        size_before = Base.summarysize(mitp)
        _ = mitp(0.5)  # Triggers lazy layout creation
        size_after = Base.summarysize(mitp)

        ratio = size_after / size_before
        @test 1.5 < ratio < 2.5  # ~2x for transpose storage
    end

    @testset "_ensure_point_layout! allocates only once" begin
        mitp = cubic_interp(x, [y1])
        _ensure_point_layout!(mitp)  # First call (warmup)

        # Use direct import to ensure proper escape analysis
        allocs = @allocated _ensure_point_layout!(mitp)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Scalar deriv=1 zero-alloc" begin
        mitp = cubic_interp(x, [y1, y2]; precompute_transpose=true)
        out = zeros(2)
        mitp(out, 0.5; deriv=1)  # Warmup

        allocs = @allocated mitp(out, 0.5; deriv=1)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Vector deriv=1 zero-alloc with pool" begin
        mitp = cubic_interp(x, [y1, y2])
        xq = collect(range(0.1, 0.9, 100))
        outputs = [zeros(100) for _ in 1:2]

        mitp(outputs, xq; deriv=1)  # Warmup × 2
        mitp(outputs, xq; deriv=1)

        allocs = @allocated mitp(outputs, xq; deriv=1)
        @test allocs <= ALLOC_THRESHOLD
    end
end

# ============================================================================
# Phase 6: Uncovered Path Coverage Tests
# ============================================================================

@testset "MultiCubicInterpolant - Uncovered Path Coverage" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "precompute_transpose! function" begin
        mitp = cubic_interp(x, [y1, y2])

        # Initially empty
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point === nothing

        # Call precompute_transpose!
        result = FI.precompute_transpose!(mitp)
        @test result === mitp

        # Now populated
        snap_after = @atomic :acquire mitp._point_snapshot
        @test snap_after.y_point !== nothing
    end

    @testset ":wrap extrapolation scalar SIMD path" begin
        # Create periodic data (endpoints match)
        y_periodic = sin.(2π .* x)  # sin(0) = sin(2π) = 0
        y_periodic2 = cos.(2π .* x)  # cos(0) = cos(2π) = 1
        y_periodic2[end] = y_periodic2[1]  # Ensure exact match

        mitp = cubic_interp(x, [y_periodic, y_periodic2]; bc=PeriodicBC())

        # Query outside domain triggers :wrap extrapolation
        out = zeros(2)
        mitp(out, -0.1)  # Below domain
        @test !any(isnan, out)

        mitp(out, 1.1)   # Above domain
        @test !any(isnan, out)
    end

    @testset "Periodic BC endpoint mismatch error" begin
        y_bad = sin.(2π .* x)
        y_bad[end] = 999.0  # Force mismatch

        @test_throws ArgumentError cubic_interp(x, [y_bad]; bc=PeriodicBC())
    end

    @testset "precompute_transpose with periodic BC" begin
        y_periodic = sin.(2π .* x)
        mitp = cubic_interp(x, [y_periodic]; bc=PeriodicBC(), precompute_transpose=true)

        # Should have point layout immediately
        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point !== nothing
    end

    @testset "Matrix input with row dimension mismatch" begin
        Y_wrong = rand(50, 3)  # 50 rows but x has 101 points
        @test_throws DimensionMismatch cubic_interp(x, Y_wrong)
    end

    @testset "Matrix input with periodic BC" begin
        Y_periodic = hcat(sin.(2π .* x), cos.(2π .* x))
        Y_periodic[end, 2] = Y_periodic[1, 2]  # Fix endpoint

        mitp = cubic_interp(x, Y_periodic; bc=PeriodicBC())
        @test mitp.extrap === Val(:wrap)
    end

    @testset "Matrix input with precompute_transpose" begin
        Y = hcat(y1, y2)
        mitp = cubic_interp(x, Y; precompute_transpose=true)

        snap = @atomic :acquire mitp._point_snapshot
        @test snap.y_point !== nothing
    end

    @testset "Anchored query path dimension errors" begin
        mitp = cubic_interp(x, [y1, y2, sin.(x)])
        xq_vec = [0.1, 0.3, 0.5]
        aq_vec = FI._anchor_query(x, xq_vec)

        # Wrong number of output buffers
        outputs_wrong = [zeros(3), zeros(3)]  # 2 instead of 3
        @test_throws DimensionMismatch mitp(outputs_wrong, aq_vec)

        # Wrong buffer length
        outputs_bad_len = [zeros(3), zeros(3), zeros(5)]  # Last has wrong length
        @test_throws DimensionMismatch mitp(outputs_bad_len, aq_vec)
    end

    @testset "Vector extrapolation :extension mode" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:extension)
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq)

        @test !any(isnan, outputs[1])
        @test !any(isnan, outputs[2])
    end

    @testset "Vector extrapolation :constant mode" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:constant)
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq)

        # Out-of-domain should clamp to boundary values
        @test outputs[1][1] ≈ y1[1] atol=1e-10  # Below domain → first value
        @test outputs[1][3] ≈ y1[end] atol=1e-10  # Above domain → last value
    end

    @testset "Vector extrapolation :wrap mode (periodic)" begin
        y_periodic = sin.(2π .* x)
        y_periodic2 = cos.(2π .* x)
        y_periodic2[end] = y_periodic2[1]

        mitp = cubic_interp(x, [y_periodic, y_periodic2]; bc=PeriodicBC())
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq)

        @test !any(isnan, outputs[1])
        @test !any(isnan, outputs[2])
    end

    @testset "Vector extrapolation :constant with derivative" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:constant)
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq; deriv=1)

        # Derivatives outside domain should be zero for :constant
        @test outputs[1][1] ≈ 0.0 atol=1e-10
        @test outputs[1][3] ≈ 0.0 atol=1e-10
    end

    @testset "DomainError message includes bounds" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:none)

        # Scalar path
        err = try
            mitp(-0.1)
            nothing
        catch e
            e
        end
        @test err isa DomainError
        @test occursin("[0.0, 1.0]", err.msg)

        # Vector path
        err2 = try
            mitp([-0.1, 0.5])
            nothing
        catch e
            e
        end
        @test err2 isa DomainError
        @test occursin("[0.0, 1.0]", err2.msg)
    end
end

# ============================================================================
# Phase 1 (Legacy): Type Definition & Constructor Tests
# ============================================================================

@testset "MultiCubicInterpolant - Type Structure" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "Construction from Vector of y-series" begin
        # 2-arg form returns MultiCubicInterpolant
        mitp = cubic_interp(x, [y1, y2, y3])
        @test mitp isa FI.MultiCubicInterpolant

        # Unified structure: has matrix storage
        @test hasfield(typeof(mitp), :y)
        @test FI.n_series(mitp) == 3
    end

    @testset "Construction from Matrix (columns as series)" begin
        Y = hcat(y1, y2, y3)  # 101×3 matrix
        mitp = cubic_interp(x, Y)
        @test mitp isa FI.MultiCubicInterpolant
        @test FI.n_series(mitp) == 3
    end

    @testset "Single series works" begin
        mitp = cubic_interp(x, [y1])
        @test mitp isa FI.MultiCubicInterpolant
        @test FI.n_series(mitp) == 1
    end

    @testset "Type inference - Float64" begin
        mitp = cubic_interp(x, [y1, y2])
        @test mitp isa FI.CubicMultiInterpolant{Float64}
    end

    @testset "Type inference - Float32" begin
        x32 = Float32.(x)
        y1_32 = Float32.(y1)
        y2_32 = Float32.(y2)
        mitp = cubic_interp(x32, [y1_32, y2_32])
        @test mitp isa FI.CubicMultiInterpolant{Float32}
    end
end

@testset "MultiCubicInterpolant - Cache Sharing" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "Single shared cache (unified struct)" begin
        mitp = cubic_interp(x, [y1, y2, y3])

        # Unified struct has a single cache field (not itps)
        @test mitp.cache isa FI.CubicSplineCache
        @test mitp.cache.x === x || mitp.cache.x == x
    end

    @testset "Cache sharing with different BC types" begin
        # NaturalBC (default)
        mitp_natural = cubic_interp(x, [y1, y2])
        @test mitp_natural.cache isa FI.CubicSplineCache
        @test mitp_natural.cache.bc_config isa BCPair

        # PeriodicBC
        y1_periodic = copy(y1); y1_periodic[end] = y1_periodic[1]
        y2_periodic = copy(y2); y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, [y1_periodic, y2_periodic]; bc=PeriodicBC())
        @test mitp_periodic.cache isa FI.CubicSplineCache
        @test mitp_periodic.cache.bc_config isa FI.PeriodicData
    end
end

@testset "MultiCubicInterpolant - Constructor Validation" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Empty ys throws error" begin
        @test_throws Union{ArgumentError, AssertionError} cubic_interp(x, Vector{Float64}[])
    end

    @testset "Mismatched lengths throws error" begin
        y_short = y1[1:50]
        @test_throws Union{DimensionMismatch, ArgumentError, AssertionError} cubic_interp(x, [y1, y_short])
    end

    @testset "BC propagation" begin
        # NaturalBC (default) - check bc_config field (unified struct)
        mitp_natural = cubic_interp(x, [y1, y2])
        @test mitp_natural.cache.bc_config isa BCPair

        # PeriodicBC - check bc_config is PeriodicData
        y1_periodic = copy(y1); y1_periodic[end] = y1_periodic[1]
        y2_periodic = copy(y2); y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, [y1_periodic, y2_periodic]; bc=PeriodicBC())
        @test mitp_periodic.cache.bc_config isa FastInterpolations.PeriodicData
    end

    @testset "Extrap propagation" begin
        for extrap_mode in (:none, :constant, :extension, :wrap)
            mitp = cubic_interp(x, [y1, y2]; extrap=extrap_mode)
            # Unified struct has single extrap field
            @test mitp.extrap === Val(extrap_mode)
        end
    end
end

# ============================================================================
# Phase 2: Scalar Evaluation Tests
# ============================================================================

@testset "MultiCubicInterpolant - Scalar Evaluation (out-of-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "Returns Vector with correct values" begin
        result = mitp(0.35)
        @test result isa AbstractVector
        @test length(result) == 3

        # Individual results match standalone interpolants
        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        @test result[1] ≈ itp1(0.35) atol=1e-14
        @test result[2] ≈ itp2(0.35) atol=1e-14
        @test result[3] ≈ itp3(0.35) atol=1e-14
    end

    @testset "Derivatives via deriv keyword" begin
        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        # First derivative
        d1 = mitp(0.5; deriv=1)
        @test d1[1] ≈ itp1(0.5; deriv=1) atol=1e-14
        @test d1[2] ≈ itp2(0.5; deriv=1) atol=1e-14
        @test d1[3] ≈ itp3(0.5; deriv=1) atol=1e-14

        # Second derivative
        d2 = mitp(0.5; deriv=2)
        @test d2[1] ≈ itp1(0.5; deriv=2) atol=1e-14
        @test d2[2] ≈ itp2(0.5; deriv=2) atol=1e-14
        @test d2[3] ≈ itp3(0.5; deriv=2) atol=1e-14
    end

    @testset "Multiple query points" begin
        for xq in [0.0, 0.15, 0.5, 0.85, 1.0]
            result = mitp(xq)
            @test length(result) == 3
        end
    end
end

@testset "MultiCubicInterpolant - Scalar Evaluation (in-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "Fills output correctly" begin
        output = Vector{Float64}(undef, 3)
        result = mitp(output, 0.35)

        @test result === output  # Returns same reference

        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        @test output[1] ≈ itp1(0.35) atol=1e-14
        @test output[2] ≈ itp2(0.35) atol=1e-14
        @test output[3] ≈ itp3(0.35) atol=1e-14
    end

    @testset "Size assertion on output mismatch" begin
        output_wrong = Vector{Float64}(undef, 2)  # Wrong size
        @test_throws DimensionMismatch mitp(output_wrong, 0.35)
    end

    @testset "In-place with derivatives" begin
        output = Vector{Float64}(undef, 3)
        mitp(output, 0.5; deriv=1)

        itp1 = cubic_interp(x, y1; extrap=:extension)
        @test output[1] ≈ itp1(0.5; deriv=1) atol=1e-14
    end
end

@testset "MultiCubicInterpolant - Scalar Extrap Modes" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "extrap=:none - throws DomainError" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:none)

        # Inside domain works
        @test isfinite(mitp(0.5)[1])

        # Outside domain throws
        @test_throws DomainError mitp(-0.1)
        @test_throws DomainError mitp(1.1)
    end

    @testset "extrap=:constant - boundary values" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:constant)

        below = mitp(-0.5)
        @test below[1] ≈ y1[1]
        @test below[2] ≈ y2[1]

        above = mitp(1.5)
        @test above[1] ≈ y1[end]
        @test above[2] ≈ y2[end]
    end

    @testset "extrap=:extension - boundary polynomial" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:extension)
        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)

        below = mitp(-0.1)
        @test below[1] ≈ itp1(-0.1) atol=1e-14
        @test below[2] ≈ itp2(-0.1) atol=1e-14

        above = mitp(1.1)
        @test above[1] ≈ itp1(1.1) atol=1e-14
        @test above[2] ≈ itp2(1.1) atol=1e-14
    end

    @testset "extrap=:wrap - wrapped coordinates" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:wrap)

        # Query outside domain should wrap
        result = mitp(1.35)  # wraps to 0.35
        expected1 = cubic_interp(x, y1; extrap=:wrap)(1.35)
        expected2 = cubic_interp(x, y2; extrap=:wrap)(1.35)

        @test result[1] ≈ expected1 atol=1e-14
        @test result[2] ≈ expected2 atol=1e-14
    end
end

# ============================================================================
# Phase 3: Vector Evaluation Tests
# ============================================================================

@testset "MultiCubicInterpolant - Vector Evaluation (out-of-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = [0.15, 0.35, 0.5, 0.75]

    @testset "Returns container of Vector{T}" begin
        result = mitp(xq)
        @test result isa AbstractVector  # Container
        @test length(result) == 3       # One per y-series
        @test all(v -> v isa AbstractVector{Float64}, result)
        @test all(v -> length(v) == 4, result)  # length(xq)
    end

    @testset "Results match individual interpolants" begin
        result = mitp(xq)

        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        @test result[1] ≈ itp1(xq) atol=1e-14
        @test result[2] ≈ itp2(xq) atol=1e-14
        @test result[3] ≈ itp3(xq) atol=1e-14
    end

    @testset "Derivatives with vector queries" begin
        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)

        mitp2 = cubic_interp(x, [y1, y2]; extrap=:extension)

        d1 = mitp2(xq; deriv=1)
        @test d1[1] ≈ itp1(xq; deriv=1) atol=1e-14
        @test d1[2] ≈ itp2(xq; deriv=1) atol=1e-14

        d2 = mitp2(xq; deriv=2)
        @test d2[1] ≈ itp1(xq; deriv=2) atol=1e-14
        @test d2[2] ≈ itp2(xq; deriv=2) atol=1e-14
    end
end

@testset "MultiCubicInterpolant - Container In-place (KILLER FEATURE)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = collect(range(0.1, 0.9, 50))

    @testset "Fills all buffers correctly" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        result = mitp(outputs, xq)

        @test result === outputs  # Returns same reference

        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        @test out1 ≈ itp1(xq) atol=1e-14
        @test out2 ≈ itp2(xq) atol=1e-14
        @test out3 ≈ itp3(xq) atol=1e-14
    end

    @testset "Size assertion on container mismatch" begin
        outputs_wrong = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50)]  # Only 2
        @test_throws DimensionMismatch mitp(outputs_wrong, xq)
    end

    @testset "Size assertion on buffer mismatch" begin
        outputs = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50), Vector{Float64}(undef, 30)]  # Last is wrong size
        @test_throws DimensionMismatch mitp(outputs, xq)
    end

    @testset "ZERO ALLOCATION with pre-built anchors (critical)" begin
        FI = FastInterpolations
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors (this is the key for zero-allocation)
        aq_vec = FI._anchor_query(x, xq)

        # Warmup
        mitp(outputs, aq_vec)
        mitp(outputs, aq_vec)

        allocs = @allocated mitp(outputs, aq_vec)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "In-place with derivatives" begin
        FI = FastInterpolations
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq)
        mitp(outputs, aq_vec; deriv=1)

        itp1 = cubic_interp(x, y1; extrap=:extension)
        # Use same anchored path for comparison
        expected = Vector{Float64}(undef, 50)
        itp1(expected, aq_vec; deriv=1)
        @test out1 ≈ expected atol=1e-14
    end
end

@testset "MultiCubicInterpolant - Vector Extrap Modes" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "extrap=:none - throws on out-of-domain" begin
        mitp = cubic_interp(x, [y1, y2]; extrap=:none)
        xq_bad = [-0.1, 0.5, 1.1]

        @test_throws DomainError mitp(xq_bad)
    end

    @testset "All extrap modes work with vector" begin
        for mode in (:constant, :extension, :wrap)
            mitp = cubic_interp(x, [y1, y2]; extrap=mode)
            xq = [0.15, 0.5, 0.85]
            result = mitp(xq)
            @test length(result) == 2
            @test all(v -> length(v) == 3, result)
        end
    end
end

# ============================================================================
# Phase 4: Safety & Integration Tests
# ============================================================================

@testset "MultiCubicInterpolant - Immutability" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    x_original = copy(x)
    y1_original = copy(y1)

    mitp = cubic_interp(x, [y1, y2])

    @testset "External x modification has no effect" begin
        result_before = mitp(0.5)
        x[1] = 999.0  # Mutate external x
        result_after = mitp(0.5)
        @test result_before ≈ result_after
    end

    @testset "External y modification has no effect" begin
        x .= x_original  # Restore
        mitp2 = cubic_interp(x, [copy(y1), copy(y2)])
        result_before = mitp2(0.5)
        y1[1] = 999.0  # Mutate external y
        result_after = mitp2(0.5)
        @test result_before ≈ result_after
    end
end

@testset "MultiCubicInterpolant - Float32 Support" begin
    FI = FastInterpolations

    x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
    y1_32 = sin.(Float32(2π) .* x32)
    y2_32 = cos.(Float32(2π) .* x32)

    mitp32 = cubic_interp(x32, [y1_32, y2_32])

    @testset "Float32 inputs produce Float32 outputs" begin
        result = mitp32(Float32(0.5))
        @test result isa AbstractVector{Float32}
    end

    @testset "Results match Float64 reference (relaxed tolerance)" begin
        x64 = Float64.(x32)
        y1_64 = Float64.(y1_32)
        y2_64 = Float64.(y2_32)
        mitp64 = cubic_interp(x64, [y1_64, y2_64])

        result32 = mitp32(Float32(0.35))
        result64 = mitp64(0.35)

        @test result32[1] ≈ Float32(result64[1]) atol=1f-5
        @test result32[2] ≈ Float32(result64[2]) atol=1f-5
    end
end

@testset "MultiCubicInterpolant - Edge Cases" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Large number of series (10+)" begin
        ys = [sin.(k .* x) for k in 1:12]
        mitp = cubic_interp(x, ys)
        @test FI.n_series(mitp) == 12

        result = mitp(0.5)
        @test length(result) == 12
    end

    @testset "Non-uniform grid works" begin
        x_nu = [0.0, 0.1, 0.15, 0.5, 0.8, 1.0]
        y1_nu = sin.(2π .* x_nu)
        y2_nu = cos.(2π .* x_nu)

        mitp = cubic_interp(x_nu, [y1_nu, y2_nu])
        result = mitp(0.3)
        @test length(result) == 2
    end

    @testset "Query at grid points" begin
        mitp = cubic_interp(x, [y1, y2])
        for xi in [0.0, 0.5, 1.0]
            result = mitp(xi)
            @test length(result) == 2
        end
    end

    @testset "Integer query auto-promoted" begin
        mitp = cubic_interp(x, [y1, y2])
        result = mitp(0)  # Int
        @test result isa AbstractVector{Float64}
    end
end

# ============================================================================
# Phase 5: Zero-Allocation Tests for Derivatives
# ============================================================================

# ============================================================================
# Phase 5b: Type Promotion Tests (coverage for Real type wrappers)
# ============================================================================

@testset "MultiCubicInterpolant - Type Promotion" begin
    FI = FastInterpolations

    @testset "Integer x with Float y vectors (promotes to Float64)" begin
        x_int = collect(1:10)  # Integer vector
        y1 = sin.(Float64.(x_int))
        y2 = cos.(Float64.(x_int))

        mitp = cubic_interp(x_int, [y1, y2])
        @test mitp isa FI.MultiCubicInterpolant{Float64}

        result = mitp(5.5)
        @test length(result) == 2
        @test all(isfinite, result)
    end

    @testset "Integer x with Integer y matrix (promotes to Float64)" begin
        x_int = collect(1:10)  # Integer vector
        Y_int = [i * j for i in 1:10, j in 1:3]  # Integer matrix

        mitp = cubic_interp(x_int, Y_int)
        @test mitp isa FI.MultiCubicInterpolant{Float64}
        @test FI.n_series(mitp) == 3

        result = mitp(5.5)
        @test length(result) == 3
        @test all(isfinite, result)
    end

    @testset "In-place vector with type-promoted xq" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp(x, [y1, y2])

        # Float32 query points with Float64 interpolant
        xq_f32 = Float32[0.1, 0.3, 0.5, 0.7, 0.9]
        out1 = Vector{Float64}(undef, 5)
        out2 = Vector{Float64}(undef, 5)
        outputs = [out1, out2]

        result = mitp(outputs, xq_f32)
        @test result === outputs
        @test all(isfinite, out1)
        @test all(isfinite, out2)

        # Verify values match Float64 path
        xq_f64 = Float64.(xq_f32)
        ref = mitp(xq_f64)
        @test out1 ≈ ref[1] atol=1e-10
        @test out2 ≈ ref[2] atol=1e-10
    end

    @testset "Mixed Integer x and y vectors" begin
        x_int = collect(1:20)
        y1_int = collect(1:20)
        y2_int = collect(20:-1:1)

        mitp = cubic_interp(x_int, [y1_int, y2_int])
        @test mitp isa FI.MultiCubicInterpolant{Float64}

        result = mitp(10.5)
        @test length(result) == 2
    end
end

# ============================================================================
# Phase 6: Zero-Allocation Tests for Derivatives
# ============================================================================

# ============================================================================
# Phase 3: Zero-Allocation Vector API (Pooled Anchors)
# ============================================================================

@testset "MultiCubicInterpolant - Zero-allocation vector API (pooled anchors)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "zero allocation after warmup (same size)" begin
        xq = collect(range(0.1, 0.9, 100))
        out1 = Vector{Float64}(undef, 100)
        out2 = Vector{Float64}(undef, 100)
        out3 = Vector{Float64}(undef, 100)
        outputs = [out1, out2, out3]

        # Warmup calls to establish pool capacity
        mitp(outputs, xq)
        mitp(outputs, xq)

        # Allocation test
        allocs = @allocated mitp(outputs, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "zero allocation for smaller query vectors (pool reuse)" begin
        # First warmup with larger size
        xq_large = collect(range(0.1, 0.9, 100))
        out_large = [Vector{Float64}(undef, 100) for _ in 1:3]
        mitp(out_large, xq_large)
        mitp(out_large, xq_large)

        # Now test with smaller size - should reuse pool capacity
        xq_small = collect(range(0.1, 0.9, 50))
        out_small = [Vector{Float64}(undef, 50) for _ in 1:3]

        # Warmup with small size
        mitp(out_small, xq_small)

        allocs = @allocated mitp(out_small, xq_small)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "bit-wise identical results (in-place vs out-of-place)" begin
        xq = collect(range(0.1, 0.9, 50))

        # Out-of-place result
        result_oop = mitp(xq)

        # In-place result
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]
        mitp(outputs, xq)

        # Bit-wise equality (not approximate)
        @test out1 == result_oop[1]
        @test out2 == result_oop[2]
        @test out3 == result_oop[3]
    end

    @testset "zero allocation with deriv=1" begin
        xq = collect(range(0.1, 0.9, 100))
        out1 = Vector{Float64}(undef, 100)
        out2 = Vector{Float64}(undef, 100)
        out3 = Vector{Float64}(undef, 100)
        outputs = [out1, out2, out3]

        # Warmup
        mitp(outputs, xq; deriv=1)
        mitp(outputs, xq; deriv=1)

        allocs = @allocated mitp(outputs, xq; deriv=1)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "zero allocation with deriv=2" begin
        xq = collect(range(0.1, 0.9, 100))
        out1 = Vector{Float64}(undef, 100)
        out2 = Vector{Float64}(undef, 100)
        out3 = Vector{Float64}(undef, 100)
        outputs = [out1, out2, out3]

        # Warmup
        mitp(outputs, xq; deriv=2)
        mitp(outputs, xq; deriv=2)

        allocs = @allocated mitp(outputs, xq; deriv=2)
        @test allocs <= ALLOC_THRESHOLD
    end
end

@testset "MultiCubicInterpolant - Zero-Allocation Derivative Tests" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = collect(range(0.1, 0.9, 50))

    @testset "Container in-place with derivatives - zero allocation (deriv=1)" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq)

        # Warmup
        mitp(outputs, aq_vec; deriv=1)
        mitp(outputs, aq_vec; deriv=1)

        allocs = @allocated mitp(outputs, aq_vec; deriv=1)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Container in-place with derivatives - zero allocation (deriv=2)" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq)

        # Warmup
        mitp(outputs, aq_vec; deriv=2)
        mitp(outputs, aq_vec; deriv=2)

        allocs = @allocated mitp(outputs, aq_vec; deriv=2)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Derivative correctness with anchors" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        aq_vec = FI._anchor_query(x, xq)

        # Get derivatives via anchored path
        mitp(outputs, aq_vec; deriv=1)

        # Compare with individual interpolants
        itp1 = cubic_interp(x, y1; extrap=:extension)
        itp2 = cubic_interp(x, y2; extrap=:extension)
        itp3 = cubic_interp(x, y3; extrap=:extension)

        expected1 = Vector{Float64}(undef, 50)
        expected2 = Vector{Float64}(undef, 50)
        expected3 = Vector{Float64}(undef, 50)

        itp1(expected1, aq_vec; deriv=1)
        itp2(expected2, aq_vec; deriv=1)
        itp3(expected3, aq_vec; deriv=1)

        @test out1 ≈ expected1 atol=1e-14
        @test out2 ≈ expected2 atol=1e-14
        @test out3 ≈ expected3 atol=1e-14
    end
end
