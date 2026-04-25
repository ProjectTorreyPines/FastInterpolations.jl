# Tests for CubicSeriesInterpolant - Multi-Y cubic interpolation
#
# Multi-Y interpolation: multiple y-data series sharing the same x-grid.
# After unification: Uses matrix storage with adaptive layout for optimal performance.
#
# ALLOC_THRESHOLD is defined in runtests.jl

# ============================================================================
# Phase 1: Unified Type Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testitem "CubicSeriesInterpolant - TransposeSnapshot Type" begin
    using FastInterpolations: _ensure_point_layout!

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

@testitem "CubicSeriesInterpolant - trait implementations" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(0.0:0.1:1.0)
    sitp = cubic_interp(x, Series(sin.(2π .* x), cos.(2π .* x)))

    @test FI.n_series(sitp) == 2
    @test FI._get_grid(sitp) ≈ x
    @test FI._get_extrap(sitp) isa FI.AbstractExtrap
    @test FI._should_wrap(sitp) == false
    @test FI._method_kind(typeof(sitp)) === Val(:cubic)
end

@testitem "CubicSeriesInterpolant - Unified Struct Fields" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3))

    @testset "Has matrix storage fields (not itps)" begin
        # After unification: has matrix storage
        @test hasfield(typeof(mitp), :y)
        @test hasfield(typeof(mitp), :z)
        @test hasfield(typeof(mitp), :cache)
        @test hasfield(typeof(mitp), :_transpose)  # LazyTransposePair for lazy point layout
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

        @test mitp.z[:, 1] ≈ itp1.z rtol = 1.0e-12
        @test mitp.z[:, 2] ≈ itp2.z rtol = 1.0e-12
        @test mitp.z[:, 3] ≈ itp3.z rtol = 1.0e-12
    end

    @testset "Cache is CubicSplineCache" begin
        @test mitp.cache isa FI.CubicSplineCache{Float64}
    end

    @testset "Extrap is AbstractExtrap type" begin
        mitp_ext = cubic_interp(x, Series(y1, y2); extrap = ExtendExtrap())
        @test mitp_ext.extrap === ExtendExtrap()

        mitp_const = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())
        @test mitp_const.extrap === ClampExtrap()
    end
end

@testitem "CubicSeriesInterpolant - Helper Functions" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "n_series and n_points helpers" begin
        mitp = cubic_interp(x, Series(y1, y2, y3))
        @test FI.n_series(mitp) == 3
        @test FI.n_points(mitp) == 101
    end

    @testset "n_series with different counts" begin
        mitp1 = cubic_interp(x, Series(y1))
        @test FI.n_series(mitp1) == 1

        mitp5 = cubic_interp(x, Series(y1, y2, y3, y1, y2))
        @test FI.n_series(mitp5) == 5
    end
end

@testitem "CubicSeriesInterpolant - Type Parameters" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    @testset "Float64 type parameter" begin
        x64 = collect(range(0.0, 1.0, 101))
        y1_64 = sin.(2π .* x64)
        y2_64 = cos.(2π .* x64)

        mitp64 = cubic_interp(x64, Series(y1_64, y2_64))
        @test mitp64 isa FI.CubicSeriesInterpolant{Float64}
    end

    @testset "Float32 type parameter" begin
        x32 = Float32.(collect(range(0.0f0, 1.0f0, 101)))
        y1_32 = sin.(2.0f0 * Float32(π) .* x32)
        y2_32 = cos.(2.0f0 * Float32(π) .* x32)

        mitp32 = cubic_interp(x32, Series(y1_32, y2_32))
        @test mitp32 isa FI.CubicSeriesInterpolant{Float32}
    end
end

# ============================================================================
# Phase 2: Constructor Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testitem "CubicSeriesInterpolant - Constructor Migration" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 51))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "precompute_transpose option" begin
        mitp = cubic_interp(x, Series(y1, y2); precompute_transpose = true)

        # _transpose should be populated immediately
        snap = FI._get_snapshot(mitp._transpose)
        @test snap !== nothing
        y_point, z_point = snap
        @test size(y_point) == (2, length(x))  # (n_series × n_points)
        @test size(z_point) == (2, length(x))
    end

    @testset "Lazy transpose (precompute_transpose=false)" begin
        mitp = cubic_interp(x, Series(y1, y2))  # default precompute_transpose=false

        # Initially empty
        snap = FI._get_snapshot(mitp._transpose)
        @test snap === nothing
    end

    @testset "Range grid preserved in cache" begin
        x_range = range(0.0, 1.0, 51)
        y1_r = sin.(2π .* collect(x_range))
        y2_r = cos.(2π .* collect(x_range))

        mitp = cubic_interp(x_range, Series(y1_r, y2_r))
        @test mitp.cache.x isa AbstractRange
    end

    @testset "PeriodicBC forces WrapExtrap() extrap" begin
        # Create periodic data (endpoints match)
        y_periodic = sin.(2π .* x)
        y_periodic[end] = y_periodic[1]  # Ensure exact periodicity

        mitp = cubic_interp(x, Series(y_periodic); bc = PeriodicBC(), extrap = NoExtrap())
        @test mitp.extrap isa WrapExtrap
        @test !(mitp.extrap isa WrapExtrap{Nothing})

        # Even if user requests :extension, periodic BC should override to :wrap
        mitp2 = cubic_interp(x, Series(y_periodic); bc = PeriodicBC(), extrap = ExtendExtrap())
        @test mitp2.extrap isa WrapExtrap
        @test !(mitp2.extrap isa WrapExtrap{Nothing})
    end
end

# ============================================================================
# Phase 3: Scalar Kernel Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testitem "CubicSeriesInterpolant - Scalar Kernel Migration" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Grid knot exactness" begin
        mitp = cubic_interp(x, Series(y1, y2))
        for (i, xi) in enumerate(x)
            vals = mitp(xi)
            @test vals[1] ≈ y1[i] atol = 1.0e-14
            @test vals[2] ≈ y2[i] atol = 1.0e-14
        end
    end

    @testset "Lazy point-layout triggered by scalar eval" begin
        mitp = cubic_interp(x, Series(y1, y2))

        # Initially empty
        snap = FI._get_snapshot(mitp._transpose)
        @test snap === nothing

        # After scalar eval → populated
        _ = mitp(0.5)
        snap_after = FI._get_snapshot(mitp._transpose)
        @test snap_after !== nothing
        @test size(snap_after[1]) == (2, length(x))  # y_point = snap_after[1]
    end

    @testset "_ensure_point_layout! idempotency" begin
        mitp = cubic_interp(x, Series(y1, y2))
        y_pt1, z_pt1 = _ensure_point_layout!(mitp)
        y_pt2, z_pt2 = _ensure_point_layout!(mitp)
        @test y_pt1 === y_pt2  # Same reference (idempotent)
        @test z_pt1 === z_pt2
    end

    @testset "DimensionMismatch for wrong output size" begin
        mitp = cubic_interp(x, Series(y1, y2))
        out_wrong = zeros(5)  # Wrong size (should be 2)
        @test_throws DimensionMismatch mitp(out_wrong, 0.5)
    end

    @testset "Scalar zero-alloc after precompute" begin
        function measure(x, y1, y2)
            mitp = cubic_interp(x, Series(y1, y2); precompute_transpose = true)
            out = zeros(2)
            mitp(out, 0.5)  # Warmup
            return @allocated mitp(out, 0.5)
        end
        @test measure(x, y1, y2) <= ALLOC_THRESHOLD
    end

    @testset "Invalid deriv=4 throws" begin
        mitp = cubic_interp(x, Series(y1))
        @test_throws TypeError mitp(0.5; deriv = 4)
    end
end

# ============================================================================
# Phase 4: Vector Kernel Migration Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testitem "CubicSeriesInterpolant - Vector Kernel Migration" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Output layout series-first" begin
        mitp = cubic_interp(x, Series(y1, y2))
        xq_vec = [0.2, 0.4, 0.6]
        outputs = mitp(xq_vec)

        for (j, xq) in enumerate(xq_vec)
            scalar_result = mitp(xq)
            @test outputs[1][j] ≈ scalar_result[1] atol = 1.0e-14
            @test outputs[2][j] ≈ scalar_result[2] atol = 1.0e-14
        end
    end

    @testset "DimensionMismatch for wrong container size" begin
        mitp = cubic_interp(x, Series(y1, y2))
        xq_vec = [0.1, 0.3, 0.5]

        outputs_wrong = [zeros(3)]  # Only 1 series (should be 2)
        @test_throws DimensionMismatch mitp(outputs_wrong, xq_vec)
    end

    @testset "Anchor reuse produces identical results" begin
        mitp = cubic_interp(x, Series(y1, y2))
        xq_vec = [0.2, 0.5, 0.8]
        aq_vec = FI._anchor_query(x, xq_vec, Val(:cubic))

        outputs1 = [zeros(3) for _ in 1:2]
        outputs2 = [zeros(3) for _ in 1:2]

        mitp(outputs1, aq_vec)
        mitp(outputs2, aq_vec)

        @test outputs1[1] ≈ outputs2[1] atol = 1.0e-14
        @test outputs1[2] ≈ outputs2[2] atol = 1.0e-14
    end

    @testset "Finite difference derivative verification" begin
        y_poly = x .^ 3 .- 2 .* x .^ 2 .+ x
        mitp = cubic_interp(x, Series(y_poly))
        xq = 0.4
        h = 1.0e-6

        fd1 = (mitp(xq + h) .- mitp(xq - h)) ./ (2h)
        d1 = mitp(xq; deriv = DerivOp(1))
        @test d1 ≈ fd1 atol = 1.0e-5
    end

    @testset "Derivatives with all BC types" begin
        for bc in [ZeroCurvBC(), ZeroSlopeBC()]
            mitp = cubic_interp(x, Series(y1, y2); bc = bc)
            itp1 = cubic_interp(x, y1; bc = bc)

            d1_multi = mitp(0.5; deriv = DerivOp(1))
            @test d1_multi[1] ≈ itp1(0.5; deriv = DerivOp(1)) atol = 1.0e-14
        end

        # PeriodicBC requires matching endpoints
        y1_periodic = sin.(2π .* x)
        y2_periodic = cos.(2π .* x)
        y1_periodic[end] = y1_periodic[1]
        y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, Series(y1_periodic, y2_periodic); bc = PeriodicBC())
        itp1_periodic = cubic_interp(x, y1_periodic; bc = PeriodicBC())

        d1_multi = mitp_periodic(0.5; deriv = DerivOp(1))
        @test d1_multi[1] ≈ itp1_periodic(0.5; deriv = DerivOp(1)) atol = 1.0e-14
    end
end

# ============================================================================
# Phase 5: Memory & Allocation Verification Tests (TDD RED → GREEN → REFACTOR)
# ============================================================================

@testitem "CubicSeriesInterpolant - Memory & Allocation" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Memory footprint - lazy transpose" begin
        ys = [rand(101) for _ in 1:10]
        mitp = cubic_interp(x, Series(ys))

        size_before = Base.summarysize(mitp)
        _ = mitp(0.5)  # Triggers lazy layout creation
        size_after = Base.summarysize(mitp)

        ratio = size_after / size_before
        @test 1.5 < ratio < 2.5  # ~2x for transpose storage
    end

    @testset "_ensure_point_layout! allocates only once" begin
        function measure(x, y1)
            mitp = cubic_interp(x, Series(y1))
            _ensure_point_layout!(mitp)  # First call (warmup)
            # Use direct import to ensure proper escape analysis
            return @allocated _ensure_point_layout!(mitp)
        end
        @test measure(x, y1) <= ALLOC_THRESHOLD
    end

    @testset "Scalar deriv=DerivOp(1) zero-alloc" begin
        function measure(x, y1, y2)
            mitp = cubic_interp(x, Series(y1, y2); precompute_transpose = true)
            out = zeros(2)
            mitp(out, 0.5; deriv = DerivOp(1))  # Warmup
            return @allocated mitp(out, 0.5; deriv = DerivOp(1))
        end
        @test measure(x, y1, y2) <= ALLOC_THRESHOLD
    end

    @testset "Vector deriv=DerivOp(1) zero-alloc with pool" begin
        function measure(x, y1, y2)
            mitp = cubic_interp(x, Series(y1, y2))
            xq = collect(range(0.1, 0.9, 100))
            outputs = [zeros(100) for _ in 1:2]

            mitp(outputs, xq; deriv = DerivOp(1))  # Warmup × 2
            mitp(outputs, xq; deriv = DerivOp(1))

            return @allocated mitp(outputs, xq; deriv = DerivOp(1))
        end
        @test measure(x, y1, y2) <= ALLOC_THRESHOLD
    end
end

# ============================================================================
# Phase 6: Uncovered Path Coverage Tests
# ============================================================================

@testitem "CubicSeriesInterpolant - Uncovered Path Coverage" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "precompute_transpose! function" begin
        mitp = cubic_interp(x, Series(y1, y2))

        # Initially empty
        snap = FI._get_snapshot(mitp._transpose)
        @test snap === nothing

        # Call precompute_transpose!
        result = FI.precompute_transpose!(mitp)
        @test result === mitp

        # Now populated
        snap_after = FI._get_snapshot(mitp._transpose)
        @test snap_after !== nothing
    end

    @testset "WrapExtrap() extrapolation scalar SIMD path" begin
        # Create periodic data (endpoints match)
        y_periodic = sin.(2π .* x)
        y_periodic[end] = y_periodic[1]  # Ensure exact periodicity
        y_periodic2 = cos.(2π .* x)
        y_periodic2[end] = y_periodic2[1]

        mitp = cubic_interp(x, Series(y_periodic, y_periodic2); bc = PeriodicBC())

        # Query outside domain triggers WrapExtrap() extrapolation
        out = zeros(2)
        mitp(out, -0.1)  # Below domain
        @test !any(isnan, out)

        mitp(out, 1.1)   # Above domain
        @test !any(isnan, out)
    end

    @testset "Periodic BC endpoint mismatch error" begin
        y_bad = sin.(2π .* x)
        y_bad[end] = 999.0  # Force mismatch

        @test_throws ArgumentError cubic_interp(x, Series(y_bad); bc = PeriodicBC())
    end

    @testset "precompute_transpose with periodic BC" begin
        y_periodic = sin.(2π .* x)
        y_periodic[end] = y_periodic[1]
        mitp = cubic_interp(x, Series(y_periodic); bc = PeriodicBC(), precompute_transpose = true)

        # Should have point layout immediately
        snap = FI._get_snapshot(mitp._transpose)
        @test snap !== nothing
    end

    @testset "Matrix input with row dimension mismatch" begin
        Y_wrong = rand(50, 3)  # 50 rows but x has 101 points
        @test_throws DimensionMismatch cubic_interp(x, Series(Y_wrong))
    end

    @testset "Matrix input with periodic BC" begin
        Y_periodic = hcat(sin.(2π .* x), cos.(2π .* x))
        Y_periodic[end, 1] = Y_periodic[1, 1]  # Fix endpoint
        Y_periodic[end, 2] = Y_periodic[1, 2]

        mitp = cubic_interp(x, Series(Y_periodic); bc = PeriodicBC())
        @test mitp.extrap isa WrapExtrap
        @test !(mitp.extrap isa WrapExtrap{Nothing})
    end

    @testset "Matrix input with precompute_transpose" begin
        Y = hcat(y1, y2)
        mitp = cubic_interp(x, Series(Y); precompute_transpose = true)

        snap = FI._get_snapshot(mitp._transpose)
        @test snap !== nothing
    end

    @testset "Anchored query path dimension errors" begin
        mitp = cubic_interp(x, Series(y1, y2, sin.(x)))
        xq_vec = [0.1, 0.3, 0.5]
        aq_vec = FI._anchor_query(x, xq_vec, Val(:cubic))

        # Wrong number of output buffers
        outputs_wrong = [zeros(3), zeros(3)]  # 2 instead of 3
        @test_throws DimensionMismatch mitp(outputs_wrong, aq_vec)

        # Wrong buffer length
        outputs_bad_len = [zeros(3), zeros(3), zeros(5)]  # Last has wrong length
        @test_throws DimensionMismatch mitp(outputs_bad_len, aq_vec)
    end

    @testset "Vector extrapolation ExtendExtrap() mode" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = ExtendExtrap())
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq)

        @test !any(isnan, outputs[1])
        @test !any(isnan, outputs[2])
    end

    @testset "Vector extrapolation ClampExtrap() mode" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq)

        # Out-of-domain should clamp to boundary values
        @test outputs[1][1] ≈ y1[1] atol = 1.0e-10  # Below domain → first value
        @test outputs[1][3] ≈ y1[end] atol = 1.0e-10  # Above domain → last value
    end

    @testset "Vector extrapolation WrapExtrap() mode (periodic)" begin
        y_periodic = sin.(2π .* x)
        y_periodic[end] = y_periodic[1]  # Ensure exact periodicity
        y_periodic2 = cos.(2π .* x)
        y_periodic2[end] = y_periodic2[1]

        mitp = cubic_interp(x, Series(y_periodic, y_periodic2); bc = PeriodicBC())
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq)

        @test !any(isnan, outputs[1])
        @test !any(isnan, outputs[2])
    end

    @testset "Scalar extrapolation ClampExtrap() with derivative" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())

        # First derivative outside domain should be zero
        out1 = zeros(2)
        mitp(out1, -0.1; deriv = DerivOp(1))
        @test out1[1] ≈ 0.0 atol = 1.0e-10
        @test out1[2] ≈ 0.0 atol = 1.0e-10

        mitp(out1, 1.1; deriv = DerivOp(1))
        @test out1[1] ≈ 0.0 atol = 1.0e-10
        @test out1[2] ≈ 0.0 atol = 1.0e-10

        # Second derivative outside domain should be zero
        out2 = zeros(2)
        mitp(out2, -0.1; deriv = DerivOp(2))
        @test out2[1] ≈ 0.0 atol = 1.0e-10
        @test out2[2] ≈ 0.0 atol = 1.0e-10

        mitp(out2, 1.1; deriv = DerivOp(2))
        @test out2[1] ≈ 0.0 atol = 1.0e-10
        @test out2[2] ≈ 0.0 atol = 1.0e-10

        # Value should still return boundary value
        out_val = zeros(2)
        mitp(out_val, -0.1)
        @test out_val[1] ≈ y1[1] atol = 1.0e-10
        @test out_val[2] ≈ y2[1] atol = 1.0e-10
    end

    @testset "Vector extrapolation ClampExtrap() with derivative" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())
        xq = [-0.1, 0.5, 1.1]  # Include out-of-domain points

        outputs = [zeros(3), zeros(3)]
        mitp(outputs, xq; deriv = DerivOp(1))

        # Derivatives outside domain should be zero for :constant
        @test outputs[1][1] ≈ 0.0 atol = 1.0e-10
        @test outputs[1][3] ≈ 0.0 atol = 1.0e-10
    end

    @testset "DomainError message includes bounds" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = NoExtrap())

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

@testitem "CubicSeriesInterpolant - Type Structure" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "Construction from Vector of y-series" begin
        # 2-arg form returns CubicSeriesInterpolant
        mitp = cubic_interp(x, Series(y1, y2, y3))
        @test mitp isa FI.CubicSeriesInterpolant

        # Unified structure: has matrix storage
        @test hasfield(typeof(mitp), :y)
        @test FI.n_series(mitp) == 3
    end

    @testset "Construction from Matrix (columns as series)" begin
        Y = hcat(y1, y2, y3)  # 101×3 matrix
        mitp = cubic_interp(x, Series(Y))
        @test mitp isa FI.CubicSeriesInterpolant
        @test FI.n_series(mitp) == 3
    end

    @testset "Single series works" begin
        mitp = cubic_interp(x, Series(y1))
        @test mitp isa FI.CubicSeriesInterpolant
        @test FI.n_series(mitp) == 1
    end

    @testset "Type inference - Float64" begin
        mitp = cubic_interp(x, Series(y1, y2))
        @test mitp isa FI.CubicSeriesInterpolant{Float64}
    end

    @testset "Type inference - Float32" begin
        x32 = Float32.(x)
        y1_32 = Float32.(y1)
        y2_32 = Float32.(y2)
        mitp = cubic_interp(x32, Series(y1_32, y2_32))
        @test mitp isa FI.CubicSeriesInterpolant{Float32}
    end
end

@testitem "CubicSeriesInterpolant - Cache Sharing" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "Single shared cache (unified struct)" begin
        mitp = cubic_interp(x, Series(y1, y2, y3))

        # Unified struct has a single cache field (not itps)
        @test mitp.cache isa FI.CubicSplineCache
        @test mitp.cache.x === x || mitp.cache.x == x
    end

    @testset "Cache sharing with different BC types" begin
        # CubicFit (default)
        mitp_natural = cubic_interp(x, Series(y1, y2))
        @test mitp_natural.cache isa FI.CubicSplineCache
        @test mitp_natural.cache.bc_config isa BCPair

        # PeriodicBC
        y1_periodic = copy(y1); y1_periodic[end] = y1_periodic[1]
        y2_periodic = copy(y2); y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, Series(y1_periodic, y2_periodic); bc = PeriodicBC())
        @test mitp_periodic.cache isa FI.CubicSplineCache
        @test mitp_periodic.cache.bc_config isa FI.PeriodicData
    end
end

@testitem "CubicSeriesInterpolant - Constructor Validation" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Empty ys throws error" begin
        @test_throws ArgumentError cubic_interp(x, Series(Vector{Float64}[]))
    end

    @testset "Mismatched lengths throws error" begin
        y_short = y1[1:50]
        @test_throws Union{DimensionMismatch, ArgumentError, AssertionError} cubic_interp(x, Series(y1, y_short))
    end

    @testset "BC propagation" begin
        # CubicFit (default) - check bc_config field (unified struct)
        mitp_natural = cubic_interp(x, Series(y1, y2))
        @test mitp_natural.cache.bc_config isa BCPair

        # PeriodicBC - check bc_config is PeriodicData
        y1_periodic = copy(y1); y1_periodic[end] = y1_periodic[1]
        y2_periodic = copy(y2); y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, Series(y1_periodic, y2_periodic); bc = PeriodicBC())
        @test mitp_periodic.cache.bc_config isa FastInterpolations.PeriodicData
    end

    @testset "Extrap propagation" begin
        for extrap_mode in (NoExtrap(), ClampExtrap(), ExtendExtrap(), WrapExtrap())
            mitp = cubic_interp(x, Series(y1, y2); extrap = extrap_mode)
            @test mitp.extrap === extrap_mode
        end
    end
end

# ============================================================================
# Phase 2: Scalar Evaluation Tests
# ============================================================================

@testitem "CubicSeriesInterpolant - Scalar Evaluation (out-of-place)" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())

    @testset "Returns Vector with correct values" begin
        result = mitp(0.35)
        @test result isa AbstractVector
        @test length(result) == 3

        # Individual results match standalone interpolants
        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        @test result[1] ≈ itp1(0.35) atol = 1.0e-14
        @test result[2] ≈ itp2(0.35) atol = 1.0e-14
        @test result[3] ≈ itp3(0.35) atol = 1.0e-14
    end

    @testset "Derivatives via deriv keyword" begin
        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        # First derivative
        d1 = mitp(0.5; deriv = DerivOp(1))
        @test d1[1] ≈ itp1(0.5; deriv = DerivOp(1)) atol = 1.0e-14
        @test d1[2] ≈ itp2(0.5; deriv = DerivOp(1)) atol = 1.0e-14
        @test d1[3] ≈ itp3(0.5; deriv = DerivOp(1)) atol = 1.0e-14

        # Second derivative
        d2 = mitp(0.5; deriv = DerivOp(2))
        @test d2[1] ≈ itp1(0.5; deriv = DerivOp(2)) atol = 1.0e-14
        @test d2[2] ≈ itp2(0.5; deriv = DerivOp(2)) atol = 1.0e-14
        @test d2[3] ≈ itp3(0.5; deriv = DerivOp(2)) atol = 1.0e-14
    end

    @testset "Multiple query points" begin
        for xq in [0.0, 0.15, 0.5, 0.85, 1.0]
            result = mitp(xq)
            @test length(result) == 3
        end
    end
end

@testitem "CubicSeriesInterpolant - Scalar Evaluation (in-place)" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())

    @testset "Fills output correctly" begin
        output = Vector{Float64}(undef, 3)
        result = mitp(output, 0.35)

        @test result === output  # Returns same reference

        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        @test output[1] ≈ itp1(0.35) atol = 1.0e-14
        @test output[2] ≈ itp2(0.35) atol = 1.0e-14
        @test output[3] ≈ itp3(0.35) atol = 1.0e-14
    end

    @testset "Size assertion on output mismatch" begin
        output_wrong = Vector{Float64}(undef, 2)  # Wrong size
        @test_throws DimensionMismatch mitp(output_wrong, 0.35)
    end

    @testset "In-place with derivatives" begin
        output = Vector{Float64}(undef, 3)
        mitp(output, 0.5; deriv = DerivOp(1))

        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        @test output[1] ≈ itp1(0.5; deriv = DerivOp(1)) atol = 1.0e-14
    end
end

@testitem "CubicSeriesInterpolant - Scalar Extrap Modes" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "extrap=NoExtrap() - throws DomainError" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = NoExtrap())

        # Inside domain works
        @test isfinite(mitp(0.5)[1])

        # Outside domain throws
        @test_throws DomainError mitp(-0.1)
        @test_throws DomainError mitp(1.1)
    end

    @testset "extrap=ClampExtrap() - boundary values" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = ClampExtrap())

        below = mitp(-0.5)
        @test below[1] ≈ y1[1]
        @test below[2] ≈ y2[1]

        above = mitp(1.5)
        @test above[1] ≈ y1[end]
        @test above[2] ≈ y2[end]
    end

    @testset "extrap=ExtendExtrap() - boundary polynomial" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = ExtendExtrap())
        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())

        below = mitp(-0.1)
        @test below[1] ≈ itp1(-0.1) atol = 1.0e-14
        @test below[2] ≈ itp2(-0.1) atol = 1.0e-14

        above = mitp(1.1)
        @test above[1] ≈ itp1(1.1) atol = 1.0e-14
        @test above[2] ≈ itp2(1.1) atol = 1.0e-14
    end

    @testset "extrap=WrapExtrap() - wrapped coordinates" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = WrapExtrap())

        # Query outside domain should wrap
        result = mitp(1.35)  # wraps to 0.35
        expected1 = cubic_interp(x, y1; extrap = WrapExtrap())(1.35)
        expected2 = cubic_interp(x, y2; extrap = WrapExtrap())(1.35)

        @test result[1] ≈ expected1 atol = 1.0e-14
        @test result[2] ≈ expected2 atol = 1.0e-14
    end
end

# ============================================================================
# Phase 3: Vector Evaluation Tests
# ============================================================================

@testitem "CubicSeriesInterpolant - Vector Evaluation (out-of-place)" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())
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

        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        @test result[1] ≈ itp1(xq) atol = 1.0e-14
        @test result[2] ≈ itp2(xq) atol = 1.0e-14
        @test result[3] ≈ itp3(xq) atol = 1.0e-14
    end

    @testset "Derivatives with vector queries" begin
        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())

        mitp2 = cubic_interp(x, Series(y1, y2); extrap = ExtendExtrap())

        d1 = mitp2(xq; deriv = DerivOp(1))
        @test d1[1] ≈ itp1(xq; deriv = DerivOp(1)) atol = 1.0e-14
        @test d1[2] ≈ itp2(xq; deriv = DerivOp(1)) atol = 1.0e-14

        d2 = mitp2(xq; deriv = DerivOp(2))
        @test d2[1] ≈ itp1(xq; deriv = DerivOp(2)) atol = 1.0e-14
        @test d2[2] ≈ itp2(xq; deriv = DerivOp(2)) atol = 1.0e-14
    end
end

@testitem "CubicSeriesInterpolant - Container In-place (KILLER FEATURE)" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())
    xq = collect(range(0.1, 0.9, 50))

    @testset "Fills all buffers correctly" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        result = mitp(outputs, xq)

        @test result === outputs  # Returns same reference

        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        @test out1 ≈ itp1(xq) atol = 1.0e-14
        @test out2 ≈ itp2(xq) atol = 1.0e-14
        @test out3 ≈ itp3(xq) atol = 1.0e-14
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
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

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
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))
        mitp(outputs, aq_vec; deriv = DerivOp(1))

        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        # Use same anchored path for comparison
        expected = Vector{Float64}(undef, 50)
        itp1(expected, aq_vec; deriv = DerivOp(1))
        @test out1 ≈ expected atol = 1.0e-13
    end
end

@testitem "CubicSeriesInterpolant - Vector Extrap Modes" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "extrap=NoExtrap() - throws on out-of-domain" begin
        mitp = cubic_interp(x, Series(y1, y2); extrap = NoExtrap())
        xq_bad = [-0.1, 0.5, 1.1]

        @test_throws DomainError mitp(xq_bad)
    end

    @testset "All extrap modes work with vector" begin
        for mode in (ClampExtrap(), ExtendExtrap(), WrapExtrap())
            mitp = cubic_interp(x, Series(y1, y2); extrap = mode)
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

@testitem "CubicSeriesInterpolant - Immutability" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    x_original = copy(x)
    y1_original = copy(y1)

    mitp = cubic_interp(x, Series(y1, y2))

    @testset "External x modification has no effect" begin
        result_before = mitp(0.5)
        x[1] = 999.0  # Mutate external x
        result_after = mitp(0.5)
        @test result_before ≈ result_after
    end

    @testset "External y modification has no effect" begin
        x .= x_original  # Restore
        mitp2 = cubic_interp(x, Series(copy(y1), copy(y2)))
        result_before = mitp2(0.5)
        y1[1] = 999.0  # Mutate external y
        result_after = mitp2(0.5)
        @test result_before ≈ result_after
    end
end

@testitem "CubicSeriesInterpolant - Float32 Support" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
    y1_32 = sin.(Float32(2π) .* x32)
    y2_32 = cos.(Float32(2π) .* x32)

    mitp32 = cubic_interp(x32, Series(y1_32, y2_32))

    @testset "Float32 inputs produce Float32 outputs" begin
        result = mitp32(Float32(0.5))
        @test result isa AbstractVector{Float32}
    end

    @testset "Results match Float64 reference (relaxed tolerance)" begin
        x64 = Float64.(x32)
        y1_64 = Float64.(y1_32)
        y2_64 = Float64.(y2_32)
        mitp64 = cubic_interp(x64, Series(y1_64, y2_64))

        result32 = mitp32(Float32(0.35))
        result64 = mitp64(0.35)

        @test result32[1] ≈ Float32(result64[1]) atol = 1.0f-5
        @test result32[2] ≈ Float32(result64[2]) atol = 1.0f-5
    end
end

@testitem "CubicSeriesInterpolant - Edge Cases" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Large number of series (10+)" begin
        ys = [sin.(k .* x) for k in 1:12]
        mitp = cubic_interp(x, Series(ys))
        @test FI.n_series(mitp) == 12

        result = mitp(0.5)
        @test length(result) == 12
    end

    @testset "Non-uniform grid works" begin
        x_nu = [0.0, 0.1, 0.15, 0.5, 0.8, 1.0]
        y1_nu = sin.(2π .* x_nu)
        y2_nu = cos.(2π .* x_nu)

        mitp = cubic_interp(x_nu, Series(y1_nu, y2_nu))
        result = mitp(0.3)
        @test length(result) == 2
    end

    @testset "Query at grid points" begin
        mitp = cubic_interp(x, Series(y1, y2))
        for xi in [0.0, 0.5, 1.0]
            result = mitp(xi)
            @test length(result) == 2
        end
    end

    @testset "Integer query auto-promoted" begin
        mitp = cubic_interp(x, Series(y1, y2))
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

@testitem "CubicSeriesInterpolant - Type Promotion" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    @testset "Integer x with Float y vectors (promotes to Float64)" begin
        x_int = collect(1:10)  # Integer vector
        y1 = sin.(Float64.(x_int))
        y2 = cos.(Float64.(x_int))

        mitp = cubic_interp(x_int, Series(y1, y2))
        @test mitp isa FI.CubicSeriesInterpolant{Float64}

        result = mitp(5.5)
        @test length(result) == 2
        @test all(isfinite, result)
    end

    @testset "Integer x with Integer y matrix (promotes to Float64)" begin
        x_int = collect(1:10)  # Integer vector
        Y_int = [i * j for i in 1:10, j in 1:3]  # Integer matrix

        mitp = cubic_interp(x_int, Series(Y_int))
        @test mitp isa FI.CubicSeriesInterpolant{Float64}
        @test FI.n_series(mitp) == 3

        result = mitp(5.5)
        @test length(result) == 3
        @test all(isfinite, result)
    end

    @testset "In-place vector with type-promoted xq" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = cubic_interp(x, Series(y1, y2))

        # Float32 query points with Float64 interpolant
        xq_f32 = Float32[0.1, 0.3, 0.5, 0.7, 0.9]
        out1 = Vector{Float64}(undef, 5)
        out2 = Vector{Float64}(undef, 5)
        outputs = [out1, out2]

        result = mitp(outputs, xq_f32)
        @test result === outputs
        @test all(isfinite, out1)
        @test all(isfinite, out2)

        # Verify values match same-type allocating path (precision preservation)
        # Float32 in-place should match Float32 allocating, not Float64 allocating
        ref = mitp(xq_f32)
        @test out1 ≈ ref[1] atol = 1.0e-10
        @test out2 ≈ ref[2] atol = 1.0e-10
    end

    @testset "Mixed Integer x and y vectors" begin
        x_int = collect(1:20)
        y1_int = collect(1:20)
        y2_int = collect(20:-1:1)

        mitp = cubic_interp(x_int, Series(y1_int, y2_int))
        @test mitp isa FI.CubicSeriesInterpolant{Float64}

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

@testitem "CubicSeriesInterpolant - Zero-allocation vector API (pooled anchors)" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())

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

    @testset "zero allocation with deriv=DerivOp(1)" begin
        xq = collect(range(0.1, 0.9, 100))
        out1 = Vector{Float64}(undef, 100)
        out2 = Vector{Float64}(undef, 100)
        out3 = Vector{Float64}(undef, 100)
        outputs = [out1, out2, out3]

        # Warmup
        mitp(outputs, xq; deriv = DerivOp(1))
        mitp(outputs, xq; deriv = DerivOp(1))

        allocs = @allocated mitp(outputs, xq; deriv = DerivOp(1))
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "zero allocation with deriv=DerivOp(2)" begin
        xq = collect(range(0.1, 0.9, 100))
        out1 = Vector{Float64}(undef, 100)
        out2 = Vector{Float64}(undef, 100)
        out3 = Vector{Float64}(undef, 100)
        outputs = [out1, out2, out3]

        # Warmup
        mitp(outputs, xq; deriv = DerivOp(2))
        mitp(outputs, xq; deriv = DerivOp(2))

        allocs = @allocated mitp(outputs, xq; deriv = DerivOp(2))
        @test allocs <= ALLOC_THRESHOLD
    end
end

# ============================================================================
# Phase 7: Scalar Extension Extrapolation with Derivatives (Coverage)
# ============================================================================

@testitem "CubicSeriesInterpolant - Scalar Extension Extrapolation with Derivatives" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    # Setup: create interpolant with ExtendExtrap() extrapolation
    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = x .^ 3 .- 2 .* x .^ 2 .+ x  # polynomial for easy derivative verification

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())

    # Create individual interpolants for reference comparison
    itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
    itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
    itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

    @testset "deriv=DerivOp(1) outside domain (left side, xq < 0)" begin
        xq = -0.15  # Outside domain on left

        out = zeros(3)
        mitp(out, xq; deriv = DerivOp(1))

        # Verify against individual interpolants
        @test out[1] ≈ itp1(xq; deriv = DerivOp(1)) atol = 1.0e-12
        @test out[2] ≈ itp2(xq; deriv = DerivOp(1)) atol = 1.0e-12
        @test out[3] ≈ itp3(xq; deriv = DerivOp(1)) atol = 1.0e-12

        # Verify values are finite and non-zero (extension continues the slope)
        @test all(isfinite, out)
    end

    @testset "deriv=DerivOp(1) outside domain (right side, xq > 1)" begin
        xq = 1.15  # Outside domain on right

        out = zeros(3)
        mitp(out, xq; deriv = DerivOp(1))

        # Verify against individual interpolants
        @test out[1] ≈ itp1(xq; deriv = DerivOp(1)) atol = 1.0e-12
        @test out[2] ≈ itp2(xq; deriv = DerivOp(1)) atol = 1.0e-12
        @test out[3] ≈ itp3(xq; deriv = DerivOp(1)) atol = 1.0e-12

        @test all(isfinite, out)
    end

    @testset "deriv=DerivOp(2) outside domain (left side)" begin
        xq = -0.2

        out = zeros(3)
        mitp(out, xq; deriv = DerivOp(2))

        # Verify against individual interpolants
        @test out[1] ≈ itp1(xq; deriv = DerivOp(2)) atol = 1.0e-12
        @test out[2] ≈ itp2(xq; deriv = DerivOp(2)) atol = 1.0e-12
        @test out[3] ≈ itp3(xq; deriv = DerivOp(2)) atol = 1.0e-12

        @test all(isfinite, out)
    end

    @testset "deriv=DerivOp(2) outside domain (right side)" begin
        xq = 1.2

        out = zeros(3)
        mitp(out, xq; deriv = DerivOp(2))

        # Verify against individual interpolants
        @test out[1] ≈ itp1(xq; deriv = DerivOp(2)) atol = 1.0e-12
        @test out[2] ≈ itp2(xq; deriv = DerivOp(2)) atol = 1.0e-12
        @test out[3] ≈ itp3(xq; deriv = DerivOp(2)) atol = 1.0e-12

        @test all(isfinite, out)
    end

    @testset "deriv=DerivOp(3) outside domain (left side)" begin
        xq = -0.1

        out = zeros(3)
        mitp(out, xq; deriv = DerivOp(3))

        # Verify against individual interpolants
        @test out[1] ≈ itp1(xq; deriv = DerivOp(3)) atol = 1.0e-12
        @test out[2] ≈ itp2(xq; deriv = DerivOp(3)) atol = 1.0e-12
        @test out[3] ≈ itp3(xq; deriv = DerivOp(3)) atol = 1.0e-12

        @test all(isfinite, out)
    end

    @testset "deriv=DerivOp(3) outside domain (right side)" begin
        xq = 1.1

        out = zeros(3)
        mitp(out, xq; deriv = DerivOp(3))

        # Verify against individual interpolants
        @test out[1] ≈ itp1(xq; deriv = DerivOp(3)) atol = 1.0e-12
        @test out[2] ≈ itp2(xq; deriv = DerivOp(3)) atol = 1.0e-12
        @test out[3] ≈ itp3(xq; deriv = DerivOp(3)) atol = 1.0e-12

        @test all(isfinite, out)
    end

    @testset "Out-of-place scalar extension derivatives" begin
        # Test out-of-place API as well (covers the allocation path)
        xq_left = -0.1
        xq_right = 1.1

        # deriv=1
        d1_left = mitp(xq_left; deriv = DerivOp(1))
        d1_right = mitp(xq_right; deriv = DerivOp(1))
        @test length(d1_left) == 3
        @test length(d1_right) == 3
        @test d1_left[1] ≈ itp1(xq_left; deriv = DerivOp(1)) atol = 1.0e-12

        # deriv=2
        d2_left = mitp(xq_left; deriv = DerivOp(2))
        d2_right = mitp(xq_right; deriv = DerivOp(2))
        @test length(d2_left) == 3
        @test d2_left[2] ≈ itp2(xq_left; deriv = DerivOp(2)) atol = 1.0e-12

        # deriv=3
        d3_left = mitp(xq_left; deriv = DerivOp(3))
        d3_right = mitp(xq_right; deriv = DerivOp(3))
        @test length(d3_left) == 3
        @test d3_right[3] ≈ itp3(xq_right; deriv = DerivOp(3)) atol = 1.0e-12
    end

    @testset "Polynomial accuracy check (extension preserves cubic)" begin
        # For y3 = x³ - 2x² + x, we know exact derivatives:
        # y3' = 3x² - 4x + 1
        # y3'' = 6x - 4
        # y3''' = 6

        # Outside domain, extension uses boundary polynomial
        # The cubic spline should approximate the polynomial well

        # At interior point for sanity check
        xq_interior = 0.5
        d1_interior = mitp(xq_interior; deriv = DerivOp(1))[3]
        d2_interior = mitp(xq_interior; deriv = DerivOp(2))[3]
        d3_interior = mitp(xq_interior; deriv = DerivOp(3))[3]

        exact_d1 = 3 * xq_interior^2 - 4 * xq_interior + 1
        exact_d2 = 6 * xq_interior - 4
        exact_d3 = 6.0

        # Interior should be very accurate
        @test d1_interior ≈ exact_d1 atol = 1.0e-10
        @test d2_interior ≈ exact_d2 atol = 1.0e-10
        @test d3_interior ≈ exact_d3 atol = 1.0e-8  # Third derivative has larger error
    end
end

@testitem "CubicSeriesInterpolant - Zero-Allocation Derivative Tests" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())
    xq = collect(range(0.1, 0.9, 50))

    @testset "Container in-place with derivatives - zero allocation (deriv=DerivOp(1))" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        # Warmup
        mitp(outputs, aq_vec; deriv = DerivOp(1))
        mitp(outputs, aq_vec; deriv = DerivOp(1))

        allocs = @allocated mitp(outputs, aq_vec; deriv = DerivOp(1))
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Container in-place with derivatives - zero allocation (deriv=DerivOp(2))" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        # Warmup
        mitp(outputs, aq_vec; deriv = DerivOp(2))
        mitp(outputs, aq_vec; deriv = DerivOp(2))

        allocs = @allocated mitp(outputs, aq_vec; deriv = DerivOp(2))
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Derivative correctness with anchors" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        # Get derivatives via anchored path
        mitp(outputs, aq_vec; deriv = DerivOp(1))

        # Compare with individual interpolants
        itp1 = cubic_interp(x, y1; extrap = ExtendExtrap())
        itp2 = cubic_interp(x, y2; extrap = ExtendExtrap())
        itp3 = cubic_interp(x, y3; extrap = ExtendExtrap())

        expected1 = Vector{Float64}(undef, 50)
        expected2 = Vector{Float64}(undef, 50)
        expected3 = Vector{Float64}(undef, 50)

        itp1(expected1, aq_vec; deriv = DerivOp(1))
        itp2(expected2, aq_vec; deriv = DerivOp(1))
        itp3(expected3, aq_vec; deriv = DerivOp(1))

        @test out1 ≈ expected1 atol = 1.0e-13
        @test out2 ≈ expected2 atol = 1.0e-13
        @test out3 ≈ expected3 atol = 1.0e-13
    end
end

# ============================================================================
# Per-Series Boundary Conditions Tests
# ============================================================================

@testitem "CubicSeriesInterpolant - Per-Series BC" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    @testset "Same BC type, different values - linear functions" begin
        x = collect(range(0.0, 1.0, 11))

        # Linear functions with different slopes
        y1 = 1.0 .* x      # slope = 1
        y2 = 2.0 .* x      # slope = 2
        y3 = 3.0 .* x      # slope = 3

        # Each series uses its actual slope as BC
        sitp = cubic_interp(
            x, Series(y1, y2, y3); bc = [
                BCPair(Deriv1(1.0), Deriv1(1.0)),
                BCPair(Deriv1(2.0), Deriv1(2.0)),
                BCPair(Deriv1(3.0), Deriv1(3.0)),
            ]
        )

        # Linear functions should be exactly reproduced
        @test sitp(0.5) ≈ [0.5, 1.0, 1.5] atol = 1.0e-14

        # Test at multiple points
        for xq in [0.0, 0.25, 0.5, 0.75, 1.0]
            result = sitp(xq)
            @test result[1] ≈ 1.0 * xq atol = 1.0e-14
            @test result[2] ≈ 2.0 * xq atol = 1.0e-14
            @test result[3] ≈ 3.0 * xq atol = 1.0e-14
        end
    end

    @testset "Mixed BC types" begin
        x = collect(range(0.0, 1.0, 21))
        y1 = sin.(π .* x)
        y2 = cos.(π .* x)
        y3 = x .^ 2

        # Different BC types for each series
        sitp = cubic_interp(
            x, Series(y1, y2, y3); bc = [
                ZeroCurvBC(),                          # Zero-Curvature BC
                BCPair(Deriv1(0.0), Deriv1(0.0)),      # Zero-Slope BC
                BCPair(Deriv2(2.0), Deriv2(2.0)),      # Constant curvature = 2
            ]
        )

        # Each series should be computed correctly
        @test sitp(0.5) isa Vector{Float64}
        @test length(sitp(0.5)) == 3

        # Compare with individual interpolants using same BC
        itp1 = cubic_interp(x, y1; bc = ZeroCurvBC())
        itp2 = cubic_interp(x, y2; bc = BCPair(Deriv1(0.0), Deriv1(0.0)))
        itp3 = cubic_interp(x, y3; bc = BCPair(Deriv2(2.0), Deriv2(2.0)))

        for xq in [0.1, 0.3, 0.5, 0.7, 0.9]
            result = sitp(xq)
            @test result[1] ≈ itp1(xq) atol = 1.0e-14
            @test result[2] ≈ itp2(xq) atol = 1.0e-14
            @test result[3] ≈ itp3(xq) atol = 1.0e-14
        end
    end

    @testset "Matrix input with BC array" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = 1.0 .* x
        y2 = 2.0 .* x
        Y = hcat(y1, y2)  # 11×2 matrix

        sitp = cubic_interp(
            x, Series(Y); bc = [
                BCPair(Deriv1(1.0), Deriv1(1.0)),
                BCPair(Deriv1(2.0), Deriv1(2.0)),
            ]
        )

        @test sitp(0.5) ≈ [0.5, 1.0] atol = 1.0e-14
    end

    @testset "BC array with Deriv2 and Deriv3" begin
        x = collect(range(0.0, 1.0, 21))

        # Quadratic: y = x^2, y'' = 2
        y_quad = x .^ 2

        # Cubic: y = x^3, y''' = 6
        y_cubic = x .^ 3

        sitp = cubic_interp(
            x, Series(y_quad, y_cubic); bc = [
                BCPair(Deriv2(2.0), Deriv2(2.0)),      # Quadratic's curvature
                BCPair(Deriv3(6.0), Deriv3(6.0)),      # Cubic's third derivative
            ]
        )

        # Verify results match individual interpolants
        itp_quad = cubic_interp(x, y_quad; bc = BCPair(Deriv2(2.0), Deriv2(2.0)))
        itp_cubic = cubic_interp(x, y_cubic; bc = BCPair(Deriv3(6.0), Deriv3(6.0)))

        for xq in [0.2, 0.5, 0.8]
            result = sitp(xq)
            @test result[1] ≈ itp_quad(xq) atol = 1.0e-13
            @test result[2] ≈ itp_cubic(xq) atol = 1.0e-13
        end
    end

    @testset "BC array length validation" begin
        x = collect(range(0.0, 1.0, 11))
        y1, y2 = sin.(x), cos.(x)

        # Length mismatch → DimensionMismatch
        @test_throws DimensionMismatch cubic_interp(x, Series(y1, y2); bc = [ZeroCurvBC()])
        @test_throws DimensionMismatch cubic_interp(x, Series(y1, y2); bc = [ZeroCurvBC(), ZeroCurvBC(), ZeroCurvBC()])
    end

    @testset "PeriodicBC in array - not supported" begin
        x = collect(range(0.0, 1.0, 11))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        # PeriodicBC not supported in arrays
        @test_throws ArgumentError cubic_interp(
            x, Series(y1, y2); bc = [
                PeriodicBC(),
                ZeroCurvBC(),
            ]
        )
    end

    @testset "Derivatives with per-series BC" begin
        x = collect(range(0.0, 1.0, 21))
        y1 = 1.0 .* x      # slope = 1
        y2 = 2.0 .* x      # slope = 2

        sitp = cubic_interp(
            x, Series(y1, y2); bc = [
                BCPair(Deriv1(1.0), Deriv1(1.0)),
                BCPair(Deriv1(2.0), Deriv1(2.0)),
            ]
        )

        # First derivative of linear function = slope
        d1 = sitp(0.5; deriv = DerivOp(1))
        @test d1[1] ≈ 1.0 atol = 1.0e-12
        @test d1[2] ≈ 2.0 atol = 1.0e-12

        # Second derivative of linear function = 0
        d2 = sitp(0.5; deriv = DerivOp(2))
        @test d2[1] ≈ 0.0 atol = 1.0e-10
        @test d2[2] ≈ 0.0 atol = 1.0e-10
    end

    # =========================================================================
    # Zero-Allocation Tests for Per-Series BC
    # =========================================================================
    # BC is only used at construction time. Evaluation should be zero-allocation
    # regardless of whether single BC or BC array was used.

    @testset "Zero-allocation: scalar query with per-series BC" begin
        function measure()
            x = collect(range(0.0, 1.0, 101))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)
            y3 = exp.(-3 .* x)

            # Create with per-series BC (mixed types)
            sitp = cubic_interp(
                x, Series(y1, y2, y3); bc = [
                    ZeroCurvBC(),
                    BCPair(Deriv1(0.0), Deriv1(0.0)),
                    BCPair(Deriv2(0.0), Deriv2(0.0)),
                ], precompute_transpose = true
            )

            out = zeros(3)

            # Warmup
            sitp(out, 0.5)

            # Scalar query - zero allocation
            return @allocated sitp(out, 0.5)
        end
        @test measure() <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: vector query with per-series BC" begin
        function measure()
            x = collect(range(0.0, 1.0, 101))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Create with same BC type, different values
            sitp = cubic_interp(
                x, Series(y1, y2); bc = [
                    BCPair(Deriv1(1.0), Deriv1(-1.0)),
                    BCPair(Deriv1(0.5), Deriv1(-0.5)),
                ]
            )

            xq = collect(range(0.1, 0.9, 50))
            out1 = Vector{Float64}(undef, 50)
            out2 = Vector{Float64}(undef, 50)
            outputs = [out1, out2]

            # Warmup
            sitp(outputs, xq)
            sitp(outputs, xq)

            # Vector query - zero allocation
            return @allocated sitp(outputs, xq)
        end
        @test measure() <= ALLOC_THRESHOLD
    end

    @testset "Zero-allocation: derivatives with per-series BC" begin
        function measure_d1()
            x = collect(range(0.0, 1.0, 101))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Create with mixed BC types
            sitp = cubic_interp(
                x, Series(y1, y2); bc = [
                    ZeroCurvBC(),
                    BCPair(Deriv1(0.0), Deriv1(0.0)),
                ], precompute_transpose = true
            )

            out = zeros(2)

            # Warmup deriv=1
            sitp(out, 0.5; deriv = DerivOp(1))

            return @allocated sitp(out, 0.5; deriv = DerivOp(1))
        end
        @test measure_d1() <= ALLOC_THRESHOLD

        function measure_d2()
            x = collect(range(0.0, 1.0, 101))
            y1 = sin.(2π .* x)
            y2 = cos.(2π .* x)

            # Create with mixed BC types
            sitp = cubic_interp(
                x, Series(y1, y2); bc = [
                    ZeroCurvBC(),
                    BCPair(Deriv1(0.0), Deriv1(0.0)),
                ], precompute_transpose = true
            )

            out = zeros(2)

            # Warmup deriv=2
            sitp(out, 0.5; deriv = DerivOp(2))

            return @allocated sitp(out, 0.5; deriv = DerivOp(2))
        end
        @test measure_d2() <= ALLOC_THRESHOLD
    end

    @testset "Fast path: BCPair array normalization is zero-allocation" begin
        # When user provides BCPair array directly, _normalize_bc_array
        # should return the same array without allocation (identity fast path)
        bc_pairs = [
            BCPair(Deriv1(0.0), Deriv1(0.0)),
            BCPair(Deriv2(0.0), Deriv2(0.0)),
            BCPair(Deriv1(1.0), Deriv3(0.0)),
        ]

        # Warmup
        FastInterpolations._normalize_bc_array(bc_pairs, Float64, 3)
        FastInterpolations._normalize_bc_array(bc_pairs, Float64, 3)

        # Should return the same object (identity)
        result = FastInterpolations._normalize_bc_array(bc_pairs, Float64, 3)
        @test result === bc_pairs  # Same object reference, not a copy

        # Zero allocation
        allocs = @allocated FastInterpolations._normalize_bc_array(bc_pairs, Float64, 3)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "General path: mixed BC array creates new Vector{BCPair}" begin
        # When BC types are mixed, normalization must create new array
        # Note: With the new type system, singleton BCs are AbstractBC{Nothing},
        # so we use a concrete type array or Any
        mixed_bcs = [
            ZeroCurvBC(),
            BCPair(Deriv1(0.0), Deriv2(3.0)),
            ZeroSlopeBC(),
        ]

        result = FastInterpolations._normalize_bc_array(mixed_bcs, Float64, 3)

        # Should be a new Vector{<:BCPair} (Type-Free design)
        @test result isa Vector{<:BCPair}
        @test result !== mixed_bcs  # Different object

        # Values should be correctly normalized (uses Julia's default structural equality for BCPair)
        @test result[1] == BCPair(Deriv2(0.0), Deriv2(0.0))  # ZeroCurvBC
        @test result[2] == BCPair(Deriv1(0.0), Deriv2(3.0))  # Already BCPair
        @test result[3] == BCPair(Deriv1(0.0), Deriv1(0.0))  # ZeroSlopeBC
    end
end

# ============================================================================
# Pre-built Anchor Tests for Coverage (deriv=DerivOp(3) and extrapolation)
# ============================================================================

@testitem "CubicSeriesInterpolant - Pre-built Anchor with deriv=DerivOp(3)" setup=[AllocConstants] begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    @testset "deriv=DerivOp(3) correctness with pre-built anchors" begin
        x = collect(range(0.0, 1.0, 21))
        # Use polynomial functions where 3rd derivative is analytically known
        y1 = x .^ 4       # d³/dx³ = 24x
        y2 = x .^ 3       # d³/dx³ = 6
        y3 = sin.(2π .* x)  # d³/dx³ = -8π³cos(2πx)

        mitp = cubic_interp(x, Series(y1, y2, y3))
        xq = [0.2, 0.4, 0.6, 0.8]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        outputs = [similar(xq) for _ in 1:3]
        mitp(outputs, aq_vec; deriv = DerivOp(3))

        # Compare with scalar evaluation
        for (j, q) in enumerate(xq)
            scalar_result = mitp(q; deriv = DerivOp(3))
            @test outputs[1][j] ≈ scalar_result[1] atol = 1.0e-10
            @test outputs[2][j] ≈ scalar_result[2] atol = 1.0e-10
            @test outputs[3][j] ≈ scalar_result[3] atol = 1.0e-10
        end
    end

    @testset "deriv=DerivOp(3) zero allocation with pre-built anchors" begin
        function measure()
            x = collect(range(0.0, 1.0, 51))
            y1 = x .^ 4
            y2 = x .^ 3

            mitp = cubic_interp(x, Series(y1, y2))
            xq = collect(range(0.1, 0.9, 20))

            # Pre-build anchors
            aq_vec = FI._anchor_query(x, xq, Val(:cubic))
            outputs = [similar(xq) for _ in 1:2]

            # Warmup
            mitp(outputs, aq_vec; deriv = DerivOp(3))
            mitp(outputs, aq_vec; deriv = DerivOp(3))

            return @allocated mitp(outputs, aq_vec; deriv = DerivOp(3))
        end
        @test measure() <= ALLOC_THRESHOLD
    end

    @testset "deriv=DerivOp(3) matches individual interpolants" begin
        x = collect(range(0.0, 1.0, 31))
        y1 = exp.(-x)
        y2 = x .^ 5

        mitp = cubic_interp(x, Series(y1, y2))
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)

        xq = [0.15, 0.35, 0.55, 0.75, 0.95]
        aq_vec = FI._anchor_query(x, xq, Val(:cubic))

        outputs = [similar(xq) for _ in 1:2]
        expected1 = similar(xq)
        expected2 = similar(xq)

        mitp(outputs, aq_vec; deriv = DerivOp(3))
        itp1(expected1, aq_vec; deriv = DerivOp(3))
        itp2(expected2, aq_vec; deriv = DerivOp(3))

        @test outputs[1] ≈ expected1 atol = 1.0e-13
        @test outputs[2] ≈ expected2 atol = 1.0e-13
    end
end

@testitem "CubicSeriesInterpolant - Pre-built Anchor with Extrapolation" begin
    using FastInterpolations: _ensure_point_layout!

    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 11))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-x)

    # Query points including out-of-domain
    xq_out = [-0.15, -0.05, 0.5, 1.05, 1.15]

    @testset ":extension mode with pre-built anchors" begin
        mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ExtendExtrap())

        # Pre-build anchors (will have state != IN_DOMAIN for out-of-domain points)
        aq_vec = FI._anchor_query(x, xq_out, Val(:cubic))
        outputs = [similar(xq_out) for _ in 1:3]

        # Should not throw, uses polynomial extension
        mitp(outputs, aq_vec)

        # Compare with scalar evaluation
        for (j, q) in enumerate(xq_out)
            scalar_result = mitp(q)
            @test outputs[1][j] ≈ scalar_result[1] atol = 1.0e-10
            @test outputs[2][j] ≈ scalar_result[2] atol = 1.0e-10
            @test outputs[3][j] ≈ scalar_result[3] atol = 1.0e-10
        end

        # Test derivatives with extension
        for deriv in [DerivOp(1), DerivOp(2), DerivOp(3)]
            mitp(outputs, aq_vec; deriv = deriv)
            for (j, q) in enumerate(xq_out)
                scalar_result = mitp(q; deriv = deriv)
                @test outputs[1][j] ≈ scalar_result[1] atol = 1.0e-10
                @test outputs[2][j] ≈ scalar_result[2] atol = 1.0e-10
                @test outputs[3][j] ≈ scalar_result[3] atol = 1.0e-10
            end
        end
    end

    @testset ":constant mode with pre-built anchors" begin
        mitp = cubic_interp(x, Series(y1, y2, y3); extrap = ClampExtrap())

        aq_vec = FI._anchor_query(x, xq_out, Val(:cubic))
        outputs = [similar(xq_out) for _ in 1:3]

        mitp(outputs, aq_vec)

        # Out-of-domain points should be clamped to boundary values
        # Left boundary (indices 1, 2 are < 0.0)
        @test outputs[1][1] ≈ y1[1] atol = 1.0e-10
        @test outputs[1][2] ≈ y1[1] atol = 1.0e-10
        @test outputs[2][1] ≈ y2[1] atol = 1.0e-10
        @test outputs[2][2] ≈ y2[1] atol = 1.0e-10
        @test outputs[3][1] ≈ y3[1] atol = 1.0e-10
        @test outputs[3][2] ≈ y3[1] atol = 1.0e-10

        # Right boundary (indices 4, 5 are > 1.0)
        @test outputs[1][4] ≈ y1[end] atol = 1.0e-10
        @test outputs[1][5] ≈ y1[end] atol = 1.0e-10
        @test outputs[2][4] ≈ y2[end] atol = 1.0e-10
        @test outputs[2][5] ≈ y2[end] atol = 1.0e-10
        @test outputs[3][4] ≈ y3[end] atol = 1.0e-10
        @test outputs[3][5] ≈ y3[end] atol = 1.0e-10

        # Inside domain should be normal interpolation
        inside_result = mitp(0.5)
        @test outputs[1][3] ≈ inside_result[1] atol = 1.0e-10
        @test outputs[2][3] ≈ inside_result[2] atol = 1.0e-10
        @test outputs[3][3] ≈ inside_result[3] atol = 1.0e-10
    end

    @testset ":none mode with pre-built anchors throws DomainError" begin
        mitp = cubic_interp(x, Series(y1, y2, y3); extrap = NoExtrap())

        aq_vec = FI._anchor_query(x, xq_out, Val(:cubic))
        outputs = [similar(xq_out) for _ in 1:3]

        @test_throws DomainError mitp(outputs, aq_vec)
    end

    @testset ":wrap mode with pre-built anchors (PeriodicBC)" begin
        # Periodic function
        x_per = collect(range(0.0, 2π, 21))
        y1_per = sin.(x_per)
        y2_per = cos.(x_per)
        y1_per[end] = y1_per[1]
        y2_per[end] = y2_per[1]

        mitp = cubic_interp(x_per, Series(y1_per, y2_per); bc = PeriodicBC())

        # Query points outside domain should wrap
        xq_wrap = [-0.5, 0.5, 2π + 0.5, 4π + 0.5]
        aq_vec = FI._anchor_query(x_per, xq_wrap, Val(:cubic), true)
        outputs = [similar(xq_wrap) for _ in 1:2]

        mitp(outputs, aq_vec)

        # Wrapped queries should match equivalent in-domain queries
        for (j, q) in enumerate(xq_wrap)
            scalar_result = mitp(q)
            @test outputs[1][j] ≈ scalar_result[1] atol = 1.0e-10
            @test outputs[2][j] ≈ scalar_result[2] atol = 1.0e-10
        end
    end
end
