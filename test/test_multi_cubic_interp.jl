# Tests for MultiCubicInterpolant - Multi-Y cubic interpolation
#
# Multi-Y interpolation: multiple y-data series sharing the same x-grid.
# Uses composition approach: wraps existing CubicInterpolant objects.
#
# ALLOC_THRESHOLD is defined in runtests.jl

# ============================================================================
# Phase 1: Type Definition & Constructor Tests
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

        # Access internal interpolants
        @test hasfield(typeof(mitp), :itps)
        @test length(mitp.itps) == 3
        @test all(itp -> itp isa CubicInterpolant, mitp.itps)
    end

    @testset "Construction from Matrix (columns as series)" begin
        Y = hcat(y1, y2, y3)  # 101×3 matrix
        mitp = cubic_interp(x, Y)
        @test mitp isa FI.MultiCubicInterpolant
        @test length(mitp.itps) == 3
    end

    @testset "Single series works" begin
        mitp = cubic_interp(x, [y1])
        @test mitp isa FI.MultiCubicInterpolant
        @test length(mitp.itps) == 1
    end

    @testset "Type inference - Float64" begin
        mitp = cubic_interp(x, [y1, y2])
        @test eltype(mitp.itps) <: CubicInterpolant{Float64}
    end

    @testset "Type inference - Float32" begin
        x32 = Float32.(x)
        y1_32 = Float32.(y1)
        y2_32 = Float32.(y2)
        mitp = cubic_interp(x32, [y1_32, y2_32])
        @test eltype(mitp.itps) <: CubicInterpolant{Float32}
    end
end

@testset "MultiCubicInterpolant - Cache Sharing" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "All interpolants share same cache reference" begin
        mitp = cubic_interp(x, [y1, y2, y3])

        # Use === for reference equality (not just value equality)
        @test mitp.itps[1].cache === mitp.itps[2].cache
        @test mitp.itps[2].cache === mitp.itps[3].cache
    end

    @testset "Cache sharing with different BC types" begin
        # NaturalBC (default)
        mitp_natural = cubic_interp(x, [y1, y2])
        @test mitp_natural.itps[1].cache === mitp_natural.itps[2].cache

        # PeriodicBC
        y1_periodic = copy(y1); y1_periodic[end] = y1_periodic[1]
        y2_periodic = copy(y2); y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, [y1_periodic, y2_periodic]; bc=PeriodicBC())
        @test mitp_periodic.itps[1].cache === mitp_periodic.itps[2].cache
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
        # NaturalBC (default) - check bc_config field
        mitp_natural = cubic_interp(x, [y1, y2])
        @test mitp_natural.itps[1].cache.bc_config isa BCPair

        # PeriodicBC - check bc_config is PeriodicData
        y1_periodic = copy(y1); y1_periodic[end] = y1_periodic[1]
        y2_periodic = copy(y2); y2_periodic[end] = y2_periodic[1]
        mitp_periodic = cubic_interp(x, [y1_periodic, y2_periodic]; bc=PeriodicBC())
        @test mitp_periodic.itps[1].cache.bc_config isa FastInterpolations.PeriodicData
    end

    @testset "Extrap propagation" begin
        for extrap_mode in (:none, :constant, :extension, :wrap)
            mitp = cubic_interp(x, [y1, y2]; extrap=extrap_mode)
            @test mitp.itps[1].extrap === Val(extrap_mode)
            @test mitp.itps[2].extrap === Val(extrap_mode)
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
        @test_throws AssertionError mitp(output_wrong, 0.35)
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
        @test_throws AssertionError mitp(outputs_wrong, xq)
    end

    @testset "Size assertion on buffer mismatch" begin
        outputs = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50), Vector{Float64}(undef, 30)]  # Last is wrong size
        @test_throws AssertionError mitp(outputs, xq)
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
        @test length(mitp.itps) == 12

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
        @test length(mitp.itps) == 3

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
