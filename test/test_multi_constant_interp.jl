# Tests for ConstantMultiInterpolant - Multi-Y constant interpolation
#
# Multi-Y interpolation: multiple y-data series sharing the same x-grid.
# Uses composition approach: wraps existing ConstantInterpolant objects.
#
# ALLOC_THRESHOLD is defined in runtests.jl

# ============================================================================
# Phase 1: Type Definition & Constructor Tests
# ============================================================================

@testset "ConstantMultiInterpolant - Type Structure" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "Construction from Vector of y-series" begin
        mitp = constant_interp(x, [y1, y2, y3])
        @test mitp isa FI.ConstantMultiInterpolant

        @test hasfield(typeof(mitp), :itps)
        @test length(mitp.itps) == 3
        @test all(itp -> itp isa ConstantInterpolant, mitp.itps)
    end

    @testset "Construction from Matrix (columns as series)" begin
        Y = hcat(y1, y2, y3)
        mitp = constant_interp(x, Y)
        @test mitp isa FI.ConstantMultiInterpolant
        @test length(mitp.itps) == 3
    end

    @testset "Single series works" begin
        mitp = constant_interp(x, [y1])
        @test mitp isa FI.ConstantMultiInterpolant
        @test length(mitp.itps) == 1
    end

    @testset "Type inference - Float64" begin
        mitp = constant_interp(x, [y1, y2])
        @test eltype(mitp.itps) <: ConstantInterpolant{Float64}
    end

    @testset "Type inference - Float32" begin
        x32 = Float32.(x)
        y1_32 = Float32.(y1)
        y2_32 = Float32.(y2)
        mitp = constant_interp(x32, [y1_32, y2_32])
        @test eltype(mitp.itps) <: ConstantInterpolant{Float32}
    end

    @testset "Subtypes AbstractMultiInterpolant" begin
        mitp = constant_interp(x, [y1, y2])
        @test mitp isa AbstractMultiInterpolant{Float64}
    end
end

@testset "ConstantMultiInterpolant - Constructor Validation" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Empty ys throws error" begin
        @test_throws Union{ArgumentError, AssertionError} constant_interp(x, Vector{Float64}[])
    end

    @testset "Mismatched lengths throws error" begin
        y_short = y1[1:50]
        @test_throws Union{DimensionMismatch, ArgumentError, AssertionError} constant_interp(x, [y1, y_short])
    end

    @testset "Extrap propagation" begin
        for extrap_mode in (:none, :constant, :extension, :wrap)
            mitp = constant_interp(x, [y1, y2]; extrap=extrap_mode)
            @test mitp.itps[1].mode === Val(extrap_mode)
            @test mitp.itps[2].mode === Val(extrap_mode)
        end
    end

    @testset "Side propagation" begin
        for side_mode in (:left, :right, :nearest)
            mitp = constant_interp(x, [y1, y2]; side=side_mode)
            @test mitp.itps[1].side === Val(side_mode)
            @test mitp.itps[2].side === Val(side_mode)
        end
    end
end

# ============================================================================
# Phase 2: Scalar Evaluation Tests
# ============================================================================

@testset "ConstantMultiInterpolant - Scalar Evaluation (out-of-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = constant_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "Returns Vector with correct values" begin
        result = mitp(0.35)
        @test result isa AbstractVector
        @test length(result) == 3

        itp1 = constant_interp(x, y1; extrap=:extension)
        itp2 = constant_interp(x, y2; extrap=:extension)
        itp3 = constant_interp(x, y3; extrap=:extension)

        @test result[1] ≈ itp1(0.35) atol=1e-14
        @test result[2] ≈ itp2(0.35) atol=1e-14
        @test result[3] ≈ itp3(0.35) atol=1e-14
    end

    @testset "Multiple query points" begin
        for xq in [0.0, 0.15, 0.5, 0.85, 1.0]
            result = mitp(xq)
            @test length(result) == 3
        end
    end
end

@testset "ConstantMultiInterpolant - Scalar Evaluation (in-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = constant_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "Fills output correctly" begin
        output = Vector{Float64}(undef, 3)
        result = mitp(output, 0.35)

        @test result === output

        itp1 = constant_interp(x, y1; extrap=:extension)
        itp2 = constant_interp(x, y2; extrap=:extension)
        itp3 = constant_interp(x, y3; extrap=:extension)

        @test output[1] ≈ itp1(0.35) atol=1e-14
        @test output[2] ≈ itp2(0.35) atol=1e-14
        @test output[3] ≈ itp3(0.35) atol=1e-14
    end

    @testset "Size assertion on output mismatch" begin
        output_wrong = Vector{Float64}(undef, 2)
        @test_throws AssertionError mitp(output_wrong, 0.35)
    end
end

@testset "ConstantMultiInterpolant - Scalar Extrap Modes" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "extrap=:none - throws DomainError" begin
        mitp = constant_interp(x, [y1, y2]; extrap=:none)

        @test isfinite(mitp(0.5)[1])

        @test_throws DomainError mitp(-0.1)
        @test_throws DomainError mitp(1.1)
    end

    @testset "extrap=:constant - boundary values" begin
        mitp = constant_interp(x, [y1, y2]; extrap=:constant)

        below = mitp(-0.5)
        @test below[1] ≈ y1[1]
        @test below[2] ≈ y2[1]

        above = mitp(1.5)
        @test above[1] ≈ y1[end]
        @test above[2] ≈ y2[end]
    end

    @testset "extrap=:wrap - wrapped coordinates" begin
        mitp = constant_interp(x, [y1, y2]; extrap=:wrap)

        result = mitp(1.35)
        expected1 = constant_interp(x, y1; extrap=:wrap)(1.35)
        expected2 = constant_interp(x, y2; extrap=:wrap)(1.35)

        @test result[1] ≈ expected1 atol=1e-14
        @test result[2] ≈ expected2 atol=1e-14
    end
end

# ============================================================================
# Phase 3: Vector Evaluation Tests
# ============================================================================

@testset "ConstantMultiInterpolant - Vector Evaluation (out-of-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = constant_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = [0.15, 0.35, 0.5, 0.75]

    @testset "Returns container of Vector{T}" begin
        result = mitp(xq)
        @test result isa AbstractVector
        @test length(result) == 3
        @test all(v -> v isa AbstractVector{Float64}, result)
        @test all(v -> length(v) == 4, result)
    end

    @testset "Results match individual interpolants" begin
        result = mitp(xq)

        itp1 = constant_interp(x, y1; extrap=:extension)
        itp2 = constant_interp(x, y2; extrap=:extension)
        itp3 = constant_interp(x, y3; extrap=:extension)

        @test result[1] ≈ itp1(xq) atol=1e-14
        @test result[2] ≈ itp2(xq) atol=1e-14
        @test result[3] ≈ itp3(xq) atol=1e-14
    end
end

@testset "ConstantMultiInterpolant - Container In-place (KILLER FEATURE)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = constant_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = collect(range(0.1, 0.9, 50))

    @testset "Fills all buffers correctly" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        result = mitp(outputs, xq)

        @test result === outputs

        itp1 = constant_interp(x, y1; extrap=:extension)
        itp2 = constant_interp(x, y2; extrap=:extension)
        itp3 = constant_interp(x, y3; extrap=:extension)

        @test out1 ≈ itp1(xq) atol=1e-14
        @test out2 ≈ itp2(xq) atol=1e-14
        @test out3 ≈ itp3(xq) atol=1e-14
    end

    @testset "Size assertion on container mismatch" begin
        outputs_wrong = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50)]
        @test_throws AssertionError mitp(outputs_wrong, xq)
    end

    @testset "ZERO ALLOCATION with pre-built anchors (critical)" begin
        FI = FastInterpolations
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        aq_vec = FI._anchor_query(x, xq, Val(:constant))

        mitp(outputs, aq_vec)
        mitp(outputs, aq_vec)

        allocs = @allocated mitp(outputs, aq_vec)
        @test allocs <= ALLOC_THRESHOLD
    end
end

# ============================================================================
# Phase 4: Safety & Integration Tests
# ============================================================================

# Note: ConstantInterpolant stores references (not copies) for efficiency.
# Use copy() if immutability is needed.

@testset "ConstantMultiInterpolant - Float32 Support" begin
    FI = FastInterpolations

    x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
    y1_32 = sin.(Float32(2π) .* x32)
    y2_32 = cos.(Float32(2π) .* x32)

    mitp32 = constant_interp(x32, [y1_32, y2_32])

    @testset "Float32 inputs produce Float32 outputs" begin
        result = mitp32(Float32(0.5))
        @test result isa AbstractVector{Float32}
    end
end

@testset "ConstantMultiInterpolant - Edge Cases" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Large number of series (10+)" begin
        ys = [sin.(k .* x) for k in 1:12]
        mitp = constant_interp(x, ys)
        @test length(mitp.itps) == 12

        result = mitp(0.5)
        @test length(result) == 12
    end

    @testset "Non-uniform grid works" begin
        x_nu = [0.0, 0.1, 0.15, 0.5, 0.8, 1.0]
        y1_nu = sin.(2π .* x_nu)
        y2_nu = cos.(2π .* x_nu)

        mitp = constant_interp(x_nu, [y1_nu, y2_nu])
        result = mitp(0.3)
        @test length(result) == 2
    end

    @testset "Query at grid points" begin
        mitp = constant_interp(x, [y1, y2])
        for xi in [0.0, 0.5, 1.0]
            result = mitp(xi)
            @test length(result) == 2
        end
    end

    @testset "Integer query auto-promoted" begin
        mitp = constant_interp(x, [y1, y2])
        result = mitp(0)
        @test result isa AbstractVector{Float64}
    end
end

# ============================================================================
# Phase 5: Type Promotion Tests
# ============================================================================

@testset "ConstantMultiInterpolant - Type Promotion" begin
    FI = FastInterpolations

    @testset "Integer x with Float y vectors (promotes to Float64)" begin
        x_int = collect(1:10)
        y1 = sin.(Float64.(x_int))
        y2 = cos.(Float64.(x_int))

        mitp = constant_interp(x_int, [y1, y2])
        @test mitp isa FI.ConstantMultiInterpolant{Float64}

        result = mitp(5.5)
        @test length(result) == 2
        @test all(isfinite, result)
    end

    @testset "Integer x with Integer y matrix (promotes to Float64)" begin
        x_int = collect(1:10)
        Y_int = [i * j for i in 1:10, j in 1:3]

        mitp = constant_interp(x_int, Y_int)
        @test mitp isa FI.ConstantMultiInterpolant{Float64}
        @test length(mitp.itps) == 3

        result = mitp(5.5)
        @test length(result) == 3
        @test all(isfinite, result)
    end

    @testset "In-place vector with type-promoted xq" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = constant_interp(x, [y1, y2])

        xq_f32 = Float32[0.1, 0.3, 0.5, 0.7, 0.9]
        out1 = Vector{Float64}(undef, 5)
        out2 = Vector{Float64}(undef, 5)
        outputs = [out1, out2]

        result = mitp(outputs, xq_f32)
        @test result === outputs
        @test all(isfinite, out1)
        @test all(isfinite, out2)

        xq_f64 = Float64.(xq_f32)
        ref = mitp(xq_f64)
        @test out1 ≈ ref[1] atol=1e-10
        @test out2 ≈ ref[2] atol=1e-10
    end
end

# ============================================================================
# Phase 6: Additional Size Assertion Tests
# ============================================================================

@testset "ConstantMultiInterpolant - Size Assertions" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = constant_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = collect(range(0.1, 0.9, 50))

    @testset "Size assertion on buffer mismatch" begin
        outputs = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50), Vector{Float64}(undef, 30)]
        @test_throws AssertionError mitp(outputs, xq)
    end

    @testset "Size assertion with pre-built anchors" begin
        aq_vec = FI._anchor_query(x, xq, Val(:constant))

        # Wrong number of output buffers
        outputs_wrong = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50)]
        @test_throws AssertionError mitp(outputs_wrong, aq_vec)

        # Wrong buffer size
        outputs_bad = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50), Vector{Float64}(undef, 30)]
        @test_throws AssertionError mitp(outputs_bad, aq_vec)
    end
end

# ============================================================================
# Phase 7: extrap=:extension Tests
# ============================================================================

@testset "ConstantMultiInterpolant - extrap=:extension" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    mitp = constant_interp(x, [y1, y2]; extrap=:extension)
    itp1 = constant_interp(x, y1; extrap=:extension)
    itp2 = constant_interp(x, y2; extrap=:extension)

    @testset "extrap=:extension outside domain" begin
        below = mitp(-0.1)
        @test below[1] ≈ itp1(-0.1) atol=1e-14
        @test below[2] ≈ itp2(-0.1) atol=1e-14

        above = mitp(1.1)
        @test above[1] ≈ itp1(1.1) atol=1e-14
        @test above[2] ≈ itp2(1.1) atol=1e-14
    end
end

# ============================================================================
# Phase 8: Zero-Allocation Vector API (Pooled Anchors)
# ============================================================================

@testset "ConstantMultiInterpolant - Zero-allocation vector API (pooled anchors)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = x .^ 2

    mitp = constant_interp(x, [y1, y2, y3])
    n_series = 3

    @testset "Zero allocation after warmup (same size)" begin
        xq = collect(range(0.05, 0.95, 50))
        outputs = [Vector{Float64}(undef, length(xq)) for _ in 1:n_series]

        # Warmup to populate pool
        mitp(outputs, xq)

        # Measure allocations on second call (same size)
        allocs = @allocated mitp(outputs, xq)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Zero allocation for smaller query vector (pool reuse)" begin
        # First call with larger vector
        xq_large = collect(range(0.05, 0.95, 100))
        outputs_large = [Vector{Float64}(undef, length(xq_large)) for _ in 1:n_series]
        mitp(outputs_large, xq_large)

        # Second call with smaller vector (pool should reuse)
        xq_small = collect(range(0.1, 0.9, 30))
        outputs_small = [Vector{Float64}(undef, length(xq_small)) for _ in 1:n_series]
        mitp(outputs_small, xq_small)

        # Third call with same small size (should be zero allocation)
        allocs = @allocated mitp(outputs_small, xq_small)
        @test allocs <= ALLOC_THRESHOLD
    end

    @testset "Bit-wise identical results" begin
        xq = collect(range(0.05, 0.95, 50))
        outputs = [Vector{Float64}(undef, length(xq)) for _ in 1:n_series]

        # Single-series reference values
        itp1 = constant_interp(x, y1)
        itp2 = constant_interp(x, y2)
        itp3 = constant_interp(x, y3)
        ref1 = itp1.(xq)
        ref2 = itp2.(xq)
        ref3 = itp3.(xq)

        # Multi-interpolant results
        mitp(outputs, xq)

        # Must be bit-wise identical
        @test outputs[1] == ref1
        @test outputs[2] == ref2
        @test outputs[3] == ref3
    end
end
