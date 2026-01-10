# Tests for LinearMultiInterpolant - Multi-Y linear interpolation
#
# Multi-Y interpolation: multiple y-data series sharing the same x-grid.
# Uses composition approach: wraps existing LinearInterpolant objects.

# ============================================================================
# Phase 1: Type Definition & Constructor Tests
# ============================================================================

@testset "LinearMultiInterpolant - Type Structure" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    @testset "Construction from Vector of y-series" begin
        mitp = linear_interp(x, [y1, y2, y3])
        @test mitp isa FI.LinearMultiInterpolant

        # Access internal interpolants
        @test hasfield(typeof(mitp), :itps)
        @test length(mitp.itps) == 3
        @test all(itp -> itp isa LinearInterpolant, mitp.itps)
    end

    @testset "Construction from Matrix (columns as series)" begin
        Y = hcat(y1, y2, y3)  # 101×3 matrix
        mitp = linear_interp(x, Y)
        @test mitp isa FI.LinearMultiInterpolant
        @test length(mitp.itps) == 3
    end

    @testset "Single series works" begin
        mitp = linear_interp(x, [y1])
        @test mitp isa FI.LinearMultiInterpolant
        @test length(mitp.itps) == 1
    end

    @testset "Type inference - Float64" begin
        mitp = linear_interp(x, [y1, y2])
        @test eltype(mitp.itps) <: LinearInterpolant{Float64}
    end

    @testset "Type inference - Float32" begin
        x32 = Float32.(x)
        y1_32 = Float32.(y1)
        y2_32 = Float32.(y2)
        mitp = linear_interp(x32, [y1_32, y2_32])
        @test eltype(mitp.itps) <: LinearInterpolant{Float32}
    end

    @testset "Subtypes AbstractMultiInterpolant" begin
        mitp = linear_interp(x, [y1, y2])
        @test mitp isa AbstractMultiInterpolant{Float64}
    end
end

@testset "LinearMultiInterpolant - Constructor Validation" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Empty ys throws error" begin
        @test_throws Union{ArgumentError, AssertionError} linear_interp(x, Vector{Float64}[])
    end

    @testset "Mismatched lengths throws error" begin
        y_short = y1[1:50]
        @test_throws Union{DimensionMismatch, ArgumentError, AssertionError} linear_interp(x, [y1, y_short])
    end

    @testset "Extrap propagation" begin
        for extrap_mode in (:none, :constant, :extension, :wrap)
            mitp = linear_interp(x, [y1, y2]; extrap=extrap_mode)
            @test mitp.itps[1].mode === Val(extrap_mode)
            @test mitp.itps[2].mode === Val(extrap_mode)
        end
    end
end

# ============================================================================
# Phase 2: Scalar Evaluation Tests
# ============================================================================

@testset "LinearMultiInterpolant - Scalar Evaluation (out-of-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = linear_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "Returns Vector with correct values" begin
        result = mitp(0.35)
        @test result isa AbstractVector
        @test length(result) == 3

        # Individual results match standalone interpolants
        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)
        itp3 = linear_interp(x, y3; extrap=:extension)

        @test result[1] ≈ itp1(0.35) atol=1e-14
        @test result[2] ≈ itp2(0.35) atol=1e-14
        @test result[3] ≈ itp3(0.35) atol=1e-14
    end

    @testset "Derivatives via deriv keyword" begin
        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)
        itp3 = linear_interp(x, y3; extrap=:extension)

        # First derivative (linear → piecewise constant derivative)
        d1 = mitp(0.5; deriv=1)
        @test d1[1] ≈ itp1(0.5; deriv=1) atol=1e-14
        @test d1[2] ≈ itp2(0.5; deriv=1) atol=1e-14
        @test d1[3] ≈ itp3(0.5; deriv=1) atol=1e-14
    end

    @testset "Multiple query points" begin
        for xq in [0.0, 0.15, 0.5, 0.85, 1.0]
            result = mitp(xq)
            @test length(result) == 3
        end
    end
end

@testset "LinearMultiInterpolant - Scalar Evaluation (in-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = linear_interp(x, [y1, y2, y3]; extrap=:extension)

    @testset "Fills output correctly" begin
        output = Vector{Float64}(undef, 3)
        result = mitp(output, 0.35)

        @test result === output  # Returns same reference

        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)
        itp3 = linear_interp(x, y3; extrap=:extension)

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

        itp1 = linear_interp(x, y1; extrap=:extension)
        @test output[1] ≈ itp1(0.5; deriv=1) atol=1e-14
    end
end

@testset "LinearMultiInterpolant - Scalar Extrap Modes" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "extrap=:none - throws DomainError" begin
        mitp = linear_interp(x, [y1, y2]; extrap=:none)

        # Inside domain works
        @test isfinite(mitp(0.5)[1])

        # Outside domain throws
        @test_throws DomainError mitp(-0.1)
        @test_throws DomainError mitp(1.1)
    end

    @testset "extrap=:constant - boundary values" begin
        mitp = linear_interp(x, [y1, y2]; extrap=:constant)

        below = mitp(-0.5)
        @test below[1] ≈ y1[1]
        @test below[2] ≈ y2[1]

        above = mitp(1.5)
        @test above[1] ≈ y1[end]
        @test above[2] ≈ y2[end]
    end

    @testset "extrap=:extension - extrapolates linearly" begin
        mitp = linear_interp(x, [y1, y2]; extrap=:extension)
        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)

        below = mitp(-0.1)
        @test below[1] ≈ itp1(-0.1) atol=1e-14
        @test below[2] ≈ itp2(-0.1) atol=1e-14

        above = mitp(1.1)
        @test above[1] ≈ itp1(1.1) atol=1e-14
        @test above[2] ≈ itp2(1.1) atol=1e-14
    end

    @testset "extrap=:wrap - wrapped coordinates" begin
        mitp = linear_interp(x, [y1, y2]; extrap=:wrap)

        result = mitp(1.35)  # wraps
        expected1 = linear_interp(x, y1; extrap=:wrap)(1.35)
        expected2 = linear_interp(x, y2; extrap=:wrap)(1.35)

        @test result[1] ≈ expected1 atol=1e-14
        @test result[2] ≈ expected2 atol=1e-14
    end
end

# ============================================================================
# Phase 3: Vector Evaluation Tests
# ============================================================================

@testset "LinearMultiInterpolant - Vector Evaluation (out-of-place)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = linear_interp(x, [y1, y2, y3]; extrap=:extension)
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

        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)
        itp3 = linear_interp(x, y3; extrap=:extension)

        @test result[1] ≈ itp1(xq) atol=1e-14
        @test result[2] ≈ itp2(xq) atol=1e-14
        @test result[3] ≈ itp3(xq) atol=1e-14
    end

    @testset "Derivatives with vector queries" begin
        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)

        mitp2 = linear_interp(x, [y1, y2]; extrap=:extension)

        d1 = mitp2(xq; deriv=1)
        @test d1[1] ≈ itp1(xq; deriv=1) atol=1e-14
        @test d1[2] ≈ itp2(xq; deriv=1) atol=1e-14
    end
end

@testset "LinearMultiInterpolant - Container In-place (KILLER FEATURE)" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)
    y3 = exp.(-3 .* x)

    mitp = linear_interp(x, [y1, y2, y3]; extrap=:extension)
    xq = collect(range(0.1, 0.9, 50))

    @testset "Fills all buffers correctly" begin
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        result = mitp(outputs, xq)

        @test result === outputs

        itp1 = linear_interp(x, y1; extrap=:extension)
        itp2 = linear_interp(x, y2; extrap=:extension)
        itp3 = linear_interp(x, y3; extrap=:extension)

        @test out1 ≈ itp1(xq) atol=1e-14
        @test out2 ≈ itp2(xq) atol=1e-14
        @test out3 ≈ itp3(xq) atol=1e-14
    end

    @testset "Size assertion on container mismatch" begin
        outputs_wrong = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50)]
        @test_throws AssertionError mitp(outputs_wrong, xq)
    end

    @testset "Size assertion on buffer mismatch" begin
        outputs = [Vector{Float64}(undef, 50), Vector{Float64}(undef, 50), Vector{Float64}(undef, 30)]
        @test_throws AssertionError mitp(outputs, xq)
    end

    @testset "ZERO ALLOCATION with pre-built anchors (critical)" begin
        FI = FastInterpolations
        out1 = Vector{Float64}(undef, 50)
        out2 = Vector{Float64}(undef, 50)
        out3 = Vector{Float64}(undef, 50)
        outputs = [out1, out2, out3]

        # Pre-build anchors
        aq_vec = FI._anchor_query(x, xq, Val(:linear))

        # Warmup
        mitp(outputs, aq_vec)
        mitp(outputs, aq_vec)

        allocs = @allocated mitp(outputs, aq_vec)
        @test allocs == 0
    end
end

# ============================================================================
# Phase 4: Safety & Integration Tests
# ============================================================================

# Note: LinearInterpolant stores references (not copies) for efficiency.
# Use copy() if immutability is needed. This differs from CubicInterpolant.

@testset "LinearMultiInterpolant - Float32 Support" begin
    FI = FastInterpolations

    x32 = Float32.(collect(range(Float32(0), Float32(1), 101)))
    y1_32 = sin.(Float32(2π) .* x32)
    y2_32 = cos.(Float32(2π) .* x32)

    mitp32 = linear_interp(x32, [y1_32, y2_32])

    @testset "Float32 inputs produce Float32 outputs" begin
        result = mitp32(Float32(0.5))
        @test result isa AbstractVector{Float32}
    end

    @testset "Results match Float64 reference (relaxed tolerance)" begin
        x64 = Float64.(x32)
        y1_64 = Float64.(y1_32)
        y2_64 = Float64.(y2_32)
        mitp64 = linear_interp(x64, [y1_64, y2_64])

        result32 = mitp32(Float32(0.35))
        result64 = mitp64(0.35)

        @test result32[1] ≈ Float32(result64[1]) atol=1f-5
        @test result32[2] ≈ Float32(result64[2]) atol=1f-5
    end
end

@testset "LinearMultiInterpolant - Edge Cases" begin
    FI = FastInterpolations

    x = collect(range(0.0, 1.0, 101))
    y1 = sin.(2π .* x)
    y2 = cos.(2π .* x)

    @testset "Large number of series (10+)" begin
        ys = [sin.(k .* x) for k in 1:12]
        mitp = linear_interp(x, ys)
        @test length(mitp.itps) == 12

        result = mitp(0.5)
        @test length(result) == 12
    end

    @testset "Non-uniform grid works" begin
        x_nu = [0.0, 0.1, 0.15, 0.5, 0.8, 1.0]
        y1_nu = sin.(2π .* x_nu)
        y2_nu = cos.(2π .* x_nu)

        mitp = linear_interp(x_nu, [y1_nu, y2_nu])
        result = mitp(0.3)
        @test length(result) == 2
    end

    @testset "Query at grid points" begin
        mitp = linear_interp(x, [y1, y2])
        for xi in [0.0, 0.5, 1.0]
            result = mitp(xi)
            @test length(result) == 2
        end
    end

    @testset "Integer query auto-promoted" begin
        mitp = linear_interp(x, [y1, y2])
        result = mitp(0)
        @test result isa AbstractVector{Float64}
    end
end

# ============================================================================
# Phase 5: Type Promotion Tests
# ============================================================================

@testset "LinearMultiInterpolant - Type Promotion" begin
    FI = FastInterpolations

    @testset "Integer x with Float y vectors (promotes to Float64)" begin
        x_int = collect(1:10)
        y1 = sin.(Float64.(x_int))
        y2 = cos.(Float64.(x_int))

        mitp = linear_interp(x_int, [y1, y2])
        @test mitp isa FI.LinearMultiInterpolant{Float64}

        result = mitp(5.5)
        @test length(result) == 2
        @test all(isfinite, result)
    end

    @testset "Integer x with Integer y matrix (promotes to Float64)" begin
        x_int = collect(1:10)
        Y_int = [i * j for i in 1:10, j in 1:3]

        mitp = linear_interp(x_int, Y_int)
        @test mitp isa FI.LinearMultiInterpolant{Float64}
        @test length(mitp.itps) == 3

        result = mitp(5.5)
        @test length(result) == 3
        @test all(isfinite, result)
    end

    @testset "In-place vector with type-promoted xq" begin
        x = collect(range(0.0, 1.0, 101))
        y1 = sin.(2π .* x)
        y2 = cos.(2π .* x)

        mitp = linear_interp(x, [y1, y2])

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
