# ========================================
# ND In-Place Batch Evaluation Tests
# ========================================
#
# Tests for in-place batch evaluation: itp(output, queries; ...)
# Covers all 4 ND interpolant types with SoA and AoS batch modes.

@testitem "ND In-Place Batch Evaluation" begin

    # ========================================
    # Test Data Setup
    # ========================================
    x = collect(range(0.0, 2.0, 21))
    y = collect(range(0.0, 1.0, 11))

    data_2d = [sin(xi) * cos(yj) for xi in x, yj in y]

    # Query points
    qxs = [0.3, 0.7, 1.1, 1.5, 1.9]
    qys = [0.1, 0.3, 0.5, 0.7, 0.9]
    queries_soa = (qxs, qys)
    queries_aos = [(qxs[i], qys[i]) for i in eachindex(qxs)]
    n = length(qxs)

    # ========================================
    # Parameterized config per interpolant type
    # ========================================
    # Each entry: (name, constructor,
    #              deriv_cases = [(queries, deriv_arg), ...],
    #              zero_deriv  = deriv order where output must be all zeros, or nothing)

    configs = [
        (
            "CubicInterpolantND", cubic_interp,
            [(queries_soa, DerivOp(1, 0)), (queries_soa, DerivOp(1, 0)), (queries_aos, DerivOp(1, 0))],
            nothing,
        ),
        (
            "QuadraticInterpolantND", quadratic_interp,
            [(queries_soa, DerivOp(1, 0)), (queries_soa, DerivOp(1, 0)), (queries_aos, DerivOp(1, 0))],
            nothing,
        ),
        (
            "LinearInterpolantND", linear_interp,
            [(queries_soa, DerivOp(1, 0)), (queries_aos, DerivOp(1, 0))],
            DerivOp(2, 0),
        ),
        (
            "ConstantInterpolantND", constant_interp,
            [],
            DerivOp(1, 0),
        ),
    ]

    @testset "$name" for (name, interp_fn, deriv_cases, zero_deriv) in configs
        itp = interp_fn((x, y), data_2d)

        @testset "SoA correctness" begin
            expected = itp(queries_soa)
            output = zeros(n)
            result = itp(output, queries_soa)
            @test result === output  # returns same object
            @test output ≈ expected
        end

        @testset "AoS correctness" begin
            expected = itp(queries_aos)
            output = zeros(n)
            result = itp(output, queries_aos)
            @test result === output
            @test output ≈ expected
        end

        if !isempty(deriv_cases)
            @testset "deriv keyword" begin
                for (qs, d) in deriv_cases
                    expected = itp(qs; deriv = d)
                    output = zeros(n)
                    itp(output, qs; deriv = d)
                    @test output ≈ expected
                end
            end
        end

        if zero_deriv !== nothing
            @testset "deriv=$zero_deriv early-return (zeros)" begin
                output = ones(n)
                itp(output, queries_soa; deriv = zero_deriv)
                @test all(iszero, output)

                output = ones(n)
                itp(output, queries_aos; deriv = zero_deriv)
                @test all(iszero, output)
            end
        end

        @testset "hint keyword" begin
            hints = (Ref(1), Ref(1))
            expected = itp(queries_soa)
            output = zeros(n)
            itp(output, queries_soa; hint = hints)
            @test output ≈ expected
        end

        @testset "DimensionMismatch" begin
            @test_throws DimensionMismatch itp(zeros(3), queries_soa)
            @test_throws DimensionMismatch itp(zeros(3), queries_aos)
        end
    end

    # ========================================
    # 3D Test (verifies N>2 works)
    # ========================================
    @testset "3D in-place" begin
        z = collect(range(0.0, 1.5, 8))
        data_3d = [sin(xi) * cos(yj) * exp(-zk) for xi in x, yj in y, zk in z]
        itp = cubic_interp((x, y, z), data_3d)

        qzs = [0.2, 0.5, 0.8, 1.0, 1.3]
        soa_3d = (qxs, qys, qzs)
        aos_3d = [(qxs[i], qys[i], qzs[i]) for i in eachindex(qxs)]

        expected_soa = itp(soa_3d)
        output = zeros(n)
        itp(output, soa_3d)
        @test output ≈ expected_soa

        expected_aos = itp(aos_3d)
        output = zeros(n)
        itp(output, aos_3d)
        @test output ≈ expected_aos
    end
end

# Regression: a Vector{Tuple} is an AoS BATCH, not a single N-vector point. Relaxing the
# vector-point method to unbounded `AbstractVector` silently captured AoS batches (a
# 5-point batch mis-read as one 5-element point). Fix = `{<:Number}` scalar-vs-container
# gate; affects plain Real grids too, not unit-specific.
@testitem "ND batch dispatch: AoS Vector{Tuple} routes to batch, not vector-point" begin
    using InteractiveUtils: @which

    x = collect(range(0.0, 2.0, 11))
    y = collect(range(0.0, 1.0, 6))
    data = [sin(xi) * cos(yj) for xi in x, yj in y]
    itp = linear_interp((x, y), data)

    qxs = [0.3, 0.7, 1.1, 1.5, 1.9]
    qys = [0.1, 0.3, 0.5, 0.7, 0.9]
    aos = [(qxs[i], qys[i]) for i in eachindex(qxs)]   # Vector{Tuple{Float64,Float64}}
    soa = (qxs, qys)

    # Functional: AoS batch yields n results (no DimensionMismatch) and matches SoA.
    @test length(itp(aos)) == length(qxs)
    @test itp(aos) ≈ itp(soa)
    @test itp(aos)[1] ≈ itp((qxs[1], qys[1]))

    # Dispatch: AoS and SoA share the batch method; a genuine 2-element numeric
    # vector is a single point on a DIFFERENT method.
    @test @which(itp(aos)) === @which(itp(soa))
    @test @which(itp(aos)) !== @which(itp([0.5, 0.5]))
end
