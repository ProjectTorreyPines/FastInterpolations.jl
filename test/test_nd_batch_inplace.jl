# ========================================
# ND In-Place Batch Evaluation Tests
# ========================================
#
# Tests for in-place batch evaluation: itp(output, queries; ...)
# Covers all 4 ND interpolant types with SoA and AoS batch modes.

using Test
using FastInterpolations

@testset "ND In-Place Batch Evaluation" begin

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
    # CubicInterpolantND
    # ========================================
    @testset "CubicInterpolantND" begin
        itp = cubic_interp((x, y), data_2d)

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

        @testset "deriv keyword" begin
            # Int deriv
            expected = itp(queries_soa; deriv=1)
            output = zeros(n)
            itp(output, queries_soa; deriv=1)
            @test output ≈ expected

            # Tuple deriv
            expected = itp(queries_soa; deriv=(1, 0))
            output = zeros(n)
            itp(output, queries_soa; deriv=(1, 0))
            @test output ≈ expected

            # Val deriv
            expected = itp(queries_soa; deriv=Val(1))
            output = zeros(n)
            itp(output, queries_soa; deriv=Val(1))
            @test output ≈ expected

            # AoS with deriv
            expected = itp(queries_aos; deriv=1)
            output = zeros(n)
            itp(output, queries_aos; deriv=1)
            @test output ≈ expected
        end

        @testset "hint keyword" begin
            hints = (Ref(1), Ref(1))
            expected = itp(queries_soa)
            output = zeros(n)
            itp(output, queries_soa; hint=hints)
            @test output ≈ expected
            @test hints[1][] >= 1
        end

        @testset "DimensionMismatch" begin
            @test_throws DimensionMismatch itp(zeros(3), queries_soa)
            @test_throws DimensionMismatch itp(zeros(3), queries_aos)
            @test_throws DimensionMismatch itp(zeros(n), (qxs, qys[1:3]))
        end
    end

    # ========================================
    # QuadraticInterpolantND
    # ========================================
    @testset "QuadraticInterpolantND" begin
        itp = quadratic_interp((x, y), data_2d)

        @testset "SoA correctness" begin
            expected = itp(queries_soa)
            output = zeros(n)
            result = itp(output, queries_soa)
            @test result === output
            @test output ≈ expected
        end

        @testset "AoS correctness" begin
            expected = itp(queries_aos)
            output = zeros(n)
            result = itp(output, queries_aos)
            @test result === output
            @test output ≈ expected
        end

        @testset "deriv keyword" begin
            expected = itp(queries_soa; deriv=1)
            output = zeros(n)
            itp(output, queries_soa; deriv=1)
            @test output ≈ expected

            expected = itp(queries_soa; deriv=(1, 0))
            output = zeros(n)
            itp(output, queries_soa; deriv=(1, 0))
            @test output ≈ expected

            expected = itp(queries_aos; deriv=Val(1))
            output = zeros(n)
            itp(output, queries_aos; deriv=Val(1))
            @test output ≈ expected
        end

        @testset "hint keyword" begin
            hints = (Ref(1), Ref(1))
            expected = itp(queries_soa)
            output = zeros(n)
            itp(output, queries_soa; hint=hints)
            @test output ≈ expected
        end

        @testset "DimensionMismatch" begin
            @test_throws DimensionMismatch itp(zeros(3), queries_soa)
            @test_throws DimensionMismatch itp(zeros(3), queries_aos)
        end
    end

    # ========================================
    # LinearInterpolantND
    # ========================================
    @testset "LinearInterpolantND" begin
        itp = linear_interp((x, y), data_2d)

        @testset "SoA correctness" begin
            expected = itp(queries_soa)
            output = zeros(n)
            result = itp(output, queries_soa)
            @test result === output
            @test output ≈ expected
        end

        @testset "AoS correctness" begin
            expected = itp(queries_aos)
            output = zeros(n)
            result = itp(output, queries_aos)
            @test result === output
            @test output ≈ expected
        end

        @testset "deriv keyword" begin
            expected = itp(queries_soa; deriv=1)
            output = zeros(n)
            itp(output, queries_soa; deriv=1)
            @test output ≈ expected

            expected = itp(queries_aos; deriv=(1, 0))
            output = zeros(n)
            itp(output, queries_aos; deriv=(1, 0))
            @test output ≈ expected
        end

        @testset "2nd derivative early-return (zeros)" begin
            output = ones(n)
            itp(output, queries_soa; deriv=2)
            @test all(iszero, output)

            output = ones(n)
            itp(output, queries_aos; deriv=2)
            @test all(iszero, output)
        end

        @testset "hint keyword" begin
            hints = (Ref(1), Ref(1))
            expected = itp(queries_soa)
            output = zeros(n)
            itp(output, queries_soa; hint=hints)
            @test output ≈ expected
        end

        @testset "DimensionMismatch" begin
            @test_throws DimensionMismatch itp(zeros(3), queries_soa)
            @test_throws DimensionMismatch itp(zeros(3), queries_aos)
        end
    end

    # ========================================
    # ConstantInterpolantND
    # ========================================
    @testset "ConstantInterpolantND" begin
        itp = constant_interp((x, y), data_2d)

        @testset "SoA correctness" begin
            expected = itp(queries_soa)
            output = zeros(n)
            result = itp(output, queries_soa)
            @test result === output
            @test output ≈ expected
        end

        @testset "AoS correctness" begin
            expected = itp(queries_aos)
            output = zeros(n)
            result = itp(output, queries_aos)
            @test result === output
            @test output ≈ expected
        end

        @testset "deriv early-return (zeros)" begin
            # Any derivative on constant returns zero
            output = ones(n)
            itp(output, queries_soa; deriv=1)
            @test all(iszero, output)

            output = ones(n)
            itp(output, queries_aos; deriv=1)
            @test all(iszero, output)
        end

        @testset "hint keyword" begin
            hints = (Ref(1), Ref(1))
            expected = itp(queries_soa)
            output = zeros(n)
            itp(output, queries_soa; hint=hints)
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
