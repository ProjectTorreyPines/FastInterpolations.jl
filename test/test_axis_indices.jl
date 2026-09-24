# Unit tests for the fixed-size per-axis index representations.

@testitem "Axis indices" begin
    using FastInterpolations: _AbstractIndices, _ContiguousIndices, _ExplicitIndices

    @testset "Construction and representation" begin
        adjacent = _ContiguousIndices{2}(5)
        @test adjacent isa _AbstractIndices{2}
        @test adjacent.first == 5

        support4 = _ContiguousIndices{4}(3)
        @test support4 isa _AbstractIndices{4}

        support6 = _ContiguousIndices{6}(7)
        @test support6 isa _AbstractIndices{6}

        seam = _ExplicitIndices(10, 1)
        @test seam isa _AbstractIndices{2}
        @test seam.indices === (10, 1)

        wrapped = _ExplicitIndices((4, 1, 2, 3))
        @test wrapped isa _AbstractIndices{4}
        @test wrapped.indices === (4, 1, 2, 3)

        repeated = _ExplicitIndices(2, 1, 2, 1, 2, 1)
        @test repeated isa _AbstractIndices{6}
        @test repeated.indices === (2, 1, 2, 1, 2, 1)
    end

    @testset "Accessors" begin
        contiguous = _ContiguousIndices{4}(7)
        @test contiguous[1] == 7
        @test contiguous[4] == 10
        @test contiguous[Val(1)] == 7
        @test contiguous[Val(4)] == 10
        @test length(contiguous) == 4
        @test firstindex(contiguous) == 1
        @test lastindex(contiguous) == 4
        @test first(contiguous) == 7
        @test last(contiguous) == 10

        explicit = _ExplicitIndices(7, 3, 9, 2)
        @test explicit[1] == 7
        @test explicit[4] == 2
        @test explicit[Val(1)] == 7
        @test explicit[Val(4)] == 2
        @test length(explicit) == 4
        @test first(explicit) == 7
        @test last(explicit) == 2
    end

    @testset "Iteration" begin
        @test collect(_ContiguousIndices{4}(2)) == [2, 3, 4, 5]
        @test collect(_ExplicitIndices(2, 5, 1, 4)) == [2, 5, 1, 4]
    end

    @testset "Semantic equality and hash" begin
        contiguous = _ContiguousIndices{4}(3)
        explicit_equal = _ExplicitIndices(3, 4, 5, 6)
        explicit_other = _ExplicitIndices(3, 4, 6, 5)

        @test contiguous == explicit_equal
        @test explicit_equal == contiguous
        @test contiguous != explicit_other
        @test hash(contiguous) == hash(explicit_equal)
    end

    @testset "Isbits layout" begin
        c2 = _ContiguousIndices{2}(1)
        c6 = _ContiguousIndices{6}(1)
        e2 = _ExplicitIndices(1, 2)
        e6 = _ExplicitIndices(1, 2, 3, 4, 5, 6)

        @test isbitstype(typeof(c2))
        @test isbitstype(typeof(c6))
        @test isbitstype(typeof(e2))
        @test isbitstype(typeof(e6))
        @test sizeof(c2) == sizeof(Int)
        @test sizeof(c6) == sizeof(Int)
        @test sizeof(e2) == 2 * sizeof(Int)
        @test sizeof(e6) == 6 * sizeof(Int)
    end

    @testset "Inference and allocation" begin
        @test @inferred(_ContiguousIndices{2}(5)) isa _ContiguousIndices{2}
        @test @inferred(_ExplicitIndices(5, 6)) isa _ExplicitIndices{2}
        @test @inferred(_ExplicitIndices((5, 6, 7, 8))) isa _ExplicitIndices{4}

        contiguous = _ContiguousIndices{4}(5)
        explicit = _ExplicitIndices(5, 6, 7, 8)
        @test @inferred(contiguous[Val(3)]) === 7
        @test @inferred(explicit[Val(3)]) === 7

        function sum_contiguous(first_idx)
            indices = _ContiguousIndices{6}(first_idx)
            return indices[Val(1)] + indices[Val(2)] + indices[Val(3)] +
                indices[Val(4)] + indices[Val(5)] + indices[Val(6)]
        end

        function sum_explicit(first_idx)
            indices = _ExplicitIndices(
                first_idx,
                first_idx + 1,
                first_idx + 2,
                first_idx + 3,
                first_idx + 4,
                first_idx + 5,
            )
            return indices[Val(1)] + indices[Val(2)] + indices[Val(3)] +
                indices[Val(4)] + indices[Val(5)] + indices[Val(6)]
        end

        sum_contiguous(3)
        sum_explicit(3)
        @test @allocated(sum_contiguous(3)) == 0
        @test @allocated(sum_explicit(3)) == 0
    end
end
