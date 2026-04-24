# Unit tests for `_IdxStencil{K}` — the wrap-aware per-axis corner-index type.
# See src/core/idx_stencil.jl for design notes.

using Test
using FastInterpolations: _IdxStencil, _IdxPair, _pair

@testset "_IdxStencil" begin

    @testset "Construction" begin
        # NTuple constructor
        s2 = _IdxStencil((5, 6))
        @test s2 isa _IdxStencil{2}
        @test s2.indices === (5, 6)

        # Explicit K
        s2b = _IdxStencil{2}((5, 6))
        @test s2b isa _IdxStencil{2}
        @test s2b.indices === (5, 6)

        # K=4 (future ND Hermite)
        s4 = _IdxStencil((3, 4, 5, 6))
        @test s4 isa _IdxStencil{4}
        @test s4.indices === (3, 4, 5, 6)

        # _pair convenience
        sp = _pair(10, 1)               # periodic-exclusive seam shape
        @test sp isa _IdxStencil{2}
        @test sp.indices === (10, 1)

        # Alias
        @test _IdxPair === _IdxStencil{2}
    end

    @testset "Accessors" begin
        s = _IdxStencil((7, 8, 9))
        @test s[1] == 7
        @test s[2] == 8
        @test s[3] == 9
        @test length(s) == 3
        @test firstindex(s) == 1
        @test lastindex(s) == 3
    end

    @testset "Iteration" begin
        s = _IdxStencil((2, 3, 4, 5))
        collected = Int[]
        for v in s
            push!(collected, v)
        end
        @test collected == [2, 3, 4, 5]

        # collect() on iteration — may allocate, but values correct
        @test collect(s) == [2, 3, 4, 5]
    end

    @testset "Equality & hash" begin
        a = _IdxStencil((1, 2))
        b = _IdxStencil((1, 2))
        c = _IdxStencil((2, 1))
        @test a == b
        @test a != c
        @test hash(a) == hash(b)
        @test hash(a) != hash(c)
    end

    @testset "Memory layout — identical to raw NTuple" begin
        s2 = _IdxStencil((5, 6))
        @test sizeof(s2) == sizeof((5, 6))                  # 16 bytes on 64-bit

        s4 = _IdxStencil((1, 2, 3, 4))
        @test sizeof(s4) == sizeof((1, 2, 3, 4))            # 32 bytes
    end

    @testset "Zero-allocation on hot path" begin
        # Function barrier for @allocated
        function bench_construct()
            s = _IdxStencil((5, 6))
            return s[1] + s[2]
        end

        function bench_iterate()
            s = _IdxStencil((1, 2, 3, 4))
            acc = 0
            for v in s
                acc += v
            end
            return acc
        end

        # Warmup
        bench_construct(); bench_construct()
        bench_iterate(); bench_iterate()

        @test (@allocated bench_construct()) == 0
        @test (@allocated bench_iterate()) == 0
    end

    @testset "Type stability" begin
        # Construction infers to concrete _IdxStencil{K, NTuple{K, Int}}
        @test @inferred(_IdxStencil((1, 2))) isa _IdxStencil{2}
        @test @inferred(_IdxStencil((1, 2, 3, 4))) isa _IdxStencil{4}
        @test @inferred(_pair(1, 2)) isa _IdxStencil{2}

        # Accessors preserve Int concrete
        s = _IdxStencil((5, 6))
        @test @inferred(s[1]) isa Int
        @test @inferred(length(s)) == 2
    end
end
