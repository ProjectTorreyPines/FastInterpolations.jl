@testitem "_CachedVector Construction" begin
    using FastInterpolations: _CachedVector

    @testset "Float64 round-trip" begin
        x = [0.0, 0.3, 0.7, 1.0]
        c = _CachedVector(x)

        @test c isa _CachedVector{Float64, Float64}
        @test eltype(c) == Float64
        @test length(c) == 4
        @test size(c) == (4,)
        @test firstindex(c) == 1
        @test lastindex(c) == 4

        # getindex forwards to inner
        for i in 1:4
            @test c[i] == x[i]
        end

        # cached spacing fields
        @test length(c.h) == 3
        @test length(c.inv_h) == 3
        @test c.h ≈ [0.3, 0.4, 0.3]
        @test c.inv_h ≈ inv.([0.3, 0.4, 0.3])
    end

    @testset "Float32" begin
        x = Float32[0.0, 0.5, 1.5, 2.0]
        c = _CachedVector(x)

        @test c isa _CachedVector{Float32, Float32}
        @test eltype(c) == Float32
        @test c.h ≈ Float32[0.5, 1.0, 0.5]
        @test eltype(c.h) == Float32
        @test eltype(c.inv_h) == Float32
    end

    @testset "Int grid → Tinv = Float64" begin
        x = [0, 1, 3, 6]
        c = _CachedVector(x)

        @test c isa _CachedVector{Int, Float64}
        @test eltype(c) == Int
        @test c.h == [1, 2, 3]
        @test eltype(c.h) == Int
        @test c.inv_h ≈ [1.0, 0.5, 1 / 3]
        @test eltype(c.inv_h) == Float64
    end

    @testset "h[i] = x[i+1] - x[i] correctness" begin
        x = [0.0, 0.1, 0.25, 0.5, 0.9, 1.0]
        c = _CachedVector(x)
        for i in 1:(length(x) - 1)
            @test c.h[i] == x[i + 1] - x[i]
            @test c.inv_h[i] == inv(c.h[i])
        end
    end

    @testset "Idempotent: _CachedVector(::_CachedVector) === input" begin
        x = [0.0, 0.3, 0.7, 1.0]
        c1 = _CachedVector(x)
        c2 = _CachedVector(c1)
        @test c2 === c1
    end

    @testset "Non-Vector input is converted to concrete Vector{T}" begin
        # SubArray
        full = collect(0.0:0.1:1.0)
        view_x = @view full[2:5]
        c = _CachedVector(view_x)

        @test c isa _CachedVector{Float64, Float64}
        @test c.inner isa Vector{Float64}  # not SubArray
        @test c.inner == collect(view_x)
    end

    @testset "ArgumentError for n < 2" begin
        @test_throws ArgumentError _CachedVector([1.0])
        @test_throws ArgumentError _CachedVector(Float64[])
    end

    @testset "AbstractVector interface compatibility" begin
        # Should accept _CachedVector wherever AbstractVector{T} is expected
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        c = _CachedVector(x)

        @test c isa AbstractVector{Float64}
        @test sum(c) == sum(x)
        @test maximum(c) == 2.0
        @test minimum(c) == 0.0

        # iteration
        s = 0.0
        for v in c
            s += v
        end
        @test s == sum(x)
    end
end

@testitem "_CachedVector with duck-typed grids (ext)" begin
    # Note: full Dual/Measurement coverage lives in ext/test_*_dual_grid.jl etc.
    # This testitem covers basic round-trip with Rational (not in ext).
    using FastInterpolations: _CachedVector

    @testset "Rational grid" begin
        x = [0 // 1, 1 // 4, 1 // 2, 3 // 4, 1 // 1]
        c = _CachedVector(x)

        @test c isa _CachedVector{Rational{Int}}
        @test eltype(c) == Rational{Int}
        @test c.h == [1 // 4, 1 // 4, 1 // 4, 1 // 4]
        @test c.inv_h == [4 // 1, 4 // 1, 4 // 1, 4 // 1]
    end
end

@testitem "_CachedVector backward-compat _create_spacing shim" begin
    using FastInterpolations: _CachedVector, _create_spacing, VectorSpacing

    @testset "Float64 shim equivalence" begin
        x = [0.0, 0.3, 0.7, 1.0]
        c = _CachedVector(x)
        s_via_shim = _create_spacing(c)
        s_direct = _create_spacing(x)

        # Shim returns VectorSpacing extracted from cached fields
        @test s_via_shim isa VectorSpacing{Float64, Float64}
        @test s_via_shim.h == s_direct.h
        @test s_via_shim.inv_h == s_direct.inv_h
    end

    @testset "Int grid shim → VectorSpacing{Int, Float64}" begin
        x = [0, 1, 3, 6]
        c = _CachedVector(x)
        s = _create_spacing(c)

        @test s isa VectorSpacing{Int, Float64}
        @test s.h == [1, 2, 3]
        @test s.inv_h ≈ [1.0, 0.5, 1 / 3]
    end
end
