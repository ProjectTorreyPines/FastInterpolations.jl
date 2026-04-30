@testitem "Unified _get_h / _get_inv_h dispatch" begin
    using FastInterpolations: _get_h, _get_inv_h, _CachedRange, _CachedVector, _to_float

    # Reference grid (mathematically identical for all 4 representations)
    x_vec_uniform = collect(0.0:0.1:1.0)         # Vector{Float64}, uniform
    x_vec_nonuniform = [0.0, 0.3, 0.7, 1.0]      # Vector{Float64}, non-uniform
    x_range = 0.0:0.1:1.0                        # StepRangeLen → AbstractRange
    x_cached_range = _to_float(x_range, Float64) # _CachedRange{Float64}
    x_cached_vec = _CachedVector(x_vec_nonuniform) # _CachedVector{Float64, Float64}

    @testset "Vector — on-the-fly diff" begin
        # h[i] = x[i+1] - x[i]
        for i in 1:(length(x_vec_nonuniform) - 1)
            expected_h = x_vec_nonuniform[i + 1] - x_vec_nonuniform[i]
            @test _get_h(x_vec_nonuniform, i) == expected_h
            @test _get_inv_h(x_vec_nonuniform, i) ≈ inv(expected_h)
        end
    end

    @testset "AbstractRange — uniform via step()" begin
        for i in 1:(length(x_range) - 1)
            @test _get_h(x_range, i) ≈ 0.1
            @test _get_inv_h(x_range, i) ≈ 10.0
        end
        # Range index argument is ignored (uniform)
        @test _get_h(x_range, 1) == _get_h(x_range, 5)
    end

    @testset "_CachedRange — cached scalar" begin
        for i in 1:(x_cached_range.len - 1)
            @test _get_h(x_cached_range, i) === x_cached_range.h
            @test _get_inv_h(x_cached_range, i) === x_cached_range.inv_h
        end
    end

    @testset "_CachedVector — cached vector lookup" begin
        for i in 1:(length(x_cached_vec) - 1)
            @test _get_h(x_cached_vec, i) === x_cached_vec.h[i]
            @test _get_inv_h(x_cached_vec, i) === x_cached_vec.inv_h[i]
        end

        # Cached values match on-the-fly computation
        for i in 1:(length(x_cached_vec) - 1)
            @test _get_h(x_cached_vec, i) == _get_h(x_vec_nonuniform, i)
            @test _get_inv_h(x_cached_vec, i) ≈ _get_inv_h(x_vec_nonuniform, i)
        end
    end

    @testset "Cross-representation numerical equivalence" begin
        # _get_h on uniform-Vector vs _CachedRange (representations of same grid)
        for i in 1:(length(x_vec_uniform) - 1)
            @test _get_h(x_vec_uniform, i) ≈ _get_h(x_cached_range, i)
            @test _get_inv_h(x_vec_uniform, i) ≈ _get_inv_h(x_cached_range, i)
        end
    end

    @testset "Dispatch specialization (no ambiguity)" begin
        # Each grid type lands on its most-specific 2-arg `_get_h(grid, i::Int)`
        @test which(_get_h, Tuple{Vector{Float64}, Int}).sig <: Tuple{Any, AbstractVector, Int}
        @test which(_get_h, Tuple{typeof(x_range), Int}).sig <: Tuple{Any, AbstractRange, Int}
        @test which(_get_h, Tuple{typeof(x_cached_range), Int}).sig <:
            Tuple{Any, _CachedRange, Int}
        @test which(_get_h, Tuple{typeof(x_cached_vec), Int}).sig <:
            Tuple{Any, _CachedVector, Int}
    end

    @testset "Int grid → Float h" begin
        # `_get_h` always promotes to float for kernel compatibility
        x_int = [0, 1, 3, 6]
        c_int = _CachedVector(x_int)

        @test _get_h(x_int, 1) === 1.0  # float() of (x[2]-x[1])=1
        @test _get_h(x_int, 2) === 2.0
        @test _get_inv_h(x_int, 2) === 0.5

        # _CachedVector for Int grid: h stays Int, inv_h is Float64
        @test _get_h(c_int, 1) === 1   # raw cached Int
        @test _get_inv_h(c_int, 2) === 0.5
    end

    @testset "Float32 grid preserves precision" begin
        x_f32 = Float32[0.0, 0.25, 0.75, 1.0]
        c_f32 = _CachedVector(x_f32)

        @test _get_h(c_f32, 1) === Float32(0.25)
        @test _get_inv_h(c_f32, 1) === Float32(4.0)
        @test eltype(c_f32.h) == Float32
        @test eltype(c_f32.inv_h) == Float32
    end
end

@testitem "Explicit _CachedVector wrapping over _store_grid result" begin
    # Step 1 keeps _store_grid lightweight (Vector → Vector). Per-method
    # constructors that want spacing caching wrap explicitly via:
    #   xc = _CachedVector(_store_grid(x, Tg))
    # This testitem documents that idiom and verifies the resulting type.
    using FastInterpolations: _store_grid, _CachedVector

    @testset "Explicit wrap idiom" begin
        x = [0.0, 0.3, 0.7, 1.0]
        xc = _CachedVector(_store_grid(x, Float64))
        @test xc isa _CachedVector{Float64, Float64}
        @test xc.h ≈ [0.3, 0.4, 0.3]
    end

    @testset "Vector{Int} promoted to Float64 → _CachedVector{Float64, Float64}" begin
        x = [0, 1, 3, 6]
        xc = _CachedVector(_store_grid(x, Float64))
        @test xc isa _CachedVector{Float64, Float64}
        @test xc.inner == [0.0, 1.0, 3.0, 6.0]
        @test xc.h == [1.0, 2.0, 3.0]
    end
end
