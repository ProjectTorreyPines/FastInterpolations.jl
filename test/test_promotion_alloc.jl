# ========================================
# Type Promotion Allocation Tests
# ========================================
#
# Verifies that type-mismatched inputs don't cause unnecessary heap allocations:
# 1. One-shot in-place: zero-alloc even with Float32 query / Int data on Float64 grid
# 2. Interpolant construction: no double-copy when y needs type conversion

# ALLOC_THRESHOLD is defined in runtests.jl (0 on Julia 1.12+, 240 on older)

@testset "Type Promotion Allocations" begin

    # ========================================
    # Setup: Float64 grid, various data/query types
    # ========================================
    x_f64 = collect(range(0.0, 5.0, 50))
    y_f64 = sin.(x_f64)
    y_f32 = Float32.(y_f64)
    xq_f64 = collect(range(0.1, 4.9, 200))
    xq_f32 = Float32.(xq_f64)
    out = Vector{Float64}(undef, 200)

    # ========================================
    # One-shot in-place: zero-alloc with mixed types
    # ========================================
    @testset "One-shot in-place: Float32 query on Float64 grid" begin
        # Function barriers for accurate @allocated measurement

        function _bench_linear!(out, x, y, xq)
            linear_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_pchip!(out, x, y, xq)
            pchip_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_cubic!(out, x, y, xq)
            cubic_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_quadratic!(out, x, y, xq)
            quadratic_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end

        # Warmup all paths
        for f in (_bench_linear!, _bench_pchip!, _bench_cubic!, _bench_quadratic!)
            f(out, x_f64, y_f64, xq_f64)
            f(out, x_f64, y_f64, xq_f32)
        end

        # Baseline: matched types → zero-alloc
        @testset "baseline (Float64 query)" begin
            @test (@allocated _bench_linear!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_pchip!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_cubic!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_quadratic!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
        end

        # Mixed query type: should also be zero-alloc (no query promotion)
        @testset "Float32 query → zero-alloc" begin
            @test (@allocated _bench_linear!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_pchip!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_cubic!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_quadratic!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD
        end
    end

    @testset "One-shot in-place: Float32 data on Float64 grid" begin
        function _bench_linear_f32data!(out, x, y, xq)
            linear_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_pchip_f32data!(out, x, y, xq)
            pchip_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end

        # Warmup
        _bench_linear_f32data!(out, x_f64, y_f32, xq_f64)
        _bench_pchip_f32data!(out, x_f64, y_f32, xq_f64)

        # Mixed data type: should also be zero-alloc (no data promotion in one-shot)
        @testset "Float32 data → zero-alloc" begin
            @test (@allocated _bench_linear_f32data!(out, x_f64, y_f32, xq_f64)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_pchip_f32data!(out, x_f64, y_f32, xq_f64)) <= ALLOC_THRESHOLD
        end
    end

    # ========================================
    # Correctness: mixed types produce correct results
    # ========================================
    @testset "Mixed-type correctness" begin
        ref_linear = linear_interp(x_f64, y_f64, xq_f64; extrap = ExtendExtrap())

        # Float32 query — slightly different due to Float32→Float64 per-element promotion
        out_f32q = similar(out)
        linear_interp!(out_f32q, x_f64, y_f64, xq_f32; extrap = ExtendExtrap())
        @test out_f32q ≈ ref_linear rtol = 1e-6

        # Float32 data
        out_f32d = similar(out)
        linear_interp!(out_f32d, x_f64, y_f32, xq_f64; extrap = ExtendExtrap())
        @test out_f32d ≈ ref_linear rtol = 1e-6  # Float32 data has less precision
    end

    # ========================================
    # Interpolant construction: no double-copy
    # ========================================
    @testset "Interpolant construction: single-copy for type conversion" begin
        # For Float32 data on Float64 grid, the constructor should:
        # - Convert + copy y in one step (not promote then copy)
        # - Total y-related allocation ≈ sizeof(Float64) * length(y), NOT 2×
        n = 1000
        x_big = collect(range(0.0, 5.0, n))
        y_big_f32 = Float32.(sin.(x_big))
        y_big_f64 = Float64.(y_big_f32)

        # Baseline: Float64 data (just copy, no conversion)
        function _bench_linear_itp_f64(x, y)
            itp = linear_interp(x, y)
            return nothing
        end
        # Mixed: Float32 data (convert + copy)
        function _bench_linear_itp_f32(x, y)
            itp = linear_interp(x, y)
            return nothing
        end

        _bench_linear_itp_f64(x_big, y_big_f64)
        _bench_linear_itp_f32(x_big, y_big_f32)

        alloc_f64 = @allocated _bench_linear_itp_f64(x_big, y_big_f64)
        alloc_f32 = @allocated _bench_linear_itp_f32(x_big, y_big_f32)

        # Float32 path should NOT allocate significantly more than Float64 path.
        # With double-copy: alloc_f32 ≈ alloc_f64 + sizeof(Float64)*n (extra promote copy)
        # Without double-copy: alloc_f32 ≈ alloc_f64 (same single copy, just converts)
        # Allow small overhead for intermediate computations but not a full extra vector.
        extra_vector_bytes = sizeof(Float64) * n  # 8000 bytes for n=1000
        @test alloc_f32 - alloc_f64 < extra_vector_bytes ÷ 2  # less than half an extra vector
    end
end
