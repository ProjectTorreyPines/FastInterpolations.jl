# ========================================
# Type Promotion Allocation Tests
# ========================================
#
# Verifies that type-mismatched inputs don't cause unnecessary heap allocations:
# 1. One-shot in-place: zero-alloc even with Float32 query / Int data on Float64 grid
# 2. Interpolant construction: no double-copy when y needs type conversion

# ALLOC_THRESHOLD is defined in test/setup.jl (0 on Julia 1.12+, 240 on older)

@testitem "Type Promotion Allocations" setup = [AllocConstants] begin

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

        # cubic_interp! oneshot leaks exactly 16 B under `--code-coverage` due
        # to instrumentation interacting with its `@with_pool` + atomic cache
        # lookup chain (linear/pchip/quadratic don't use that machinery, so they
        # stay strict). Real allocs are always 0 — single-file runs (no coverage)
        # still verify that.
        _cubic_cov_slack = Base.JLOptions().code_coverage != 0 ? 16 : 0

        # Baseline: matched types → zero-alloc
        @testset "baseline (Float64 query)" begin
            @test (@allocated _bench_linear!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_pchip!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_cubic!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD + _cubic_cov_slack
            @test (@allocated _bench_quadratic!(out, x_f64, y_f64, xq_f64)) <= ALLOC_THRESHOLD
        end

        # Mixed query type: should also be zero-alloc (no query promotion)
        @testset "Float32 query → zero-alloc" begin
            @test (@allocated _bench_linear!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_pchip!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD
            @test (@allocated _bench_cubic!(out, x_f64, y_f64, xq_f32)) <= ALLOC_THRESHOLD + _cubic_cov_slack
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
        @test out_f32q ≈ ref_linear rtol = 1.0e-6

        # Float32 data
        out_f32d = similar(out)
        linear_interp!(out_f32d, x_f64, y_f32, xq_f64; extrap = ExtendExtrap())
        @test out_f32d ≈ ref_linear rtol = 1.0e-6  # Float32 data has less precision
    end

    # ========================================
    # Int grid + Int y: all methods produce correct Float results
    # ========================================
    @testset "Int grid + Int y correctness" begin
        x_int = [0, 1, 2, 3, 4]
        y_int = [0, 1, 3, 2, 5]
        xq_test = [0.5, 1.5, 2.5, 3.5]

        # Allocating batch (all methods)
        @test linear_interp(x_int, y_int, xq_test) isa Vector{Float64}
        @test constant_interp(x_int, y_int, xq_test) isa Vector{Float64}
        @test quadratic_interp(x_int, y_int, xq_test) isa Vector{Float64}
        @test cubic_interp(x_int, y_int, xq_test; extrap = ExtendExtrap()) isa Vector{Float64}
        @test pchip_interp(x_int, y_int, xq_test; extrap = ExtendExtrap()) isa Vector{Float64}
        @test cardinal_interp(x_int, y_int, xq_test; extrap = ExtendExtrap()) isa Vector{Float64}
        @test akima_interp(x_int, y_int, xq_test; extrap = ExtendExtrap()) isa Vector{Float64}

        # In-place batch with PreCompute (Hermite family)
        out = zeros(4)
        @test begin
            pchip_interp!(out, x_int, y_int, xq_test; coeffs = PreCompute(), extrap = ExtendExtrap()); true
        end
        @test begin
            cardinal_interp!(out, x_int, y_int, xq_test; coeffs = PreCompute(), extrap = ExtendExtrap()); true
        end
        @test begin
            akima_interp!(out, x_int, y_int, xq_test; coeffs = PreCompute(), extrap = ExtendExtrap()); true
        end

        # Cubic + Quadratic in-place batch
        @test begin
            cubic_interp!(out, x_int, y_int, xq_test; bc = ZeroCurvBC(), extrap = ExtendExtrap()); true
        end
        @test begin
            quadratic_interp!(out, x_int, y_int, xq_test; extrap = ExtendExtrap()); true
        end
    end

    # ========================================
    # Int grid: zero-alloc scalar and in-place batch one-shot
    # ========================================
    @testset "Int grid one-shot: zero-alloc scalar" begin
        # Function barriers with typed args — avoid closure capture boxing
        function _bench_linear_int(x, y, xq)
            linear_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_constant_int(x, y, xq)
            constant_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_quadratic_int(x, y, xq)
            quadratic_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_cubic_int(x, y, xq)
            cubic_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_pchip_int(x, y, xq)
            pchip_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_cardinal_int(x, y, xq)
            cardinal_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_akima_int(x, y, xq)
            akima_interp(x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_hermite_int(x, y, dy, xq)
            hermite_interp(x, y, dy, xq; extrap = ExtendExtrap())
            return nothing
        end

        x_int = [0, 1, 2, 3, 4]
        y_flt = sin.(Float64.(x_int))
        dy_flt = cos.(Float64.(x_int))
        xq_s = 1.5

        # Warmup
        for f in (
                _bench_linear_int, _bench_constant_int, _bench_quadratic_int,
                _bench_cubic_int, _bench_pchip_int, _bench_cardinal_int, _bench_akima_int,
            )
            f(x_int, y_flt, xq_s); f(x_int, y_flt, xq_s)
        end
        _bench_hermite_int(x_int, y_flt, dy_flt, xq_s)
        _bench_hermite_int(x_int, y_flt, dy_flt, xq_s)

        @test (@allocated _bench_linear_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_constant_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_quadratic_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_cubic_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_pchip_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_cardinal_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_akima_int(x_int, y_flt, xq_s)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_hermite_int(x_int, y_flt, dy_flt, xq_s)) <= ALLOC_THRESHOLD
    end

    @testset "Int grid one-shot: zero-alloc in-place batch" begin
        function _bench_linear_int!(out, x, y, xq)
            linear_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_constant_int!(out, x, y, xq)
            constant_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_quadratic_int!(out, x, y, xq)
            quadratic_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_cubic_int!(out, x, y, xq)
            cubic_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_pchip_int!(out, x, y, xq)
            pchip_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_cardinal_int!(out, x, y, xq)
            cardinal_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_akima_int!(out, x, y, xq)
            akima_interp!(out, x, y, xq; extrap = ExtendExtrap())
            return nothing
        end
        function _bench_hermite_int!(out, x, y, dy, xq)
            hermite_interp!(out, x, y, dy, xq; extrap = ExtendExtrap())
            return nothing
        end

        x_int = [0, 1, 2, 3, 4]
        y_flt = sin.(Float64.(x_int))
        dy_flt = cos.(Float64.(x_int))
        xq_v = [0.5, 1.5, 2.5, 3.5]
        out4 = Vector{Float64}(undef, 4)

        for f in (
                _bench_linear_int!, _bench_constant_int!, _bench_quadratic_int!,
                _bench_cubic_int!, _bench_pchip_int!, _bench_cardinal_int!, _bench_akima_int!,
            )
            f(out4, x_int, y_flt, xq_v); f(out4, x_int, y_flt, xq_v)
        end
        _bench_hermite_int!(out4, x_int, y_flt, dy_flt, xq_v)
        _bench_hermite_int!(out4, x_int, y_flt, dy_flt, xq_v)

        @test (@allocated _bench_linear_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_constant_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_quadratic_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_cubic_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_pchip_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_cardinal_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_akima_int!(out4, x_int, y_flt, xq_v)) <= ALLOC_THRESHOLD
        @test (@allocated _bench_hermite_int!(out4, x_int, y_flt, dy_flt, xq_v)) <= ALLOC_THRESHOLD
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
