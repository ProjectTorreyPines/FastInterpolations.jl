#=
Thread-Safety Tests for FastInterpolations.jl

Comprehensive tests for:
- Phase 1: Ring buffer safety (DCL, sizehint)
- Phase 2: Workspace safety (AdaptiveArrayPools)

Run with: julia -t 8 --project -e 'using Pkg; Pkg.test(test_args=["test_thread_safety.jl"])'
=#

using Test
using FastInterpolations
using Base.Threads

# Skip all tests if single-threaded
if nthreads() == 1
    @warn "Thread-safety tests require multiple threads. Run with: julia -t 4"
    @testset "Thread Safety (skipped)" begin
        @test_skip "Need multiple threads"
    end
else

# =========================================================================
# Group 1: Ring Buffer Safety (Phase 1)
# =========================================================================
@testset "Ring Buffer Safety" begin
    @testset "Registry race" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)

        @threads for i in 1:1000
            try
                n_points = 10 + (i % 20)
                x = collect(range(0.0, 1.0, n_points))
                y = sin.(2π .* x)
                cubic_interp(x, y, 0.5; autocache=true)
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Store push race" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)

        @threads for i in 1:1000
            try
                offset = i * 0.001
                x = collect(range(offset, 1.0 + offset, 51))
                y = sin.(x)
                cubic_interp(x, y, 0.5 + offset; autocache=true)
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Mixed BC" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)

        @threads for i in 1:1000
            try
                x = collect(range(0.0, 1.0, 51))
                y = sin.(2π .* x)

                bc = if i % 3 == 0
                    NaturalBC()
                elseif i % 3 == 1
                    ClampedBC()
                else
                    y[end] = y[1]
                    PeriodicBC()
                end

                cubic_interp(x, y, 0.5; bc=bc, autocache=true)
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end
end

# =========================================================================
# Group 2: Workspace Correctness (Phase 2)
# =========================================================================
@testset "Workspace Correctness" begin
    @testset "Scalar query" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        n_iter = 1000

        results_cached = Vector{Float64}(undef, n_iter)
        results_nocache = Vector{Float64}(undef, n_iter)

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)
            results_cached[i] = cubic_interp(x, y, 0.25; autocache=true)
            results_nocache[i] = cubic_interp(x, y, 0.25; autocache=false)
        end

        errors = abs.(results_cached .- results_nocache)
        @test count(e -> e > 1e-10, errors) == 0
    end

    @testset "Vector query" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        x_query = [0.2, 0.4, 0.6, 0.8]
        n_iter = 1000

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            result_cached = cubic_interp(x, y, x_query; autocache=true)
            result_nocache = cubic_interp(x, y, x_query; autocache=false)

            err = maximum(abs.(result_cached .- result_nocache))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end

    @testset "Periodic BC" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 2π, 51))
        n_iter = 1000

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(x)

            result_cached = cubic_interp(x, y, π; bc=PeriodicBC(), autocache=true)
            result_nocache = cubic_interp(x, y, π; bc=PeriodicBC(), autocache=false)

            err = abs(result_cached - result_nocache)
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end
end

# =========================================================================
# Group 3: In-place API & CubicInterpolant
# =========================================================================
@testset "In-place & Interpolant" begin
    @testset "cubic_interp!" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        x_query = [0.25, 0.5, 0.75]
        n_iter = 1000

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            output_cached = Vector{Float64}(undef, 3)
            output_nocache = Vector{Float64}(undef, 3)

            cubic_interp!(output_cached, x, y, x_query; autocache=true)
            cubic_interp!(output_nocache, x, y, x_query; autocache=false)

            err = maximum(abs.(output_cached .- output_nocache))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end

    @testset "Explicit cache" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        x_query = [0.25, 0.5, 0.75]
        cache = FastInterpolations._get_cubic_cache(x, NaturalBC())
        n_iter = 300

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            output = Vector{Float64}(undef, 3)
            cubic_interp!(output, cache, y, x_query)

            expected = amplitude .* sin.(2π .* x_query)
            err = maximum(abs.(output .- expected))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        # Tolerance: cubic interpolation of sin() with 51 points has ~1% max error
        @test max_err[] < 0.01
    end

    @testset "Interpolant creation" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)

        @threads for i in 1:200
            try
                x = collect(range(0.0, 1.0, 51))
                y = sin.(2π .* x) .* Float64(mod1(i, 5))
                itp = cubic_interp(x, y; autocache=true)
                @assert isfinite(itp(0.5))
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Interpolant correctness" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        x_query = [0.2, 0.4, 0.6, 0.8]
        n_iter = 200

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            itp = cubic_interp(x, y; autocache=true)
            result = itp(x_query)
            expected = cubic_interp(x, y, x_query; autocache=false)

            err = maximum(abs.(result .- expected))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end
end

# =========================================================================
# Group 4: Float32 & Derivatives
# =========================================================================
@testset "Float32 & Derivatives" begin
    @testset "Float32 scalar" begin
        FastInterpolations.clear_cubic_cache!()
        x = Float32.(collect(range(0.0, 1.0, 51)))
        n_iter = 300

        max_err = Ref(0.0f0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float32(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            result_cached = cubic_interp(x, y, 0.25f0; autocache=true)
            result_nocache = cubic_interp(x, y, 0.25f0; autocache=false)

            err = abs(result_cached - result_nocache)
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1f-5
    end

    @testset "Float32 vector" begin
        FastInterpolations.clear_cubic_cache!()
        x = Float32.(collect(range(0.0, 1.0, 51)))
        x_query = Float32[0.25, 0.5, 0.75]
        n_iter = 200

        max_err = Ref(0.0f0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float32(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            result_cached = cubic_interp(x, y, x_query; autocache=true)
            result_nocache = cubic_interp(x, y, x_query; autocache=false)

            err = maximum(abs.(result_cached .- result_nocache))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1f-5
    end

    @testset "First derivative evaluation (deriv=DerivOp(1))" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        x_query = [0.25, 0.5, 0.75]
        n_iter = 200

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            result_cached = cubic_interp(x, y, x_query; autocache=true, deriv=DerivOp(1))
            result_nocache = cubic_interp(x, y, x_query; autocache=false, deriv=DerivOp(1))

            err = maximum(abs.(result_cached .- result_nocache))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end

    @testset "deriv=DerivOp(2)" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        x_query = [0.25, 0.5, 0.75]
        n_iter = 200

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            result_cached = cubic_interp(x, y, x_query; autocache=true, deriv=DerivOp(2))
            result_nocache = cubic_interp(x, y, x_query; autocache=false, deriv=DerivOp(2))

            err = maximum(abs.(result_cached .- result_nocache))
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end
end

# =========================================================================
# Group 5: BCPair (Mixed Boundary Conditions)
# =========================================================================
@testset "BCPair" begin
    @testset "Mixed BC creation" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)

        @threads for i in 1:150
            try
                x = collect(range(0.0, 1.0, 51))
                y = sin.(2π .* x) .* Float64(mod1(i, 5))
                bc = BCPair(Deriv1(0.0), Deriv2(0.0))
                result = cubic_interp(x, y, 0.5; bc=bc, autocache=true)
                @assert isfinite(result)
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "BCPair correctness" begin
        FastInterpolations.clear_cubic_cache!()
        x = collect(range(0.0, 1.0, 51))
        bc = BCPair(Deriv1(0.0), Deriv1(0.0))
        n_iter = 200

        max_err = Ref(0.0)
        lk = ReentrantLock()

        @threads for i in 1:n_iter
            amplitude = Float64(mod1(i, 10))
            y = amplitude .* sin.(2π .* x)

            result_cached = cubic_interp(x, y, 0.5; bc=bc, autocache=true)
            result_nocache = cubic_interp(x, y, 0.5; bc=bc, autocache=false)

            err = abs(result_cached - result_nocache)
            lock(lk) do
                max_err[] = max(max_err[], err)
            end
        end

        @test max_err[] < 1e-10
    end
end

# =========================================================================
# Group 6: Concurrent Insert (Writer Lock Contention)
# =========================================================================
@testset "Concurrent Insert" begin
    @testset "Derivative BC" begin
        # Repeated clear + concurrent insert to test writer lock contention
        # Two threads racing to insert same cache key → one waits, then re-checks
        # Using @threads for guaranteed parallel execution on separate OS threads
        errors = Atomic{Int}(0)

        for _ in 1:5000
            FastInterpolations.clear_cubic_cache!()
            x = collect(range(0.0, 1.0, 51))
            y = sin.(2π .* x)

            @threads for _ in 1:2
                try
                    cubic_interp(x, y, 0.5; autocache=true)
                catch
                    atomic_add!(errors, 1)
                end
            end
        end

        @test errors[] == 0
    end

    @testset "Periodic BC" begin
        errors = Atomic{Int}(0)

        for _ in 1:5000
            FastInterpolations.clear_cubic_cache!()
            x = collect(range(0.0, 2π, 51))
            y = sin.(x)

            @threads for _ in 1:2
                try
                    cubic_interp(x, y, π; bc=PeriodicBC(), autocache=true)
                catch
                    atomic_add!(errors, 1)
                end
            end
        end

        @test errors[] == 0
    end
end

# =========================================================================
# Group 5: CubicSeriesInterpolant Thread Safety
# =========================================================================
@testset "CubicSeriesInterpolant Thread Safety" begin
    @testset "Lazy transpose concurrent initialization" begin
        # Test that concurrent scalar queries safely initialize the lazy transpose
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)
        n_iter = 500

        x = collect(range(0.0, 1.0, 101))
        y1, y2 = sin.(2π .* x), cos.(2π .* x)
        mitp = cubic_interp(x, [y1, y2])  # No precompute_transpose

        # Pre-compute expected values with deterministic query points
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        xq_values = collect(range(0.1, 0.9, n_iter))

        _ = mitp(0.5)  # Trigger lazy transpose (warm up)

        @threads for i in 1:n_iter
            try
                xq = xq_values[i]
                result = mitp(xq)

                # Verify correctness against single-interpolant results
                @assert result[1] ≈ itp1(xq) atol=1e-12
                @assert result[2] ≈ itp2(xq) atol=1e-12
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Concurrent scalar evaluation" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)
        n_iter = 500

        x = collect(range(0.0, 1.0, 101))
        y1, y2, y3 = sin.(2π .* x), cos.(2π .* x), exp.(-3 .* x)
        mitp = cubic_interp(x, [y1, y2, y3]; precompute_transpose=true)

        # Pre-compute expected values with deterministic query points
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        itp3 = cubic_interp(x, y3)
        xq_values = collect(range(0.1, 0.9, n_iter))

        @threads for i in 1:n_iter
            try
                xq = xq_values[i]
                result = mitp(xq)

                # Verify correctness
                @assert result[1] ≈ itp1(xq) atol=1e-12
                @assert result[2] ≈ itp2(xq) atol=1e-12
                @assert result[3] ≈ itp3(xq) atol=1e-12
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Concurrent in-place scalar evaluation" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)
        n_iter = 500

        x = collect(range(0.0, 1.0, 101))
        y1, y2 = sin.(2π .* x), cos.(2π .* x)
        mitp = cubic_interp(x, [y1, y2]; precompute_transpose=true)

        # Pre-compute expected values with deterministic query points
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        xq_values = collect(range(0.1, 0.9, n_iter))

        @threads for i in 1:n_iter
            try
                # Task-local output buffer (safe from task migration)
                output = zeros(2)
                xq = xq_values[i]
                mitp(output, xq)

                # Verify correctness against single-interpolant results
                @assert output[1] ≈ itp1(xq) atol=1e-12
                @assert output[2] ≈ itp2(xq) atol=1e-12
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Concurrent vector evaluation" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)
        n_iter = 200

        x = collect(range(0.0, 1.0, 101))
        y1, y2 = sin.(2π .* x), cos.(2π .* x)
        mitp = cubic_interp(x, [y1, y2])

        # Pre-compute expected values
        itp1 = cubic_interp(x, y1)
        itp2 = cubic_interp(x, y2)
        xq = collect(range(0.1, 0.9, 50))
        expected1 = itp1.(xq)
        expected2 = itp2.(xq)

        @threads for i in 1:n_iter
            try
                # Task-local output buffers (safe from task migration)
                outputs = [zeros(50), zeros(50)]
                mitp(outputs, xq)

                # Verify correctness against pre-computed expected values
                @assert outputs[1] ≈ expected1 atol=1e-12
                @assert outputs[2] ≈ expected2 atol=1e-12
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end

    @testset "Concurrent construction with autocache" begin
        FastInterpolations.clear_cubic_cache!()
        errors = Atomic{Int}(0)
        n_iter = 200

        x = collect(range(0.0, 1.0, 51))

        @threads for i in 1:n_iter
            try
                y1 = sin.(2π .* x .+ i * 0.01)
                y2 = cos.(2π .* x .+ i * 0.01)
                mitp = cubic_interp(x, [y1, y2]; autocache=true)
                result = mitp(0.5)
                @assert length(result) == 2
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
    end
end

# =========================================================================
# Group 7: Search Policy Thread Safety
# =========================================================================
@testset "Search Policy Thread Safety" begin
    @testset "LinearBinary concurrent access" begin
        x = collect(range(0.0, 1.0, 1001))
        y = sin.(2π .* x)
        itp = linear_interp(x, y)

        results = Vector{Float64}(undef, nthreads() * 1000)
        errors = Atomic{Int}(0)

        @threads for i in eachindex(results)
            try
                results[i] = itp(rand(); search=LinearBinary())
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
        @test all(isfinite, results)
    end

    @testset "HintedBinary concurrent access" begin
        x = collect(range(0.0, 1.0, 1001))
        y = sin.(2π .* x)
        itp = linear_interp(x, y)

        results = Vector{Float64}(undef, nthreads() * 1000)
        errors = Atomic{Int}(0)

        @threads for i in eachindex(results)
            try
                results[i] = itp(rand(); search=HintedBinary())
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
        @test all(isfinite, results)
    end

    @testset "Binary concurrent access" begin
        x = collect(range(0.0, 1.0, 1001))
        y = sin.(2π .* x)
        itp = linear_interp(x, y)

        results = Vector{Float64}(undef, nthreads() * 1000)
        errors = Atomic{Int}(0)

        @threads for i in eachindex(results)
            try
                results[i] = itp(rand(); search=Binary())
            catch
                atomic_add!(errors, 1)
            end
        end

        @test errors[] == 0
        @test all(isfinite, results)
    end
end

end  # if nthreads() > 1

# Print summary if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("Thread Safety Tests Complete")
    println("Threads used: $(nthreads())")
    println("="^60)
end
