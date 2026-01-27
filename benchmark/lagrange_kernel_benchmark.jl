# ========================================
# Lagrange Kernel Benchmark: Precomputed vs On-the-fly
# ========================================
# Compares performance of:
#   1. On-the-fly: computing coefficients every call
#   2. Precomputed: computing coefficients once, reusing for batch

using FastInterpolations
using BenchmarkTools
using Printf

# Access internal functions
const _lagrange_coeffs_left = FastInterpolations._lagrange_coeffs_left
const _lagrange_coeffs_right = FastInterpolations._lagrange_coeffs_right
const _lagrange_d1_nonuniform = FastInterpolations._lagrange_d1_nonuniform
const _estimate_endpoint_derivative = FastInterpolations._estimate_endpoint_derivative
const _lagrange_d1_left_uniform = FastInterpolations._lagrange_d1_left_uniform
const _lagrange_d1_right_uniform = FastInterpolations._lagrange_d1_right_uniform

println("=" ^ 70)
println("Lagrange Endpoint Derivative Kernel Benchmark")
println("=" ^ 70)
println()

# ========================================
# Setup
# ========================================

# Non-uniform grid
n_points = 21
xs_nonuniform = sort(rand(n_points)) .* 2.0  # Random points in [0, 2]
xs_nonuniform[1] = 0.0
xs_nonuniform[end] = 2.0

# Uniform grid
xs_uniform = range(0.0, 2.0, n_points)

# Multiple y vectors (simulating batch)
n_vectors = 10_000
ys_batch = [sin.(2π .* xs_nonuniform .+ rand()) .+ 0.1 .* randn(n_points) for _ in 1:n_vectors]
ys_batch_uniform = [sin.(2π .* collect(xs_uniform) .+ rand()) .+ 0.1 .* randn(n_points) for _ in 1:n_vectors]

println("Setup:")
println("  Grid points: $n_points")
println("  Batch size: $n_vectors vectors")
println()

# ========================================
# Benchmark 1: Non-Uniform Grid
# ========================================

println("-" ^ 70)
println("NON-UNIFORM GRID")
println("-" ^ 70)

# Precompute coefficients
c_left = _lagrange_coeffs_left(xs_nonuniform[1], xs_nonuniform[2], xs_nonuniform[3], xs_nonuniform[4])
c_right = _lagrange_coeffs_right(xs_nonuniform[end-3], xs_nonuniform[end-2], xs_nonuniform[end-1], xs_nonuniform[end])

# Method 1: On-the-fly (existing implementation)
function bench_onfly_nonuniform(xs, ys_batch)
    results = zeros(2, length(ys_batch))
    @inbounds for (j, ys) in enumerate(ys_batch)
        results[1, j] = _estimate_endpoint_derivative(xs, ys, Val(:left))
        results[2, j] = _estimate_endpoint_derivative(xs, ys, Val(:right))
    end
    return results
end

# Method 2: Precomputed coefficients
function bench_precomputed_nonuniform(c_left, c_right, ys_batch)
    results = zeros(2, length(ys_batch))
    @inbounds for (j, ys) in enumerate(ys_batch)
        results[1, j] = _lagrange_d1_nonuniform(c_left..., ys[1], ys[2], ys[3], ys[4])
        n = length(ys)
        results[2, j] = _lagrange_d1_nonuniform(c_right..., ys[n-3], ys[n-2], ys[n-1], ys[n])
    end
    return results
end

# Warmup
bench_onfly_nonuniform(xs_nonuniform, ys_batch[1:10])
bench_precomputed_nonuniform(c_left, c_right, ys_batch[1:10])

# Benchmark
println("\n[1] On-the-fly (computing coeffs each call):")
t_onfly = @benchmark bench_onfly_nonuniform($xs_nonuniform, $ys_batch) samples=20 evals=1
display(t_onfly)

println("\n[2] Precomputed coefficients:")
t_precomp = @benchmark bench_precomputed_nonuniform($c_left, $c_right, $ys_batch) samples=20 evals=1
display(t_precomp)

# Verify correctness
r1 = bench_onfly_nonuniform(xs_nonuniform, ys_batch)
r2 = bench_precomputed_nonuniform(c_left, c_right, ys_batch)
@assert r1 ≈ r2 "Results don't match!"

speedup_nonuniform = median(t_onfly).time / median(t_precomp).time
@printf("\n→ Speedup (precomputed vs on-the-fly): %.2fx\n", speedup_nonuniform)

# ========================================
# Benchmark 2: Uniform Grid
# ========================================

println()
println("-" ^ 70)
println("UNIFORM GRID")
println("-" ^ 70)

inv_h = 1 / step(xs_uniform)

# Method 1: On-the-fly (using Range dispatch)
function bench_onfly_uniform(xs, ys_batch)
    results = zeros(2, length(ys_batch))
    @inbounds for (j, ys) in enumerate(ys_batch)
        results[1, j] = _estimate_endpoint_derivative(xs, ys, Val(:left))
        results[2, j] = _estimate_endpoint_derivative(xs, ys, Val(:right))
    end
    return results
end

# Method 2: Direct kernel call with precomputed inv_h
function bench_direct_uniform(inv_h, ys_batch)
    results = zeros(2, length(ys_batch))
    @inbounds for (j, ys) in enumerate(ys_batch)
        results[1, j] = _lagrange_d1_left_uniform(ys[1], ys[2], ys[3], ys[4], inv_h)
        n = length(ys)
        results[2, j] = _lagrange_d1_right_uniform(ys[n-3], ys[n-2], ys[n-1], ys[n], inv_h)
    end
    return results
end

# Warmup
bench_onfly_uniform(xs_uniform, ys_batch_uniform[1:10])
bench_direct_uniform(inv_h, ys_batch_uniform[1:10])

println("\n[1] On-the-fly (via _estimate_endpoint_derivative):")
t_onfly_u = @benchmark bench_onfly_uniform($xs_uniform, $ys_batch_uniform) samples=20 evals=1
display(t_onfly_u)

println("\n[2] Direct kernel call (precomputed inv_h):")
t_direct_u = @benchmark bench_direct_uniform($inv_h, $ys_batch_uniform) samples=20 evals=1
display(t_direct_u)

# Verify correctness
r1u = bench_onfly_uniform(xs_uniform, ys_batch_uniform)
r2u = bench_direct_uniform(inv_h, ys_batch_uniform)
@assert r1u ≈ r2u "Results don't match!"

speedup_uniform = median(t_onfly_u).time / median(t_direct_u).time
@printf("\n→ Speedup (direct vs on-the-fly): %.2fx\n", speedup_uniform)

# ========================================
# Summary
# ========================================

println()
println("=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println()
@printf("Non-uniform grid:\n")
@printf("  On-the-fly:  %8.3f ms\n", median(t_onfly).time / 1e6)
@printf("  Precomputed: %8.3f ms\n", median(t_precomp).time / 1e6)
@printf("  Speedup:     %8.2fx\n", speedup_nonuniform)
println()
@printf("Uniform grid:\n")
@printf("  On-the-fly:  %8.3f ms\n", median(t_onfly_u).time / 1e6)
@printf("  Direct:      %8.3f ms\n", median(t_direct_u).time / 1e6)
@printf("  Speedup:     %8.2fx\n", speedup_uniform)
println()
@printf("Per-vector cost (non-uniform):\n")
@printf("  On-the-fly:  %8.1f ns/vector\n", median(t_onfly).time / n_vectors)
@printf("  Precomputed: %8.1f ns/vector\n", median(t_precomp).time / n_vectors)
println()
