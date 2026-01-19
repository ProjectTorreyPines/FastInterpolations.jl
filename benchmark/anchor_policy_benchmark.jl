#!/usr/bin/env julia
# ========================================
# Anchor Policy Integration Benchmark
# ========================================
# Measures the performance benefit of LinearBoundedAlg{8} for batch anchor operations.
# The optimization targets monotonic-ish queries where hint caching can avoid binary search.
#
# Expected results:
# - Monotonic queries: Improvement due to linear search hitting the hint
# - Random queries: No regression (falls back to binary search)

using FastInterpolations
using BenchmarkTools

println("=" ^ 60)
println("Anchor Policy Integration Benchmark")
println("=" ^ 60)

# ========================================
# Setup
# ========================================

x = collect(range(0.0, 1.0, 10001))  # Large grid (10001 points)

# Monotonic queries (best case for bounded linear - sequential access)
xq_monotonic = collect(range(0.001, 0.999, 1000))

# Random queries (worst case - no locality benefit)
xq_random = rand(1000)

# ========================================
# Linear Interpolation Anchors
# ========================================

println("\n## Linear Anchor Benchmark (10001 grid points, 1000 queries)")
println("-" ^ 60)

buffer_linear = Vector{FastInterpolations._LinearAnchoredQuery{Float64}}(undef, 1000)

print("Monotonic queries: ")
t_mono_linear = @belapsed FastInterpolations._fill_anchors!($buffer_linear, $x, $xq_monotonic, Val(:linear))
println("$(round(t_mono_linear * 1e6, digits=2)) μs ($(round(t_mono_linear * 1e9 / 1000, digits=2)) ns/query)")

print("Random queries:    ")
t_rand_linear = @belapsed FastInterpolations._fill_anchors!($buffer_linear, $x, $xq_random, Val(:linear))
println("$(round(t_rand_linear * 1e6, digits=2)) μs ($(round(t_rand_linear * 1e9 / 1000, digits=2)) ns/query)")

improvement_linear = (1 - t_mono_linear / t_rand_linear) * 100
println("Monotonic improvement: $(round(improvement_linear, digits=1))%")

# ========================================
# Constant Interpolation Anchors
# ========================================

println("\n## Constant Anchor Benchmark")
println("-" ^ 60)

buffer_constant = Vector{FastInterpolations._ConstantAnchoredQuery{Float64}}(undef, 1000)

print("Monotonic queries: ")
t_mono_constant = @belapsed FastInterpolations._fill_anchors!($buffer_constant, $x, $xq_monotonic, Val(:constant))
println("$(round(t_mono_constant * 1e6, digits=2)) μs ($(round(t_mono_constant * 1e9 / 1000, digits=2)) ns/query)")

print("Random queries:    ")
t_rand_constant = @belapsed FastInterpolations._fill_anchors!($buffer_constant, $x, $xq_random, Val(:constant))
println("$(round(t_rand_constant * 1e6, digits=2)) μs ($(round(t_rand_constant * 1e9 / 1000, digits=2)) ns/query)")

improvement_constant = (1 - t_mono_constant / t_rand_constant) * 100
println("Monotonic improvement: $(round(improvement_constant, digits=1))%")

# ========================================
# Quadratic Interpolation Anchors
# ========================================

println("\n## Quadratic Anchor Benchmark")
println("-" ^ 60)

buffer_quadratic = Vector{FastInterpolations._QuadraticAnchoredQuery{Float64}}(undef, 1000)

print("Monotonic queries: ")
t_mono_quadratic = @belapsed FastInterpolations._fill_anchors!($buffer_quadratic, $x, $xq_monotonic, Val(:quadratic))
println("$(round(t_mono_quadratic * 1e6, digits=2)) μs ($(round(t_mono_quadratic * 1e9 / 1000, digits=2)) ns/query)")

print("Random queries:    ")
t_rand_quadratic = @belapsed FastInterpolations._fill_anchors!($buffer_quadratic, $x, $xq_random, Val(:quadratic))
println("$(round(t_rand_quadratic * 1e6, digits=2)) μs ($(round(t_rand_quadratic * 1e9 / 1000, digits=2)) ns/query)")

improvement_quadratic = (1 - t_mono_quadratic / t_rand_quadratic) * 100
println("Monotonic improvement: $(round(improvement_quadratic, digits=1))%")

# ========================================
# Cubic Interpolation Anchors
# ========================================

println("\n## Cubic Anchor Benchmark")
println("-" ^ 60)

buffer_cubic = Vector{FastInterpolations._CubicAnchoredQuery{Float64}}(undef, 1000)

print("Monotonic queries: ")
t_mono_cubic = @belapsed FastInterpolations._fill_anchors!($buffer_cubic, $x, $xq_monotonic)
println("$(round(t_mono_cubic * 1e6, digits=2)) μs ($(round(t_mono_cubic * 1e9 / 1000, digits=2)) ns/query)")

print("Random queries:    ")
t_rand_cubic = @belapsed FastInterpolations._fill_anchors!($buffer_cubic, $x, $xq_random)
println("$(round(t_rand_cubic * 1e6, digits=2)) μs ($(round(t_rand_cubic * 1e9 / 1000, digits=2)) ns/query)")

improvement_cubic = (1 - t_mono_cubic / t_rand_cubic) * 100
println("Monotonic improvement: $(round(improvement_cubic, digits=1))%")

# ========================================
# Summary
# ========================================

println("\n" * "=" ^ 60)
println("Summary: Monotonic Query Improvement")
println("=" ^ 60)
println("Linear:    $(round(improvement_linear, digits=1))%")
println("Constant:  $(round(improvement_constant, digits=1))%")
println("Quadratic: $(round(improvement_quadratic, digits=1))%")
println("Cubic:     $(round(improvement_cubic, digits=1))%")
println()

# Note on interpretation
println("Note: Positive values indicate monotonic queries are faster than random.")
println("      This demonstrates the benefit of LinearBoundedAlg{8} hint caching.")
println("      Negative or near-zero values indicate the search is dominated by")
println("      anchor weight computation rather than interval search.")
