"""
    ci_benchmark.jl

Benchmark script for GitHub Actions CI.
Outputs JSON compatible with github-action-benchmark.

Usage:
    julia --project=benchmark benchmark/ci_benchmark.jl
"""

using BenchmarkTools
using FastInterpolations

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Representative sizes for benchmarking
const QUERY_SIZES = [1, 100, 10_000]      # queries: single, medium, large batch
const GRID_SIZES = [10, 100, 1000]        # grids: small, medium, large

# Benchmark parameters for noise reduction in CI
# BenchmarkTools stops when EITHER limit is reached (whichever comes first)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100_000

# Fixed evals by speed category (skip tuning for faster CI)
# Higher evals = more stable ns-level measurements
const EVALS_FAST = 10_000   # ~10-50ns benchmarks
const EVALS_MED  = 1_000    # ~500ns-2μs benchmarks
const EVALS_SLOW = 100       # ~30-100μs benchmarks

# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

suite = BenchmarkGroup()

# Use medium grid for oneshot and eval benchmarks
const N_GRID = 100
x = range(0.0, 10.0, N_GRID)
y = sin.(x) .+ 0.1 .* collect(x)

# Pre-build interpolants for evaluation benchmarks
clear_cubic_cache!()
const itp_linear = linear_interp(x, y)
const itp_cubic = cubic_interp(x, y; autocache=false)

# ══════════════════════════════════════════════════════════════════════════════
# Cubic Benchmarks (shown first)
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up cubic benchmarks...")

# 1. Cubic One-Shot (construct + evaluate)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime cache
    label = lpad(nq, 5, '0')  # 00001, 00100, 10000
    b = @benchmarkable cubic_interp($x, $y, $xi)
    b.params.evals = nq >= 10_000 ? EVALS_SLOW : EVALS_MED
    suite["1_cubic_oneshot"]["q$label"] = b
end

# 2. Cubic Construction (varying grid size)
for ng in GRID_SIZES
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)
    clear_cubic_cache!()
    label = lpad(ng, 4, '0')  # 0010, 0100, 1000
    b = @benchmarkable cubic_interp($x_grid, $y_grid; autocache=false)
    b.params.evals = ng >= 1000 ? EVALS_SLOW : EVALS_MED
    suite["2_cubic_construct"]["g$label"] = b
end

# 3. Cubic Evaluation (reuse interpolant)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    b = @benchmarkable $itp_cubic($xi)
    b.params.evals = nq == 1 ? EVALS_FAST : nq == 100 ? EVALS_MED : EVALS_SLOW
    suite["3_cubic_eval"]["q$label"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# Linear Benchmarks (shown second)
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up linear benchmarks...")

# 4. Linear One-Shot (construct + evaluate)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    b = @benchmarkable linear_interp($x, $y, $xi)
    b.params.evals = nq == 1 ? EVALS_FAST : nq == 100 ? EVALS_MED : EVALS_SLOW
    suite["4_linear_oneshot"]["q$label"] = b
end

# 5. Linear Construction (varying grid size) - nearly instant, use high evals
for ng in GRID_SIZES
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)
    label = lpad(ng, 4, '0')
    b = @benchmarkable linear_interp($x_grid, $y_grid)
    b.params.evals = EVALS_FAST
    suite["5_linear_construct"]["g$label"] = b
end

# 6. Linear Evaluation (reuse interpolant)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    b = @benchmarkable $itp_linear($xi)
    b.params.evals = nq == 1 ? EVALS_FAST : nq == 100 ? EVALS_MED : EVALS_SLOW
    suite["6_linear_eval"]["q$label"] = b
end

# ══════════════════════════════════════════════════════════════════════════════
# Run and Save
# ══════════════════════════════════════════════════════════════════════════════

# Skip tuning - we set evals manually for consistent CI results
println("\nRunning benchmarks (evals preset, no tuning)...")
results = run(suite, verbose=true)

println("\nSaving results to output.json...")
BenchmarkTools.save("output.json", median(results))

# Sort JSON keys recursively for consistent dashboard ordering
println("Sorting JSON keys for dashboard display...")
using JSON
using OrderedCollections

function sort_keys_recursive(obj)
    if obj isa AbstractDict
        sorted = OrderedDict{String,Any}()
        for k in sort(collect(keys(obj)); by=string)
            sorted[string(k)] = sort_keys_recursive(obj[k])
        end
        return sorted
    elseif obj isa AbstractVector
        return [sort_keys_recursive(item) for item in obj]
    else
        return obj
    end
end

json_data = JSON.parsefile("output.json")
sorted_data = sort_keys_recursive(json_data)
open("output.json", "w") do io
    JSON.print(io, sorted_data)
end
println("Saved $(length(collect(BenchmarkTools.leaves(median(results))))) benchmarks (sorted)")

# ══════════════════════════════════════════════════════════════════════════════
# Print Summary
# ══════════════════════════════════════════════════════════════════════════════

function format_time(ns::Float64)
    if ns < 1000
        return "$(round(ns, digits=1)) ns"
    elseif ns < 1_000_000
        return "$(round(ns/1000, digits=2)) μs"
    else
        return "$(round(ns/1_000_000, digits=2)) ms"
    end
end

println("\n" * "="^70)
println("BENCHMARK SUMMARY (median times)")
println("="^70)

med_results = median(results)

for group_name in sort(collect(keys(med_results)))
    group = med_results[group_name]
    println("\n[$group_name]")
    for bench_name in sort(collect(keys(group)))
        trial = group[bench_name]
        println("  $(rpad(bench_name, 20)) $(format_time(trial.time))")
    end
end

println("\nDone! (18 benchmarks total)")
