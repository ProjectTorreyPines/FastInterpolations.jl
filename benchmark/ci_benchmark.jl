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

# Benchmark time budget (slower benchmarks will use more time)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0

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
    suite["1_cubic_oneshot"]["q$label"] = @benchmarkable cubic_interp($x, $y, $xi)
end

# 2. Cubic Construction (varying grid size)
for ng in GRID_SIZES
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)
    clear_cubic_cache!()
    label = lpad(ng, 4, '0')  # 0010, 0100, 1000
    suite["2_cubic_construct"]["g$label"] = @benchmarkable cubic_interp($x_grid, $y_grid; autocache=false)
end

# 3. Cubic Evaluation (reuse interpolant)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    suite["3_cubic_eval"]["q$label"] = @benchmarkable $itp_cubic($xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Linear Benchmarks (shown second)
# ══════════════════════════════════════════════════════════════════════════════

println("Setting up linear benchmarks...")

# 4. Linear One-Shot (construct + evaluate)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    suite["4_linear_oneshot"]["q$label"] = @benchmarkable linear_interp($x, $y, $xi)
end

# 5. Linear Construction (varying grid size)
for ng in GRID_SIZES
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)
    label = lpad(ng, 4, '0')
    suite["5_linear_construct"]["g$label"] = @benchmarkable linear_interp($x_grid, $y_grid)
end

# 6. Linear Evaluation (reuse interpolant)
for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
    label = lpad(nq, 5, '0')
    suite["6_linear_eval"]["q$label"] = @benchmarkable $itp_linear($xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Run and Save
# ══════════════════════════════════════════════════════════════════════════════

println("\nTuning benchmarks...")
tune!(suite)

println("Running benchmarks...")
results = run(suite, verbose=true)

println("\nSaving results to output.json...")

# Build sorted JSON output directly (github-action-benchmark 'julia' format)
using JSON
med_results = median(results)

# Collect all benchmarks as (name, data) pairs
entries = Tuple{String, Dict{String,Any}}[]
for group_name in keys(med_results)
    group = med_results[group_name]
    for bench_name in keys(group)
        trial = group[bench_name]
        full_name = "$group_name/$bench_name"
        data = Dict{String,Any}(
            "time" => trial.time,
            "gctime" => trial.gctime,
            "memory" => trial.memory,
            "allocs" => trial.allocs
        )
        push!(entries, (full_name, data))
    end
end

# Sort alphabetically by benchmark name
sort!(entries, by = first)

# Write in BenchmarkTools-compatible format
open("output.json", "w") do io
    JSON.print(io, [[name, data] for (name, data) in entries])
end
println("Saved $(length(entries)) benchmarks (sorted)")

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
