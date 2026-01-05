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
# One-Shot Benchmarks (construct + evaluate)
# ══════════════════════════════════════════════════════════════════════════════
# Typical user workflow: build interpolant and evaluate in one call

println("Setting up one-shot benchmarks...")

for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

    suite["oneshot"]["linear_q$nq"] = @benchmarkable linear_interp($x, $y, $xi)

    # Prime cache, then benchmark cache-hit performance
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime
    suite["oneshot"]["cubic_q$nq"] = @benchmarkable cubic_interp($x, $y, $xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Construction Benchmarks (varying grid size)
# ══════════════════════════════════════════════════════════════════════════════
# Track how construction scales with grid size

println("Setting up construction benchmarks...")

for ng in GRID_SIZES
    x_grid = range(0.0, 10.0, ng)
    y_grid = sin.(x_grid) .+ 0.1 .* collect(x_grid)

    suite["construct"]["linear_g$ng"] = @benchmarkable linear_interp($x_grid, $y_grid)

    clear_cubic_cache!()
    suite["construct"]["cubic_g$ng"] = @benchmarkable cubic_interp($x_grid, $y_grid; autocache=false)
end

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Benchmarks (reuse interpolant)
# ══════════════════════════════════════════════════════════════════════════════
# Performance when interpolant is reused across many evaluations

println("Setting up evaluation benchmarks...")

for nq in QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

    suite["eval"]["linear_q$nq"] = @benchmarkable $itp_linear($xi)
    suite["eval"]["cubic_q$nq"] = @benchmarkable $itp_cubic($xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Run and Save
# ══════════════════════════════════════════════════════════════════════════════

println("\nTuning benchmarks...")
tune!(suite)

println("Running benchmarks...")
results = run(suite, verbose=true)

println("\nSaving results to output.json...")
BenchmarkTools.save("output.json", median(results))

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
