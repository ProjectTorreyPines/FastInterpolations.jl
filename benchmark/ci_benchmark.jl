"""
    ci_benchmark.jl

Benchmark script for GitHub Actions CI.
Outputs JSON compatible with github-action-benchmark.

Usage:
    julia --project benchmark/ci_benchmark.jl
"""

using BenchmarkTools
using FastInterpolations
import Interpolations
import DataInterpolations

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

const N_GRID = 100
const QUERY_SIZES = [10, 100, 1000]
const COMPARISON_QUERY_SIZES = [1, 10, 100, 1000]

# Reduce benchmark time for CI (default is 5s per benchmark)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

# ══════════════════════════════════════════════════════════════════════════════
# Setup Data
# ══════════════════════════════════════════════════════════════════════════════

x = range(0.0, 10.0, N_GRID)
y = sin.(x) .+ 0.1 .* collect(x)

# Pre-build interpolants for evaluation benchmarks
clear_cubic_cache!()
const itp_linear = linear_interp(x, y)
const itp_cubic = cubic_interp(x, y; autocache=false)

suite = BenchmarkGroup()

# ══════════════════════════════════════════════════════════════════════════════
# One-Shot Benchmarks (construct + evaluate)
# ══════════════════════════════════════════════════════════════════════════════
# Typical user workflow: interpolate once per dataset

println("Setting up one-shot benchmarks...")

for nq in QUERY_SIZES
    xi = collect(range(0.1, 9.9, nq))

    suite["oneshot"]["linear_$nq"] = @benchmarkable linear_interp($x, $y, $xi)

    # Prime cache, then benchmark cache-hit performance
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime
    suite["oneshot"]["cubic_$nq"] = @benchmarkable cubic_interp($x, $y, $xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Benchmarks (reuse interpolant)
# ══════════════════════════════════════════════════════════════════════════════
# Performance when interpolant is reused across many evaluations

println("Setting up evaluation benchmarks...")

for nq in QUERY_SIZES
    xi = collect(range(0.1, 9.9, nq))

    suite["eval"]["linear_$nq"] = @benchmarkable $itp_linear($xi)
    suite["eval"]["cubic_$nq"] = @benchmarkable $itp_cubic($xi)
end

# ══════════════════════════════════════════════════════════════════════════════
# Construction Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
# Track construction overhead separately

println("Setting up construction benchmarks...")

suite["construct"]["linear"] = @benchmarkable linear_interp($x, $y)

clear_cubic_cache!()
suite["construct"]["cubic"] = @benchmarkable cubic_interp($x, $y; autocache=false)

# ══════════════════════════════════════════════════════════════════════════════
# Package Comparison (cubic one-shot)
# ══════════════════════════════════════════════════════════════════════════════
# Compare against Interpolations.jl and DataInterpolations.jl

println("Setting up comparison benchmarks...")

for nq in COMPARISON_QUERY_SIZES
    xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

    # FastInterpolations (cache-hit)
    clear_cubic_cache!()
    cubic_interp(x, y, xi)  # prime cache
    suite["compare"]["FastInterp_$nq"] = @benchmarkable cubic_interp($x, $y, $xi)

    # Interpolations.jl
    suite["compare"]["Interpolations_$nq"] = @benchmarkable begin
        itp = Interpolations.cubic_spline_interpolation($x, $y)
        itp($xi)
    end

    # DataInterpolations.jl
    suite["compare"]["DataInterp_$nq"] = @benchmarkable begin
        itp = DataInterpolations.CubicSpline($y, $x)
        itp($xi)
    end
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

println("Done!")
