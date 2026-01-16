"""
    run_readme_benchmark.jl

Benchmark for CI release workflow.
Generates the oneshot_detail plot for README with high accuracy.

Usage:
```julia
julia --project=benchmark benchmark/run_readme_benchmark.jl
```
"""

using BenchmarkTools
using FastInterpolations
import Interpolations
import DataInterpolations
using Plots
using DataFrames

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Benchmark parameters for noise reduction
# BenchmarkTools stops when EITHER limit is reached (whichever comes first)
const BENCH_SECONDS = 5.0      # seconds per benchmark point
const BENCH_SAMPLES = 10_000   # max samples per benchmark

BenchmarkTools.DEFAULT_PARAMETERS.seconds = BENCH_SECONDS
BenchmarkTools.DEFAULT_PARAMETERS.samples = BENCH_SAMPLES

# Query sizes for the plot (10^0 ~ 10^5)
const QUERY_SIZES = [1,2,5,10,20,50,100,200,500,1000,2000,5000, 10_000, 20_000, 50_000, 100_000]

# Grid size (fixed)
const N_GRID = 100

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Functions
# ══════════════════════════════════════════════════════════════════════════════

function run_readme_benchmark(; verbose::Bool=true)
    x = range(0.0, 10.0, N_GRID)
    y = sin.(x) .+ 0.1 .* collect(x)

    ns_to_sec(ns) = ns / 1e9

    n_benchmarks = length(QUERY_SIZES) * 4  # 4 packages/configs per query size
    est_time_min = n_benchmarks * BENCH_SECONDS / 60

    if verbose
        println("Running one-shot benchmark (n_grid=$N_GRID)")
        println("  • $(length(QUERY_SIZES)) query sizes × 4 configs = $n_benchmarks benchmarks")
        println("  • $(BENCH_SECONDS)s per benchmark → ~$(round(est_time_min, digits=1)) min total")
        println("  • Using in-place APIs for zero-allocation evaluation")
        println()
    end

    rows = []
    bench_count = 0

    for nq in QUERY_SIZES
        xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
        out = Vector{Float64}(undef, nq)  # Pre-allocate output buffer

        # ── FastInterpolations (no cache) ──
        # One-shot: construct + evaluate in-place
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] FastInterp(cache-miss) n=$nq... ")
        clear_cubic_cache!()
        b = @benchmark begin
            itp = cubic_interp($x, $y; autocache=false)
            itp($out, $xi)
        end
        t_fast_nocache = ns_to_sec(median(b.times))
        verbose && println("$(round(t_fast_nocache*1e6, digits=2)) μs")

        # ── FastInterpolations (cache hit) ──
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] FastInterp(cache)    n=$nq... ")
        clear_cubic_cache!()
        cubic_interp(x, y)  # prime cache
        b = @benchmark begin
            itp = cubic_interp($x, $y)
            itp($out, $xi)
        end
        t_fast_cache = ns_to_sec(median(b.times))
        verbose && println("$(round(t_fast_cache*1e6, digits=2)) μs")

        # ── Interpolations.jl ──
        # One-shot: construct + broadcast into pre-allocated output
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] Interpolations.jl   n=$nq... ")
        b = @benchmark begin
            itp = Interpolations.cubic_spline_interpolation($x, $y)
            @. $out = itp($xi)
        end
        t_itp = ns_to_sec(median(b.times))
        verbose && println("$(round(t_itp*1e6, digits=2)) μs")

        # ── DataInterpolations.jl ──
        # One-shot: construct + evaluate in-place
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] DataInterpolations  n=$nq... ")
        b = @benchmark begin
            itp = DataInterpolations.CubicSpline($y, $x)
            itp($out, $xi)
        end
        t_di = ns_to_sec(median(b.times))
        verbose && println("$(round(t_di*1e6, digits=2)) μs")

        push!(rows, (
            n=nq,
            FastInterp_nocache=t_fast_nocache,
            FastInterp_cached=t_fast_cache,
            Interpolations=t_itp,
            DataInterp=t_di
        ))

        verbose && println()
    end

    return DataFrame(rows)
end

# ══════════════════════════════════════════════════════════════════════════════
# Plot Function
# ══════════════════════════════════════════════════════════════════════════════

function save_readme_plot(df; save_path::String="docs/images/benchmark_oneshot_detail.png", dpi::Int=250)
    colors = [:orange, :green, :blue]

    # Calculate y-axis limits: min(1e-6, data_min) to auto
    all_times = vcat(df.Interpolations, df.DataInterp, df.FastInterp_cached, df.FastInterp_nocache)
    ymin = min(1e-6, minimum(all_times) * 0.5)

    p = plot(
        df.n, [df.Interpolations df.DataInterp df.FastInterp_cached],
        label=["Interpolations.jl" "DataInterpolations.jl" "FastInterpolations.jl (cache-hit)"],
        xlabel="Query points",
        ylabel="Time (s)",
        title="One-Shot (Construction + Evaluation)",
        xscale=:log10,
        yscale=:log10,
        xlims=(0.8, 1.25e5),
        ylims=(ymin, :auto),
        xticks=10.0 .^ (0:5),
        marker=:circle,
        markersize=6,
        linewidth=2,
        color=permutedims(colors),
        legend=:topleft,
        grid=true,
        minorgrid=true,
        tickfontsize=12,
        guidefontsize=14,
        titlefontsize=16,
        legendfontsize=10,
        size=(600, 450),
        dpi=dpi
    )

    # Add uncached FastInterpolations as dashed line
    plot!(p, df.n, df.FastInterp_nocache,
        label="FastInterpolations.jl (cache-miss)",
        linestyle=:dash,
        linewidth=2,
        color=:blue,
        marker=:none
    )

    mkpath(dirname(save_path))
    savefig(p, save_path)
    println("Saved: $save_path")

    return p
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 60)
    println("README Benchmark (High Accuracy)")
    println("=" ^ 60)
    println()

    df = run_readme_benchmark()

    println("=" ^ 60)
    println("Results Summary")
    println("=" ^ 60)
    println(df)
    println()

    save_readme_plot(df)

    println("\nDone!")
end
