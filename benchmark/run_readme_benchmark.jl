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
import Dierckx
using Plots
using DataFrames
using Statistics
using JSON

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Benchmark parameters for noise reduction
# BenchmarkTools stops when EITHER limit is reached (whichever comes first)
const BENCH_SAMPLES = 10_000   # max samples per benchmark

BenchmarkTools.DEFAULT_PARAMETERS.samples = BENCH_SAMPLES

# Fixed evals and seconds by query size (skip auto-tuning for more stable results)
# Higher evals/seconds = more stable measurements for fast operations
const EVALS_TINY = 200      # nq ≤ 20: ~1-5μs benchmarks (prevent long samples impacting minimum)
const EVALS_SMALL = 100     # nq ≤ 100: ~5-20μs benchmarks
const EVALS_MED = 10        # nq ≤ 2000: ~20-100μs benchmarks
const EVALS_LARGE = 1       # nq > 2000: ~100μs+ benchmarks (single call is long enough)

const SECS_TINY = 10.0      # nq ≤ 20: longer for stability
const SECS_SMALL = 10.0      # nq ≤ 100
const SECS_MED = 5.0        # nq ≤ 2000
const SECS_LARGE = 5.0      # nq > 2000: already stable

"""Get appropriate (evals, seconds) based on query size."""
function get_bench_params(nq::Int)
    nq ≤ 20 && return (EVALS_TINY, SECS_TINY)
    nq ≤ 100 && return (EVALS_SMALL, SECS_SMALL)
    nq ≤ 2000 && return (EVALS_MED, SECS_MED)
    return (EVALS_LARGE, SECS_LARGE)
end

# Query sizes for the plot (10^0 ~ 10^5)
const QUERY_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]

# Grid size (fixed)
const N_GRID = 100

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Functions
# ══════════════════════════════════════════════════════════════════════════════

"""Format time in appropriate units (ns/μs/ms)."""
function format_time(ns::Float64)
    if ns < 1000
        return "$(round(ns, digits=1)) ns"
    elseif ns < 1_000_000
        return "$(round(ns/1000, digits=2)) μs"
    else
        return "$(round(ns/1_000_000, digits=2)) ms"
    end
end

"""Format benchmark stats for verbose output."""
function format_bench_stats(b)
    # b.times contains per-eval times in ns
    t_med = median(b.times)
    
    total_time = sum(b.times)
    total_gc = sum(b.gctimes)
    gc_pct = total_time > 0 ? round(100 * total_gc / total_time, digits=1) : 0.0
    
    elapsed_s = total_time * b.params.evals / 1e9
    n_samples = length(b.times)
    n_evals = b.params.evals
    
    return "| med: $(format_time(t_med)) | gc: $(gc_pct)% | mem: $(b.memory)B | ($(round(elapsed_s, digits=1))s, $(n_samples) smp, $(n_evals) evl)"
end

function run_readme_benchmark(; verbose::Bool=true)
    x = range(0.0, 10.0, N_GRID)
    y = sin.(x) .+ 0.1 .* collect(x)

    ns_to_sec(ns) = ns / 1e9

    n_benchmarks = length(QUERY_SIZES) * 5  # 5 packages/configs per query size
    # Estimate total time based on variable seconds per query size
    est_time_sec = sum(nq -> get_bench_params(nq)[2] * 4, QUERY_SIZES)
    est_time_min = est_time_sec / 60

    if verbose
        println("Running one-shot benchmark (n_grid=$N_GRID)")
        println("  • $(length(QUERY_SIZES)) query sizes × 4 configs = $n_benchmarks benchmarks")
        println("  • Variable seconds: tiny=$(SECS_TINY)s, small=$(SECS_SMALL)s, med=$(SECS_MED)s, large=$(SECS_LARGE)s")
        println("  • Estimated total: ~$(round(est_time_min, digits=1)) min")
        println("  • GC.gc() before each benchmark for fair comparison")
        println()
    end

    rows = []
    bench_count = 0

    for nq in QUERY_SIZES
        xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
        out = Vector{Float64}(undef, nq)  # Pre-allocate output buffer
        evals, secs = get_bench_params(nq)

        # ── FastInterpolations (autocache=true) ──
        # One-shot API with cache primed
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] FastInterp(autocache=true)  n=$(lpad(nq, 6))... ")
        clear_cubic_cache!()
        cubic_interp!(out, x, y, xi)  # prime cache
        GC.gc()  # Clear GC state before benchmark
        bench = @benchmarkable begin 
            cubic_interp!($out, $x, $y, $xi; autocache=true)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_fast_cache = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # ── FastInterpolations (autocache=false) ──
        # One-shot API: cubic_interp! does construction + evaluation in one call
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] FastInterp(autocache=false) n=$(lpad(nq, 6))... ")
        clear_cubic_cache!()
        GC.gc()  # Clear GC state before benchmark
        bench = @benchmarkable begin
            cubic_interp!($out, $x, $y, $xi; autocache=false)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_fast_nocache = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # ── Interpolations.jl ──
        # One-shot: construct + broadcast into pre-allocated output
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] Interpolations.jl     n=$(lpad(nq, 6))... ")
        GC.gc()  # Clear GC state before benchmark
        bench = @benchmarkable begin
            itp = Interpolations.cubic_spline_interpolation($x, $y)
            @. $out = itp($xi)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_itp = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # ── DataInterpolations.jl ──
        # One-shot: construct + evaluate in-place
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] DataInterpolations    n=$(lpad(nq, 6))... ")
        GC.gc()  # Clear GC state before benchmark
        bench = @benchmarkable begin
            itp = DataInterpolations.CubicSpline($y, $x)
            itp($out, $xi)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_di = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # ── Dierckx.jl ──
        # One-shot: construct + evaluate (FITPACK wrapper)
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] Dierckx.jl            n=$(lpad(nq, 6))... ")
        x_vec = collect(x)  # Dierckx requires Vector
        GC.gc()  # Clear GC state before benchmark
        bench = @benchmarkable begin
            itp = Dierckx.Spline1D($x_vec, $y; k=3, s=0.0)
            @. $out = itp($xi)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_dierckx = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        push!(rows, (
            n=nq,
            FastInterp_nocache=t_fast_nocache,
            FastInterp_cached=t_fast_cache,
            Interpolations=t_itp,
            DataInterp=t_di,
            Dierckx=t_dierckx
        ))

        # Show speedup summary for this query size
        if verbose
            speedup_itp = t_itp / t_fast_cache
            speedup_di = t_di / t_fast_cache
            speedup_dierckx = t_dierckx / t_fast_cache
            speedup_nocache_itp = t_itp / t_fast_nocache
            speedup_nocache_di = t_di / t_fast_nocache
            speedup_nocache_dierckx = t_dierckx / t_fast_nocache
            println("       → FastInterp(autocache=true) speedup: $(round(speedup_itp, digits=1))× vs Interpolations, $(round(speedup_di, digits=1))× vs DataInterp, $(round(speedup_dierckx, digits=1))× vs Dierckx")
            println("       → FastInterp(autocache=false) speedup: $(round(speedup_nocache_itp, digits=1))× vs Interpolations, $(round(speedup_nocache_di, digits=1))× vs DataInterp, $(round(speedup_nocache_dierckx, digits=1))× vs Dierckx")
            println()
        end
    end

    return DataFrame(rows)
end

"""Print summary table with speedups."""
function print_summary_table(df)
    println("=" ^ 110)
    println("SPEEDUP SUMMARY (FastInterpolations.jl vs others)")
    println("=" ^ 110)
    println()

    # Header
    println("┌─────────┬────────────────────────────────┬────────────────────────────────┬────────────────────────────────┐")
    println("│  Query  │      vs Interpolations.jl      │    vs DataInterpolations.jl    │         vs Dierckx.jl          │")
    println("│    n    │  autocache=T   autocache=F    │  autocache=T   autocache=F    │  autocache=T   autocache=F    │")
    println("├─────────┼────────────────────────────────┼────────────────────────────────┼────────────────────────────────┤")

    for row in eachrow(df)
        speedup_itp_cache = row.Interpolations / row.FastInterp_cached
        speedup_itp_nocache = row.Interpolations / row.FastInterp_nocache
        speedup_di_cache = row.DataInterp / row.FastInterp_cached
        speedup_di_nocache = row.DataInterp / row.FastInterp_nocache
        speedup_dierckx_cache = row.Dierckx / row.FastInterp_cached
        speedup_dierckx_nocache = row.Dierckx / row.FastInterp_nocache

        n_str = lpad(row.n, 6)
        itp_cache_str = lpad("$(round(speedup_itp_cache, digits=1))×", 8)
        itp_nocache_str = lpad("$(round(speedup_itp_nocache, digits=1))×", 8)
        di_cache_str = lpad("$(round(speedup_di_cache, digits=1))×", 8)
        di_nocache_str = lpad("$(round(speedup_di_nocache, digits=1))×", 8)
        dierckx_cache_str = lpad("$(round(speedup_dierckx_cache, digits=1))×", 8)
        dierckx_nocache_str = lpad("$(round(speedup_dierckx_nocache, digits=1))×", 8)

        println("│ $n_str  │    $itp_cache_str       $itp_nocache_str      │    $di_cache_str       $di_nocache_str      │    $dierckx_cache_str       $dierckx_nocache_str      │")
    end

    println("└─────────┴────────────────────────────────┴────────────────────────────────┴────────────────────────────────┘")
    println()

    # Overall average speedup
    println("Average speedup (geometric mean):")
    geo_mean(x) = exp(mean(log.(x)))
    geo_itp_cache = geo_mean(df.Interpolations ./ df.FastInterp_cached)
    geo_itp_nocache = geo_mean(df.Interpolations ./ df.FastInterp_nocache)
    geo_di_cache = geo_mean(df.DataInterp ./ df.FastInterp_cached)
    geo_di_nocache = geo_mean(df.DataInterp ./ df.FastInterp_nocache)
    geo_dierckx_cache = geo_mean(df.Dierckx ./ df.FastInterp_cached)
    geo_dierckx_nocache = geo_mean(df.Dierckx ./ df.FastInterp_nocache)

    println("  vs Interpolations.jl:     $(round(geo_itp_cache, digits=1))× (autocache=true), $(round(geo_itp_nocache, digits=1))× (autocache=false)")
    println("  vs DataInterpolations.jl: $(round(geo_di_cache, digits=1))× (autocache=true), $(round(geo_di_nocache, digits=1))× (autocache=false)")
    println("  vs Dierckx.jl:            $(round(geo_dierckx_cache, digits=1))× (autocache=true), $(round(geo_dierckx_nocache, digits=1))× (autocache=false)")

    # Calculate ranges
    s_itp = df.Interpolations ./ df.FastInterp_cached
    s_di = df.DataInterp ./ df.FastInterp_cached
    s_dierckx = df.Dierckx ./ df.FastInterp_cached

    # Save summary for Release workflow
    summary = Dict(
        "itp_min" => minimum(s_itp),
        "itp_max" => maximum(s_itp),
        "di_min" => minimum(s_di),
        "di_max" => maximum(s_di),
        "dierckx_min" => minimum(s_dierckx),
        "dierckx_max" => maximum(s_dierckx)
    )
    open(joinpath(@__DIR__, "speedup_summary.json"), "w") do io
        JSON.print(io, summary)
    end
    println("Saved speedup summary to $(joinpath(@__DIR__, "speedup_summary.json"))")
    println()
end

# ══════════════════════════════════════════════════════════════════════════════
# Plot Function
# ══════════════════════════════════════════════════════════════════════════════

function save_readme_plot(df; save_path::String="docs/images/benchmark_oneshot_detail.png", dpi::Int=150)
    colors = [:orange, :green, :magenta, :blue]  # Added magenta for Dierckx

    # Calculate y-axis limits: min(1e-6, data_min) to auto
    all_times = vcat(df.Interpolations, df.DataInterp, df.Dierckx, df.FastInterp_cached, df.FastInterp_nocache)
    ymin = min(1e-6, minimum(all_times) * 0.5)

    p = plot(
        df.n, [df.Interpolations df.DataInterp df.Dierckx df.FastInterp_cached],
        label=["Interpolations.jl" "DataInterpolations.jl" "Dierckx.jl" "FastInterpolations.jl (autocache=true)"],
        xlabel="Query points",
        ylabel="Time (s)",
        title="One-Shot (Construction + Evaluation)",
        xscale=:log10,
        yscale=:log10,
        xlims=(0.8, 1.25e5),
        ylims=(ymin, :auto),
        xticks=10.0 .^ (0:5),
        yticks=10.0 .^ (-7:-2),
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
        size=(700, 450),  # Slightly wider to accommodate legend
        dpi=dpi
    )

    # Add uncached FastInterpolations as dotted line
    plot!(p, df.n, df.FastInterp_nocache,
        label="FastInterpolations.jl (autocache=false)",
        linestyle=:dot,
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

    # Print speedup summary table
    print_summary_table(df)

    # Save plot
    save_readme_plot(df)

    println("Done!")
end
