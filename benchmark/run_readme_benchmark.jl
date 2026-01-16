"""
    run_readme_benchmark.jl

Lightweight benchmark for CI release workflow.
Only generates the oneshot_detail plot for README.

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

# Faster benchmark settings for CI
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.3

function run_readme_benchmark(; verbose::Bool=true)
    # Query sizes: 10^0 to 10^5
    # query_sizes = [1, 10, 100, 1000, 10000, 100000]
    query_sizes = [1, 10, 1000, 100000]

    n_grid = 100
    x = range(0.0, 10.0, n_grid)
    y = sin.(x) .+ 0.1 .* collect(x)

    ns_to_sec(ns) = ns / 1e9

    verbose && println("Running one-shot benchmark (n_grid=$n_grid)...")

    rows = []
    for nq in query_sizes
        verbose && print("  n_query=$nq... ")

        xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))

        # FastInterpolations - autocache OFF
        b = @benchmark cubic_interp($x, $y, $xi; autocache=false)
        t_fast_nocache = ns_to_sec(median(b.times))

        # FastInterpolations - autocache ON (cache hit)
        clear_cubic_cache!()
        cubic_interp(x, y, xi)  # prime cache
        b = @benchmark cubic_interp($x, $y, $xi)
        t_fast_cache = ns_to_sec(median(b.times))

        # Interpolations.jl
        b = @benchmark Interpolations.cubic_spline_interpolation($x, $y)($xi)
        t_itp = ns_to_sec(median(b.times))

        # DataInterpolations.jl
        b = @benchmark DataInterpolations.CubicSpline($y, $x)($xi)
        t_di = ns_to_sec(median(b.times))

        push!(rows, (
            n=nq,
            FastInterp_nocache=t_fast_nocache,
            FastInterp_cached=t_fast_cache,
            Interpolations=t_itp,
            DataInterp=t_di
        ))
        verbose && println("done")
    end

    return DataFrame(rows)
end

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
        xlims=(0.5, 1.5e5),
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
        label="FastInterpolations.jl (no cache)",
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

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 50)
    println("CI Release Benchmark")
    println("=" ^ 50)

    df = run_readme_benchmark()
    save_readme_plot(df)

    println("\nDone!")
end
