#!/usr/bin/env julia
"""
    update_readme.jl

All-in-one script: Run benchmark, generate plot, and update README.md.

Usage:
    julia --project=benchmark benchmark/update_readme.jl [OPTIONS]

Options:
    --dry-run        Show what would be changed without modifying files
    --skip-benchmark Skip benchmark execution (use existing speedup_summary.json)
    --plot-only      Skip benchmark, regenerate plot and update README from existing results
"""

using Pkg
using JSON
using BenchmarkTools
using FastInterpolations
import Interpolations
import DataInterpolations
import Dierckx
using Plots
using DataFrames
using Statistics

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

const BENCHMARK_DIR = @__DIR__
const PROJECT_ROOT = dirname(BENCHMARK_DIR)
const README_PATH = joinpath(PROJECT_ROOT, "README.md")
const SPEEDUP_JSON = joinpath(BENCHMARK_DIR, "speedup_summary.json")
const RESULTS_JSON = joinpath(BENCHMARK_DIR, "benchmark_results.json")
const PLOT_PATH = joinpath(PROJECT_ROOT, "docs", "images", "benchmark_oneshot_detail.png")

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Configuration
# ══════════════════════════════════════════════════════════════════════════════

const BENCH_SAMPLES = 10_000
BenchmarkTools.DEFAULT_PARAMETERS.samples = BENCH_SAMPLES

const EVALS_TINY = 10       # nq ≤ 20
const EVALS_SMALL = 10      # nq ≤ 100
const EVALS_MED = 10        # nq ≤ 2000
const EVALS_LARGE = 1       # nq > 2000

const SECS_TINY = 3.0
const SECS_SMALL = 3.0
const SECS_MED = 3.0
const SECS_LARGE = 3.0

const QUERY_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
const N_GRID = 100

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Functions
# ══════════════════════════════════════════════════════════════════════════════

function get_bench_params(nq::Int)
    nq ≤ 20 && return (EVALS_TINY, SECS_TINY)
    nq ≤ 100 && return (EVALS_SMALL, SECS_SMALL)
    nq ≤ 2000 && return (EVALS_MED, SECS_MED)
    return (EVALS_LARGE, SECS_LARGE)
end

function format_time(ns::Float64)
    ns < 1000 && return "$(round(ns, digits=1)) ns"
    ns < 1_000_000 && return "$(round(ns/1000, digits=2)) μs"
    return "$(round(ns/1_000_000, digits=2)) ms"
end

function format_bench_stats(b)
    t_med = median(b.times)
    total_time = sum(b.times)
    total_gc = sum(b.gctimes)
    gc_pct = total_time > 0 ? round(100 * total_gc / total_time, digits=1) : 0.0
    elapsed_s = total_time * b.params.evals / 1e9
    n_samples = length(b.times)
    n_evals = b.params.evals
    return "| med: $(format_time(t_med)) | gc: $(gc_pct)% | mem: $(b.memory)B | ($(round(elapsed_s, digits=1))s, $(n_samples) smp, $(n_evals) evl)"
end

function run_benchmark(; verbose::Bool=true)
    x = range(0.0, 10.0, N_GRID)
    y = sin.(x) .+ 0.1 .* collect(x)
    ns_to_sec(ns) = ns / 1e9

    n_benchmarks = length(QUERY_SIZES) * 5
    est_time_sec = sum(nq -> get_bench_params(nq)[2] * 4, QUERY_SIZES)
    est_time_min = est_time_sec / 60

    if verbose
        println("Running one-shot benchmark (n_grid=$N_GRID)")
        println("  • $(length(QUERY_SIZES)) query sizes × 5 configs = $n_benchmarks benchmarks")
        println("  • Variable seconds: tiny=$(SECS_TINY)s, small=$(SECS_SMALL)s, med=$(SECS_MED)s, large=$(SECS_LARGE)s")
        println("  • Estimated total: ~$(round(est_time_min, digits=1)) min")
        println("  • GC.gc() before each benchmark for fair comparison")
        println()
    end

    rows = []
    bench_count = 0

    for nq in QUERY_SIZES
        xi = nq == 1 ? [5.0] : collect(range(0.1, 9.9, nq))
        out = Vector{Float64}(undef, nq)
        evals, secs = get_bench_params(nq)

        # FastInterpolations (autocache=true)
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] FastInterp(autocache=true)  n=$(lpad(nq, 6))... ")
        clear_cubic_cache!()
        cubic_interp!(out, x, y, xi)
        GC.gc()
        bench = @benchmarkable cubic_interp!($out, $x, $y, $xi; autocache=true)
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_fast_cache = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # FastInterpolations (autocache=false)
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] FastInterp(autocache=false) n=$(lpad(nq, 6))... ")
        clear_cubic_cache!()
        GC.gc()
        bench = @benchmarkable cubic_interp!($out, $x, $y, $xi; autocache=false)
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_fast_nocache = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # Interpolations.jl
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] Interpolations.jl     n=$(lpad(nq, 6))... ")
        GC.gc()
        bench = @benchmarkable begin
            itp = Interpolations.cubic_spline_interpolation($x, $y)
            @. $out = itp($xi)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_itp = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # DataInterpolations.jl
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] DataInterpolations    n=$(lpad(nq, 6))... ")
        GC.gc()
        bench = @benchmarkable begin
            itp = DataInterpolations.CubicSpline($y, $x)
            itp($out, $xi)
        end
        bench.params.evals = evals
        bench.params.seconds = secs
        b = run(bench)
        t_di = ns_to_sec(minimum(b.times))
        verbose && println("$(lpad(format_time(minimum(b.times)), 10)) $(format_bench_stats(b))")

        # Dierckx.jl
        bench_count += 1
        verbose && print("  [$bench_count/$n_benchmarks] Dierckx.jl            n=$(lpad(nq, 6))... ")
        x_vec = collect(x)
        GC.gc()
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

function save_results(df)
    results = Dict(
        "n" => df.n,
        "FastInterp_nocache" => df.FastInterp_nocache,
        "FastInterp_cached" => df.FastInterp_cached,
        "Interpolations" => df.Interpolations,
        "DataInterp" => df.DataInterp,
        "Dierckx" => df.Dierckx
    )
    open(RESULTS_JSON, "w") do io
        JSON.print(io, results, 2)
    end
    println("Saved: $RESULTS_JSON")
end

function load_results()
    if !isfile(RESULTS_JSON)
        error("Benchmark results not found: $RESULTS_JSON\nRun benchmark first without --plot-only")
    end
    data = JSON.parsefile(RESULTS_JSON)
    return DataFrame(
        n = data["n"],
        FastInterp_nocache = data["FastInterp_nocache"],
        FastInterp_cached = data["FastInterp_cached"],
        Interpolations = data["Interpolations"],
        DataInterp = data["DataInterp"],
        Dierckx = data["Dierckx"]
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# Summary & Plot
# ══════════════════════════════════════════════════════════════════════════════

function print_summary_table(df)
    println("=" ^ 110)
    println("SPEEDUP SUMMARY (FastInterpolations.jl vs others)")
    println("=" ^ 110)
    println()

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

    # Calculate ranges and save summary
    s_itp = df.Interpolations ./ df.FastInterp_cached
    s_di = df.DataInterp ./ df.FastInterp_cached
    s_dierckx = df.Dierckx ./ df.FastInterp_cached

    summary = Dict(
        "itp_min" => minimum(s_itp),
        "itp_max" => maximum(s_itp),
        "di_min" => minimum(s_di),
        "di_max" => maximum(s_di),
        "dierckx_min" => minimum(s_dierckx),
        "dierckx_max" => maximum(s_dierckx)
    )
    open(SPEEDUP_JSON, "w") do io
        JSON.print(io, summary)
    end
    println("\nSaved: $SPEEDUP_JSON")
end

function save_plot(df; dpi::Int=150)
    colors = [:orange, :green, :magenta, :blue]

    all_times = vcat(df.Interpolations, df.DataInterp, df.Dierckx, df.FastInterp_cached, df.FastInterp_nocache)
    ymin = min(1e-6, minimum(all_times) * 0.5)

    p = plot(
        df.n, [df.Interpolations df.DataInterp df.Dierckx df.FastInterp_cached],
        label=["Interpolations.jl" "DataInterpolations.jl" "Dierckx.jl" "FastInterpolations.jl (cache-hit)"],
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
        size=(600, 450),
        alpha=0.8,
        dpi=dpi
    )

    plot!(p, df.n, df.FastInterp_nocache,
        label="FastInterpolations.jl (cache-miss)",
        linestyle=:dot,
        linewidth=2,
        color=:blue,
        marker=:none,
        alpha=0.8
    )

    mkpath(dirname(PLOT_PATH))
    savefig(p, PLOT_PATH)
    println("Saved: $PLOT_PATH")
    return p
end

# ══════════════════════════════════════════════════════════════════════════════
# README Update
# ══════════════════════════════════════════════════════════════════════════════

function get_pkg_version(name::String)
    deps = Pkg.dependencies()
    for (_, info) in deps
        if info.name == name && info.version !== nothing
            return string(info.version)
        end
    end
    return "?"
end

format_range(min_val, max_val) = "$(round(min_val, digits=1)) ~ $(round(max_val, digits=1))"

function get_local_env_info()
    os_name = Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : Sys.iswindows() ? "Windows" : "Unknown"

    os_detail = try
        if Sys.isapple()
            strip(read(`sw_vers -productVersion`, String))
        elseif Sys.islinux()
            if isfile("/etc/os-release")
                for line in eachline("/etc/os-release")
                    startswith(line, "PRETTY_NAME=") && return replace(line[14:end], "\"" => "")
                end
            end
            "Linux"
        else
            os_name
        end
    catch
        os_name
    end

    cpu_info = try
        if Sys.isapple()
            strip(read(`sysctl -n machdep.cpu.brand_string`, String))
        elseif Sys.islinux()
            for line in eachline("/proc/cpuinfo")
                startswith(line, "model name") && return split(line, ":")[2] |> strip
            end
            "Unknown CPU"
        else
            "Unknown CPU"
        end
    catch
        "Unknown CPU"
    end

    return "$os_name $os_detail · $cpu_info"
end

function update_readme(; dry_run::Bool=false)
    println()
    println("=" ^ 60)
    println("Updating README.md")
    println("=" ^ 60)
    println()

    if !isfile(SPEEDUP_JSON)
        error("Speedup summary not found: $SPEEDUP_JSON")
    end

    summary = JSON.parsefile(SPEEDUP_JSON)

    itp_ver = get_pkg_version("Interpolations")
    di_ver = get_pkg_version("DataInterpolations")
    dierckx_ver = get_pkg_version("Dierckx")

    project_toml = joinpath(PROJECT_ROOT, "Project.toml")
    pkg_ver = "?"
    for line in eachline(project_toml)
        m = match(r"^version\s*=\s*\"([^\"]+)\"", line)
        if m !== nothing
            pkg_ver = m.captures[1]
            break
        end
    end

    julia_ver = "$(VERSION.major).$(VERSION.minor).$(VERSION.patch)"
    env_info = get_local_env_info()

    s_itp = format_range(summary["itp_min"], summary["itp_max"])
    s_di = format_range(summary["di_min"], summary["di_max"])
    s_dierckx = format_range(summary["dierckx_min"], summary["dierckx_max"])

    println("Environment: $env_info")
    println("Julia: $julia_ver")
    println("Versions: FastInterpolations v$pkg_ver, Interpolations v$itp_ver, DataInterpolations v$di_ver, Dierckx v$dierckx_ver")
    println("Speedups: $(s_itp)× (vs Itp), $(s_di)× (vs DataItp), $(s_dierckx)× (vs Dierckx)")
    println()

    meta_content = """<!-- BENCHMARK_VERSIONS_START -->
> **Env:** Local · $env_info · Julia $julia_ver<br>
> **Pkg:** FastInterpolations (v$pkg_ver) · Interpolations (v$itp_ver) · DataInterpolations (v$di_ver) · Dierckx (v$dierckx_ver)
<!-- BENCHMARK_VERSIONS_END -->"""

    speedup_content = """<!-- BENCHMARK_SPEEDUP_START -->
**Speedup:** ($(s_itp))× vs `Interpolations.jl` · ($(s_di))× vs `DataInterpolations.jl` · ($(s_dierckx))× vs `Dierckx.jl`
<!-- BENCHMARK_SPEEDUP_END -->"""

    readme = read(README_PATH, String)

    pattern_ver = r"<!-- BENCHMARK_VERSIONS_START -->.*?<!-- BENCHMARK_VERSIONS_END -->"s
    pattern_spd = r"<!-- BENCHMARK_SPEEDUP_START -->.*?<!-- BENCHMARK_SPEEDUP_END -->"s

    new_readme = replace(readme, pattern_ver => meta_content)
    new_readme = replace(new_readme, pattern_spd => speedup_content)

    if readme == new_readme
        println("ℹ️  No changes needed - README is already up to date")
        return
    end

    if dry_run
        println("🔍 DRY RUN - Would update README.md")
        println("Run without --dry-run to apply changes")
    else
        write(README_PATH, new_readme)
        println("✅ README.md updated")
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

function main()
    args = ARGS
    dry_run = "--dry-run" in args
    skip_benchmark = "--skip-benchmark" in args
    plot_only = "--plot-only" in args

    if "--help" in args || "-h" in args
        println(@doc update_readme_benchmark)
        return
    end

    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║     FastInterpolations.jl - README Benchmark Updater         ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
    println("Options: dry_run=$dry_run, skip_benchmark=$skip_benchmark, plot_only=$plot_only")
    println()

    if plot_only
        println("=" ^ 60)
        println("Regenerating Plot from Existing Results")
        println("=" ^ 60)
        println()

        df = load_results()
        println("✅ Loaded $(nrow(df)) benchmark results from $RESULTS_JSON")
        print_summary_table(df)
        save_plot(df)
    elseif !skip_benchmark
        println("=" ^ 60)
        println("Running Benchmark")
        println("=" ^ 60)
        println()

        df = run_benchmark()
        save_results(df)
        print_summary_table(df)
        save_plot(df)
    else
        println("⏭️  Skipping benchmark (using existing speedup_summary.json)")
    end

    update_readme(; dry_run=dry_run)

    println()
    println("=" ^ 60)
    println("Done! Updated files:")
    println("  • $README_PATH")
    println("  • $PLOT_PATH")
    println("  • $SPEEDUP_JSON")
    println("=" ^ 60)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
