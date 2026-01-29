#!/usr/bin/env julia
"""
    update_readme_benchmark.jl

Local script to run benchmark and update README.md.
Replaces the GitHub Actions workflow for local development.

Usage:
    julia --project=benchmark scripts/update_readme_benchmark.jl [--dry-run] [--skip-benchmark]

Options:
    --dry-run        Show what would be changed without modifying files
    --skip-benchmark Skip benchmark execution (use existing speedup_summary.json)
"""

using Pkg
using JSON

const PROJECT_ROOT = dirname(@__DIR__)
const BENCHMARK_DIR = joinpath(PROJECT_ROOT, "benchmark")
const README_PATH = joinpath(PROJECT_ROOT, "README.md")
const SPEEDUP_JSON = joinpath(BENCHMARK_DIR, "speedup_summary.json")
const PLOT_PATH = joinpath(PROJECT_ROOT, "docs", "images", "benchmark_oneshot_detail.png")

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

"""Get package version from Pkg.dependencies()."""
function get_pkg_version(name::String)
    deps = Pkg.dependencies()
    for (_, info) in deps
        if info.name == name && info.version !== nothing
            return string(info.version)
        end
    end
    return "?"
end

"""Format speedup range as string."""
format_range(min_val, max_val) = "$(round(min_val, digits=1)) ~ $(round(max_val, digits=1))"

"""Get local environment description."""
function get_local_env_info()
    # OS info
    os_name = Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : Sys.iswindows() ? "Windows" : "Unknown"

    # Try to get more specific OS version
    os_detail = try
        if Sys.isapple()
            strip(read(`sw_vers -productVersion`, String))
        elseif Sys.islinux()
            if isfile("/etc/os-release")
                for line in eachline("/etc/os-release")
                    if startswith(line, "PRETTY_NAME=")
                        return replace(line[14:end], "\"" => "")
                    end
                end
            end
            "Linux"
        else
            os_name
        end
    catch
        os_name
    end

    # CPU info
    cpu_info = try
        if Sys.isapple()
            strip(read(`sysctl -n machdep.cpu.brand_string`, String))
        elseif Sys.islinux()
            for line in eachline("/proc/cpuinfo")
                if startswith(line, "model name")
                    return split(line, ":")[2] |> strip
                end
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

# ══════════════════════════════════════════════════════════════════════════════
# Main Logic
# ══════════════════════════════════════════════════════════════════════════════

function run_benchmark()
    println("=" ^ 60)
    println("Running README Benchmark")
    println("=" ^ 60)
    println()

    benchmark_script = joinpath(BENCHMARK_DIR, "run_readme_benchmark.jl")

    if !isfile(benchmark_script)
        error("Benchmark script not found: $benchmark_script")
    end

    # Run benchmark in subprocess to isolate environment
    cmd = `julia --project=$BENCHMARK_DIR $benchmark_script`
    println("Executing: $cmd")
    println()

    run(cmd)

    println()
    println("✅ Benchmark completed")
end

function update_readme(; dry_run::Bool=false)
    println()
    println("=" ^ 60)
    println("Updating README.md")
    println("=" ^ 60)
    println()

    # Check required files exist
    if !isfile(SPEEDUP_JSON)
        error("Speedup summary not found: $SPEEDUP_JSON\nRun benchmark first or remove --skip-benchmark flag")
    end

    if !isfile(PLOT_PATH)
        @warn "Plot not found at $PLOT_PATH - README will reference missing image"
    end

    # Read speedup summary
    summary = JSON.parsefile(SPEEDUP_JSON)

    # Get package versions
    itp_ver = get_pkg_version("Interpolations")
    di_ver = get_pkg_version("DataInterpolations")
    dierckx_ver = get_pkg_version("Dierckx")

    # Read FastInterpolations version from Project.toml
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

    # Format speedup ranges
    s_itp = format_range(summary["itp_min"], summary["itp_max"])
    s_di = format_range(summary["di_min"], summary["di_max"])
    s_dierckx = format_range(summary["dierckx_min"], summary["dierckx_max"])

    println("Environment: $env_info")
    println("Julia: $julia_ver")
    println("Versions:")
    println("  FastInterpolations: v$pkg_ver")
    println("  Interpolations: v$itp_ver")
    println("  DataInterpolations: v$di_ver")
    println("  Dierckx: v$dierckx_ver")
    println()
    println("Speedups:")
    println("  vs Interpolations.jl: $(s_itp)×")
    println("  vs DataInterpolations.jl: $(s_di)×")
    println("  vs Dierckx.jl: $(s_dierckx)×")
    println()

    # Build replacement blocks
    meta_content = """<!-- BENCHMARK_VERSIONS_START -->
> **Env:** Local · $env_info · Julia $julia_ver<br>
> **Pkg:** FastInterpolations (v$pkg_ver) · Interpolations (v$itp_ver) · DataInterpolations (v$di_ver) · Dierckx (v$dierckx_ver)
<!-- BENCHMARK_VERSIONS_END -->"""

    speedup_content = """<!-- BENCHMARK_SPEEDUP_START -->
**Speedup:** ($(s_itp))× vs `Interpolations.jl` · ($(s_di))× vs `DataInterpolations.jl` · ($(s_dierckx))× vs `Dierckx.jl`
<!-- BENCHMARK_SPEEDUP_END -->"""

    # Read current README
    readme = read(README_PATH, String)

    # Replace content between markers
    pattern_ver = r"<!-- BENCHMARK_VERSIONS_START -->.*?<!-- BENCHMARK_VERSIONS_END -->"s
    pattern_spd = r"<!-- BENCHMARK_SPEEDUP_START -->.*?<!-- BENCHMARK_SPEEDUP_END -->"s

    new_readme = replace(readme, pattern_ver => meta_content)
    new_readme = replace(new_readme, pattern_spd => speedup_content)

    if readme == new_readme
        println("ℹ️  No changes needed - README is already up to date")
        return
    end

    if dry_run
        println("🔍 DRY RUN - Would update README.md with:")
        println()
        println("  Versions block:")
        println("  ", replace(meta_content, "\n" => "\n  "))
        println()
        println("  Speedup block:")
        println("  ", replace(speedup_content, "\n" => "\n  "))
        println()
        println("Run without --dry-run to apply changes")
    else
        write(README_PATH, new_readme)
        println("✅ README.md updated")
        println()
        println("Updated files:")
        println("  • $README_PATH")
        println("  • $PLOT_PATH")
    end
end

function main()
    args = ARGS
    dry_run = "--dry-run" in args
    skip_benchmark = "--skip-benchmark" in args

    if "--help" in args || "-h" in args
        println(@doc update_readme_benchmark)
        return
    end

    println()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║          FastInterpolations.jl README Benchmark              ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()
    println("Options:")
    println("  dry_run: $dry_run")
    println("  skip_benchmark: $skip_benchmark")
    println()

    # Step 1: Run benchmark (unless skipped)
    if !skip_benchmark
        run_benchmark()
    else
        println("⏭️  Skipping benchmark (using existing speedup_summary.json)")
    end

    # Step 2: Update README
    update_readme(; dry_run=dry_run)

    println()
    println("=" ^ 60)
    println("Done!")
    println("=" ^ 60)
end

# Run main if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
