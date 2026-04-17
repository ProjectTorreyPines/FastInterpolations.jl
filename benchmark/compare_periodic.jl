#!/usr/bin/env julia
#
# Compare two JSON outputs from `periodic_oneshot_benchmark.jl`.
# Usage:
#   julia --project=benchmark benchmark/compare_periodic.jl master.json branch.json
#
# Joins entries by `(dim, method, n, grid_type, bc, query)` tuple keys,
# computes speedup ratio = master / branch, prints a table sorted by ratio.

using JSON
using Printf

function load(path)
    data = JSON.parsefile(path)
    dict = Dict{NTuple{6, String}, Float64}()
    for e in data["results"]
        key = (
            string(e["dim"]), e["method"], string(e["n"]),
            e["grid_type"], e["bc"], e["query"],
        )
        # Skip entries that errored ("skipped" string) or are non-numeric
        e["ns_per_call"] isa Number || continue
        dict[key] = e["ns_per_call"]
    end
    return data["julia_version"], dict
end

function main()
    length(ARGS) == 2 ||
        (println(stderr, "Usage: compare_periodic.jl <master.json> <branch.json>"); exit(1))
    master_file, branch_file = ARGS

    master_v, master = load(master_file)
    branch_v, branch = load(branch_file)

    println("Julia: master=$master_v   branch=$branch_v")
    println()

    common = intersect(keys(master), keys(branch))
    rows = [(k, master[k], branch[k], master[k] / branch[k]) for k in common]
    sort!(rows, by=r -> -r[4])  # biggest speedup first

    println(@sprintf("%-4s %-10s %-6s %-7s %-11s %-12s %12s %12s %8s",
        "dim", "method", "n", "grid", "bc", "query",
        "master_ns", "branch_ns", "speedup"))
    println(repeat("─", 100))

    for (key, m, b, ratio) in rows
        println(@sprintf("%-4s %-10s %-6s %-7s %-11s %-12s %12.2f %12.2f %7.2fx",
            key..., m, b, ratio))
    end

    # Highlight regressions (ratio < 0.98 means branch is SLOWER)
    println()
    regressions = [(k, m, b, r) for (k, m, b, r) in rows if r < 0.98]
    if !isempty(regressions)
        println("⚠ Regressions (branch slower than master by > 2%):")
        for (k, m, b, r) in regressions
            println(@sprintf("   %s : %.2fns → %.2fns (%.2fx)", join(k, "/"), m, b, r))
        end
    else
        println("✓ No regressions (all branch timings within 2% of master or faster)")
    end

    # Highlight big wins
    big_wins = [(k, m, b, r) for (k, m, b, r) in rows if r >= 1.5]
    if !isempty(big_wins)
        println()
        println("🚀 Significant speedups (≥ 1.5×):")
        for (k, m, b, r) in big_wins
            println(@sprintf("   %s : %.2fns → %.2fns (%.2fx)", join(k, "/"), m, b, r))
        end
    end
end

main()
