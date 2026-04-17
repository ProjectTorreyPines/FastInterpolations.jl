#!/usr/bin/env julia
#
# Benchmark: Linear/Constant oneshot — NoBC / PeriodicBC(:inclusive) / PeriodicBC(:exclusive)
#   1D + 2D, Range + Vector grid, scalar + vector queries.
#
# Emits JSON to stdout for A/B comparison (master vs this branch).
# Usage:
#   julia --project=benchmark benchmark/periodic_oneshot_benchmark.jl > out.json
#
# Measures min(ns/call) over 5 trials × NREPS reps after 3 warmup iterations.
# Intentionally avoids BenchmarkTools.jl to keep the benchmark self-contained and
# eligible to run under old Julia versions without installing additional packages.

using FastInterpolations
using JSON

const NWARMUP = 3
const NTRIALS = 5
const NREPS_SCALAR = 100_000
const NREPS_VECTOR = 1_000
const NREPS_BULK_2D = 1_000

# ─────────────────────────────────────────────────────────────
# Function-barriers (prevent captured-var boxes from @elapsed)
# ─────────────────────────────────────────────────────────────

@inline call_linear_scalar(x, y, xq, bc) = linear_interp(x, y, xq; bc=bc)
@inline call_constant_scalar(x, y, xq, bc) = constant_interp(x, y, xq; bc=bc)

@inline function call_linear_vector!(output, x, y, xqs, bc)
    linear_interp!(output, x, y, xqs; bc=bc)
end

@inline function call_constant_vector!(output, x, y, xqs, bc)
    constant_interp!(output, x, y, xqs; bc=bc)
end

@inline call_linear_2d_scalar(grids, data, q, bcs) = linear_interp(grids, data, q; bc=bcs)
@inline call_constant_2d_scalar(grids, data, q, bcs) = constant_interp(grids, data, q; bc=bcs)

# ─────────────────────────────────────────────────────────────
# Timing helper — returns ns/call (min over trials, after warmup)
# ─────────────────────────────────────────────────────────────

function time_call(f, args...; nreps)
    try
        for _ in 1:NWARMUP
            f(args...)
        end
    catch e
        @info "time_call: skipped (warmup error): $(typeof(e).name.name)"
        return NaN
    end
    best = Inf
    for _ in 1:NTRIALS
        t = @elapsed for _ in 1:nreps
            f(args...)
        end
        best = min(best, t)
    end
    return best * 1e9 / nreps
end

# ─────────────────────────────────────────────────────────────
# Benchmark configurations
# ─────────────────────────────────────────────────────────────

const RESULTS = Dict{String, Any}[]

function record!(entry)
    # Convert NaN ns_per_call to "skipped" string (JSON spec disallows NaN)
    if haskey(entry, "ns_per_call") && entry["ns_per_call"] isa Float64 && isnan(entry["ns_per_call"])
        entry["ns_per_call"] = "skipped"
    end
    push!(RESULTS, entry)
    return entry
end

function bc_from_spec(spec::Symbol, period=nothing)
    if spec === :nobc
        return NoBC()
    elseif spec === :inclusive
        return PeriodicBC()
    elseif spec === :exclusive
        return PeriodicBC(endpoint=:exclusive, period=period)
    else
        error("unknown bc spec: $spec")
    end
end

# ─────────────────────────────────────────────────────────────
# 1D benchmarks
# ─────────────────────────────────────────────────────────────

function bench_1d()
    for n in (50, 100, 1000), method in (:linear, :constant), grid_type in (:range, :vector)
        x_range = range(0.0, step=2π/n, length=n)
        x = grid_type === :range ? x_range : collect(x_range)
        y = sin.(x)
        xq = 1.5

        for bc_spec in (:nobc, :inclusive, :exclusive)
            bc = bc_from_spec(bc_spec, 2π)
            # For :inclusive, grid must have matched endpoints — reconstruct
            if bc_spec === :inclusive
                x_inc_range = range(0.0, 2π, length=n+1)
                x_use = grid_type === :range ? x_inc_range : collect(x_inc_range)
                y_use = sin.(x_use)
            else
                x_use, y_use = x, y
            end

            f = method === :linear ? call_linear_scalar : call_constant_scalar
            t = time_call(f, x_use, y_use, xq, bc; nreps=NREPS_SCALAR)
            record!(Dict(
                "dim" => 1, "method" => string(method), "n" => n,
                "grid_type" => string(grid_type), "bc" => string(bc_spec),
                "query" => "scalar", "ns_per_call" => t,
            ))
        end
    end

    # 1D vector query (100 queries)
    for n in (100, 1000), method in (:linear, :constant), grid_type in (:range, :vector)
        x_range = range(0.0, step=2π/n, length=n)
        x = grid_type === :range ? x_range : collect(x_range)
        y = sin.(x)
        xqs = collect(range(0.1, 2π-0.1, length=100))
        output = zeros(length(xqs))

        for bc_spec in (:nobc, :exclusive)
            bc = bc_from_spec(bc_spec, 2π)
            f! = method === :linear ? call_linear_vector! : call_constant_vector!
            t = time_call(f!, output, x, y, xqs, bc; nreps=NREPS_VECTOR)
            record!(Dict(
                "dim" => 1, "method" => string(method), "n" => n,
                "grid_type" => string(grid_type), "bc" => string(bc_spec),
                "query" => "vector100", "ns_per_call" => t,
            ))
        end
    end
end

# ─────────────────────────────────────────────────────────────
# 2D benchmarks
# ─────────────────────────────────────────────────────────────

function bench_2d()
    for n in (20, 50, 100), method in (:linear, :constant), grid_type in (:range, :vector)
        x_range = range(0.0, step=2π/n, length=n)
        y_range = range(-1.0, 1.0, length=n)
        x = grid_type === :range ? x_range : collect(x_range)
        y = grid_type === :range ? y_range : collect(y_range)
        data = [sin(xi) * exp(-yj^2) for xi in x, yj in y]
        q = (1.5, 0.3)

        for bc_spec in (:nobc, :exclusive)
            bc = bc_from_spec(bc_spec, 2π)
            bcs = bc_spec === :exclusive ? (bc, NoBC()) : (NoBC(), NoBC())

            if bc_spec === :inclusive
                # inclusive 2D requires matched endpoints — skip to keep perf comparison clean
                continue
            end

            f = method === :linear ? call_linear_2d_scalar : call_constant_2d_scalar
            t = time_call(f, (x, y), data, q, bcs; nreps=NREPS_BULK_2D)
            record!(Dict(
                "dim" => 2, "method" => string(method), "n" => n,
                "grid_type" => string(grid_type), "bc" => string(bc_spec),
                "query" => "scalar", "ns_per_call" => t,
            ))
        end
    end
end

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

function main()
    println(stderr, "Running 1D benchmarks ...")
    bench_1d()
    println(stderr, "Running 2D benchmarks ...")
    bench_2d()
    println(stderr, "Done. $(length(RESULTS)) entries.")

    # Emit JSON to stdout
    JSON.print(stdout, Dict(
        "julia_version" => string(VERSION),
        "n_entries" => length(RESULTS),
        "results" => RESULTS,
    ), 2)
    println()
end

main()
