"""
    regression_check.jl

Two-tier regression detection with automatic re-verification.
Included by ci_benchmark.jl when --baseline is provided.

Tier 1 (Immediate):  current / latest_master > IMMEDIATE_THRESHOLD
Tier 2 (Gradual):    current / mean(window around M commits ago) > GRADUAL_THRESHOLD
"""

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

const IMMEDIATE_THRESHOLD = 1.10   # vs latest master commit
const GRADUAL_THRESHOLD = 1.10     # vs M-ago window average
const LOOKBACK_M = 10              # how far back for gradual baseline
const WINDOW_W = 5                 # averaging window size around M
const RERUN_N = 3                  # re-run count for suspected regressions

# ══════════════════════════════════════════════════════════════════════════════
# Baseline Loading
# ══════════════════════════════════════════════════════════════════════════════

"""
    load_baseline(data_js_path) -> (latest, window_avg)

Parse gh-pages `data.js` and extract two baselines:
- `latest`: Dict{String,Float64} from the most recent master push
- `window_avg`: Dict{String,Float64} averaged over WINDOW_W commits around LOOKBACK_M ago
"""
function load_baseline(data_js_path::String)
    empty_result = (Dict{String, Float64}(), Dict{String, Float64}())

    raw = read(data_js_path, String)
    isempty(strip(raw)) && return empty_result

    # Strip window.BENCHMARK_DATA = ...; wrapper
    json_str = replace(raw, r"^window\.BENCHMARK_DATA\s*=\s*" => "")
    json_str = replace(json_str, r";\s*$" => "")

    data = JSON.parse(json_str)
    entries_dict = get(data, "entries", Dict())
    isempty(entries_dict) && return empty_result

    # Get benchmark suite entries (typically one key)
    entries = first(values(entries_dict))
    isempty(entries) && return empty_result

    # --- Latest commit baseline ---
    latest_benches = entries[end]["benches"]
    latest = Dict{String, Float64}(b["name"] => Float64(b["value"]) for b in latest_benches)

    # --- Window average around M commits ago ---
    n = length(entries)
    if n < 2
        return (latest, Dict{String, Float64}())
    end

    center = clamp(n - LOOKBACK_M, 1, n)
    w_start = max(1, center - WINDOW_W ÷ 2)
    w_end = min(n, center + WINDOW_W ÷ 2)
    window_entries = entries[w_start:w_end]

    # Accumulate per-benchmark averages across window
    window_sum = Dict{String, Float64}()
    window_count = Dict{String, Int}()
    for entry in window_entries
        for b in entry["benches"]
            name = b["name"]
            window_sum[name] = get(window_sum, name, 0.0) + Float64(b["value"])
            window_count[name] = get(window_count, name, 0) + 1
        end
    end
    window_avg = Dict{String, Float64}(
        name => total / window_count[name] for (name, total) in window_sum
    )

    return (latest, window_avg)
end

# ══════════════════════════════════════════════════════════════════════════════
# Regression Detection
# ══════════════════════════════════════════════════════════════════════════════

struct FlaggedBench
    group_key::String
    bench_key::String
    full_name::String
    current_ns::Float64
    ratio_immediate::Union{Float64, Nothing}
    ratio_gradual::Union{Float64, Nothing}
    tier::Symbol   # :immediate, :gradual, or :both
end

"""
    detect_regressions(results, latest, window_avg) -> Vector{FlaggedBench}

Two-tier comparison against baselines. Returns flagged benchmarks.
"""
function detect_regressions(
    results::BenchmarkGroup,
    latest::Dict{String, Float64},
    window_avg::Dict{String, Float64},
)
    flagged = FlaggedBench[]

    for group_name in keys(results)
        group = results[group_name]
        for bench_name in keys(group)
            trial = group[bench_name]
            current_ns = minimum(trial).time
            full_name = "$group_name/$bench_name"

            r_imm = nothing
            r_grad = nothing
            hit_imm = false
            hit_grad = false

            # Tier 1: Immediate (vs latest master)
            if haskey(latest, full_name)
                r_imm = current_ns / latest[full_name]
                hit_imm = r_imm > IMMEDIATE_THRESHOLD
            end

            # Tier 2: Gradual (vs M-ago window average)
            if haskey(window_avg, full_name)
                r_grad = current_ns / window_avg[full_name]
                hit_grad = r_grad > GRADUAL_THRESHOLD
            end

            if hit_imm || hit_grad
                tier = (hit_imm && hit_grad) ? :both : hit_imm ? :immediate : :gradual
                push!(flagged, FlaggedBench(group_name, bench_name, full_name,
                    current_ns, r_imm, r_grad, tier))
            end
        end
    end

    return flagged
end

# ══════════════════════════════════════════════════════════════════════════════
# Re-run and Merge
# ══════════════════════════════════════════════════════════════════════════════

"""
    rerun_and_merge!(suite, results, flagged, n_reruns)

Re-run each flagged benchmark n_reruns times.
If a re-run produces a lower minimum time, replace the trial in results.
"""
function rerun_and_merge!(
    suite::BenchmarkGroup,
    results::BenchmarkGroup,
    flagged::Vector{FlaggedBench},
    n_reruns::Int,
)
    for i in 1:n_reruns
        println("  Re-run $i/$n_reruns ($(length(flagged)) benchmarks)...")
        for fb in flagged
            GC.gc()
            rerun_trial = run(suite[fb.group_key][fb.bench_key])
            if minimum(rerun_trial).time < minimum(results[fb.group_key][fb.bench_key]).time
                results[fb.group_key][fb.bench_key] = rerun_trial
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════════════

"""
    write_regression_report(path, results, latest, window_avg, initially_flagged, confirmed)

Write regression_report.json containing per-benchmark details and summary.
"""
function write_regression_report(
    path::String,
    results::BenchmarkGroup,
    latest::Dict{String, Float64},
    window_avg::Dict{String, Float64},
    initially_flagged::Vector{FlaggedBench},
    confirmed::Vector{FlaggedBench},
)
    flagged_names = Set(fb.full_name for fb in initially_flagged)
    confirmed_names = Set(fb.full_name for fb in confirmed)

    benchmarks = Dict{String, Any}[]

    for group_name in sort(collect(keys(results)))
        group = results[group_name]
        for bench_name in sort(collect(keys(group)))
            trial = group[bench_name]
            current_ns = minimum(trial).time
            full_name = "$group_name/$bench_name"

            entry = Dict{String, Any}(
                "name" => full_name,
                "value" => round(current_ns, digits = 1),
                "unit" => "ns",
            )

            if haskey(latest, full_name)
                entry["prev_value"] = round(latest[full_name], digits = 1)
                entry["ratio_immediate"] = round(current_ns / latest[full_name], digits = 3)
            end
            if haskey(window_avg, full_name)
                entry["window_avg_value"] = round(window_avg[full_name], digits = 1)
                entry["ratio_gradual"] = round(current_ns / window_avg[full_name], digits = 3)
            end

            if full_name in flagged_names
                entry["was_flagged"] = true
                entry["rerun_verified"] = true
                entry["regression"] = full_name in confirmed_names
            else
                entry["was_flagged"] = false
                entry["regression"] = false
            end

            push!(benchmarks, entry)
        end
    end

    report = Dict{String, Any}(
        "thresholds" => Dict{String, Any}(
            "immediate" => IMMEDIATE_THRESHOLD,
            "gradual" => GRADUAL_THRESHOLD,
            "lookback_m" => LOOKBACK_M,
            "window_w" => WINDOW_W,
            "rerun_n" => RERUN_N,
        ),
        "summary" => Dict{String, Any}(
            "total" => length(benchmarks),
            "initially_flagged" => length(initially_flagged),
            "confirmed_regressions" => length(confirmed),
        ),
        "benchmarks" => benchmarks,
    )

    open(path, "w") do io
        JSON.print(io, report, 2)
    end
end
