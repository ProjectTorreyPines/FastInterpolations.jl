"""
    regression_check.jl

Two-tier regression detection with automatic re-verification.
Included by ci_benchmark.jl when --baseline is provided.

Tier 1 (Immediate):  current / latest_master > IMMEDIATE_THRESHOLD
Tier 2 (Gradual):    current / mean(window around M commits ago) > GRADUAL_THRESHOLD
"""

using Printf

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

const IMMEDIATE_THRESHOLD = 1.1   # vs latest master commit
const GRADUAL_THRESHOLD = 1.1     # vs M-ago window average
const LOOKBACK_M = 10              # how far back for gradual baseline
const WINDOW_W = 5                 # averaging window size around M
const RERUN_N = 10                 # re-run count for suspected regressions

# ══════════════════════════════════════════════════════════════════════════════
# Baseline Loading
# ══════════════════════════════════════════════════════════════════════════════

const SUITE_NAME = "FastInterpolations.jl Benchmarks"

"""
    read_suite_entries(data_js_path) -> Vector

Parse a gh-pages `data.js` and return the chronological entry array for this
package's benchmark suite (each entry: `Dict` with "commit", "date",
"benches"). Returns an empty vector on missing / empty / unparseable input.
"""
function read_suite_entries(data_js_path::String)
    (isfile(data_js_path) && filesize(data_js_path) > 0) || return Any[]
    raw = strip(read(data_js_path, String))
    isempty(raw) && return Any[]

    # Strip window.BENCHMARK_DATA = ...; wrapper
    json_str = replace(raw, r"^window\.BENCHMARK_DATA\s*=\s*" => "")
    json_str = replace(json_str, r";\s*$" => "")

    data = JSON.parse(json_str)
    entries_dict = get(data, "entries", Dict())
    isempty(entries_dict) && return Any[]
    return get(entries_dict, SUITE_NAME, get(entries_dict, first(keys(entries_dict)), Any[]))
end

_commit_id(entry) = get(get(entry, "commit", Dict{String, Any}()), "id", "")

_benches_to_dict(benches) = Dict{String, Float64}(b["name"] => Float64(b["value"]) for b in benches)

"""
    _window_average(entries) -> Dict{String,Float64}

Per-benchmark average over WINDOW_W entries centred LOOKBACK_M back from the end
of `entries`. Empty when there are too few entries.
"""
function _window_average(entries)
    n = length(entries)
    n < 2 && return Dict{String, Float64}()

    center = clamp(n - LOOKBACK_M, 1, n)
    w_start = max(1, center - WINDOW_W ÷ 2)
    w_end = min(n, center + WINDOW_W ÷ 2)

    window_sum = Dict{String, Float64}()
    window_count = Dict{String, Int}()
    for entry in entries[w_start:w_end]
        for b in entry["benches"]
            name = b["name"]
            window_sum[name] = get(window_sum, name, 0.0) + Float64(b["value"])
            window_count[name] = get(window_count, name, 0) + 1
        end
    end
    return Dict{String, Float64}(name => total / window_count[name] for (name, total) in window_sum)
end

"""
    load_baseline(data_js_path) -> (latest, window_avg)

Parse gh-pages `data.js` and extract two baselines (PR mode):
- `latest`: Dict{String,Float64} from the most recent push
- `window_avg`: Dict{String,Float64} averaged over WINDOW_W commits around LOOKBACK_M ago
"""
function load_baseline(data_js_path::String)
    empty_result = (Dict{String, Float64}(), Dict{String, Float64}())
    entries = try
        read_suite_entries(data_js_path)
    catch
        return empty_result
    end
    isempty(entries) && return empty_result

    latest = _benches_to_dict(entries[end]["benches"])
    return (latest, _window_average(entries))
end

"""
    load_master_baseline(data_js_path, current_sha) -> (prev_best, latest, window_avg)

Master-store variant. Splits the history relative to `current_sha`:
- `prev_best`: per-benchmark min across ALL existing entries for `current_sha`
  (robust to pre-existing duplicate points) — the cross-run floor that makes a
  re-run of the same master commit only lower the stored value.
- `latest` / `window_avg`: baselines from entries of *other* commits (the true
  previous master state), so regression detection never compares a commit to
  its own earlier run.
"""
function load_master_baseline(data_js_path::String, current_sha::AbstractString)
    empty_result = (Dict{String, Float64}(), Dict{String, Float64}(), Dict{String, Float64}())
    entries = try
        read_suite_entries(data_js_path)
    catch
        return empty_result
    end
    isempty(entries) && return empty_result

    prev_best = Dict{String, Float64}()
    for e in entries
        _commit_id(e) == current_sha || continue
        for b in e["benches"]
            name = b["name"]
            v = Float64(b["value"])
            prev_best[name] = haskey(prev_best, name) ? min(prev_best[name], v) : v
        end
    end

    others = [e for e in entries if _commit_id(e) != current_sha]
    latest = isempty(others) ? Dict{String, Float64}() : _benches_to_dict(others[end]["benches"])
    return (prev_best, latest, _window_average(others))
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
    split_full_name(full_name) -> (group_key, bench_key)

Recover the `suite[group][bench]` keys from a `"group/bench"` full name.
Group and benchmark keys never contain '/', so a single split on the first
'/' is unambiguous.
"""
function split_full_name(full_name::AbstractString)
    parts = split(full_name, '/', limit = 2)
    return length(parts) == 2 ? (String(parts[1]), String(parts[2])) : (String(parts[1]), "")
end

"""
    compute_effective(results, prev_best) -> Dict{String,Float64}

Build the per-benchmark *effective* minimum time used for all regression
decisions. This is the single place where the min-merge across re-runs of the
same commit happens:

- Every freshly measured benchmark contributes `min(this_run_min, prev_best)`.
- Any benchmark present only in `prev_best` (e.g. not re-run in a flagged-only
  subset run) is carried over verbatim.

With an empty `prev_best` and a full-suite `results`, this reduces to the plain
per-benchmark minimum — i.e. identical to the pre-merge behaviour.
"""
function compute_effective(
        results::BenchmarkGroup,
        prev_best::Dict{String, Float64},
    )
    eff = Dict{String, Float64}()
    # Seed with prior-run values (carried over for benches not re-measured here)
    for (name, v) in prev_best
        eff[name] = v
    end
    # Override / lower with this run's measurements
    for group_name in keys(results)
        group = results[group_name]
        for bench_name in keys(group)
            full_name = "$group_name/$bench_name"
            this_min = minimum(group[bench_name]).time
            eff[full_name] = min(this_min, get(prev_best, full_name, Inf))
        end
    end
    return eff
end

"""
    detect_regressions(effective, latest, window_avg) -> Vector{FlaggedBench}

Two-tier comparison of the effective per-benchmark minimums against baselines.
Returns flagged benchmarks sorted by name (deterministic ordering).
"""
function detect_regressions(
        effective::Dict{String, Float64},
        latest::Dict{String, Float64},
        window_avg::Dict{String, Float64},
    )
    flagged = FlaggedBench[]

    for full_name in sort(collect(keys(effective)))
        current_ns = effective[full_name]
        group_key, bench_key = split_full_name(full_name)

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
            push!(
                flagged, FlaggedBench(
                    group_key, bench_key, full_name,
                    current_ns, r_imm, r_grad, tier
                )
            )
        end
    end

    return flagged
end

# ══════════════════════════════════════════════════════════════════════════════
# Re-run and Merge
# ══════════════════════════════════════════════════════════════════════════════

"""
    rerun_and_merge!(suite, results, effective, flagged, n_reruns, prev_best, latest, window_avg)

Re-run each flagged benchmark n_reruns times. If a re-run produces a lower
minimum time, replace the trial in `results` and lower its `effective` entry
(never below the `prev_best` floor is not enforced here — instead the effective
value is `min(best_trial, prev_best)` so the cross-run floor is preserved).

Flagged benchmarks that are not present in `suite` (e.g. carried over from
`prev_best` in a flagged-only subset run) are left untouched — we trust their
prior measurement rather than fabricate one.

Prints current effective min and imm/grad ratios after each re-run.
"""
function rerun_and_merge!(
        suite::BenchmarkGroup,
        results::BenchmarkGroup,
        effective::Dict{String, Float64},
        flagged::Vector{FlaggedBench},
        n_reruns::Int,
        prev_best::Dict{String, Float64},
        latest::Dict{String, Float64},
        window_avg::Dict{String, Float64},
    )
    for i in 1:n_reruns
        println("  Re-run $i/$n_reruns ($(length(flagged)) benchmarks)...")
        for fb in flagged
            # Skip benches not in this run's suite (carried over from prev_best)
            (haskey(suite, fb.group_key) && haskey(suite[fb.group_key], fb.bench_key)) || continue

            GC.gc()
            rerun_trial = run(suite[fb.group_key][fb.bench_key])
            old_min = minimum(results[fb.group_key][fb.bench_key]).time
            new_min = minimum(rerun_trial).time
            updated = new_min < old_min
            if updated
                results[fb.group_key][fb.bench_key] = rerun_trial
            end
            # Effective value keeps the cross-run floor from prev_best
            cur = min(updated ? new_min : old_min, get(prev_best, fb.full_name, Inf))
            effective[fb.full_name] = cur

            r_imm = haskey(latest, fb.full_name) ? @sprintf("%.3f", cur / latest[fb.full_name]) : "-"
            r_grad = haskey(window_avg, fb.full_name) ? @sprintf("%.3f", cur / window_avg[fb.full_name]) : "-"
            tag = updated ? "↓" : "="
            @printf("    %s %-40s  min: %8.1f ns  imm: %s  grad: %s\n", tag, fb.full_name, cur, r_imm, r_grad)
        end
    end
    return
end

# ══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════════════

"""
    write_regression_report(path, effective, latest, window_avg, initially_flagged, confirmed)

Write regression_report.json containing per-benchmark details and summary.
Values are the *effective* (cross-run min-merged) times, so the report — and
the hidden data blob derived from it — reflect the best measurement seen for
this commit across all re-runs.
"""
function write_regression_report(
        path::String,
        effective::Dict{String, Float64},
        latest::Dict{String, Float64},
        window_avg::Dict{String, Float64},
        initially_flagged::Vector{FlaggedBench},
        confirmed::Vector{FlaggedBench},
    )
    flagged_names = Set(fb.full_name for fb in initially_flagged)
    confirmed_names = Set(fb.full_name for fb in confirmed)

    benchmarks = Dict{String, Any}[]

    for full_name in sort(collect(keys(effective)))
        current_ns = effective[full_name]

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

    return open(path, "w") do io
        JSON.print(io, report, 2)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Master-store serialization (data.js benches format)
# ══════════════════════════════════════════════════════════════════════════════

"""
    _bench_extra(trial) -> String

github-action-benchmark "extra" field for a Julia benchmark: gctime / memory /
allocs / params, matching the format already stored in gh-pages `data.js`.
"""
function _bench_extra(trial)
    t = minimum(trial)
    p = trial.params
    params_dict = Dict{String, Any}(
        "evals" => p.evals,
        "samples" => p.samples,
        "seconds" => p.seconds,
        "overhead" => p.overhead,
        "gctrial" => p.gctrial,
        "gcsample" => p.gcsample,
        "time_tolerance" => p.time_tolerance,
        "memory_tolerance" => p.memory_tolerance,
    )
    return "gctime=$(t.gctime)\nmemory=$(trial.memory)\nallocs=$(trial.allocs)\nparams=$(JSON.json(params_dict))"
end

"""
    write_master_benches(path, effective, results)

Write the current commit's benches in gh-pages `data.js` format
(`[{name, value, unit, extra}]`) using the *effective* (min-merged) times.
`update_gh_pages_data.jl` consumes this to replace/append the commit's entry.
memory/allocs/params come from this run's trial (deterministic for the same
code, so valid even when the time floor came from a prior run).
"""
function write_master_benches(
        path::String,
        effective::Dict{String, Float64},
        results::BenchmarkGroup,
    )
    benches = Dict{String, Any}[]
    for full_name in sort(collect(keys(effective)))
        group_key, bench_key = split_full_name(full_name)
        extra = (haskey(results, group_key) && haskey(results[group_key], bench_key)) ?
            _bench_extra(results[group_key][bench_key]) : ""
        push!(
            benches, Dict{String, Any}(
                "name" => full_name,
                "value" => round(effective[full_name], digits = 3),
                "unit" => "ns",
                "extra" => extra,
            )
        )
    end
    return open(path, "w") do io
        JSON.print(io, benches)
    end
end
