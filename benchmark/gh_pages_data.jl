"""
    gh_pages_data.jl

Shared helpers for reading / collapsing / writing the github-action-benchmark
`data.js` file stored on the `gh-pages` branch. Used by:
- `update_gh_pages_data.jl` — CI store step (forward-only: touches one commit)
- `dedup_history.jl`        — one-time history cleanup (all commits)

Schema (per suite):
    window.BENCHMARK_DATA = { "entries": { "<suite>": [ <entry>, ... ] }, ... };
    <entry> = { "commit": {"id", "message", "timestamp", "url", ...},
                "date": <ms epoch>, "benches": [ {"name","value","unit","extra"} ] }
"""

using JSON

include(joinpath(@__DIR__, "bench_machine.jl"))   # machine_key / hardware_fingerprint

const SUITE = "FastInterpolations.jl Benchmarks"

"""
    parse_data_js(path) -> Dict

Parse a `data.js` file into its underlying object. Returns a fresh skeleton when
the file is missing or empty.
"""
function parse_data_js(path::AbstractString)
    if !isfile(path) || filesize(path) == 0
        return Dict{String, Any}(
            "entries" => Dict{String, Any}(),
            "lastUpdate" => 0,
            "repoUrl" => get(ENV, "BENCH_REPO_URL", ""),
        )
    end
    raw = strip(read(path, String))
    js = replace(raw, r"^window\.BENCHMARK_DATA\s*=\s*" => "")
    js = replace(js, r";\s*$" => "")
    return JSON.parse(js)
end

commit_id(entry) = get(get(entry, "commit", Dict{String, Any}()), "id", "")

_date(entry) = Float64(entry["date"])

"""
    merge_benches(entries) -> Vector

Per-benchmark minimum across all `entries` (which should share a commit). Keeps
the whole bench object (value + unit + extra) of the lowest-value measurement,
sorted by name.
"""
function merge_benches(entries)
    by_name = Dict{String, Any}()
    for e in entries
        for b in e["benches"]
            n = b["name"]
            if !haskey(by_name, n) || Float64(b["value"]) < Float64(by_name[n]["value"])
                by_name[n] = b
            end
        end
    end
    return [by_name[n] for n in sort(collect(keys(by_name)))]
end

"""
    collapse_entries(entries) -> entry

Collapse entries that share a commit into one: per-benchmark min value, the
earliest date, and the commit metadata from the earliest-dated entry.
"""
function collapse_entries(entries)
    i_earliest = argmin([_date(e) for e in entries])
    collapsed = Dict{String, Any}(
        "commit" => entries[i_earliest]["commit"],
        "date" => entries[i_earliest]["date"],
        "benches" => merge_benches(entries),
    )
    # Preserve a hardware fingerprint if any of the collapsed entries carries one.
    for e in entries
        if haskey(e, "cpu")
            collapsed["cpu"] = e["cpu"]
            break
        end
    end
    return collapsed
end

# ══════════════════════════════════════════════════════════════════════════════
# Machine-aware keying
# ══════════════════════════════════════════════════════════════════════════════
#
# The runner fleet mixes CPUs, so a bare per-commit line jitters as consecutive
# commits land on different boxes and a same-commit re-run on a faster box would
# wrongly `min` away a real number. We key every point by machine and never merge
# or compare across keys.

"""
    entry_machine_key(entry) -> String

Machine key of a stored point, from its `cpu` fingerprint. Legacy / fingerprint-
less points (or malformed ones) key as `"unknown"` so they form their own series
rather than contaminating a real machine's.
"""
function entry_machine_key(entry)
    haskey(entry, "cpu") || return "unknown"
    hw = entry["cpu"]
    (haskey(hw, "cpu_name") && haskey(hw, "ncores")) || return "unknown"
    return machine_key(hw)
end

"""
    store_measurement(all_entries, sha, key, new_benches, cpu, fallback_commit, fallback_date)
        -> (updated_entries, merged_entry)

Merge this run's measurement into `all_entries`, keyed by **(commit, machine)**:

- Entries sharing this `sha` **and** `key` are collapsed with `new_benches` to the
  per-benchmark minimum (a same-box re-run only ever lowers the point).
- Every other entry — different commit **or** different machine — is left
  untouched (forward-only; no cross-machine `min`).
- On a re-run the earliest date + original commit metadata are kept; a first-time
  (commit, machine) uses `fallback_commit` / `fallback_date`.
"""
function store_measurement(all_entries, sha, key, new_benches, cpu, fallback_commit, fallback_date)
    same = [e for e in all_entries if commit_id(e) == sha && entry_machine_key(e) == key]
    others = [e for e in all_entries if !(commit_id(e) == sha && entry_machine_key(e) == key)]

    merged_benches = merge_benches(vcat(same, [Dict{String, Any}("benches" => new_benches)]))

    if !isempty(same)
        i_earliest = argmin([_date(e) for e in same])
        date = same[i_earliest]["date"]
        commit = same[i_earliest]["commit"]
    else
        date = fallback_date
        commit = fallback_commit
    end

    merged_entry = Dict{String, Any}("commit" => commit, "date" => date, "benches" => merged_benches, "cpu" => cpu)
    return (vcat(others, [merged_entry]), merged_entry)
end

"""
    primary_machine(all_entries, override) -> String

The machine whose series is shown as the canonical trend line. `override`
(`BENCH_PRIMARY_MACHINE`) wins when non-empty — this is the knob a future
self-hosted "official" runner sets. Otherwise the most-common key across the
history is used (ties broken lexicographically for determinism).
"""
function primary_machine(all_entries, override)
    isempty(override) || return override
    isempty(all_entries) && return "unknown"
    counts = Dict{String, Int}()
    for e in all_entries
        k = entry_machine_key(e)
        counts[k] = get(counts, k, 0) + 1
    end
    best, bestn = "unknown", -1
    for k in sort(collect(keys(counts)))   # deterministic tie-break
        if counts[k] > bestn
            best, bestn = k, counts[k]
        end
    end
    return best
end

"""
    split_by_machine(all_entries, primary_key) -> Dict{suite_name => Vector{entry}}

Presentation transform: the primary machine's points go under the canonical
`SUITE` (the main, first chart), every other machine under `"SUITE (<key>)"`
(secondary charts). Each series is sorted by date. Storage stays lossless — this
only decides which suite name each point renders under.
"""
function split_by_machine(all_entries, primary_key)
    suites = Dict{String, Any}()
    for e in all_entries
        k = entry_machine_key(e)
        name = k == primary_key ? SUITE : "$SUITE ($k)"
        push!(get!(suites, name, Any[]), e)
    end
    for es in values(suites)
        sort!(es, by = _date)
    end
    return suites
end

"""
    collect_suite_entries(data) -> Vector

Gather every FastInterpolations point across the canonical suite and any
per-machine secondary suites into one flat list (for re-keying / re-splitting).
"""
function collect_suite_entries(data)
    entries_all = get(data, "entries", Dict{String, Any}())
    out = Any[]
    for (name, es) in entries_all
        (name == SUITE || startswith(name, "$SUITE (")) && append!(out, es)
    end
    return out
end

"""
    apply_suite_split!(data, suites)

Replace all FastInterpolations suites in `data` with the machine-split `suites`
(canonical + secondaries), leaving unrelated suites untouched.
"""
function apply_suite_split!(data, suites)
    entries_all = get!(data, "entries", Dict{String, Any}())
    for name in collect(keys(entries_all))
        (name == SUITE || startswith(name, "$SUITE (")) && delete!(entries_all, name)
    end
    for (name, es) in suites
        entries_all[name] = es
    end
    return data
end

"""
    write_data_js(path, data)

Serialize back to the `window.BENCHMARK_DATA = {...};` wrapper (minified,
matching github-action-benchmark's on-disk format).
"""
function write_data_js(path::AbstractString, data)
    return open(path, "w") do io
        print(io, "window.BENCHMARK_DATA = ")
        JSON.print(io, data)
        print(io, ";")
    end
end
