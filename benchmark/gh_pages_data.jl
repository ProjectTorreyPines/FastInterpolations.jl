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
