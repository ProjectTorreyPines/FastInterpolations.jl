"""
    update_gh_pages_data.jl <existing_data.js> <master_benches.json> <out_data.js>

CI store step. Merges the current master commit's benches into `data.js`,
keyed by commit SHA:

- If the commit already has one or more entries (e.g. a re-run), they are
  collapsed with the new measurement to the per-benchmark **minimum** and
  replaced by a single entry — so re-running a commit only *lowers* its stored
  point instead of appending a noisy duplicate.
- Otherwise a new entry is appended.

**Forward-only**: entries for *other* commits are left untouched. Bulk cleanup
of pre-existing duplicates is a separate, deliberate step (`dedup_history.jl`).

Environment:
    BENCH_SHA          (required) commit id used as the merge key
    BENCH_COMMIT_META  (optional) path to a JSON file with commit metadata for a
                       brand-new entry: {id,message,timestamp,url,author,
                       committer,date_ms}. Built by the workflow via `gh api`
                       (avoids injecting multi-line commit messages through env).
    BENCH_DATE_MS      (fallback) entry date in ms when no meta file is given.
"""

include(joinpath(@__DIR__, "gh_pages_data.jl"))

length(ARGS) == 3 || error("usage: update_gh_pages_data.jl <data.js> <master_benches.json> <out.js>")
const DATA_PATH = ARGS[1]
const NEW_BENCHES_PATH = ARGS[2]
const OUT_PATH = ARGS[3]

const SHA = get(ENV, "BENCH_SHA", "")
isempty(SHA) && error("BENCH_SHA is required")

data = parse_data_js(DATA_PATH)
entries_all = get!(data, "entries", Dict{String, Any}())
suite_entries = get(entries_all, SUITE, Any[])

new_benches = JSON.parsefile(NEW_BENCHES_PATH)  # [{name,value,unit,extra}]

same = [e for e in suite_entries if commit_id(e) == SHA]
others = [e for e in suite_entries if commit_id(e) != SHA]

# Per-benchmark min over existing same-SHA entries ∪ this run's benches.
new_entry_wrapper = Dict{String, Any}("benches" => new_benches)
merged_benches = merge_benches(vcat(same, [new_entry_wrapper]))

if isempty(merged_benches)
    @warn "No benches to store — writing data.js unchanged"
    write_data_js(OUT_PATH, data)
    exit(0)
end

# Date + commit metadata: keep the commit's original (earliest) identity on a
# re-run; synthesize for a first-time commit from the meta file (or env).
if !isempty(same)
    i_earliest = argmin([_date(e) for e in same])
    entry_date = same[i_earliest]["date"]
    commit = same[i_earliest]["commit"]
else
    meta_path = get(ENV, "BENCH_COMMIT_META", "")
    if !isempty(meta_path) && isfile(meta_path)
        m = JSON.parsefile(meta_path)
        author = get(m, "author", Dict{String, Any}("name" => "", "email" => ""))
        commit = Dict{String, Any}(
            "id" => get(m, "id", SHA),
            "message" => get(m, "message", ""),
            "timestamp" => get(m, "timestamp", ""),
            "url" => get(m, "url", ""),
            "author" => author,
            "committer" => get(m, "committer", author),
        )
        entry_date = round(Int, Float64(get(m, "date_ms", parse(Float64, get(ENV, "BENCH_DATE_MS", "0")))))
    else
        commit = Dict{String, Any}("id" => SHA)
        entry_date = parse(Int, get(ENV, "BENCH_DATE_MS", "0"))
    end
end

merged_entry = Dict{String, Any}("commit" => commit, "date" => entry_date, "benches" => merged_benches)

# Diagnostic hardware fingerprint of the run that produced this point (annotation
# only — the graph tooltip / our tooling can flag anomalously-fast hardware).
hw_path = get(ENV, "BENCH_HARDWARE", "")
if !isempty(hw_path) && isfile(hw_path)
    merged_entry["cpu"] = JSON.parsefile(hw_path)
else
    # Carry a prior fingerprint forward on a re-run with no fresh one.
    for e in same
        if haskey(e, "cpu")
            merged_entry["cpu"] = e["cpu"]
            break
        end
    end
end

new_suite = vcat(others, [merged_entry])
sort!(new_suite, by = _date)
entries_all[SUITE] = new_suite
data["lastUpdate"] = round(Int, maximum(_date, new_suite))

write_data_js(OUT_PATH, data)
println(
    "Stored SHA $(SHA[1:min(8, lastindex(SHA))]): $(length(merged_benches)) benches; " *
        "collapsed $(length(same)) prior same-commit entr$(length(same) == 1 ? "y" : "ies"); " *
        "suite now $(length(new_suite)) entries"
)
