"""
    update_gh_pages_data.jl <existing_data.js> <master_benches.json> <out_data.js>

CI store step. Merges the current master commit's benches into `data.js`, keyed
by **(commit SHA, machine)**:

- If this (commit, machine) already has one or more entries (a same-box re-run),
  they are collapsed with the new measurement to the per-benchmark **minimum**
  and replaced by a single entry — a re-run only *lowers* its stored point.
- A different machine on the same commit becomes a **separate** entry; we never
  `min` across machines (the fleet mixes CPUs, so that would mask real numbers).
- **Forward-only**: entries for other commits/machines are left untouched.

For display, points are split by machine across suite names: the primary
machine's series is the canonical suite (the main chart), every other machine a
secondary `"<suite> (<key>)"`. Storage stays lossless; the split only decides the
suite name each point renders under.

Environment:
    BENCH_SHA              (required) commit id used as the merge key
    BENCH_HARDWARE         (required) path to hardware.json — keys the point by machine
    BENCH_PRIMARY_MACHINE  (optional) machine key to pin as the canonical series;
                           defaults to the most-common key in the history
    BENCH_COMMIT_META      (optional) JSON metadata for a brand-new entry:
                           {id,message,timestamp,url,author,committer,date_ms}
    BENCH_DATE_MS          (fallback) entry date in ms when no meta file is given
"""

include(joinpath(@__DIR__, "gh_pages_data.jl"))

length(ARGS) == 3 || error("usage: update_gh_pages_data.jl <data.js> <master_benches.json> <out.js>")
const DATA_PATH = ARGS[1]
const NEW_BENCHES_PATH = ARGS[2]
const OUT_PATH = ARGS[3]

const SHA = get(ENV, "BENCH_SHA", "")
isempty(SHA) && error("BENCH_SHA is required")

# Hardware fingerprint is REQUIRED — it keys the point onto a machine series.
# Without it we cannot place the measurement on the right line, so fail loudly
# rather than store a mis-keyed point.
const HW_PATH = get(ENV, "BENCH_HARDWARE", "")
(!isempty(HW_PATH) && isfile(HW_PATH)) ||
    error("BENCH_HARDWARE must point to an existing hardware.json (required to key the point by machine)")
const CPU = JSON.parsefile(HW_PATH)
const CURRENT_KEY = machine_key(CPU)

data = parse_data_js(DATA_PATH)
all_entries = collect_suite_entries(data)

new_benches = JSON.parsefile(NEW_BENCHES_PATH)  # [{name,value,unit,extra}]

# Fallback commit metadata / date for a first-time (commit, machine) point.
meta_path = get(ENV, "BENCH_COMMIT_META", "")
if !isempty(meta_path) && isfile(meta_path)
    m = JSON.parsefile(meta_path)
    author = get(m, "author", Dict{String, Any}("name" => "", "email" => ""))
    fallback_commit = Dict{String, Any}(
        "id" => get(m, "id", SHA),
        "message" => get(m, "message", ""),
        "timestamp" => get(m, "timestamp", ""),
        "url" => get(m, "url", ""),
        "author" => author,
        "committer" => get(m, "committer", author),
    )
    fallback_date = round(Int, Float64(get(m, "date_ms", parse(Float64, get(ENV, "BENCH_DATE_MS", "0")))))
else
    fallback_commit = Dict{String, Any}("id" => SHA)
    fallback_date = parse(Int, get(ENV, "BENCH_DATE_MS", "0"))
end

updated, merged_entry = store_measurement(all_entries, SHA, CURRENT_KEY, new_benches, CPU, fallback_commit, fallback_date)

if isempty(merged_entry["benches"])
    @warn "No benches to store — writing data.js unchanged"
    write_data_js(OUT_PATH, data)   # `data` not yet mutated (split happens below)
    exit(0)
end

primary = primary_machine(updated, get(ENV, "BENCH_PRIMARY_MACHINE", ""))
suites = split_by_machine(updated, primary)
apply_suite_split!(data, suites)
data["lastUpdate"] = round(Int, maximum(_date, updated))

write_data_js(OUT_PATH, data)

n_collapsed = count(e -> commit_id(e) == SHA && entry_machine_key(e) == CURRENT_KEY, all_entries)
println(
    "Stored SHA $(SHA[1:min(8, lastindex(SHA))]) on machine $CURRENT_KEY (primary=$primary): " *
        "$(length(merged_entry["benches"])) benches; collapsed $n_collapsed prior same-(commit,machine) " *
        "entr$(n_collapsed == 1 ? "y" : "ies"); history now $(length(updated)) points across $(length(suites)) machine series"
)
