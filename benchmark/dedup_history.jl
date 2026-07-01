"""
    dedup_history.jl <data.js> [<out.js>]

One-time (idempotent) cleanup of the gh-pages benchmark history: collapse every
group of entries that share a commit id into a single entry holding the
per-benchmark **minimum** across that commit's runs, keeping the earliest date
and commit metadata. Entries are re-sorted chronologically.

This de-noises the graph (each commit becomes one point at its best measurement)
and removes the duplicate points left by past re-runs of the stock benchmark
action. Running it again is a no-op once the history is already unique.

Intended to be run locally against a fresh `gh-pages` checkout; the resulting
`data.js` is then pushed deliberately (this script never pushes).

    out defaults to overwriting the input path.
"""

include(joinpath(@__DIR__, "gh_pages_data.jl"))

length(ARGS) >= 1 || error("usage: dedup_history.jl <data.js> [<out.js>]")
const DATA_PATH = ARGS[1]
const OUT_PATH = length(ARGS) >= 2 ? ARGS[2] : ARGS[1]

data = parse_data_js(DATA_PATH)
entries_all = get(data, "entries", Dict{String, Any}())
suite_entries = get(entries_all, SUITE, Any[])

if isempty(suite_entries)
    println("No entries for suite '$SUITE' — nothing to do")
    exit(0)
end

# Group by commit id (dupes from past re-runs land in the same group).
by_id = Dict{String, Vector{Any}}()
for e in suite_entries
    push!(get!(by_id, commit_id(e), Any[]), e)
end

collapsed = [collapse_entries(group) for group in values(by_id)]
sort!(collapsed, by = _date)

entries_all[SUITE] = collapsed
data["entries"] = entries_all

write_data_js(OUT_PATH, data)

n_dupe_commits = count(>(1) ∘ length, values(by_id))
println(
    "Deduped '$SUITE': $(length(suite_entries)) entries → $(length(collapsed)) unique commits " *
        "($(n_dupe_commits) commit(s) had duplicates)"
)
