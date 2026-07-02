"""
    dedup_history.jl <data.js> [<out.js>]

One-time (idempotent) cleanup of the gh-pages benchmark history: collapse every
group of entries that share a **(commit id, machine)** into a single entry
holding the per-benchmark **minimum** across that group's runs, keeping the
earliest date and commit metadata. Points are then re-split by machine into the
canonical suite (primary machine) + per-machine secondary suites and re-sorted
chronologically.

This de-noises the graph (each commit becomes one point per machine at its best
measurement) and removes duplicate points left by past re-runs. Grouping is
per-machine so a commit measured on two different boxes stays two points —
collapsing across machines would mix incomparable timings.

Intended to be run locally against a fresh `gh-pages` checkout; the resulting
`data.js` is then pushed deliberately (this script never pushes).

    out defaults to overwriting the input path.
    BENCH_PRIMARY_MACHINE (optional) pins the canonical series; else most-common.
"""

include(joinpath(@__DIR__, "gh_pages_data.jl"))

length(ARGS) >= 1 || error("usage: dedup_history.jl <data.js> [<out.js>]")
const DATA_PATH = ARGS[1]
const OUT_PATH = length(ARGS) >= 2 ? ARGS[2] : ARGS[1]

data = parse_data_js(DATA_PATH)
all_entries = collect_suite_entries(data)

if isempty(all_entries)
    println("No FastInterpolations entries — nothing to do")
    exit(0)
end

# Group by (commit id, machine): dupes from past re-runs of the SAME box land in
# the same group; different boxes stay separate.
by_key = Dict{Tuple{String, String}, Vector{Any}}()
for e in all_entries
    push!(get!(by_key, (commit_id(e), entry_machine_key(e)), Any[]), e)
end

collapsed = [collapse_entries(group) for group in values(by_key)]

primary = primary_machine(collapsed, get(ENV, "BENCH_PRIMARY_MACHINE", ""))
suites = split_by_machine(collapsed, primary)
apply_suite_split!(data, suites)
data["lastUpdate"] = round(Int, maximum(_date, collapsed))

write_data_js(OUT_PATH, data)

n_dupe = count(>(1) ∘ length, values(by_key))
println(
    "Deduped: $(length(all_entries)) entries → $(length(collapsed)) unique (commit,machine) points " *
        "($(n_dupe) had duplicates); $(length(suites)) machine series (primary=$primary)"
)
