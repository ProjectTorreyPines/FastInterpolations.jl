# Regenerate test/shard_weights.toml from a ReTestItems CI log.
#
# The per-file weights that balance the sharded CI (RETESTITEMS_SHARD=i/N, see
# test/runtests_parallel.jl) come straight from a real run's log: ReTestItems prints
# per-item START/DONE lines carrying the source file and wall time, e.g.
#   ... START (  4/732) test item "..." at test/test_x.jl:233
#   ... DONE  (  4/732) test item "..." 0.7 secs (78.9% compile), ...
# We join START→file with DONE→secs on the item index and sum per file (whole-file
# granularity, matching how the runner ships files to shards).
#
# This is a MANUAL, on-demand refresh — CI never runs it. Weights drift slowly, so
# regenerate only occasionally. TWO per-OS tables (the runner picks by Sys.iswindows):
#   test/shard_weights.toml          DEFAULT (ubuntu/macOS; low CI variance)
#   test/shard_weights_windows.toml  windows (HIGH per-file variance ±50% + its own heavy
#                                    files, so its OWN ranking must drive the split — a
#                                    shared table leaves windows at ~1.6, its own ~1.1)
# The suite is sharded, so pass BOTH shard logs of a run (per-item indices restart per
# shard → each log joined independently); pass 2+ runs' logs to average out the noise.
#
#   R=ProjectTorreyPines/FastInterpolations.jl
#   # windows shard logs of a green run (repeat with 'ubuntu' for the default table):
#   for id in $(gh run view <RUN_ID> -R $R --json jobs \
#       --jq '.jobs[]|select(.name|test("1.x - windows.*\\[")).databaseId'); do
#     gh api repos/$R/actions/jobs/$id/logs > win_$id.log; done
#   julia scripts/gen_shard_weights.jl --out test/shard_weights_windows.toml win_*.log
#   # DEFAULT: same with ubuntu(+windows) logs and no --out.
#
# Find <RUN_ID> from a recent green run:  gh run list --workflow=CI.yml -R $R

const ROOT = dirname(@__DIR__)
const TESTDIR = joinpath(ROOT, "test")
const DEFAULT_OUT = joinpath(TESTDIR, "shard_weights.toml")

const RE_ANSI = r"\e\[[0-9;]*m"
const RE_START = r"START \(\s*(\d+)/\d+\).* at test/(\S+?\.jl):\d+"
const RE_DONE = r"DONE\s+\(\s*(\d+)/\d+\).*\"\s+<?([\d.]+) secs"   # <? : the '<0.1 secs' fast items

# Join one log's START (idx→file) and DONE (idx→secs) by item index, adding each file's
# seconds into `weight`. Per-item indices restart per shard, so EACH log must be joined
# independently (never cat shard logs). Returns (items_started, items_matched).
function parse_log!(weight, io)
    idx_file = Dict{Int, String}()
    idx_secs = Dict{Int, Float64}()
    for raw in eachline(io)
        line = replace(raw, RE_ANSI => "")
        if (m = match(RE_START, line)) !== nothing
            idx_file[parse(Int, m[1])] = m[2]
        elseif (m = match(RE_DONE, line)) !== nothing
            idx_secs[parse(Int, m[1])] = parse(Float64, m[2])
        end
    end
    n = 0
    for (idx, file) in idx_file
        haskey(idx_secs, idx) || continue
        weight[file] = get(weight, file, 0.0) + idx_secs[idx]
        n += 1
    end
    return length(idx_file), n
end

# every test_*.jl on disk gets an entry (0 if it had no items in these logs, so the runner
# treats it as weightless rather than median-guessing a phantom weight).
weight = Dict{String, Float64}(
    basename(p) => 0.0 for p in readdir(TESTDIR; join = true)
        if startswith(basename(p), "test_") && endswith(p, ".jl")
)

# One or more logs. A SHARDED run splits the files across shards, so a full refresh must
# pass BOTH shard logs of a run — they are merged here. "-"/none reads stdin.
# optional `--out <path>` selects the output table (default = shard_weights.toml, the
# ubuntu/macOS table; windows uses shard_weights_windows.toml). The rest are log files.
_args = collect(String, ARGS)
_oi = findfirst(==("--out"), _args)
OUT = _oi === nothing ? DEFAULT_OUT : _args[_oi + 1]
_oi === nothing || deleteat!(_args, _oi:(_oi + 1))
logs = isempty(_args) ? ["-"] : _args
started = matched = 0
for logpath in logs
    io = logpath == "-" ? stdin : open(logpath)
    try
        s, n = parse_log!(weight, io)
        global started += s
        global matched += n
    finally
        logpath == "-" || close(io)
    end
end
started == 0 && error("no ReTestItems 'START (…) at test/….jl' lines found — is this a parallel (RETESTITEMS_NWORKERS>0) run log?")

open(OUT, "w") do io
    println(io, "# Per-file test wall-time weights (seconds), summed across a file's testitems.")
    println(io, "# Per-OS (runner picks by Sys.iswindows): shard_weights.toml = DEFAULT for")
    println(io, "# ubuntu/macOS (low variance); shard_weights_windows.toml = windows (high per-")
    println(io, "# file variance + its own heavy files → its own ranking must drive the split).")
    println(io, "# Used by test/runtests_parallel.jl RETESTITEMS_SHARD=i/N to LPT-balance shards.")
    println(io, "# Regenerate: julia scripts/gen_shard_weights.jl <shard1.log> <shard2.log>.")
    println(io)
    println(io, "[weights]")
    for (fn, w) in sort(collect(weight); by = kv -> (-kv[2], kv[1]))
        println(io, "\"$fn\" = $(round(w; digits = 1))")
    end
end

total = sum(values(weight))
measured = count(>(0), values(weight))
println("parsed: $started items started, $matched joined, across $(length(logs)) log(s)")
println("files: $(length(weight)) on disk, $measured with measured time")
println("total wall time: $(round(total; digits = 1))s = $(round(total / 60; digits = 1)) min (summed across workers)")
println("wrote $OUT")
