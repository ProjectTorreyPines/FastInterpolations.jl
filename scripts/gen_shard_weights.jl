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
# regenerate only occasionally (e.g. after adding a heavy test file). Prefer the 1.x
# windows log: it is the binding-constraint platform the split is tuned for.
#
#   gh run view --job <JOB_ID> --log -R ProjectTorreyPines/FastInterpolations.jl > run.log
#   julia scripts/gen_shard_weights.jl run.log        # (or: … --log | julia scripts/gen_shard_weights.jl)
#
# Pick <JOB_ID> from a recent green "Julia 1.x - windows-latest" job:
#   gh run list --workflow=CI.yml -R ProjectTorreyPines/FastInterpolations.jl

const ROOT = dirname(@__DIR__)
const TESTDIR = joinpath(ROOT, "test")
const OUT = joinpath(TESTDIR, "shard_weights.toml")

const RE_ANSI = r"\e\[[0-9;]*m"
const RE_START = r"START \(\s*(\d+)/\d+\).* at test/(\S+?\.jl):\d+"
const RE_DONE = r"DONE\s+\(\s*(\d+)/\d+\).*\"\s+<?([\d.]+) secs"   # <? : the '<0.1 secs' fast items

# stream lines from ARGS[1] (a saved log) or stdin ("-"/none) so the gh output can pipe in
lines() = (isempty(ARGS) || ARGS[1] == "-") ? eachline(stdin) : eachline(ARGS[1])

idx_file = Dict{Int, String}()
idx_secs = Dict{Int, Float64}()
for raw in lines()
    line = replace(raw, RE_ANSI => "")
    if (m = match(RE_START, line)) !== nothing
        idx_file[parse(Int, m[1])] = m[2]
    elseif (m = match(RE_DONE, line)) !== nothing
        idx_secs[parse(Int, m[1])] = parse(Float64, m[2])
    end
end
isempty(idx_file) && error("no ReTestItems 'START (…) at test/….jl' lines found — is this a parallel (RETESTITEMS_NWORKERS>0) run log?")

# sum seconds per file; every test_*.jl on disk gets an entry (0 if it had no items this
# run, so the runner treats it as weightless rather than median-guessing a phantom weight).
weight = Dict{String, Float64}(
    basename(p) => 0.0 for p in readdir(TESTDIR; join = true)
        if startswith(basename(p), "test_") && endswith(p, ".jl")
)
matched = 0
for (idx, file) in idx_file
    haskey(idx_secs, idx) || continue
    weight[file] = get(weight, file, 0.0) + idx_secs[idx]
    global matched += 1
end

open(OUT, "w") do io
    println(io, "# Per-file test wall-time weights (seconds), summed across a file's testitems.")
    println(io, "# Source: ReTestItems CI log, Julia 1.x windows-latest (binding-constraint platform).")
    println(io, "# Used by test/runtests_parallel.jl RETESTITEMS_SHARD=i/N to LPT-balance shards.")
    println(io, "# Regenerate: julia scripts/gen_shard_weights.jl <log>  (see header for the gh one-liner).")
    println(io)
    println(io, "[weights]")
    for (fn, w) in sort(collect(weight); by = kv -> (-kv[2], kv[1]))
        println(io, "\"$fn\" = $(round(w; digits = 1))")
    end
end

total = sum(values(weight))
measured = count(>(0), values(weight))
println("parsed items: START=$(length(idx_file)) DONE=$(length(idx_secs)) joined=$matched")
println("files: $(length(weight)) on disk, $measured with measured time")
println("total wall time: $(round(total; digits = 1))s = $(round(total / 60; digits = 1)) min (summed across workers)")
println("wrote $OUT")
