#!/usr/bin/env julia
#
# runtests_parallel.jl — run the FastInterpolations test suite in PARALLEL via ReTestItems,
# WITHOUT migrating the repo. Nothing under test/ is renamed or edited, so the
# VS Code "Julia test" integration (TestItemRunner) keeps working exactly as before.
#
# How it works (all transient — the shadow dir is git-ignored and removed on exit):
#   1. `test_*.jl`  ->  symlinked as `*_tests.jl`   (ReTestItems only discovers the
#                                                    *_test(s).jl suffix, not our prefix)
#   2. every `@testsnippet Name begin … end`  ->  `@testsetup module Name … end`
#      (ReTestItems has no @testsnippet; it only knows @testsetup modules)
#   3. ReTestItems.runtests(...) runs on its "clean" path (dynamic work-stealing
#      across `nworkers` worker processes), then the shadow dir is removed.
#
# Prereq (once): instantiate the test env — ReTestItems is a test-only dependency there:
#     julia --project=test -e 'using Pkg; Pkg.instantiate()'
#
# Usage (from the repo root):
#     julia test/runtests_parallel.jl --keyword linear              # linear files, 2 workers
#     julia test/runtests_parallel.jl --keyword cubic  --nworkers 4
#     julia test/runtests_parallel.jl --keyword linear --nworkers 0 # sequential baseline (compare!)
#     julia test/runtests_parallel.jl                               # whole suite, 2 workers
#     julia --code-coverage=user test/runtests_parallel.jl          # + coverage (.cov in src/; merge as usual)
#
# --keyword matches the file NAME (like `cc-julia-test-runner . linear`, case-insensitive).
# NOTE: parallelism pays off on LARGE runs (full suite / CI). On a small keyword subset
# dominated by a few heavy-compile items, workers can't share compilation, so
# `--nworkers 0` (or plain cc-julia-test-runner) is often faster locally.

import Pkg

const FI = normpath(joinpath(@__DIR__, ".."))
const REALTEST = joinpath(FI, "test")
const SHADOW = joinpath(FI, "_partest_shadow")   # in-repo (git-ignored); NOT dot-prefixed

# Activate the test env — it provides the package's test deps AND ReTestItems (a
# test-only dependency, invisible to users who install the package).
Pkg.activate(REALTEST; io = devnull)
try
    @eval using ReTestItems
catch
    error("""
    ReTestItems not found in the test environment. Instantiate it once:
        julia --project=test -e 'using Pkg; Pkg.instantiate()'
    """)
end

# ── args (in a function so assignments aren't trapped in a hard local scope) ──
function parse_args(args)
    keyword = ""
    nworkers = 2
    i = firstindex(args)
    while i <= lastindex(args)
        a = args[i]
        if a in ("--keyword", "-k")
            i < lastindex(args) || error("--keyword needs a value")
            keyword = args[i + 1]; i += 2
        elseif a in ("--nworkers", "-n")
            i < lastindex(args) || error("--nworkers needs a value")
            nworkers = parse(Int, args[i + 1]); i += 2
        else
            error("unknown arg: $(repr(a))  (use --keyword KW / --nworkers N)")
        end
    end
    return keyword, nworkers
end

# ── file scan: keep @testitem/@testsetup, blank everything else ──────────────
# ReTestItems requires files to contain ONLY @testitem/@testsetup. Walk a file's
# top-level expressions and (a) capture each `@testsnippet Name begin … end` to convert
# into a @testsetup module, and (b) record the char range of EVERY non-@testitem/@testsetup
# top-level expr (docstrings, stray `using`/`const`, @testsnippet, …) so the shadow copy
# can blank them (line numbers preserved). This is safe: testitems run in isolated modules
# and never saw that top-level code, so blanking it changes no behavior.
function scan_file(path)
    src = read(path, String)
    snips = Tuple{String, String}[]     # @testsnippet (name, body-source) to convert
    cuts = Tuple{Int, Int}[]            # ranges to blank in the shadow copy
    pos = firstindex(src)
    while true
        ex, np = Meta.parse(src, pos; raise = false)
        (ex === nothing || np <= pos) && break     # nothing left / no progress → stop
        stop = prevind(src, np)
        keep = false
        if ex isa Expr && ex.head === :macrocall
            mname = ex.args[1]
            if mname === Symbol("@testitem") || mname === Symbol("@testsetup")
                keep = true
            elseif mname === Symbol("@testsnippet")
                chunk = src[pos:stop]
                b = findfirst("begin", chunk)
                e = findlast("end", chunk)
                if b !== nothing && e !== nothing
                    name = string(ex.args[3])
                    body = strip(chunk[nextind(chunk, last(b)):prevind(chunk, first(e))])
                    push!(snips, (name, String(body)))
                end
            end
        end
        keep || push!(cuts, (pos, stop))   # blank anything that isn't @testitem/@testsetup
        np > lastindex(src) && break
        pos = np
    end
    return src, snips, cuts
end

# A @testsetup module needs its OWN imports (unlike @testsnippet, which inlines into
# a testitem that already has `using FastInterpolations`). Add it, then auto-export
# every binding so `using` makes them available unqualified — reproducing the flat
# name scope @testsnippet gives.
function to_testsetup(name, body)
    shim = """
        for var"#en" in names(@__MODULE__; all = true)
            startswith(string(var"#en"), "#") && continue
            var"#en" in (:eval, :include) && continue
            try
                Core.eval(@__MODULE__, Expr(:export, var"#en"))
            catch
            end
        end
    """
    return "@testsetup module $name\n    using FastInterpolations\n$body\n$shim\nend\n"
end

# Replace a char range with the SAME number of newlines, so every following line keeps
# its original line number (failures point at the right line even in excised files).
function blank_range(s, a, b)
    nl = count(==('\n'), s[a:b])
    return s[1:prevind(s, a)] * ("\n"^nl) * s[nextind(s, b):end]
end

# Run `f(shadow)` in a freshly-created shadow dir, guaranteeing removal afterwards.
function with_shadow(f, dir)
    isdir(dir) && rm(dir; recursive = true, force = true)
    mkpath(dir)
    try
        return f(dir)
    finally
        rm(dir; recursive = true, force = true)
    end
end

# Stream ReTestItems' output through a rewrite that maps shadow paths back to the real
# test/ files, so failures point at the original source (line numbers already match).
function run_rewriting(f)
    prev_out, prev_err = stdout, stderr
    # <shadow>/X_tests.jl -> test/X.jl, matching both absolute and relative occurrences.
    # Keep the literal dir name in sync with SHADOW's basename ("_partest_shadow").
    rx = r"(^|[/\s])_partest_shadow/([^\s/:]+?)_tests\.jl"
    pipe = Pipe()
    Base.link_pipe!(pipe; reader_supports_async = true, writer_supports_async = true)
    redirect_stdout(pipe.in)
    redirect_stderr(pipe.in)
    reader = @async for line in eachline(pipe.out)
        if occursin("_partest_shadow", line)
            line = replace(line, "_partest_shadow/aa_wrapper_testsetup.jl" => "test/setup.jl")
            line = replace(line, rx => s"\1test/\2.jl")
            line = replace(line, "_partest_shadow" => "test")   # bare dir name (e.g. summary group)
        end
        println(prev_out, line)
        flush(prev_out)        # stream progress live — CI stdout is block-buffered (non-TTY)
    end
    try
        return f()
    finally
        flush(pipe.in)
        redirect_stdout(prev_out)
        redirect_stderr(prev_err)
        close(pipe.in)
        wait(reader)
        close(pipe.out)
    end
end

# Populate `shadow`, then run the selected tests through ReTestItems.
function build_and_run(shadow, keyword, nworkers)
    # Scan every test/ file once: collect @testsnippet defs (→ @testsetup modules) and,
    # per file, the ranges of non-@testitem/@testsetup top-level exprs to blank.
    scanned = Dict{String, Tuple{String, Vector{Tuple{Int, Int}}}}()   # path => (src, cuts)
    snippet_defs = String[]
    for f in readdir(REALTEST; join = true)
        endswith(f, ".jl") || continue
        src, snips, cuts = scan_file(f)
        scanned[f] = (src, cuts)
        for (nm, body) in snips
            push!(snippet_defs, to_testsetup(nm, body))
        end
    end
    write(joinpath(shadow, "aa_wrapper_testsetup.jl"), join(snippet_defs, "\n\n"))

    # Select test_*.jl by filename keyword; shadow each as *_tests.jl. Files with only
    # @testitem/@testsetup are symlinked; those with other top-level exprs get a copy with
    # those ranges blanked (line numbers preserved).
    selected = String[]
    for f in sort(readdir(REALTEST))
        (startswith(f, "test_") && endswith(f, ".jl")) || continue
        isempty(keyword) || occursin(lowercase(keyword), lowercase(f)) || continue
        real = joinpath(REALTEST, f)
        dst = joinpath(shadow, replace(f, r"\.jl$" => "") * "_tests.jl")
        src, cuts = scanned[real]
        if isempty(cuts)
            symlink(real, dst)
        else
            buf = src
            for (a, b) in sort(cuts; by = first, rev = true)
                buf = blank_range(buf, a, b)
            end
            write(dst, buf)
        end
        push!(selected, dst)
    end
    isempty(selected) && error("no test files matched keyword $(repr(keyword))")

    # Spoof :SOURCE_PATH = the package's real runtests.jl so ReTestItems takes its clean
    # path (skips TestEnv.activate) and uses the already-active test env. ReTestItems walks
    # up from the shadow files to FI/Project.toml, so `testfile(FI)` = test/runtests.jl.
    task_local_storage(:SOURCE_PATH, joinpath(REALTEST, "runtests.jl"))
    setupfile = joinpath(shadow, "aa_wrapper_testsetup.jl")

    println("── partest ─ files=$(length(selected))  nworkers=$nworkers  keyword=$(repr(keyword)) ──")
    flush(stdout)
    # Rewrite shadow paths → real test/ paths in ReTestItems' streamed output.
    t = run_rewriting() do
        @elapsed runtests(
            setupfile, selected...;
            nworkers = nworkers,
            nworker_threads = 1,
            verbose_results = false,
            report = false,
            logs = :issues,
        )
    end
    println("\n>>> partest done: $(length(selected)) files · nworkers=$nworkers · $(round(t, digits = 2)) s")
end

function main()
    keyword, nworkers = parse_args(ARGS)
    with_shadow(SHADOW) do shadow
        build_and_run(shadow, keyword, nworkers)
    end
end

main()
