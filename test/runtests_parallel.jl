# runtests_parallel.jl — run the FastInterpolations @testitem suite in PARALLEL via
# ReTestItems, WITHOUT migrating the repo. Nothing under test/ is renamed or edited,
# so the VS Code "Julia test" integration (TestItemRunner) keeps working as before.
#
# How it works (all transient — the shadow dir is git-ignored and removed on exit):
#   1. `test_*.jl`  ->  symlinked as `*_tests.jl`   (ReTestItems only discovers the
#                                                    *_test(s).jl suffix, not our prefix)
#   2. every `@testsnippet Name begin … end`  ->  `@testsetup module Name … end`
#      (ReTestItems has no @testsnippet; it only knows @testsetup modules)
#   3. ReTestItems.runtests(...) runs with dynamic work-stealing across `nworkers`
#      worker processes, then the shadow dir is removed.
#
# ── Canonical use: through Pkg.test (recommended) ────────────────────────────
# test/runtests.jl routes here when RETESTITEMS_NWORKERS > 0, so the usual entry
# points just work — Pkg.test provides the sandbox (this checkout auto-dev'd, so
# the Julia-1.10 `[sources]` pitfall is moot), --check-bounds=yes, and
# package-scoped coverage, exactly like the sequential CI:
#     RETESTITEMS_NWORKERS=4 julia --project -e 'using Pkg; Pkg.test()'
#     RETESTITEMS_NWORKERS=2 cc-julia-test-runner . linear
#     RETESTITEMS_NWORKERS=2 cc-julia-test-runner . "Store Policy"       # testitem name
#     RETESTITEMS_NWORKERS=2 cc-julia-test-runner . "re:^Cubic .* Anchor" # regex (re: prefix —
#                                       # the --regex flag can't pass the bash wrapper)
#
# ── Standalone use: direct invocation (fast local iteration) ─────────────────
# Skips the Pkg.test sandbox; the test env must be instantiated once with the
# package dev-pinned (a guard below fails loud if not):
#     julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
# Cross-version note: test manifests are per-Julia-version (`Manifest-vX.Y.toml`),
# so 1.12 ↔ lts switching can't clobber. The CANONICAL path needs no manifest at
# all (the sandbox resolves fresh) — prefer it on non-primary versions: a
# dev-pinned manifest for a 1.10 version makes that version's Pkg.test unable to
# merge ("can not merge projects").
#
#     julia test/runtests_parallel.jl linear                        # linear files, 2 workers
#     julia test/runtests_parallel.jl cubic --nworkers 4
#     julia test/runtests_parallel.jl linear --nworkers 0           # sequential baseline (compare!)
#     julia test/runtests_parallel.jl "Store Policy"                # testitem NAMES match too
#     julia test/runtests_parallel.jl --regex '^Cubic .* Adjoint'   # regex, name or filename
#     julia --check-bounds=yes test/runtests_parallel.jl linear     # CI-exact codegen (see below)
#
# Patterns match the FILENAME or the @testitem NAME — same semantics as runtests.jl's
# ARGS filter (`cc-julia-test-runner . "Cubic Adjoint"`). Plain patterns (positional or
# --keyword) are case-insensitive substrings; --regex takes a Julia regex. A bare "."
# is ignored (cc-julia-test-runner muscle memory).
# NOTE: parallelism pays off on LARGE runs (full suite / CI). On a small keyword subset
# dominated by a few heavy-compile items, workers can't share compilation, so
# `--nworkers 0` (or plain cc-julia-test-runner) is often faster locally.
# CI parity: `Pkg.test` always runs `--check-bounds=yes`, which pins the FP codegen the
# suite's tight ULP tolerances were calibrated against — standalone runs wanting
# CI-exact results must pass it explicitly. Workers inherit flags via Base.julia_cmd().

const FI = normpath(joinpath(@__DIR__, ".."))
const REALTEST = joinpath(FI, "test")
const SHADOW = joinpath(FI, "_partest_shadow")   # in-repo (git-ignored); NOT dot-prefixed

# Standalone = launched as the program (`julia test/runtests_parallel.jl ...`).
# Canonical = include()d by test/runtests.jl under Pkg.test — the sandbox is already
# active with the package dev'd and all test deps present, so no bootstrap needed.
const STANDALONE = !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(@__FILE__)

if STANDALONE
    # Pkg/TOML load from @stdlib in a normal session; under the Pkg.test sandbox
    # (canonical mode) they are NOT loadable — and not needed.
    import Pkg
    import TOML

    # Activate the test env — it provides the package's test deps AND ReTestItems (a
    # test-only dependency, invisible to users who install the package).
    Pkg.activate(REALTEST; io = devnull)

    # Guard: the test env must resolve FastInterpolations to THIS checkout, not the
    # registered release. `[sources]` pins it on Julia 1.11+, but LTS (1.10) ignores
    # that section and falls back to the registry — silently testing the wrong code
    # (UndefVarError on unreleased internals, stale behavior). Fail loud instead.
    # Manifests are per-Julia-version (`Manifest-vX.Y.toml`, Julia 1.10.8+): each
    # version resolves its own, so switching 1.12 ↔ lts can't clobber or conflict.
    let mfv = joinpath(REALTEST, "Manifest-v$(VERSION.major).$(VERSION.minor).toml"),
            mf = isfile(mfv) ? mfv : joinpath(REALTEST, "Manifest.toml")

        entry = nothing
        if isfile(mf)
            deps = get(TOML.parsefile(mf), "deps", Dict{String, Any}())
            e = get(deps, "FastInterpolations", nothing)
            entry = e isa Vector ? first(e) : e
        end
        if !(entry isa AbstractDict && haskey(entry, "path"))
            error(
                """
                test/$(basename(mf)) does not pin FastInterpolations to this checkout
                (no `path` entry — on Julia $(VERSION) the `[sources]` section may be
                unsupported, so Pkg resolved the REGISTERED release instead). Fix once with:
                    julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
                (Or skip standalone mode on this Julia version and use the canonical path,
                which needs no manifest: RETESTITEMS_NWORKERS=2 cc-julia-test-runner .)
                """
            )
        end
    end
end

try
    @eval using ReTestItems
catch
    error(
        """
        ReTestItems not found in the active environment. Instantiate the test env once:
            julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
        """
    )
end

# ── args (in a function so assignments aren't trapped in a hard local scope) ──
# Returns (patterns, nworkers). Patterns: String = case-insensitive substring,
# Regex = as-is; matched against filenames AND @testitem names, OR-combined.
# nworkers is `nothing` unless --nworkers was passed (mode default applied in main).
function parse_args(args)
    patterns = Union{String, Regex}[]
    nworkers = nothing
    i = firstindex(args)
    while i <= lastindex(args)
        a = args[i]
        if a in ("--keyword", "-k")
            i < lastindex(args) || error("--keyword needs a value")
            push!(patterns, String(args[i + 1])); i += 2
        elseif a in ("--regex", "-r")
            i < lastindex(args) || error("--regex needs a value")
            push!(patterns, Regex(args[i + 1])); i += 2
        elseif a in ("--nworkers", "-n")
            i < lastindex(args) || error("--nworkers needs a value")
            nworkers = parse(Int, args[i + 1]); i += 2
        elseif a == "."
            i += 1                       # cc-julia-test-runner habit: project-path arg
        elseif startswith(a, "re:")
            # positional regex — flags like --regex can't pass through cc-julia-test-runner
            # (its bash parser rejects unknown options), a `re:` prefix survives any wrapper
            push!(patterns, Regex(chopprefix(a, "re:"))); i += 1
        elseif startswith(a, "-")
            error("unknown flag: $(repr(a))  (use PATTERN / re:REGEX / --keyword KW / --regex RX / --nworkers N)")
        else
            push!(patterns, String(a)); i += 1   # positional pattern
        end
    end
    return patterns, nworkers
end

_matches(p::String, s) = occursin(lowercase(p), lowercase(s))
_matches(p::Regex, s) = occursin(p, s)
_matches_any(patterns, s) = any(p -> _matches(p, s), patterns)

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
    names = String[]                    # literal @testitem names (for pattern matching)
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
                if mname === Symbol("@testitem") && length(ex.args) >= 3 && ex.args[3] isa String
                    push!(names, ex.args[3])
                end
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
    return src, snips, cuts, names
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
# Exception: under --code-coverage the runtime writes .cov files for shadow-loaded
# code AT PROCESS EXIT — after this `finally` — so the dir must outlive us then.
# It is gitignored and recreated fresh on the next run.
function with_shadow(f, dir)
    isdir(dir) && rm(dir; recursive = true, force = true)
    mkpath(dir)
    try
        return f(dir)
    finally
        if Base.JLOptions().code_coverage == 0
            rm(dir; recursive = true, force = true)
        else
            println("(kept $(basename(dir))/ for the exit-time coverage writeout — gitignored)")
        end
    end
end

# Stream ReTestItems' output through a rewrite that maps shadow paths back to the real
# test/ files, so failures point at the original source (line numbers already match).
function run_rewriting(f)
    prev_out, prev_err = stdout, stderr
    # <shadow>/X_tests.jl -> test/X.jl, matching absolute and relative occurrences with
    # either path separator (Windows prints backslashes). Keep the literal dir name in
    # sync with SHADOW's basename ("_partest_shadow").
    rx = r"(^|[/\\\s])_partest_shadow[/\\]([^\s/\\:]+?)_tests\.jl"
    pipe = Pipe()
    Base.link_pipe!(pipe; reader_supports_async = true, writer_supports_async = true)
    redirect_stdout(pipe.in)
    redirect_stderr(pipe.in)
    reader = @async for line in eachline(pipe.out)
        if occursin("_partest_shadow", line)
            line = replace(
                line,
                "_partest_shadow/aa_wrapper_testsetup.jl" => "test/setup.jl",
                "_partest_shadow\\aa_wrapper_testsetup.jl" => "test/setup.jl",
            )
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
function build_and_run(shadow, patterns, nworkers)
    # Scan every test/ file once: collect @testsnippet defs (→ @testsetup modules), per-file
    # blank ranges (non-@testitem/@testsetup top-level exprs), and @testitem names.
    scanned = Dict{String, Tuple{String, Vector{Tuple{Int, Int}}, Vector{String}}}()
    snippet_defs = String[]
    for f in readdir(REALTEST; join = true)
        endswith(f, ".jl") || continue
        src, snips, cuts, names = scan_file(f)
        scanned[f] = (src, cuts, names)
        for (nm, body) in snips
            push!(snippet_defs, to_testsetup(nm, body))
        end
    end
    write(joinpath(shadow, "aa_wrapper_testsetup.jl"), join(snippet_defs, "\n\n"))

    # Select test_*.jl whose FILENAME or any @testitem NAME matches a pattern (parity with
    # runtests.jl's ARGS filter); shadow each as *_tests.jl. `allowed` collects the item
    # names to run: all of them for filename-matched files, the matching ones otherwise.
    # (An item whose name is computed rather than a literal can only be selected via its
    # filename.) Files with only @testitem/@testsetup are symlinked; those with other
    # top-level exprs get a copy with those ranges blanked (line numbers preserved).
    selected = String[]
    allowed = Set{String}()
    for f in sort(readdir(REALTEST))
        (startswith(f, "test_") && endswith(f, ".jl")) || continue
        real = joinpath(REALTEST, f)
        src, cuts, names = scanned[real]
        file_hit = isempty(patterns) || _matches_any(patterns, f)
        hits = file_hit ? names : Base.filter(n -> _matches_any(patterns, n), names)
        (file_hit || !isempty(hits)) || continue
        dst = joinpath(shadow, replace(f, r"\.jl$" => "") * "_tests.jl")
        if isempty(cuts)
            try
                symlink(real, dst)
            catch                    # Windows / filesystems without symlink privilege
                cp(real, dst)
            end
        else
            buf = src
            for (a, b) in sort(cuts; by = first, rev = true)
                buf = blank_range(buf, a, b)
            end
            write(dst, buf)
        end
        push!(selected, dst)
        union!(allowed, hits)
    end
    isempty(selected) && error("no test files or testitem names matched $(join(repr.(patterns), ", "))")
    ti_filter = isempty(patterns) ? Returns(true) : (ti -> ti.name in allowed)

    # Set :SOURCE_PATH = the package's runtests.jl so ReTestItems takes its "running
    # under Pkg.test" path (skips TestEnv.activate) and uses the already-active env.
    # Canonical mode: truthful — we ARE one include() below runtests.jl (include just
    # re-pointed the tls to this file). Standalone: the same spoof opts into that path.
    task_local_storage(:SOURCE_PATH, joinpath(REALTEST, "runtests.jl"))
    setupfile = joinpath(shadow, "aa_wrapper_testsetup.jl")

    sel = isempty(patterns) ? "all" : join(repr.(patterns), ", ") * " → $(length(allowed)) item(s)"
    println("── partest ─ files=$(length(selected))  nworkers=$nworkers  patterns: $sel ──")
    flush(stdout)
    # Rewrite shadow paths → real test/ paths in ReTestItems' streamed output.
    t = run_rewriting() do
        @elapsed runtests(
            ti_filter, setupfile, selected...;
            nworkers = nworkers,
            nworker_threads = 1,
            verbose_results = false,
            report = false,
            logs = :issues,
        )
    end
    return println("\n>>> partest done: $(length(selected)) files · nworkers=$nworkers · $(round(t, digits = 2)) s")
end

function main()
    patterns, nworkers_cli = parse_args(ARGS)
    # --nworkers wins; otherwise canonical mode follows RETESTITEMS_NWORKERS (what
    # routed runtests.jl here) and standalone defaults to 2.
    nworkers = something(
        nworkers_cli,
        STANDALONE ? 2 : parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "2"))
    )
    return with_shadow(SHADOW) do shadow
        build_and_run(shadow, patterns, nworkers)
    end
end

main()
