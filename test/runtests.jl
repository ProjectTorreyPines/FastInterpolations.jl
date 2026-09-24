# Main test entry point — TestItemRunner auto-discovers all @testitem
# files under test/. The 36 test/ext/ files use legacy @testset and run
# in a separate Julia process to isolate AD/ChainRulesCore loading order.

using Test

# ── Extension tests via ARGS shortcut ────────────────────────────────
# `Pkg.test(test_args=["ext/runtests.jl"])` and
# `cc-julia-test-runner . ext/runtests.jl` route straight to the legacy
# @testset extension runner without invoking the @testitem dispatcher
# (no testitem matches that path, so the filter would otherwise leave
# the run silently empty).
# ── Parallel opt-in (ReTestItems) ────────────────────────────────────
# RETESTITEMS_NWORKERS=N (N>0) routes the @testitem suite through ReTestItems across
# N worker processes — same ARGS filters, driven by the usual Pkg.test entry points
# (`RETESTITEMS_NWORKERS=4 cc-julia-test-runner .`). See test/runtests_parallel.jl:
# it runs a transient shadow copy of the suite; nothing under test/ changes and the
# TestItemRunner/VS Code path below is untouched. Extension tests keep their own path.
if parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "0")) > 0 && !("ext/runtests.jl" in ARGS)
    include("runtests_parallel.jl")
elseif "ext/runtests.jl" in ARGS
    include("ext/runtests.jl")
else
    # TestItemRunner is loaded HERE (not at the top): it and ReTestItems both export
    # `@testitem`, and importing both into Main makes the macro ambiguous — the
    # parallel branch above must be the only runner in its process. Consequence: the
    # suite is invoked through the function form (run_tests) instead of
    # @run_package_tests — a macro in this branch would be expanded when the whole
    # `if` lowers, i.e. before the `using` above has ever run.
    using TestItemRunner

    # ── Auto-discovered @testitem files ──────────────────────────────
    # Allocation thresholds and AAP_RUNTIME_CHECK live in @testsnippet
    # AllocConstants (test/setup.jl); each @testitem opts in via setup=[AllocConstants].
    # Extension tests in test/ext/ are self-contained (see test/ext/runtests.jl).
    # ARGS-based filter: pass testitem name OR filename substring, or a `re:` regex.
    # Examples:
    #   cc-julia-test-runner . cubic                # all testitems matching "cubic" in name or filename
    #   cc-julia-test-runner . test_grid_spacing    # by filename
    #   cc-julia-test-runner . "Cubic Adjoint"      # by testitem name
    #   cc-julia-test-runner . "re:^Cubic .* Anchor"  # regex on name or filename
    # NB: the PACKAGE ROOT, exactly what @run_package_tests expands to — run_tests
    # reads the package name from root Project.toml to auto-inject
    # `using FastInterpolations` into every testitem.
    TestItemRunner.run_tests(
        joinpath(@__DIR__, "..");
        verbose = true,
        filter = ti -> begin
            isempty(ARGS) && return true
            return any(ARGS) do arg
                p = startswith(arg, "re:") ? Regex(chopprefix(arg, "re:")) : arg
                return occursin(p, ti.name) || occursin(p, ti.filename)
            end
        end,
    )

    # ── Extension tests on default (no-ARGS) full runs ───────────────
    # Still legacy @testset. Run in separate process to avoid
    # ChainRulesCore contamination from Interpolations.jl.
    if isempty(ARGS)
        if get(ENV, "CI", nothing) !== nothing && get(ENV, "SKIP_EXTENSIONS", nothing) === nothing
            include("ext/runtests.jl")
        elseif get(ENV, "CI", nothing) === nothing
            @info "Skipping extension tests (run via: julia --project=test test/ext/runtests.jl)"
        end
    end
end
