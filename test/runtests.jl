# Main test entry point — TestItemRunner auto-discovers all @testitem
# files under test/. The 36 test/ext/ files use legacy @testset and run
# in a separate Julia process to isolate AD/ChainRulesCore loading order.

using Test
using TestItemRunner

# ── Extension tests via ARGS shortcut ────────────────────────────────
# `Pkg.test(test_args=["ext/runtests.jl"])` and
# `cc-julia-test-runner . ext/runtests.jl` route straight to the legacy
# @testset extension runner without invoking the @testitem dispatcher
# (no testitem matches that path, so the filter would otherwise leave
# the run silently empty).
if "ext/runtests.jl" in ARGS
    include("ext/runtests.jl")
else
    # ── Auto-discovered @testitem files ──────────────────────────────
    # Allocation thresholds and AAP_RUNTIME_CHECK live in @testsnippet
    # AllocConstants (test/setup.jl); each @testitem opts in via setup=[AllocConstants].
    # Extension tests in test/ext/ are self-contained (see test/ext/runtests.jl).
    # ARGS-based filter: pass testitem name OR filename substring.
    # Examples:
    #   cc-julia-test-runner . cubic                # all testitems matching "cubic" in name or filename
    #   cc-julia-test-runner . test_grid_spacing    # by filename
    #   cc-julia-test-runner . "Cubic Adjoint"      # by testitem name
    @run_package_tests verbose = true filter = ti -> begin
        isempty(ARGS) && return true
        return any(arg -> occursin(arg, ti.name) || occursin(arg, ti.filename), ARGS)
    end

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
