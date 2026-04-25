# Main test entry point — TestItemRunner auto-discovers all @testitem
# files under test/. The 36 test/ext/ files use legacy @testset and run
# in a separate Julia process to isolate AD/ChainRulesCore loading order.

using Test
using TestItemRunner
using FastInterpolations

# AdaptiveArrayPools RUNTIME_CHECK (debug mode flag)
# These constants are needed by legacy @testset files in test/ext/.
# Migrated @testitem files get the same constants via @testsnippet
# AllocConstants (defined in test/setup.jl).
const AAP_RUNTIME_CHECK = FastInterpolations.AdaptiveArrayPools.RUNTIME_CHECK
const _COV_OVERHEAD = 16
const ALLOC_THRESHOLD = (VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240) + _COV_OVERHEAD
const ND_ALLOC_THRESHOLD = (VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240) + _COV_OVERHEAD

@info "Running tests with ALLOC_THRESHOLD = $ALLOC_THRESHOLD bytes (AAP_RUNTIME_CHECK = $AAP_RUNTIME_CHECK)"

# ── Auto-discovered @testitem files ──────────────────────────────────
# ARGS-based filter: pass testitem name OR filename substring.
# Examples:
#   cc-julia-test-runner . cubic                # all testitems matching "cubic" in name or filename
#   cc-julia-test-runner . test_grid_spacing    # by filename
#   cc-julia-test-runner . "Cubic Adjoint"      # by testitem name
@run_package_tests verbose = true filter = ti -> begin
    isempty(ARGS) && return true
    return any(arg -> occursin(arg, ti.name) || occursin(arg, ti.filename), ARGS)
end

# ── Extension tests (AD / Symbolics / Recipes) ───────────────────────
# Still legacy @testset. Run in separate process to avoid
# ChainRulesCore contamination from Interpolations.jl. Only included
# on full local/CI runs (skipped when ARGS filter is active).
if isempty(ARGS)
    if get(ENV, "CI", nothing) !== nothing && get(ENV, "SKIP_EXTENSIONS", nothing) === nothing
        include("ext/runtests.jl")
    elseif get(ENV, "CI", nothing) === nothing
        @info "Skipping extension tests (run via: julia --project=test test/ext/runtests.jl)"
    end
end
