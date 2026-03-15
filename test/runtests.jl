using Test
using FastInterpolations
using Random

# Julia 1.12+ achieves true zero-allocation via improved escape analysis.
# Older versions have small runtime overhead from mutable struct field access.
# Note: 4-way Val dispatch (extrap modes) increases overhead on older Julia (~160 bytes).
const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240

# ND oneshot dispatch has higher fixed overhead from tuple construction/resolution.
# This is O(1) overhead, not O(n), so a separate higher threshold is appropriate.
const ND_ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Default behavior: run all tests
    include("test_aqua.jl")
    include("test_grid_spacing.jl")
    include("test_search.jl")
    include("test_factory.jl")
    include("test_constant.jl")
    include("test_constextrap_fill.jl")
    include("test_complex_constant.jl")
    include("test_linear.jl")
    include("test_quadratic.jl")
    include("test_cubic.jl")
    include("test_cubic_autocache.jl")
    include("test_cubic_interpolant.jl")
    include("test_cubic_anchor.jl")
    include("test_cubic_adjoint.jl")
    include("test_linear_adjoint.jl")   # Linear 1D adjoint (W^T * y_bar)
    include("test_constant_adjoint.jl")  # Constant 1D adjoint (W^T * y_bar)
    include("test_quadratic_adjoint.jl") # Quadratic 1D adjoint (W^T * y_bar)
    include("test_linear_anchor.jl")
    include("test_constant_anchor.jl")
    include("test_quadratic_anchor.jl")
    include("test_cubic_series_interp.jl")
    include("test_linear_series_interp.jl")
    include("test_constant_series_interp.jl")
    include("test_quadratic_series_interp.jl")
    include("test_complex_linear_series.jl")
    include("test_complex_constant_series.jl")
    include("test_complex_quadratic_series.jl")
    include("test_complex_cubic_series.jl")
    include("test_series_range_grid.jl")
    include("test_series_wrapper.jl")
    include("test_series_utils.jl")
    include("test_allocation.jl")
    include("test_random_grid.jl")
    include("test_periodic_bc.jl")
    include("test_periodic_exclusive.jl")
    include("test_thomas_lu_solver.jl")
    include("test_generic_bc.jl")
    include("test_polyfit_bc.jl")
    include("test_bc_structure.jl")
    include("test_bc_complex_int.jl")
    include("test_type_stability.jl")
    include("test_mixed_precision_extrap.jl")
    include("test_derivatives.jl")
    include("test_packages_comparison.jl")
    include("test_thread_safety.jl")
    include("test_rcu.jl")
    include("test_nonuniform_grid.jl")
    include("test_show.jl")
    include("test_recipes.jl")
    include("test_mutation_safety.jl")

    # ND Interpolation
    include("test_nd_utils_shared.jl")  # Shared ND utilities (phase 1)
    include("test_nd_constant.jl")      # Constant ND interpolation (phase 2)
    include("test_nd_linear.jl")        # Linear ND interpolation (phase 3)
    include("test_nd_quadratic.jl")     # Quadratic ND interpolation
    include("test_cubic_nd.jl")
    include("test_cubic_nd_adjoint.jl")  # Cubic ND adjoint (W^T * y_bar)
    include("test_linear_nd_adjoint.jl") # Linear ND adjoint (W^T * y_bar)
    include("test_constant_nd_adjoint.jl") # Constant ND adjoint (W^T * y_bar)
    include("test_quadratic_nd_adjoint.jl") # Quadratic ND adjoint (W^T * y_bar)
    include("test_cubic_nd_oneshot.jl")  # Cubic ND one-shot (pool-based, zero-alloc)
    include("test_nd_noextrap_oob.jl")  # ND NoExtrap domain validation (all paths)
    include("test_nd_comprehensive.jl")
    include("test_nd_coverage.jl")
    include("test_nd_heterogeneous_grids.jl")
    include("test_nd_hint.jl")
    include("test_nd_oneshot_hint.jl")
    include("test_nd_autosearch_peraxis.jl")
    include("test_nd_batch_inplace.jl")
    include("test_gradient_hessian.jl")

    # Duck typing (custom value types)
    include("test_duck_typing_comprehensive.jl")

    # Coefficients API
    include("test_coeffs.jl")

    # Integration API
    include("test_integral_api.jl")
    include("test_integral_cubic_1d.jl")
    include("test_integral_1d.jl")
    include("test_integral_nd_cubic.jl")
    include("test_integral_nd.jl")
    include("test_integral_nd_exactness.jl")
    include("test_integral_extrap.jl")
    include("test_integral_allocation.jl")
    include("test_integral_series.jl")
    include("test_integral_fulldomain.jl")
    include("test_cumulative_integrate.jl")

    # ── Extension tests (AD / Symbolics) ──────────────────────────────
    # Heavy package loads (~5 min compile). Always in CI, skip locally by default.
    # Run individually via ARGS: Pkg.test(test_args=["test_autodiff_Zygote.jl"])
    # NOTE: Enzyme MUST run before Zygote. Zygote loads ChainRulesCore, which
    # triggers FastInterpolationsChainRulesCoreExt (rrule fallback). Enzyme then
    # silently uses the rrule instead of our custom EnzymeRules, leaving the
    # Enzyme extension with 0% body coverage.
    if get(ENV, "CI", nothing) !== nothing
        include("test_autodiff_Enzyme.jl")
        include("test_autodiff_ForwardDiff.jl")
        include("test_autodiff_Zygote.jl")
        include("test_symbolics.jl")
    else
        @info "Skipping extension tests (run in CI, or use cc-julia-test-runner for individual files)"
    end
end
