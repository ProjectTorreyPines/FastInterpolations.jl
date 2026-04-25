using Test
using TestItemRunner
using FastInterpolations
using Random

# ── Migrated @testitem files (auto-discovered by TestItemRunner) ─────
# As files are migrated from @testset → @testitem, they are removed
# from the legacy include cascade below. The set here is informational
# and used to skip these files in the ARGS-based legacy include path
# below (so `cc-julia-test-runner . test_grid_spacing.jl` doesn't try
# to include() a @testitem file).
const MIGRATED_TESTITEM_FILES = Set(
    [
        "test_abstract_types.jl",
        "test_akima_1d.jl",
        "test_allocation.jl",
        "test_anchor_common.jl",
        "test_aqua.jl",
        "test_bc_complex_int.jl",
        "test_bc_structure.jl",
        "test_cardinal_1d.jl",
        "test_coeffs.jl",
        "test_complex_constant.jl",
        "test_complex_constant_series.jl",
        "test_complex_cubic.jl",
        "test_complex_cubic_series.jl",
        "test_complex_linear.jl",
        "test_complex_linear_series.jl",
        "test_complex_quadratic.jl",
        "test_complex_quadratic_series.jl",
        "test_constant.jl",
        "test_constant_adjoint.jl",
        "test_constant_anchor.jl",
        "test_constant_nd_adjoint.jl",
        "test_constant_oneshot_series.jl",
        "test_constant_periodic.jl",
        "test_constant_series_interp.jl",
        "test_constextrap_fill.jl",
        "test_cubic.jl",
        "test_cubic_adjoint.jl",
        "test_cubic_anchor.jl",
        "test_cubic_autocache.jl",
        "test_cubic_interpolant.jl",
        "test_cubic_nd_adjoint.jl",
        "test_cubic_oneshot_series.jl",
        "test_cubic_series_interp.jl",
        "test_cubic_series_naming.jl",
        "test_cumulative_integrate.jl",
        "test_derivatives.jl",
        "test_factory.jl",
        "test_generic_bc.jl",
        "test_gradient_hessian.jl",
        "test_grid_spacing.jl",
        "test_hermite_1d.jl",
        "test_hermite_adjoint.jl",
        "test_hermite_nd_graceful_errors.jl",
        "test_hermite_onthefly.jl",
        "test_hetero_adjoint.jl",
        "test_hetero_nd.jl",
        "test_hetero_oneshot.jl",
        "test_hetero_precomputed.jl",
        "test_idx_stencil.jl",
        "test_inbounds_extrap.jl",
        "test_integral_1d.jl",
        "test_integral_allocation.jl",
        "test_integral_api.jl",
        "test_integral_cubic_1d.jl",
        "test_integral_extrap.jl",
        "test_integral_fulldomain.jl",
        "test_integral_nd.jl",
        "test_integral_nd_cubic.jl",
        "test_integral_nd_exactness.jl",
        "test_integral_series.jl",
        "test_linear.jl",
        "test_linear_adjoint.jl",
        "test_linear_anchor.jl",
        "test_linear_nd_adjoint.jl",
        "test_linear_oneshot_series.jl",
        "test_linear_series_interp.jl",
        "test_local_hermite_nd_forward.jl",
        "test_local_slope_comparison.jl",
        "test_mixed_precision_extrap.jl",
        "test_nd_autosearch_peraxis.jl",
        "test_nd_batch_hint_persistence.jl",
        "test_nd_batch_inplace.jl",
        "test_nd_comprehensive.jl",
        "test_nd_coverage.jl",
        "test_nd_heterogeneous_grids.jl",
        "test_nd_hint.jl",
        "test_nd_mixed_partial_bc_consistency.jl",
        "test_nd_oneshot_hint.jl",
        "test_nd_utils_shared.jl",
        "test_nodal_partials.jl",
        "test_packages_comparison.jl",
        "test_pchip_1d.jl",
        "test_periodic_bc.jl",
        "test_periodic_exclusive.jl",
        "test_periodic_resolvers.jl",
        "test_periodic_search_4value.jl",
        "test_polyfit_bc.jl",
        "test_promotion_alloc.jl",
        "test_quadratic.jl",
        "test_quadratic_adjoint.jl",
        "test_quadratic_anchor.jl",
        "test_quadratic_nd_adjoint.jl",
        "test_quadratic_oneshot_series.jl",
        "test_quadratic_series_interp.jl",
        "test_random_grid.jl",
        "test_rcu.jl",
        "test_search.jl",
        "test_search_anchor_integration.jl",
        "test_search_context_normalization.jl",
        "test_series_matrix.jl",
        "test_series_range_grid.jl",
        "test_series_utils.jl",
        "test_series_wrapper.jl",
        "test_show.jl",
        "test_thomas_lu_solver.jl",
        "test_type_stability.jl",
    ]
)

# AdaptiveArrayPools RUNTIME_CHECK (debug mode flag)
# These constants are needed by legacy @testset files include()d below.
# Migrated @testitem files get the same constants via @testsnippet AllocConstants
# (defined in test/setup.jl).
const AAP_RUNTIME_CHECK = FastInterpolations.AdaptiveArrayPools.RUNTIME_CHECK
const _COV_OVERHEAD = 16
const ALLOC_THRESHOLD = (VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240) + _COV_OVERHEAD
const ND_ALLOC_THRESHOLD = (VERSION >= v"1.12" ? 0 : (2 * AAP_RUNTIME_CHECK + 1) * 240) + _COV_OVERHEAD

@info "Running tests with ALLOC_THRESHOLD = $ALLOC_THRESHOLD bytes (AAP_RUNTIME_CHECK = $AAP_RUNTIME_CHECK)"

# ── Run migrated @testitem files via TestItemRunner ──────────────────
@run_package_tests verbose = true filter = ti -> begin
    if !isempty(ARGS)
        return any(arg -> occursin(arg, ti.name) || occursin(arg, ti.filename), ARGS)
    end
    return true
end

# ── Legacy @testset files via include cascade (36 files) ─────────────
# These files have file-level helpers (function/const outside @testset)
# and are pending per-file migration that extracts helpers to @testsnippet
# or @testmodule. They run after @run_package_tests in the runtests
# scope, where the alloc constants above are visible.
if !isempty(ARGS)
    for arg in ARGS
        # Normalize: append .jl if missing (so "test_foo" works like "test_foo.jl")
        candidate = endswith(arg, ".jl") ? arg : arg * ".jl"
        # Skip migrated files — already run by @run_package_tests above
        basename(candidate) in MIGRATED_TESTITEM_FILES && continue
        # Skip if file doesn't exist — arg was likely a testitem name pattern
        # (handled by the @run_package_tests filter above) rather than a legacy filename
        isfile(joinpath(@__DIR__, candidate)) || continue
        @info "Running legacy test file: $candidate"
        include(candidate)
    end
else
    # Default behavior: run all legacy files
    include("test_cubic_nd.jl")
    include("test_cubic_nd_oneshot.jl")
    include("test_duck_typing_comprehensive.jl")
    include("test_linear_periodic.jl")
    include("test_mutation_safety.jl")
    include("test_nd_constant.jl")
    include("test_nd_linear.jl")
    include("test_nd_noextrap_oob.jl")
    include("test_nd_oneshot_onthefly.jl")
    include("test_nd_quadratic.jl")
    include("test_nointerp.jl")
    include("test_nonuniform_grid.jl")
    include("test_precision_vector_queries.jl")
    include("test_thread_safety.jl")

    # ── Extension tests (AD / Symbolics) ──────────────────────────────
    # In CI, extensions run in a SEPARATE job (clean Julia process) to prevent
    # ChainRulesCore contamination from Interpolations.jl (test_packages_comparison).
    if get(ENV, "CI", nothing) !== nothing && get(ENV, "SKIP_EXTENSIONS", nothing) === nothing
        include("ext/runtests.jl")
    elseif get(ENV, "CI", nothing) === nothing
        @info "Skipping extension tests (run via: julia --project=test test/ext/runtests.jl)"
    end
end
