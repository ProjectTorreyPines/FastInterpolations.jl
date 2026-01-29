using Test
using FastInterpolations
using Random

# Julia 1.12+ achieves true zero-allocation via improved escape analysis.
# Older versions have small runtime overhead from mutable struct field access.
# Note: 4-way Val dispatch (extrap modes) increases overhead on older Julia (~160 bytes).
const ALLOC_THRESHOLD = VERSION >= v"1.12" ? 0 : 240

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Default behavior: run all tests
    include("test_grid_spacing.jl")
    include("test_search.jl")
    include("test_constant.jl")
    include("test_linear.jl")
    include("test_quadratic.jl")
    include("test_cubic.jl")
    include("test_cubic_autocache.jl")
    include("test_cubic_interpolant.jl")
    include("test_cubic_anchor.jl")
    include("test_linear_anchor.jl")
    include("test_constant_anchor.jl")
    include("test_quadratic_anchor.jl")
    include("test_cubic_series_interp.jl")
    include("test_linear_series_interp.jl")
    include("test_constant_series_interp.jl")
    include("test_quadratic_series_interp.jl")
    include("test_allocation.jl")
    include("test_random_grid.jl")
    include("test_periodic_bc.jl")
    include("test_ldiv_tridiagonal_nopiv.jl")
    include("test_generic_bc.jl")
    include("test_polyfit_bc.jl")
    include("test_type_stability.jl")
    include("test_derivatives.jl")
    include("test_packages_comparison.jl")
    include("test_thread_safety.jl")
    include("test_rcu.jl")
    include("test_nonuniform_grid.jl")
    include("test_show.jl")
    include("test_recipes.jl")
end
