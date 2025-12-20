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
    include("test_linear.jl")
    include("test_cubic.jl")
    include("test_cubic_autocache.jl")
    include("test_cubic_callable.jl")
    include("test_allocation.jl")
    include("test_random_grid.jl")
    include("test_periodic_bc.jl")
    include("test_packages_comparison.jl")
end
