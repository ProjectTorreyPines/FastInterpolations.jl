using Test
using FastInterpolations

# Check if specific test files are requested via ARGS
if !isempty(ARGS)
    for testfile in ARGS
        @info "Running test file: $testfile"
        include(testfile)
    end
else
    # Default behavior: run all tests
    @testset "FastInterpolations.jl" begin
        include("test_linear.jl")
        include("test_cubic.jl")
        include("test_cubic_autocache.jl")
        include("test_cubic_callable.jl")
        include("test_allocation.jl")
    end
end
