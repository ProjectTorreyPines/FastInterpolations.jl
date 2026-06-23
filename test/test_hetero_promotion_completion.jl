# Hetero coefficient (partials) eltype contract: partials are Dual-concrete on a Dual
# grid, and the Float64 scalar-eval path is inferred + zero-alloc.

@testitem "Hetero partials concrete on Dual grid" begin
    using ForwardDiff: Dual
    g = [Dual{Nothing}(Float64(v), 1.0) for v in 0.0:1.0:5.0]
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]
    # Cubic × Pchip hetero: pchip axis builds slope partials → must be Dual-concrete.
    itp = interp((g, g), data; method = (CubicInterp(), PchipInterp()))
    rt = Base.return_types(itp, Tuple{Float64, Float64})
    @test length(rt) == 1 && isconcretetype(rt[1])
    @test (@inferred itp(2.5, 3.5)) isa Dual
end

@testitem "Hetero partials — Float64 path inferred + zero-alloc" setup = [AllocConstants] begin
    g = collect(0.0:1.0:5.0)
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]
    itp = interp((g, g), data; method = (CubicInterp(), PchipInterp()))
    @test (@inferred itp(2.5, 3.5)) isa Float64
    h(it, a, b) = (it(a, b); @allocated it(a, b))
    h(itp, 2.5, 3.5)                            # warmup
    @test h(itp, 2.5, 3.5) <= ALLOC_THRESHOLD   # scalar eval pool-bounded
end
