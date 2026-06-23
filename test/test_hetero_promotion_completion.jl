# Pins the hetero promotion-completion contract: coefficient partials are Dual-concrete
# on a Dual grid (Phase 2), and the per-axis output fold keeps Int for all-Constant and
# floats for any dividing axis (Phase 4). See completion spec phases 2 & 4.

@testitem "Hetero partials concrete on Dual grid (Phase 2 coefficient eltype)" begin
    using ForwardDiff: Dual
    g = [Dual{Nothing}(Float64(v), 1.0) for v in 0.0:1.0:5.0]
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]
    # Cubic × Pchip hetero: pchip axis builds slope partials → must be Dual-concrete.
    itp = interp((g, g), data; method = (CubicInterp(), PchipInterp()))
    rt = Base.return_types(itp, Tuple{Float64, Float64})
    @test length(rt) == 1 && isconcretetype(rt[1])
    @test (@inferred itp(2.5, 3.5)) isa Dual
end

@testitem "Hetero partials — Float64 path inferred + zero-alloc (I5)" setup=[AllocConstants] begin
    g = collect(0.0:1.0:5.0)
    data = [Float64(i + 2j) for i in 1:6, j in 1:6]
    itp = interp((g, g), data; method = (CubicInterp(), PchipInterp()))
    @test (@inferred itp(2.5, 3.5)) isa Float64
    h(it, a, b) = (it(a, b); @allocated it(a, b))
    h(itp, 2.5, 3.5)                            # warmup
    @test h(itp, 2.5, 3.5) <= ALLOC_THRESHOLD   # scalar eval pool-bounded
end

@testitem "_hetero_output_eltype — per-axis fold keeps/floats/duals correctly" begin
    using FastInterpolations: _hetero_output_eltype
    C = ConstantInterp(RightSide(), NoBC())   # Int-keeping axis
    L = LinearInterp(NoBC()); K = CubicInterp(NoBC())        # dividing axes
    # all-Constant + all-Int → keeps Int (the corner the legacy form over-floats)
    @test _hetero_output_eltype((C, C), Int, Int, (1, 1)) === Int
    # any dividing axis → floats
    @test _hetero_output_eltype((C, L), Int, Int, (1, 1)) === Float64
    @test _hetero_output_eltype((K, C), Int, Int, (1, 1)) === Float64
    # Float64 everywhere → Float64
    @test _hetero_output_eltype((K, L), Float64, Float64, (1.0, 1.0)) === Float64
end

@testitem "_hetero_output_eltype — Dual rides through; type-level (zero-arg-eval)" begin
    using FastInterpolations: _hetero_output_eltype
    using ForwardDiff: Dual
    DT = Dual{Nothing, Float64, 1}
    L = LinearInterp(NoBC()); K = CubicInterp(NoBC())
    @test _hetero_output_eltype((K, L), DT, Float64, (1.0, 1.0)) <: Dual          # Dual grid
    @test _hetero_output_eltype((K, L), Float64, DT, (1.0, 1.0)) <: Dual          # Dual data
    # Pure type-level: return_types is concrete (the fold constant-folds)
    rt = Base.return_types(_hetero_output_eltype, Tuple{Tuple{typeof(K), typeof(L)}, Type{Float64}, Type{Float64}, Tuple{Float64, Float64}})
    @test length(rt) == 1 && rt[1] === Type{Float64}
end
