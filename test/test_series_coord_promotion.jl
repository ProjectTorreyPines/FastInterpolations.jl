# Series/anchor-pool coordinate-type contract: vector evaluation of Cubic/Quadratic
# Series interpolants stays concrete and rides the grid (I2) for a Dual grid, stays
# real for Dual data (I3), keeps Int for Int grids, and is zero-alloc/inferred on Float64.

@testitem "Cubic Series vector eval — Dual grid concrete (I2) + Float64 zero-alloc (I5)" setup = [AllocConstants] begin
    using ForwardDiff: Dual
    g = collect(0.0:1.0:9.0)
    Y = [Float64(i + 2s) for i in 1:10, s in 1:3]      # 10 points × 3 series
    sitp = cubic_interp(g, Series(Y))                   # Series PreCompute
    xq = [1.5, 4.5, 7.5]
    outs = [Vector{Float64}(undef, length(xq)) for _ in 1:3]
    e(it, o, q) = (it(o, q); @allocated it(o, q))
    e(sitp, outs, xq)                                   # warmup
    @test e(sitp, outs, xq) <= ALLOC_THRESHOLD

    gd = [Dual{Nothing}(v, 1.0) for v in g]
    sitp_d = cubic_interp(gd, Series(Y))
    outs_d = [Vector{Dual{Nothing, Float64, 1}}(undef, length(xq)) for _ in 1:3]
    sitp_d(outs_d, xq)
    @test all(o -> isconcretetype(eltype(o)) && eltype(o) <: Dual, outs_d)
end

@testitem "Quadratic Series vector eval — Dual grid concrete + Int grid keeps real" begin
    using ForwardDiff: Dual
    g = collect(0.0:1.0:9.0)
    Y = [Float64(i + 2s) for i in 1:10, s in 1:2]
    gd = [Dual{Nothing}(v, 1.0) for v in g]
    sitp_d = quadratic_interp(gd, Series(Y))
    xq = [2.5, 6.5]
    outs_d = [Vector{Dual{Nothing, Float64, 1}}(undef, length(xq)) for _ in 1:2]
    sitp_d(outs_d, xq)
    @test all(o -> eltype(o) <: Dual, outs_d)
end
