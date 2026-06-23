# Series/anchor-pool coordinate-type contract: vector evaluation of Cubic/Quadratic Series
# interpolants. Series is a build-once/cached path — it floats the grid at construction
# (`_promote_grid_float` / `_to_float` / `_cache_axis_pooled`), so the pooled-anchor
# coordinate `_coord_eltype(Tq, Tg)` rides a Dual grid (AD wrt grid nodes), widens
# Float32→Float64 precision, stays real for Dual *data* (data never contaminates the
# coordinate), and is zero-alloc/inferred on Float64.
#
# NOTE: Int grids are *floated* here, not kept (`float(Int) == Float64`). The raw-grid
# Int-coordinate contract belongs to the single-pass one-shot paths, which keep the grid
# raw for zero-alloc — see test_promotion_alloc.jl ("Int grid one-shot: zero-alloc scalar")
# and the `_coord_eltype` witness identities in test_dual_grid_coord_promotion.jl.

@testitem "Cubic Series vector eval — Dual grid concrete + Float64 zero-alloc" setup = [AllocConstants] begin
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

@testitem "Quadratic Series vector eval — Dual grid concrete (grid floats; Dual rides)" begin
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
