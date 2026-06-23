# Adjoint operators keep the coordinate type param pinned to the grid type
# (_LinearAnchoredQuery{Tg, Tg}, _CubicAnchoredQuery{Tg, Tg}): they operate on baked
# coefficients, so AD-through-adjoint is unsupported. Guards against an accidental
# coordinate-widening refactor.

@testitem "Adjoint coordinate type stays grid-pinned" begin
    using FastInterpolations: _LinearAnchoredQuery, _CubicAnchoredQuery
    x = collect(0.0:1.0:9.0); xq = [1.5, 4.5, 7.5]
    la = linear_adjoint(x, xq)
    @test eltype(la.anchors) <: _LinearAnchoredQuery{Float64, Float64}   # {Tg, Tg}
    ca = cubic_adjoint(x, xq)
    @test eltype(ca.anchors) <: _CubicAnchoredQuery{Float64, Float64}    # {Tg, Tg}
end

@testitem "Linear anchor coordinate derivation matches _coord_eltype" begin
    using FastInterpolations: _coord_eltype
    # The _LinearAnchoredQuery outer ctor derives Tc from typeof((xq-xL)*inv_h),
    # which equals _coord_eltype(Tq, Tg) for these pairs.
    @test _coord_eltype(Float64, Float64) === Float64
    @test _coord_eltype(Int, Float64) === Float64
    @test _coord_eltype(Float64, Int) === Float64
end
