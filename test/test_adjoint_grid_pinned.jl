# Pins design decision D4: adjoint operators keep coordinate type param == grid type
# (_LinearAnchoredQuery{Tg, Tg}, _CubicAnchoredQuery{Tg, Tg}). AD-through-adjoint is a
# non-goal; this guards against an accidental coordinate-widening refactor.

@testitem "Adjoint coordinate type stays grid-pinned (D4)" begin
    using FastInterpolations: _LinearAnchoredQuery, _CubicAnchoredQuery
    x = collect(0.0:1.0:9.0); y = collect(Float64, 1:10); xq = [1.5, 4.5, 7.5]
    la = linear_adjoint(x, xq)
    @test eltype(la.anchors) <: _LinearAnchoredQuery{Float64, Float64}   # {Tg, Tg}
    ca = cubic_adjoint(x, xq)
    @test eltype(ca.anchors) <: _CubicAnchoredQuery{Float64, Float64}    # {Tg, Tg}
end

@testitem "Linear anchor coordinate derivation is already canonical (verify-only)" begin
    using FastInterpolations: _coord_eltype
    # The _LinearAnchoredQuery outer ctor derives Tc from typeof((xq-xL)*inv_h);
    # for these pairs that equals _coord_eltype(Tq, Tg). Pin the equivalence.
    @test _coord_eltype(Float64, Float64) === Float64
    @test _coord_eltype(Int, Float64) === Float64
    @test _coord_eltype(Float64, Int) === Float64
end
