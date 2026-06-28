@testitem "no-wrap helpers" begin
    using FastInterpolations: _fielddiff, _fieldsum, _linear_value_blend

    # promote-operands difference: wrap-free for UInt8, bit-identical for Float
    @test _fielddiff(Float64, UInt8(50), UInt8(200)) == -150.0   # raw UInt8 sub would be +106
    @test _fielddiff(Float64, 3.0, 1.0) === 2.0                  # Float identity
    @test _fieldsum(Float64, UInt8(200), UInt8(200)) == 400.0    # raw UInt8 add would wrap
    @test _fieldsum(Float64, 1.5, 2.5) === 4.0

    # convex linear blend: endpoint-exact + correct on descending finite cell
    @test _linear_value_blend(1.0, UInt8(200), UInt8(50)) == 50.0   # t=1 → yR exactly
    @test _linear_value_blend(0.0, UInt8(200), UInt8(50)) == 200.0  # t=0 → yL exactly
    @test _linear_value_blend(0.5, UInt8(200), UInt8(50)) == 125.0  # true midpoint (raw wrap → 253)
    @test _linear_value_blend(0.5, 0.2, 0.8) ≈ 0.5
end
