@testitem "no-wrap helpers" begin
    using FastInterpolations: _fielddiff, _fieldsum, _linear_value_blend

    # promote-operands difference: wrap-free for UInt8, bit-identical for Float
    @test _fielddiff(Float64, UInt8(50), UInt8(200)) == -150.0   # raw UInt8 sub would be +106
    @test _fielddiff(Float64, 3.0, 1.0) === 2.0                  # Float identity
    @test _fieldsum(Float64, UInt8(200), UInt8(200)) == 400.0    # raw UInt8 add would wrap
    @test _fieldsum(Float64, 1.5, 2.5) === 4.0

    # signed-overflow (Int8) is also fixed: raw Int8(-128) - Int8(1) wraps to +127
    @test _fielddiff(Float64, Int8(-128), Int8(1)) == -129.0
    @test _fieldsum(Float64, Int8(100), Int8(100)) == 200.0   # raw Int8 add would wrap to -56

    # convex linear blend: endpoint-exact + correct on descending finite cell
    @test _linear_value_blend(1.0, UInt8(200), UInt8(50)) == 50.0   # t=1 → yR exactly
    @test _linear_value_blend(0.0, UInt8(200), UInt8(50)) == 200.0  # t=0 → yL exactly
    @test _linear_value_blend(0.5, UInt8(200), UInt8(50)) == 125.0  # true midpoint (raw wrap → 253)
    @test _linear_value_blend(0.5, 0.2, 0.8) ≈ 0.5
end

@testitem "linear value kernel no-wrap" begin
    using FastInterpolations
    const FI = FastInterpolations

    # Raw 1D kernel with native UInt8 (external-consumer style): descending cell.
    @test FI._linear_kernel(FI.EvalValue(), UInt8(200), UInt8(50), 1.0, 0.5) == 125.0

    # Anchored value kernel: build a real anchor on a unit cell (alpha=0.5,
    # inv_h=1.0) via the internal builder, then feed native UInt8 corner values.
    aq = FI._anchor_query([1.0, 2.0], 1.5, Val(:linear))   # alpha=0.5, inv_h=1.0
    @test FI._linear_kernel(FI.EvalValue(), UInt8(200), UInt8(50), aq) == 125.0

    # Bilinear (ND) collapses through the 1D value kernel — spot-check the helper
    # form is endpoint-exact so an N0f8-range write stays in-range.
    @test FI._linear_kernel(FI.EvalValue(), UInt8(200), UInt8(50), 1.0, 1.0) == 50.0
end

@testitem "linear value convex: Float behavior" begin
    using FastInterpolations
    const FI = FastInterpolations
    # Endpoint-exact (the reason for convex over slope form)
    @test FI._linear_kernel(FI.EvalValue(), 0.2, 0.9, 1.0, 1.0) === 0.9
    @test FI._linear_kernel(FI.EvalValue(), 0.2, 0.9, 1.0, 0.0) === 0.2
    # Bounded: interior stays within [yL,yR]
    for α in 0.0:0.1:1.0
        v = FI._linear_kernel(FI.EvalValue(), 0.2, 0.9, 1.0, α)
        @test 0.2 <= v <= 0.9
    end
end

@testitem "no-wrap helpers preserve natural promotion (no forced convert)" begin
    using FastInterpolations: _fielddiff, _fieldsum
    using ForwardDiff: Dual

    # Field types: the fast-path method (a::Tc, b::Tc) is plain `a - b`/`a + b` —
    # byte-for-byte identical to the old code, NO convert.
    @test _fielddiff(Float64, 3.0, 1.0) === 3.0 - 1.0
    @test _fieldsum(Float64, 3.0, 1.0) === 3.0 + 1.0
    @test _fielddiff(Float32, 3.0f0, 1.0f0) === 3.0f0 - 1.0f0
    @test _fielddiff(ComplexF64, 2.0 + 1im, 1.0 + 0im) === (2.0 + 1im) - (1.0 + 0im)

    # Duck/AD types take the natural-promotion path (Tc === the duck type) — the
    # result is the exact Dual, partials intact, never flattened through Float.
    d1 = Dual(3.0, 1.0); d2 = Dual(1.0, 0.0)
    Td = typeof(d1)
    @test _fielddiff(Td, d1, d2) === d1 - d2          # identity, partials preserved
    @test _fieldsum(Td, d1, d2) === d1 + d2

    # Mixed field lift (e.g. Float operand in a Dual coefficient field, as in
    # AD-wrt-grid): convert lifts to the field exactly as natural promotion would.
    @test _fielddiff(Td, 3.0, 1.0) === Td(3.0) - Td(1.0)
end
