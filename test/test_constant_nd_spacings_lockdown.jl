@testitem "ConstantInterpolantND — spacings field removed (lock-down)" begin
    using FastInterpolations: constant_interp

    x = 0.0:1.0:3.0
    y = 0.0:1.0:3.0
    data = [Float64(i + j) for i in 1:4, j in 1:4]
    itp = constant_interp((x, y), data)

    @test !hasfield(typeof(itp), :spacings)
    # Was 8 (Tg, Tv, N, G, S, E, SD, P), now 7 (drops S)
    @test length(typeof(itp).parameters) == 7

    # Sanity: itp callable, returns a finite value at a query point
    val = itp((1.5, 1.5))
    @test val isa Float64
    @test isfinite(val)
end
