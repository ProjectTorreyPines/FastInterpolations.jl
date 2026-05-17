@testitem "Linear batch WrapExtrap — exact last(x) takes fast path" begin
    using FastInterpolations
    x = collect(range(0.0, 1.0, 11))
    y = collect(1.0:11.0)
    xq = [0.0, 0.5, 1.0]
    out = linear_interp(x, y, xq; extrap=WrapExtrap())
    # Closed: last entry should be y[end] = 11.0, NOT y[1] = 1.0
    @test out[end] == y[end]
    @test out[1] == y[1]
end
