@testitem "LinearInterpolantND — spacings field removed (lock-down)" begin
    using FastInterpolations: linear_interp

    x = 0.0:1.0:3.0
    y = 0.0:1.0:3.0
    data = [Float64(i + j) for i in 1:4, j in 1:4]
    itp = linear_interp((x, y), data)

    # Field-level lock-down: no `spacings` field after PR1
    @test !hasfield(typeof(itp), :spacings)

    # Type-parameter count: was 7 (Tg, Tv, N, G, S, E, P), now 6 (drops S)
    @test length(typeof(itp).parameters) == 6

    # Sanity: itp still works correctly. data[i,j] = i+j (1-indexed), bilinear at
    # (1.5, 1.5) reads corners at x[2]=1, x[3]=2 → data[2..3, 2..3] = (4,5,5,6),
    # average = 5.0
    @test itp((1.5, 1.5)) ≈ 5.0
end
