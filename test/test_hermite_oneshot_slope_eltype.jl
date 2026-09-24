# Hermite ONESHOT (3-arg) slope-buffer eltype contract across grid/query eltypes.
# Float64/Int/Dual only: the Hermite eval kernel divides by the grid (`/(value, Float64)`),
# so a `/`-less duck value type cannot evaluate here.

@testitem "Hermite oneshot slope eltype — Float64 inferred + zero-alloc + value matches PreCompute" setup = [AllocConstants] begin
    x = collect(0.0:1.0:9.0)
    y = @. sin(x) + 0.5x
    xq = 3.7
    for f in (pchip_interp, cardinal_interp, akima_interp)
        oneshot = @inferred f(x, y, xq)        # 3-arg ONESHOT
        precomp = f(x, y)(xq)                   # 2-arg PreCompute twin
        @test oneshot ≈ precomp
        @test oneshot isa Float64
        # zero-alloc measured in a function barrier
        g(fn, xx, yy, q) = (fn(xx, yy, q); @allocated fn(xx, yy, q))
        g(f, x, y, xq)                          # warmup
        @test g(f, x, y, xq) <= ALLOC_THRESHOLD
    end
end

@testitem "Hermite oneshot slope eltype — Int grid floats to Float64" begin
    xi = collect(0:1:9)                         # Int grid
    y = Float64[sin(v) + 0.5v for v in xi]
    for f in (pchip_interp, cardinal_interp, akima_interp)
        r = @inferred f(xi, y, 3.7)
        @test r isa Float64                     # Int grid → kernel floats the output
    end
end

@testitem "Hermite oneshot slope eltype — Dual grid stays concrete Dual" begin
    using ForwardDiff: Dual
    g = [Dual{Nothing}(Float64(v), 1.0) for v in 0.5:1.0:9.5]
    y = collect(Float64, 1:10)
    for f in (pchip_interp, cardinal_interp, akima_interp)
        rt = Base.return_types(f, Tuple{typeof(g), typeof(y), Float64})
        @test length(rt) == 1 && isconcretetype(rt[1])
        @test (@inferred f(g, y, 3.0)) isa Dual
    end
end
