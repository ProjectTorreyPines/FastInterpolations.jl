@testitem "colorant Gray{N0f8} interpolation no-wrap" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    x = [1.0, 2.0, 3.0, 4.0]
    g = Gray{N0f8}.([0.9, 0.1, 0.8, 0.2])     # descending cells

    itpG = FI.linear_interp(x, g)
    itpF = FI.linear_interp(x, Float64.(gray.(g)))

    for q in (1.5, 2.5, 3.5, 2.25, 3.75)
        @test isapprox(Float64(gray(itpG(q))), Float64(itpF(q)); atol = 1.0e-2)
    end

    # Bounded/monotone safety: every interpolated Gray{N0f8} value is in [0,1],
    # so converting back to N0f8 (the upsample write) cannot throw.
    for q in range(1.0, 4.0; length = 31)
        v = gray(itpG(q))
        @test 0.0 <= Float64(v) <= 1.0
        @test (N0f8(clamp(Float64(v), 0.0, 1.0)); true)   # round-trip into N0f8 succeeds
    end
end

@testitem "colorant RGB{N0f8} smoke" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations
    x = [1.0, 2.0, 3.0]
    c = [RGB{N0f8}(0.9, 0.1, 0.5), RGB{N0f8}(0.1, 0.8, 0.2), RGB{N0f8}(0.7, 0.3, 0.9)]
    itp = FI.linear_interp(x, c)
    v = itp(1.5)
    @test 0.0 <= Float64(red(v)) <= 1.0
    @test 0.0 <= Float64(green(v)) <= 1.0
    @test 0.0 <= Float64(blue(v)) <= 1.0
end

@testitem "colorant 2D bilinear vs Interpolations.jl" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    import Interpolations as IP
    const FI = FastInterpolations
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 2.0, 3.0, 4.0]
    A = Gray{N0f8}.(reshape(range(0.05, 0.95; length = 16), 4, 4))
    fi = FI.interp((xs, ys), A; method = FI.LinearInterp())
    ip = IP.linear_interpolation((xs, ys), A)
    for qx in (1.5, 2.5, 3.25), qy in (1.5, 2.5, 3.75)
        @test isapprox(Float64(gray(fi(qx, qy))), Float64(gray(ip(qx, qy))); atol = 1.0e-2)
    end
end
