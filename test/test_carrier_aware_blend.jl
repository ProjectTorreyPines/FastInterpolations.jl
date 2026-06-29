@testitem "carrier-aware blend: _blend_fuses classification" setup = [AllocConstants] begin
    using FastInterpolations: _blend_fuses
    using FixedPointNumbers, ColorTypes, ColorVectorSpace, ForwardDiff

    # hardware-FMA-fusable result types → Val(true)
    @test _blend_fuses(Float64) === Val(true)
    @test _blend_fuses(Float32) === Val(true)
    @test _blend_fuses(Float16) === Val(true)
    @test _blend_fuses(ComplexF64) === Val(true)
    @test _blend_fuses(ComplexF32) === Val(true)
    # fixed-point promotes to float under a real weight → fused
    @test _blend_fuses(Base.promote_op(*, Float64, N0f8)) === Val(true)   # == Float64
    # non-fusable carriers → Val(false)
    @test _blend_fuses(Gray{Float64}) === Val(false)
    @test _blend_fuses(RGB{Float64}) === Val(false)
    @test _blend_fuses(BigFloat) === Val(false)
    @test _blend_fuses(ForwardDiff.Dual{Nothing, Float64, 2}) === Val(false)
    @test _blend_fuses(Union{}) === Val(false)        # undefined `*` (no ColorVectorSpace) → safe ALT
end

@testitem "carrier-aware blend: linear FMA path unchanged + ALT faithful" setup = [AllocConstants] begin
    using FastInterpolations: _linear_value_blend
    using Random

    # Float path: dispatched form == verbatim FMA form (bit-identical), endpoint-exact
    @test _linear_value_blend(0.3, 0.2, 0.9) === muladd(0.3, 0.9, muladd(-0.3, 0.2, 0.2))
    @test _linear_value_blend(0.0, 0.2, 0.9) === 0.2
    @test _linear_value_blend(1.0, 0.2, 0.9) === 0.9
    @test _linear_value_blend(0.0f0, 0.2f0, 0.9f0) === 0.2f0
    @test _linear_value_blend(1.0f0, 0.2f0, 0.9f0) === 0.9f0

    # ALT form is a faithful refactor of the FMA form on floats (≤1 ULP; measured 0)
    function _maxulp_probe(n)
        rng = MersenneTwister(1)
        mu = 0
        for _ in 1:n
            α = rand(rng)
            yL = (rand(rng) - 0.5) * 20
            yR = (rand(rng) - 0.5) * 20
            a = _linear_value_blend(Val(true), α, yL, yR)
            b = _linear_value_blend(Val(false), α, yL, yR)
            mu = max(mu, abs(reinterpret(Int64, a) - reinterpret(Int64, b)))
        end
        return mu
    end
    @test _maxulp_probe(200_000) <= 1
end

@testitem "carrier-aware blend: linear type-stable + zero-alloc + carriers correct" setup = [AllocConstants] begin
    import FastInterpolations as FI
    using FixedPointNumbers, ColorTypes, ColorVectorSpace, ForwardDiff, Random

    x = collect(0.0:0.1:1.0)
    y = sin.(x)
    itp = FI.linear_interp(x, y)
    itp(0.55)                                              # warmup
    @test (@inferred itp(0.55)) isa Float64
    @test (@allocated itp(0.55)) <= ALLOC_THRESHOLD        # Float path: no new alloc

    # ComplexF64 stays fused, type-stable
    yc = ComplexF64.(y, reverse(y))
    itpc = FI.linear_interp(x, yc)
    @test (@inferred itpc(0.55)) isa ComplexF64

    # color/N0f8 routes through ALT and stays correct vs the Float reference (≤1e-2 for N0f8 quantization)
    g = Gray{N0f8}.(rand(MersenneTwister(2), length(x)))
    itpg = FI.linear_interp(x, g)
    itpf = FI.linear_interp(x, Float64.(gray.(g)))
    for q in (0.05, 0.37, 0.62, 0.91)
        @test isapprox(Float64(gray(itpg(q))), itpf(q); atol = 1.0e-2)
    end
    # endpoint-exactness on N0f8 (wrap-free, exact)
    @test itpg(x[3]) == g[3]
end
