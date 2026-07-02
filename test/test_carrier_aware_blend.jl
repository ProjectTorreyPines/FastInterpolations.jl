@testitem "carrier-aware blend: native-FMA classification" setup = [AllocConstants] begin
    using FastInterpolations: _LinearBlendFMA, _LinearBlendGeneric, _linear_blend_style
    using FixedPointNumbers, ColorTypes, ColorVectorSpace, ForwardDiff

    # hardware-FMA-fusable result types → FMA style
    @test _linear_blend_style(Float64) === _LinearBlendFMA()
    @test _linear_blend_style(Float32) === _LinearBlendFMA()
    @test _linear_blend_style(Float16) === _LinearBlendFMA()
    @test _linear_blend_style(ComplexF64) === _LinearBlendFMA()
    @test _linear_blend_style(ComplexF32) === _LinearBlendFMA()
    # fixed-point promotes to float under a real weight → FMA
    @test _linear_blend_style(Base.promote_op(*, Float64, N0f8)) === _LinearBlendFMA()   # == Float64
    # non-fusable carriers → generic (the colorant COMPONENTWISE opt-in lives on
    # the ext's colorant blend ENTRIES, not on this result-type classifier)
    @test _linear_blend_style(Gray{Float64}) === _LinearBlendGeneric()
    @test _linear_blend_style(RGB{Float64}) === _LinearBlendGeneric()
    @test _linear_blend_style(BigFloat) === _LinearBlendGeneric()
    @test _linear_blend_style(ForwardDiff.Dual{Nothing, Float64, 2}) === _LinearBlendGeneric()
    @test _linear_blend_style(Union{}) === _LinearBlendGeneric()   # undefined `*` → safe generic
end

@testitem "carrier-aware blend: linear FMA path unchanged + generic faithful" setup = [AllocConstants] begin
    using FastInterpolations: _LinearBlendFMA, _LinearBlendGeneric, _linear_value_blend
    using Random

    # Float path: dispatched form == verbatim FMA form (bit-identical), endpoint-exact
    @test _linear_value_blend(0.3, 0.2, 0.9) === muladd(0.3, 0.9, muladd(-0.3, 0.2, 0.2))
    @test _linear_value_blend(0.0, 0.2, 0.9) === 0.2
    @test _linear_value_blend(1.0, 0.2, 0.9) === 0.9
    @test _linear_value_blend(0.0f0, 0.2f0, 0.9f0) === 0.2f0
    @test _linear_value_blend(1.0f0, 0.2f0, 0.9f0) === 0.9f0

    # The FMA and generic forms are algebraically identical; they agree to ~1 ULP
    # *relative to the input scale*. (Bit-identical on Julia ≥1.12, where LLVM
    # contracts both to the same FMA; on older LLVM the rounding differs slightly.
    # A result-ULP metric is the wrong gauge — it explodes near cancellation, where
    # the result is tiny but |a−b| is still ~1 ULP of the inputs.) Floats ship the
    # FMA form regardless, so this only pins the generic form as a faithful refactor.
    function _maxreldiff(n)
        rng = MersenneTwister(1)
        m = 0.0
        for _ in 1:n
            α = rand(rng)
            yL = (rand(rng) - 0.5) * 20
            yR = (rand(rng) - 0.5) * 20
            a = _linear_value_blend(_LinearBlendFMA(), α, yL, yR)
            b = _linear_value_blend(_LinearBlendGeneric(), α, yL, yR)
            m = max(m, abs(a - b) / max(abs(yL), abs(yR), abs(a)))
        end
        return m
    end
    @test _maxreldiff(200_000) <= 8 * eps(Float64)
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
