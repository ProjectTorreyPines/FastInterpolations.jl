@testitem "lincomb2: styles, generic expression, safety net" begin
    using FastInterpolations:
        _LincombComponentwise,
        _LincombGeneric,
        _LincombStyle,
        _channel_blend_fuses,
        _lincomb2,
        _lincomb_style,
        _style_from_fuses
    using FixedPointNumbers

    # default style is generic for arbitrary type pairs
    @test @inferred(_lincomb_style(Float64, Float64)) === _LincombGeneric()
    @test @inferred(_lincomb_style(Float64, BigFloat)) === _LincombGeneric()
    @test _lincomb_style(Int, String) === _LincombGeneric()

    # generic expression is exactly muladd(b, y, a*x)
    a, x, b, y = 0.3, 1.7, 0.7, -2.2
    @test @inferred(_lincomb2(a, x, b, y)) === muladd(b, y, a * x)
    @test _lincomb2(0.25f0, 2.0f0, 0.75f0, 4.0f0) ===
        muladd(0.75f0, 4.0f0, 0.25f0 * 2.0f0)
    # complex values ride the generic expression too (Base componentwise muladd)
    @test _lincomb2(0.3, 1.0 + 2.0im, 0.7, 3.0 - 1.0im) ===
        muladd(0.7, 3.0 - 1.0im, 0.3 * (1.0 + 2.0im))

    # safety net: componentwise style with no specialized method degrades to generic
    @test _lincomb2(_LincombComponentwise(), a, x, b, y) === muladd(b, y, a * x)

    # channel-eligibility helpers
    @test _channel_blend_fuses(Float64, N0f8) === Val(true)   # Float64 × N0f8 → Float64
    @test _channel_blend_fuses(Float32, N0f8) === Val(true)
    @test _channel_blend_fuses(Float64, Float64) === Val(true)
    @test _channel_blend_fuses(BigFloat, BigFloat) === Val(false)
    @test _style_from_fuses(Val(true)) === _LincombComponentwise()
    @test _style_from_fuses(Val(false)) === _LincombGeneric()
end

@testitem "lincomb2: blend delegation is a faithful refactor" begin
    using FastInterpolations: _linear_value_blend

    # Val(true) FMA path untouched: verbatim FMA form, endpoint-exact
    @test _linear_value_blend(0.3, 0.2, 0.9) ===
        muladd(0.3, 0.9, muladd(-0.3, 0.2, 0.2))
    @test _linear_value_blend(0.0, 0.2, 0.9) === 0.2
    @test _linear_value_blend(1.0, 0.2, 0.9) === 0.9
    z1, z2 = 1.0 + 2.0im, 3.0 - 1.0im
    @test _linear_value_blend(0.3, z1, z2) ===
        muladd(0.3, z2, muladd(-0.3, z1, z1))

    # Val(false) generic path: delegation through _lincomb2 reproduces the
    # pre-lincomb expression muladd(α, yR, (one(α) - α) * yL) bit-identically
    α, yL, yR = big"0.3", big"0.2", big"0.9"
    @test _linear_value_blend(α, yL, yR) == muladd(α, yR, (one(α) - α) * yL)
    @test _linear_value_blend(big"0.0", yL, yR) == yL
    @test _linear_value_blend(big"1.0", yL, yR) == yR
    @test @inferred(_linear_value_blend(α, yL, yR)) isa BigFloat
end

@testitem "componentwise colorants: style gating" begin
    using FastInterpolations: _LincombComponentwise, _LincombGeneric, _lincomb_style
    using FixedPointNumbers, ColorTypes, ColorVectorSpace, ForwardDiff

    # eligible: weight × channel lands in a native-FMA type
    @test @inferred(_lincomb_style(Float64, Gray{N0f8})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float32, RGB{N0f8})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float64, RGB{Float64})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float64, AGray{N0f8})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float64, GrayA{N0f8})) === _LincombComponentwise()

    # 4-channel families: all eligible channels — wins hinge on the convex
    # blend hook preserving α into the channels (benchmark-gated; ext comment)
    @test @inferred(_lincomb_style(Float64, RGBA{Float64})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float64, ARGB{Float64})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float64, RGBA{N0f8})) === _LincombComponentwise()
    @test @inferred(_lincomb_style(Float64, ARGB{N0f8})) === _LincombComponentwise()

    # ineligible weight or channel → generic (gate includes the weight type)
    @test _lincomb_style(BigFloat, Gray{BigFloat}) === _LincombGeneric()
    @test _lincomb_style(ForwardDiff.Dual{Nothing, Float64, 2}, RGB{Float64}) ===
        _LincombGeneric()

    # packed colorants have no style method → generic (never re-quantized)
    @test _lincomb_style(Float64, Gray24) === _LincombGeneric()
    @test _lincomb_style(Float64, RGB24) === _LincombGeneric()
    @test _lincomb_style(Float64, ARGB32) === _LincombGeneric()
end

@testitem "componentwise colorants: output types stay widened" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    x = [1.0, 2.0, 3.0]
    cases = (
        (Gray{N0f8}.([0.2, 0.7, 0.4]), Gray{Float64}),
        (Gray{Float64}.([0.2, 0.7, 0.4]), Gray{Float64}),
        ([RGB{N0f8}(0.9, 0.1, 0.5), RGB{N0f8}(0.1, 0.8, 0.2), RGB{N0f8}(0.7, 0.3, 0.9)], RGB{Float64}),
        ([RGBA{N0f8}(0.9, 0.1, 0.5, 0.8), RGBA{N0f8}(0.1, 0.8, 0.2, 0.3), RGBA{N0f8}(0.7, 0.3, 0.9, 0.6)], RGBA{Float64}),
        # transparent families (componentwise-opted; alpha widens like color)
        ([AGray{N0f8}(0.2, 0.9), AGray{N0f8}(0.7, 0.3), AGray{N0f8}(0.4, 0.6)], AGray{Float64}),
        ([GrayA{N0f8}(0.2, 0.9), GrayA{N0f8}(0.7, 0.3), GrayA{N0f8}(0.4, 0.6)], GrayA{Float64}),
        ([RGBA{Float64}(0.9, 0.1, 0.5, 0.8), RGBA{Float64}(0.1, 0.8, 0.2, 0.3), RGBA{Float64}(0.7, 0.3, 0.9, 0.6)], RGBA{Float64}),
        ([ARGB{Float64}(0.9, 0.1, 0.5, 0.8), ARGB{Float64}(0.1, 0.8, 0.2, 0.3), ARGB{Float64}(0.7, 0.3, 0.9, 0.6)], ARGB{Float64}),
        # packed: still widen — the componentwise path must NOT capture these
        ([Gray24(0.2), Gray24(0.7), Gray24(0.4)], Gray{Float64}),
        ([RGB24(0.9, 0.1, 0.5), RGB24(0.1, 0.8, 0.2), RGB24(0.7, 0.3, 0.9)], RGB{Float64}),
        ([ARGB32(0.9, 0.1, 0.5, 0.8), ARGB32(0.1, 0.8, 0.2, 0.3), ARGB32(0.7, 0.3, 0.9, 0.6)], ARGB{Float64}),
    )
    for (data, W) in cases
        itp = FI.linear_interp(x, data)
        @test typeof(itp(1.5)) === W
    end
end

@testitem "componentwise colorants: parity, endpoints, safety net" begin
    using FastInterpolations
    using FastInterpolations: _lincomb2, _linear_value_blend
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    # direct blend parity vs the generic CVS expression (≤ few-ulp FMA rounding)
    g0 = RGB{N0f8}(0.9, 0.1, 0.5)
    g1 = RGB{N0f8}(0.1, 0.8, 0.2)
    cw = _linear_value_blend(0.3, g0, g1)
    gen = muladd(0.3, g1, 0.7 * g0)
    @test typeof(cw) === typeof(gen) === RGB{Float64}
    for ch in (red, green, blue)
        @test isapprox(ch(cw), ch(gen); rtol = 1.0e-13, atol = 1.0e-15)
    end
    @test @inferred(_linear_value_blend(0.3, g0, g1)) isa RGB{Float64}

    # endpoints exact for fixed-point colorants
    v0 = _linear_value_blend(0.0, g0, g1)
    v1 = _linear_value_blend(1.0, g0, g1)
    for ch in (red, green, blue)
        @test ch(v0) == Float64(ch(g0))
        @test ch(v1) == Float64(ch(g1))
    end

    # interpolation-level parity vs per-channel scalar FI (ground truth)
    x = [1.0, 2.0, 3.0, 4.0]
    c = [
        RGB{N0f8}(0.9, 0.1, 0.5), RGB{N0f8}(0.1, 0.8, 0.2),
        RGB{N0f8}(0.7, 0.3, 0.9), RGB{N0f8}(0.2, 0.6, 0.4),
    ]
    itpC = FI.linear_interp(x, c)
    chans = (red, green, blue)
    itpS = map(ch -> FI.linear_interp(x, Float64.(ch.(c))), chans)
    for q in (1.25, 1.5, 2.75, 3.5)
        v = itpC(q)
        for (k, ch) in enumerate(chans)
            @test isapprox(ch(v), itpS[k](q); rtol = 1.0e-12, atol = 1.0e-14)
        end
    end

    # safety net: mixed concrete colorant types — no MethodError, result
    # equals the generic expression bit-identically. (Unit-level by design:
    # FI's linear kernel constrains both endpoints to one Tv, so mixed pairs
    # can only reach _lincomb2 through direct calls / future kernels.)
    ga = Gray{N0f8}(0.2)
    gb = Gray{Float32}(0.5)
    v = _lincomb2(0.3, ga, 0.7, gb)
    @test v == muladd(0.7, gb, 0.3 * ga)
end

@testitem "componentwise colorants: alpha-channel parity (AGray/ARGB)" begin
    using FastInterpolations: _linear_value_blend
    using FixedPointNumbers, ColorTypes, ColorVectorSpace

    # AGray{N0f8}: componentwise (2-channel, incl. alpha) vs generic CVS expression
    a0 = AGray{N0f8}(0.2, 0.9)
    a1 = AGray{N0f8}(0.7, 0.3)
    cw = _linear_value_blend(0.3, a0, a1)
    gen = muladd(0.3, a1, 0.7 * a0)
    @test typeof(cw) === typeof(gen) === AGray{Float64}
    @test isapprox(gray(cw), gray(gen); rtol = 1.0e-13, atol = 1.0e-15)
    @test isapprox(alpha(cw), alpha(gen); rtol = 1.0e-13, atol = 1.0e-15)

    # ARGB{Float64}: the positive branch of the IEEEFloat-restricted 4-channel
    # opt-in — alpha-first storage must not permute channels through mapc
    r0 = ARGB{Float64}(0.9, 0.1, 0.5, 0.8)
    r1 = ARGB{Float64}(0.1, 0.8, 0.2, 0.3)
    cw4 = _linear_value_blend(0.25, r0, r1)
    gen4 = muladd(0.25, r1, 0.75 * r0)
    @test typeof(cw4) === ARGB{Float64}
    for ch in (red, green, blue, alpha)
        @test isapprox(ch(cw4), ch(gen4); rtol = 1.0e-13, atol = 1.0e-15)
    end

    # endpoints exact (incl. alpha) for the fixed-point transparent family
    v0 = _linear_value_blend(0.0, a0, a1)
    v1 = _linear_value_blend(1.0, a0, a1)
    @test gray(v0) == Float64(gray(a0)) && alpha(v0) == Float64(alpha(a0))
    @test gray(v1) == Float64(gray(a1)) && alpha(v1) == Float64(alpha(a1))
end

@testitem "componentwise colorants: 2D bilinear parity + widened stage re-entry" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    # 2D collapses axis-by-axis: stage 1 blends RGB{N0f8} corners into
    # RGB{Float64}, stage 2 re-enters the style lookup with the WIDENED
    # carrier — a consultation no 1D test reaches.
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 2.0, 3.0, 4.0]
    A = [
        RGB{N0f8}(0.9 * i / 4, 0.1 + 0.2 * j / 4, mod(0.3 * i + 0.5 * j, 1.0))
            for i in 1:4, j in 1:4
    ]
    itp2 = FI.linear_interp((xs, ys), A)
    chans = (red, green, blue)
    itpS = map(ch -> FI.linear_interp((xs, ys), Float64.(ch.(A))), chans)
    for qx in (1.25, 2.5, 3.75), qy in (1.5, 2.75, 3.25)
        v = itp2(qx, qy)
        @test typeof(v) === RGB{Float64}
        for (k, ch) in enumerate(chans)
            @test isapprox(ch(v), itpS[k](qx, qy); rtol = 1.0e-12, atol = 1.0e-14)
        end
    end

    # transparent 2D: alpha rides the same collapse (AGray{N0f8} → AGray{Float64})
    B = [AGray{N0f8}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:4, j in 1:4]
    itpB = FI.linear_interp((xs, ys), B)
    itpG = FI.linear_interp((xs, ys), Float64.(gray.(B)))
    itpA = FI.linear_interp((xs, ys), Float64.(alpha.(B)))
    for qx in (1.75, 3.25), qy in (1.25, 2.5)
        v = itpB(qx, qy)
        @test typeof(v) === AGray{Float64}
        @test isapprox(gray(v), itpG(qx, qy); rtol = 1.0e-12, atol = 1.0e-14)
        @test isapprox(alpha(v), itpA(qx, qy); rtol = 1.0e-12, atol = 1.0e-14)
    end
end

@testitem "componentwise colorants: itp-level inference + zero-alloc" setup = [AllocConstants] begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    x = [1.0, 2.0, 3.0, 4.0]
    c = [
        RGB{N0f8}(0.9, 0.1, 0.5), RGB{N0f8}(0.1, 0.8, 0.2),
        RGB{N0f8}(0.7, 0.3, 0.9), RGB{N0f8}(0.2, 0.6, 0.4),
    ]
    itp = FI.linear_interp(x, c)
    @test @inferred(itp(1.5)) isa RGB{Float64}
    itp(2.5)   # warm
    @test (@allocated itp(2.5)) <= ALLOC_THRESHOLD

    xs = [1.0, 2.0, 3.0]
    A = [RGB{N0f8}(0.2 * i, 0.3 * j, 0.1) for i in 1:3, j in 1:3]
    itp2 = FI.linear_interp((xs, xs), A)
    @test @inferred(itp2(1.5, 2.5)) isa RGB{Float64}
    itp2(1.5, 2.5)   # warm
    @test (@allocated itp2(1.5, 2.5)) <= ND_ALLOC_THRESHOLD
end

@testitem "componentwise colorants: Float32 grid end-to-end" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    # Float32 weights ⇒ Float32 channel math ⇒ RGB{Float32} output — a later
    # Float64 hard-coding in the blend would silently widen this
    x32 = Float32[1.0, 2.0, 3.0]
    c = [RGB{N0f8}(0.9, 0.1, 0.5), RGB{N0f8}(0.1, 0.8, 0.2), RGB{N0f8}(0.7, 0.3, 0.9)]
    itp = FI.linear_interp(x32, c)
    v = itp(1.5f0)
    @test typeof(v) === RGB{Float32}
    gen = muladd(0.5f0, c[2], 0.5f0 * c[1])
    for ch in (red, green, blue)
        @test isapprox(ch(v), ch(gen); rtol = 1.0f-6)
    end
end
