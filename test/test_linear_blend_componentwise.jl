@testitem "linear blend: style rule + styled forms" begin
    using FastInterpolations:
        _LinearBlendFMA,
        _LinearBlendGeneric,
        _LinearBlendStyle,
        _linear_blend_style,
        _linear_value_blend
    using FixedPointNumbers

    # ONE rule on the weighted RESULT type: native-FMA → FMA, else generic
    @test @inferred(_linear_blend_style(Float64, Float64)) === _LinearBlendFMA()
    @test @inferred(_linear_blend_style(Float64, N0f8)) === _LinearBlendFMA()    # result Float64
    @test @inferred(_linear_blend_style(Float32, N0f8)) === _LinearBlendFMA()    # result Float32
    @test @inferred(_linear_blend_style(Float64, ComplexF64)) === _LinearBlendFMA()
    @test @inferred(_linear_blend_style(Float64, BigFloat)) === _LinearBlendGeneric()
    @test @inferred(_linear_blend_style(BigFloat, BigFloat)) === _LinearBlendGeneric()
    @test _linear_blend_style(Int, String) === _LinearBlendGeneric()             # undefined `*` ⇒ Union{}
    @test _linear_blend_style(Union{}) === _LinearBlendGeneric()

    # styled bodies are directly callable (the ext's generic escape relies on
    # this). NOT bitwise: `muladd` may legally fuse or not per compilation
    # context, so the constant-folded literal reference can land one
    # contraction rounding away from the runtime body — tolerance, not bits.
    @test isapprox(
        _linear_value_blend(_LinearBlendFMA(), 0.3, 0.2, 0.9),
        muladd(0.3, 0.9, muladd(-0.3, 0.2, 0.2)); rtol = 1.0e-15,
    )
    @test isapprox(
        _linear_value_blend(_LinearBlendGeneric(), 0.3, 0.2, 0.9),
        muladd(0.3, 0.9, (1 - 0.3) * 0.2); rtol = 1.0e-15,
    )
end

@testitem "linear blend: entry dispatch is a faithful refactor" begin
    using FastInterpolations: _linear_value_blend

    # FMA style: verbatim 2-FMA form (tolerance ∵ muladd contraction is
    # compilation-context-dependent — see the styled-forms testitem), and
    # endpoint-exact (α∈{0,1} is exact under either contraction choice)
    @test isapprox(
        _linear_value_blend(0.3, 0.2, 0.9),
        muladd(0.3, 0.9, muladd(-0.3, 0.2, 0.2)); rtol = 1.0e-15,
    )
    @test _linear_value_blend(0.0, 0.2, 0.9) === 0.2
    @test _linear_value_blend(1.0, 0.2, 0.9) === 0.9
    z1, z2 = 1.0 + 2.0im, 3.0 - 1.0im
    @test isapprox(
        _linear_value_blend(0.3, z1, z2),
        muladd(0.3, z2, muladd(-0.3, z1, z1)); rtol = 1.0e-15,
    )

    # generic style: exact pre-refactor expression, bit-identically
    α, yL, yR = big"0.3", big"0.2", big"0.9"
    @test _linear_value_blend(α, yL, yR) == muladd(α, yR, (one(α) - α) * yL)
    @test _linear_value_blend(big"0.0", yL, yR) == yL
    @test _linear_value_blend(big"1.0", yL, yR) == yR
    @test @inferred(_linear_value_blend(α, yL, yR)) isa BigFloat
end

@testitem "componentwise colorants: ext style rule + ownership" begin
    using FastInterpolations
    using FastInterpolations: _LinearBlendGeneric, _linear_blend_style, _linear_value_blend
    using FixedPointNumbers, ColorTypes, ColorVectorSpace, ForwardDiff
    const FI = FastInterpolations
    EXT = Base.get_extension(FI, :FastInterpolationsColorVectorSpaceExt)
    @test EXT !== nothing
    CW = EXT._LinearBlendComponentwise()

    # the ext extends the CORE style rule: eligible colorant pairs classify
    # as a third, ext-owned style (channel is native-FMA under the weight)
    @test @inferred(_linear_blend_style(Float64, Gray{N0f8})) === CW
    @test @inferred(_linear_blend_style(Float32, RGB{N0f8})) === CW
    @test @inferred(_linear_blend_style(Float64, RGB{Float64})) === CW
    @test @inferred(_linear_blend_style(Float64, AGray{N0f8})) === CW
    @test @inferred(_linear_blend_style(Float64, GrayA{N0f8})) === CW
    @test @inferred(_linear_blend_style(Float64, RGBA{N0f8})) === CW
    @test @inferred(_linear_blend_style(Float64, RGBA{Float64})) === CW
    @test @inferred(_linear_blend_style(Float64, ARGB{N0f8})) === CW
    @test @inferred(_linear_blend_style(Float64, ARGB{Float64})) === CW

    # ineligible weight or channel → generic (rule condition 1)
    @test _linear_blend_style(BigFloat, Gray{BigFloat}) === _LinearBlendGeneric()
    @test _linear_blend_style(ForwardDiff.Dual{Nothing, Float64, 2}, RGB{Float64}) ===
        _LinearBlendGeneric()

    # packed colorants → generic (rule condition 2: mapc reconstruction type
    # Gray24 ≠ natural arithmetic type Gray{Float64} — never re-quantized)
    @test @inferred(_linear_blend_style(Float64, Gray24)) === _LinearBlendGeneric()
    @test @inferred(_linear_blend_style(Float64, RGB24)) === _LinearBlendGeneric()
    @test @inferred(_linear_blend_style(Float64, ARGB32)) === _LinearBlendGeneric()
    # …and the rule's `if` constant-folds: were the branch alive, the packed
    # entry would infer Union{Gray24, Gray{Float64}} and @inferred would fail
    @test @inferred(_linear_value_blend(0.3, Gray24(0.2), Gray24(0.6))) isa Gray{Float64}

    # promotion-match rule generalizes: memory-layout RGB variants opt in
    # automatically (mapc BGR{N0f8} → BGR{Float64} ≡ CVS arithmetic result)
    @test @inferred(_linear_blend_style(Float64, BGR{N0f8})) === CW

    # the inference probe must compute exactly what the componentwise path
    # computes — `promote_op` on it is only truthful while the two stay in
    # sync (tolerance: muladd contraction is compilation-context-dependent)
    p0, p1 = RGB{N0f8}(0.9, 0.1, 0.5), RGB{N0f8}(0.1, 0.8, 0.2)
    probe = EXT._cw_blend_result(0.3, p0, p1)
    shipped = _linear_value_blend(0.3, p0, p1)
    @test typeof(probe) === typeof(shipped)
    for ch in (red, green, blue)
        @test isapprox(ch(probe), ch(shipped); rtol = 1.0e-15)
    end

    # ownership: the ext owns ONLY the style rule for arithmetic colorants
    # (incl. packed — rule-escaped) and the componentwise styled body; the
    # blend ENTRY is the core duck method for every value type, and
    # non-arithmetic color spaces (HSV — CVS defines no arithmetic;
    # channelwise hue blending would be wrong) classify through the core rule
    @test which(_linear_blend_style, Tuple{Type{Float64}, Type{RGB{N0f8}}}).module === EXT
    @test which(_linear_blend_style, Tuple{Type{Float64}, Type{Gray24}}).module === EXT
    @test which(_linear_blend_style, Tuple{Type{Float64}, Type{HSV{Float32}}}).module === FI
    @test which(_linear_value_blend, Tuple{Float64, RGB{N0f8}, RGB{N0f8}}).module === FI
    @test which(
        _linear_value_blend, Tuple{typeof(CW), Float64, RGB{N0f8}, RGB{N0f8}}
    ).module === EXT

    # mixed concrete pair: the entry styles on promote_type → componentwise,
    # but the styled body requires a same-type pair → escape → generic body
    yL, yR = Gray{N0f8}(0.2), Gray{Float32}(0.6)
    @test which(
        _linear_value_blend, Tuple{typeof(CW), Float64, typeof(yL), typeof(yR)}
    ).module === EXT
    @test @inferred(_linear_value_blend(0.3, yL, yR)) ===
        _linear_value_blend(_LinearBlendGeneric(), 0.3, yL, yR)
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
        # layout-variant RGB: promotion-match gate opts it in automatically
        ([BGR{N0f8}(0.9, 0.1, 0.5), BGR{N0f8}(0.1, 0.8, 0.2), BGR{N0f8}(0.7, 0.3, 0.9)], BGR{Float64}),
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

@testitem "componentwise colorants: parity, endpoints, mixed fallback" begin
    using FastInterpolations
    using FastInterpolations: _linear_value_blend
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

    # mixed concrete colorant types miss the ext's same-`C` entries and fall
    # to the CORE generic path — no MethodError, exact generic expression.
    # (Unit-level by design: FI's linear kernel constrains both endpoints to
    # one Tv, so mixed pairs only arise from direct calls / future kernels.)
    ga = Gray{N0f8}(0.2)
    gb = Gray{Float32}(0.5)
    v = _linear_value_blend(0.3, ga, gb)
    @test v == muladd(0.3, gb, (1 - 0.3) * ga)
end

@testitem "componentwise colorants: flat extrapolation inference across 1D ND GQ" begin
    using FastInterpolations
    using FixedPointNumbers, ColorTypes, ColorVectorSpace
    const FI = FastInterpolations

    x = [1.0, 2.0, 3.0]
    c = [
        RGB{N0f8}(0.9, 0.1, 0.5),
        RGB{N0f8}(0.1, 0.8, 0.2),
        RGB{N0f8}(0.7, 0.3, 0.9),
    ]
    fill = RGB{N0f8}(0.2, 0.3, 0.4)
    A = [
        RGB{N0f8}(0.15 * i, 0.2 * j, 0.1 + 0.05 * i)
            for i in 1:3, j in 1:3
    ]

    let extrap = FI.ClampExtrap()
        itp1 = FI.linear_interp(x, c; extrap)

        # 1D flat-extrap shortcut: OOB returns the boundary sample without
        # entering the interpolation kernel, so it must still widen like the
        # in-domain colorant kernel.
        @test @inferred(itp1(1.5)) isa RGB{Float64}
        @test @inferred(itp1(0.5)) isa RGB{Float64}
        @test @inferred(itp1(3.5)) isa RGB{Float64}
        @test @inferred(itp1(0.5; deriv = FI.EvalDeriv1())) isa RGB{Float64}
        @test @inferred(FI.linear_interp(x, c, 0.5; extrap)) isa RGB{Float64}
        @test itp1(0.5) == RGB{Float64}(c[1])
        @test itp1(3.5) == RGB{Float64}(c[end])

        out1 = Vector{RGB{Float64}}(undef, 2)
        @test @inferred(itp1(out1, [0.5, 1.5])) === out1

        # ND ClampExtrap clamps coordinates then uses the normal kernel; keep it
        # in the same contract so a future shortcut cannot reintroduce a raw
        # RGB{N0f8} OOB result.
        itp2 = FI.linear_interp((x, x), A; extrap, store = FI.StorePolicy(; copy = false))
        @test @inferred(itp2(0.5, 1.5)) isa RGB{Float64}
        @test @inferred(itp2((0.5, 1.5))) isa RGB{Float64}
        @test @inferred(itp2(0.5, 1.5; deriv = (FI.EvalDeriv1(), FI.EvalValue()))) isa RGB{Float64}
        @test @inferred(FI.linear_interp((x, x), A, (0.5, 1.5); extrap)) isa RGB{Float64}

        outb = Vector{RGB{Float64}}(undef, 2)
        @test @inferred(itp2(outb, ([0.5, 1.5], [1.5, 2.5]))) === outb

        gq = FI.GriddedQuery(([0.5, 1.5], [1.5, 2.5]))
        @test @inferred(itp2(gq)) isa Matrix{RGB{Float64}}

        outg = Matrix{RGB{Float64}}(undef, 2, 2)
        @test @inferred(itp2(outg, gq)) === outg
        @test (
            @inferred(FI.interp((x, x), A, gq; method = FI.LinearInterp(), extrap))
        ) isa Matrix{RGB{Float64}}

        outg2 = similar(outg)
        @test @inferred(FI.interp!(outg2, (x, x), A, gq; method = FI.LinearInterp(), extrap)) === outg2
    end

    let extrap = FI.FillExtrap(fill)
        itp1 = FI.linear_interp(x, c; extrap)

        # 1D FillExtrap takes the same OOB shortcut as ClampExtrap, but with the
        # fill value as the OOB cell data.
        @test @inferred(itp1(1.5)) isa RGB{Float64}
        @test @inferred(itp1(0.5)) isa RGB{Float64}
        @test @inferred(itp1(3.5)) isa RGB{Float64}
        @test @inferred(itp1(0.5; deriv = FI.EvalDeriv1())) isa RGB{Float64}
        @test @inferred(FI.linear_interp(x, c, 0.5; extrap)) isa RGB{Float64}
        @test itp1(0.5) == RGB{Float64}(fill)
        @test itp1(3.5) == RGB{Float64}(fill)

        out1 = Vector{RGB{Float64}}(undef, 2)
        @test @inferred(itp1(out1, [0.5, 1.5])) === out1

        # ND FillExtrap has its own `_try_fill_oob` shortcut before cell
        # location/evaluation. That shortcut must match the in-domain widened
        # kernel result type for both scalar and batch calls.
        itp2 = FI.linear_interp((x, x), A; extrap, store = FI.StorePolicy(; copy = false))
        @test @inferred(itp2(0.5, 1.5)) isa RGB{Float64}
        @test @inferred(itp2((0.5, 1.5))) isa RGB{Float64}
        @test @inferred(itp2(0.5, 1.5; deriv = (FI.EvalDeriv1(), FI.EvalValue()))) isa RGB{Float64}
        @test @inferred(FI.linear_interp((x, x), A, (0.5, 1.5); extrap)) isa RGB{Float64}
        @test itp2(0.5, 1.5) == RGB{Float64}(fill)

        outb = Vector{RGB{Float64}}(undef, 2)
        @test @inferred(itp2(outb, ([0.5, 1.5], [1.5, 2.5]))) === outb
        @test outb[1] == RGB{Float64}(fill)

        # GriddedQuery FillExtrap uses the separable kernel first, then a slab
        # fill post-pass. The filled slabs must keep the same widened output
        # type as the separable in-domain cells.
        gq = FI.GriddedQuery(([0.5, 1.5], [1.5, 3.5]))
        G = @inferred(itp2(gq))
        @test G isa Matrix{RGB{Float64}}
        @test G[1, 1] == RGB{Float64}(fill)
        @test G[1, 2] == RGB{Float64}(fill)
        @test G[2, 2] == RGB{Float64}(fill)

        outg = Matrix{RGB{Float64}}(undef, 2, 2)
        @test @inferred(itp2(outg, gq)) === outg
        @test outg[1, 1] == RGB{Float64}(fill)

        @test (
            @inferred(FI.interp((x, x), A, gq; method = FI.LinearInterp(), extrap))
        ) isa Matrix{RGB{Float64}}

        outg2 = similar(outg)
        @test @inferred(FI.interp!(outg2, (x, x), A, gq; method = FI.LinearInterp(), extrap)) === outg2
        @test outg2[1, 1] == RGB{Float64}(fill)
    end
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

    # ARGB{Float64}: 4-channel, alpha-first storage — mapc must not permute
    # channels
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
