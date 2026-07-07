# GriddedQuery separable 2D-linear evaluation.
# Per-axis anchors (idx + weight) are precomputed once and reused by two
# strategies: fused (pure downsampling) and two-pass full buffer (otherwise).

@testitem "_GriddedAnchor: builder + extrap fold" begin
    using FastInterpolations
    using FastInterpolations: _GriddedAnchor, _gridded_anchors

    g = collect(1.0:10.0)

    # in-domain: interval index + normalized in-cell weight
    b = _gridded_anchors(g, [3.25], ExtendExtrap(), 1)
    @test eltype(b) == _GriddedAnchor{Float64}
    @test isbits(b[1]) && sizeof(b[1]) == 24
    @test b[1].idx == 3
    @test b[1].alpha ≈ 0.25
    @test b[1].inv_h == 1.0   # unit-step grid → reciprocal cell width is exactly 1

    # Clamp folds the OOB weight to the boundary node; Extend keeps it
    bC = _gridded_anchors(g, [0.0], ClampExtrap(), 1)
    @test bC[1].idx == 1 && bC[1].alpha === 0.0
    bE = _gridded_anchors(g, [0.0], ExtendExtrap(), 1)
    @test bE[1].idx == 1 && bE[1].alpha === -1.0

    # NoExtrap throws on the first OOB coordinate, naming the axis
    @test_throws DomainError _gridded_anchors(g, [0.5], NoExtrap(), 2)
    err = try
        _gridded_anchors(g, [0.5], NoExtrap(), 2)
    catch e
        e
    end
    @test err isa DomainError && occursin("axis 2", sprint(showerror, err))
    # domain endpoints are in-domain
    @test length(_gridded_anchors(g, [1.0, 10.0], NoExtrap(), 1)) == 2

    # Float32 grid + targets stay Float32 (no silent widening)
    b32 = _gridded_anchors(collect(Float32, 1:10), Float32[2.5], ClampExtrap(), 1)
    @test eltype(b32) == _GriddedAnchor{Float32}
    @test b32[1].alpha === 0.5f0
end

@testitem "anchor-build searcher: hint chaining for clustered targets" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_build_searcher, Searcher, BinarySearch,
        LinearBinarySearch, NoHint

    gv = sort!(rand(512)) .* 511.0 .+ 1.0                    # Vector grid
    clustered = collect(range(100.0, 110.0, 256))            # zoom: stride ≪ 1 cell
    sparse = collect(range(2.0, 511.0, 32))                  # stride ≈ 16 cells
    @test _gridded_build_searcher(gv, clustered) isa Searcher{<:LinearBinarySearch}
    @test _gridded_build_searcher(gv, sparse) isa Searcher{BinarySearch, NoHint}
    @test _gridded_build_searcher(gv, [5.0]) isa Searcher{BinarySearch, NoHint}   # M < 2
    # range grids keep their direct arm regardless of clustering
    itp_r = FI.linear_interp((1.0:512.0, 1.0:512.0), rand(512, 512))
    @test !(_gridded_build_searcher(itp_r.grids[1], clustered) isa Searcher{<:LinearBinarySearch})

    # hinted arm produces correct results end-to-end (Vector grid, zoom query)
    A = rand(512, 384)
    gy = sort!(rand(384)) .* 383.0 .+ 1.0
    itp = FI.linear_interp((gv, gy), A; extrap = FI.ClampExtrap())
    tx = collect(range(100.0, 110.0, 96))                    # clustered → hinted
    ty = collect(range(200.0, 204.0, 72))
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "GriddedQuery correctness matrix vs point-wise" begin
    using FastInterpolations
    import FastInterpolations as FI

    # {F64, F32} × {Clamp, Extend} × {up, down, mixed, non-monotonic, M == 1}
    function check(itp, tx, ty; rtol = 1.0e-14, atol = 1.0e-14)
        C = itp(GriddedQuery((tx, ty)))
        ref = [itp((x, y)) for x in tx, y in ty]
        @test eltype(C) == typeof(itp((first(tx), first(ty))))   # output-eltype pin
        @test size(C) == (length(tx), length(ty))
        @test all(isapprox.(C, ref; rtol, atol))
    end

    for TF in (Float64, Float32), ex in (ClampExtrap(), ExtendExtrap())
        rt = TF == Float64 ? 1.0e-14 : 2.0f-6
        A = rand(TF, 32, 24)
        itp = FI.linear_interp((collect(TF, 1:32), collect(TF, 1:24)), A; extrap = ex)
        up = (range(TF(1), TF(32), 57), range(TF(1), TF(24), 41))
        down = (range(TF(1), TF(32), 13), range(TF(1), TF(24), 9))
        mixed = (range(TF(1), TF(32), 57), range(TF(1), TF(24), 9))
        oob = (range(TF(0), TF(33), 20), range(TF(0.5), TF(24.5), 20))  # extrap exercise
        nonmono = (TF[7.3, 2.1, 30.9, 2.1], TF[11.0, 3.5])              # unsorted + repeat
        single = (TF[15.5], TF[8.25])                                    # M == 1
        zoom = (range(TF(10), TF(12), 41), range(TF(5), TF(6), 33))     # dense sub-window
        for (tx, ty) in (up, down, mixed, oob, nonmono, single, zoom)
            check(itp, collect(tx), collect(ty); rtol = rt, atol = rt)
        end
    end

    # Range query axes (image-resize shape) through the public entry
    A = rand(33, 27)
    itpr = FI.linear_interp(
        (range(0.0, 2.0, length = 33), range(-1.0, 3.0, length = 27)), A;
        extrap = FI.ClampExtrap()
    )
    tx = range(0.02, 1.98, length = 71)
    ty = range(-0.9, 2.9, length = 59)
    C = itpr(GriddedQuery((tx, ty)))
    ref = [itpr((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))

    # mixed per-axis extraps (Clamp × Extend)
    A2 = rand(16, 16)
    itp2 = FI.linear_interp((1.0:16.0, 1.0:16.0), A2; extrap = (ClampExtrap(), ExtendExtrap()))
    tx2 = collect(range(0.0, 17.0, 21))
    ty2 = collect(range(0.0, 17.0, 21))
    C2 = itp2(GriddedQuery((tx2, ty2)))
    ref2 = [itp2((x, y)) for x in tx2, y in ty2]
    @test all(isapprox.(C2, ref2; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "strategy cores: fused + both pass orders agree" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_anchors, _gridded_fused!, _gridded_pass!, EvalValue

    A = rand(24, 20)
    itp = FI.linear_interp((1.0:24.0, 1.0:20.0), A; extrap = FI.ClampExtrap())
    # sorted non-range query vectors with repeats (same-cell runs)
    tx = sort!(vcat(1.0 .+ 23.0 .* rand(60), collect(range(2.25, 22.75, 17))))
    ty = sort!(vcat(1.0 .+ 19.0 .* rand(45), collect(range(3.5, 18.5, 12))))
    M, N = length(tx), length(ty)
    ax = _gridded_anchors(itp.grids[1], tx, itp.extraps[1], 1)
    ay = _gridded_anchors(itp.grids[2], ty, itp.extraps[2], 2)
    ref = [itp((x, y)) for x in tx, y in ty]

    Cf = Matrix{Float64}(undef, M, N)
    _gridded_fused!(Cf, A, (ax, ay))
    # explicit pass compositions: axis-2 first vs axis-1 first
    B1 = Matrix{Float64}(undef, 24, N)
    _gridded_pass!(B1, A, ay, EvalValue(), Val(2))
    Cb1 = Matrix{Float64}(undef, M, N)
    _gridded_pass!(Cb1, B1, ax, EvalValue(), Val(1))
    B2 = Matrix{Float64}(undef, M, 20)
    _gridded_pass!(B2, A, ax, EvalValue(), Val(1))
    Cb2 = Matrix{Float64}(undef, M, N)
    _gridded_pass!(Cb2, B2, ay, EvalValue(), Val(2))

    # all strategies are mathematically equivalent — machine-eps, NOT bit-identity
    @test all(isapprox.(Cf, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test all(isapprox.(Cb1, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test all(isapprox.(Cb2, ref; rtol = 1.0e-14, atol = 1.0e-14))

    # the public entry agrees with the cores
    C = itp(GriddedQuery((tx, ty)))
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "pass hull restriction: identical entries, untouched outside" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_anchors, _gridded_hull, _gridded_pass!, EvalValue

    A = rand(64, 48)
    itp = FI.linear_interp((1.0:64.0, 1.0:48.0), A; extrap = FI.ClampExtrap())
    tx = collect(range(20.0, 24.0, 37))     # clustered windows (zoom profile)
    ty = collect(range(30.0, 33.0, 29))
    ax = _gridded_anchors(itp.grids[1], tx, itp.extraps[1], 1)
    ay = _gridded_anchors(itp.grids[2], ty, itp.extraps[2], 2)

    # coordinate-extrema hull == anchor tap extrema (monotone coord → idx map)
    hx = _gridded_hull(itp.grids[1], tx, itp.extraps[1])
    hy = _gridded_hull(itp.grids[2], ty, itp.extraps[2])
    @test first(hx) == minimum(a -> a.idx, ax) && last(hx) == maximum(a -> a.idx, ax) + 1
    @test first(hy) == minimum(a -> a.idx, ay) && last(hy) == maximum(a -> a.idx, ay) + 1
    @test issubset(hx, 1:64) && issubset(hy, 1:48)
    @test length(hy) < 48   # the zoom window is a strict sub-range
    # unsorted targets: extrema (not first/last) drives the hull
    @test _gridded_hull(itp.grids[1], [24.0, 20.0, 22.5], itp.extraps[1]) == hx
    # Wrap folds coordinates → conservative full axis
    @test _gridded_hull(itp.grids[1], tx, FI.WrapExtrap()) == 1:64

    # hull-restricted pass: bit-identical on computed entries, untouched outside
    full = fill(NaN, 37, 48)
    _gridded_pass!(full, A, ax, EvalValue(), Val(1))
    part = fill(NaN, 37, 48)
    _gridded_pass!(part, A, ax, EvalValue(), Val(1), (1:37, hy))   # ranges[1] ignored (D = 1)
    @test part[:, hy] == full[:, hy]
    @test all(isnan, part[:, setdiff(1:48, hy)])

    # public zoom-up (fullbuffer strategy + hull) still matches point-wise
    txu = collect(range(20.0, 24.0, 96))
    tyu = collect(range(30.0, 33.0, 72))
    C = itp(GriddedQuery((txu, tyu)))
    ref = [itp((x, y)) for x in txu, y in tyu]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "3D public path: value/deriv/wrap/fill via GriddedQuery" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: EvalValue, EvalDeriv1

    A = rand(24, 20, 16)
    itp = FI.linear_interp((1.0:24.0, 1.0:20.0, 1.0:16.0), A; extrap = FI.ClampExtrap())
    down = (collect(range(1.5, 23.5, 12)), collect(range(1.5, 19.5, 10)), collect(range(1.5, 15.5, 8)))
    upmix = (collect(range(0.5, 24.5, 31)), collect(range(1.0, 20.0, 27)), collect(range(1.0, 16.0, 9)))
    for ts in (down, upmix)   # fused strategy / multi-pass strategy
        gq = GriddedQuery(ts)
        C = itp(gq)
        ref = [itp((x, y, z)) for x in ts[1], y in ts[2], z in ts[3]]
        @test eltype(C) == eltype(ref)
        @test size(C) == map(length, ts)
        @test all(isapprox.(C, ref; rtol = 1.0e-13, atol = 1.0e-13))

        Cd = itp(gq; deriv = (EvalDeriv1(), EvalValue(), EvalValue()))
        refd = [itp((x, y, z); deriv = (EvalDeriv1(), EvalValue(), EvalValue())) for x in ts[1], y in ts[2], z in ts[3]]
        @test all(isapprox.(Cd, refd; rtol = 1.0e-13, atol = 1.0e-13))

        out = Array{Float64, 3}(undef, map(length, ts))
        @test itp(out, gq) === out
        @test out == C
    end

    # Wrap through the public path
    itpw = FI.linear_interp((1.0:24.0, 1.0:20.0, 1.0:16.0), A; extrap = FI.WrapExtrap())
    tsw = (collect(range(-10.0, 50.0, 17)), collect(range(0.0, 21.0, 11)), [-1.0, 8.5, 17.0])
    Cw = itpw(GriddedQuery(tsw))
    refw = [itpw((x, y, z)) for x in tsw[1], y in tsw[2], z in tsw[3]]
    @test all(isapprox.(Cw, refw; rtol = 1.0e-13, atol = 1.0e-13))

    # Fill through the public path (OOB slab along the MIDDLE axis)
    itpf = FI.linear_interp((1.0:24.0, 1.0:20.0, 1.0:16.0), A; extrap = FI.FillExtrap(NaN))
    tsf = ([2.0, 12.5], [0.5, 8.0, 20.5], [1.0, 15.9])
    Cfill = itpf(GriddedQuery(tsf))
    reff = [itpf((x, y, z)) for x in tsf[1], y in tsf[2], z in tsf[3]]
    @test all(
        i -> isnan(reff[i]) ? isnan(Cfill[i]) :
            isapprox(Cfill[i], reff[i]; rtol = 1.0e-13, atol = 1.0e-13),
        eachindex(Cfill)
    )
    @test count(isnan, Cfill) == count(isnan, reff) > 0
end

@testitem "generated fused kernel: 3D/4D vs point-wise" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_anchors, _gridded_fused!

    # 3D: {Clamp, Extend} incl. OOB targets — anchors fold extrap per axis
    A3 = rand(12, 10, 8)
    grids3 = (1.0:12.0, 1.0:10.0, 1.0:8.0)
    for ex in (FI.ClampExtrap(), FI.ExtendExtrap())
        itp = FI.linear_interp(grids3, A3; extrap = ex)
        ts = (
            collect(range(0.5, 12.5, 9)),
            collect(range(1.0, 10.0, 7)),
            [1.0, 3.75, 7.9],
        )
        anchors = ntuple(d -> _gridded_anchors(itp.grids[d], ts[d], itp.extraps[d], d), 3)
        out = Array{Float64, 3}(undef, map(length, ts)...)
        _gridded_fused!(out, A3, anchors)
        ref = [itp((x, y, z)) for x in ts[1], y in ts[2], z in ts[3]]
        @test all(isapprox.(out, ref; rtol = 1.0e-14, atol = 1.0e-14))
    end

    # 4D: expression nesting depth (15 blends / 16 corners per output)
    A4 = rand(7, 6, 5, 4)
    grids4 = (1.0:7.0, 1.0:6.0, 1.0:5.0, 1.0:4.0)
    itp4 = FI.linear_interp(grids4, A4; extrap = FI.ClampExtrap())
    ts4 = (
        collect(range(1.0, 7.0, 5)),
        [1.5, 4.25],
        collect(range(1.0, 5.0, 4)),
        [2.0, 3.5, 3.9],
    )
    anchors4 = ntuple(d -> _gridded_anchors(itp4.grids[d], ts4[d], itp4.extraps[d], d), 4)
    out4 = Array{Float64, 4}(undef, map(length, ts4)...)
    _gridded_fused!(out4, A4, anchors4)
    ref4 = [itp4((x, y, z, w)) for x in ts4[1], y in ts4[2], z in ts4[3], w in ts4[4]]
    @test all(isapprox.(out4, ref4; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "deriv ops: gridded == point-wise (2D public, 3D kernel)" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_anchors, _gridded_fused!,
        EvalValue, EvalDeriv1, EvalDeriv2

    # 2D public path: {Clamp, Extend} × op combinations, OOB included
    A = rand(24, 20)
    for ex in (FI.ClampExtrap(), FI.ExtendExtrap())
        itp = FI.linear_interp((1.0:24.0, 1.0:20.0), A; extrap = ex)
        tx = collect(range(0.5, 24.5, 31))
        ty = collect(range(1.0, 20.0, 27))
        gq = GriddedQuery((tx, ty))
        for deriv in (
                (EvalDeriv1(), EvalValue()),
                (EvalValue(), EvalDeriv1()),
                (EvalDeriv1(), EvalDeriv1()),   # mixed partial
                (EvalDeriv2(), EvalValue()),    # linear → carrier-aware zero
            )
            C = itp(gq; deriv = deriv)
            ref = [itp((x, y); deriv = deriv) for x in tx, y in ty]
            @test eltype(C) == eltype(ref)
            @test all(isapprox.(C, ref; rtol = 1.0e-13, atol = 1.0e-13))
        end
        # in-place form carries the kwarg too
        out = Matrix{Float64}(undef, 31, 27)
        itp(out, gq; deriv = (EvalDeriv1(), EvalValue()))
        refd = [itp((x, y); deriv = (EvalDeriv1(), EvalValue())) for x in tx, y in ty]
        @test all(isapprox.(out, refd; rtol = 1.0e-13, atol = 1.0e-13))
    end

    # non-uniform Vector grid: inv_h is genuinely per-cell
    gx = cumsum(0.5 .+ rand(24))
    gy = cumsum(0.3 .+ rand(20))
    itpv = FI.linear_interp((gx, gy), rand(24, 20); extrap = FI.ClampExtrap())
    txv = collect(range(gx[1], gx[end], 29))
    tyv = collect(range(gy[1], gy[end], 23))
    Cv = itpv(GriddedQuery((txv, tyv)); deriv = (EvalDeriv1(), EvalValue()))
    refv = [itpv((x, y); deriv = (EvalDeriv1(), EvalValue())) for x in txv, y in tyv]
    @test all(isapprox.(Cv, refv; rtol = 1.0e-12, atol = 1.0e-12))

    # 3D kernel level: per-axis ops through the generated collapse
    A3 = rand(12, 10, 8)
    itp3 = FI.linear_interp((1.0:12.0, 1.0:10.0, 1.0:8.0), A3; extrap = FI.ClampExtrap())
    ts = (collect(range(1.0, 12.0, 9)), collect(range(0.5, 10.5, 7)), [1.0, 3.75, 7.9])
    anchors = ntuple(d -> _gridded_anchors(itp3.grids[d], ts[d], itp3.extraps[d], d), 3)
    for ops in (
            (EvalDeriv1(), EvalValue(), EvalValue()),
            (EvalValue(), EvalDeriv1(), EvalValue()),
            (EvalValue(), EvalDeriv1(), EvalDeriv1()),
        )
        out3 = Array{Float64, 3}(undef, map(length, ts)...)
        _gridded_fused!(out3, A3, anchors, ops)
        ref3 = [itp3((x, y, z); deriv = ops) for x in ts[1], y in ts[2], z in ts[3]]
        @test all(isapprox.(out3, ref3; rtol = 1.0e-13, atol = 1.0e-13))
    end
end

@testitem "WrapExtrap: gridded == point-wise (2D public, 3D kernel)" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_anchors, _gridded_fused!

    A = rand(16, 12)
    itp = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FI.WrapExtrap())
    tx = collect(range(-14.0, 45.0, 41))    # several periods out on both sides
    ty = collect(range(-3.0, 26.0, 29))
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-13, atol = 1.0e-13))

    # closed-domain contract: xq == last(x) hits the endpoint exactly
    Ce = itp(GriddedQuery(([16.0], [12.0])))
    @test Ce[1, 1] == A[16, 12]

    # mixed per-axis (Wrap × Clamp)
    itpm = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = (FI.WrapExtrap(), FI.ClampExtrap()))
    Cm = itpm(GriddedQuery((tx, ty)))
    refm = [itpm((x, y)) for x in tx, y in ty]
    @test all(isapprox.(Cm, refm; rtol = 1.0e-13, atol = 1.0e-13))

    # 3D kernel level
    A3 = rand(10, 9, 8)
    itp3 = FI.linear_interp((1.0:10.0, 1.0:9.0, 1.0:8.0), A3; extrap = FI.WrapExtrap())
    ts = (collect(range(-8.0, 28.0, 13)), collect(range(0.0, 17.0, 9)), [-1.0, 4.5, 9.25])
    anchors = ntuple(d -> _gridded_anchors(itp3.grids[d], ts[d], itp3.extraps[d], d), 3)
    out3 = Array{Float64, 3}(undef, map(length, ts)...)
    _gridded_fused!(out3, A3, anchors)
    ref3 = [itp3((x, y, z)) for x in ts[1], y in ts[2], z in ts[3]]
    @test all(isapprox.(out3, ref3; rtol = 1.0e-13, atol = 1.0e-13))
end

@testitem "FillExtrap: OOB slabs filled, matches point-wise" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: EvalValue, EvalDeriv1

    agree(C, ref) = all(
        i -> isnan(ref[i]) ? isnan(C[i]) :
            isapprox(C[i], ref[i]; rtol = 1.0e-13, atol = 1.0e-13),
        eachindex(C)
    )

    A = rand(16, 12)
    itp = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FI.FillExtrap(NaN))
    tx = [0.25, 1.5, 8.0, 16.75]    # 2 OOB rows
    ty = [1.0, 6.5, 12.5]           # 1 OOB column
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test agree(C, ref)
    @test count(isnan, C) == count(isnan, ref) > 0

    # numeric fill + deriv (fill result for any deriv op is a carrier zero)
    itp0 = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FI.FillExtrap(-7.5))
    C0 = itp0(GriddedQuery((tx, ty)))
    ref0 = [itp0((x, y)) for x in tx, y in ty]
    @test agree(C0, ref0)
    Cd = itp0(GriddedQuery((tx, ty)); deriv = (EvalDeriv1(), EvalValue()))
    refd = [itp0((x, y); deriv = (EvalDeriv1(), EvalValue())) for x in tx, y in ty]
    @test agree(Cd, refd)

    # mixed Fill × Clamp: OOB on the Clamp axis clamps (not filled)
    itpm = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = (FI.FillExtrap(NaN), FI.ClampExtrap()))
    Cm = itpm(GriddedQuery((tx, ty)))
    refm = [itpm((x, y)) for x in tx, y in ty]
    @test agree(Cm, refm)
    @test !isnan(Cm[3, 3])   # in-domain x, OOB y on Clamp axis → clamped value
end

@testitem "NoExtrap validation + unsupported-extrap guards" begin
    using FastInterpolations
    import FastInterpolations as FI

    A = rand(16, 12)
    # NoExtrap: in-domain == point-wise; OOB throws BEFORE any work, naming the axis
    itp = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FI.NoExtrap())
    tx = collect(range(1.0, 16.0, 21))
    ty = collect(range(1.0, 12.0, 17))
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test_throws DomainError itp(GriddedQuery(([0.5, 8.0], ty)))          # axis 1 OOB
    @test_throws DomainError itp(GriddedQuery((tx, [1.0, 12.001])))       # axis 2 OOB
    err = try
        itp(GriddedQuery(([0.5], ty)))
    catch e
        e
    end
    @test err isa DomainError && occursin("axis 1", sprint(showerror, err))

    # Structural guarantee: the allocating path builds (and validates) the
    # anchors BEFORE allocating the O(M·N) output, so a throwing query must
    # not leave the full output matrix behind.
    big_tx = collect(range(1.0, 16.0, 500))
    bad_ty = vcat(collect(range(1.0, 12.0, 399)), [12.5])   # OOB tail on axis 2
    gq_big = GriddedQuery((big_tx, bad_ty))
    @test_throws DomainError itp(gq_big)
    function alloc_on_throw(itp, gq)
        try
            itp(gq)   # warmup: compile the throwing method path
        catch e
            e isa DomainError || rethrow()
        end
        return @allocated try
            itp(gq)
        catch e
            e isa DomainError || rethrow()
        end
    end
    a_throw = alloc_on_throw(itp, gq_big)
    @test a_throw < length(big_tx) * length(bad_ty) * sizeof(Float64)
end

@testitem "call-time extrap override: InBounds anchor fast path" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: EvalValue, EvalDeriv1

    A = rand(24, 20)
    itp = FI.linear_interp((1.0:24.0, 1.0:20.0), A; extrap = FI.NoExtrap())
    tx = collect(range(1.0, 24.0, 31))
    ty = collect(range(1.0, 20.0, 27))
    gq = GriddedQuery((tx, ty))
    C0 = itp(gq)

    # single InBounds broadcasts to all axes; in-domain queries take the same
    # lean search the default path uses internally → bit-identical
    C1 = itp(gq; extrap = FI.InBounds())
    @test C1 == C0
    # per-axis tuple form
    C2 = itp(gq; extrap = (FI.InBounds(), FI.InBounds()))
    @test C2 == C0
    # in-place form
    out = similar(C0)
    @test itp(out, gq; extrap = FI.InBounds()) === out
    @test out == C0
    # composes with deriv
    Cd0 = itp(gq; deriv = (EvalDeriv1(), EvalValue()))
    Cd1 = itp(gq; deriv = (EvalDeriv1(), EvalValue()), extrap = FI.InBounds())
    @test Cd1 == Cd0
    # matches point-wise under the same override
    refI = [itp((x, y); extrap = FI.InBounds()) for x in tx, y in ty]
    @test all(isapprox.(C1, refI; rtol = 1.0e-14, atol = 1.0e-14))
    # single non-InBounds mode never broadcasts (point-wise contract)
    @test_throws ArgumentError itp(gq; extrap = FI.ClampExtrap())

    # 3D
    A3 = rand(12, 10, 8)
    itp3 = FI.linear_interp((1.0:12.0, 1.0:10.0, 1.0:8.0), A3; extrap = FI.ClampExtrap())
    ts3 = (collect(range(1.0, 12.0, 15)), collect(range(1.0, 10.0, 13)), [1.0, 4.5, 8.0])
    C3 = itp3(GriddedQuery(ts3))
    C3I = itp3(GriddedQuery(ts3); extrap = FI.InBounds())
    @test C3I == C3
end

@testitem "in-place API: pooled buffers, zero-alloc, edges" begin
    using FastInterpolations
    import FastInterpolations as FI

    A = rand(48, 40)
    itp = FI.linear_interp((1.0:48.0, 1.0:40.0), A; extrap = FI.ClampExtrap())
    tx = collect(range(1.0, 48.0, 100))
    ty = collect(range(1.0, 40.0, 80))
    gq = GriddedQuery((tx, ty))

    # in-place == allocating (same anchors + same strategy → bit-equal)
    out = Matrix{Float64}(undef, 100, 80)
    ret = itp(out, gq)
    @test ret === out
    @test out == itp(gq)

    # wrong size → DimensionMismatch
    @test_throws DimensionMismatch itp(Matrix{Float64}(undef, 99, 80), gq)

    # empty axes: empty output, no throw
    @test size(itp(GriddedQuery((Float64[], ty)))) == (0, 80)
    @test size(itp(GriddedQuery((tx, Float64[])))) == (100, 0)
    e0 = Matrix{Float64}(undef, 0, 80)
    @test itp(e0, GriddedQuery((Float64[], ty))) === e0

    # zero allocation after warmup: anchors AND the fullbuffer intermediate
    # are pool-backed. Bound allows fixed pool/bookkeeping drift but stays far
    # below one output-sized Float64 scratch (100*80*8 = 64000 B).
    function alloc_count(itp, out, gq)
        itp(out, gq)               # warmup (pool grows here)
        return @allocated itp(out, gq)
    end
    allocs = alloc_count(itp, out, gq)
    @test allocs <= 1024

    # a pure-downsampling query exercises the fused strategy's pool scope
    gq_down = GriddedQuery((collect(range(1.0, 48.0, 20)), collect(range(1.0, 40.0, 16))))
    out_down = Matrix{Float64}(undef, 20, 16)
    allocs_down = alloc_count(itp, out_down, gq_down)
    @test allocs_down <= 1024
end

@testitem "exact-node queries: bit-exact via endpoint-exact blend" begin
    using FastInterpolations
    import FastInterpolations as FI

    A = rand(32, 24)
    itp = FI.linear_interp((1.0:32.0, 1.0:24.0), A; extrap = FI.ClampExtrap())

    # node-aligned axis-2 targets (alpha ∈ {0, 1}): `_linear_value_blend` is
    # endpoint-exact, so node hits reproduce point-wise results bit-exactly —
    # no special-case copy path needed.
    ty_nodes = collect(3.0:2.0:23.0)
    tx = collect(range(1.0, 32.0, 45))
    C = itp(GriddedQuery((tx, ty_nodes)))
    ref = [itp((x, y)) for x in tx, y in ty_nodes]
    @test C == ref

    # both axes ≡ grid nodes → exact reproduction of the data
    Cid = itp(GriddedQuery((collect(1.0:32.0), collect(1.0:24.0))))
    @test Cid == A
    out = Matrix{Float64}(undef, 32, 24)
    @test itp(out, GriddedQuery((collect(1.0:32.0), collect(1.0:24.0)))) == A
end
