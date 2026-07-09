# GriddedQuery separable 2D-linear evaluation.
# Per-axis anchors are precomputed once and reused by two strategies: fused
# (pure downsampling) and multi-pass full buffer (otherwise).

@testitem "_AxisAnchor: linear builder + extrap fold" setup = [Basic] begin
    using FastInterpolations: _AxisAnchor, _gridded_anchors

    g = collect(1.0:10.0)
    linear_anchor_type = _AxisAnchor{typeof(LinearInterp(NoBC())), Tuple{Float64, Float64}}
    @test !isdefined(FI, :_GriddedAnchor)

    # in-domain: interval index + normalized in-cell weight
    b = _gridded_anchors(g, [3.25], ExtendExtrap(), 1)
    @test eltype(b) == linear_anchor_type
    @test isbits(b[1]) && sizeof(b[1]) == 24
    @test b[1].idx == 3
    @test b[1].payload[1] ≈ 0.25
    @test b[1].payload[2] == 1.0   # unit-step grid → reciprocal cell width is exactly 1

    # Clamp folds the OOB weight to the boundary node; Extend keeps it
    bC = _gridded_anchors(g, [0.0], ClampExtrap(), 1)
    @test bC[1].idx == 1 && bC[1].payload[1] === 0.0
    bE = _gridded_anchors(g, [0.0], ExtendExtrap(), 1)
    @test bE[1].idx == 1 && bE[1].payload[1] === -1.0

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
    @test eltype(b32) == _AxisAnchor{typeof(LinearInterp(NoBC())), Tuple{Float32, Float32}}
    @test b32[1].payload[1] === 0.5f0

    # Exclusive periodic axes are the only linear gridded anchors that need to
    # retain a right tap; the seam cell wraps to the physical first node.
    bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
    gx = FI._cache_axis(collect(0.0:3.0), bc)
    bx = _gridded_anchors(gx, [3.75], WrapExtrap(), 1)
    @test eltype(bx) == _AxisAnchor{typeof(LinearInterp(NoBC())), Tuple{Int, Float64, Float64}}
    @test bx[1].idx == 4
    @test bx[1].payload[1] == 1
    @test bx[1].payload[2] ≈ 0.75
end

@testitem "anchor-build searcher: hint chaining for clustered targets" setup = [Basic] begin
    using FastInterpolations: _gridded_build_searcher, Searcher, BinarySearch,
        LinearBinarySearch, NoHint

    gv = sort!(rand(512)) .* 511.0 .+ 1.0                    # Vector grid
    clustered = range(100.0, 110.0, 256)            # zoom: stride ≪ 1 cell
    sparse = range(2.0, 511.0, 32)                  # stride ≈ 16 cells
    @test _gridded_build_searcher(gv, clustered) isa Searcher{<:LinearBinarySearch}
    @test _gridded_build_searcher(gv, sparse) isa Searcher{BinarySearch, NoHint}
    @test _gridded_build_searcher(gv, [5.0]) isa Searcher{BinarySearch, NoHint}   # M < 2
    # range grids keep their direct arm regardless of clustering
    itp_r = linear_interp((1.0:512.0, 1.0:512.0), rand(512, 512))
    @test !(_gridded_build_searcher(itp_r.grids[1], clustered) isa Searcher{<:LinearBinarySearch})

    # hinted arm produces correct results end-to-end (Vector grid, zoom query)
    A = rand(512, 384)
    gy = sort!(rand(384)) .* 383.0 .+ 1.0
    itp = linear_interp((gv, gy), A; extrap = ClampExtrap())
    tx = range(100.0, 110.0, 96)                    # clustered → hinted
    ty = range(200.0, 204.0, 72)
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "GriddedQuery correctness matrix vs point-wise" setup = [Basic] begin
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
        itp = linear_interp((collect(TF, 1:32), collect(TF, 1:24)), A; extrap = ex)
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
    itpr = linear_interp(
        (range(0.0, 2.0, length = 33), range(-1.0, 3.0, length = 27)), A;
        extrap = ClampExtrap()
    )
    tx = range(0.02, 1.98, length = 71)
    ty = range(-0.9, 2.9, length = 59)
    C = itpr(GriddedQuery((tx, ty)))
    ref = [itpr((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))

    # mixed per-axis extraps (Clamp × Extend)
    A2 = rand(16, 16)
    itp2 = linear_interp((1.0:16.0, 1.0:16.0), A2; extrap = (ClampExtrap(), ExtendExtrap()))
    tx2 = range(0.0, 17.0, 21)
    ty2 = range(0.0, 17.0, 21)
    C2 = itp2(GriddedQuery((tx2, ty2)))
    ref2 = [itp2((x, y)) for x in tx2, y in ty2]
    @test all(isapprox.(C2, ref2; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "strategy cores: fused + both pass orders agree" setup = [Basic] begin
    using FastInterpolations: _gridded_anchors, _gridded_fused!, _gridded_pass!, EvalValue

    A = rand(24, 20)
    itp = linear_interp((1.0:24.0, 1.0:20.0), A; extrap = ClampExtrap())
    # sorted non-range query vectors with repeats (same-cell runs)
    tx = sort!(vcat(1.0 .+ 23.0 .* rand(60), range(2.25, 22.75, 17)))
    ty = sort!(vcat(1.0 .+ 19.0 .* rand(45), range(3.5, 18.5, 12)))
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

@testitem "pass hull restriction: identical entries, untouched outside" setup = [Basic] begin
    using FastInterpolations: _gridded_anchors, _gridded_hull, _gridded_pass!, EvalValue

    A = rand(64, 48)
    itp = linear_interp((1.0:64.0, 1.0:48.0), A; extrap = ClampExtrap())
    tx = range(20.0, 24.0, 37)     # clustered windows (zoom profile)
    ty = range(30.0, 33.0, 29)
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
    @test _gridded_hull(itp.grids[1], tx, WrapExtrap()) == 1:64

    # hull-restricted pass: bit-identical on computed entries, untouched outside
    full = fill(NaN, 37, 48)
    _gridded_pass!(full, A, ax, EvalValue(), Val(1))
    part = fill(NaN, 37, 48)
    _gridded_pass!(part, A, ax, EvalValue(), Val(1), (1:37, hy))   # ranges[1] ignored (D = 1)
    @test part[:, hy] == full[:, hy]
    @test all(isnan, part[:, setdiff(1:48, hy)])

    # public zoom-up (fullbuffer strategy + hull) still matches point-wise
    txu = range(20.0, 24.0, 96)
    tyu = range(30.0, 33.0, 72)
    C = itp(GriddedQuery((txu, tyu)))
    ref = [itp((x, y)) for x in txu, y in tyu]
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "3D public path: value/deriv/wrap/fill via GriddedQuery" setup = [Basic] begin
    using FastInterpolations: EvalValue, EvalDeriv1

    A = rand(24, 20, 16)
    itp = linear_interp((1.0:24.0, 1.0:20.0, 1.0:16.0), A; extrap = ClampExtrap())
    down = (range(1.5, 23.5, 12), range(1.5, 19.5, 10), range(1.5, 15.5, 8))
    upmix = (range(0.5, 24.5, 31), range(1.0, 20.0, 27), range(1.0, 16.0, 9))
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
    itpw = linear_interp((1.0:24.0, 1.0:20.0, 1.0:16.0), A; extrap = WrapExtrap())
    tsw = (range(-10.0, 50.0, 17), range(0.0, 21.0, 11), [-1.0, 8.5, 17.0])
    Cw = itpw(GriddedQuery(tsw))
    refw = [itpw((x, y, z)) for x in tsw[1], y in tsw[2], z in tsw[3]]
    @test all(isapprox.(Cw, refw; rtol = 1.0e-13, atol = 1.0e-13))

    # Fill through the public path (OOB slab along the MIDDLE axis)
    itpf = linear_interp((1.0:24.0, 1.0:20.0, 1.0:16.0), A; extrap = FillExtrap(NaN))
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

@testitem "generated fused kernel: 3D/4D vs point-wise" setup = [Basic] begin
    using FastInterpolations: _gridded_anchors, _gridded_fused!

    # 3D: {Clamp, Extend} incl. OOB targets — anchors fold extrap per axis
    A3 = rand(12, 10, 8)
    grids3 = (1.0:12.0, 1.0:10.0, 1.0:8.0)
    for ex in (ClampExtrap(), ExtendExtrap())
        itp = linear_interp(grids3, A3; extrap = ex)
        ts = (
            range(0.5, 12.5, 9),
            range(1.0, 10.0, 7),
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
    itp4 = linear_interp(grids4, A4; extrap = ClampExtrap())
    ts4 = (
        range(1.0, 7.0, 5),
        [1.5, 4.25],
        range(1.0, 5.0, 4),
        [2.0, 3.5, 3.9],
    )
    anchors4 = ntuple(d -> _gridded_anchors(itp4.grids[d], ts4[d], itp4.extraps[d], d), 4)
    out4 = Array{Float64, 4}(undef, map(length, ts4)...)
    _gridded_fused!(out4, A4, anchors4)
    ref4 = [itp4((x, y, z, w)) for x in ts4[1], y in ts4[2], z in ts4[3], w in ts4[4]]
    @test all(isapprox.(out4, ref4; rtol = 1.0e-14, atol = 1.0e-14))
end

@testitem "deriv ops: gridded == point-wise (2D public, 3D kernel)" setup = [Basic] begin
    using FastInterpolations: _gridded_anchors, _gridded_fused!,
        EvalValue, EvalDeriv1, EvalDeriv2

    # 2D public path: {Clamp, Extend} × op combinations, OOB included
    A = rand(24, 20)
    for ex in (ClampExtrap(), ExtendExtrap())
        itp = linear_interp((1.0:24.0, 1.0:20.0), A; extrap = ex)
        tx = range(0.5, 24.5, 31)
        ty = range(1.0, 20.0, 27)
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
    itpv = linear_interp((gx, gy), rand(24, 20); extrap = ClampExtrap())
    txv = range(gx[1], gx[end], 29)
    tyv = range(gy[1], gy[end], 23)
    Cv = itpv(GriddedQuery((txv, tyv)); deriv = (EvalDeriv1(), EvalValue()))
    refv = [itpv((x, y); deriv = (EvalDeriv1(), EvalValue())) for x in txv, y in tyv]
    @test all(isapprox.(Cv, refv; rtol = 1.0e-12, atol = 1.0e-12))

    # 3D kernel level: per-axis ops through the generated collapse
    A3 = rand(12, 10, 8)
    itp3 = linear_interp((1.0:12.0, 1.0:10.0, 1.0:8.0), A3; extrap = ClampExtrap())
    ts = (range(1.0, 12.0, 9), range(0.5, 10.5, 7), [1.0, 3.75, 7.9])
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

@testitem "WrapExtrap: gridded == point-wise (2D public, 3D kernel)" setup = [Basic] begin
    using FastInterpolations: _gridded_anchors, _gridded_fused!

    A = rand(16, 12)
    itp = linear_interp((1.0:16.0, 1.0:12.0), A; extrap = WrapExtrap())
    tx = range(-14.0, 45.0, 41)    # several periods out on both sides
    ty = range(-3.0, 26.0, 29)
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test all(isapprox.(C, ref; rtol = 1.0e-13, atol = 1.0e-13))

    # closed-domain contract: xq == last(x) hits the endpoint exactly
    Ce = itp(GriddedQuery(([16.0], [12.0])))
    @test Ce[1, 1] == A[16, 12]

    # mixed per-axis (Wrap × Clamp)
    itpm = linear_interp((1.0:16.0, 1.0:12.0), A; extrap = (WrapExtrap(), ClampExtrap()))
    Cm = itpm(GriddedQuery((tx, ty)))
    refm = [itpm((x, y)) for x in tx, y in ty]
    @test all(isapprox.(Cm, refm; rtol = 1.0e-13, atol = 1.0e-13))

    # 3D kernel level
    A3 = rand(10, 9, 8)
    itp3 = linear_interp((1.0:10.0, 1.0:9.0, 1.0:8.0), A3; extrap = WrapExtrap())
    ts = (range(-8.0, 28.0, 13), range(0.0, 17.0, 9), [-1.0, 4.5, 9.25])
    anchors = ntuple(d -> _gridded_anchors(itp3.grids[d], ts[d], itp3.extraps[d], d), 3)
    out3 = Array{Float64, 3}(undef, map(length, ts)...)
    _gridded_fused!(out3, A3, anchors)
    ref3 = [itp3((x, y, z)) for x in ts[1], y in ts[2], z in ts[3]]
    @test all(isapprox.(out3, ref3; rtol = 1.0e-13, atol = 1.0e-13))
end

@testitem "FillExtrap: OOB slabs filled, matches point-wise" setup = [Basic] begin
    using FastInterpolations: EvalValue, EvalDeriv1

    agree(C, ref) = all(
        i -> isnan(ref[i]) ? isnan(C[i]) :
            isapprox(C[i], ref[i]; rtol = 1.0e-13, atol = 1.0e-13),
        eachindex(C)
    )

    A = rand(16, 12)
    itp = linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FillExtrap(NaN))
    tx = [0.25, 1.5, 8.0, 16.75]    # 2 OOB rows
    ty = [1.0, 6.5, 12.5]           # 1 OOB column
    C = itp(GriddedQuery((tx, ty)))
    ref = [itp((x, y)) for x in tx, y in ty]
    @test agree(C, ref)
    @test count(isnan, C) == count(isnan, ref) > 0

    # numeric fill + deriv (fill result for any deriv op is a carrier zero)
    itp0 = linear_interp((1.0:16.0, 1.0:12.0), A; extrap = FillExtrap(-7.5))
    C0 = itp0(GriddedQuery((tx, ty)))
    ref0 = [itp0((x, y)) for x in tx, y in ty]
    @test agree(C0, ref0)
    Cd = itp0(GriddedQuery((tx, ty)); deriv = (EvalDeriv1(), EvalValue()))
    refd = [itp0((x, y); deriv = (EvalDeriv1(), EvalValue())) for x in tx, y in ty]
    @test agree(Cd, refd)

    # mixed Fill × Clamp: OOB on the Clamp axis clamps (not filled)
    itpm = linear_interp((1.0:16.0, 1.0:12.0), A; extrap = (FillExtrap(NaN), ClampExtrap()))
    Cm = itpm(GriddedQuery((tx, ty)))
    refm = [itpm((x, y)) for x in tx, y in ty]
    @test agree(Cm, refm)
    @test !isnan(Cm[3, 3])   # in-domain x, OOB y on Clamp axis → clamped value
end

@testitem "NoExtrap validation + unsupported-extrap guards" setup = [Basic] begin
    A = rand(16, 12)
    # NoExtrap: in-domain == point-wise; OOB throws BEFORE any work, naming the axis
    itp = linear_interp((1.0:16.0, 1.0:12.0), A; extrap = NoExtrap())
    tx = range(1.0, 16.0, 21)
    ty = range(1.0, 12.0, 17)
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
    # canonical axis-named message: same phrasing as scalar/ND, plus the offending
    # axis and the physical domain bounds
    msg = sprint(showerror, err)
    @test err isa DomainError
    @test occursin("query point on axis 1 outside interpolation domain", msg)
    @test occursin("[1.0, 16.0]", msg)

    # Structural guarantee: the allocating path builds (and validates) the
    # anchors BEFORE allocating the O(M·N) output, so a throwing query must
    # not leave the full output matrix behind.
    big_tx = range(1.0, 16.0, 500)
    bad_ty = vcat(range(1.0, 12.0, 399), [12.5])   # OOB tail on axis 2
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

@testitem "call-time extrap override: InBounds anchor fast path" setup = [Basic] begin
    using FastInterpolations: EvalValue, EvalDeriv1

    A = rand(24, 20)
    itp = linear_interp((1.0:24.0, 1.0:20.0), A; extrap = NoExtrap())
    tx = range(1.0, 24.0, 31)
    ty = range(1.0, 20.0, 27)
    gq = GriddedQuery((tx, ty))
    C0 = itp(gq)

    # single InBounds broadcasts to all axes; in-domain queries take the same
    # lean search the default path uses internally → bit-identical
    C1 = itp(gq; extrap = InBounds())
    @test C1 == C0
    # per-axis tuple form
    C2 = itp(gq; extrap = (InBounds(), InBounds()))
    @test C2 == C0
    # in-place form
    out = similar(C0)
    @test itp(out, gq; extrap = InBounds()) === out
    @test out == C0
    # composes with deriv
    Cd0 = itp(gq; deriv = (EvalDeriv1(), EvalValue()))
    Cd1 = itp(gq; deriv = (EvalDeriv1(), EvalValue()), extrap = InBounds())
    @test Cd1 == Cd0
    # matches point-wise under the same override
    refI = [itp((x, y); extrap = InBounds()) for x in tx, y in ty]
    @test all(isapprox.(C1, refI; rtol = 1.0e-14, atol = 1.0e-14))
    # single non-InBounds mode never broadcasts (point-wise contract)
    @test_throws ArgumentError itp(gq; extrap = ClampExtrap())

    # 3D
    A3 = rand(12, 10, 8)
    itp3 = linear_interp((1.0:12.0, 1.0:10.0, 1.0:8.0), A3; extrap = ClampExtrap())
    ts3 = (range(1.0, 12.0, 15), range(1.0, 10.0, 13), [1.0, 4.5, 8.0])
    C3 = itp3(GriddedQuery(ts3))
    C3I = itp3(GriddedQuery(ts3); extrap = InBounds())
    @test C3I == C3
end

@testitem "in-place API: pooled buffers, zero-alloc, edges" setup = [Basic] begin
    A = rand(48, 40)
    itp = linear_interp((1.0:48.0, 1.0:40.0), A; extrap = ClampExtrap())
    tx = range(1.0, 48.0, 100)
    ty = range(1.0, 40.0, 80)
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
    gq_down = GriddedQuery((range(1.0, 48.0, 20), range(1.0, 40.0, 16)))
    out_down = Matrix{Float64}(undef, 20, 16)
    allocs_down = alloc_count(itp, out_down, gq_down)
    @test allocs_down <= 1024
end

@testitem "exact-node queries: bit-exact via endpoint-exact blend" setup = [Basic] begin
    A = rand(32, 24)
    itp = linear_interp((1.0:32.0, 1.0:24.0), A; extrap = ClampExtrap())

    # node-aligned axis-2 targets (alpha ∈ {0, 1}): `_linear_value_blend` is
    # endpoint-exact, so node hits reproduce point-wise results bit-exactly —
    # no special-case copy path needed.
    ty_nodes = collect(3.0:2.0:23.0)
    tx = range(1.0, 32.0, 45)
    C = itp(GriddedQuery((tx, ty_nodes)))
    ref = [itp((x, y)) for x in tx, y in ty_nodes]
    @test C == ref

    # both axes ≡ grid nodes → exact reproduction of the data
    Cid = itp(GriddedQuery((collect(1.0:32.0), collect(1.0:24.0))))
    @test Cid == A
    out = Matrix{Float64}(undef, 32, 24)
    @test itp(out, GriddedQuery((collect(1.0:32.0), collect(1.0:24.0)))) == A
end

@testitem "one-shot GriddedQuery: linear_interp(grids, data, gq) == persistent" setup = [Basic] begin
    # Reference persistent interpolants for each grid flavor; the one-shot form
    # must reproduce them bit-for-bit (same value-matched Tg, same anchors,
    # same fused/fullbuffer strategy) — it is a thin front over the same eval.
    for (grids, data, extrap) in (
            ((1.0:64.0, 1.0:48.0), rand(64, 48), ClampExtrap()),
            ((collect(1.0:64.0), collect(1.0:48.0)), rand(64, 48), ClampExtrap()),
            ((Base.OneTo(64), Base.OneTo(48)), rand(64, 48), NoExtrap()),
            ((sort!(rand(64)) .* 63 .+ 1, sort!(rand(48)) .* 47 .+ 1), rand(64, 48), ClampExtrap()),
        )
        itp = linear_interp(grids, data; extrap = extrap, store = StorePolicy(; copy = false))
        tx = range(2.0, 63.0, 120)
        ty = range(2.0, 47.0, 90)
        gq = GriddedQuery((tx, ty))

        # allocating one-shot == persistent
        os = linear_interp(grids, data, gq; extrap = extrap)
        @test os == itp(gq)

        # in-place one-shot == persistent
        out = Matrix{eltype(os)}(undef, 120, 90)
        @test linear_interp!(out, grids, data, gq; extrap = extrap) === out
        @test out == itp(gq)

        # per-axis derivative op matches persistent
        osd = linear_interp(grids, data, gq; extrap = extrap, deriv = (EvalDeriv1(), EvalValue()))
        @test osd == itp(gq; deriv = (EvalDeriv1(), EvalValue()))
    end
end

@testitem "one-shot GriddedQuery: linear PeriodicBC exclusive seam parity" setup = [Basic] begin
    # 1D named one-shot GriddedQuery must follow the BC-aware 1D path: the
    # exclusive seam cell is virtual, so the right endpoint wraps to y[1].
    x = range(0.0, 3.0, length = 4)
    y = [0.0, 10.0, 20.0, 30.0]
    bc = PeriodicBC(endpoint = :exclusive, period = 4.0)
    tx = [-0.5, 0.0, 0.25, 2.5, 3.25, 3.75, 4.0, 4.5]
    gq = GriddedQuery((tx,))
    os = linear_interp((x,), y, gq; bc = bc)
    ref_vec = linear_interp(x, y, tx; bc = bc)
    itp = linear_interp(x, y; bc = bc)
    @test vec(os) ≈ ref_vec atol = 1.0e-14 rtol = 1.0e-14
    @test vec(os) ≈ itp(tx) atol = 1.0e-14 rtol = 1.0e-14
    osd = linear_interp((x,), y, gq; bc = bc, deriv = EvalDeriv1())
    ref_vec_d = linear_interp(x, y, tx; bc = bc, deriv = EvalDeriv1())
    @test vec(osd) ≈ ref_vec_d atol = 1.0e-14 rtol = 1.0e-14

    out = similar(os)
    @test linear_interp!(out, (x,), y, gq; bc = (bc,)) === out
    @test vec(out) ≈ ref_vec atol = 1.0e-14 rtol = 1.0e-14
    @test_throws ArgumentError linear_interp((x,), y, gq; bc = PeriodicBC(endpoint = :exclusive, period = 2.0))
    @test_throws ArgumentError linear_interp((x,), y, GriddedQuery(([0.25],)); bc = PeriodicBC())

    # 2D one-shot must match point-wise scalar one-shot on seam and OOB
    # wrapped coordinates, including a mixed periodic/non-periodic BC tuple.
    xs = range(0.0, 3.0, length = 4)
    ys = range(-1.0, 1.0, length = 5)
    data = [sin(xi) + 0.2cos(3yj) for xi in xs, yj in ys]
    tx2 = [-0.25, 0.0, 2.75, 3.25, 3.75, 4.25]
    ty2 = [-1.0, -0.25, 0.7, 1.0]
    bc2 = (bc, NoBC())
    gq2 = GriddedQuery((tx2, ty2))
    got = linear_interp((xs, ys), data, gq2; bc = bc2, extrap = (NoExtrap(), NoExtrap()))
    ref = [
        linear_interp((xs, ys), data, (qx, qy); bc = bc2, extrap = (NoExtrap(), NoExtrap()))
            for qx in tx2, qy in ty2
    ]
    itp2 = linear_interp((xs, ys), data; bc = bc2, extrap = NoExtrap())
    @test got ≈ ref atol = 1.0e-14 rtol = 1.0e-14
    @test itp2(gq2) ≈ ref atol = 1.0e-14 rtol = 1.0e-14

    out2 = similar(got)
    @test linear_interp!(out2, (xs, ys), data, gq2; bc = bc2, extrap = NoExtrap()) === out2
    @test out2 ≈ ref atol = 1.0e-14 rtol = 1.0e-14

    for deriv in (
            (EvalDeriv1(), EvalValue()),
            (EvalValue(), EvalDeriv1()),
            (EvalDeriv1(), EvalDeriv1()),
            (EvalDeriv2(), EvalValue()),
        )
        gotd = linear_interp((xs, ys), data, gq2; bc = bc2, extrap = NoExtrap(), deriv = deriv)
        refd = [
            linear_interp((xs, ys), data, (qx, qy); bc = bc2, extrap = NoExtrap(), deriv = deriv)
                for qx in tx2, qy in ty2
        ]
        @test gotd ≈ refd atol = 1.0e-14 rtol = 1.0e-14
    end

    # Periodic linear GriddedQuery must still use the separable gridded kernel,
    # not the generic point-wise batch fallback.
    fast = similar(got)
    @test FI._try_gridded_oneshot_methods!(
        fast, (xs, ys), data, gq2, (LinearInterp(bc), LinearInterp(NoBC())),
        NoExtrap(), EvalValue(), AutoCoeffs(),
    ) === true
    @test fast ≈ ref atol = 1.0e-14 rtol = 1.0e-14

    @test_throws ArgumentError linear_interp((xs, ys), data, gq2; bc = (PeriodicBC(), NoBC()))
end

@testitem "one-shot GriddedQuery: value-match narrow float + zero-alloc" setup = [Basic] begin
    # OneTo/Int grid beside Float32 data must solve at Float32 (value-matched Tg),
    # not blindly widen through the grid — same rule as scalar/batch one-shot.
    A32 = rand(Float32, 48, 40)
    gq32 = GriddedQuery(
        (
            collect(Float32, range(2.0f0, 47.0f0, 100)),
            collect(Float32, range(2.0f0, 39.0f0, 80)),
        )
    )
    os32 = linear_interp((Base.OneTo(48), Base.OneTo(40)), A32, gq32; extrap = ClampExtrap())
    @test eltype(os32) === Float32
    itp32 = linear_interp((Base.OneTo(48), Base.OneTo(40)), A32; extrap = ClampExtrap(), store = StorePolicy(; copy = false))
    @test os32 == itp32(gq32)

    # zero net allocation for the in-place one-shot even on Vector grids: the
    # value-matched grid cache is pool-backed (`_cache_axis_pooled`), released
    # at call scope. Bound stays far below one output Float64 scratch (100*80*8).
    gx = sort!(rand(48)) .* 47 .+ 1
    gy = sort!(rand(40)) .* 39 .+ 1
    A = rand(48, 40)
    tx = range(2.0, 47.0, 100)
    ty = range(2.0, 39.0, 80)
    gq = GriddedQuery((tx, ty))
    out = Matrix{Float64}(undef, 100, 80)
    function os_alloc(out, gx, gy, A, gq)
        linear_interp!(out, (gx, gy), A, gq; extrap = ClampExtrap())  # warmup
        return @allocated linear_interp!(out, (gx, gy), A, gq; extrap = ClampExtrap())
    end
    @test os_alloc(out, gx, gy, A, gq) <= 1024
end

@testitem "GriddedQuery strategy: fused + fullbuffer both zero-alloc (incl. Int axes)" setup = [Basic, AllocConstants] begin
    # Int grid beside Int-`UnitRange` query axes — the combination whose
    # heterogeneous `targets` tuple boxed the fullbuffer hull closure before
    # `_gridded_hulls` was made @generated. The strategy (fused vs fullbuffer) is a
    # runtime `all(out_size .< size(data))` branch, NOT a dispatch, so `@which`
    # can't distinguish it; instead each case asserts that branch condition to
    # document which path it takes, and both must be allocation-free.
    grid = (1:10, 2:20)
    data = rand(10, 19)
    function alloc(o, g, d, q)
        linear_interp!(o, g, d, q; extrap = ClampExtrap())   # warmup
        return @allocated linear_interp!(o, g, d, q; extrap = ClampExtrap())
    end

    # FULLBUFFER: axis-1 out width == grid width → NOT strictly smaller everywhere
    qfull = GriddedQuery((1:10, range(3.0, 4.1, 50)))     # 10 × 50
    ofull = Matrix{Float64}(undef, 10, 50)
    @test !all(map(<, size(ofull), size(data)))                    # → fullbuffer branch
    a_full = alloc(ofull, grid, data, qfull)                       # fills ofull, then measures
    @test ofull == linear_interp(grid, data, qfull; extrap = ClampExtrap())
    @test a_full <= ALLOC_THRESHOLD

    # FUSED: out strictly smaller than the grid on every axis (pure downsampling),
    # both query axes raw Int step-ranges
    qfused = GriddedQuery((1:2:9, 2:3:17))                         # 5 × 6, both < (10, 19)
    ofused = Matrix{Float64}(undef, 5, 6)
    @test all(map(<, size(ofused), size(data)))                    # → fused branch
    a_fused = alloc(ofused, grid, data, qfused)                    # fills ofused, then measures
    @test ofused == linear_interp(grid, data, qfused; extrap = ClampExtrap())
    @test a_fused <= ALLOC_THRESHOLD

    # 3D pure downsampling hits the generated fused kernel. This is a small
    # allocation pin for the path that carries the largest regression risk.
    grid3 = (1:12, 1:10, 1:8)
    data3 = rand(12, 10, 8)
    q3 = GriddedQuery((1:3:10, 1:3:10, 1:3:7))                    # 4 × 4 × 3
    out3 = Array{Float64, 3}(undef, size(q3))
    @test all(map(<, size(out3), size(data3)))                    # → fused branch
    a3 = alloc(out3, grid3, data3, q3)
    ref3 = linear_interp(grid3, data3, q3; extrap = ClampExtrap())
    @test out3 == ref3
    @test a3 <= ALLOC_THRESHOLD
end

@testitem "one-shot GriddedQuery: 3D + Fill parity" setup = [Basic] begin
    A = rand(24, 20, 16)
    grids = (1.0:24.0, 1.0:20.0, 1.0:16.0)
    itp = linear_interp(grids, A; extrap = ClampExtrap(), store = StorePolicy(; copy = false))
    gq = GriddedQuery(
        (
            range(2.0, 23.0, 40),
            range(2.0, 19.0, 32),
            range(2.0, 15.0, 24),
        )
    )
    @test linear_interp(grids, A, gq; extrap = ClampExtrap()) == itp(gq)

    # Fill parity (OOB slabs) — one-shot passes the resolved per-axis extraps
    itpf = linear_interp(grids, A; extrap = FillExtrap(0.0), store = StorePolicy(; copy = false))
    gqf = GriddedQuery(
        (
            range(-2.0, 27.0, 40),
            range(2.0, 19.0, 32),
            range(2.0, 15.0, 24),
        )
    )
    @test linear_interp(grids, A, gqf; extrap = FillExtrap(0.0)) == itpf(gqf)
end

@testitem "unified interp: GriddedQuery flows through the batch query protocol" setup = [Basic] begin
    using FastInterpolations: _query_length, _query_eltype, _query_extract

    grids = (1.0:16.0, 1.0:12.0)
    A = rand(16, 12)
    gq = GriddedQuery((range(2.0, 15.0, 20), range(2.0, 11.0, 14)))

    # `interp` (no GriddedQuery-specific method) routes through the generic batch
    # query path (pointwise) and returns an N-D array of size(gq)
    C = interp(grids, A, gq; method = LinearInterp(), extrap = ClampExtrap())
    @test size(C) == (20, 14)
    itp = linear_interp(grids, A; extrap = ClampExtrap())
    @test C ≈ [itp((x, y)) for x in gq.axes[1], y in gq.axes[2]]

    # in-place interp! writes into a matching N-D array (shape from _query_size)
    out = Matrix{Float64}(undef, 20, 14)
    @test interp!(out, grids, A, gq; method = LinearInterp(), extrap = ClampExtrap()) === out
    @test out == C
    # GriddedQuery's in-place one-shot output is N-D: callers must provide size(gq),
    # not a flat batch buffer.
    vout = Vector{Float64}(undef, 20 * 14)
    @test_throws DimensionMismatch interp!(vout, grids, A, gq; method = LinearInterp(), extrap = ClampExtrap())

    # the generic path supports ANY method (the reason to route through it)
    Cc = interp(grids, A, gq; method = CubicInterp(), extrap = ClampExtrap())
    @test size(Cc) == (20, 14)
    ref_c = [
        interp(grids, A, (x, y); method = CubicInterp(), extrap = ClampExtrap())
            for x in gq.axes[1], y in gq.axes[2]
    ]
    @test Cc ≈ ref_c

    # protocol methods directly (column-major: axis 1 varies fastest)
    @test _query_length(gq) == 20 * 14
    @test _query_eltype(gq) === Float64
    @test @inferred(_query_extract(gq, 1)) == (gq.axes[1][1], gq.axes[2][1])
    @test _query_extract(gq, 2) == (gq.axes[1][2], gq.axes[2][1])
    @test _query_extract(gq, 21) == (gq.axes[1][1], gq.axes[2][2])
end

@testitem "unified interp: GriddedQuery + linear uses separable fast path" setup = [Basic] begin
    grids = (1.0:64.0, 1.0:48.0)
    A = rand(64, 48)
    gq = GriddedQuery((range(2.0, 63.0, 120), range(2.0, 47.0, 90)))
    fast = linear_interp(grids, A, gq; extrap = ClampExtrap())

    # Correctness: interp on a GriddedQuery equals the separable path bit-for-bit.
    # (For LINEAR this also equals pointwise batch — both share the `_linear_kernel`
    # collapse — so equality alone can't prove routing; the `@which` block below does.)
    @test interp(grids, A, gq; method = LinearInterp(), extrap = ClampExtrap()) == fast
    @test interp(grids, A, gq; method = (LinearInterp(), LinearInterp()), extrap = ClampExtrap()) == fast

    # in-place also routes to separable
    out = Matrix{Float64}(undef, 120, 90)
    @test interp!(out, grids, A, gq; method = LinearInterp(), extrap = ClampExtrap()) === out
    @test out == fast

    # ROUTING PROOF (not values): interp!'s GriddedQuery hook is one canonical
    # gate, and the method-tuple dispatcher behind it resolves all-linear to the
    # separable arm. `which` inspects dispatch directly, so this proves routing
    # even though separable and pointwise produce bit-identical linear values.
    hook = FI._try_gridded_separable!
    A2 = Matrix{Float64}(undef, 2, 3)
    gqs = GriddedQuery(([2.0, 3.0], [2.0, 3.0, 4.0]))
    mhook_gq = which(hook, typeof((A2, grids, A, gqs, (LinearInterp(), LinearInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    mhook_soa = which(hook, typeof((A2, grids, A, ([2.0, 3.0], [2.0, 3.0]), (LinearInterp(), LinearInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    @test occursin("gridded_query", String(mhook_gq.file))
    @test mhook_gq !== mhook_soa

    dispatch = FI._try_gridded_oneshot_methods!
    m_lin = which(dispatch, typeof((A2, grids, A, gqs, (LinearInterp(), LinearInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    m_cub = which(dispatch, typeof((A2, grids, A, gqs, (CubicInterp(), CubicInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    @test occursin("gridded_query", String(m_lin.file))   # separable arm
    @test occursin("gridded_partials", String(m_cub.file)) # cubic fused-anchor arm

    # a non-linear method still works, N-D shaped
    Cc = interp(grids, A, gq; method = CubicInterp(), extrap = ClampExtrap())
    @test size(Cc) == (120, 90)
end

# ═══════════════════════════════════════════════════════════════════════════
# Multi-method separable evaluation (_AxisAnchor backbone)
# ═══════════════════════════════════════════════════════════════════════════

@testitem "gridded constant: gather == point-wise (sides × extraps, deriv, Int grids)" setup = [Basic] begin
    using FastInterpolations: GriddedQuery, ConstantInterp, NearestSide, LeftSide, RightSide

    x = range(0.0, 10.0, 24)
    y = range(-3.0, 5.0, 20)
    A = rand(24, 20)
    tx = range(-0.8, 10.9, 17)    # OOB both sides
    ty = range(-3.4, 5.6, 13)
    gq = GriddedQuery((tx, ty))

    # index selection is arithmetic-free → exact equality, OOB included.
    # RightSide × Clamp pins the coordinate-fold: a raw OOB-left dL < 0 would
    # flip its `iszero` test to the wrong node.
    for side in (NearestSide(), LeftSide(), RightSide()),
            ex in (ClampExtrap(), ExtendExtrap(), WrapExtrap())

        C = interp((x, y), A, gq; method = ConstantInterp(side = side), extrap = ex)
        ref = [
            interp((x, y), A, (qx, qy); method = ConstantInterp(side = side), extrap = ex)
                for qx in tx, qy in ty
        ]
        @test C == ref
        @test eltype(C) == eltype(ref)
    end

    # any-deriv arm (`* 0` short-circuit mirrors `_constant_nd_evaluate`)
    for deriv in ((EvalDeriv1(), EvalValue()), (EvalDeriv1(), EvalDeriv1()))
        Cd = interp((x, y), A, gq; method = ConstantInterp(), extrap = ClampExtrap(), deriv = deriv)
        refd = [
            interp((x, y), A, (qx, qy); method = ConstantInterp(), extrap = ClampExtrap(), deriv = deriv)
                for qx in tx, qy in ty
        ]
        @test Cd == refd
    end

    # Fill parity (post-pass slabs)
    CF = interp((x, y), A, gq; method = ConstantInterp(), extrap = FillExtrap(-7.5))
    refF = [
        interp((x, y), A, (qx, qy); method = ConstantInterp(), extrap = FillExtrap(-7.5))
            for qx in tx, qy in ty
    ]
    @test CF == refF

    # named one-shot + persistent functor agree with the unified path
    Cu = interp((x, y), A, gq; method = ConstantInterp(), extrap = ClampExtrap())
    @test constant_interp((x, y), A, gq; extrap = ClampExtrap()) == Cu
    itp = constant_interp((x, y), A; extrap = ClampExtrap())
    @test itp(gq) == Cu
    out = similar(Cu)
    @test itp(out, gq) === out
    @test out == Cu

    # Int grid + Int query axis (heterogeneous-tuple boxing guard case)
    gi = (1:10, 2:20)
    Ai = rand(10, 19)
    gqi = GriddedQuery((1:10, range(3.0, 15.0, 30)))
    Ci = interp(gi, Ai, gqi; method = ConstantInterp(), extrap = ClampExtrap())
    refi = [
        interp(gi, Ai, (qx, qy); method = ConstantInterp(), extrap = ClampExtrap())
            for qx in gqi.axes[1], qy in gqi.axes[2]
    ]
    @test Ci == refi

    # 3D
    A3 = rand(12, 10, 8)
    g3 = (1.0:12.0, 1.0:10.0, 1.0:8.0)
    t3 = (range(0.5, 12.4, 7), range(1.2, 9.7, 6), range(0.5, 8.5, 5))
    C3 = interp(g3, A3, GriddedQuery(t3); method = ConstantInterp(), extrap = ClampExtrap())
    ref3 = [
        interp(g3, A3, (a, b, c); method = ConstantInterp(), extrap = ClampExtrap())
            for a in t3[1], b in t3[2], c in t3[3]
    ]
    @test C3 == ref3

    # NoExtrap throws the canonical axis-named error BEFORE writing output
    err = try
        interp((x, y), A, gq; method = ConstantInterp(), extrap = NoExtrap())
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test occursin("axis 1", sprint(showerror, err))

    @test size(interp((x, y), A, GriddedQuery((Float64[], ty)); method = ConstantInterp(), extrap = ClampExtrap())) == (0, length(ty))
    @test size(interp((x, y), A, GriddedQuery((tx, Float64[])); method = ConstantInterp(), extrap = ClampExtrap())) == (length(tx), 0)
end

@testitem "gridded hermite: fullbuffer == point-wise (flavors × extraps × ops)" setup = [Basic] begin
    using FastInterpolations: GriddedQuery, PchipInterp, CardinalInterp, AkimaInterp

    x = range(0.0, 10.0, 24)
    y = range(-3.0, 5.0, 20)
    A = rand(24, 20)
    tx = range(-0.8, 10.9, 17)    # OOB both sides
    ty = range(-3.4, 5.6, 13)
    gq = GriddedQuery((tx, ty))

    for m in (PchipInterp(), CardinalInterp(0.0), CardinalInterp(0.5), AkimaInterp()),
            ex in (ClampExtrap(), ExtendExtrap(), WrapExtrap())

        H = interp((x, y), A, gq; method = m, extrap = ex)
        ref = [interp((x, y), A, (qx, qy); method = m, extrap = ex) for qx in tx, qy in ty]
        @test isclose(H, ref)
        @test eltype(H) == eltype(ref)
    end

    # deriv ops through Clamp with OOB targets — the point-wise ND surface
    # clamps the COORDINATE and still evaluates the kernel (a Clamp-OOB
    # derivative is the boundary cell polynomial's slope, not zero); the
    # anchor's dL fold must reproduce that.
    for deriv in (
            (EvalDeriv1(), EvalValue()),
            (EvalValue(), EvalDeriv1()),
            (EvalDeriv1(), EvalDeriv1()),
            (EvalDeriv2(), EvalValue()),
        )
        Hd = interp((x, y), A, gq; method = PchipInterp(), extrap = ClampExtrap(), deriv = deriv)
        refd = [
            interp((x, y), A, (qx, qy); method = PchipInterp(), extrap = ClampExtrap(), deriv = deriv)
                for qx in tx, qy in ty
        ]
        @test isclose(Hd, refd)
    end

    # Fill parity (post-pass slabs, value + deriv carrier zero)
    for deriv in (EvalValue(), (EvalDeriv1(), EvalValue()))
        HF = interp((x, y), A, gq; method = PchipInterp(), extrap = FillExtrap(-7.5), deriv = deriv)
        refF = [
            interp((x, y), A, (qx, qy); method = PchipInterp(), extrap = FillExtrap(-7.5), deriv = deriv)
                for qx in tx, qy in ty
        ]
        @test isclose(HF, refF)
    end

    # in-domain queries share every operation with point-wise → bit-identical here
    # (kept as a strict pin on THIS comparison; cross-context FMA drift is why the
    # OOB/deriv comparisons above use rtol)
    txi = range(0.2, 9.9, 17)
    tyi = range(-2.9, 4.9, 13)
    gq_in = GriddedQuery((txi, tyi))
    Hin = interp((x, y), A, gq_in; method = PchipInterp(), extrap = NoExtrap())
    @test isclose(Hin, [interp((x, y), A, (qx, qy); method = PchipInterp(), extrap = NoExtrap()) for qx in txi, qy in tyi])

    # mixed flavors compose per axis
    Hm = interp((x, y), A, gq_in; method = (PchipInterp(), AkimaInterp()), extrap = NoExtrap())
    refm = [
        interp((x, y), A, (qx, qy); method = (PchipInterp(), AkimaInterp()), extrap = NoExtrap())
            for qx in txi, qy in tyi
    ]
    @test isclose(Hm, refm)

    # 3D, fixed pass order == collapse order
    A3 = rand(12, 10, 8)
    g3 = (1.0:12.0, 1.0:10.0, 1.0:8.0)
    t3 = (range(0.5, 12.4, 7), range(1.2, 9.7, 6), range(0.5, 8.5, 5))
    H3 = interp(g3, A3, GriddedQuery(t3); method = PchipInterp(), extrap = ClampExtrap())
    ref3 = [
        interp(g3, A3, (a, b, c); method = PchipInterp(), extrap = ClampExtrap())
            for a in t3[1], b in t3[2], c in t3[3]
    ]
    @test isclose(H3, ref3)

    # Float32 value-matched width; Int grid beside Float targets
    A32 = rand(Float32, 16, 12)
    g32 = (collect(Float32, 1:16), collect(Float32, 1:12))
    t32 = (collect(Float32, range(1.5f0, 15.5f0, 9)), collect(Float32, range(1.2f0, 11.8f0, 7)))
    H32 = interp(g32, A32, GriddedQuery(t32); method = PchipInterp(), extrap = ClampExtrap())
    ref32 = [
        interp(g32, A32, (qx, qy); method = PchipInterp(), extrap = ClampExtrap())
            for qx in t32[1], qy in t32[2]
    ]
    @test eltype(H32) == eltype(ref32)
    @test isclose(H32, ref32)

    gi = (1:10, 2:20)
    Ai = rand(10, 19)
    gqi = GriddedQuery((1:10, range(3.0, 15.0, 30)))
    Hi = interp(gi, Ai, gqi; method = PchipInterp(), extrap = ClampExtrap())
    refi = [
        interp(gi, Ai, (qx, qy); method = PchipInterp(), extrap = ClampExtrap())
            for qx in gqi.axes[1], qy in gqi.axes[2]
    ]
    @test isclose(Hi, refi)

    # NoExtrap throws axis-named before writing
    err = try
        interp((x, y), A, gq; method = PchipInterp(), extrap = NoExtrap())
        nothing
    catch e
        e
    end
    @test err isa DomainError
    @test occursin("axis 1", sprint(showerror, err))

    @test size(interp((x, y), A, GriddedQuery((Float64[], ty)); method = PchipInterp(), extrap = ClampExtrap())) == (0, length(ty))
    @test size(interp((x, y), A, GriddedQuery((tx, Float64[])); method = PchipInterp(), extrap = ClampExtrap())) == (length(tx), 0)
end

@testitem "gridded cubic/quadratic: fused anchors == point-wise" setup = [Basic] begin
    using FastInterpolations: GriddedQuery, CubicInterp, QuadraticInterp, EvalDeriv1
    using InteractiveUtils: which

    x = range(0.0, 1.0, 22)
    y = range(-1.0, 2.0, 19)
    A = [sin(4xi) + cos(3yi) + 0.1sin(xi * yi) for xi in x, yi in y]
    tx = range(-0.05, 1.05, 13)
    ty = range(-0.8, 1.8, 11)
    gq = GriddedQuery((tx, ty))
    close(a, b) = size(a) == size(b) && all(isapprox.(a, b; rtol = 2.0e-12, atol = 2.0e-12))

    for method in (CubicInterp(), QuadraticInterp())
        for ex in (ClampExtrap(), FillExtrap(-7.25), WrapExtrap())
            C = interp((x, y), A, gq; method = method, extrap = ex)
            ref = [
                interp((x, y), A, (qx, qy); method = method, extrap = ex)
                    for qx in tx, qy in ty
            ]
            @test close(C, ref)

            itp = interp((x, y), A; method = method, extrap = ex)
            @test close(itp(gq), ref)
            out = similar(C)
            @test itp(out, gq) === out
            @test close(out, ref)
        end

        Cd = interp(
            (x, y), A, gq;
            method = method, extrap = ClampExtrap(),
            deriv = (EvalDeriv1(), EvalValue()),
        )
        refd = [
            interp(
                    (x, y), A, (qx, qy);
                    method = method, extrap = ClampExtrap(),
                    deriv = (EvalDeriv1(), EvalValue()),
                )
                for qx in tx, qy in ty
        ]
        @test close(Cd, refd)

        err = try
            interp((x, y), A, gq; method = method, extrap = NoExtrap())
            nothing
        catch e
            e
        end
        @test err isa DomainError
        @test occursin("axis 1", sprint(showerror, err))

        @test size(interp((x, y), A, GriddedQuery((Float64[], ty)); method = method, extrap = ClampExtrap())) == (0, length(ty))
        @test size(interp((x, y), A, GriddedQuery((tx, Float64[])); method = method, extrap = ClampExtrap())) == (length(tx), 0)
    end

    out = zeros(length(tx), length(ty))
    hook = FI._try_gridded_oneshot_methods!
    m_cub = which(hook, typeof((out, (x, y), A, gq, (CubicInterp(), CubicInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    m_quad = which(hook, typeof((out, (x, y), A, gq, (QuadraticInterp(), QuadraticInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    @test occursin("gridded_partials", String(m_cub.file))
    @test occursin("gridded_partials", String(m_quad.file))

    out_onfly = fill(NaN, length(tx), length(ty))
    @test hook(out_onfly, (x, y), A, gq, (CubicInterp(), CubicInterp()), ClampExtrap(), EvalValue(), OnTheFly()) === false
    @test all(isnan, out_onfly)
    @test hook(out_onfly, (x, y), A, gq, (QuadraticInterp(), QuadraticInterp()), ClampExtrap(), EvalValue(), OnTheFly()) === false
    @test all(isnan, out_onfly)
end

@testitem "F7 gridded cubic/quadratic: persistent + one-shot zero-alloc warm" setup = [Basic, AllocConstants] begin
    using FastInterpolations: GriddedQuery, CubicInterp, QuadraticInterp

    function persistent_alloc(out, grids, data, gq, method)
        itp = interp(grids, data; method = method, extrap = ClampExtrap())
        itp(out, gq)
        itp(out, gq)
        return @allocated itp(out, gq)
    end

    function oneshot_alloc(out, grids, data, gq, method)
        interp!(out, grids, data, gq; method = method, extrap = ClampExtrap())
        interp!(out, grids, data, gq; method = method, extrap = ClampExtrap())
        return @allocated interp!(out, grids, data, gq; method = method, extrap = ClampExtrap())
    end

    x = range(0.0, 1.0, 22)
    y = range(-1.0, 2.0, 19)
    A = [sin(4xi) + cos(3yi) + 0.1sin(xi * yi) for xi in x, yi in y]
    gq = GriddedQuery((range(0.05, 0.95, 13), range(-0.8, 1.8, 11)))
    out = zeros(size(gq))
    for method in (CubicInterp(), QuadraticInterp())
        @test persistent_alloc(out, (x, y), A, gq, method) <= ALLOC_THRESHOLD
        @test oneshot_alloc(out, (x, y), A, gq, method) <= ALLOC_THRESHOLD
    end

    g3 = (
        range(0.0, 1.0, 12),
        range(-1.0, 1.0, 10),
        range(0.0, 2.0, 8),
    )
    A3 = [sin(3x1) + cos(2x2) + 0.1x3 for x1 in g3[1], x2 in g3[2], x3 in g3[3]]
    gq3 = GriddedQuery(
        (
            range(0.1, 0.9, 5),
            range(-0.8, 0.8, 4),
            range(0.2, 1.8, 3),
        )
    )
    out3 = zeros(size(gq3))
    for method in (CubicInterp(), QuadraticInterp())
        @test persistent_alloc(out3, g3, A3, gq3, method) <= ALLOC_THRESHOLD
        @test oneshot_alloc(out3, g3, A3, gq3, method) <= ALLOC_THRESHOLD
    end
end

@testitem "gridded routing: constant/hermite hooks, PeriodicBC + mixed-method fallback" setup = [Basic] begin
    using FastInterpolations: GriddedQuery, ConstantInterp, PchipInterp, LinearInterp, PeriodicBC
    using InteractiveUtils: which

    x = range(0.0, 2π, 25)
    A = [sin(xi) + cos(yi) for xi in x, yi in x]
    tq = range(0.3, 6.0, 11)
    gq = GriddedQuery((tq, tq))
    out = zeros(11, 11)
    hook = FI._try_gridded_oneshot_methods!

    m_bridge = which(hook, typeof((out, (x, x), A, gq, (ConstantInterp(), ConstantInterp()), ClampExtrap(), EvalValue(), AutoCoeffs())))
    @test occursin("gridded_query", String(m_bridge.file))

    m_c = which(hook, typeof((out, (x, x), A, gq.axes, (ConstantInterp(), ConstantInterp()), ClampExtrap(), EvalValue())))
    m_h = which(hook, typeof((out, (x, x), A, gq.axes, (PchipInterp(), PchipInterp()), ClampExtrap(), EvalValue())))
    @test occursin("gridded_constant", String(m_c.file))
    @test occursin("gridded_hermite", String(m_h.file))

    # PeriodicBC slope stencils and mixed method families fall through to the
    # generic `false` fallback (point-wise batch) — and stay correct there.
    mper = PchipInterp(PeriodicBC())
    m_p = which(hook, typeof((out, (x, x), A, gq.axes, (mper, mper), WrapExtrap(), EvalValue())))
    m_mix = which(hook, typeof((out, (x, x), A, gq.axes, (ConstantInterp(), LinearInterp()), ClampExtrap(), EvalValue())))
    m_gen = m_mix
    @test m_p === m_gen
    @test m_mix === m_gen

    Hp = interp((x, x), A, gq; method = mper, extrap = WrapExtrap())
    refp = [interp((x, x), A, (qx, qy); method = mper, extrap = WrapExtrap()) for qx in tq, qy in tq]
    @test Hp ≈ refp
    Hmix = interp((x, x), A, gq; method = (ConstantInterp(), LinearInterp()), extrap = ClampExtrap())
    refmix = [
        interp((x, x), A, (qx, qy); method = (ConstantInterp(), LinearInterp()), extrap = ClampExtrap())
            for qx in tq, qy in tq
    ]
    @test Hmix ≈ refmix
end

@testitem "gridded constant/hermite: zero-alloc warm (Vector + Int axes)" setup = [Basic, AllocConstants] begin
    using FastInterpolations: GriddedQuery, ConstantInterp, PchipInterp, AkimaInterp

    # First execution of an @allocated call site can catch one-time pool
    # first-touch growth (~290 KB, shared with the linear path on the same
    # axes) — measure twice, assert the steady state.
    function meas(out, grids, data, gq, m)
        interp!(out, grids, data, gq; method = m, extrap = ClampExtrap())
        return @allocated interp!(out, grids, data, gq; method = m, extrap = ClampExtrap())
    end
    steady(out, grids, data, gq, m) = (meas(out, grids, data, gq, m); meas(out, grids, data, gq, m))

    x = range(0.0, 10.0, 24)
    y = range(-3.0, 5.0, 20)
    A = rand(24, 20)
    gq = GriddedQuery((range(-0.8, 10.9, 17), range(-3.4, 5.6, 13)))
    out = zeros(17, 13)
    @test steady(out, (x, y), A, gq, ConstantInterp()) <= ALLOC_THRESHOLD
    @test steady(out, (x, y), A, gq, PchipInterp()) <= ALLOC_THRESHOLD
    @test steady(out, (x, y), A, gq, AkimaInterp()) <= ALLOC_THRESHOLD

    # Int grid + Int query axis (heterogeneous-tuple boxing guard)
    gi = (1:10, 2:20)
    Ai = rand(10, 19)
    gqi = GriddedQuery((1:10, range(3.0, 15.0, 30)))
    outi = zeros(10, 30)
    @test steady(outi, gi, Ai, gqi, ConstantInterp()) <= ALLOC_THRESHOLD
    @test steady(outi, gi, Ai, gqi, PchipInterp()) <= ALLOC_THRESHOLD

    # 3D hermite (pooled intermediates recycled across passes)
    A3 = rand(12, 10, 8)
    g3 = (collect(1.0:12.0), collect(1.0:10.0), collect(1.0:8.0))
    gq3 = GriddedQuery((range(1.0, 12.0, 7), range(1.2, 9.7, 6), range(0.5, 8.5, 5)))
    out3 = zeros(7, 6, 5)
    @test steady(out3, g3, A3, gq3, PchipInterp()) <= ALLOC_THRESHOLD
end

@testitem "gridded N=1: flat-vector out disambiguation vs batch protocol" setup = [Basic] begin
    using FastInterpolations: GriddedQuery, ConstantInterp, PchipInterp

    # `out::AbstractVector` + 1-axis GriddedQuery sits in the ambiguity
    # intersection with the generic batch entries (`queries::Any`) — these pins
    # keep the N = 1 disambiguation methods routed to the gridded arm.
    x = range(0.0, 10.0, 24)
    v = sin.(x)
    tq = range(-0.5, 10.5, 9)
    gq = GriddedQuery((tq,))
    out = zeros(9)

    # 1D-scalar reference crosses inline contexts → elementwise rtol=1e-15
    # (FMA contraction is inline-context-dependent; established policy)
    ref = [linear_interp(x, v, q; extrap = ClampExtrap()) for q in tq]
    linear_interp!(out, (x,), v, gq; extrap = ClampExtrap())
    @test all(isapprox.(out, ref; rtol = 1.0e-15))

    refc = [interp((x,), v, (q,); method = ConstantInterp(), extrap = ClampExtrap()) for q in tq]
    constant_interp!(out, (x,), v, gq; extrap = ClampExtrap())
    @test out == refc
    interp!(out, (x,), v, gq; method = ConstantInterp(), extrap = ClampExtrap())
    @test out == refc

    H = interp((x,), v, gq; method = PchipInterp(), extrap = ClampExtrap())
    refh = [interp((x,), v, (q,); method = PchipInterp(), extrap = ClampExtrap()) for q in tq]
    @test size(H) == (9,)
    @test all(isapprox.(H, refh; rtol = 1.0e-15, atol = 1.0e-13))
end

@testitem "unified itp(gq): one AbstractInterpolantND callable, N-D shape for all methods" setup = [Basic] begin
    using InteractiveUtils: which

    x = range(0.0, 10.0, 24)
    y = range(-3.0, 5.0, 20)
    A = rand(24, 20)
    tx = range(-0.5, 10.5, 15)
    ty = range(-3.2, 5.4, 11)
    gq = GriddedQuery((tx, ty))
    rt(a, b) = size(a) == size(b) && all(isapprox.(a, b; rtol = 1.0e-13, atol = 1.0e-12))

    # Every persistent ND interpolant evaluates a GriddedQuery through ONE
    # callable and returns an N-D size(gq) array — including cubic/quadratic,
    # which previously fell through to the generic batch functor and returned a
    # FLAT vector (the inconsistency this unification fixes).
    for (label, itp) in (
            ("linear", linear_interp((x, y), A; extrap = ClampExtrap())),
            ("constant", constant_interp((x, y), A; extrap = ClampExtrap())),
            ("cubic", cubic_interp((x, y), A; extrap = ClampExtrap())),
            ("quadratic", quadratic_interp((x, y), A; extrap = ClampExtrap())),
        )
        C = itp(gq)
        @test C isa AbstractMatrix
        @test size(C) == (15, 11)
        ref = [itp((qx, qy)) for qx in tx, qy in ty]
        @test rt(C, ref)
        out = similar(C)
        @test itp(out, gq) === out
        @test rt(out, ref)
    end

    # The unified callable resolves to gridded_dispatch.jl for ALL ND methods
    # (the concrete per-type functors are gone). The separable/pointwise choice
    # is delegated to the prepared method-tuple dispatcher.
    itpL = linear_interp((x, y), A; extrap = ClampExtrap())
    itpCub = cubic_interp((x, y), A; extrap = ClampExtrap())
    @test occursin("gridded_dispatch", String(which(itpL, typeof((gq,))).file))
    @test occursin("gridded_dispatch", String(which(itpCub, typeof((gq,))).file))
    eit = FI._gridded_eval_itp!
    args = (zeros(15, 11), gq, (EvalValue(), EvalValue()), (ClampExtrap(), ClampExtrap()), nothing, nothing)
    @test occursin("gridded_dispatch", String(which(eit, typeof((args[1], itpL, args[2:end]...))).file))
    @test occursin("gridded_dispatch", String(which(eit, typeof((args[1], constant_interp((x, y), A), args[2:end]...))).file))
    @test occursin("gridded_dispatch", String(which(eit, typeof((args[1], itpCub, args[2:end]...))).file))
    emit = FI._gridded_eval_methods!
    ops = (EvalValue(), EvalValue())
    extraps = (ClampExtrap(), ClampExtrap())
    @test occursin("gridded_query", String(which(emit, typeof((args[1], itpL.grids, itpL.data, gq.axes, (LinearInterp(), LinearInterp()), ops, extraps))).file))
    itpConst = constant_interp((x, y), A; extrap = ClampExtrap())
    @test occursin("gridded_constant", String(which(emit, typeof((args[1], itpConst.grids, itpConst.data, gq.axes, (ConstantInterp(), ConstantInterp()), ops, extraps))).file))
    @test occursin("gridded_dispatch", String(which(emit, typeof((args[1], itpCub.grids, A, gq.axes, (CubicInterp(), CubicInterp()), ops, extraps))).file))

    # deriv + call-time InBounds override still thread through the unified path
    Cd = itpL(gq; deriv = (EvalDeriv1(), EvalValue()))
    @test rt(Cd, [itpL((qx, qy); deriv = (EvalDeriv1(), EvalValue())) for qx in tx, qy in ty])

    # ordinary (non-gridded) batch queries are unaffected — still a flat Vector
    b = itpL(([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]))
    @test b isa AbstractVector && length(b) == 3
end

@testitem "persistent hermite itp(gq): fast arm (not pointwise default), == one-shot" setup = [Basic] begin
    using InteractiveUtils: which

    x = range(0.0, 10.0, 24)
    y = range(-3.0, 5.0, 20)
    A = rand(24, 20)
    gq = GriddedQuery((range(-0.5, 10.5, 15), range(-3.2, 5.4, 11)))
    arm(itp) = String(which(FI._gridded_eval_methods!, typeof((zeros(15, 11), itp.grids, itp.data, gq.axes, itp.methods, (EvalValue(), EvalValue()), (ClampExtrap(), ClampExtrap())))).file)

    # A local-hermite HeteroInterpolantND (OnTheFly) evaluates a GriddedQuery via
    # the SAME separable kernel as the one-shot — persistent itp(gq) is no longer
    # the point-wise default. Values must equal the one-shot and the point-wise ref.
    for m in (PchipInterp(), AkimaInterp(), CardinalInterp(0.5))
        itp = interp((x, y), A; method = m, extrap = ClampExtrap())
        @test occursin("gridded_hermite", arm(itp))              # fast arm, not gridded_dispatch default
        G = itp(gq)
        ref = [itp((qx, qy)) for qx in gq.axes[1], qy in gq.axes[2]]
        oneshot = interp((x, y), A, gq; method = m, extrap = ClampExtrap())
        @test isclose(G, ref)
        @test isclose(G, oneshot)
        out = similar(G)
        @test itp(out, gq) === out
        @test isclose(out, ref)
    end

    # deriv threads through the persistent fast arm
    itp = interp((x, y), A; method = PchipInterp(), extrap = ClampExtrap())
    Gd = itp(gq; deriv = (EvalDeriv1(), EvalValue()))
    @test isclose(Gd, [itp((qx, qy); deriv = (EvalDeriv1(), EvalValue())) for qx in gq.axes[1], qy in gq.axes[2]])

    # mixed families (linear × pchip) and PreCompute (cubic) hetero stay on the
    # point-wise default arm — and stay correct there.
    itpm = interp((x, y), A; method = (LinearInterp(), PchipInterp()), extrap = ClampExtrap())
    @test isclose(itpm(gq), [itpm((qx, qy)) for qx in gq.axes[1], qy in gq.axes[2]])
    itpc = interp((x, y), A; method = CubicInterp(), extrap = ClampExtrap())
    @test isclose(itpc(gq), [itpc((qx, qy)) for qx in gq.axes[1], qy in gq.axes[2]])
end
