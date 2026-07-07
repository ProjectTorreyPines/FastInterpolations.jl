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
    @test isbits(b[1]) && sizeof(b[1]) == 16
    @test b[1].idx == 3
    @test b[1].alpha ≈ 0.25

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
        for (tx, ty) in (up, down, mixed, oob, nonmono, single)
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

@testitem "strategy cores: fused + both fullbuffer orders agree" begin
    using FastInterpolations
    import FastInterpolations as FI
    using FastInterpolations: _gridded_anchors, _gridded_fused!, _gridded_fullbuffer!

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
    _gridded_fused!(Cf, A, ax, ay)
    Cb1 = Matrix{Float64}(undef, M, N)
    _gridded_fullbuffer!(Cb1, A, ax, ay, Float64, true)    # blend dim2 first
    Cb2 = Matrix{Float64}(undef, M, N)
    _gridded_fullbuffer!(Cb2, A, ax, ay, Float64, false)   # gather dim1 first

    # all strategies are mathematically equivalent — machine-eps, NOT bit-identity
    @test all(isapprox.(Cf, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test all(isapprox.(Cb1, ref; rtol = 1.0e-14, atol = 1.0e-14))
    @test all(isapprox.(Cb2, ref; rtol = 1.0e-14, atol = 1.0e-14))

    # the public entry agrees with the cores
    C = itp(GriddedQuery((tx, ty)))
    @test all(isapprox.(C, ref; rtol = 1.0e-14, atol = 1.0e-14))
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

    # Wrap / Fill: explicit ArgumentError at dispatch (both call forms)
    for ex in (FI.WrapExtrap(), FI.FillExtrap(NaN))
        itpu = FI.linear_interp((1.0:16.0, 1.0:12.0), A; extrap = ex)
        @test_throws ArgumentError itpu(GriddedQuery((tx, ty)))
        out = Matrix{Float64}(undef, length(tx), length(ty))
        @test_throws ArgumentError itpu(out, GriddedQuery((tx, ty)))
    end
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
